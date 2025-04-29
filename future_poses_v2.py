#!/usr/bin/env python3
import os
import json
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.distance import cdist

# ─── Arg parsing ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Project future‐pose footprints onto thermal frames"
)
parser.add_argument(
    "-r","--root-dir", required=True,
    help="root folder containing .pkl, image subdirs, all_poses/"
)
parser.add_argument(
    "-p","--pkl-path", required=True,
    help="path to one .pkl file to process"
)
parser.add_argument(
    "-n","--n-samples", type=int, default=5,
    help="how many sample points per frame"
)
args = parser.parse_args()

ROOT_DIR  = args.root_dir
PKL_PATH  = args.pkl_path
N_SAMPLES = args.n_samples

# ─── Helper fns & constants ───────────────────────────────────────────────────
VIZ_IMAGE_SIZE = (1280, 1024)

def gen_camera_matrix(fx, fy, cx, cy):
    return np.array([[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]])

def project_points(xy, h, x_off, K, dist_coeffs):
    n,_ = xy.shape
    # lift to (X,Y,Z) in robot frame
    pts = np.concatenate([xy, -h*np.ones((n,1))], axis=1)
    pts[:,0] += x_off
    # CV convention
    cv_pts = np.stack([pts[:,1], -pts[:,2], pts[:,0]], axis=1)
    uv,_ = cv2.projectPoints(cv_pts, (0,0,0), (0,0,0), K, dist_coeffs)
    return uv.reshape(-1,2)

def get_pos_pixels(xy, h, x_off, K, dist_coeffs):
    pix = project_points(xy, h, x_off, K, dist_coeffs)
    # flip x
    pix[:,0] = VIZ_IMAGE_SIZE[0] - pix[:,0]
    # clip
    pix = np.stack([
        [np.clip(u,0,VIZ_IMAGE_SIZE[0]), np.clip(v,0,VIZ_IMAGE_SIZE[1])]
        for u,v in pix
    ])
    return pix

def farthest_point_sampling(points, n_samples):
    n = len(points)
    if n_samples >= n:
        return np.arange(n)
    idxs = [np.random.randint(n)]
    dists = np.full(n, np.inf)
    for _ in range(1, n_samples):
        last = points[idxs[-1]]
        newd = cdist(points, last.reshape(1,-1)).squeeze()
        dists = np.minimum(dists, newd)
        nexti = np.argmax(dists)
        idxs.append(nexti)
        dists[nexti] = 0
    return np.array(idxs)

def yaw_from_quaternion(qx,qy,qz,qw):
    siny = 2*(qw*qz + qx*qy)
    cosy = 1 - 2*(qy*qy + qz*qz)
    return np.arctan2(siny,cosy)

def transform_lg(world_pts, X0, Y0, psi0):
    T = np.array([
        [ np.cos(psi0), -np.sin(psi0), X0],
        [ np.sin(psi0),  np.cos(psi0), Y0],
        [           0.0,          0.0, 1.0]
    ])
    invT = np.linalg.inv(T)
    out = []
    for x,y in world_pts:
        v = invT @ np.array([x,y,1.0])
        out.append([v[0], v[1]])
    return np.array(out)

def plot_footprints(ax, pix, w, l, K, h, x_off, w_scale, l_scale):
    for i in range(len(pix)-1):
        # back‐project pixel to depth
        def depth(p):
            # normalized coords
            uv = np.hstack([p,1.0])
            norm = np.linalg.inv(K) @ uv
            return h / norm[1]
        d0 = depth(pix[i])
        d1 = depth(pix[i+1])
        # scale box size
        f = K[0,0]
        w0 = w_scale * w * f / d0
        l0 = l_scale * l * f / d0
        w1 = w_scale * w * f / d1
        l1 = l_scale * l * f / d1
        # quad corners
        p1 = pix[i]   + np.array([-w0/2,0])
        p2 = pix[i]   + np.array([ w0/2,0])
        p3 = pix[i+1] + np.array([ w1/2,0])
        p4 = pix[i+1] + np.array([-w1/2,0])
        poly = Polygon([p1,p2,p3,p4], closed=True,
                       edgecolor='blue', facecolor='blue', alpha=0.3)
        ax.add_patch(poly)

# ─── Load pickle (images + timestamps) ────────────────────────────────────────
with open(PKL_PATH,'rb') as f:
    data = pickle.load(f)
thermal_images    = data['thermal_paths']
thermal_ts_list   = data['thermal_timestamps']

# derive basename, output dir
basename = os.path.splitext(os.path.basename(PKL_PATH))[0]
out_dir  = os.path.join(ROOT_DIR, f"Traj_Footprints_{basename}")
os.makedirs(out_dir, exist_ok=True)

# ─── Read odometry from all_poses/<basename>/path_poses.txt ──────────────────
poses_dir = os.path.join(ROOT_DIR, 'all_poses', basename)
txt_path  = os.path.join(poses_dir, 'path_poses.txt')
if not os.path.isfile(txt_path):
    raise FileNotFoundError(f"cannot find {txt_path}")

original_odom_poses = []
with open(txt_path,'r') as fp:
    for line in fp:
        # if line.startswith('#') or not line.strip():
        #     continue
        # parts = line.split()
        # # assume: ts x y z qx qy qz qw
        # ts  = float(parts[0])
        for line in fp:
            line = line.strip()
            if (not line or line.startswith("#") or line.lower().startswith('timestamp')):
                continue
            parts = line.split()
            try:
                ts = float(parts[0])
            except ValueError:
                print(f"Skipping line: {line}")
                continue
        x,y = float(parts[1]), float(parts[2])
        qx,qy,qz,qw = map(float, parts[4:8])
        original_odom_poses.append({
            'timestamp': ts,
            'x': x, 'y': y,
            'qx': qx,'qy': qy,'qz': qz,'qw': qw
        })

# ─── Camera intrinsics/extrinsics ─────────────────────────────────────────────
camera_height   = 0.409 + 0.1
camera_x_offset = 0.451
fx,fy,cx,cy     = 935.2356, 935.7905, 656.1572, 513.7144
K               = gen_camera_matrix(fx,fy,cx,cy)
dist_coeffs     = np.array([
    -0.08194476107782814,
    -0.06592640858415261,
    -0.0007043163003212235,
     0.002577256982584405
])

# robot dims & scales
robot_w, robot_l = 0.6, 1.0
w_scale, l_scale = 3.0, 0.0

# ─── Rebase thermal image paths ───────────────────────────────────────────────
# we assume each thermal_images entry ends in ".../<basename>/<file>.png"
img_dir = os.path.join(ROOT_DIR, basename)
thermal_images = [
    os.path.join(img_dir, os.path.basename(p)) for p in thermal_images
]

# ─── Main loop: project + sample + dump JSON ─────────────────────────────────
all_samples = []

for idx, t_ts in enumerate(thermal_ts_list):
    # find future odom
    future = [p for p in original_odom_poses if p['timestamp'] >= t_ts]
    if not future:
        print(f"skip idx={idx}: no odom ≥ {t_ts}")
        all_samples.append([])
        continue
    future = future[:250]

    # build traj in ego frame
    trajs = []
    for i,pose in enumerate(future):
        X0, Y0 = pose['x'], pose['y']
        psi0    = yaw_from_quaternion(
            pose['qx'], pose['qy'], pose['qz'], pose['qw']
        )
        ptsG = np.array([[p['x'],p['y']] for p in future[i:]])
        trajs.append(transform_lg(ptsG, X0, Y0, psi0))
    traj = trajs[0]

    # project to pixels
    pix = get_pos_pixels(traj, camera_height, camera_x_offset, K, dist_coeffs)

    # sample
    idxs = farthest_point_sampling(pix, N_SAMPLES)
    samp = pix[idxs].astype(int).tolist()
    all_samples.append(samp)

    # Optional: draw & save image
    # fig,ax = plt.subplots(figsize=(12,9))
    # plot_footprints(ax, pix, robot_w, robot_l,
    #                 K, camera_height, camera_x_offset, w_scale, l_scale)
    # ax.scatter(*np.array(samp).T, color='red', s=50, zorder=5)
    # ax.imshow(cv2.cvtColor(cv2.imread(thermal_images[idx]), cv2.COLOR_BGR2RGB))
    # ax.axis('off')
    # fig.savefig(os.path.join(out_dir, f"{idx:04d}.png"),
    #             bbox_inches='tight', pad_inches=0)
    # plt.close(fig)

# write JSON
with open(os.path.join(out_dir, f"{basename}_samples.json"), 'w') as jf:
    json.dump(all_samples, jf, indent=2)

print("Wrote JSON to", os.path.join(out_dir, f"{basename}_samples.json"))
