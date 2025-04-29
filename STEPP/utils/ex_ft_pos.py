#!/usr/bin/python3
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from nav_msgs.msg import Odometry as odom
import os
import matplotlib.pyplot as plt
import json

class CameraPinhole:
    def __init__(self, width, height, camera_name, distortion_model, K, D, Rect, P):
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.distortion_model = distortion_model
        self.K = K
        self.D = D
        self.Rect = Rect
        self.P = P

    def undistort(self, image):
        undistorted_image = cv2.undistort(image, self.K, self.D)
        return undistorted_image

    def project(self, point):
        point_2d, _ = cv2.projectPoints(point.reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.K, self.D)
        return point_2d[0, 0]

def main():    
    D = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])
    K = np.array([935.2355857804463, 0.0, 656.1572332633887, 0.0, 935.7905325732659, 513.7144019593092, 0.0, 0.0, 1.0]).reshape(3, 3)
    Rect = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    P = np.array([935.2355857804463, 0, 656.1572332633887, 0, 0, 935.7905325732659, 513.7144019593092, 0, 0, 0, 1, 0]).reshape(3, 4) 
    camera_pinhole = CameraPinhole(width=1280, height=1024, camera_name='custom_camera', 
                                distortion_model='plumb_bob', 
                                K=K, D=D, Rect=Rect, P=P)

    # Load images
    folder_path = '/home/vader/RobotiXX/STEPP-Code_Thermal/STEPP/Data/subset_data/thermal_BL_2024-09-04_19-09-17_chunk0000_processed/'
    folder_path = os.path.expanduser(folder_path)  # Expand ~ to full path
    images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])
    img_file_names = [os.path.basename(img) for img in images]
    print(f"Number of images found: {len(images)}")

    # ROS setup (unchanged)
    rospy.init_node('trajectory_publisher', anonymous=True)
    pub = rospy.Publisher('/trajectory', odom, queue_size=10)
    pub2 = rospy.Publisher('/trajectory2', odom, queue_size=10)

    # Load coordinates
    # coordinates_path = '~/RobotiXX/STEPP-Code_Thermal/odometry.txt'  # Update this to your actual path
    coordinates_path = '/home/vader/RobotiXX/STEPP-Code_Thermal/STEPP/Data/subset_data/BL_2024-09-04_19-09-17_chunk0000_processed/path_poses.txt'
    # coordinates_path = '/home/vader/RobotiXX/STEPP-Code_Thermal/STEPP/Data/subset_data/BL_2024-09-04_19-09-17_chunk0000_processed/path_poses.txt'
    coordinates_path = os.path.expanduser(coordinates_path)
    coordinates = []
    orientations = []
    T_odom_list = []
    with open(coordinates_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split()
            if parts:
                coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                orientations.append(np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]))
                T_odom = np.eye(4, 4)
                T_odom[:3, :3] = R.from_quat(orientations[-1]).as_matrix()[:3, :3]
                T_odom[:3, 3] = coordinates[-1]
                T_odom_list.append(T_odom)

    # Frame transformations (unchanged)
    translation = [-0.739, -0.056, -0.205]
    path_translation = [0.0, 0.0, 0.0]
    rotation = [0.466, -0.469, -0.533, 0.528]
    T_imu_camera = np.eye(4, 4)
    T_imu_camera[:3, :3] = R.from_quat(rotation).as_matrix()[:3, :3]
    T_imu_camera[:3, 3] = translation

    for i in range(len(coordinates)):
        T_world_camera = np.linalg.inv(T_imu_camera) @ T_odom_list[i] @ T_imu_camera
        coordinates[i] = T_world_camera[:3, 3]
        orientations[i] = R.from_matrix(T_world_camera[:3, :3]).as_quat()

    # Odometry messages (unchanged)
    directions = []
    for i in range(len(coordinates)):
        odom_msg = odom()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = coordinates[i][0]
        odom_msg.pose.pose.position.y = coordinates[i][1]
        odom_msg.pose.pose.position.z = coordinates[i][2]
        odom_msg.pose.pose.orientation.x = orientations[i][0]
        odom_msg.pose.pose.orientation.y = orientations[i][1]
        odom_msg.pose.pose.orientation.z = orientations[i][2]
        odom_msg.pose.pose.orientation.w = orientations[i][3]
        directions.append(odom_msg)

    def unit_vector(vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector
        return vector / magnitude

    def trasnform_coord(quat, coord):
        R1 = R.from_quat(quat).as_matrix()
        return R1.T @ coord

    def translate_to_frame(coords, point, quat):
        New_frame_coord = []
        for i in range(1, len(coords)):
            c = trasnform_coord(quat, point - coords[i] - path_translation)
            New_frame_coord.append(c)
        return np.array(New_frame_coord)

    u_C2_past = np.zeros((2, 1))
    future_steps = 10
    all_points = []

    print('length of coordinates:', len(coordinates))

    for i in range(1, len(coordinates) - future_steps + 1):  # Adjusted loop range
        point = i
        points = translate_to_frame(coordinates[point:], coordinates[point], orientations[point])
        p_C2 = points.T
        img_path = folder_path + img_file_names[point]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        print(f"Processing image {point}: {img_path}")
        print(f"Number of future points to project: {points.shape[0]}")

        img_points = []
        u_C2_past[0] = 1280 / 2
        u_C2_past[1] = 900

 # for j in range(1, future_steps):
        #     p_C = p_C2[:, j]
        #     tmp_p = camera_pinhole.project(p_C)
        #     tmp_p = tmp_p.reshape(2, 1)
        #     u_C1 = tmp_p[:, 0]
        #     tmp_p, _ = cv2.projectPoints(p_C.reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)
        #     u_C2 = tmp_p[0, 0, :2]
        #     print(f"Projected point {j}: {u_C2}")
        #     if u_C2[0] < camera_pinhole.width and u_C2[0] > 30 and u_C2[1] < camera_pinhole.height-20 and u_C2[1] > 0:
        #         cv2.circle(img, (int(u_C2[0]), int(u_C2[1])), 5, (0, 0, 255), -1)
        #         cv2.line(img, (int(u_C2_past[0]), int(u_C2_past[1])), (int(u_C2[0]), int(u_C2[1])), (255 - j*(255/future_steps), j*(255/future_steps), 0), 2)
        #         img_points.append([int(u_C2[0]), int(u_C2[1])])
        #         print(f"Valid point added: {u_C2}")
        #     else:
        #         print(f"Point out of bounds: {u_C2}")
        #     u_C2_past = u_C2
        
        for j in range(min(future_steps, points.shape[0])):  # Use points.shape[0] for rows
            p_C = p_C2[:, j]
            print(f"3D point in camera frame {j+1}: {p_C}")  # Debug 3D point
            tmp_p = camera_pinhole.project(p_C)
            u_C2 = tmp_p
            u_C2[1] += 490
            print(f"Projected point {j+1}: {u_C2}")
            print(f"Projected point {j+1}: {u_C2}")
            if u_C2[0] < camera_pinhole.width and u_C2[0] > 30 and u_C2[1] < camera_pinhole.height-20 and u_C2[1] > 0:
                cv2.circle(img, (int(u_C2[0]), int(u_C2[1])), 5, (0, 0, 255), -1)
                cv2.line(img, (int(u_C2_past[0]), int(u_C2_past[1])), (int(u_C2[0]), int(u_C2[1])), (255 - j*(255/future_steps), j*(255/future_steps), 0), 2)
                img_points.append([int(u_C2[0]), int(u_C2[1])])
                print(f"Valid point added: {u_C2}")
            else:
                print(f"Point out of bounds: {u_C2}")
            u_C2_past = u_C2

        print(f"Number of valid points for image {point}: {len(img_points)}")
        all_points.append(img_points)

        # Display image (optional, comment out if not needed)
        save_file_name = img_file_names[point]
        filename = os.path.join(folder_path, save_file_name)
        cv2.imwrite(filename, img)
        # cv2.imshow('image', img)
        # cv2.waitKey(1)  # Reduced delay for faster debugging

    cv2.destroyAllWindows()
    print(f"Total number of images processed: {len(all_points)}")
    print(f"Sample of all_points: {all_points[:5]}")

    with open('OPS_trial.json', 'w') as f:
        json.dump(all_points, f)
    print("Saved to OPS_grass_pixels.json")

if __name__ == '__main__':
    main()