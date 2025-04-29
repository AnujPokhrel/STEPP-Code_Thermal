#!/usr/bin/env python3
import os
import subprocess
import argparse
import pdb

def main():
    p = argparse.ArgumentParser(
        description="Batch-run future_pose.py on every folder/.pkl pair"
    )
    p.add_argument(
        "-r", "--root-dir", required=True,
        help="Parent folder containing subfolders and matching .pkl files"
    )
    p.add_argument(
        "-s", "--script", required=True,
        help="Path to your future_pose.py script"
    )
    p.add_argument(
        "-n", "--n-samples", type=int, default=5,
        help="How many sample points per frame"
    )
    args = p.parse_args()

    root = os.path.abspath(args.root_dir)
    script = os.path.abspath(args.script)

    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir) or entry == "all_poses":
            continue

        pkl = os.path.join(root, f"{entry}.pkl")
        if not os.path.isfile(pkl):
            print(f"⚠️  Skipping '{entry}': no matching {entry}.pkl")
            continue
        # pdb.set_trace()
        cmd = [
            "python3", script,
            "-r", entry,
            # "-p", pkl,
            "-n", str(args.n_samples),
        ]
        print(f"▶️  Processing '{entry}' …")
        subprocess.run(cmd, check=True)

    print("✅  Batch processing complete.")

if __name__ == "__main__":
    main()
