#!/usr/bin/env python3
"""
Batch wrapper to run FeatureDataSet + main() over all pixel JSONs and image folders,
producing .npy datasets per sequence.
"""
import os
import argparse
from STEPP.utils.misc import load_image  # ensure STEPP is on PYTHONPATH
from make_dataset import FeatureDataSet, main as process_sequence
import numpy as np; 


def find_jsons(json_root):
    """Recursively find all '*_samples.json' under json_root."""
    matches = []
    for root, _, files in os.walk(json_root):
        for f in files:
            if f.endswith("_samples.json"):
                matches.append(os.path.join(root, f))
    return matches


def derive_image_folder(json_path, image_root):
    """Given JSON path .../Traj_Footprints_<base>/<base>_samples.json,
    return corresponding image folder: <image_root>/thermal_<base>_processed/"""
    base = os.path.basename(json_path).replace("_samples.json", "")
    folder = os.path.join(image_root, f"{base}")
    return folder, base


def main():
    parser = argparse.ArgumentParser(
        description="Batch-create .npy datasets from pixel JSONs and image folders"
    )
    parser.add_argument(
        "--json-root", required=True,
        help="Root directory containing Traj_Footprints_<base> folders"
    )
    parser.add_argument(
        "--image-root", required=True,
        help="Root directory containing thermal_<base>_processed image subfolders"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save resulting .npy files"
    )
    # parser.add_argument(
    #     "-n", "--n-samples", type=int, default=5,
    #     help="Number of sample points per frame (passed into FeatureDataSet)"
    # )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = find_jsons(args.json_root)
    if not json_files:
        print(f"No '*_samples.json' files found under {args.json_root}")
        return

    for json_path in sorted(json_files):
        image_folder, base = derive_image_folder(json_path, args.image_root)
        if not os.path.isdir(image_folder):
            print(f"⚠️  Missing image folder for '{base}': {image_folder}")
            continue

        print(f"▶️  Processing sequence '{base}' …")
        try:
            feat = FeatureDataSet(image_folder, json_path)
            data = process_sequence(feat)
            arr = np.array(data)
            good = ~np.isnan(arr).any(axis=1)
            arr = arr[good]
            out_path = os.path.join(args.output_dir, f"{base}.npy")
            with open(out_path, 'wb') as f:
                np.save(f, arr)
            print(f"✅  Saved dataset: {out_path}\n")
        except Exception as e:
            print(f"❌  Failed on '{base}': {e}\n")

if __name__ == '__main__':
    main()
