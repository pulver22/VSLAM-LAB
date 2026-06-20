import rosbag
from cv_bridge import CvBridge
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--rosbag_path', type=str, help=f"rosbag path")
    parser.add_argument('--sequence_path', type=str, help=f"sequence_path")
    parser.add_argument('--image_topic', type=str, help=f"image topic")
    parser.add_argument('--cam', type=str, help=f"camera index")

    args = parser.parse_args()

    bridge = CvBridge()
    rosbag_path = args.rosbag_path
    image_topic = args.image_topic
    sequence_path = args.sequence_path
    rgb_path = os.path.join(sequence_path, f'rgb_{args.cam}')
    print(f"Extracting frames from {rosbag_path} with topic {image_topic} to {rgb_path} ...")

    with rosbag.Bag(rosbag_path, 'r') as bag:
        rgb_files = []
        ts = []
        for topic, msg, t in tqdm(bag.read_messages(topics=[image_topic]), desc=f'Extracting frames from {image_topic} ...'):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as e:
                print(f"Could not convert image: {e}")
                continue

            image_name = f"{t}.png"
            image_path = os.path.join(rgb_path,image_name)

            rgb_files.append(f"rgb_{args.cam}/{image_name}")
            ts_ns = t.to_nsec()
            ts.append(ts_ns)
            cv2.imwrite(image_path, cv_image)

    rgb_csv = Path(sequence_path) / "rgb.csv"
    ts_col = f"ts_rgb_{args.cam} (ns)"
    path_col = f"path_rgb_{args.cam}"

    new_rgb = pd.DataFrame({
        ts_col: pd.Series(ts, dtype="int64"),
        path_col: rgb_files,
    })
    
    if rgb_csv.exists():
        rgb = pd.read_csv(rgb_csv)

        # Prevent accidental duplicate column names
        overlap = [c for c in new_rgb.columns if c in rgb.columns]
        if overlap:
            raise ValueError(f"Columns already exist in {rgb_csv}: {overlap}")

        # Merge side-by-side, allowing different row counts
        out = rgb.join(new_rgb, how="outer")
    else:
        out = new_rgb

    tmp = rgb_csv.with_name(f"{rgb_csv.name}.tmp")
    try:
        out[ts_col] = out[ts_col].astype("Int64")
        out.to_csv(tmp, index=False)
        tmp.replace(rgb_csv)
    finally:
        if tmp.exists():
            tmp.unlink()

if __name__ == "__main__":
    main()

  
