import os
import cv2
import argparse
import pandas as pd
from pathlib import Path

import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CompressedImage

def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--rosbag_path', type=str, help=f"rosbag path")
    parser.add_argument('--sequence_path', type=str, help=f"sequence_path")
    parser.add_argument('--image_topic', type=str, help=f"image topic")
    parser.add_argument('--cam', type=str, help=f"camera index")
    parser.add_argument('--storage_id', type=str, default='sqlite3',
                        help='Bag storage backend: sqlite3 or mcap')
    
    args = parser.parse_args()

    bridge = CvBridge()
    rosbag_path = args.rosbag_path
    image_topic = args.image_topic
    sequence_path = args.sequence_path
    rgb_path = os.path.join(sequence_path, f'rgb_{args.cam}')
    print(f"Extracting frames from {rosbag_path} with topic {image_topic} to {rgb_path} ...")

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=rosbag_path,
        storage_id=args.storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    is_compressed = image_topic.endswith('/compressed')
    msg_type = CompressedImage if is_compressed else Image

    rgb_files = []
    ts = []
    print(f"Extracting frames from {image_topic} ...")
    while reader.has_next():
        topic, data, t = reader.read_next()
        print(f"Read message on topic {topic} at time {t} ...")
        if topic != image_topic:
            continue

        try:
            msg = deserialize_message(data, msg_type)

            if is_compressed:
                cv_image = bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding='passthrough'
                )
            else:
                cv_image = bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough'
                )
        except Exception as e:
            print(f"Could not convert image on topic {topic}: {e}")
            continue
    
        image_name = f"{t}.png"
        image_path = os.path.join(rgb_path, image_name)

        rgb_files.append(f"rgb_{args.cam}/{image_name}")
        ts.append(int(t))

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

  
