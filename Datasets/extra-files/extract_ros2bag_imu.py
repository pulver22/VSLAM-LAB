import ros2bag
import os
import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path


import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu

def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--rosbag_path', type=str, required=True, help="Path to rosbag")
    parser.add_argument('--sequence_path', type=str, required=True, help="Output sequence path")
    parser.add_argument('--imu_topic', type=str, required=True, help="IMU topic name")
    parser.add_argument('--imu_name', type=str, default='0', help="IMU index/name, e.g. 0 -> imu_0.csv")
    parser.add_argument('--storage_id', type=str, default='sqlite3',
                        help='Bag storage backend: sqlite3 or mcap')
    
    args = parser.parse_args()

    rosbag_path = args.rosbag_path
    imu_topic = args.imu_topic
    sequence_path = args.sequence_path
    imu_csv = Path(sequence_path) / f"imu_{args.imu_name}.csv"

    print(f"Extracting IMU messages from {rosbag_path} with topic {imu_topic} to {imu_csv} ...")

    data = {
        "ts (ns)": [],
        "wx (rad s^-1)": [],
        "wy (rad s^-1)": [],
        "wz (rad s^-1)": [],
        "ax (m s^-2)": [],
        "ay (m s^-2)": [],
        "az (m s^-2)": [],
    }

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=rosbag_path,
        storage_id=args.storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    while reader.has_next():
        topic, raw_data, t = reader.read_next()

        if topic != imu_topic:
            continue

        try:
            msg = deserialize_message(raw_data, Imu)
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                ts_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                if ts_ns == 0:
                    ts_ns = int(t)
            else:
                ts_ns = int(t)

            data["ts (ns)"].append(ts_ns)

            data["wx (rad s^-1)"].append(msg.angular_velocity.x)
            data["wy (rad s^-1)"].append(msg.angular_velocity.y)
            data["wz (rad s^-1)"].append(msg.angular_velocity.z)

            data["ax (m s^-2)"].append(msg.linear_acceleration.x)
            data["ay (m s^-2)"].append(msg.linear_acceleration.y)
            data["az (m s^-2)"].append(msg.linear_acceleration.z)

        except Exception as e:
            print(f"Could not parse IMU message at {t}: {e}")
            continue

    out = pd.DataFrame(data)

    tmp = imu_csv.with_name(f"{imu_csv.name}.tmp")
    try:
        out.to_csv(tmp, index=False)
        tmp.replace(imu_csv)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"Saved {len(out)} IMU messages to {imu_csv}")


if __name__ == "__main__":
    main()