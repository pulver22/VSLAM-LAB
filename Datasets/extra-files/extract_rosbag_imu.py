import rosbag
import os
import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--rosbag_path', type=str, required=True, help="Path to rosbag")
    parser.add_argument('--sequence_path', type=str, required=True, help="Output sequence path")
    parser.add_argument('--imu_topic', type=str, required=True, help="IMU topic name")
    parser.add_argument('--imu_name', type=str, default='0', help="IMU index/name, e.g. 0 -> imu_0.csv")

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

    with rosbag.Bag(rosbag_path, 'r') as bag:
        for topic, msg, t in tqdm(bag.read_messages(topics=[imu_topic]),
                                  desc=f'Extracting IMU from {imu_topic} ...'):
            try:
                # Prefer message header timestamp if available
                if hasattr(msg, 'header') and msg.header.stamp is not None:
                    ts_ns = msg.header.stamp.to_nsec()
                else:
                    ts_ns = t.to_nsec()

                data["ts (ns)"].append(int(ts_ns))

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