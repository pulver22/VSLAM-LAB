from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import cv2
import pandas

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention


class StrayScanner_dataset(DatasetVSLAMLab):
    """Iphone datsets from StrayScanner app"""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "strayscanner") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        # self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        pass
        # todo add hf

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for raw, dst in (("depth", "depth_0"),):
            src = sequence_path / raw
            tgt = sequence_path / dst
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

        if not (sequence_path / "rgb_0").exists():
            self.extract_png_frames(sequence_path / "rgb.mp4", sequence_path / "rgb_0")



    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        rgb_csv = sequence_path / "rgb.csv"
        tmp = rgb_csv.with_suffix(".csv.tmp")

        odometry_csv = pandas.read_csv(sequence_path / "odometry.csv")
        assert isinstance(odometry_csv, pandas.DataFrame) and not odometry_csv.empty, "odometry.csv is missing or empty"

        rgb_path = "rgb_0/{idx:05d}.png"
        depth_path = "depth_0/{idx:06d}.png"

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])
            for i, (_, row) in enumerate(odometry_csv.iterrows()):
                ts = row["timestamp"]
                ts_r0_ns = int(float(ts) * 1e9)
                ts_d_ns = int(float(ts) * 1e9)

                w.writerow([ts_r0_ns, rgb_path.format(idx=i), ts_d_ns, depth_path.format(idx=i)])
        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        odometry_csv = pandas.read_csv(sequence_path / "odometry.csv")
        assert isinstance(odometry_csv, pandas.DataFrame) and not odometry_csv.empty, "odometry.csv is missing or empty"
        fx = odometry_csv[" fx"].iloc[0].item()
        fy = odometry_csv[" fy"].iloc[0].item()
        cx = odometry_csv[" cx"].iloc[0].item()
        cy = odometry_csv[" cy"].iloc[0].item()

        rgbd0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb+depth",
            "depth_name": "depth_0",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        odometry_csv = pandas.read_csv(sequence_path / "odometry.csv")
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            # Pending
        tmp.replace(groundtruth_csv)

    def extract_png_frames(self, video_path: Path, output_dir: Path):
        """
        Extract frames from a video based on a frequency in Hertz (frames per second) and save as PNG images.
        Also creates an rgb.txt file with timestamps and image paths.
        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the PNG files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Failed to get FPS from video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Validate and clamp ti/tf
        ti = 0.0
        tf = video_duration

        # Seek to start frame
        start_frame = int(round(ti * fps))
        end_frame = int(round(tf * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_interval = int(round(fps / self.rgb_hz))
        print(f"Video opened: {video_path}")
        print(f"Video FPS: {fps:.2f}")
        print(f"Extracting {self.rgb_hz} frames per second (every {frame_interval} frames).")
        print(f"Time range: {ti:.2f}s to {tf:.2f}s (frames {start_frame} to {end_frame})")

        frame_idx = start_frame
        saved_idx = 0
        timestamp_list = []

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval == 0:
                # Compute timestamp from the beginning of the video
                timestamp_nsec = int(1e9 * frame_idx / fps)

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save as PNG with 5-digit padded integer filename
                filename = output_dir / f"{saved_idx:05d}.png"
                cv2.imwrite(str(filename), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

                # Save timestamp and image path
                image_relative_path = output_dir / f"{saved_idx:05d}.png"
                timestamp_list.append((timestamp_nsec, str(image_relative_path)))
                saved_idx += 1

            frame_idx += 1

        cap.release()

    @staticmethod
    def _iter_entries(txt_path: Path, old_prefix: str, new_prefix: str) -> Iterable[tuple[str, str]]:
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing file: {txt_path}")
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                ts, path = s.split(None, 1)
                if path.startswith(old_prefix):
                    path = new_prefix + path[len(old_prefix):]
                yield ts, path
