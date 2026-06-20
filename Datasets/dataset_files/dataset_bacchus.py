from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab


class BACCHUS_dataset(DatasetVSLAMLab):
    """BACCHUS ktima local rosbag dataset helper."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "bacchus") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]
        self.source_root = Path(
            os.environ.get(cfg["source_root_env"], cfg["source_root_default"])
        ).expanduser()
        self.source_bags = {
            name: self.source_root / rel_path
            for name, rel_path in cfg["source_bags"].items()
        }
        self.image_topic = os.environ.get(cfg["image_topic_env"], cfg["image_topic"])
        self.image_transport = cfg.get("image_transport", "raw")
        self.decompressed_image_topic = cfg.get("decompressed_image_topic", self.image_topic)
        self.max_frames = int(os.environ.get(cfg["max_frames_env"], cfg.get("max_frames", 0)))
        self.calibration = cfg["calibration"]

    def download_sequence_data(self, sequence_name: str) -> None:
        bag_path = self.get_source_bag_path(sequence_name)
        if not bag_path.is_file():
            raise FileNotFoundError(
                f"Missing BACCHUS source bag for '{sequence_name}': {bag_path}"
            )
        (self.dataset_path / sequence_name).mkdir(parents=True, exist_ok=True)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        if rgb_path.is_dir() and any(rgb_path.iterdir()):
            return

        rgb_path.mkdir(parents=True, exist_ok=True)
        self._extract_rgb_images(
            bag_path=self.get_source_bag_path(sequence_name),
            image_topics=self.get_image_topic_candidates(),
            output_path=rgb_path,
            max_frames=self.max_frames,
        )

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        tmp = rgb_csv.with_suffix(".csv.tmp")

        image_paths = sorted(
            p for p in rgb_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No BACCHUS RGB images found in {rgb_path}")

        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            for image_path in image_paths:
                writer.writerow([self._timestamp_from_image_name(image_path), f"rgb_0/{image_path.name}"])

        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        cal = self.calibration[sequence_name]
        rgb0: dict[str, Any] = {
            "cam_name": cal["cam_name"],
            "cam_type": cal["cam_type"],
            "cam_model": cal["cam_model"],
            "focal_length": cal["focal_length"],
            "principal_point": cal["principal_point"],
            "fps": float(cal.get("fps", self.rgb_hz)),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def get_source_bag_path(self, sequence_name: str) -> Path:
        if sequence_name not in self.source_bags:
            raise ValueError(f"Unknown BACCHUS sequence: {sequence_name}")
        return Path(self.source_bags[sequence_name]).expanduser()

    def get_image_topic_candidates(self) -> list[str]:
        image_topic = self.image_topic.rstrip("/")
        candidates: list[str] = []

        if self.image_transport == "compressed":
            if image_topic.endswith("/compressed"):
                candidates.append(image_topic)
                candidates.append(image_topic.removesuffix("/compressed"))
            else:
                candidates.append(f"{image_topic}/compressed")
                candidates.append(image_topic)
        else:
            candidates.append(image_topic)

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _timestamp_from_image_name(image_path: Path) -> int:
        try:
            return int(image_path.stem)
        except ValueError as exc:
            raise ValueError(
                f"BACCHUS RGB image filenames must be nanosecond timestamps: {image_path.name}"
            ) from exc

    @staticmethod
    def _extract_rgb_images(
        bag_path: Path,
        image_topics: list[str],
        output_path: Path,
        max_frames: int = 0,
    ) -> None:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise RuntimeError(
                "The BACCHUS dataset requires the 'rosbags' Python package to read source bags."
            ) from exc

        written = 0
        with AnyReader([bag_path]) as reader:
            connections = []
            selected_topic = None
            for image_topic in image_topics:
                connections = [c for c in reader.connections if c.topic == image_topic]
                if connections:
                    selected_topic = image_topic
                    break

            if not connections:
                topics = ", ".join(sorted({c.topic for c in reader.connections}))
                raise ValueError(
                    f"Image topics {image_topics} not found in {bag_path}. Available topics: {topics}"
                )

            messages = reader.messages(connections=connections)
            for connection, timestamp_ns, rawdata in tqdm(messages, desc=f"Extracting {selected_topic}"):
                msg = reader.deserialize(rawdata, connection.msgtype)
                image = BACCHUS_dataset._message_to_bgr_image(connection.msgtype, msg)
                cv2.imwrite(str(output_path / f"{timestamp_ns}.png"), image)
                written += 1
                if max_frames > 0 and written >= max_frames:
                    break

        if written == 0:
            raise RuntimeError(f"No images extracted from {bag_path}:{image_topic}")

    @staticmethod
    def _message_to_bgr_image(msgtype: str, msg: Any) -> np.ndarray:
        if msgtype.endswith("/CompressedImage"):
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode BACCHUS compressed image message")
            return image

        if not msgtype.endswith("/Image"):
            raise TypeError(f"Unsupported BACCHUS image message type: {msgtype}")

        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if encoding in {"rgb8", "bgr8"}:
            image = data.reshape((height, width, 3))
            if encoding == "rgb8":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        if encoding in {"mono8", "8uc1"}:
            return data.reshape((height, width))

        if encoding in {"bgra8", "rgba8"}:
            image = data.reshape((height, width, 4))
            if encoding == "rgba8":
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        raise ValueError(f"Unsupported BACCHUS image encoding: {msg.encoding}")
