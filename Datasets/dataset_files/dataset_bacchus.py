from __future__ import annotations

import csv
import os
from collections import deque
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
        self.groundtruth_topic = os.environ.get(
            cfg["groundtruth_topic_env"], cfg["groundtruth_topic"]
        )
        self.groundtruth_frame_source = cfg.get("groundtruth_frame_source", "tf")
        self.tf_topics = list(cfg.get("tf_topics", ["/tf_static", "/tf"]))
        self.camera_frame = os.environ.get(
            cfg.get("camera_frame_env", "BACCHUS_CAMERA_FRAME"),
            cfg.get("camera_frame", ""),
        )
        self.camera_frame_candidates = list(cfg.get("camera_frame_candidates", []))
        self.max_frames = int(os.environ.get(cfg["max_frames_env"], cfg.get("max_frames", 0)))
        self.calibration = cfg["calibration"]

    def download_sequence(self, sequence_name: str) -> None:
        if self.check_sequence_availability(sequence_name, verbose=True) == "available":
            return
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.download_process(sequence_name)

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
        existing_images = self._rgb_image_paths(rgb_path)
        if self.max_frames > 0 and len(existing_images) >= self.max_frames:
            return
        if self.max_frames <= 0 and existing_images:
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

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.dataset_path / sequence_name / "groundtruth.csv"
        if groundtruth_csv.is_file():
            return

        rows = self._extract_groundtruth_rows(
            bag_path=self.get_source_bag_path(sequence_name),
            rgb_time_bounds=self._rgb_time_bounds(sequence_name),
        )
        self._write_groundtruth_csv(sequence_name, rows)

    def check_sequence_integrity(self, sequence_name: str, verbose: bool) -> bool:
        complete_sequence = super().check_sequence_integrity(sequence_name, verbose)
        groundtruth_csv = self.dataset_path / sequence_name / "groundtruth.csv"
        if not groundtruth_csv.is_file():
            if verbose:
                from loguru import logger

                from utilities import ws

                logger.error(f"\n{ws(4)}Missing Groundtruth CSV: {groundtruth_csv} !!!!!")
            complete_sequence = False
        return complete_sequence

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

    def _rgb_time_bounds(self, sequence_name: str) -> tuple[int, int] | None:
        rgb_csv = self.dataset_path / sequence_name / "rgb.csv"
        if not rgb_csv.is_file():
            return None

        timestamps: list[int] = []
        with open(rgb_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                timestamps.append(int(row["ts_rgb_0 (ns)"]))

        if not timestamps:
            return None
        return min(timestamps), max(timestamps)

    def _resolve_groundtruth_target_frame(self, tf_edges: dict[str, list[tuple[str, np.ndarray]]]) -> str:
        if self.camera_frame:
            return self._clean_frame(self.camera_frame)

        available_frames = self._available_tf_frames(tf_edges)
        for frame in self.camera_frame_candidates:
            clean_frame = self._clean_frame(frame)
            if clean_frame in available_frames:
                return clean_frame

        camera_like = [
            frame for frame in available_frames
            if self._is_camera_like_frame(frame)
        ]
        if camera_like:
            return sorted(camera_like)[0]

        frames = ", ".join(sorted(available_frames)) or "<none>"
        raise ValueError(
            "Could not infer BACCHUS camera frame from TF. "
            f"Set BACCHUS_CAMERA_FRAME. Available frames: {frames}"
        )

    def _extract_groundtruth_rows(
        self,
        bag_path: Path,
        rgb_time_bounds: tuple[int, int] | None = None,
    ) -> list[list[Any]]:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise RuntimeError(
                "The BACCHUS dataset requires the 'rosbags' Python package to read source bags."
            ) from exc

        odometry_messages: list[tuple[int, Any]] = []
        tf_transforms: list[Any] = []
        min_rgb_ts = rgb_time_bounds[0] if rgb_time_bounds else None
        max_rgb_ts = rgb_time_bounds[1] if rgb_time_bounds else None
        margin_ns = int(2e9)

        with AnyReader([bag_path]) as reader:
            groundtruth_connections = [
                c for c in reader.connections if c.topic == self.groundtruth_topic
            ]
            if not groundtruth_connections:
                topics = ", ".join(sorted({c.topic for c in reader.connections}))
                raise ValueError(
                    f"Groundtruth topic '{self.groundtruth_topic}' not found in {bag_path}. "
                    f"Available topics: {topics}"
                )

            tf_connections = [
                c for c in reader.connections if c.topic in set(self.tf_topics)
            ]
            selected_connections = groundtruth_connections + tf_connections
            for connection, timestamp_ns, rawdata in tqdm(
                reader.messages(connections=selected_connections),
                desc=f"Extracting {self.groundtruth_topic} ground truth",
            ):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic in self.tf_topics:
                    tf_transforms.extend(self._transforms_from_tf_message(msg))
                    continue

                if min_rgb_ts is not None and timestamp_ns < min_rgb_ts - margin_ns:
                    continue
                if max_rgb_ts is not None and timestamp_ns > max_rgb_ts + margin_ns:
                    if odometry_messages:
                        break
                    continue

                if not connection.msgtype.endswith("/Odometry"):
                    raise TypeError(
                        f"Unsupported BACCHUS groundtruth message type on "
                        f"{self.groundtruth_topic}: {connection.msgtype}"
                    )
                odometry_messages.append((timestamp_ns, msg))

        if not odometry_messages:
            raise RuntimeError(
                f"No BACCHUS odometry groundtruth messages extracted from {bag_path}:"
                f"{self.groundtruth_topic}"
            )

        tf_edges = self._tf_edges_from_transforms(tf_transforms)
        target_frame = self._resolve_groundtruth_target_frame(tf_edges)
        return self._groundtruth_rows_from_odometry_messages(
            odometry_messages,
            tf_edges=tf_edges,
            target_frame=target_frame,
        )

    def _write_groundtruth_csv(self, sequence_name: str, rows: list[list[Any]]) -> None:
        sequence_path = self.dataset_path / sequence_name
        sequence_path.mkdir(parents=True, exist_ok=True)
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for row in rows:
                writer.writerow([row[0], *[self._format_csv_number(v) for v in row[1:]]])

        tmp.replace(groundtruth_csv)

    @staticmethod
    def _rgb_image_paths(rgb_path: Path) -> list[Path]:
        if not rgb_path.is_dir():
            return []
        return sorted(
            p for p in rgb_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

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
                BACCHUS_dataset._write_image_message(
                    output_path=output_path,
                    timestamp_ns=timestamp_ns,
                    msgtype=connection.msgtype,
                    msg=msg,
                )
                written += 1
                if max_frames > 0 and written >= max_frames:
                    break

        if written == 0:
            raise RuntimeError(f"No images extracted from {bag_path}:{image_topic}")

    @classmethod
    def _groundtruth_rows_from_odometry_messages(
        cls,
        odometry_messages: list[tuple[int, Any]],
        tf_edges: dict[str, list[tuple[str, np.ndarray]]],
        target_frame: str,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        target_frame = cls._clean_frame(target_frame)
        transform_cache: dict[tuple[str, str], np.ndarray] = {}
        for timestamp_ns, msg in odometry_messages:
            source_frame = cls._odometry_pose_frame(msg)
            pose_matrix = cls._pose_msg_to_matrix(msg.pose.pose)
            transform_key = (source_frame, target_frame)
            if transform_key not in transform_cache:
                transform_cache[transform_key] = cls._find_tf_chain_matrix(
                    tf_edges,
                    source_frame,
                    target_frame,
                )
            source_to_target = transform_cache[transform_key]
            target_pose = pose_matrix @ source_to_target
            translation = target_pose[:3, 3]
            quaternion = cls._quaternion_from_matrix(target_pose[:3, :3])
            rows.append([
                int(timestamp_ns),
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ])

        return rows

    @classmethod
    def _tf_edges_from_transforms(cls, transforms: list[Any]) -> dict[str, list[tuple[str, np.ndarray]]]:
        deduped_edges: dict[str, dict[str, np.ndarray]] = {}
        for transform_msg in transforms:
            parent_frame = cls._clean_frame(transform_msg.header.frame_id)
            child_frame = cls._clean_frame(transform_msg.child_frame_id)
            matrix = cls._transform_msg_to_matrix(transform_msg.transform)
            deduped_edges.setdefault(parent_frame, {}).setdefault(child_frame, matrix)
            deduped_edges.setdefault(child_frame, {}).setdefault(parent_frame, np.linalg.inv(matrix))
        return {
            frame: list(edges.items())
            for frame, edges in deduped_edges.items()
        }

    @staticmethod
    def _transforms_from_tf_message(msg: Any) -> list[Any]:
        if hasattr(msg, "transforms"):
            return list(msg.transforms)
        return [msg]

    @classmethod
    def _find_tf_chain_matrix(
        cls,
        tf_edges: dict[str, list[tuple[str, np.ndarray]]],
        source_frame: str,
        target_frame: str,
    ) -> np.ndarray:
        source_frame = cls._clean_frame(source_frame)
        target_frame = cls._clean_frame(target_frame)
        if source_frame == target_frame:
            return np.eye(4)

        queue = deque([(source_frame, np.eye(4))])
        visited = {source_frame}
        while queue:
            frame, matrix = queue.popleft()
            for next_frame, edge_matrix in tf_edges.get(frame, []):
                if next_frame in visited:
                    continue
                next_matrix = matrix @ edge_matrix
                if next_frame == target_frame:
                    return next_matrix
                visited.add(next_frame)
                queue.append((next_frame, next_matrix))

        available_frames = ", ".join(sorted(cls._available_tf_frames(tf_edges))) or "<none>"
        raise ValueError(
            f"No TF chain from '{source_frame}' to '{target_frame}'. "
            f"available frames: {available_frames}"
        )

    @classmethod
    def _odometry_pose_frame(cls, msg: Any) -> str:
        child_frame = cls._clean_frame(getattr(msg, "child_frame_id", ""))
        if child_frame:
            return child_frame
        return cls._clean_frame(msg.header.frame_id)

    @classmethod
    def _pose_msg_to_matrix(cls, pose: Any) -> np.ndarray:
        translation = cls._vector_to_array(pose.position)
        quaternion = cls._quaternion_to_array(pose.orientation)
        return cls._matrix_from_translation_quaternion(translation, quaternion)

    @classmethod
    def _transform_msg_to_matrix(cls, transform: Any) -> np.ndarray:
        translation = cls._vector_to_array(transform.translation)
        quaternion = cls._quaternion_to_array(transform.rotation)
        return cls._matrix_from_translation_quaternion(translation, quaternion)

    @staticmethod
    def _matrix_from_translation_quaternion(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = BACCHUS_dataset._rotation_matrix_from_quaternion(quaternion)
        matrix[:3, 3] = translation
        return matrix

    @staticmethod
    def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = BACCHUS_dataset._normalize_quaternion(quaternion)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
        trace = np.trace(rotation)
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            idx = int(np.argmax(np.diag(rotation)))
            if idx == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                w = (rotation[2, 1] - rotation[1, 2]) / scale
                x = 0.25 * scale
                y = (rotation[0, 1] + rotation[1, 0]) / scale
                z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif idx == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                w = (rotation[0, 2] - rotation[2, 0]) / scale
                x = (rotation[0, 1] + rotation[1, 0]) / scale
                y = 0.25 * scale
                z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                w = (rotation[1, 0] - rotation[0, 1]) / scale
                x = (rotation[0, 2] + rotation[2, 0]) / scale
                y = (rotation[1, 2] + rotation[2, 1]) / scale
                z = 0.25 * scale
        return BACCHUS_dataset._normalize_quaternion(np.array([x, y, z, w]))

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Zero-length quaternion is not valid")
        return quaternion / norm

    @staticmethod
    def _vector_to_array(vector: Any) -> np.ndarray:
        return np.array([float(vector.x), float(vector.y), float(vector.z)])

    @staticmethod
    def _quaternion_to_array(quaternion: Any) -> np.ndarray:
        return np.array([
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        ])

    @staticmethod
    def _clean_frame(frame: str) -> str:
        return str(frame or "").strip().lstrip("/")

    @staticmethod
    def _available_tf_frames(tf_edges: dict[str, list[tuple[str, np.ndarray]]]) -> set[str]:
        frames = set(tf_edges.keys())
        for edges in tf_edges.values():
            frames.update(frame for frame, _ in edges)
        return frames

    @staticmethod
    def _is_camera_like_frame(frame: str) -> bool:
        frame_lower = frame.lower()
        return (
            "camera" in frame_lower
            or "zed" in frame_lower
            or "rgb" in frame_lower
        ) and (
            "front" in frame_lower
            or "zed" in frame_lower
            or "rgb" in frame_lower
            or "camera" in frame_lower
        )

    @staticmethod
    def _format_csv_number(value: float) -> str:
        return format(float(value), ".12g")

    @staticmethod
    def _write_image_message(
        output_path: Path,
        timestamp_ns: int,
        msgtype: str,
        msg: Any,
    ) -> Path:
        if msgtype.endswith("/CompressedImage"):
            payload = bytes(msg.data)
            format_hint = str(getattr(msg, "format", "") or "").lower()
            if "jpeg" in format_hint or "jpg" in format_hint or payload.startswith(b"\xff\xd8"):
                image_path = output_path / f"{timestamp_ns}.jpg"
                image_path.write_bytes(payload)
                return image_path
            if "png" in format_hint or payload.startswith(b"\x89PNG\r\n\x1a\n"):
                image_path = output_path / f"{timestamp_ns}.png"
                image_path.write_bytes(payload)
                return image_path

        image = BACCHUS_dataset._message_to_bgr_image(msgtype, msg)
        image_path = output_path / f"{timestamp_ns}.png"
        cv2.imwrite(str(image_path), image)
        return image_path

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
