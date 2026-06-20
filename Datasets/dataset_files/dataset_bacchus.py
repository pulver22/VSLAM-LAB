from __future__ import annotations

import csv
import os
from collections import defaultdict, deque
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
        self.camera_info_topics = list(cfg.get("camera_info_topics", []))
        camera_info_override = os.environ.get(cfg.get("camera_info_topic_env", "BACCHUS_CAMERA_INFO_TOPIC"), "")
        if camera_info_override:
            self.camera_info_topics = [camera_info_override]
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
        self.max_seconds = float(os.environ.get(cfg.get("max_seconds_env", "BACCHUS_MAX_SECONDS"), cfg.get("max_seconds", 0.0)))
        self.allow_placeholder_calibration = os.environ.get(
            cfg.get("allow_placeholder_calibration_env", "BACCHUS_ALLOW_PLACEHOLDER_CALIBRATION"),
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self.calibration = cfg["calibration"]
        self._calibration_info_by_sequence: dict[str, Any] = {}
        self._calibration_info_topic_by_sequence: dict[str, str] = {}

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
        if self.max_seconds > 0 and self._rgb_images_cover_seconds(existing_images, self.max_seconds):
            return
        if self.max_frames <= 0 and self.max_seconds <= 0 and existing_images:
            return

        rgb_path.mkdir(parents=True, exist_ok=True)
        extraction_info = self._extract_rgb_images(
            bag_path=self.get_source_bag_path(sequence_name),
            image_topics=self.get_image_topic_candidates(),
            output_path=rgb_path,
            max_frames=self.max_frames,
            max_seconds=self.max_seconds,
        )
        self._update_diagnostics(sequence_name, extraction_info)

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
        cal = dict(self.calibration[sequence_name])
        camera_info = self._calibration_info_by_sequence.get(sequence_name)
        if camera_info is None and self.camera_info_topics:
            camera_info_result = self._extract_camera_info(
                bag_path=self.get_source_bag_path(sequence_name),
                camera_info_topics=self.camera_info_topics,
                rgb_time_bounds=self._rgb_time_bounds(sequence_name),
            )
            if camera_info_result is not None:
                camera_info_topic, camera_info = camera_info_result
                self._calibration_info_by_sequence[sequence_name] = camera_info
                self._calibration_info_topic_by_sequence[sequence_name] = camera_info_topic

        calibration_source = "dataset_yaml"
        calibration_diagnostics: dict[str, Any] = {}
        if camera_info is not None:
            cal.update(self._camera_info_to_calibration(camera_info))
            calibration_source = "camera_info"
            calibration_diagnostics = {
                "camera_info_topic": self._calibration_info_topic_by_sequence.get(sequence_name),
                "camera_info_frame": self._clean_frame(getattr(getattr(camera_info, "header", None), "frame_id", "")),
                "camera_info_width": int(getattr(camera_info, "width", 0)),
                "camera_info_height": int(getattr(camera_info, "height", 0)),
            }
        else:
            self._validate_placeholder_calibration(sequence_name, cal)

        rgb0: dict[str, Any] = {
            "cam_name": cal["cam_name"],
            "cam_type": cal["cam_type"],
            "cam_model": cal["cam_model"],
            "focal_length": cal["focal_length"],
            "principal_point": cal["principal_point"],
            "fps": float(cal.get("fps", self.rgb_hz)),
            "T_BS": np.eye(4),
        }
        if "distortion_type" in cal:
            rgb0["distortion_type"] = cal["distortion_type"]
        if "distortion_coefficients" in cal:
            rgb0["distortion_coefficients"] = cal["distortion_coefficients"]
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])
        self._update_diagnostics(
            sequence_name,
            {
                "calibration_source": calibration_source,
                "camera_info_topics": self.camera_info_topics,
                "focal_length": cal["focal_length"],
                "principal_point": cal["principal_point"],
                "distortion_coefficients": cal.get("distortion_coefficients", []),
                **calibration_diagnostics,
            },
        )

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.dataset_path / sequence_name / "groundtruth.csv"
        if groundtruth_csv.is_file():
            return

        rows, diagnostics = self._extract_groundtruth_rows(
            bag_path=self.get_source_bag_path(sequence_name),
            rgb_time_bounds=self._rgb_time_bounds(sequence_name),
        )
        self._write_groundtruth_csv(sequence_name, rows, diagnostics=diagnostics)

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

    @staticmethod
    def _rgb_images_cover_seconds(image_paths: list[Path], max_seconds: float) -> bool:
        if len(image_paths) < 2:
            return False
        timestamps = [BACCHUS_dataset._timestamp_from_image_name(path) for path in image_paths]
        duration_s = (max(timestamps) - min(timestamps)) / 1e9
        return duration_s >= max_seconds

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
    ) -> tuple[list[list[Any]], dict[str, Any]]:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise RuntimeError(
                "The BACCHUS dataset requires the 'rosbags' Python package to read source bags."
            ) from exc

        odometry_messages: list[tuple[int, Any]] = []
        tf_transforms: list[Any] = []
        tf_timestamps: list[int] = []
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
                    transforms = self._transforms_from_tf_message(msg)
                    tf_transforms.extend(transforms)
                    tf_timestamps.extend([timestamp_ns] * len(transforms))
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

        tf_edges = self._tf_edges_from_transforms(tf_transforms, timestamps_ns=tf_timestamps)
        target_frame = self._resolve_groundtruth_target_frame(tf_edges)
        diagnostics: dict[str, Any] = {}
        rows = self._groundtruth_rows_from_odometry_messages(
            odometry_messages,
            tf_edges=tf_edges,
            target_frame=target_frame,
            diagnostics=diagnostics,
        )
        diagnostics.update(
            {
                "groundtruth_topic": self.groundtruth_topic,
                "tf_topics": self.tf_topics,
                "groundtruth_target_frame": target_frame,
                "rgb_time_bounds_ns": list(rgb_time_bounds) if rgb_time_bounds else None,
            }
        )
        return rows, diagnostics

    def _write_groundtruth_csv(
        self,
        sequence_name: str,
        rows: list[list[Any]],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
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
        if diagnostics is not None:
            diagnostics = dict(diagnostics)
            diagnostics.update(self._groundtruth_row_diagnostics(rows, diagnostics))
            self._update_diagnostics(sequence_name, diagnostics)

    def _extract_camera_info(
        self,
        bag_path: Path,
        camera_info_topics: list[str],
        rgb_time_bounds: tuple[int, int] | None = None,
    ) -> tuple[str, Any] | None:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise RuntimeError(
                "The BACCHUS dataset requires the 'rosbags' Python package to read source bags."
            ) from exc

        min_rgb_ts = rgb_time_bounds[0] if rgb_time_bounds else None
        max_rgb_ts = rgb_time_bounds[1] if rgb_time_bounds else None
        margin_ns = int(2e9)
        with AnyReader([bag_path]) as reader:
            for camera_info_topic in camera_info_topics:
                connections = [c for c in reader.connections if c.topic == camera_info_topic]
                if not connections:
                    continue
                for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                    if min_rgb_ts is not None and timestamp_ns < min_rgb_ts - margin_ns:
                        continue
                    if max_rgb_ts is not None and timestamp_ns > max_rgb_ts + margin_ns:
                        break
                    return camera_info_topic, reader.deserialize(rawdata, connection.msgtype)
        return None

    @staticmethod
    def _camera_info_to_calibration(camera_info: Any) -> dict[str, Any]:
        distortion_values = getattr(camera_info, "d", getattr(camera_info, "D", []))
        distortion_coefficients = [float(v) for v in list(distortion_values)]
        distortion_model = str(getattr(camera_info, "distortion_model", "") or "").lower()
        if distortion_model in {"plumb_bob", "radtan"}:
            distortion_type = "radtan"
            if len(distortion_coefficients) >= 5:
                cam_model = "radtan5"
            elif len(distortion_coefficients) >= 4:
                cam_model = "radtan4"
            else:
                cam_model = "pinhole"
        else:
            distortion_type = distortion_model or "unknown"
            cam_model = "pinhole"

        k = list(getattr(camera_info, "k", getattr(camera_info, "K", [])))
        if len(k) != 9:
            raise ValueError("BACCHUS CameraInfo must contain a 3x3 K matrix")
        calibration = {
            "cam_model": cam_model,
            "focal_length": [float(k[0]), float(k[4])],
            "principal_point": [float(k[2]), float(k[5])],
        }
        if distortion_coefficients:
            calibration["distortion_type"] = distortion_type
            calibration["distortion_coefficients"] = distortion_coefficients
        return calibration

    def _validate_placeholder_calibration(self, sequence_name: str, calibration: dict[str, Any]) -> None:
        if self.allow_placeholder_calibration:
            return
        sequence_path = self.dataset_path / sequence_name
        image_dimension = self._first_rgb_image_dimension(sequence_path / calibration["cam_name"])
        if image_dimension is None:
            return
        width, height = image_dimension
        focal = [float(v) for v in calibration.get("focal_length", [])]
        principal = [float(v) for v in calibration.get("principal_point", [])]
        looks_placeholder = focal == [525.0, 525.0] and principal == [319.5, 239.5]
        looks_hd = width >= 1280 or height >= 720
        if looks_placeholder and looks_hd:
            raise ValueError(
                "BACCHUS placeholder calibration cannot be used for "
                f"{width}x{height} images. Extract CameraInfo or set "
                "BACCHUS_ALLOW_PLACEHOLDER_CALIBRATION=1 to acknowledge the limitation."
            )

    @staticmethod
    def _first_rgb_image_dimension(rgb_path: Path) -> tuple[int, int] | None:
        image_paths = BACCHUS_dataset._rgb_image_paths(rgb_path)
        if not image_paths:
            return None
        image = cv2.imread(str(image_paths[0]))
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height

    def _diagnostics_path(self, sequence_name: str) -> Path:
        return self.dataset_path / sequence_name / "bacchus_diagnostics.yaml"

    def _update_diagnostics(self, sequence_name: str, values: dict[str, Any]) -> None:
        diagnostics_path = self._diagnostics_path(sequence_name)
        diagnostics = self._read_diagnostics(diagnostics_path)
        diagnostics.update(self._yaml_safe(values))
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = diagnostics_path.with_suffix(".yaml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(diagnostics, f, sort_keys=True)
        tmp.replace(diagnostics_path)

    @staticmethod
    def _read_diagnostics(diagnostics_path: Path) -> dict[str, Any]:
        if not diagnostics_path.is_file():
            return {}
        with open(diagnostics_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def _groundtruth_row_diagnostics(
        cls,
        rows: list[list[Any]],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        if not rows:
            return {
                "groundtruth_count": 0,
                "groundtruth_path_length_m": 0.0,
                "groundtruth_time_bounds_ns": None,
                "rgb_groundtruth_overlap_ns": None,
            }
        positions = np.array([[row[1], row[2], row[3]] for row in rows], dtype=float)
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(rows) > 1 else 0.0
        gt_bounds = [int(rows[0][0]), int(rows[-1][0])]
        rgb_bounds = diagnostics.get("rgb_time_bounds_ns")
        overlap = None
        if rgb_bounds and rgb_bounds[0] is not None and rgb_bounds[1] is not None:
            overlap_start = max(int(rgb_bounds[0]), gt_bounds[0])
            overlap_end = min(int(rgb_bounds[1]), gt_bounds[1])
            if overlap_start <= overlap_end:
                overlap = [overlap_start, overlap_end]
        return {
            "groundtruth_count": len(rows),
            "groundtruth_path_length_m": path_length,
            "groundtruth_time_bounds_ns": gt_bounds,
            "rgb_groundtruth_overlap_ns": overlap,
        }

    @classmethod
    def _yaml_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._yaml_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._yaml_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return value

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

    @classmethod
    def _extract_rgb_images(
        cls,
        bag_path: Path,
        image_topics: list[str],
        output_path: Path,
        max_frames: int = 0,
        max_seconds: float = 0.0,
    ) -> dict[str, Any]:
        written = 0
        first_timestamp_ns: int | None = None
        last_timestamp_ns: int | None = None
        selected_topic: str | None = None
        for selected_topic, msgtype, timestamp_ns, msg in cls._iter_image_messages(
            bag_path,
            image_topics,
        ):
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            if max_seconds > 0 and (timestamp_ns - first_timestamp_ns) / 1e9 >= max_seconds:
                break
            cls._write_image_message(
                output_path=output_path,
                timestamp_ns=timestamp_ns,
                msgtype=msgtype,
                msg=msg,
            )
            written += 1
            last_timestamp_ns = timestamp_ns
            if max_frames > 0 and written >= max_frames:
                break

        if written == 0:
            raise RuntimeError(f"No images extracted from {bag_path}:{image_topics}")

        duration_s = 0.0
        if first_timestamp_ns is not None and last_timestamp_ns is not None:
            duration_s = (last_timestamp_ns - first_timestamp_ns) / 1e9
        fps = written / duration_s if duration_s > 0 else 0.0
        return {
            "image_topic": selected_topic,
            "image_count": written,
            "rgb_time_bounds_ns": [first_timestamp_ns, last_timestamp_ns],
            "rgb_duration_s": duration_s,
            "rgb_inferred_fps": fps,
            "max_frames": max_frames,
            "max_seconds": max_seconds,
        }

    @staticmethod
    def _iter_image_messages(
        bag_path: Path,
        image_topics: list[str],
    ):
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise RuntimeError(
                "The BACCHUS dataset requires the 'rosbags' Python package to read source bags."
            ) from exc

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
                yield selected_topic, connection.msgtype, timestamp_ns, msg

    @classmethod
    def _groundtruth_rows_from_odometry_messages(
        cls,
        odometry_messages: list[tuple[int, Any]],
        tf_edges: dict[str, list[tuple[str, np.ndarray]]],
        target_frame: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        target_frame = cls._clean_frame(target_frame)
        transform_cache: dict[tuple[str, str], np.ndarray] = {}
        chain_cache: dict[tuple[str, str], tuple[np.ndarray, list[str], bool]] = {}
        dynamic_tf = cls._tf_edges_are_dynamic(tf_edges)
        for timestamp_ns, msg in odometry_messages:
            source_frame = cls._odometry_pose_frame(msg)
            pose_matrix = cls._pose_msg_to_matrix(msg.pose.pose)
            transform_key = (source_frame, target_frame)
            if dynamic_tf:
                if transform_key in transform_cache:
                    source_to_target, tf_chain, tf_chain_dynamic = chain_cache[transform_key]
                else:
                    source_to_target, tf_chain, tf_chain_dynamic = cls._find_tf_chain(
                        tf_edges,
                        source_frame,
                        target_frame,
                        timestamp_ns=timestamp_ns,
                    )
                    if not tf_chain_dynamic:
                        chain_cache[transform_key] = (source_to_target, tf_chain, tf_chain_dynamic)
                        transform_cache[transform_key] = source_to_target
            else:
                if transform_key not in transform_cache:
                    if diagnostics is None:
                        source_to_target = cls._find_tf_chain_matrix(
                            tf_edges,
                            source_frame,
                            target_frame,
                        )
                        chain_cache[transform_key] = (source_to_target, [source_frame, target_frame], False)
                    else:
                        chain_cache[transform_key] = cls._find_tf_chain(
                            tf_edges,
                            source_frame,
                            target_frame,
                            timestamp_ns=timestamp_ns,
                        )
                        source_to_target = chain_cache[transform_key][0]
                    transform_cache[transform_key] = source_to_target
                source_to_target, tf_chain, tf_chain_dynamic = chain_cache[transform_key]

            if diagnostics is not None and "groundtruth_source_frame" not in diagnostics:
                diagnostics.update(
                    {
                        "groundtruth_source_frame": source_frame,
                        "groundtruth_target_frame": target_frame,
                        "tf_chain": tf_chain,
                        "tf_chain_dynamic": tf_chain_dynamic,
                    }
                )
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
    def _tf_edges_from_transforms(
        cls,
        transforms: list[Any],
        timestamps_ns: list[int] | None = None,
    ) -> dict[str, list[tuple]]:
        if timestamps_ns is None:
            timestamps_ns = [None] * len(transforms)
        deduped_edges: dict[str, dict[tuple[str, int | None], tuple]] = {}
        seen_static_pairs: set[tuple[str, str]] = set()
        for transform_msg, timestamp_ns in zip(transforms, timestamps_ns):
            parent_frame = cls._clean_frame(transform_msg.header.frame_id)
            child_frame = cls._clean_frame(transform_msg.child_frame_id)
            matrix = cls._transform_msg_to_matrix(transform_msg.transform)
            if timestamp_ns is None:
                pair = (parent_frame, child_frame)
                if pair in seen_static_pairs:
                    continue
                seen_static_pairs.add(pair)
                seen_static_pairs.add((child_frame, parent_frame))
            deduped_edges.setdefault(parent_frame, {})[(child_frame, timestamp_ns)] = (child_frame, matrix, timestamp_ns)
            deduped_edges.setdefault(child_frame, {})[(parent_frame, timestamp_ns)] = (parent_frame, np.linalg.inv(matrix), timestamp_ns)
        return {
            frame: list(edge_map.values())
            for frame, edge_map in deduped_edges.items()
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
        matrix, _, _ = cls._find_tf_chain(tf_edges, source_frame, target_frame)
        return matrix

    @classmethod
    def _find_tf_chain(
        cls,
        tf_edges: dict[str, list[tuple]],
        source_frame: str,
        target_frame: str,
        timestamp_ns: int | None = None,
    ) -> tuple[np.ndarray, list[str], bool]:
        source_frame = cls._clean_frame(source_frame)
        target_frame = cls._clean_frame(target_frame)
        if source_frame == target_frame:
            return np.eye(4), [source_frame], False

        queue = deque([(source_frame, np.eye(4), [source_frame], False)])
        visited = {source_frame}
        while queue:
            frame, matrix, chain, chain_dynamic = queue.popleft()
            for next_frame, edge_matrix, edge_dynamic in cls._selected_tf_edges(
                tf_edges.get(frame, []),
                timestamp_ns,
            ):
                if next_frame in visited:
                    continue
                next_matrix = matrix @ edge_matrix
                next_chain = [*chain, next_frame]
                next_dynamic = chain_dynamic or edge_dynamic
                if next_frame == target_frame:
                    return next_matrix, next_chain, next_dynamic
                visited.add(next_frame)
                queue.append((next_frame, next_matrix, next_chain, next_dynamic))

        available_frames = ", ".join(sorted(cls._available_tf_frames(tf_edges))) or "<none>"
        raise ValueError(
            f"No TF chain from '{source_frame}' to '{target_frame}'. "
            f"available frames: {available_frames}"
        )

    @classmethod
    def _selected_tf_edges(
        cls,
        edges: list[tuple],
        timestamp_ns: int | None,
    ) -> list[tuple[str, np.ndarray, bool]]:
        grouped: dict[str, list[tuple[np.ndarray, int | None]]] = defaultdict(list)
        for edge in edges:
            next_frame = cls._edge_frame(edge)
            grouped[next_frame].append((cls._edge_matrix(edge), cls._edge_timestamp(edge)))

        selected = []
        for next_frame, candidates in grouped.items():
            dynamic = any(ts is not None for _, ts in candidates) and len(candidates) > 1
            if timestamp_ns is None:
                matrix, _ = candidates[0]
            else:
                matrix, _ = min(
                    candidates,
                    key=lambda item: abs((item[1] if item[1] is not None else timestamp_ns) - timestamp_ns),
                )
            selected.append((next_frame, matrix, dynamic))
        return selected

    @staticmethod
    def _edge_frame(edge: tuple) -> str:
        return edge[0]

    @staticmethod
    def _edge_matrix(edge: tuple) -> np.ndarray:
        return edge[1]

    @staticmethod
    def _edge_timestamp(edge: tuple) -> int | None:
        return edge[2] if len(edge) > 2 else None

    @classmethod
    def _tf_edges_are_dynamic(cls, tf_edges: dict[str, list[tuple]]) -> bool:
        by_pair: dict[tuple[str, str], set[int | None]] = defaultdict(set)
        for frame, edges in tf_edges.items():
            for edge in edges:
                by_pair[(frame, cls._edge_frame(edge))].add(cls._edge_timestamp(edge))
        return any(len(timestamps) > 1 for timestamps in by_pair.values())

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
            frames.update(BACCHUS_dataset._edge_frame(edge) for edge in edges)
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
