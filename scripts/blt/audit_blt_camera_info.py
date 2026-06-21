from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Datasets.get_dataset import get_dataset  # noqa: E402


SEQUENCES = [
    "ktima_2022_03",
    "ktima_2022_04",
    "ktima_2022_05",
    "ktima_2022_06",
    "ktima_2022_07",
    "ktima_2022_09",
]


def _field(message: Any, lowercase: str, uppercase: str, default: Any = None) -> Any:
    return getattr(message, lowercase, getattr(message, uppercase, default))


def _camera_info_row(sequence: str, topic: str, timestamp_ns: int, camera_info: Any, tracked: dict[str, Any]) -> dict[str, Any]:
    k = [float(v) for v in list(_field(camera_info, "k", "K", []))]
    if len(k) != 9:
        raise ValueError(f"{sequence}:{topic} CameraInfo has malformed K matrix")
    d = [float(v) for v in list(_field(camera_info, "d", "D", []))]
    width = int(getattr(camera_info, "width", 0))
    height = int(getattr(camera_info, "height", 0))
    fx, fy = k[0], k[4]
    cx, cy = k[2], k[5]
    tracked_fx, tracked_fy = [float(v) for v in tracked.get("focal_length", [0.0, 0.0])]
    tracked_cx, tracked_cy = [float(v) for v in tracked.get("principal_point", [0.0, 0.0])]
    diag = math.hypot(width, height)
    focal_geom_mean = math.sqrt(max(fx * fy, 1e-9))
    h_fov = math.degrees(2.0 * math.atan(width / (2.0 * fx))) if fx else 0.0
    v_fov = math.degrees(2.0 * math.atan(height / (2.0 * fy))) if fy else 0.0
    diag_fov = math.degrees(2.0 * math.atan(diag / (2.0 * focal_geom_mean))) if focal_geom_mean else 0.0
    focal_delta = max(
        abs(tracked_fx - fx) / max(abs(fx), 1.0),
        abs(tracked_fy - fy) / max(abs(fy), 1.0),
    )
    principal_delta = max(
        abs(tracked_cx - cx) / max(float(width), 1.0),
        abs(tracked_cy - cy) / max(float(height), 1.0),
    )
    return {
        "sequence": sequence,
        "topic": topic,
        "timestamp_ns": timestamp_ns,
        "frame_id": str(getattr(getattr(camera_info, "header", None), "frame_id", "")),
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_model": str(getattr(camera_info, "distortion_model", "")),
        "distortion_coefficients": " ".join(f"{value:.12g}" for value in d),
        "h_fov_deg": h_fov,
        "v_fov_deg": v_fov,
        "diag_fov_deg": diag_fov,
        "tracked_fx": tracked_fx,
        "tracked_fy": tracked_fy,
        "tracked_cx": tracked_cx,
        "tracked_cy": tracked_cy,
        "tracked_vs_camera_info_focal_max_rel_delta": focal_delta,
        "tracked_vs_camera_info_principal_max_rel_delta": principal_delta,
    }


def _decode_first_image_dimension(dataset: Any, bag_path: Path) -> tuple[str, int, int] | None:
    try:
        import cv2
    except ImportError:
        return None
    for topic, msgtype, _timestamp_ns, msg in dataset._iter_image_messages(
        bag_path,
        dataset.get_image_topic_candidates(),
    ):
        if "CompressedImage" in msgtype:
            image = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            height, width = image.shape[:2]
            return topic, int(width), int(height)
        width = int(getattr(msg, "width", 0))
        height = int(getattr(msg, "height", 0))
        if width and height:
            return topic, width, height
        return None
    return None


def _camera_info_priority(dataset: Any) -> list[str]:
    image_topic = dataset.image_topic.rstrip("/")
    if image_topic.endswith("/compressed"):
        image_topic = image_topic.removesuffix("/compressed")
    image_info_topic = f"{image_topic}/camera_info"
    return list(dict.fromkeys([image_info_topic, *dataset.camera_info_topics]))


def _sample_camera_info(dataset: Any, sequence: str, sample_count: int) -> tuple[list[dict[str, Any]], set[str], tuple[str, int, int] | None]:
    bag_path = dataset.get_source_bag_path(sequence)
    tracked = dataset.calibration[sequence]
    rows: list[dict[str, Any]] = []
    image_dimension = _decode_first_image_dimension(dataset, bag_path)
    with dataset._open_fast_ros1_stream(bag_path) as reader:
        available_topics = {connection.topic for connection in reader.connections}
        for topic in dataset.camera_info_topics:
            connections = [connection for connection in reader.connections if connection.topic == topic]
            if not connections:
                continue
            seen = 0
            for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                row = _camera_info_row(sequence, topic, timestamp_ns, msg, tracked)
                if image_dimension is not None:
                    image_topic, image_width, image_height = image_dimension
                    row["image_topic"] = image_topic
                    row["image_width"] = image_width
                    row["image_height"] = image_height
                    row["matches_first_image_dimension"] = (
                        row["width"] == image_width and row["height"] == image_height
                    )
                rows.append(row)
                seen += 1
                if seen >= sample_count:
                    break
    return rows, available_topics, image_dimension


def _preferred_rows(dataset: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = _camera_info_priority(dataset)
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(row["sequence"], []).append(row)

    preferred: list[dict[str, Any]] = []
    for sequence, sequence_rows in by_sequence.items():
        selected: dict[str, Any] | None = None
        for topic in priority:
            topic_rows = [row for row in sequence_rows if row["topic"] == topic]
            if topic_rows:
                selected = topic_rows[0]
                break
        if selected is None:
            selected = sequence_rows[0]
        selected = dict(selected)
        selected["preferred_for_calibration_candidate"] = True
        preferred.append(selected)
    return preferred


def _calibration_from_row(dataset: Any, row: dict[str, Any], source: str) -> dict[str, Any]:
    d = [float(v) for v in str(row["distortion_coefficients"]).split()] if row["distortion_coefficients"] else []
    distortion_model = row["distortion_model"].lower()
    if distortion_model in {"plumb_bob", "radtan"}:
        distortion_type = "radtan"
        cam_model = "radtan5" if len(d) >= 5 else "radtan4" if len(d) >= 4 else "pinhole"
    else:
        distortion_type = distortion_model or "unknown"
        cam_model = "pinhole"
    return {
        "source": source,
        "cam_name": "rgb_0",
        "cam_type": "rgb",
        "cam_model": cam_model,
        "distortion_type": distortion_type,
        "focal_length": [float(row["fx"]), float(row["fy"])],
        "principal_point": [float(row["cx"]), float(row["cy"])],
        "distortion_coefficients": d,
        "image_dimension": [int(row["width"]), int(row["height"])],
        "fps": float(dataset.rgb_hz),
    }


def _write_candidate_yamls(dataset: Any, preferred: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    by_sequence = {row["sequence"]: row for row in preferred}
    factory = {
        "metadata": {
            "source": "BLT bag CameraInfo sampled from configured ZED2i topics.",
            "note": "Per-month calibration candidate generated by scripts/blt/audit_blt_camera_info.py.",
        },
        "calibration": {},
    }
    for sequence in dataset.sequence_names:
        row = by_sequence[sequence]
        source = f"bag CameraInfo {row['topic']} at {row['timestamp_ns']}"
        factory["calibration"][sequence] = _calibration_from_row(dataset, row, source)

    dims = Counter((int(row["width"]), int(row["height"])) for row in preferred)
    width, height = dims.most_common(1)[0][0]
    distortion_lengths = Counter(len(str(row["distortion_coefficients"]).split()) for row in preferred)
    distortion_len = distortion_lengths.most_common(1)[0][0]
    median_row = {
        "fx": float(np.median([float(row["fx"]) for row in preferred])),
        "fy": float(np.median([float(row["fy"]) for row in preferred])),
        "cx": float(np.median([float(row["cx"]) for row in preferred])),
        "cy": float(np.median([float(row["cy"]) for row in preferred])),
        "width": width,
        "height": height,
        "distortion_model": Counter(row["distortion_model"] for row in preferred).most_common(1)[0][0],
        "distortion_coefficients": " ".join(
            f"{float(np.median([float(str(row['distortion_coefficients']).split()[i]) for row in preferred if len(str(row['distortion_coefficients']).split()) > i])):.12g}"
            for i in range(distortion_len)
        ),
    }
    common = {
        "metadata": {
            "source": "Median of preferred BLT bag CameraInfo samples.",
            "note": "Common calibration candidate generated by scripts/blt/audit_blt_camera_info.py.",
        },
        "calibration": {},
    }
    common_calibration = _calibration_from_row(dataset, median_row, "median preferred BLT bag CameraInfo")
    for sequence in dataset.sequence_names:
        common["calibration"][sequence] = dict(common_calibration)

    factory_path = output_dir / "dataset_blt_calibration_factory_camera_info.yaml"
    common_path = output_dir / "dataset_blt_calibration_common_camera_info_median.yaml"
    factory_path.write_text(yaml.safe_dump(factory, sort_keys=False), encoding="utf-8")
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    return factory_path, common_path


def _write_markdown(rows: list[dict[str, Any]], preferred: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# BLT CameraInfo Audit",
        "",
        "## Preferred Calibration Samples",
        "",
        "| sequence | topic | fx | fy | cx | cy | size | diag_fov_deg | focal_delta | principal_delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in preferred:
        lines.append(
            "| {sequence} | {topic} | {fx:.3f} | {fy:.3f} | {cx:.3f} | {cy:.3f} | {width}x{height} | {diag_fov_deg:.2f} | {tracked_vs_camera_info_focal_max_rel_delta:.3f} | {tracked_vs_camera_info_principal_max_rel_delta:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "## All Sampled Topics",
        "",
        "| sequence | topic | frame | size | fx | fy | cx | cy | distortion_model |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        lines.append(
            "| {sequence} | {topic} | {frame_id} | {width}x{height} | {fx:.3f} | {fy:.3f} | {cx:.3f} | {cy:.3f} | {distortion_model} |".format(**row)
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit BLT bag CameraInfo and write calibration candidates.")
    parser.add_argument("--benchmark-path", default="/tmp/vslamlab_blt_camera_info_audit_benchmark")
    parser.add_argument("--output-dir", default="/tmp/vslamlab_blt_camera_info_audit")
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("blt", args.benchmark_path)

    all_rows: list[dict[str, Any]] = []
    for sequence in SEQUENCES:
        bag_path = dataset.get_source_bag_path(sequence)
        if not bag_path.is_file():
            raise FileNotFoundError(bag_path)
        rows, available_topics, image_dimension = _sample_camera_info(dataset, sequence, args.sample_count)
        print(
            f"{sequence}: rows={len(rows)} camera_info_topics={sorted(topic for topic in dataset.camera_info_topics if topic in available_topics)} image_dimension={image_dimension}",
            flush=True,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No CameraInfo messages were sampled from BLT bags")

    preferred = _preferred_rows(dataset, all_rows)
    csv_path = output_dir / "blt_camera_info_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    preferred_csv_path = output_dir / "blt_camera_info_preferred.csv"
    with open(preferred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in preferred for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(preferred)
    md_path = output_dir / "blt_camera_info_audit.md"
    _write_markdown(all_rows, preferred, md_path)
    factory_path, common_path = _write_candidate_yamls(dataset, preferred, output_dir)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {preferred_csv_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    print(f"wrote {factory_path}", flush=True)
    print(f"wrote {common_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
