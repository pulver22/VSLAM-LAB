from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Datasets.dataset_files.dataset_blt import BLT_dataset  # noqa: E402
from Datasets.get_dataset import get_dataset  # noqa: E402
from path_constants import VSLAMLAB_BENCHMARK  # noqa: E402


SEQUENCES = [
    "ktima_2022_03",
    "ktima_2022_04",
    "ktima_2022_05",
    "ktima_2022_06",
    "ktima_2022_07",
    "ktima_2022_09",
]
IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
CAMERA_INFO_TYPES = {"sensor_msgs/msg/CameraInfo"}


def _parse_sequences(value: str) -> list[str]:
    if not value:
        return list(SEQUENCES)
    return [item.strip() for item in value.split(",") if item.strip()]


def _connection_counts(reader: Any) -> dict[int, int]:
    counts = {int(connection.id): 0 for connection in reader.connections}
    for chunk_info in getattr(reader.reader, "chunk_infos", []):
        for conn_id, count in getattr(chunk_info, "connection_counts", {}).items():
            counts[int(conn_id)] = counts.get(int(conn_id), 0) + int(count)
    return counts


def _topic_counts(reader: Any) -> dict[str, int]:
    by_id = _connection_counts(reader)
    counts: dict[str, int] = {}
    for connection in reader.connections:
        counts[connection.topic] = counts.get(connection.topic, 0) + by_id.get(int(connection.id), 0)
    return counts


def _topic_chunk_bounds(reader: Any) -> dict[str, tuple[int | None, int | None]]:
    connection_by_id = {int(connection.id): connection for connection in reader.connections}
    bounds: dict[str, tuple[int | None, int | None]] = {}
    for chunk_info in getattr(reader.reader, "chunk_infos", []):
        for conn_id, count in getattr(chunk_info, "connection_counts", {}).items():
            if int(count) <= 0:
                continue
            connection = connection_by_id.get(int(conn_id))
            if connection is None:
                continue
            first, last = bounds.get(connection.topic, (None, None))
            start = int(chunk_info.start_time)
            end = int(chunk_info.end_time)
            bounds[connection.topic] = (
                start if first is None else min(int(first), start),
                end if last is None else max(int(last), end),
            )
    return bounds


def _unique_topics(connections: list[Any]) -> list[str]:
    return sorted({connection.topic for connection in connections})


def _topic_connections(reader: Any, topics: set[str]) -> list[Any]:
    return [connection for connection in reader.connections if connection.topic in topics]


def _is_image_connection(connection: Any) -> bool:
    return connection.msgtype in IMAGE_TYPES


def _is_camera_info_connection(connection: Any) -> bool:
    return connection.msgtype in CAMERA_INFO_TYPES


def _depth_priority(topic: str) -> tuple[int, str]:
    lower = topic.lower()
    if lower.endswith("/compresseddepth"):
        return (0, topic)
    if "registered" in lower:
        return (1, topic)
    if lower.endswith("/image_raw") or lower.endswith("/image"):
        return (2, topic)
    return (3, topic)


def _candidate_depth_topics(connections: list[Any]) -> list[str]:
    topics = {
        connection.topic
        for connection in connections
        if _is_image_connection(connection)
        and "depth" in connection.topic.lower()
        and "camera_info" not in connection.topic.lower()
    }
    return sorted(topics, key=_depth_priority)


def _candidate_depth_camera_info_topics(connections: list[Any]) -> list[str]:
    return sorted(
        {
            connection.topic
            for connection in connections
            if _is_camera_info_connection(connection)
            and "depth" in connection.topic.lower()
        }
    )


def _first_available(preferred: list[str], available: set[str]) -> str:
    for topic in preferred:
        if topic in available:
            return topic
    return ""


def _sample_info(stream_kind: str, msgtype: str, msg: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "encoding": "",
        "width": "",
        "height": "",
        "dtype": "",
        "sample_decode_ok": False,
        "sample_error": "",
    }
    try:
        if msgtype.endswith("/CameraInfo"):
            info.update(
                {
                    "width": int(getattr(msg, "width", 0)),
                    "height": int(getattr(msg, "height", 0)),
                    "encoding": "CameraInfo",
                    "sample_decode_ok": True,
                }
            )
            return info
        if msgtype.endswith("/Image") and not msgtype.endswith("/CompressedImage"):
            info["encoding"] = str(getattr(msg, "encoding", ""))
        if msgtype.endswith("/CompressedImage"):
            info["encoding"] = str(getattr(msg, "format", ""))
        if stream_kind == "depth":
            image = BLT_dataset._message_to_depth_image(msgtype, msg)
        else:
            image = BLT_dataset._message_to_bgr_image(msgtype, msg)
        info.update(
            {
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "dtype": str(image.dtype),
                "sample_decode_ok": True,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised by local bag evidence.
        info["sample_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _nearest_delta_ms(source: list[int], target: list[int]) -> tuple[float | None, float | None]:
    if not source or not target:
        return None, None
    target_sorted = sorted(target)
    deltas: list[float] = []
    for timestamp in source:
        pos = bisect_left(target_sorted, timestamp)
        candidates = []
        if pos < len(target_sorted):
            candidates.append(target_sorted[pos])
        if pos > 0:
            candidates.append(target_sorted[pos - 1])
        if candidates:
            deltas.append(min(abs(timestamp - candidate) for candidate in candidates) / 1e6)
    if not deltas:
        return None, None
    deltas.sort()
    mid = len(deltas) // 2
    median = deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2.0
    return median, max(deltas)


def _infer_hz(count: int, first_ns: int | None, last_ns: int | None) -> float | None:
    if not first_ns or not last_ns or last_ns <= first_ns:
        return None
    return float(count) / ((last_ns - first_ns) / 1e9)


def _duration_s(first_ns: int | None, last_ns: int | None) -> float | None:
    if first_ns is None or last_ns is None or last_ns < first_ns:
        return None
    return (last_ns - first_ns) / 1e9


def _overlap_s(a_first: int | None, a_last: int | None, b_first: int | None, b_last: int | None) -> float:
    if None in {a_first, a_last, b_first, b_last}:
        return 0.0
    start = max(int(a_first), int(b_first))
    end = min(int(a_last), int(b_last))
    return max(0.0, (end - start) / 1e9)


def _stream_rows(
    sequence: str,
    bag_path: Path,
    reader: Any,
    rgb_topic: str,
    depth_topics: list[str],
    depth_camera_info_topics: list[str],
    pair_sample: int,
    expected_size: tuple[int, int],
    max_pair_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topics = {rgb_topic, *depth_topics, *depth_camera_info_topics}
    topics.discard("")
    counts = _topic_counts(reader)
    chunk_bounds = _topic_chunk_bounds(reader)
    connections = _topic_connections(reader, topics)
    msgtype_by_topic = {connection.topic: connection.msgtype for connection in connections}
    stream_kind_by_topic = {
        topic: "depth"
        for topic in depth_topics
    }
    if rgb_topic:
        stream_kind_by_topic[rgb_topic] = "rgb"
    for topic in depth_camera_info_topics:
        stream_kind_by_topic[topic] = "depth_camera_info"

    stats: dict[str, dict[str, Any]] = {
        topic: {
            "first_ns": chunk_bounds.get(topic, (None, None))[0],
            "last_ns": chunk_bounds.get(topic, (None, None))[1],
            "sample": {},
            "timestamps": [],
        }
        for topic in topics
    }

    pair_topics = {rgb_topic, *depth_topics}
    sample_topics = set(topics)
    for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
        topic = connection.topic
        item = stats[topic]
        if topic in pair_topics and len(item["timestamps"]) < min(pair_sample, counts.get(topic, 0)):
            item["timestamps"].append(int(timestamp_ns))
        if not item["sample"]:
            msg = reader.deserialize(rawdata, connection.msgtype)
            item["sample"] = _sample_info(stream_kind_by_topic[topic], connection.msgtype, msg)
        if all(stats[item_topic]["sample"] or counts.get(item_topic, 0) == 0 for item_topic in sample_topics) and all(
            len(stats[pair_topic]["timestamps"]) >= min(pair_sample, counts.get(pair_topic, 0))
            for pair_topic in pair_topics
            if pair_topic
        ):
            break

    rgb_stats = stats.get(rgb_topic, {})
    rows: list[dict[str, Any]] = []
    best_depth: dict[str, Any] | None = None
    width_expected, height_expected = expected_size
    camera_info_joined = ",".join(depth_camera_info_topics)

    for topic in [rgb_topic, *depth_topics, *depth_camera_info_topics]:
        if not topic:
            continue
        topic_stats = stats.get(topic, {})
        sample = topic_stats.get("sample") or {}
        row = {
            "sequence": sequence,
            "bag": str(bag_path),
            "stream_kind": stream_kind_by_topic.get(topic, ""),
            "topic": topic,
            "msgtype": msgtype_by_topic.get(topic, ""),
            "msg_count": counts.get(topic, 0),
            "first_ts_ns": topic_stats.get("first_ns"),
            "last_ts_ns": topic_stats.get("last_ns"),
            "duration_s": _duration_s(topic_stats.get("first_ns"), topic_stats.get("last_ns")),
            "inferred_hz": _infer_hz(
                counts.get(topic, 0),
                topic_stats.get("first_ns"),
                topic_stats.get("last_ns"),
            ),
            "encoding": sample.get("encoding", ""),
            "width": sample.get("width", ""),
            "height": sample.get("height", ""),
            "dtype": sample.get("dtype", ""),
            "sample_decode_ok": sample.get("sample_decode_ok", False),
            "sample_error": sample.get("sample_error", ""),
            "depth_camera_info_topics": camera_info_joined,
            "overlap_rgb_s": "",
            "median_pair_delta_ms_sample": "",
            "max_pair_delta_ms_sample": "",
            "dimension_matches_rgb_calibration": "",
            "usable_depth_candidate": "",
            "failure_cause": "",
        }
        if stream_kind_by_topic.get(topic) == "depth":
            median_delta, max_delta = _nearest_delta_ms(
                rgb_stats.get("timestamps", []),
                topic_stats.get("timestamps", []),
            )
            overlap = _overlap_s(
                rgb_stats.get("first_ns"),
                rgb_stats.get("last_ns"),
                topic_stats.get("first_ns"),
                topic_stats.get("last_ns"),
            )
            dimension_ok = (
                int(sample.get("width") or 0) == width_expected
                and int(sample.get("height") or 0) == height_expected
            )
            pairing_ok = median_delta is not None and median_delta <= max_pair_ms
            usable = bool(sample.get("sample_decode_ok")) and overlap > 0 and dimension_ok and pairing_ok
            failures = []
            if not sample.get("sample_decode_ok"):
                failures.append("sample_decode_failed")
            if overlap <= 0:
                failures.append("no_rgb_overlap")
            if not dimension_ok:
                failures.append("depth_dimension_mismatch")
            if not pairing_ok:
                failures.append("timestamp_pairing_exceeds_threshold")
            row.update(
                {
                    "overlap_rgb_s": overlap,
                    "median_pair_delta_ms_sample": median_delta,
                    "max_pair_delta_ms_sample": max_delta,
                    "dimension_matches_rgb_calibration": dimension_ok,
                    "usable_depth_candidate": usable,
                    "failure_cause": ",".join(failures),
                }
            )
            if usable and (
                best_depth is None
                or (
                    _depth_priority(topic),
                    float(median_delta or 1e12),
                )
                < (
                    _depth_priority(str(best_depth["topic"])),
                    float(best_depth["median_pair_delta_ms_sample"] or 1e12),
                )
            ):
                best_depth = row
        rows.append(row)

    summary = {
        "sequence": sequence,
        "rgb_topic": rgb_topic,
        "depth_camera_info_topics": depth_camera_info_topics,
        "best_depth_topic": best_depth["topic"] if best_depth else "",
        "usable_rgbd": best_depth is not None,
        "failure_cause": "" if best_depth else "no_usable_depth_topic",
        "expected_width": width_expected,
        "expected_height": height_expected,
        "max_pair_delta_ms": max_pair_ms,
    }
    return rows, summary


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    values = [{column: "" if row.get(column) is None else str(row.get(column, "")) for column in columns} for row in rows]
    widths = {
        column: max(len(column), *(len(row[column]) for row in values))
        for column in columns
    }
    lines = [
        "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |",
        "| " + " | ".join("-" * widths[column] for column in columns) + " |",
    ]
    for row in values:
        lines.append("| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover BLT RGB-D candidate topics without full extraction.")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "blt_rgbd_discovery"))
    parser.add_argument("--pair-sample", type=int, default=500)
    parser.add_argument("--max-pair-delta-ms", type=float, default=100.0)
    parser.add_argument("--expected-width", type=int, default=1920)
    parser.add_argument("--expected-height", type=int, default=1080)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("blt", VSLAMLAB_BENCHMARK)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for sequence in _parse_sequences(args.sequences):
        bag_path = dataset.get_source_bag_path(sequence)
        if not bag_path.is_file():
            summaries.append(
                {
                    "sequence": sequence,
                    "rgb_topic": "",
                    "depth_camera_info_topics": [],
                    "best_depth_topic": "",
                    "usable_rgbd": False,
                    "failure_cause": f"missing_bag:{bag_path}",
                }
            )
            continue
        with BLT_dataset._open_fast_ros1_stream(bag_path) as reader:
            available = {connection.topic for connection in reader.connections}
            rgb_topic = _first_available(dataset.get_image_topic_candidates(), available)
            depth_topics = _candidate_depth_topics(reader.connections)
            depth_camera_info_topics = _candidate_depth_camera_info_topics(reader.connections)
            if not rgb_topic:
                summaries.append(
                    {
                        "sequence": sequence,
                        "rgb_topic": "",
                        "depth_camera_info_topics": depth_camera_info_topics,
                        "best_depth_topic": "",
                        "usable_rgbd": False,
                        "failure_cause": "missing_rgb_topic",
                    }
                )
                continue
            topic_rows, summary = _stream_rows(
                sequence=sequence,
                bag_path=bag_path,
                reader=reader,
                rgb_topic=rgb_topic,
                depth_topics=depth_topics,
                depth_camera_info_topics=depth_camera_info_topics,
                pair_sample=args.pair_sample,
                expected_size=(args.expected_width, args.expected_height),
                max_pair_ms=args.max_pair_delta_ms,
            )
            rows.extend(topic_rows)
            summaries.append(summary)

    table_csv = output_dir / "blt_rgbd_topic_discovery.csv"
    summary_csv = output_dir / "blt_rgbd_discovery_summary.csv"
    summary_json = output_dir / "blt_rgbd_discovery_summary.json"
    table_md = output_dir / "blt_rgbd_topic_discovery.md"
    summary_md = output_dir / "blt_rgbd_discovery_summary.md"
    _write_csv(rows, table_csv)
    _write_csv(summaries, summary_csv)
    _write_markdown(
        rows,
        table_md,
        [
            "sequence",
            "stream_kind",
            "topic",
            "msgtype",
            "msg_count",
            "inferred_hz",
            "encoding",
            "width",
            "height",
            "overlap_rgb_s",
            "median_pair_delta_ms_sample",
            "usable_depth_candidate",
            "failure_cause",
        ],
    )
    _write_markdown(
        summaries,
        summary_md,
        [
            "sequence",
            "rgb_topic",
            "best_depth_topic",
            "depth_camera_info_topics",
            "usable_rgbd",
            "failure_cause",
        ],
    )
    payload = {
        "rgbd_ready_all_sequences": all(item.get("usable_rgbd") for item in summaries),
        "sequences": summaries,
        "topic_table": str(table_csv),
        "summary_table": str(summary_csv),
        "recommended_env": {},
    }
    depth_topics = {str(item.get("best_depth_topic")) for item in summaries if item.get("usable_rgbd")}
    camera_info_sets = {
        tuple(item.get("depth_camera_info_topics", []))
        for item in summaries
        if item.get("usable_rgbd")
    }
    if payload["rgbd_ready_all_sequences"] and len(depth_topics) == 1 and len(camera_info_sets) == 1:
        camera_info_topics = list(next(iter(camera_info_sets)))
        payload["recommended_env"] = {
            "BLT_MODES": "mono,rgbd",
            "BLT_DEPTH_TOPIC": next(iter(depth_topics)),
            "BLT_DEPTH_CAMERA_INFO_TOPIC": camera_info_topics[0] if camera_info_topics else "",
            "BLT_DEPTH_FACTOR": "1000.0",
        }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"topic_csv={table_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_json={summary_json}")
    print(f"rgbd_ready_all_sequences={payload['rgbd_ready_all_sequences']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
