import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Baselines.baseline_files.baseline_dpvo import DPVO_baseline
from Datasets.dataset_files.dataset_bacchus import BACCHUS_dataset
from Datasets.get_dataset import get_dataset, list_available_datasets
from Evaluate.evaluate_functions import _count_csv_data_rows, _count_text_data_rows
from Evaluate.plot_functions import _trajectory_grid_shape


def _vector(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _quaternion(x=0.0, y=0.0, z=0.0, w=1.0):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


def _header(frame_id):
    return SimpleNamespace(frame_id=frame_id)


def _pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=_vector(*position),
        orientation=_quaternion(*orientation),
    )


def _odometry(frame_id, child_frame_id, position, orientation=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        header=_header(frame_id),
        child_frame_id=child_frame_id,
        pose=SimpleNamespace(pose=_pose(position, orientation)),
    )


def _transform(parent_frame, child_frame, translation, rotation=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        header=_header(parent_frame),
        child_frame_id=child_frame,
        transform=SimpleNamespace(
            translation=_vector(*translation),
            rotation=_quaternion(*rotation),
        ),
    )


def _camera_info(
    width=1920,
    height=1080,
    frame_id="zed_left_camera_optical_frame",
    k=(1050.0, 0.0, 960.0, 0.0, 1055.0, 540.0, 0.0, 0.0, 1.0),
    d=(0.1, -0.05, 0.001, 0.002, 0.0),
    distortion_model="plumb_bob",
):
    return SimpleNamespace(
        header=_header(frame_id),
        width=width,
        height=height,
        k=list(k),
        d=list(d),
        distortion_model=distortion_model,
    )


def test_bacchus_dataset_is_registered(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)

    assert "bacchus" in list_available_datasets()
    assert dataset.dataset_name == "bacchus"
    assert dataset.dataset_folder == "BACCHUS"
    assert dataset.sequence_names == ["ktima_2022_06_08"]
    assert dataset.sequence_nicknames == ["ktima 2022 06 08"]
    assert dataset.modes == ["mono"]
    assert dataset.cam_models == ["pinhole"]


def test_bacchus_reports_missing_source_bag(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    dataset.source_bags["ktima_2022_06_08"] = tmp_path / "missing.bag"

    with pytest.raises(FileNotFoundError, match="missing.bag"):
        dataset.download_sequence_data("ktima_2022_06_08")


def test_bacchus_uses_ros_compressed_transport_launch_defaults(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)

    assert dataset.image_topic == "/front/zed_node/rgb/image_rect_color"
    assert dataset.camera_info_topics == [
        "/front/zed_node/rgb/camera_info",
        "/front/zed_node/left/camera_info",
        "/front/zed_node/rgb/image_rect_color/camera_info",
    ]
    assert dataset.image_transport == "compressed"
    assert dataset.decompressed_image_topic == "/front/zed_node/rgb_decompressed"
    assert dataset.get_image_topic_candidates() == [
        "/front/zed_node/rgb/image_rect_color/compressed",
        "/front/zed_node/rgb/image_rect_color",
    ]
    assert dataset.groundtruth_topic == "/odometry/gps"
    assert dataset.groundtruth_frame_source == "tf"
    assert dataset.tf_topics == ["/tf_static", "/tf"]
    assert dataset.max_seconds == 0.0


def test_bacchus_groundtruth_and_camera_frame_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("BACCHUS_GROUNDTRUTH_TOPIC", "/custom/rtk")
    monkeypatch.setenv("BACCHUS_CAMERA_FRAME", "front_camera_optical")

    dataset = get_dataset("bacchus", tmp_path)

    assert dataset.groundtruth_topic == "/custom/rtk"
    assert dataset.camera_frame == "front_camera_optical"


def test_bacchus_max_seconds_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("BACCHUS_MAX_SECONDS", "360")

    dataset = get_dataset("bacchus", tmp_path)

    assert dataset.max_seconds == 360.0


def test_bacchus_writes_groundtruth_csv_from_odometry(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / "ktima_2022_06_08"
    sequence_path.mkdir(parents=True)
    odom = _odometry("odom", "gps", (1.0, 2.0, 3.0))

    rows = BACCHUS_dataset._groundtruth_rows_from_odometry_messages(
        [(1654690000000000000, odom)],
        tf_edges={},
        target_frame="gps",
    )
    dataset._write_groundtruth_csv("ktima_2022_06_08", rows)

    with open(sequence_path / "groundtruth.csv", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    assert csv_rows == [
        {
            "ts (ns)": "1654690000000000000",
            "tx (m)": "1",
            "ty (m)": "2",
            "tz (m)": "3",
            "qx": "0",
            "qy": "0",
            "qz": "0",
            "qw": "1",
        }
    ]


def test_bacchus_applies_tf_chain_from_gps_to_camera_frame():
    odom = _odometry("odom", "gps", (10.0, 0.0, 0.0))
    tf_edges = BACCHUS_dataset._tf_edges_from_transforms(
        [_transform("gps", "front_camera", (1.0, 2.0, 3.0))]
    )

    rows = BACCHUS_dataset._groundtruth_rows_from_odometry_messages(
        [(1654690000000000000, odom)],
        tf_edges=tf_edges,
        target_frame="front_camera",
    )

    assert rows[0][0] == 1654690000000000000
    np.testing.assert_allclose(rows[0][1:4], [11.0, 2.0, 3.0])
    np.testing.assert_allclose(rows[0][4:8], [0.0, 0.0, 0.0, 1.0])


def test_bacchus_deduplicates_repeated_tf_edges():
    tf_edges = BACCHUS_dataset._tf_edges_from_transforms(
        [
            _transform("gps", "front_camera", (1.0, 2.0, 3.0)),
            _transform("gps", "front_camera", (1.0, 2.0, 3.0)),
        ]
    )

    assert len(tf_edges["gps"]) == 1
    assert len(tf_edges["front_camera"]) == 1


def test_bacchus_reuses_resolved_tf_chain_for_repeated_odometry_frame(monkeypatch):
    calls = []

    def fake_find(cls, tf_edges, source_frame, target_frame):
        calls.append((source_frame, target_frame))
        return np.eye(4)

    monkeypatch.setattr(
        BACCHUS_dataset,
        "_find_tf_chain_matrix",
        classmethod(fake_find),
    )

    odometry_messages = [
        (1654690000000000000, _odometry("odom", "gps", (1.0, 0.0, 0.0))),
        (1654690000100000000, _odometry("odom", "gps", (2.0, 0.0, 0.0))),
    ]
    rows = BACCHUS_dataset._groundtruth_rows_from_odometry_messages(
        odometry_messages,
        tf_edges={},
        target_frame="front_camera",
    )

    assert len(rows) == 2
    assert calls == [("gps", "front_camera")]


def test_bacchus_missing_tf_chain_names_frames_in_error():
    odom = _odometry("odom", "gps", (10.0, 0.0, 0.0))
    tf_edges = BACCHUS_dataset._tf_edges_from_transforms(
        [_transform("map", "base_link", (1.0, 2.0, 3.0))]
    )

    with pytest.raises(ValueError, match="gps.*front_camera.*available frames.*base_link.*map"):
        BACCHUS_dataset._groundtruth_rows_from_odometry_messages(
            [(1654690000000000000, odom)],
            tf_edges=tf_edges,
            target_frame="front_camera",
        )


def test_bacchus_decodes_compressed_image_messages():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    decoded = BACCHUS_dataset._message_to_bgr_image(
        "sensor_msgs/msg/CompressedImage",
        SimpleNamespace(data=encoded.tobytes()),
    )

    assert decoded.shape == image.shape
    assert decoded.dtype == image.dtype


def test_bacchus_writes_jpeg_compressed_messages_without_reencoding(tmp_path):
    jpeg_bytes = b"\xff\xd8fake-jpeg-payload\xff\xd9"

    image_path = BACCHUS_dataset._write_image_message(
        output_path=tmp_path,
        timestamp_ns=1654690000000000000,
        msgtype="sensor_msgs/msg/CompressedImage",
        msg=SimpleNamespace(format="jpeg", data=jpeg_bytes),
    )

    assert image_path == tmp_path / "1654690000000000000.jpg"
    assert image_path.read_bytes() == jpeg_bytes


def test_bacchus_create_rgb_csv_uses_current_vslamlab_contract(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / "ktima_2022_06_08"
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    (rgb_path / "1654690000000000000.png").write_bytes(b"fake")
    (rgb_path / "1654690000100000000.png").write_bytes(b"fake")

    dataset.create_rgb_csv("ktima_2022_06_08")

    with open(sequence_path / "rgb.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "ts_rgb_0 (ns)": "1654690000000000000",
            "path_rgb_0": "rgb_0/1654690000000000000.png",
        },
        {
            "ts_rgb_0 (ns)": "1654690000100000000",
            "path_rgb_0": "rgb_0/1654690000100000000.png",
        },
    ]


def test_bacchus_writes_calibration_yaml_from_camera_info(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / "ktima_2022_06_08"
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path / "1654690000000000000.png"), image)
    dataset._calibration_info_by_sequence["ktima_2022_06_08"] = _camera_info(
        width=640,
        height=480,
        k=(610.0, 0.0, 320.0, 0.0, 612.0, 240.0, 0.0, 0.0, 1.0),
        d=(0.01, -0.02, 0.001, 0.002, 0.0),
    )

    dataset.create_calibration_yaml("ktima_2022_06_08")

    calibration_yaml = (sequence_path / "calibration.yaml").read_text(encoding="utf-8")
    assert "cam_name: rgb_0" in calibration_yaml
    assert "cam_model: radtan5" in calibration_yaml
    assert "focal_length: [610.0, 612.0]" in calibration_yaml
    assert "principal_point: [320.0, 240.0]" in calibration_yaml
    assert "distortion_type: radtan" in calibration_yaml
    assert "distortion_coefficients: [0.01, -0.02, 0.001, 0.002, 0.0]" in calibration_yaml
    assert "image_dimension: [640, 480]" in calibration_yaml


def test_bacchus_rejects_placeholder_calibration_for_hd_images(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / "ktima_2022_06_08"
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path / "1654690000000000000.jpg"), image)
    dataset.camera_info_topics = []

    with pytest.raises(ValueError, match="placeholder.*1920x1080.*BACCHUS_ALLOW_PLACEHOLDER_CALIBRATION"):
        dataset.create_calibration_yaml("ktima_2022_06_08")


def test_bacchus_duration_limit_stops_after_elapsed_time(tmp_path):
    class FakeDataset(BACCHUS_dataset):
        written_timestamps = []

        @staticmethod
        def _iter_image_messages(*_args, **_kwargs):
            for ts in (1_000_000_000, 1_500_000_000, 2_100_000_000, 3_000_000_000):
                yield "/image", "sensor_msgs/msg/CompressedImage", ts, SimpleNamespace(format="jpeg", data=b"\xff\xd8x\xff\xd9")

        @staticmethod
        def _write_image_message(output_path, timestamp_ns, msgtype, msg):
            FakeDataset.written_timestamps.append(timestamp_ns)
            path = output_path / f"{timestamp_ns}.jpg"
            path.write_bytes(bytes(msg.data))
            return path

    FakeDataset.written_timestamps = []
    FakeDataset._extract_rgb_images(
        bag_path=tmp_path / "fake.bag",
        image_topics=["/image"],
        output_path=tmp_path,
        max_frames=0,
        max_seconds=1.0,
    )

    assert FakeDataset.written_timestamps == [1_000_000_000, 1_500_000_000]


def test_bacchus_groundtruth_diagnostics_record_frame_chain_and_overlap(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / "ktima_2022_06_08"
    sequence_path.mkdir(parents=True)
    rows = [
        [1_000_000_000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2_000_000_000, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    dataset._write_groundtruth_csv(
        "ktima_2022_06_08",
        rows,
        diagnostics={
            "groundtruth_topic": "/odometry/gps",
            "groundtruth_source_frame": "gps",
            "groundtruth_target_frame": "zed_left_camera_optical_frame",
            "tf_chain": ["gps", "base_link", "zed_left_camera_optical_frame"],
            "tf_chain_dynamic": False,
            "rgb_time_bounds_ns": [900_000_000, 2_100_000_000],
        },
    )

    diagnostics = dataset._read_diagnostics(sequence_path / "bacchus_diagnostics.yaml")

    assert diagnostics["groundtruth_topic"] == "/odometry/gps"
    assert diagnostics["groundtruth_source_frame"] == "gps"
    assert diagnostics["groundtruth_target_frame"] == "zed_left_camera_optical_frame"
    assert diagnostics["tf_chain"] == ["gps", "base_link", "zed_left_camera_optical_frame"]
    assert diagnostics["tf_chain_dynamic"] is False
    assert diagnostics["groundtruth_path_length_m"] == pytest.approx(1.0)
    assert diagnostics["rgb_groundtruth_overlap_ns"] == [1_000_000_000, 2_000_000_000]


def test_dynamic_tf_chain_uses_timestamp_specific_transform():
    odometry_messages = [
        (1_000_000_000, _odometry("odom", "gps", (0.0, 0.0, 0.0))),
        (2_000_000_000, _odometry("odom", "gps", (0.0, 0.0, 0.0))),
    ]
    tf_edges = BACCHUS_dataset._tf_edges_from_transforms(
        [
            _transform("gps", "front_camera", (1.0, 0.0, 0.0)),
            _transform("gps", "front_camera", (2.0, 0.0, 0.0)),
        ],
        timestamps_ns=[1_000_000_000, 2_000_000_000],
    )

    rows = BACCHUS_dataset._groundtruth_rows_from_odometry_messages(
        odometry_messages,
        tf_edges=tf_edges,
        target_frame="front_camera",
    )

    np.testing.assert_allclose(rows[0][1:4], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rows[1][1:4], [2.0, 0.0, 0.0])


def test_evaluation_counts_exclude_csv_headers(tmp_path):
    csv_path = tmp_path / "rgb_exp.csv"
    csv_path.write_text("ts,path\n1,a\n2,b\n", encoding="utf-8")
    text_path = tmp_path / "trajectory.tum"
    text_path.write_text("ts tx ty tz qx qy qz qw\n1 0 0 0 0 0 0 1\n", encoding="utf-8")

    assert _count_csv_data_rows(csv_path) == 2
    assert _count_text_data_rows(text_path, has_header=True) == 1


def test_trajectory_grid_shape_does_not_create_empty_five_panel_row():
    assert _trajectory_grid_shape(1) == (1, 1)
    assert _trajectory_grid_shape(4) == (1, 4)
    assert _trajectory_grid_shape(6) == (2, 4)


def test_dpvo_experiment_can_override_settings_yaml(tmp_path):
    baseline = DPVO_baseline()
    exp = SimpleNamespace(
        folder=tmp_path / "eval",
        parameters={
            "mode": "mono",
            "verbose": 0,
            "network": tmp_path / "dpvo.pth",
            "settings_yaml": tmp_path / "dpvo_bacchus_loop.yaml",
        },
    )
    dataset = SimpleNamespace(
        dataset_path=tmp_path / "benchmark" / "BACCHUS",
        dataset_folder="BACCHUS",
    )

    command = baseline.build_execute_command("00000", exp, dataset, "ktima_2022_06_08")

    assert f"--settings_yaml {tmp_path / 'dpvo_bacchus_loop.yaml'}" in command
    assert str(baseline.settings_yaml) not in command
