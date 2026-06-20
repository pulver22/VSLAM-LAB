import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Datasets.dataset_files.dataset_bacchus import BACCHUS_dataset
from Datasets.get_dataset import get_dataset, list_available_datasets


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
    assert dataset.image_transport == "compressed"
    assert dataset.decompressed_image_topic == "/front/zed_node/rgb_decompressed"
    assert dataset.get_image_topic_candidates() == [
        "/front/zed_node/rgb/image_rect_color/compressed",
        "/front/zed_node/rgb/image_rect_color",
    ]
    assert dataset.groundtruth_topic == "/odometry/gps"
    assert dataset.groundtruth_frame_source == "tf"
    assert dataset.tf_topics == ["/tf_static", "/tf"]


def test_bacchus_groundtruth_and_camera_frame_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("BACCHUS_GROUNDTRUTH_TOPIC", "/custom/rtk")
    monkeypatch.setenv("BACCHUS_CAMERA_FRAME", "front_camera_optical")

    dataset = get_dataset("bacchus", tmp_path)

    assert dataset.groundtruth_topic == "/custom/rtk"
    assert dataset.camera_frame == "front_camera_optical"


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


def test_bacchus_writes_calibration_yaml_from_metadata(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / "ktima_2022_06_08"
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path / "1654690000000000000.png"), image)

    dataset.create_calibration_yaml("ktima_2022_06_08")

    calibration_yaml = (sequence_path / "calibration.yaml").read_text(encoding="utf-8")
    assert "cam_name: rgb_0" in calibration_yaml
    assert "cam_model: pinhole" in calibration_yaml
    assert "focal_length: [525.0, 525.0]" in calibration_yaml
    assert "principal_point: [319.5, 239.5]" in calibration_yaml
    assert "image_dimension: [640, 480]" in calibration_yaml
