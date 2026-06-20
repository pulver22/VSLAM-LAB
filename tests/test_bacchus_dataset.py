import csv
import yaml
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Datasets.dataset_files.dataset_bacchus import BACCHUS_dataset
from Datasets.get_dataset import get_dataset, list_available_datasets
from Evaluate.evaluate_functions import _count_csv_data_rows, _count_text_data_rows
from Evaluate.plot_functions import _trajectory_grid_shape


BACCHUS_SEQUENCES = {
    "ktima_2022_03": "03_march/rosbag_certh_compressed_2022-03-23-12-27-15.bag",
    "ktima_2022_04": "04_april/rosbag_compressed_2022-04-06-11-02-34.bag",
    "ktima_2022_05": "05_may/rosbag_compressed_2022-05-06.bag",
    "ktima_2022_06": "06_june/rosbag_compressed_2022-06-22-13-02-08.bag",
    "ktima_2022_07": "07_july/rosbag_compressed_2022-07-13-15-38-32.bag",
    "ktima_2022_09": "09_september/rosbag_compressed_2022-09-15-14-23-20.bag",
}
BACCHUS_SEQUENCE = "ktima_2022_04"
BACCHUS_BAG = "04_april/rosbag_compressed_2022-04-06-11-02-34.bag"
BACCHUS_KTIMA_ROOT = Path("/media/pulver/PulverHDD/BACCHUS/ktima")
BACCHUS_METADATA_BAG = BACCHUS_KTIMA_ROOT / BACCHUS_SEQUENCES["ktima_2022_03"]


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
    assert dataset.sequence_names == list(BACCHUS_SEQUENCES)
    assert dataset.sequence_nicknames == [name.replace("_", " ") for name in BACCHUS_SEQUENCES]
    assert dataset.modes == ["mono"]
    assert dataset.cam_models == ["pinhole"]
    assert dataset.source_bags == BACCHUS_SEQUENCES


@pytest.mark.parametrize("sequence_name", BACCHUS_SEQUENCES)
def test_bacchus_resolves_full_monthly_local_bag_paths(tmp_path, monkeypatch, sequence_name):
    root = tmp_path / "ktima"
    monkeypatch.setenv("BACCHUS_KTIMA_ROOT", str(root))
    dataset = get_dataset("bacchus", tmp_path)

    assert dataset.get_source_bag_path(sequence_name) == (
        root / BACCHUS_SEQUENCES[sequence_name]
    )


def test_bacchus_default_matrix_excludes_trimmed_and_2023_bags(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)

    source_paths = list(dataset.source_bags.values())

    assert all("trimmed" not in path for path in source_paths)
    assert all("/2023/" not in f"/{path}" for path in source_paths)
    assert "03_march/rosbag_certh_compressed_2022-03-23-12-27-15.bag" in source_paths
    assert "07_july/rosbag_compressed_2022-07-13-15-38-32.bag" in source_paths


def test_bacchus_accepts_existing_local_source_bag(tmp_path, monkeypatch):
    root = tmp_path / "ktima"
    bag_path = root / BACCHUS_BAG
    bag_path.parent.mkdir(parents=True)
    bag_path.write_bytes(b"bag")
    monkeypatch.setenv("BACCHUS_KTIMA_ROOT", str(root))
    dataset = get_dataset("bacchus", tmp_path)

    dataset.download_sequence_data(BACCHUS_SEQUENCE)

    assert (tmp_path / "BACCHUS" / BACCHUS_SEQUENCE).is_dir()


def test_bacchus_reports_missing_local_source_bag(tmp_path, monkeypatch):
    root = tmp_path / "ktima"
    monkeypatch.setenv("BACCHUS_KTIMA_ROOT", str(root))
    dataset = get_dataset("bacchus", tmp_path)

    with pytest.raises(FileNotFoundError, match="ktima_2022_04.*04_april"):
        dataset.download_sequence_data(BACCHUS_SEQUENCE)


def test_bacchus_uses_ros_compressed_transport_launch_defaults(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)

    assert dataset.image_topic == "/front/zed_node/rgb/image_rect_color/compressed"
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
    assert dataset.camera_frame == "front_left_camera_optical_frame"
    assert dataset.camera_frame_candidates == [
        "front_left_camera_optical_frame",
        "front_left_camera_frame",
        "front_camera_center",
        "front_base_link",
        "base_link",
    ]
    assert dataset.max_seconds == 0.0


def test_bacchus_local_metadata_bag_matches_configured_topic_contract(tmp_path):
    pytest.importorskip("rosbags")

    if not BACCHUS_METADATA_BAG.is_file():
        pytest.skip(f"Local BACCHUS metadata bag is not available: {BACCHUS_METADATA_BAG}")

    dataset = get_dataset("bacchus", tmp_path)
    expected_topics = {
        dataset.image_topic: "sensor_msgs/msg/CompressedImage",
        dataset.camera_info_topics[0]: "sensor_msgs/msg/CameraInfo",
        dataset.groundtruth_topic: "nav_msgs/msg/Odometry",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/tf": "tf2_msgs/msg/TFMessage",
    }
    samples = {}

    with BACCHUS_dataset._open_fast_ros1_stream(BACCHUS_METADATA_BAG) as reader:
        actual_topics = {connection.topic: connection.msgtype for connection in reader.connections}
        for topic, msgtype in expected_topics.items():
            assert actual_topics[topic] == msgtype

        sample_connections = [
            connection
            for connection in reader.connections
            if connection.topic in expected_topics
        ]
        for connection, timestamp_ns, rawdata in reader.messages(connections=sample_connections):
            if connection.topic in samples:
                continue
            samples[connection.topic] = reader.deserialize(rawdata, connection.msgtype)
            if set(samples) == set(expected_topics):
                break

    rgb_camera_info = samples["/front/zed_node/rgb/camera_info"]
    assert rgb_camera_info.header.frame_id == "front_left_camera_optical_frame"
    assert int(rgb_camera_info.width) == 1920
    assert int(rgb_camera_info.height) == 1080
    rgb_k = getattr(rgb_camera_info, "K", None)
    if rgb_k is None:
        rgb_k = getattr(rgb_camera_info, "k")
    assert float(rgb_k[0]) > 0.0
    assert float(rgb_k[4]) > 0.0
    assert samples["/front/zed_node/rgb/image_rect_color/compressed"].format == (
        "bgra8; jpeg compressed bgr8"
    )
    assert samples["/odometry/gps"].header.frame_id == "map"
    assert samples["/odometry/gps"].child_frame_id == "base_link"


def test_bacchus_selects_preferred_image_topic_from_available_topics():
    topics = [
        "/front/zed_node/rgb/image_rect_color",
        "/front/zed_node/rgb/image_rect_color/compressed",
        "/rear/image/compressed",
    ]

    selected = BACCHUS_dataset._select_first_available_topic(
        ["/front/zed_node/rgb/image_rect_color/compressed", "/front/zed_node/rgb/image_rect_color"],
        topics,
        label="image",
        bag_path=Path("bag.bag"),
    )

    assert selected == "/front/zed_node/rgb/image_rect_color/compressed"


def test_bacchus_missing_preferred_image_topic_names_available_topics():
    with pytest.raises(ValueError, match="Image topics.*bag.bag.*Available topics.*/rear/image"):
        BACCHUS_dataset._select_first_available_topic(
            ["/front/zed_node/rgb/image_rect_color/compressed"],
            ["/rear/image/compressed"],
            label="Image",
            bag_path=Path("bag.bag"),
        )


def test_bacchus_iter_image_messages_uses_fast_ros1_stream(monkeypatch, tmp_path):
    calls = []

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @property
        def connections(self):
            return [
                SimpleNamespace(
                    topic="/front/zed_node/rgb/image_rect_color/compressed",
                    msgtype="sensor_msgs/msg/CompressedImage",
                )
            ]

        def messages(self, connections):
            calls.append(connections)
            yield connections[0], 123, b"raw"

        def deserialize(self, rawdata, msgtype):
            return SimpleNamespace(format="jpeg", data=rawdata, msgtype=msgtype)

    monkeypatch.setattr(BACCHUS_dataset, "_open_fast_ros1_stream", lambda _path: FakeStream())

    rows = list(
        BACCHUS_dataset._iter_image_messages(
            tmp_path / "bag.bag",
            ["/front/zed_node/rgb/image_rect_color/compressed"],
        )
    )

    assert calls
    assert rows[0][0] == "/front/zed_node/rgb/image_rect_color/compressed"
    assert rows[0][1] == "sensor_msgs/msg/CompressedImage"
    assert rows[0][2] == 123
    assert rows[0][3].data == b"raw"


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
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    sequence_path.mkdir(parents=True)
    odom = _odometry("odom", "gps", (1.0, 2.0, 3.0))

    rows = BACCHUS_dataset._groundtruth_rows_from_odometry_messages(
        [(1654690000000000000, odom)],
        tf_edges={},
        target_frame="gps",
    )
    dataset._write_groundtruth_csv(BACCHUS_SEQUENCE, rows)

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


def test_bacchus_writes_decompressed_rgb_images_from_compressed_messages(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg_bytes = encoded.tobytes()

    image_path = BACCHUS_dataset._write_image_message(
        output_path=tmp_path,
        timestamp_ns=1654690000000000000,
        msgtype="sensor_msgs/msg/CompressedImage",
        msg=SimpleNamespace(format="bgra8; jpeg compressed bgr8", data=jpeg_bytes),
    )

    assert image_path == tmp_path / "1654690000000000000.png"
    assert image_path.read_bytes() != jpeg_bytes
    decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
    assert decoded.dtype == image.dtype
    assert decoded[:, :, 2].mean() > 240


def test_bacchus_writes_compressed_depth_messages(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    depth = np.array([[1000, 2000], [0, 1500]], dtype=np.uint16)
    ok, encoded = cv2.imencode(".png", depth)
    assert ok
    compressed_depth_header = b"\x00" * 12

    image_path = BACCHUS_dataset._write_depth_message(
        output_path=tmp_path,
        timestamp_ns=1654690000000000000,
        msgtype="sensor_msgs/msg/CompressedImage",
        msg=SimpleNamespace(
            format="16UC1; compressedDepth",
            data=compressed_depth_header + encoded.tobytes(),
        ),
    )

    assert image_path == tmp_path / "1654690000000000000.png"
    decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, depth)


def test_bacchus_create_rgb_csv_uses_current_vslamlab_contract(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    (rgb_path / "1654690000000000000.png").write_bytes(b"fake")
    (rgb_path / "1654690000100000000.png").write_bytes(b"fake")

    dataset.create_rgb_csv(BACCHUS_SEQUENCE)

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


def test_bacchus_create_rgb_csv_pairs_nearest_depth_frames(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    dataset.modes = ["mono", "rgbd"]
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    rgb_path = sequence_path / "rgb_0"
    depth_path = sequence_path / "depth_0"
    rgb_path.mkdir(parents=True)
    depth_path.mkdir()
    (rgb_path / "1000.png").write_bytes(b"fake")
    (rgb_path / "3000.png").write_bytes(b"fake")
    (depth_path / "900.png").write_bytes(b"fake")
    (depth_path / "2900.png").write_bytes(b"fake")
    (depth_path / "7000.png").write_bytes(b"fake")

    dataset.create_rgb_csv(BACCHUS_SEQUENCE)

    with open(sequence_path / "rgb.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "ts_rgb_0 (ns)": "1000",
            "path_rgb_0": "rgb_0/1000.png",
            "ts_depth_0 (ns)": "900",
            "path_depth_0": "depth_0/900.png",
        },
        {
            "ts_rgb_0 (ns)": "3000",
            "path_rgb_0": "rgb_0/3000.png",
            "ts_depth_0 (ns)": "2900",
            "path_depth_0": "depth_0/2900.png",
        },
    ]


def test_bacchus_writes_calibration_yaml_from_camera_info(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path / "1654690000000000000.png"), image)
    dataset._calibration_info_by_sequence[BACCHUS_SEQUENCE] = _camera_info(
        width=640,
        height=480,
        k=(610.0, 0.0, 320.0, 0.0, 612.0, 240.0, 0.0, 0.0, 1.0),
        d=(0.01, -0.02, 0.001, 0.002, 0.0),
    )

    dataset.create_calibration_yaml(BACCHUS_SEQUENCE)

    calibration_yaml = (sequence_path / "calibration.yaml").read_text(encoding="utf-8")
    assert "cam_name: rgb_0" in calibration_yaml
    assert "cam_type: rgb" in calibration_yaml
    assert "depth_name: depth_0" not in calibration_yaml
    assert "depth_factor:" not in calibration_yaml
    assert "cam_model: radtan5" in calibration_yaml
    assert "focal_length: [610.0, 612.0]" in calibration_yaml
    assert "principal_point: [320.0, 240.0]" in calibration_yaml
    assert "distortion_type: radtan" in calibration_yaml
    assert "distortion_coefficients: [0.01, -0.02, 0.001, 0.002, 0.0]" in calibration_yaml
    assert "image_dimension: [640, 480]" in calibration_yaml


def test_bacchus_rejects_camera_info_with_mismatched_image_dimensions(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path / "1654690000000000000.jpg"), image)
    dataset._calibration_info_by_sequence[BACCHUS_SEQUENCE] = _camera_info(
        width=640,
        height=480,
    )

    with pytest.raises(ValueError, match="CameraInfo.*640x480.*image dimensions.*1920x1080"):
        dataset.create_calibration_yaml(BACCHUS_SEQUENCE)


def test_bacchus_rejects_placeholder_calibration_for_hd_images(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path / "1654690000000000000.jpg"), image)
    dataset.camera_info_topics = []

    with pytest.raises(ValueError, match="placeholder.*1920x1080.*BACCHUS_ALLOW_PLACEHOLDER_CALIBRATION"):
        dataset.create_calibration_yaml(BACCHUS_SEQUENCE)


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


def test_bacchus_extraction_fingerprint_changes_with_source_and_limits(tmp_path, monkeypatch):
    root = tmp_path / "ktima"
    monkeypatch.setenv("BACCHUS_KTIMA_ROOT", str(root))
    monkeypatch.setenv("BACCHUS_MAX_SECONDS", "30")
    dataset = get_dataset("bacchus", tmp_path)

    fingerprint = dataset._extraction_fingerprint(BACCHUS_SEQUENCE)

    assert fingerprint["source_bag"] == str(root / BACCHUS_BAG)
    assert fingerprint["max_seconds"] == 30.0
    assert fingerprint["image_topic"] == "/front/zed_node/rgb/image_rect_color/compressed"
    assert "depth_topic" not in fingerprint
    assert "depth_camera_info_topics" not in fingerprint
    assert "depth_factor" not in fingerprint
    assert fingerprint["groundtruth_topic"] == "/odometry/gps"
    assert fingerprint["camera_frame"] == "front_left_camera_optical_frame"


def test_bacchus_stale_rgb_outputs_are_regenerated_when_fingerprint_changes(tmp_path, monkeypatch):
    root = tmp_path / "ktima"
    monkeypatch.setenv("BACCHUS_KTIMA_ROOT", str(root))
    monkeypatch.setenv("BACCHUS_MAX_SECONDS", "30")
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    rgb_path = sequence_path / "rgb_0"
    rgb_path.mkdir(parents=True)
    stale = rgb_path / "1000.jpg"
    stale.write_bytes(b"stale")
    dataset._update_diagnostics(
        BACCHUS_SEQUENCE,
        {"extraction_fingerprint": {"max_seconds": 360.0}},
    )

    calls = []

    def fake_extract(**kwargs):
        calls.append(kwargs)
        image_path = kwargs["output_path"] / "2000.jpg"
        image_path.write_bytes(b"fresh")
        return {
            "image_topic": "/front/zed_node/rgb/image_rect_color/compressed",
            "image_count": 1,
            "rgb_time_bounds_ns": [2000, 2000],
            "rgb_duration_s": 0.0,
            "rgb_inferred_fps": 0.0,
            "max_frames": 200,
            "max_seconds": 30.0,
        }

    monkeypatch.setattr(dataset, "_extract_rgb_images", fake_extract)

    dataset.create_rgb_folder(BACCHUS_SEQUENCE)

    assert calls
    assert not stale.exists()
    assert (rgb_path / "2000.jpg").exists()
    diagnostics = dataset._read_diagnostics(sequence_path / "bacchus_diagnostics.yaml")
    assert diagnostics["extraction_fingerprint"] == dataset._extraction_fingerprint(BACCHUS_SEQUENCE)


def test_bacchus_download_sequence_reruns_when_available_outputs_are_stale(tmp_path, monkeypatch):
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    (sequence_path / "rgb_0").mkdir(parents=True)
    for name in ("rgb.csv", "calibration.yaml", "groundtruth.csv"):
        (sequence_path / name).write_text("header\n1\n", encoding="utf-8")
    (sequence_path / "bacchus_diagnostics.yaml").write_text(
        yaml.safe_dump({"extraction_fingerprint": {"max_seconds": 999.0}}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(dataset, "download_process", lambda sequence_name: calls.append(sequence_name))

    dataset.download_sequence(BACCHUS_SEQUENCE)

    assert calls == [BACCHUS_SEQUENCE]


def test_bacchus_groundtruth_diagnostics_record_frame_chain_and_overlap(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    sequence_path = dataset.dataset_path / BACCHUS_SEQUENCE
    sequence_path.mkdir(parents=True)
    rows = [
        [1_000_000_000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2_000_000_000, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    dataset._write_groundtruth_csv(
        BACCHUS_SEQUENCE,
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


def test_bacchus_extraction_gate_reports_failures_and_blocks_experiments(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    good_sequence = "ktima_2022_03"
    bad_sequence = "ktima_2022_04"
    good_path = dataset.dataset_path / good_sequence
    bad_path = dataset.dataset_path / bad_sequence
    (good_path / "rgb_0").mkdir(parents=True)
    (bad_path / "rgb_0").mkdir(parents=True)
    for name in ("rgb.csv", "calibration.yaml", "groundtruth.csv"):
        (good_path / name).write_text("header\n1\n", encoding="utf-8")
    (good_path / "bacchus_diagnostics.yaml").write_text(
        yaml.safe_dump(
            {
                "image_topic": "/front/zed_node/rgb/image_rect_color/compressed",
                "image_count": 20,
                "rgb_inferred_fps": 15.0,
                "rgb_duration_s": 1.3,
                "calibration_source": "camera_info",
                "camera_info_topic": "/front/zed_node/rgb/camera_info",
                "camera_info_width": 1920,
                "camera_info_height": 1080,
                "groundtruth_topic": "/odometry/gps",
                "groundtruth_count": 40,
                "groundtruth_path_length_m": 2.0,
                "groundtruth_source_frame": "base_link",
                "groundtruth_target_frame": "front_left_camera_optical_frame",
                "tf_chain": ["base_link", "front_left_camera_optical_frame"],
                "tf_chain_dynamic": False,
                "rgb_groundtruth_overlap_ns": [1, 2],
                "extraction_fingerprint": dataset._extraction_fingerprint(good_sequence),
            }
        ),
        encoding="utf-8",
    )

    report = dataset.validate_extraction_gate([good_sequence, bad_sequence])

    assert report["ready_for_experiments"] is False
    assert report["sequences"][good_sequence]["status"] == "ok"
    assert report["sequences"][bad_sequence]["status"] == "failed"
    assert any("rgb.csv" in item for item in report["later_inspection"])
    assert any(item["sequence"] == bad_sequence for item in report["failures"])


def test_bacchus_extraction_gate_requires_identical_contract(tmp_path):
    dataset = get_dataset("bacchus", tmp_path)
    for sequence_name, image_topic in [
        ("ktima_2022_03", "/front/zed_node/rgb/image_rect_color/compressed"),
        ("ktima_2022_04", "/other/image/compressed"),
    ]:
        sequence_path = dataset.dataset_path / sequence_name
        (sequence_path / "rgb_0").mkdir(parents=True)
        for name in ("rgb.csv", "calibration.yaml", "groundtruth.csv"):
            (sequence_path / name).write_text("header\n1\n", encoding="utf-8")
        (sequence_path / "bacchus_diagnostics.yaml").write_text(
            yaml.safe_dump(
                {
                    "image_topic": image_topic,
                    "image_count": 20,
                    "rgb_inferred_fps": 15.0,
                    "rgb_duration_s": 1.3,
                    "calibration_source": "camera_info",
                    "camera_info_topic": "/front/zed_node/rgb/camera_info",
                    "camera_info_width": 1920,
                    "camera_info_height": 1080,
                    "groundtruth_topic": "/odometry/gps",
                    "groundtruth_count": 40,
                    "groundtruth_path_length_m": 2.0,
                    "groundtruth_source_frame": "base_link",
                    "groundtruth_target_frame": "front_left_camera_optical_frame",
                    "tf_chain": ["base_link", "front_left_camera_optical_frame"],
                    "tf_chain_dynamic": False,
                    "rgb_groundtruth_overlap_ns": [1, 2],
                    "extraction_fingerprint": dataset._extraction_fingerprint(sequence_name),
                }
            ),
            encoding="utf-8",
        )

    report = dataset.validate_extraction_gate(["ktima_2022_03", "ktima_2022_04"])

    assert report["ready_for_experiments"] is False
    assert any("image_topic differs" in item for item in report["later_inspection"])


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
    pytest.importorskip("huggingface_hub")
    from Baselines.baseline_files.baseline_dpvo import DPVO_baseline

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

    command = baseline.build_execute_command("00000", exp, dataset, BACCHUS_SEQUENCE)

    assert f"--settings_yaml {tmp_path / 'dpvo_bacchus_loop.yaml'}" in command
    assert str(baseline.settings_yaml) not in command
