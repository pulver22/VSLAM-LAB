import csv
from pathlib import Path

import pytest

from Datasets.get_dataset import get_dataset, list_available_datasets


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
