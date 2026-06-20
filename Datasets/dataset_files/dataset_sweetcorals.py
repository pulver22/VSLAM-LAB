import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from huggingface_hub import HfApi, HfFileSystem, login
from huggingface_hub.utils import disable_progress_bars
from path_constants import HUGGINGFACE_TOKEN
from PIL import Image
from tqdm import tqdm


class SWEETCORALS_dataset(DatasetVSLAMLab):
    """SWEETCORALS dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "sweetcorals") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.repo_id = cfg["repo_id"]

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]

        # Get resolution size
        self.target_resolution = cfg["target_resolution"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0_raw"

        if rgb_path.exists():
            return
        rgb_path.mkdir(parents=True, exist_ok=True)

        remote_folder = self._remote_folder(sequence_name)

        if HUGGINGFACE_TOKEN is not None:
            login(token=HUGGINGFACE_TOKEN)
            token = HUGGINGFACE_TOKEN
        else:
            token = os.environ.get("HF_TOKEN")

        api = HfApi(token=token)
        fs = HfFileSystem(token=token)

        cache_file = self.dataset_path / "all_files_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                all_files = json.load(f)
        else:
            all_files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(all_files, f, indent=2)

        files = [f for f in all_files if f.startswith(remote_folder + "/")]

        disable_progress_bars()
        for remote_file in tqdm(files, desc="Downloading files", unit="file"):
            local_file = rgb_path / Path(remote_file).name
            fs.get_file(f"datasets/{self.repo_id}/{remote_file}", str(local_file))

    def create_rgb_folder(self, sequence_name: str) -> None:
        IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_raw_path = sequence_path / "rgb_0_raw"

        if rgb_path.exists():
            return
        if not rgb_raw_path.exists():
            return

        rgb_path.mkdir(parents=True, exist_ok=True)
        target_size = None
        init_size = None
        for file_path in tqdm(sorted(rgb_raw_path.iterdir()), desc="    resizing images"):
            if file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            with Image.open(file_path) as img:
                img.load()
                if target_size is None:
                    init_size = img.size
                    target_size = self._compute_scaled_size(img.size)

                if img.size != init_size:
                    print(f"{file_path.name} {img.size} != {init_size}")

                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(rgb_path / file_path.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        rgb_files = sorted(file_path.name for file_path in rgb_path.iterdir() if file_path.is_file())

        rgb = pd.DataFrame(
            {
                "ts_rgb_0 (ns)": [int(i * 1e9 / self.rgb_hz) for i in range(len(rgb_files))],
                "path_rgb_0": [f"rgb_0/{filename}" for filename in rgb_files],
            }
        )

        out = rgb[["ts_rgb_0 (ns)", "path_rgb_0"]]
        tmp = rgb_csv.with_suffix(".csv.tmp")
        try:
            out.to_csv(tmp, index=False)
            tmp.replace(rgb_csv)
        finally:
            tmp.unlink(missing_ok=True)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "unknown",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        tmp.replace(groundtruth_csv)

    def _compute_scaled_size(self, original_size: tuple[int, int]) -> tuple[int, int]:
        target_w, target_h = self.target_resolution
        orig_w, orig_h = original_size
        target_area = target_w * target_h

        scaled_h = int(np.sqrt(target_area * orig_h / orig_w))
        scaled_w = int(target_area / scaled_h)
        return scaled_w, scaled_h

    def _remote_folder(self, sequence_name: str) -> str:
        if sequence_name == "indonesia_tabuhan_p1":
            return "_indonesia_tabuhan_p1_20250210/corrected/images"

        if sequence_name == "indonesia_tabuhan_p2":
            return "indonesia_tabuhan_p2_20250210/raw/Q8_Left"

        if sequence_name == "indonesia_tabuhan_p3":
            return " indonesia_tabuhan_p3_20250210/raw/Q9_Left"

        if sequence_name == "indonesia_pemuteran_p1":
            return "indonesia_pemuteran_p1_20250213/raw/B1_Left"

        if sequence_name == "indonesia_pemuteran_p2":
            return "indonesia_pemuteran_p2_20250213/raw/B2_Left"

        if sequence_name == "indonesia_pemuteran_p3":
            return "indonesia_pemuteran_p3_20250213/raw/B3_Left"

        if sequence_name == "indonesia_watudodol_p1":
            return "indonesia_watudodol_p1_20250208/raw/Q1_Left"

        if sequence_name == "indonesia_watudodol_p2":
            return "indonesia_watudodol_p2_20250208/raw/Q2_Right"

        if sequence_name == "indonesia_watudodol_p3":
            return "indonesia_watudodol_p3_20250208/raw/Q3_Left"

        if sequence_name == "indonesia_watudodol_p4":
            return "indonesia_watudodol_p4_20250209/raw/Q4_Left"

        if sequence_name == "indonesia_watudodol_p5":
            return "indonesia_watudodol_p5_20250209/raw/Q5_Left"

        if sequence_name == "indonesia_watudodol_p6":
            return "indonesia_watudodol_p6_20250209/raw/Q6_Left"

        if sequence_name == "indonesia_banyuwangi_farm":
            return "indonesia_banyuwangi_farm_20250211/raw/F1_Left"
