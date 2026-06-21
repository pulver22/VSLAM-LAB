from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ["PATH"] = (
    str(REPO_ROOT / ".pixi/envs/vslamlab/bin")
    + os.pathsep
    + str(REPO_ROOT / ".pixi/envs/dpvo/bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from Datasets.get_dataset import get_dataset  # noqa: E402
from Evaluate.evaluate_functions import (  # noqa: E402
    _count_csv_data_rows,
    _count_text_data_rows,
    _rgb_exp_max_time_difference,
)
from Run.downsample_rgb_frames import downsample_rgb_frames  # noqa: E402
from path_constants import (  # noqa: E402
    TRAJECTORY_FILE_NAME,
    VSLAMLAB_BENCHMARK,
    VSLAMLAB_EVALUATION,
    VSLAM_LAB_EVALUATION_FOLDER,
)


SEQUENCES = {
    "ktima_2022_03": "March",
    "ktima_2022_04": "April",
    "ktima_2022_05": "May",
    "ktima_2022_06": "June",
    "ktima_2022_07": "July",
    "ktima_2022_09": "September",
}
METHODS = {
    "nominal_635": {
        "calibration_file": "dataset_blt_calibration_zed2i_nominal_635.yaml",
        "override": "1",
    },
    "factory_camera_info": {
        "calibration_file": "dataset_blt_calibration_factory_camera_info.yaml",
        "override": "",
    },
    "common_camera_info_median": {
        "calibration_file": "dataset_blt_calibration_common_camera_info_median.yaml",
        "override": "",
    },
}
DPVO_EXE = REPO_ROOT / ".pixi/envs/dpvo/bin/vslamlab_dpvo_mono"
SETTINGS_YAML = REPO_ROOT / "configs/baselines/dpvo_blt_no_loop.yaml"
NETWORK = REPO_ROOT / "Baselines/DPVO/dpvo.pth"


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    rendered = df.fillna("").astype(str)
    widths = {
        col: max(len(str(col)), *(len(value) for value in rendered[col].tolist()))
        for col in rendered.columns
    }
    lines = [
        "| " + " | ".join(str(col).ljust(widths[col]) for col in rendered.columns) + " |",
        "| " + " | ".join("-" * widths[col] for col in rendered.columns) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in rendered.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_traj(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: str(col).split()[0] for col in df.columns}
    df = df.rename(columns=rename)
    return df[["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]].sort_values("ts")


def _associate_by_timestamp(gt_df: pd.DataFrame, traj_df: pd.DataFrame, max_diff: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = gt_df.sort_values("ts").reset_index(drop=True)
    traj = traj_df.sort_values("ts").reset_index(drop=True)
    gt_ts = gt["ts"].to_numpy(dtype=np.float64)
    gt_indices: list[int] = []
    traj_indices: list[int] = []
    for traj_i, ts in enumerate(traj["ts"].to_numpy(dtype=np.float64)):
        pos = int(np.searchsorted(gt_ts, ts))
        candidates = []
        if pos < len(gt_ts):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(gt_ts[idx] - ts))
        if abs(gt_ts[best] - ts) <= max_diff:
            gt_indices.append(best)
            traj_indices.append(traj_i)
    return gt.iloc[gt_indices].reset_index(drop=True), traj.iloc[traj_indices].reset_index(drop=True)


def _sim3_align(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src_xyz)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ sign) / src_var) if src_var > 0 else 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    return (scale * (rotation @ src_xyz.T)).T + translation


def _read_tum(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _write_rgb_exp(sequence_path: Path, exp_folder: Path, dataset: Any, max_rgb: int) -> int:
    _, _, rows = downsample_rgb_frames(
        sequence_path / "rgb.csv",
        max_rgb,
        dataset.rgb_hz / 10,
        True,
    )
    with open(exp_folder / "rgb_exp.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _run_dpvo(sequence_path: Path, exp_folder: Path, env: dict[str, str]) -> tuple[bool, str, float]:
    cmd = [
        str(DPVO_EXE),
        "--sequence_path",
        str(sequence_path),
        "--calibration_yaml",
        str(sequence_path / "calibration.yaml"),
        "--rgb_csv",
        str(exp_folder / "rgb_exp.csv"),
        "--exp_folder",
        str(exp_folder),
        "--exp_it",
        "0",
        "--settings_yaml",
        str(SETTINGS_YAML),
        "--verbose",
        "0",
        "--mode",
        "mono",
        "--network",
        str(NETWORK),
    ]
    start = time.time()
    with open(exp_folder / "system_output_00000.txt", "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=f, text=True, env=env)
    duration = time.time() - start
    traj = exp_folder / f"00000_{TRAJECTORY_FILE_NAME}.csv"
    success = proc.returncode == 0 and traj.is_file() and traj.stat().st_size > 0
    comments = "" if success else f"dpvo returncode={proc.returncode}; trajectory_exists={traj.exists()}"
    return success, comments, duration


def _run_evaluation(exp_folder: Path, dataset: Any, env: dict[str, str]) -> tuple[bool, str, dict[str, Any]]:
    eval_dir = exp_folder / VSLAM_LAB_EVALUATION_FOLDER
    eval_dir.mkdir(parents=True, exist_ok=True)
    traj_csv = exp_folder / f"00000_{TRAJECTORY_FILE_NAME}.csv"
    gt_csv = exp_folder / "groundtruth.csv"
    rgb_exp_csv = exp_folder / "rgb_exp.csv"
    max_time_difference = _rgb_exp_max_time_difference(rgb_exp_csv, dataset.rgb_hz)
    traj_txt = eval_dir / f"00000_{TRAJECTORY_FILE_NAME}.txt"
    gt_txt = eval_dir / "groundtruth.txt"
    traj_tum = eval_dir / f"00000_{TRAJECTORY_FILE_NAME}.tum"
    gt_tum = eval_dir / "00000_gt.tum"

    traj_df = _normalize_traj(pd.read_csv(traj_csv))
    gt_df = _normalize_traj(pd.read_csv(gt_csv))
    traj_df.to_csv(traj_txt, header=False, index=False, sep=" ", lineterminator="\n")
    gt_df.to_csv(gt_txt, header=False, index=False, sep=" ", lineterminator="\n")

    zip_file = eval_dir / f"00000_{TRAJECTORY_FILE_NAME}.zip"
    if zip_file.exists():
        zip_file.unlink()
    proc = subprocess.run(
        [
            "evo_ape",
            "tum",
            str(gt_txt),
            str(traj_txt),
            "-va",
            "-as",
            "--t_max_diff",
            str(max_time_difference),
            "--save_results",
            str(zip_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if proc.returncode != 0 or not zip_file.exists():
        return False, f"evo_ape failed returncode={proc.returncode}: {(proc.stderr or proc.stdout)[-700:]}", {}

    with zipfile.ZipFile(zip_file, "r") as zf:
        stats = json.loads(zf.read("stats.json").decode("utf-8"))

    associated_gt, associated_traj = _associate_by_timestamp(gt_df, traj_df, max_time_difference)
    if len(associated_gt) == 0:
        return False, "no timestamp-associated poses for aligned plot", {}
    aligned_xyz = _sim3_align(
        associated_traj[["tx", "ty", "tz"]].to_numpy(dtype=float),
        associated_gt[["tx", "ty", "tz"]].to_numpy(dtype=float),
    )
    aligned_traj = associated_traj.copy()
    aligned_traj.loc[:, ["tx", "ty", "tz"]] = aligned_xyz
    associated_gt.to_csv(gt_tum, index=False, sep=" ")
    aligned_traj.to_csv(traj_tum, index=False, sep=" ")

    acc = pd.DataFrame([{"traj_name": f"00000_{TRAJECTORY_FILE_NAME}.txt", **stats}])
    acc.loc[0, "num_frames"] = _count_csv_data_rows(rgb_exp_csv)
    acc.loc[0, "num_tracked_frames"] = _count_text_data_rows(traj_txt) if traj_txt.exists() else 0
    acc.loc[0, "num_evaluated_frames"] = _count_text_data_rows(traj_tum, has_header=True) if traj_tum.exists() else 0
    acc.to_csv(eval_dir / "ate.csv", index=False)
    return True, "ate", acc.iloc[0].to_dict()


def _step_summary(exp_folder: Path) -> dict[str, Any]:
    eval_dir = exp_folder / VSLAM_LAB_EVALUATION_FOLDER
    summary: dict[str, Any] = {}
    for prefix, file_name in [
        ("estimated", f"00000_{TRAJECTORY_FILE_NAME}.tum"),
        ("gt", "00000_gt.tum"),
    ]:
        path = eval_dir / file_name
        if not path.exists():
            continue
        df = _read_tum(path)
        xyz = df[["tx", "ty", "tz"]].to_numpy(dtype=float)
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if len(steps) == 0:
            continue
        max_index = int(steps.argmax())
        summary[f"{prefix}_max_step_m"] = float(steps[max_index])
        summary[f"{prefix}_max_step_index"] = max_index
    return summary


def _step_rows(exp_root: Path, dataset_folder: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_dir in sorted((exp_root / "runs").glob("*")):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for seq_dir in sorted((method_dir / dataset_folder).glob("ktima_*")):
            eval_dir = seq_dir / VSLAM_LAB_EVALUATION_FOLDER
            for kind, file_name in [
                ("estimated", f"00000_{TRAJECTORY_FILE_NAME}.tum"),
                ("ground_truth", "00000_gt.tum"),
            ]:
                path = eval_dir / file_name
                if not path.exists():
                    continue
                df = _read_tum(path)
                xyz = df[["tx", "ty", "tz"]].to_numpy(dtype=float)
                ts = df["ts"].to_numpy(dtype=float)
                for i, step in enumerate(np.linalg.norm(np.diff(xyz, axis=0), axis=1)):
                    rows.append(
                        {
                            "method": method,
                            "sequence": seq_dir.name,
                            "month": SEQUENCES.get(seq_dir.name, seq_dir.name),
                            "kind": kind,
                            "index": i,
                            "ts0": ts[i],
                            "ts1": ts[i + 1],
                            "step_m": float(step),
                        }
                    )
    return rows


def _plot_trajectories(exp_root: Path, dataset_folder: str, method: str, result_rows: list[dict[str, Any]], fig_dir: Path) -> None:
    successful = [row for row in result_rows if row["method"] == method and row["success"]]
    if not successful:
        return
    cols = min(3, len(successful))
    rows = (len(successful) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, row in zip(axes_flat, successful):
        eval_dir = exp_root / "runs" / method / dataset_folder / row["sequence"] / VSLAM_LAB_EVALUATION_FOLDER
        gt = _read_tum(eval_dir / "00000_gt.tum")
        traj = _read_tum(eval_dir / f"00000_{TRAJECTORY_FILE_NAME}.tum")
        pca = PCA(n_components=2)
        gt_xy = pca.fit_transform(gt[["tx", "ty", "tz"]])
        traj_xy = pca.transform(traj[["tx", "ty", "tz"]])
        shift = gt_xy.min(axis=0)
        ax.plot(gt_xy[:, 0] - shift[0], gt_xy[:, 1] - shift[1], color="black", linewidth=1.4, label="ground truth")
        ax.plot(traj_xy[:, 0] - shift[0], traj_xy[:, 1] - shift[1], color="tab:red", linewidth=1.0, label="DPVO")
        ax.set_title(f"{row['month']} ({row['sequence']})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    for ax in axes_flat[len(successful):]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(fig_dir / f"trajectories_{method}.png", dpi=180, bbox_inches="tight")
    fig.savefig(fig_dir / f"trajectories_{method}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_rmse(results: pd.DataFrame, fig_dir: Path) -> None:
    ok = results[results["success"]].copy()
    if ok.empty:
        return
    ok["rmse"] = pd.to_numeric(ok["rmse"])
    pivot = ok.pivot(index="month", columns="method", values="rmse").reindex(list(SEQUENCES.values()))
    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=25)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_dir / "rmse_by_method.png", dpi=180, bbox_inches="tight")
    fig.savefig(fig_dir / "rmse_by_method.pdf", bbox_inches="tight")
    plt.close(fig)


def _method_env(base_env: dict[str, str], method: str, max_seconds: float) -> dict[str, str]:
    env = dict(base_env)
    env["BLT_CALIBRATION_FILE"] = METHODS[method]["calibration_file"]
    env["BLT_MAX_SECONDS"] = str(max_seconds)
    env["BLT_MAX_FRAMES"] = "0"
    if METHODS[method]["override"]:
        env["BLT_EXPERIMENTAL_CALIBRATION_OVERRIDE"] = METHODS[method]["override"]
    else:
        env.pop("BLT_EXPERIMENTAL_CALIBRATION_OVERRIDE", None)
    return env


def _run_one(method: str, sequence: str, exp_root: Path, max_rgb: int, max_seconds: float, base_env: dict[str, str]) -> dict[str, Any]:
    env = _method_env(base_env, method, max_seconds)
    os.environ.pop("BLT_EXPERIMENTAL_CALIBRATION_OVERRIDE", None)
    os.environ.update(env)
    dataset = get_dataset("blt", VSLAMLAB_BENCHMARK)
    seq_path = dataset.dataset_path / sequence
    exp_folder = exp_root / "runs" / method / dataset.dataset_folder / sequence
    if seq_path.exists():
        shutil.rmtree(seq_path)
    exp_folder.mkdir(parents=True, exist_ok=True)
    success = False
    evaluation = "failed"
    comments = ""
    duration = 0.0
    acc_info: dict[str, Any] = {}
    extracted_removed = False
    try:
        print(f"[{method}:{sequence}] extracting max_seconds={max_seconds}", flush=True)
        dataset.download_sequence(sequence)
        selected_frames = _write_rgb_exp(seq_path, exp_folder, dataset, max_rgb)
        shutil.copy2(seq_path / "groundtruth.csv", exp_folder / "groundtruth.csv")
        print(f"[{method}:{sequence}] selected_frames={selected_frames}; running DPVO", flush=True)
        success, comments, duration = _run_dpvo(seq_path, exp_folder, env)
        if success:
            eval_ok, evaluation_msg, acc_info = _run_evaluation(exp_folder, dataset, env)
            evaluation = evaluation_msg if eval_ok else "failed"
            success = bool(eval_ok)
            if not eval_ok:
                comments = (comments + "; " if comments else "") + evaluation_msg
    except Exception as exc:
        comments = f"{type(exc).__name__}: {exc}"
        print(f"[{method}:{sequence}] ERROR {comments}", flush=True)
    finally:
        if seq_path.exists():
            shutil.rmtree(seq_path)
        extracted_removed = not seq_path.exists()

    row = {
        "method": method,
        "calibration_file": METHODS[method]["calibration_file"],
        "month": SEQUENCES.get(sequence, sequence),
        "sequence": sequence,
        "success": bool(success),
        "evaluation": evaluation,
        "rmse": acc_info.get("rmse", ""),
        "mean": acc_info.get("mean", ""),
        "median": acc_info.get("median", ""),
        "std": acc_info.get("std", ""),
        "min": acc_info.get("min", ""),
        "max": acc_info.get("max", ""),
        "num_frames": acc_info.get("num_frames", 0),
        "num_tracked_frames": acc_info.get("num_tracked_frames", 0),
        "num_evaluated_frames": acc_info.get("num_evaluated_frames", 0),
        "duration_time_s": round(duration, 3),
        "comments": comments,
        "extracted_removed": extracted_removed,
    }
    row.update(_step_summary(exp_folder) if success else {})
    (exp_folder / "run_status.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    return row


def _preflight(sequences: list[str], methods: list[str], base_env: dict[str, str]) -> None:
    for required in [DPVO_EXE, SETTINGS_YAML, NETWORK]:
        if not required.exists():
            raise FileNotFoundError(required)
    for method in methods:
        cal_path = REPO_ROOT / "Datasets/dataset_files" / METHODS[method]["calibration_file"]
        if not cal_path.exists():
            raise FileNotFoundError(cal_path)
    env = _method_env(base_env, methods[0], 1.0)
    os.environ.pop("BLT_EXPERIMENTAL_CALIBRATION_OVERRIDE", None)
    os.environ.update(env)
    dataset = get_dataset("blt", VSLAMLAB_BENCHMARK)
    print(f"[preflight] dataset_path={dataset.dataset_path}", flush=True)
    print(f"[preflight] methods={methods}", flush=True)
    for sequence in sequences:
        bag = dataset.get_source_bag_path(sequence)
        print(f"[preflight] {sequence} bag_exists={bag.is_file()} path={bag}", flush=True)
        if not bag.is_file():
            raise FileNotFoundError(bag)
    for path in [Path("/tmp"), Path.home(), VSLAMLAB_EVALUATION]:
        usage = shutil.disk_usage(path)
        print(f"[preflight] free {path}: {usage.free / (1024 ** 3):.1f} GiB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BLT DPVO calibration matrix.")
    parser.add_argument("--output-name", default="exp_blt_dpvo_calibration_matrix")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--max-rgb", type=int, default=600)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    sequences = [sequence.strip() for sequence in args.sequences.split(",") if sequence.strip()]
    unknown_methods = sorted(set(methods).difference(METHODS))
    unknown_sequences = sorted(set(sequences).difference(SEQUENCES))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")

    base_env = os.environ.copy()
    if "BLT_KTIMA_ROOT" not in base_env:
        base_env["BLT_KTIMA_ROOT"] = "/media/pulver/PulverHDD/BACCHUS/ktima"
    Path(base_env.get("MPLCONFIGDIR", str(Path("/tmp") / "vslamlab_mpl"))).mkdir(parents=True, exist_ok=True)
    base_env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "vslamlab_mpl"))

    exp_root = VSLAMLAB_EVALUATION / args.output_name
    if exp_root.exists() and args.clean_output:
        shutil.rmtree(exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _preflight(sequences, methods, base_env)

    rows: list[dict[str, Any]] = []
    results_csv = exp_root / "blt_dpvo_calibration_matrix_results.csv"
    results_md = exp_root / "blt_dpvo_calibration_matrix_results.md"
    for method in methods:
        for sequence in sequences:
            row = _run_one(method, sequence, exp_root, args.max_rgb, args.max_seconds, base_env)
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            _write_markdown_table(pd.DataFrame(rows), results_md)

    results = pd.DataFrame(rows)
    results.to_csv(results_csv, index=False)
    display_cols = [
        "method",
        "month",
        "sequence",
        "success",
        "rmse",
        "mean",
        "median",
        "num_frames",
        "num_tracked_frames",
        "num_evaluated_frames",
        "estimated_max_step_m",
        "estimated_max_step_index",
        "comments",
    ]
    _write_markdown_table(results[display_cols], results_md)
    for method in methods:
        _plot_trajectories(exp_root, "BLT_dataset", method, rows, fig_dir)
    _plot_rmse(results, fig_dir)

    step_diag = pd.DataFrame(_step_rows(exp_root, "BLT_dataset"))
    step_diag.to_csv(exp_root / "step_diagnostics.csv", index=False)
    summary = (
        results[results["success"]]
        .assign(rmse=lambda frame: pd.to_numeric(frame["rmse"]))
        .groupby("method", as_index=False)["rmse"]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values(["mean", "median"])
    )
    summary.to_csv(exp_root / "method_summary.csv", index=False)
    _write_markdown_table(summary, exp_root / "method_summary.md")

    print(f"[done] results_csv={results_csv}", flush=True)
    print(f"[done] results_md={results_md}", flush=True)
    print(f"[done] figures={fig_dir}", flush=True)
    print(results.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
