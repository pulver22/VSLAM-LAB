from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PIXI_EXE = Path.home() / ".pixi/bin/pixi"
os.environ["PATH"] = (
    (str(PIXI_EXE.parent) + os.pathsep if PIXI_EXE.exists() else "")
    + str(REPO_ROOT / ".pixi/envs/vslamlab/bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from Baselines.get_baseline import get_baseline  # noqa: E402
from Datasets.get_dataset import get_dataset  # noqa: E402
from Evaluate.evaluate_functions import (  # noqa: E402
    _count_csv_data_rows,
    _count_text_data_rows,
    _rgb_exp_max_time_difference,
)
from path_constants import (  # noqa: E402
    TRAJECTORY_FILE_NAME,
    VSLAMLAB_EVALUATION,
    VSLAM_LAB_EVALUATION_FOLDER,
)
from scripts.blt.run_blt_dpvo_calibration_matrix import (  # noqa: E402
    SEQUENCES,
    _associate_by_timestamp,
    _normalize_traj,
    _read_tum,
    _sim3_align,
    _step_rows,
    _step_summary,
    _write_markdown_table,
    _write_rgb_exp,
)


COMMON_CALIBRATION_FILE = "dataset_blt_calibration_common_camera_info_median.yaml"
DEFAULT_DISCOVERY_SUMMARY = (
    VSLAMLAB_EVALUATION
    / "exp_blt_rgbd_discovery"
    / "blt_rgbd_discovery_summary.json"
)
DROIDSLAM_RGBD_EXE = REPO_ROOT / ".pixi/envs/droidslam/bin/vslamlab_droidslam_rgbd"
DROIDSLAM_BLT_RGBD_LOWMEM_SETTINGS = (
    REPO_ROOT
    / "configs/baselines/droidslam_blt_rgbd_lowmem.yaml"
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _baseline_availability(baseline_name: str) -> tuple[bool, str, Any]:
    baseline = get_baseline(baseline_name)
    installed, install_msg = baseline.is_installed()
    settings_ok = baseline.settings_yaml.is_file()
    weights_param = baseline.default_parameters.get("weights")
    weights = Path(str(weights_param)) if weights_param else None
    weights_ok = weights.is_file() if weights is not None else True
    pixi_cmd = str(PIXI_EXE) if PIXI_EXE.exists() else shutil.which("pixi")
    if not pixi_cmd:
        details = (
            f"is_installed={installed} ({install_msg}); "
            f"settings_exists={settings_ok} path={baseline.settings_yaml}; "
            f"weights_exists={weights_ok} path={weights}; "
            "pixi_executable_missing"
        )
        return False, details, baseline
    env_check = subprocess.run(
        [pixi_cmd, "run", "--frozen", "-e", baseline.baseline_name, "python", "-c", "import sys; print(sys.version.split()[0])"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ok = bool(installed and settings_ok and weights_ok and env_check.returncode == 0)
    details = (
        f"is_installed={installed} ({install_msg}); "
        f"settings_exists={settings_ok} path={baseline.settings_yaml}; "
        f"weights_exists={weights_ok} path={weights}; "
        f"pixi_env_returncode={env_check.returncode}; "
        f"pixi_env_tail={(env_check.stderr or env_check.stdout)[-500:]}"
    )
    return ok, details, baseline


def _patch_droidslam_torch_load_compat(baseline_name: str) -> str:
    if baseline_name != "droidslam":
        return "not_required"
    pixi_cmd = str(PIXI_EXE) if PIXI_EXE.exists() else shutil.which("pixi")
    if not pixi_cmd:
        return "pixi_missing"
    module_path_proc = subprocess.run(
        [
            pixi_cmd,
            "run",
            "--frozen",
            "-e",
            "droidslam",
            "python",
            "-c",
            "import droid_slam.droid as d; print(d.__file__)",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if module_path_proc.returncode != 0:
        tail = (module_path_proc.stderr or module_path_proc.stdout)[-500:]
        return f"module_lookup_failed returncode={module_path_proc.returncode}; tail={tail}"
    module_path = Path(module_path_proc.stdout.strip().splitlines()[-1])
    text = module_path.read_text(encoding="utf-8")
    notes = []
    old_load = "torch.load(weights).items()"
    new_load = "torch.load(weights, weights_only=False).items()"
    if new_load in text:
        notes.append("torch_load_already_patched")
    elif old_load in text:
        text = text.replace(old_load, new_load)
        notes.append("torch_load_patched")
    else:
        notes.append("torch_load_unexpected_source")

    old_pixels = "target_pixels: int = 384*512"
    new_pixels = 'target_pixels: int = int(os.environ.get("DROIDSLAM_TARGET_PIXELS", 384*512))'
    if new_pixels in text:
        notes.append("target_pixels_env_already_patched")
    elif old_pixels in text:
        text = text.replace(old_pixels, new_pixels)
        notes.append("target_pixels_env_patched")
    else:
        notes.append("target_pixels_unexpected_source")

    module_path.write_text(text, encoding="utf-8")
    return f"path={module_path}; " + "; ".join(notes)


def _load_discovery(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _discovery_config_for_sequence(
    discovery: dict[str, Any],
    sequence: str,
    fallback_depth_topic: str,
    fallback_depth_camera_info_topic: str,
) -> tuple[str, str, str]:
    if fallback_depth_topic:
        return fallback_depth_topic, fallback_depth_camera_info_topic, ""
    for item in discovery.get("sequences", []):
        if item.get("sequence") != sequence:
            continue
        if not item.get("usable_rgbd"):
            return "", "", str(item.get("failure_cause", "discovery_not_usable"))
        camera_info_topics = item.get("depth_camera_info_topics") or []
        if isinstance(camera_info_topics, str):
            camera_info_topic = camera_info_topics.split(",")[0] if camera_info_topics else ""
        else:
            camera_info_topic = camera_info_topics[0] if camera_info_topics else ""
        return str(item.get("best_depth_topic", "")), str(camera_info_topic), ""
    return "", "", "sequence_missing_from_discovery"


def _method_env(
    base_env: dict[str, str],
    max_seconds: float,
    depth_topic: str,
    depth_camera_info_topic: str,
    depth_factor: float,
) -> dict[str, str]:
    env = dict(base_env)
    env["BLT_MODES"] = "mono,rgbd"
    env["BLT_CALIBRATION_FILE"] = COMMON_CALIBRATION_FILE
    env["BLT_MAX_SECONDS"] = str(max_seconds)
    env["BLT_MAX_FRAMES"] = "0"
    env["BLT_DEPTH_TOPIC"] = depth_topic
    if depth_camera_info_topic:
        env["BLT_DEPTH_CAMERA_INFO_TOPIC"] = depth_camera_info_topic
    env["BLT_DEPTH_FACTOR"] = str(depth_factor)
    return env


def _direct_rgbd_command(command: str) -> str:
    tokens = shlex.split(command)
    if "execute-rgbd" not in tokens:
        return command
    execute_index = tokens.index("execute-rgbd")
    if not DROIDSLAM_RGBD_EXE.is_file():
        return command
    return shlex.join([str(DROIDSLAM_RGBD_EXE), *tokens[execute_index + 1 :]])


def _run_baseline(exp_folder: Path, command: str, env: dict[str, str], expected_frames: int) -> tuple[bool, str, float]:
    start = time.time()
    log_path = exp_folder / "system_output_00000.txt"
    traj = exp_folder / f"00000_{TRAJECTORY_FILE_NAME}.csv"
    complete_since: float | None = None
    shutdown_hang_recovered = False
    next_heartbeat = start
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=log_file,
            text=True,
            env=env,
            start_new_session=True,
        )
        while proc.poll() is None:
            now = time.time()
            if expected_frames > 0 and traj.exists():
                try:
                    tracked_frames = _count_csv_data_rows(traj)
                except Exception:
                    tracked_frames = 0
                if tracked_frames >= expected_frames:
                    if complete_since is None:
                        complete_since = now
                        print(
                            f"[rgbd-baseline] complete trajectory detected rows={tracked_frames}; "
                            "waiting for clean shutdown",
                            flush=True,
                        )
                    elif now - complete_since >= 15.0:
                        print(
                            "[rgbd-baseline] baseline shutdown hang detected after complete trajectory; "
                            "terminating subprocess",
                            flush=True,
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                        shutdown_hang_recovered = True
                        break
                else:
                    complete_since = None
            if now >= next_heartbeat:
                elapsed = now - start
                print(f"[rgbd-baseline] still running elapsed_s={elapsed:.1f} log={log_path}", flush=True)
                next_heartbeat = now + 30.0
            time.sleep(5)
    duration = time.time() - start
    complete_trajectory = False
    if traj.is_file() and traj.stat().st_size > 0:
        try:
            complete_trajectory = expected_frames > 0 and _count_csv_data_rows(traj) >= expected_frames
        except Exception:
            complete_trajectory = False
    success = proc.returncode == 0 and traj.is_file() and traj.stat().st_size > 0
    if shutdown_hang_recovered and traj.is_file() and _count_csv_data_rows(traj) >= expected_frames:
        success = True
    if proc.returncode in {-15, 143} and complete_trajectory:
        success = True
    if success:
        comments = ""
        if shutdown_hang_recovered:
            comments = "baseline_shutdown_hang_recovered"
        elif proc.returncode in {-15, 143}:
            comments = "baseline_self_terminated_after_complete_trajectory"
        return True, comments, duration
    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:] if log_path.exists() else ""
    return (
        False,
        f"baseline_runtime_failure returncode={proc.returncode}; "
        f"trajectory_exists={traj.exists()}; log_tail={log_tail}",
        duration,
    )


def _run_logged_command(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    label: str,
    heartbeat_seconds: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=log_file,
            text=True,
            env=env,
        )
        while proc.poll() is None:
            elapsed = time.time() - start
            print(f"[{label}] still running elapsed_s={elapsed:.1f} log={log_path}", flush=True)
            time.sleep(heartbeat_seconds)
    return subprocess.CompletedProcess(command, proc.returncode, "", "")


def _run_evaluation_with_heartbeat(exp_folder: Path, dataset: Any, env: dict[str, str]) -> tuple[bool, str, dict[str, Any]]:
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
    log_path = eval_dir / "evo_ape.log"
    proc = _run_logged_command(
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
        log_path,
        env,
        "rgbd-evaluation",
    )
    if proc.returncode != 0 or not zip_file.exists():
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:] if log_path.exists() else ""
        return False, f"evo_ape failed returncode={proc.returncode}: {log_tail}", {}

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


def _run_one(
    baseline: Any,
    sequence: str,
    exp_root: Path,
    benchmark_root: Path,
    max_rgb: int,
    max_seconds: float,
    depth_topic: str,
    depth_camera_info_topic: str,
    depth_factor: float,
    base_env: dict[str, str],
) -> dict[str, Any]:
    env = _method_env(base_env, max_seconds, depth_topic, depth_camera_info_topic, depth_factor)
    os.environ.update(env)
    dataset = get_dataset("blt", benchmark_root)
    seq_path = dataset.dataset_path / sequence
    method = f"{baseline.baseline_name}_rgbd"
    exp_base = exp_root / "runs" / method
    exp_folder = exp_base / dataset.dataset_folder / sequence
    if seq_path.exists():
        shutil.rmtree(seq_path)
    exp_folder.mkdir(parents=True, exist_ok=True)
    success = False
    evaluation = "failed"
    comments = ""
    duration = 0.0
    acc_info: dict[str, Any] = {}
    extracted_removed = False
    selected_frames = 0
    try:
        print(f"[{method}:{sequence}] extracting RGB-D max_seconds={max_seconds}", flush=True)
        dataset.download_sequence(sequence)
        selected_frames = _write_rgb_exp(seq_path, exp_folder, dataset, max_rgb)
        shutil.copy2(seq_path / "groundtruth.csv", exp_folder / "groundtruth.csv")
        exp = SimpleNamespace(
            folder=exp_base,
            parameters={
                "verbose": 0,
                "mode": "rgbd",
                "max_rgb": max_rgb,
                "upsample": 0,
            },
            num_runs=1,
        )
        pixi_command = baseline.build_execute_command(0, exp, dataset, sequence)
        command = _direct_rgbd_command(pixi_command)
        (exp_folder / "execute_command.txt").write_text(command + "\n", encoding="utf-8")
        if command != pixi_command:
            (exp_folder / "execute_command_pixi.txt").write_text(pixi_command + "\n", encoding="utf-8")
        print(f"[{method}:{sequence}] selected_frames={selected_frames}; running {baseline.baseline_name} rgbd", flush=True)
        success, comments, duration = _run_baseline(exp_folder, command, env, selected_frames)
        if success:
            print(f"[{method}:{sequence}] baseline complete; running ATE evaluation", flush=True)
            eval_ok, evaluation_msg, acc_info = _run_evaluation_with_heartbeat(exp_folder, dataset, env)
            evaluation = evaluation_msg if eval_ok else "failed"
            success = bool(eval_ok)
            if not eval_ok:
                comments = evaluation_msg
    except Exception as exc:
        comments = f"{type(exc).__name__}: {exc}"
        print(f"[{method}:{sequence}] ERROR {comments}", flush=True)
    finally:
        if seq_path.exists():
            shutil.rmtree(seq_path)
        extracted_removed = not seq_path.exists()

    row = {
        "method": method,
        "baseline": baseline.baseline_name,
        "mode": "rgbd",
        "calibration_file": COMMON_CALIBRATION_FILE,
        "settings_yaml": str(baseline.settings_yaml),
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
        "num_frames": acc_info.get("num_frames", selected_frames),
        "num_tracked_frames": acc_info.get("num_tracked_frames", 0),
        "num_evaluated_frames": acc_info.get("num_evaluated_frames", 0),
        "duration_time_s": round(duration, 3),
        "depth_topic": depth_topic,
        "depth_camera_info_topic": depth_camera_info_topic,
        "depth_factor": depth_factor,
        "comments": comments,
        "extracted_removed": extracted_removed,
    }
    row.update(_step_summary(exp_folder) if success else {})
    (exp_folder / "run_status.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    return row


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
        ax.plot(traj_xy[:, 0] - shift[0], traj_xy[:, 1] - shift[1], color="tab:blue", linewidth=1.0, label=method)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BLT RGB-D baseline matrix after topic discovery.")
    parser.add_argument("--output-name", default="exp_blt_droidslam_rgbd_matrix")
    parser.add_argument("--baseline", default="droidslam")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--max-rgb", type=int, default=600)
    parser.add_argument("--depth-topic", default="")
    parser.add_argument("--depth-camera-info-topic", default="")
    parser.add_argument("--depth-factor", type=float, default=1000.0)
    parser.add_argument("--discovery-summary", default=str(DEFAULT_DISCOVERY_SUMMARY))
    parser.add_argument(
        "--benchmark-root",
        default=os.environ.get("BLT_RGBD_BENCHMARK_ROOT", "/tmp/vslamlab_blt_rgbd_benchmark"),
        help="Temporary extraction root for RGB-D benchmark files; defaults to /tmp to avoid filling /home.",
    )
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    sequences = _parse_csv(args.sequences)
    unknown_sequences = sorted(set(sequences).difference(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown BLT sequences: {unknown_sequences}")

    base_env = os.environ.copy()
    base_env.setdefault("BLT_KTIMA_ROOT", "/media/pulver/PulverHDD/BACCHUS/ktima")
    base_env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "vslamlab_mpl"))
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    base_env.setdefault("DROIDSLAM_TARGET_PIXELS", str(384 * 384))
    Path(base_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    availability_ok, availability_details, baseline = _baseline_availability(args.baseline)
    if args.baseline == "droidslam" and DROIDSLAM_BLT_RGBD_LOWMEM_SETTINGS.is_file():
        baseline.settings_yaml = DROIDSLAM_BLT_RGBD_LOWMEM_SETTINGS
    compat_details = _patch_droidslam_torch_load_compat(args.baseline) if availability_ok else "not_attempted"
    exp_root = VSLAMLAB_EVALUATION / args.output_name
    if exp_root.exists() and args.clean_output:
        shutil.rmtree(exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = Path(args.benchmark_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "baseline_availability.txt").write_text(availability_details + "\n", encoding="utf-8")
    (exp_root / "baseline_compatibility.txt").write_text(compat_details + "\n", encoding="utf-8")

    discovery = _load_discovery(Path(args.discovery_summary))
    rows: list[dict[str, Any]] = []
    if not availability_ok:
        for sequence in sequences:
            rows.append(
                {
                    "method": f"{args.baseline}_rgbd",
                    "baseline": args.baseline,
                    "mode": "rgbd",
                    "month": SEQUENCES.get(sequence, sequence),
                    "sequence": sequence,
                    "success": False,
                    "evaluation": "blocked",
                    "comments": f"baseline_unavailable: {availability_details}",
                    "extracted_removed": True,
                }
            )
    else:
        for sequence in sequences:
            depth_topic, camera_info_topic, discovery_failure = _discovery_config_for_sequence(
                discovery,
                sequence,
                args.depth_topic,
                args.depth_camera_info_topic,
            )
            if not depth_topic:
                rows.append(
                    {
                        "method": f"{args.baseline}_rgbd",
                        "baseline": args.baseline,
                        "mode": "rgbd",
                        "month": SEQUENCES.get(sequence, sequence),
                        "sequence": sequence,
                        "success": False,
                        "evaluation": "blocked",
                        "comments": f"depth_topic_unavailable: {discovery_failure}",
                        "extracted_removed": True,
                    }
                )
                continue
            rows.append(
                _run_one(
                    baseline=baseline,
                    sequence=sequence,
                    exp_root=exp_root,
                    benchmark_root=benchmark_root,
                    max_rgb=args.max_rgb,
                    max_seconds=args.max_seconds,
                    depth_topic=depth_topic,
                    depth_camera_info_topic=camera_info_topic,
                    depth_factor=args.depth_factor,
                    base_env=base_env,
                )
            )

    results = pd.DataFrame(rows)
    results_csv = exp_root / "blt_rgbd_baseline_results.csv"
    results_md = exp_root / "blt_rgbd_baseline_results.md"
    results.to_csv(results_csv, index=False)
    display_cols = [
        "method",
        "month",
        "sequence",
        "success",
        "evaluation",
        "rmse",
        "num_frames",
        "num_tracked_frames",
        "num_evaluated_frames",
        "depth_topic",
        "comments",
    ]
    existing_cols = [column for column in display_cols if column in results.columns]
    _write_markdown_table(results[existing_cols], results_md)
    for method in sorted(set(results["method"])):
        _plot_trajectories(exp_root, "BLT_dataset", method, rows, fig_dir)
    step_diag = pd.DataFrame(_step_rows(exp_root, "BLT_dataset"))
    step_diag.to_csv(exp_root / "step_diagnostics.csv", index=False)

    print(f"[done] results_csv={results_csv}", flush=True)
    print(f"[done] results_md={results_md}", flush=True)
    print(f"[done] figures={fig_dir}", flush=True)
    print(results.to_string(index=False), flush=True)
    return 0 if bool(results["success"].any()) or not availability_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
