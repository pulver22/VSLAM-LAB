#!/usr/bin/env python3
"""Plot March/April BLT follow-up trajectory and step diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SEQUENCES = {
    "ktima_2022_03": "March",
    "ktima_2022_04": "April",
}


def read_tum(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", comment="#")
    return frame.rename(columns={"tx": "x", "ty": "y", "tz": "z"})


def load_followup(base_dir: Path, sequence: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = base_dir / "runs" / "common_camera_info_median" / "BLT_dataset" / sequence
    eval_dir = run_dir / "vslamlab_evaluation"
    estimated = read_tum(eval_dir / "00000_KeyFrameTrajectory.tum")
    groundtruth = read_tum(eval_dir / "00000_gt.tum")
    return estimated, groundtruth


def window(frame: pd.DataFrame, center: int, radius: int) -> pd.DataFrame:
    start = max(0, center - radius)
    stop = min(len(frame), center + radius + 1)
    return frame.iloc[start:stop].copy()


def plot_zoom(base_dir: Path, output_path: Path, center_index: int, radius: int) -> None:
    steps = pd.read_csv(base_dir / "step_diagnostics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (sequence, month) in enumerate(SEQUENCES.items()):
        estimated, groundtruth = load_followup(base_dir, sequence)
        est_win = window(estimated, center_index, radius)
        gt_win = window(groundtruth, center_index, radius)

        ax_xy = axes[0, col]
        ax_xy.plot(gt_win["x"], gt_win["y"], color="#444444", linewidth=2, label="GT")
        ax_xy.plot(est_win["x"], est_win["y"], color="#0072B2", linewidth=1.5, label="Estimated")
        ax_xy.scatter(est_win.iloc[[0, -1]]["x"], est_win.iloc[[0, -1]]["y"], color="#0072B2", s=22)
        ax_xy.set_title(f"{month} XY trajectory, indices {center_index-radius}-{center_index+radius}")
        ax_xy.set_xlabel("x (m)")
        ax_xy.set_ylabel("y (m)")
        ax_xy.axis("equal")
        ax_xy.grid(True, alpha=0.25)
        ax_xy.legend(loc="best")

        ax_step = axes[1, col]
        seq_steps = steps[steps["sequence"] == sequence]
        for kind, color in [("estimated", "#D55E00"), ("groundtruth", "#009E73")]:
            kind_steps = seq_steps[seq_steps["kind"] == kind]
            step_win = kind_steps[
                (kind_steps["index"] >= center_index - radius)
                & (kind_steps["index"] <= center_index + radius)
            ]
            ax_step.plot(step_win["index"], step_win["step_m"], color=color, label=kind)
        ax_step.axvline(center_index, color="#000000", linestyle="--", linewidth=1)
        ax_step.set_title(f"{month} inter-frame step size")
        ax_step.set_xlabel("step index")
        ax_step.set_ylabel("step (m)")
        ax_step.grid(True, alpha=0.25)
        ax_step.legend(loc="best")

    fig.suptitle("BLT March vs April follow-up diagnostics around March jump index 1253")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center-index", type=int, default=1253)
    parser.add_argument("--radius", type=int, default=60)
    args = parser.parse_args()
    plot_zoom(args.base_dir, args.output, args.center_index, args.radius)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
