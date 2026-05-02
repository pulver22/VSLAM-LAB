#!/usr/bin/env python3
"""
rerun_visualize.py
------------------
Visualizes a VSLAM-LAB sequence (RGB, depth, groundtruth) using Rerun.

Usage:
    python Utilities/rerun_visualize.py --sequence_path /path/to/sequence
    python Utilities/rerun_visualize.py --sequence_path /path/to/sequence --max_frames 200

Dataset format expected:
    sequence_path/
        rgb.csv           (ts_rgb_0 (ns), path_rgb_0 [, ts_depth_0 (ns), path_depth_0])
        groundtruth.csv   (ts (ns), tx (m), ty (m), tz (m), qx, qy, qz, qw) [optional]
        calibration.yaml
        rgb_0/            PNG images
        depth_0/          PNG images, 16-bit, divided by depth_factor to get meters [optional]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import yaml


def load_calibration(calib_path: Path) -> dict:
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    cameras = {}
    for cam in calib.get("cameras", []):
        cameras[cam["cam_name"]] = cam
    return cameras


def nearest_pose(gt_df: pd.DataFrame, ts_ns: int) -> np.ndarray:
    """Return (tx, ty, tz, qx, qy, qz, qw) at the nearest groundtruth timestamp."""
    idx = (gt_df.iloc[:, 0] - ts_ns).abs().argmin()
    return gt_df.iloc[idx, 1:8].to_numpy(float)


def build_intrinsic(cam: dict) -> tuple[np.ndarray, int, int]:
    fx, fy = cam["focal_length"]
    cx, cy = cam["principal_point"]
    w, h = cam["image_dimension"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, int(w), int(h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a VSLAM-LAB sequence with Rerun")
    parser.add_argument("--sequence_path", required=True, type=Path,
                        help="Path to the sequence directory")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap the number of frames to load (default: all)")
    rr.script_add_args(parser)
    args = parser.parse_args()

    seq_path = args.sequence_path.resolve()
    if not seq_path.exists():
        print(f"Error: sequence path does not exist: {seq_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load metadata ──────────────────────────────────────────────────────────
    cameras = load_calibration(seq_path / "calibration.yaml")
    rgb_df = pd.read_csv(seq_path / "rgb.csv")

    gt_df = None
    gt_path = seq_path / "groundtruth.csv"
    if gt_path.exists():
        _gt = pd.read_csv(gt_path)
        if not _gt.empty:
            gt_df = _gt
        else:
            print("Warning: groundtruth.csv is empty, skipping pose visualization")

    cam_name = next(iter(cameras))
    cam = cameras[cam_name]
    K, w, h = build_intrinsic(cam)
    depth_factor = float(cam.get("depth_factor", 1000.0))

    # rgb.csv columns: ts_rgb, path_rgb [, ts_depth, path_depth]
    has_depth = rgb_df.shape[1] >= 4

    # ── Blueprint ──────────────────────────────────────────────────────────────
    camera_origin = f"world/{cam_name}"
    rgb_view = rrb.Spatial2DView(
        name="RGB",
        origin=camera_origin,
        contents=[f"{camera_origin}/rgb"],
    )

    if has_depth:
        depth_view = rrb.Spatial2DView(
            name="Depth",
            origin=camera_origin,
            contents=[f"{camera_origin}/depth"],
        )
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(name="3D World"),
            rrb.Horizontal(rgb_view, depth_view),
            row_shares=[2, 1],
        )
    else:
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(name="3D World"),
            rgb_view,
            row_shares=[2, 1],
        )

    rr.script_setup(args, "vslamlab_sequence_viewer", default_blueprint=blueprint)

    # ── Static camera intrinsics ───────────────────────────────────────────────
    rr.log(
        camera_origin,
        rr.Pinhole(image_from_camera=K, resolution=[w, h]),
        static=True,
    )

    # ── Full trajectory + initial view orientation ─────────────────────────────
    if gt_df is not None:
        traj_pts = gt_df.iloc[:, 1:4].to_numpy(float).astype(np.float32)
        rr.log("world/trajectory", rr.Points3D(traj_pts, radii=0.003, colors=[0, 200, 255]), static=True)
        rr.log("world/trajectory_path", rr.LineStrips3D([traj_pts], colors=[[0, 200, 255, 160]]), static=True)

        # SVD: row 2 of Vt = normal to the dominant plane of motion → use as "up"
        if len(traj_pts) >= 3:
            centroid = traj_pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(traj_pts - centroid, full_matrices=False)
            normal = Vt[2]
            dominant = int(np.argmax(np.abs(normal)))
            view_coords = [
                rr.ViewCoordinates.RIGHT_HAND_X_UP,
                rr.ViewCoordinates.RIGHT_HAND_Y_UP,
                rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            ][dominant]
            rr.log("world", view_coords, static=True)

    # ── Per-frame logging ──────────────────────────────────────────────────────
    rows = rgb_df.head(args.max_frames) if args.max_frames else rgb_df

    for frame_idx, (_, row) in enumerate(rows.iterrows()):
        ts_ns = int(row.iloc[0])
        rgb_path = seq_path / row.iloc[1]

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("timestamp_ns", sequence=ts_ns)

        # Camera pose
        if gt_df is not None:
            tx, ty, tz, qx, qy, qz, qw = nearest_pose(gt_df, ts_ns)
            rr.log(
                camera_origin,
                rr.Transform3D(
                    translation=[tx, ty, tz],
                    rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
                ),
            )

        rr.log(f"{camera_origin}/rgb", rr.Image(rgb).compress(jpeg_quality=85))

        if has_depth:
            depth_path = seq_path / row.iloc[3]
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                rr.log(f"{camera_origin}/depth", rr.DepthImage(depth, meter=depth_factor))

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
