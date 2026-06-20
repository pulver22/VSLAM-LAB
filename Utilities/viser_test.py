"""
Viser: optional colored 3D point cloud loaded from a PLY + camera frustums loaded from a CSV,
with quaternion convention fixes (invert + OpenCV camera-axis conversion).

CSV columns expected:
ts (ns), tx (m), ty (m), tz (m), qx, qy, qz, qw

Install:
  pip install viser numpy pandas open3d

Run (poses only):
  python viser_from_csv.py --csv poses.csv

Run (poses + point cloud):
  python viser_from_csv.py --csv poses.csv --ply cloud.ply

Open:
  http://localhost:8080
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import viser

import open3d as o3d
from PIL import Image

from path_constants import VSLAMLAB_BENCHMARK
SHOW_TRAJECTORY = True
TRAJ_LINE_WIDTH = 3.0

# ---------------------------
# Quaternion helpers (w,x,y,z)
# ---------------------------
def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


# ---------------------------
# CSV loading
# ---------------------------
def read_pose_csv(csv_path: Path) -> pd.DataFrame:
    text = csv_path.read_text(errors="ignore")
    first_line = text.splitlines()[0] if text else ""

    if "," in first_line:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, sep=r"\s+", engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        c2 = c.replace("(m)", "").replace("(ns)", "").strip()
        c2 = " ".join(c2.split())
        rename[c] = c2
    df = df.rename(columns=rename)

    required = ["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(np.int64)

    return df

def read_rgb_exp_csv(rgb_csv_path: Path) -> pd.DataFrame:
    """
    Reads rgb_exp.csv that may be comma-separated, tab-separated, or whitespace-separated.
    Expected columns include:
      ts_rgb_0 (ns), path_rgb_0
    """
    text = rgb_csv_path.read_text(errors="ignore")
    first_line = text.splitlines()[0] if text else ""

    if "," in first_line:
        df = pd.read_csv(rgb_csv_path)
    else:
        df = pd.read_csv(rgb_csv_path, sep=r"\s+", engine="python")

    # Normalize column names (strip, collapse spaces)
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]

    # Handle headers like "ts_rgb_0 (ns)"
    rename = {}
    for c in df.columns:
        c2 = c.replace("(ns)", "").strip()
        c2 = " ".join(c2.split())
        rename[c] = c2
    df = df.rename(columns=rename)

    required = ["ts_rgb_0", "path_rgb_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in rgb csv: {missing}\nFound: {list(df.columns)}")

    # Force timestamps to int64 (handles scientific notation)
    df["ts_rgb_0"] = pd.to_numeric(df["ts_rgb_0"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts_rgb_0"]).copy()
    df["ts_rgb_0"] = df["ts_rgb_0"].astype(np.int64)

    df = df.sort_values("ts_rgb_0").reset_index(drop=True)
    return df


def find_nearest_rgb_path(ts_ns: int, rgb_df: pd.DataFrame) -> tuple[str, int]:
    """
    Returns (path_rgb_0, dt_ns) for the nearest timestamp in rgb_df to ts_ns.
    """
    ts_arr = rgb_df["ts_rgb_0"].to_numpy(np.int64)
    idx = int(np.searchsorted(ts_arr, ts_ns, side="left"))

    if idx <= 0:
        best = 0
    elif idx >= len(ts_arr):
        best = len(ts_arr) - 1
    else:
        # choose closer of idx-1 and idx
        if abs(ts_arr[idx] - ts_ns) < abs(ts_arr[idx - 1] - ts_ns):
            best = idx
        else:
            best = idx - 1

    dt = int(abs(ts_arr[best] - ts_ns))
    path = str(rgb_df.loc[best, "path_rgb_0"])
    return path, dt


def load_rgb_image(image_path: Path) -> np.ndarray:
    """
    Loads an image as uint8 HxWx3 RGB.
    """
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# ---------------------------
# PLY loading (XYZ + RGB)
# ---------------------------
def load_ply_point_cloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {ply_path}")

    pts = np.asarray(pcd.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Unexpected points shape {pts.shape} in {ply_path}")
    points = pts.astype(np.float32)

    cols = np.asarray(pcd.colors)
    if cols.size == 0:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    else:
        colors = np.clip(cols * 255.0, 0, 255).astype(np.uint8)

    return points, colors


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", type=str, required=True, help="Path to exp folder")
    parser.add_argument("--port", type=int, default=8080)
    
    args = parser.parse_args()
    exp_folder = Path(args.exp_folder)
    csv_path = exp_folder / "00000_KeyFrameTrajectory.csv"
    rgb_csv_path = exp_folder / "rgb_exp.csv"
    ply_path = exp_folder / "00000_PointCloud.ply"

    image_root = Path(str(exp_folder).replace("VSLAM-LAB-Evaluation", "VSLAM-LAB-Benchmark"))
    image_root = VSLAMLAB_BENCHMARK / exp_folder.parts[-2] / exp_folder.parts[-1]

    df = read_pose_csv(csv_path)
    rgb_df = read_rgb_exp_csv(rgb_csv_path)
   
    #rng = np.random.default_rng(0)
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # World axes at origin
    #server.scene.add_frame(name="/world", axes_length=0.5, axes_radius=0.02)

    # ---- Point cloud publish ----
    if ply_path.exists():
        points, colors = load_ply_point_cloud(ply_path)
        server.scene.add_point_cloud(
            name="/point_cloud",
            points=points,
            colors=colors,
            point_size=0.01,
        )

    # ---- Frustums from CSV ----
    frustum_handles = []
    frustum_sizes = []
    trajectory_positions = []
    for i, row in enumerate(df.itertuples(index=False)):
        tx = float(getattr(row, "tx"))
        ty = float(getattr(row, "ty"))
        tz = float(getattr(row, "tz"))

        qx = float(getattr(row, "qx"))
        qy = float(getattr(row, "qy"))
        qz = float(getattr(row, "qz"))
        qw = float(getattr(row, "qw"))

        q_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
        #q_wxyz = quat_normalize_wxyz(q_wxyz)

        wxyz = tuple(q_wxyz.tolist())
        pos = np.array([tx, ty, tz], dtype=np.float32)
        trajectory_positions.append(pos.astype(np.float32))

        # Load associated RGB image (if any)
        if rgb_df is not None:
            ts_pose = int(getattr(row, "ts"))
            rel_path, dt_ns = find_nearest_rgb_path(ts_pose, rgb_df)
            print(f"Frame {i:05d}: ts={ts_pose} ns, nearest rgb='{rel_path}' dt={dt_ns} ns")
            if dt_ns <= 1000:
                img_path_exp = exp_folder / Path(rel_path).name
                img_path_raw = (image_root / rel_path).resolve()
                print(f"  Trying exp image path: {img_path_exp}")
                if img_path_exp.exists():
                    img = load_rgb_image(img_path_exp)
                else:
                    if img_path_raw.exists():
                        img = load_rgb_image(img_path_raw)
                    else:
                        img = None

        if img is None:
            img = np.full((480, 640, 3), 128, dtype=np.uint8)
        h, w = img.shape[:2]

        fy = 1.1 * h
        fov = float(2 * np.arctan2(h / 2, fy))

        server.scene.add_frame(
            name=f"/frustums/{i:05d}/axes",
            wxyz=wxyz,
            position=pos,
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )

        frustum_cam = server.scene.add_camera_frustum(
            name=f"/frustums/{i:05d}/frustum",
            fov=fov,
            aspect=w / h,
            scale=0.05,
            image=img,
            line_width=1.0,
            wxyz=wxyz,
            position=pos,
        )

        frustum_handles.append(frustum_cam)
        frustum_sizes.append((h, w))

    # ---- Trajectory: connect frustums sequentially with a polyline ----
    if len(trajectory_positions) >= 2:
        traj = np.stack(trajectory_positions, axis=0)          # (N,3)
        segments = np.stack([traj[:-1], traj[1:]], axis=1)     # (N-1,2,3)

        server.scene.add_line_segments(
            name="/trajectory",
            points=segments,
            colors=(0.6471,0.6157,0.8745),     # single RGB color for all segments
            line_width=2.0,
        )

    print(f"Serving at http://localhost:{args.port}")
    print(f"Loaded {len(frustum_handles)} frustums from {csv_path}")

    while True:
        time.sleep(1.0)



if __name__ == "__main__":
    main()

