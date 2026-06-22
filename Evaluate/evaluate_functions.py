import csv
import subprocess
import os, shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from Evaluate.evo_functions import evo_metric, evo_get_accuracy
from path_constants import VSLAM_LAB_EVALUATION_FOLDER, TRAJECTORY_FILE_NAME, GROUNTRUTH_FILE
from utilities import print_msg, ws, format_msg, read_csv

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def _count_csv_data_rows(csv_path: str | Path) -> int:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

def _count_text_data_rows(text_path: str | Path, has_header: bool = False) -> int:
    with open(text_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    if has_header and count > 0:
        count -= 1
    return count

def _rgb_exp_max_time_difference(rgb_exp_csv: str | Path, fallback_rgb_hz: float) -> float:
    try:
        rgb_df = pd.read_csv(rgb_exp_csv)
        timestamps = rgb_df.iloc[:, 0].astype("int64").sort_values().to_numpy()
        if len(timestamps) >= 2:
            median_dt_ns = float(pd.Series(timestamps).diff().dropna().median())
            if median_dt_ns > 0:
                return max(median_dt_ns * 1.5, 2e7)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, IndexError):
        pass
    return 1.5e9 / float(fallback_rgb_hz)

def evaluate_sequence(exp, dataset, sequence_name, overwrite=False):
    sequence_name = str(sequence_name)

    # Enable evo to save trajectories in zip format
    # TODO : Remove this
    command =  "pixi run -e vslamlab evo_config set save_traj_in_zip true"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    METRIC = 'ate'
    
    # Define input paths
    trajectories_path = exp.folder / dataset.dataset_folder / sequence_name
    groundtruth_csv = exp.folder / dataset.dataset_folder / sequence_name /  GROUNTRUTH_FILE

    # Define output paths
    evaluation_folder = exp.folder / dataset.dataset_folder / sequence_name / VSLAM_LAB_EVALUATION_FOLDER
    accuracy_csv = evaluation_folder / f'{METRIC}.csv'

    # Load experiments log
    exp_log = read_csv(exp.log_csv)
    if overwrite:
        if evaluation_folder.exists():
            shutil.rmtree(evaluation_folder)        
        exp_log.loc[exp_log["sequence_name"] == sequence_name, "EVALUATION"] = "none"

    evaluation_folder.mkdir(parents=True, exist_ok=True)

    # Find runs to evaluate
    runs_to_evaluate = []
    for _, row in exp_log.iterrows():
        if row["SUCCESS"] and (row["EVALUATION"] == 'none') and (row["sequence_name"] == sequence_name):
            exp_it = str(row["exp_it"]).zfill(5) 
            runs_to_evaluate.append(exp_it)

    print_msg(
        SCRIPT_LABEL,
        f"Evaluating '{str(evaluation_folder).replace(sequence_name, f'{dataset.dataset_color}{sequence_name}\033[0m')}'"
    )
    if len(runs_to_evaluate) == 0:
        exp_log.to_csv(exp.log_csv, index=False)
        return
    
    # Evaluate runs
    zip_files = []
    for exp_it in tqdm(runs_to_evaluate):
        trajectory_file = trajectories_path / f"{exp_it}_{TRAJECTORY_FILE_NAME}.csv"
        rgb_exp_csv = trajectories_path / "rgb_exp.csv"
        success = evo_metric(METRIC, groundtruth_csv, trajectory_file, evaluation_folder, _rgb_exp_max_time_difference(rgb_exp_csv, dataset.rgb_hz))
        if success[0]:
            zip_files.append(evaluation_folder / f"{exp_it}_{TRAJECTORY_FILE_NAME}.zip")
        else:
            exp_log.loc[(exp_log["exp_it"] == int(exp_it)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
            tqdm.write(format_msg(ws(8), f"{success[1]}", "error"))
    if len(zip_files) == 0:
        exp_log.to_csv(exp.log_csv, index=False)
        return   
    
    # Retrieve accuracies
    evo_get_accuracy(zip_files, accuracy_csv)

    # Final Checks
    if not os.path.exists(accuracy_csv):
        exp_log.to_csv(exp.log_csv, index=False)
        return
    
    accuracy = pd.read_csv(accuracy_csv)
    for evaluated_run in runs_to_evaluate:
        if exp_log.loc[(exp_log["exp_it"] == int(exp_it)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"].any() == 'failed':
            continue
        trajectory_name_txt = f"{evaluated_run}_{TRAJECTORY_FILE_NAME}.txt"
        exists = (accuracy["traj_name"] == trajectory_name_txt).any()
        if exists:
            run_mask = (exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name)
            exp_log.loc[run_mask, "EVALUATION"] = METRIC

            # Find number of frames in the sequence
            rgb_exp_csv = trajectories_path / "rgb_exp.csv"
            num_frames = _count_csv_data_rows(rgb_exp_csv)
            accuracy.loc[accuracy["traj_name"] == trajectory_name_txt,"num_frames"] = num_frames
            exp_log.loc[run_mask, "num_frames"] = num_frames

            # Find number of tracked frames
            trajectory_file_txt = evaluation_folder / trajectory_name_txt
            if not trajectory_file_txt.exists():
                exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
                continue
            num_tracked_frames = _count_text_data_rows(trajectory_file_txt)
            accuracy.loc[accuracy["traj_name"] == trajectory_name_txt,"num_tracked_frames"] = num_tracked_frames    
            exp_log.loc[run_mask, "num_tracked_frames"] = num_tracked_frames

            # Find number of evaluated frames
            trajectory_file_tum = trajectories_path / VSLAM_LAB_EVALUATION_FOLDER / trajectory_name_txt.replace(".txt", ".tum")
            if not trajectory_file_tum.exists():
                exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
                continue
            num_evaluated_frames = _count_text_data_rows(trajectory_file_tum, has_header=True)
            accuracy.loc[accuracy["traj_name"] == trajectory_name_txt,"num_evaluated_frames"] = num_evaluated_frames   
            exp_log.loc[run_mask, "num_evaluated_frames"] = num_evaluated_frames 
        else:
            exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'

    exp_log.to_csv(exp.log_csv, index=False)
    accuracy.to_csv(accuracy_csv, index=False)
