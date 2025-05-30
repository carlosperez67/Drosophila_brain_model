#!/usr/bin/env python3
"""
Run neuron activation experiments.

This script can launch:

1. Group 1 neuron activations.
2. Group 1 + Group 2 neuron activations.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from brian2 import Hz
from brian2 import prefs

from model import run_exp
from model import default_params

# ----------------------
# Global default config
#
# Change here to run script with different parameters
# ----------------------
DEFAULT_SUBSET         = "group-1"
DEFAULT_RES_DIR        = Path("/Volumes/T7/GordonLab/EGG_LAYING/test")
DEFAULT_GROUP_1_HZ     = list(range(20, 201, 100))
DEFAULT_GROUP_2_HZ     = list(range(20, 201, 100))
DEFAULT_GROUP_1_CSV    = Path("./Data/sweet.csv")
DEFAULT_GROUP_2_CSV    = None                     # Set to None if not using
DEFAULT_GROUP_1_NAME   = "group-1"
DEFAULT_GROUP_2_NAME   = "group-2"


CONFIG = {
    "path_comp": "./Completeness_783.csv",
    "path_con" : "./Connectivity_783.parquet",
    "path_res" : DEFAULT_RES_DIR,
    "n_proc"   : 10,
}
GROUP_1_F = list(range(20, 201, 10))
GROUP_2_F   = list(range(20, 201, 10))
prefs.codegen.target = "numpy"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GRN frequency sweeps")
    p.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        choices=["group-1", "group-2", "both"],
        help=f"Which part of the sweep to run (default: {DEFAULT_SUBSET})",
    )
    p.add_argument(
        "--res-dir",
        type=Path,
        default=DEFAULT_RES_DIR,
        help=f"Root directory to save results (default: {DEFAULT_RES_DIR})",
    )
    p.add_argument(
        "--group-1-hz",
        type=float,
        nargs="+",
        default=DEFAULT_GROUP_1_HZ,
        help=f"List of sugar frequencies in Hz (default: {DEFAULT_GROUP_1_HZ})",
    )
    p.add_argument(
        "--group-2-hz",
        type=float,
        nargs="+",
        default=DEFAULT_GROUP_2_HZ,
        help=f"List of group-2 frequencies in Hz (default: {DEFAULT_GROUP_2_HZ})",
    )
    p.add_argument(
        "--group-1-csv",
        type=Path,
        default=DEFAULT_GROUP_1_CSV,
        help=f"Path to CSV containing group-1 root_id (default: {DEFAULT_GROUP_1_CSV})",
    )
    p.add_argument(
        "--group-2-csv",
        type=Path,
        default=DEFAULT_GROUP_2_CSV,
        help=f"Path to CSV containing group-2 root_id (default: {DEFAULT_GROUP_2_CSV})",
    )
    p.add_argument(
        "--group-1-name",
        type=str,
        default=DEFAULT_GROUP_1_NAME,
        help=f"Name of group-1 neurons (default: {DEFAULT_GROUP_1_NAME})",
    )
    p.add_argument(
        "--group-2-name",
        type=str,
        default=DEFAULT_GROUP_2_NAME,
        help=f"Name of group-2 neurons (default: {DEFAULT_GROUP_2_NAME})",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    res_root: Path = args.res_dir
    group_1_ids: Path = args.group_1_csv
    group_2_ids: Path = args.group_2_csv

    group_1_name = args.group_1_name
    group_2_name = args.group_2_name

    # prepare output subdirectories
    dir_controls    = res_root / "controls"
    dir_experiments = res_root / "experiments"

    dir_controls.mkdir(parents=True, exist_ok=True)
    dir_experiments.mkdir(parents=True, exist_ok=True)

    # load neuron populations
    group_1_df  = pd.read_csv(group_1_ids)
    group_1_neurons = [int(i) for i in group_1_df["root_id"]]

    exp_names: list[str]     = []
    file_paths: list[Path]   = []

    if not args.group_1_csv:
        raise ValueError("group_1_csv must be specified")

    for group_1_rate in args.group_1_hz:
        default_params["r_poi"] = group_1_rate * Hz
        exp_name = f"{group_1_name}_{int(group_1_rate)}Hz"
        run_exp(
            exp_name=exp_name,
            neu_exc=group_1_neurons,
            params=default_params,
            **CONFIG,
            force_overwrite=False,
        )
        exp_names.append(exp_name)
        file_paths.append(res_root / f"{exp_name}.parquet")

    if args.group_2_csv:
        group_2_df = pd.read_csv(group_2_ids)
        group_2_neurons = [int(i) for i in group_2_df["root_id"]]
        for group_1_rate in args.group_1_hz:
            default_params["r_poi"] = group_1_rate * Hz

            for group_2_rate in args.group_1_hz:
                default_params["r_poi2"]  = group_2_rate * Hz

                exp_name = f"{group_1_name}_{int(group_1_rate)}Hz_AND_{group_2_name}_{int(group_2_rate)}Hz"
                run_exp(
                    exp_name=exp_name,
                    neu_exc=group_1_neurons,
                    neu_exc2=group_2_neurons,
                    params=default_params,
                    **CONFIG,
                    force_overwrite=False,
                )
                exp_names.append(exp_name)
                file_paths.append(res_root / f"{exp_name}.parquet")

    print(f"Finished running {len(exp_names)} experiments")


if __name__ == "__main__":
    main()
