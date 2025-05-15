#!/usr/bin/env python3
"""
control_compute.py

Run control experiments and save MN9 firing-rate runs to parquet,
automatically continuing the run numbering based on what’s already there.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from brian2 import Hz

from model import run_exp, default_params as params
import utils as utl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run control experiments and auto‑number the runs."
    )
    parser.add_argument(
        "--hz",
        type=float,
        required=True,
        help="Activation frequency for sugar neurons (in Hz).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of new repetitions to perform.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        # default=Path("/project/def‑mdgordon‑ab/cperez67/RESULTS/control_trials"),
        default=Path("./sweet_results/GRN/controls_hist"),
        help="Directory for control‑trial parquet files (default: network share).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hz_int = int(args.hz)
    n_new = args.n_runs

    # 1) where parquet files live
    out_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) scan for existing run numbers
    pattern = re.compile(rf"sugar_only_{hz_int}Hz_run_(\d+)\.parquet$")
    existing = [
        int(m.group(1))
        for p in out_dir.glob(f"sugar_only_{hz_int}Hz_run_*.parquet")
        if (m := pattern.match(p.name))
    ]
    max_existing = max(existing) if existing else 0

    # 3) set up parameters
    # os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
    # os.environ["CXX"] = "/opt/homebrew/bin/g++-14"
    sweet_df = pd.read_csv("Data/sweet.csv")
    neu_sugar = [int(i) for i in sweet_df["root_id"]]
    params["r_poi"]  = args.hz * Hz
    params["r_poi2"] = 0 * Hz

    # 4) run the new experiments, numbering from max_existing + 1
    for offset in range(1, n_new + 1):
        run_idx   = max_existing + offset
        exp_name  = f"control_trials/sugar_only_{hz_int}Hz_run_{run_idx}"
        run_exp(
            exp_name=exp_name,
            neu_exc=neu_sugar,
            neu_exc2=[],
            params=params,
            path_comp="./Completeness_783.csv",
            path_con="./Connectivity_783.parquet",
            n_proc=-1,
            path_res=out_dir,
            force_overwrite=False,
        )

    print(
        f"Completed runs {max_existing+1} through "
        f"{max_existing+n_new} at {hz_int}Hz; files in {out_dir.resolve()}"
    )

if __name__ == "__main__":
    main()