#!/usr/bin/env python3
"""
Run GRN frequency sweep experiments.

This script can launch:

1. Sugar‑only controls.
2. Sugar + ORN combinations (experimental).

Choose what to run with the --subset flag:

    --subset controls      # run controls only
    --subset experiments   # run experiments only
    --subset both          # run both (default)

Example:
    python run_frequency_sweeps.py --subset controls
"""

import argparse
from collections import defaultdict
from pathlib import Path
import os

import pandas as pd
from brian2 import Hz

from model import run_exp
from model import default_params as params
import utils as utl

# ---------------------------------------------------------------------------
# paths and environment
# ---------------------------------------------------------------------------
os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
os.environ["CXX"] = "/opt/homebrew/bin/g++-14"

CONFIG = {
    "path_comp": "./Completeness_783.csv",
    "path_con" : "./Connectivity_783.parquet",
    "n_proc"   : -1,
    "path_res" : Path("./sweet_results/GRN"),
}

# output sub‑directories
DIR_CONTROLS    = CONFIG["path_res"] / "controls"
DIR_EXPERIMENTS = CONFIG["path_res"] / "experiments"
DIR_CONTROLS.mkdir(parents=True, exist_ok=True)
DIR_EXPERIMENTS.mkdir(parents=True, exist_ok=True)

# frequency sweeps
FREQS_SUGAR = [20, 25, 30]              # Hz for sweet neurons
FREQS_ORN   = list(range(20, 201, 10))  # Hz for ORNs

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GRN frequency sweeps")
    p.add_argument(
        "--subset",
        default="both",
        choices=["controls", "experiments", "both"],
        help="Which part of the sweep to run (default: both)",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # load neuron populations
    sweet_df = pd.read_csv("Data/sweet.csv")
    neu_sugar = [int(i) for i in sweet_df["root_id"]]

    orn_df = pd.read_csv("Data/filtered_orn.csv")
    cell_type_dict = defaultdict(list)
    for _, row in orn_df.iterrows():
        cell_type_dict[row["primary_type"].strip()].append(int(row["root_id"]))

    exp_names  = []
    file_paths = []

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    if args.subset in ("controls", "both"):
        for sugar_rate in FREQS_SUGAR:
            params["r_poi"]  = sugar_rate * Hz
            params["r_poi2"] = 0 * Hz

            exp_name = f"controls/sugar_only_{sugar_rate}Hz"
            run_exp(
                exp_name=exp_name,
                neu_exc=neu_sugar,
                neu_exc2=[],
                params=params,
                **CONFIG,
                force_overwrite=False,
            )
            exp_names.append(exp_name)
            file_paths.append(CONFIG["path_res"] / f"{exp_name}.parquet")

    # ------------------------------------------------------------------
    # experiments
    # ------------------------------------------------------------------
    if args.subset in ("experiments", "both"):
        for sugar_rate in FREQS_SUGAR:
            params["r_poi"] = sugar_rate * Hz

            for cell_type, orn_ids in cell_type_dict.items():
                for orn_rate in FREQS_ORN:
                    params["r_poi2"] = orn_rate * Hz

                    exp_name = (
                        f"experiments/"
                        f"sugar{str(sugar_rate)}Hz_plus_{cell_type}_{orn_rate}Hz"
                    )
                    run_exp(
                        exp_name=exp_name,
                        neu_exc=neu_sugar,
                        neu_exc2=orn_ids,
                        params=params,
                        **CONFIG,
                        force_overwrite=False,
                    )
                    exp_names.append(exp_name)
                    file_paths.append(CONFIG["path_res"] / f"{exp_name}.parquet")

    print(f"Finished running {len(exp_names)} experiments")

    # ------------------------------------------------------------------
    # rate matrices
    # ------------------------------------------------------------------
    if not file_paths:
        print("No experiments selected; exiting without analysis")
        return

    flyid2name = {nid: f"sugar_{idx + 1}" for idx, nid in enumerate(neu_sugar)}
    print("Computing firing rates…")
    df_spike = utl.load_exps([str(p) for p in file_paths])
    df_rate, df_std = utl.get_rate(
        df_spike,
        t_run=params["t_run"],
        n_run=params["n_run"],
        flyid2name=flyid2name,
    )

    root_out = CONFIG["path_res"].parent
    df_rate.fillna(0).to_csv(root_out / "all_rates.csv")
    df_std.fillna(0).to_csv(root_out / "all_rates_std.csv")
    print(f"Saved rate matrices to {root_out.resolve()}")

if __name__ == "__main__":
    main()
