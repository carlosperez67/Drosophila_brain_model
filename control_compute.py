# control_compute.py
#!/usr/bin/env python3
"""
Run control experiments and save MN9 firing rates to CSV.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from brian2 import Hz

from model import run_exp, default_params as params
import utils as utl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run control experiments and save MN9 firing rates."
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
        help="Number of repetitions of the control experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hz_value = args.hz
    n_runs = args.n_runs

    # prepare output directory
    out_dir = Path("./sweet_results/GRN/controls_hist")
    out_dir.mkdir(parents=True, exist_ok=True)

    # set compilers
    # os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
    # os.environ["CXX"] = "/opt/homebrew/bin/g++-14"

    # load sugar neuron IDs
    sweet_df = pd.read_csv("Data/sweet.csv")
    neu_sugar = [int(i) for i in sweet_df["root_id"]]

    # configure rates
    params["r_poi"] = hz_value * Hz
    params["r_poi2"] = 0 * Hz

    # run experiments
    parquet_paths = []
    for i in range(1, n_runs + 1):
        exp_name = f"controls_hist/sugar_only_{int(hz_value)}Hz_run_{i}"
        run_exp(
            exp_name=exp_name,
            neu_exc=neu_sugar,
            neu_exc2=[],
            params=params,
            path_comp="./Completeness_783.csv",
            path_con="./Connectivity_783.parquet",
            n_proc=-1,
            path_res=Path("./sweet_results/GRN"),
            force_overwrite=False,
        )
        parquet_paths.append(Path("./sweet_results/GRN") / f"{exp_name}.parquet")

    # compute firing rates
    df_spike = utl.load_exps([str(p) for p in parquet_paths])
    df_rate, _ = utl.get_rate(
        df_spike,
        t_run=params["t_run"],
        n_run=params["n_run"],
        flyid2name={nid: f"sugar_{idx+1}" for idx, nid in enumerate(neu_sugar)},
    )

    # save numeric-only rates to CSV
    df_rate = df_rate.select_dtypes(include="number")
    out_file = out_dir / f"mn9_rates_{int(hz_value)}Hz_{n_runs}runs.csv"
    df_rate.to_csv(out_file)

    print(f"Firing rates saved to {out_file.resolve()}")


if __name__ == "__main__":
    main()