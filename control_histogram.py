#!/usr/bin/env python3
"""
control_histogram.py

Run a series of control experiments by activating all sugar neurons
at a specified frequency multiple times, then plot and save a histogram
showing the variation in MN9 motor neuron firing rates across runs.
You can choose MN9 neuron 1, 2, or both.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from brian2 import Hz

from model import run_exp
from model import default_params as params
import utils as utl


# motor neuron IDs
MN9_LOOKUP = {
    "1": 720575940660219265,
    "2": 720575940618238523,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run control experiments and plot MN9 firing-rate histogram."
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
    parser.add_argument(
        "--mn9",
        choices=["1", "2", "both"],
        default="1",
        help="Which MN9 neuron to histogram (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hz_value = args.hz
    n_runs = args.n_runs
    mn9_choice = args.mn9

    # directories
    res_dir = Path("./sweet_results/GRN/controls_hist")
    res_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path("control_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # set compilers
    os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
    os.environ["CXX"] = "/opt/homebrew/bin/g++-14"

    # load sugar neuron IDs
    sweet_df = pd.read_csv("Data/sweet.csv")
    neu_sugar = [int(i) for i in sweet_df["root_id"]]

    # run control experiments
    parquet_paths = []
    params["r_poi"] = hz_value * Hz
    params["r_poi2"] = 0 * Hz

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

    # ensure only numeric columns (experiments)
    df_rate = df_rate.select_dtypes(include="number")

    # pick MN9 IDs
    if mn9_choice == "both":
        mn9_ids = list(MN9_LOOKUP.values())
    else:
        mn9_ids = [MN9_LOOKUP[mn9_choice]]

    # plot histogram
    plt.figure()
    for mn9_id in mn9_ids:
        try:
            rates = df_rate.loc[mn9_id]
        except KeyError:
            raise KeyError(f"MN9 ID {mn9_id} not found in results")
        plt.hist(
            rates,
            bins="auto",
            alpha=0.5 if len(mn9_ids) == 2 else 1.0,
            edgecolor="black",
            label=f"MN9 {mn9_id}",
        )
        m = rates.mean()
        med = rates.median()
        plt.axvline(m, linestyle="--", linewidth=1.5, label=f"Mean MN9 {mn9_id}: {m:.2f} Hz")
        plt.axvline(med, linestyle="-", linewidth=1.5, label=f"Median MN9 {mn9_id}: {med:.2f} Hz")

    plt.title(f"MN9 Firing Rates @ {hz_value:.0f} Hz over {n_runs} Runs")
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Count")
    plt.legend(fontsize="small")
    plt.tight_layout()

    # save and close
    out_file = plot_dir / f"mn9_hist_{int(hz_value)}Hz_{n_runs}runs_mn9{mn9_choice}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Histogram saved to {out_file.resolve()}")


if __name__ == "__main__":
    main()