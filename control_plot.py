# control_plot.py
#!/usr/bin/env python3
"""
Scan the controls_hist directory for any completed runs at a given Hz,
compute MN9 firing rates from the parquet files, and plot a histogram.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import utils as utl
from model import default_params as params

MN9_LOOKUP = {
    "1": 720575940660219265,
    "2": 720575940618238523,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MN9 histogram from existing parquet runs."
    )
    parser.add_argument(
        "--hz",
        type=float,
        required=True,
        help="Activation frequency that identifies the runs (in Hz).",
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
    hz_int = int(args.hz)

    base = Path("./sweet_results/GRN/controls_hist")
    pattern = f"sugar_only_{hz_int}Hz_run_*.parquet"
    paths = sorted(base.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files matching {pattern}")

    # load spikes and compute rates
    df_spike = utl.load_exps([str(p) for p in paths])
    df_rate, _ = utl.get_rate(
        df_spike,
        t_run=params["t_run"],
        n_run=params["n_run"],
        flyid2name={},
    )
    df_rate = df_rate.select_dtypes(include="number")

    # select target MN9(s)
    if args.mn9 == "both":
        target_ids = MN9_LOOKUP.values()
    else:
        target_ids = [MN9_LOOKUP[args.mn9]]

    # plot
    out_dir = Path("control_plots")
    out_dir.mkdir(exist_ok=True)

    plt.figure()
    for mid in target_ids:
        if mid not in df_rate.index:
            raise KeyError(f"MN9 ID {mid} not in results")
        rates = df_rate.loc[mid]
        plt.hist(rates, bins="auto",
                 alpha=0.5 if len(target_ids) == 2 else 1.0,
                 edgecolor="black",
                 label=f"MN9 {mid}")
        mu, med = rates.mean(), rates.median()
        plt.axvline(mu, linestyle="--", linewidth=1.5,
                    label=f"Mean {mu:.2f} Hz")
        plt.axvline(med, linestyle="-", linewidth=1.5,
                    label=f"Median {med:.2f} Hz")

    plt.title(f"MN9 Firing Rates @ {hz_int} Hz over {len(paths)} runs")
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Count")
    plt.legend(fontsize="small")
    plt.tight_layout()

    out_file = out_dir / f"mn9_hist_{hz_int}Hz_{len(paths)}runs_mn9_{args.mn9}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Histogram saved to {out_file.resolve()}")


if __name__ == "__main__":
    main()