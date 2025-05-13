# control_plot.py
#!/usr/bin/env python3
"""
Load saved MN9 firing rates and plot histogram.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MN9_LOOKUP = {
    "1": 720575940660219265,
    "2": 720575940618238523,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MN9 firing-rate histogram from CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV file with MN9 firing rates.",
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
    df_rate = pd.read_csv(args.csv, index_col=0)

    # select MN9 IDs
    if args.mn9 == "both":
        mn9_ids = MN9_LOOKUP.values()
    else:
        mn9_ids = [MN9_LOOKUP[args.mn9]]

    # prepare plot directory
    plot_dir = Path("control_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # plot
    plt.figure()
    for mn9_id in mn9_ids:
        try:
            rates = df_rate.loc[mn9_id]
        except KeyError:
            raise KeyError(f"MN9 ID {mn9_id} not found in {args.csv}")
        plt.hist(
            rates,
            bins="auto",
            alpha=0.5 if len(mn9_ids) == 2 else 1.0,
            edgecolor="black",
            label=f"MN9 {mn9_id}",
        )
        mean = rates.mean()
        median = rates.median()
        plt.axvline(mean, linestyle="--", linewidth=1.5,
                    label=f"Mean: {mean:.2f} Hz")
        plt.axvline(median, linestyle="-", linewidth=1.5,
                    label=f"Median: {median:.2f} Hz")

    hz_str, runs_str = args.csv.stem.split("_")[2:4]
    plt.title(f"MN9 Firing Rates @ {hz_str} Hz over {runs_str}")
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Count")
    plt.legend(fontsize="small")
    plt.tight_layout()

    out_file = plot_dir / f"mn9_hist_{hz_str}_{runs_str}_mn9{args.mn9}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Histogram saved to {out_file.resolve()}")


if __name__ == "__main__":
    main()