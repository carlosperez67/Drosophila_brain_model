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
    "2": 720575940660618238523,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MN9 histogram and update summary stats."
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
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=Path("/Volumes/T7/GordonLab/control_trials"),
        help="Directory containing control_trial parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Directory to save plots (default: current working directory).",
    )
    return parser.parse_args()

def load_summary(summary_file: Path) -> pd.DataFrame:
    if summary_file.exists():
        return pd.read_csv(summary_file, index_col="hz")
    else:
        return pd.DataFrame(
            columns=["hz", "n_runs", "mean", "median", "std"]
        ).set_index("hz")


def save_summary(df: pd.DataFrame, summary_file: Path):
    df.to_csv(summary_file, index=True)
def main():
    args = parse_args()
    hz_int = int(args.hz)

    control_dir = args.control_dir
    summary_file = control_dir / "summary_stats.csv"
    summary_df = load_summary(summary_file)

    # find all parquet runs
    pattern = f"sugar_only_{hz_int}Hz_run_*.parquet"
    paths = sorted(control_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files matching {pattern}")
    n_runs = len(paths)

    # compute rates
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

    # update summary if n_runs increased
    for mid in target_ids:
        rates = df_rate.loc[mid]
        mu, med, sd = rates.mean(), rates.median(), rates.std()
        old = summary_df.loc[hz_int]["n_runs"] if hz_int in summary_df.index else 0
        if n_runs > old:
            summary_df.loc[hz_int] = [n_runs, mu, med, sd]

    save_summary(summary_df, summary_file)

    # create output directory
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # plot histograms
    plt.figure()
    for mid in target_ids:
        if mid not in df_rate.index:
            raise KeyError(f"MN9 ID {mid} not in results")
        rates = df_rate.loc[mid]
        mu = summary_df.loc[hz_int, "mean"]
        med = summary_df.loc[hz_int, "median"]
        sd = summary_df.loc[hz_int, "std"]

        plt.hist(
            rates,
            bins="auto",
            alpha=0.5 if len(target_ids) == 2 else 1.0,
            edgecolor="black",
            label=f"MN9 {mid}",
        )
        plt.axvline(mu, linestyle="--", linewidth=1.5, label=f"Mean {mu:.2f} Hz")
        plt.axvline(med, linestyle="-", linewidth=1.5, label=f"Median {med:.2f} Hz")
        plt.axvline(mu + sd, linestyle=":", linewidth=1.5, label=f"Mean +1σ {mu+sd:.2f} Hz")
        plt.axvline(mu - sd, linestyle=":", linewidth=1.5, label=f"Mean -1σ {mu-sd:.2f} Hz")

    plt.title(f"MN9 Firing Rates @ {hz_int} Hz over {n_runs} runs")
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Count")
    plt.legend(fontsize="small")
    plt.tight_layout()

    out_file = out_dir / f"mn9_hist_{hz_int}Hz_{n_runs}runs_mn9_{args.mn9}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Histogram saved to {out_file.resolve()}")
    print(f"Updated summary stats at {summary_file}")


if __name__ == "__main__":
    main()