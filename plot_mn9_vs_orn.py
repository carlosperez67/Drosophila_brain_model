#!/usr/bin/env python3
"""
plot_mn9_vs_orn.py

Scatter + errorbar panels of MN9 firing rate vs ORN frequency,
for a given sugar Hz, comparing 5 ORN cell types and the control baseline,
using precomputed control summary stats.
"""

import argparse
import re
from pathlib import Path
import warnings


import numpy as np
import pandas as pd
# pd.set_option("io.parquet.engine", "pyarrow")

import matplotlib.pyplot as plt
from model import default_params
import utils as utl  # provides load_exps & get_rate

MN9_LOOKUP = {
    "1": 720575940660219265,
    "2": 720575940618238523,
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MN9 firing rate vs ORN frequency panels"
    )
    p.add_argument(
        "--summary-stats", "-s",
        type=Path,
        default=Path("/project/def-mdgordon-ab/cperez67/RESULTS/control_trials/summary_stats.csv"),
        help="Path to the control_trials summary_stats.csv",
    )
    p.add_argument(
        "--experiments-dir", "-e",
        type=Path,
        default=Path("/home/cperez67/projects/cperez67/grn_sweeps/grn_sweeps_45Hz/experiments"),
        help="Directory of ORN experiment parquet files",
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("/home/cperez67/projects/cperez67/RESULTS/grn_plots"),
        help="Directory to save the PNG plot",
    )
    p.add_argument(
        "--hz",
        type=int,
        required=True,
        help="Sugar stimulation frequency (Hz) to plot",
    )
    p.add_argument(
        "--mn9",
        choices=["1", "2", "both"],
        default="1",
        help="Which MN9 neuron to plot (default: 1)",
    )
    return p.parse_args()

def load_control_stats(summary_csv: Path, hz: int):
    df = pd.read_csv(summary_csv, index_col="hz")
    if hz not in df.index:
        raise KeyError(f"No summary for hz={hz} in {summary_csv}")
    row = df.loc[hz]
    return float(row["mean"]), float(row["std"]), int(row["n_runs"])

def load_experiment_rates(exp_dir: Path, hz: int, mn9_idx: int):
    # find all parquet files for this sugar Hz
    files = sorted(exp_dir.glob(f"sugar{hz}Hz_plus_*_[0-9]*Hz.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files matching sugar{hz}Hz_plus_*_XHz.parquet in {exp_dir}")

    # extract cell type and ORN freq, keep raw stem for mapping
    pattern = re.compile(rf"sugar{hz}Hz_plus_(.+)_(\d+)Hz\.parquet")
    preliminary = []
    for file in files:
        m = pattern.match(file.name)
        if not m:
            warnings.warn(f"Skipping unrecognized filename: {file.name}")
            continue
        cell, orn = m.group(1), int(m.group(2))
        preliminary.append((cell, orn, file.stem))

    # load spike DataFrame and compute rate & std
    df_spike = utl.load_exps([str(file) for file in files])
    df_rate, df_std = utl.get_rate(
        df_spike,
        t_run=default_params["t_run"],
        n_run=default_params["n_run"],
        flyid2name={}
    )

    # keep only numeric columns (the exp_name groups)
    df_rate = df_rate.select_dtypes(include="number")
    df_std  = df_std.select_dtypes(include="number")

    # map each preliminary stem to the actual column in df_rate
    exp_info = []
    for cell, orn, stem in preliminary:
        if stem in df_rate.columns:
            col = stem
        elif f"{stem}.parquet" in df_rate.columns:
            col = f"{stem}.parquet"
        else:
            # try any column that ends with the stem
            matches = [c for c in df_rate.columns if c.endswith(stem)]
            if len(matches) == 1:
                col = matches[0]
            elif len(matches) > 1:
                # further narrow to exact endmatch
                exact = [c for c in matches if c.split("/")[-1].startswith(stem)]
                col = exact[0] if len(exact) == 1 else None
            else:
                col = None

        if col is None:
            raise KeyError(f"Cannot find matching df_rate column for '{stem}'")
        exp_info.append((cell, orn, col))

    # select MN9 by integer index
    mn9_id = df_rate.index[mn9_idx - 1]
    return df_rate, df_std, mn9_id, exp_info

def make_plot(df_rate, df_std, mn9_id, exp_info,
              ctrl_mean, ctrl_sd, hz, mn9_label, out_path):
    # Determine unique cell types and ORN range
    cells = sorted({cell for cell, _, _ in exp_info})
    orn_vals = sorted({orn for _, orn, _ in exp_info})
    xmin, xmax = min(orn_vals), max(orn_vals)

    plt.figure(figsize=(8, 6))

    # Plot each cell type’s curve with error bars
    for cell in cells:
        pts = sorted([
            (orn,
             df_rate.at[mn9_id, col],
             df_std.at[mn9_id, col])
            for (c, orn, col) in exp_info if c == cell
        ], key=lambda x: x[0])
        xs, ys, errs = zip(*pts)
        plt.errorbar(xs, ys, yerr=errs, fmt='o-', label=cell)

    # Overlay control mean ±1 SD
    plt.axhline(ctrl_mean, linestyle='--', label='Control mean')
    plt.fill_between(
        [xmin, xmax],
        [ctrl_mean - ctrl_sd] * 2,
        [ctrl_mean + ctrl_sd] * 2,
        alpha=0.2,
        label='Control ±1 SD'
    )

    # Labels, legend, title
    plt.xlabel("ORN freq (Hz)")
    plt.ylabel(f"MN9 {mn9_label} rate (Hz)")
    plt.title(f"MN9 {mn9_label} firing rate vs ORN freq (sugar {hz}Hz)")
    plt.legend(fontsize="small")
    plt.tight_layout()

    # Save and close
    plt.savefig(out_path, dpi=200)
    plt.close()

def run_for_mn9(args, mn9_int: int):
    ctrl_mean, ctrl_sd, _ = load_control_stats(args.summary_stats, args.hz)
    df_rate, df_std, mn9_id, exp_info = load_experiment_rates(
        args.experiments_dir, args.hz, mn9_int
    )
    mn9_id = MN9_LOOKUP[str(mn9_int)]
    out_file = args.output_dir / f"mn9{mn9_int}_vs_orn_{args.hz}Hz.png"
    make_plot(df_rate, df_std, mn9_id, exp_info,
              ctrl_mean, ctrl_sd, args.hz, str(mn9_int), out_file)
    print(f"Saved plot: {out_file}")

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mn9 in ("1", "2"):
        run_for_mn9(args, int(args.mn9))
    else:
        run_for_mn9(args, 1)
        run_for_mn9(args, 2)

if __name__ == "__main__":
    main()