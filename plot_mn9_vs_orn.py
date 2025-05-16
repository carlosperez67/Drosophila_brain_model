#!/usr/bin/env python3
"""
plot_mn9_vs_orn.py

Scatter + errorbar panels of MN9 firing rate vs ORN frequency,
for a given sugar Hz, comparing 5 ORN cell types and control baseline.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

# Force PyArrow as the Parquet engine (avoids fastparquet OSError issues)
pd.set_option("io.parquet.engine", "pyarrow")

import matplotlib.pyplot as plt

# model.default_params provides t_run, n_run
from model import default_params
import utils as utl  # provides load_exps & get_rate

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MN9 firing rate vs ORN frequency panels"
    )
    p.add_argument(
        "--control-dir",
        "-c",
        type=Path,
        default=Path("/project/def-mdgordon-ab/cperez67/RESULTS/control_trials"),
        help="Directory of control-trial parquet files",
    )
    p.add_argument(
        "--experiments-dir",
        "-e",
        type=Path,
        default=Path("./sweet_results/GRN/experiments"),
        help="Directory of ORN experiment parquet files",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to save the generated PNG plot(s)",
    )
    p.add_argument(
        "--hz",
        type=int,
        required=True,
        help="Sugar stimulation frequency (Hz) to plot",
    )
    p.add_argument(
        "--mn9",
        choices=["1","2","both"],
        default="1",
        help="Which MN9 neuron to plot (default: 1)",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# data collection & plotting
# ---------------------------------------------------------------------------
def collect_files(ctrl_dir, exp_dir, hz):
    ctrl_pattern = f"sugar_only_{hz}Hz_run_*.parquet"
    ctrls = sorted(ctrl_dir.glob(ctrl_pattern))
    if not ctrls:
        raise FileNotFoundError(f"No control files matching {ctrl_pattern}")
    exp_rx = re.compile(rf"sugar{hz}Hz_plus_(.+)_(\d+)Hz\.parquet$")
    exps = [p for p in sorted(exp_dir.glob(f"sugar{hz}Hz_plus_*_*.parquet"))
            if exp_rx.match(p.name)]
    if not exps:
        raise FileNotFoundError(f"No experiment files for sugar{hz}Hz in {exp_dir}")
    return ctrls, exps

def load_rates(parquet_paths):
    df_spike = utl.load_exps([str(p) for p in parquet_paths])
    return utl.get_rate(
        df_spike,
        t_run=default_params["t_run"],
        n_run=default_params["n_run"],
        flyid2name=None
    )

def parse_exp_info(exp_paths, hz):
    rx = re.compile(rf"sugar{hz}Hz_plus_(.+)_(\d+)Hz\.parquet$")
    info = []
    for p in exp_paths:
        m = rx.match(p.name)
        if m:
            cell, orn = m.group(1), int(m.group(2))
            info.append((cell, orn, p.stem))
    return info

def make_plot(df_rate, df_std, mn9_id, ctrl_cols, exp_info, hz, mn9_label, out_path):
    # control stats
    ctrl_vals = df_rate.loc[mn9_id, ctrl_cols].values
    ctrl_mean = np.mean(ctrl_vals)
    ctrl_sd   = np.std(ctrl_vals)

    # pick first five cell types
    cells = sorted({c for c,_,_ in exp_info})[:5]
    panels = cells + ["Control"]

    # global axis limits
    all_x, all_y = [], []
    for cell in cells:
        pts = [(orn, df_rate.at[mn9_id,stem]) for (cell0,orn,stem) in exp_info if cell0==cell]
        pts.sort(key=lambda x: x[0])
        xs, ys = zip(*pts)
        all_x += xs; all_y += ys
    all_y += list(ctrl_vals)
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    ncols = 3
    nrows = (len(panels) + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, key in enumerate(panels):
        ax = axes[idx]
        if key == "Control":
            ax.axhline(ctrl_mean, linestyle="--", label="mean")
            ax.fill_between([xmin-1, xmax+1],
                            [ctrl_mean-ctrl_sd]*2,
                            [ctrl_mean+ctrl_sd]*2,
                            alpha=0.2, label="+/-1SD")
        else:
            pts = [(orn,
                    df_rate.at[mn9_id,stem],
                    df_std.at[mn9_id,stem])
                   for (cell0,orn,stem) in exp_info if cell0==key]
            pts.sort(key=lambda x: x[0])
            xs, ys, errs = zip(*pts)
            ax.errorbar(xs, ys, yerr=errs, fmt="o-", label=key)
            for x,y in zip(xs,ys):
                if abs(y - ctrl_mean) > 2*ctrl_sd:
                    ax.text(x, y, "*", color="red", fontsize=12,
                            ha="center", va="bottom")
        ax.set_title(key)
        ax.set_xlim(xmin-1, xmax+1)
        ax.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))
        ax.set_xlabel("ORN freq (Hz)")
        ax.set_ylabel(f"MN9 {mn9_label} rate (Hz)")
        ax.legend(fontsize="small")

    # hide any extra axes
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"MN9 {mn9_label} firing rate vs ORN freq (sugar {hz}Hz)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def run_for_mn9(args, mn9_int):
    ctrls, exps = collect_files(args.control_dir, args.experiments_dir, args.hz)
    all_files = ctrls + exps

    df_rate, df_std = load_rates(all_files)
    mn9_id = df_rate.index[mn9_int-1]  # select row by position

    ctrl_cols = [p.stem for p in ctrls]
    exp_info  = parse_exp_info(exps, args.hz)

    out_name = f"mn9{mn9_int}_vs_orn_{args.hz}Hz.png"
    out_path = args.output_dir / out_name
    make_plot(df_rate, df_std, mn9_id, ctrl_cols, exp_info,
              args.hz, mn9_int, out_path)
    print(f"Saved plot: {out_path}")

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mn9 in ("1","2"):
        run_for_mn9(args, int(args.mn9))
    else:
        run_for_mn9(args, 1)
        run_for_mn9(args, 2)

if __name__ == "__main__":
    main()