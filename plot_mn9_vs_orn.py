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
import matplotlib.pyplot as plt

import utils as utl  # assumes load_exps & get_rate live here

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MN9 firing rate vs ORN frequency panels"
    )
    p.add_argument(
        "--control-dir",
        type=Path,
        default=Path("/project/def‑mdgordon‑ab/cperez67/RESULTS/control_trials"),
        help="Directory of control-trial parquets (sugar_only_{hz}Hz_run_*.parquet)",
    )
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("./sweet_results/GRN/experiments"),
        help="Directory of ORN experiment parquets (sugar{hz}Hz_plus_*_{orn}Hz.parquet)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to save the PNG plots",
    )
    p.add_argument(
        "--hz",
        type=int,
        required=True,
        help="Sugar stimulation frequency (Hz) to plot",
    )
    p.add_argument(
        "--mn9",
        type=int,
        choices=[1,2],
        default=1,
        help="Which MN9 neuron to plot (1 or 2)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    ctrl_dir = args.control_dir
    exp_dir  = args.experiments_dir
    out_dir  = args.output_dir
    hz       = args.hz
    mn9      = args.mn9

    out_dir.mkdir(parents=True, exist_ok=True)

    # find control files for this sugar Hz
    ctrl_pattern = f"sugar_only_{hz}Hz_run_*.parquet"
    ctrl_files = sorted(ctrl_dir.glob(ctrl_pattern))
    if not ctrl_files:
        raise FileNotFoundError(f"No controls found: {ctrl_pattern}")

    # find ORN experiment files for this sugar Hz
    exp_pattern = f"sugar{hz}Hz_plus_*_?????Hz.parquet"
    # the ????: block matches the ORN-frequency number
    exp_files = sorted(exp_dir.glob(f"sugar{hz}Hz_plus_*_*.parquet"))
    if not exp_files:
        raise FileNotFoundError(f"No experiments found in {exp_dir} for sugar{hz}Hz")

    # load & compute rates
    all_files = ctrl_files + exp_files
    df_spike = utl.load_exps([str(p) for p in all_files])
    df_rate, df_std = utl.get_rate(df_spike,
                                   t_run=utl.default_params["t_run"],
                                   n_run=utl.default_params["n_run"],
                                   flyid2name=None)

    # neuron ID lookup: assume returned index is the actual neuron id
    mn9_id = list(df_rate.index)[mn9-1]  # if index sorted so that MN9 1 is first, else adjust

    # control statistics
    ctrl_cols = [p.stem for p in ctrl_files]
    ctrl_vals = df_rate.loc[mn9_id, ctrl_cols].values
    ctrl_mean = float(np.mean(ctrl_vals))
    ctrl_sd   = float(np.std(ctrl_vals))

    # parse experiments by cell type
    # col names = file stem = 'sugar{hz}Hz_plus_{cell}_{orn}Hz'
    exp_info = []
    rx = re.compile(rf"sugar{hz}Hz_plus_(.+)_(\d+)Hz$")
    for p in exp_files:
        stem = p.stem
        m = rx.match(stem)
        if not m:
            continue
        cell, orn = m.group(1), int(m.group(2))
        exp_info.append((cell, orn, stem))

    # pick first five cell types alphabetically
    cells = sorted({info[0] for info in exp_info})[:5]
    panels = cells + ["Control"]
    n_panels = len(panels)
    ncols = 3
    nrows = (n_panels + ncols - 1)//ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 3*nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, key in enumerate(panels):
        ax = axes[i]
        if key == "Control":
            # horizontal band
            ax.axhline(ctrl_mean, linestyle="--", label="mean")
            ax.fill_between(
                [0, max([orn for (_,orn,_) in exp_info])+5],
                [ctrl_mean-ctrl_sd]*2,
                [ctrl_mean+ctrl_sd]*2,
                alpha=0.2, label="±1 SD"
            )
            ax.set_title("Control")
        else:
            # gather points
            pts = [(orn, float(df_rate.loc[mn9_id, stem]),
                    float(df_std.loc[mn9_id, stem]))
                   for (cell,orn,stem) in exp_info if cell==key]
            pts = sorted(pts, key=lambda x: x[0])
            x, y, yerr = zip(*pts)

            # scatter + errorbars
            ax.errorbar(x, y, yerr=yerr, fmt="o-", label=key)

            # significance: |mean - ctrl_mean| > 2*ctrl_sd
            for xi, yi in zip(x,y):
                if abs(yi - ctrl_mean) > 2*ctrl_sd:
                    ax.text(xi, yi, "*", color="red", fontsize=12,
                            ha="center", va="bottom")

            ax.set_title(key)

        ax.set_xlabel("ORN freq (Hz)")
        ax.set_ylabel(f"MN9_{mn9} rate (Hz)")
        ax.legend(fontsize="small")

    # hide unused
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"MN9_{mn9} firing rate vs ORN freq  (sugar {hz} Hz)", y=1.02)
    plt.tight_layout()

    out_file = out_dir / f"mn9{mn9}_vs_orn_{hz}Hz.png"
    fig.savefig(out_file, dpi=200)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()