#!/usr/bin/env python3
"""
Plot MN9 firing rate responses from parquet files.

Folder layout assumed:

    ./sweet_results/GRN/controls       # sugar only experiments
    ./sweet_results/GRN/experiments   # sugar plus ORN experiments

Examples
--------
Plot both MN9 neurons for every experiment::

    python plot_parquet_results.py

Plot only MN9 1 for experiments conditions::

    python plot_parquet_results.py --mn9 1 --subset experiments

Parameters
----------
--mn9 {1,2,both}        Which MN9 neuron to plot (default: both)
--subset {controls,experiments,both}
                        Select which experiment set to include (default: both)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# project helper modules
from model import default_params as params  # provides t_run, n_run
import utils as utl

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path("./sweet_results/GRN")
DIR_PLOTS = ROOT_DIR / "plots"
DIR_PLOTS.mkdir(parents=True, exist_ok=True)
DIR_CONTROLS = ROOT_DIR / "controls"
DIR_EXPER = ROOT_DIR / "experiments"
PATH_SWEET = Path("Data/sweet.csv")

MN9_LOOKUP: Dict[str, int] = {
    "1": 720575940660219265,
    "2": 720575940618238523,
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MN9 firing rates across experiments")
    p.add_argument("--mn9", default="2", choices=["1", "2", "both"],
                   help="MN9 neuron to plot (default: 2)")
    p.add_argument("--subset", default="both", choices=["controls", "experiments", "both"],
                   help="Plot controls, experiments, or both (default: both)")
    p.add_argument("--save", action="store_true", help="Save plots as PNG files")
    return p.parse_args()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def gather_parquet_files(subset: str) -> List[Path]:
    files: List[Path] = []
    if subset in ("controls", "both") and DIR_CONTROLS.is_dir():
        files.extend(sorted(DIR_CONTROLS.glob("*.parquet")))
    if subset in ("experiments", "both") and DIR_EXPER.is_dir():
        files.extend(sorted(DIR_EXPER.glob("*.parquet")))
    return files

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # choose MN9 ids
    mn9_ids = list(MN9_LOOKUP.values()) if args.mn9 == "both" else [MN9_LOOKUP[args.mn9]]

    # gather parquet files based on subset flag
    file_paths = gather_parquet_files(args.subset)
    if not file_paths:
        raise FileNotFoundError("No parquet files found for requested subset")
    print(f"Found {len(file_paths)} parquet files to process")

    # build sugar id label mapping
    sweet_df = pd.read_csv(PATH_SWEET)
    neu_sugar = [int(i) for i in sweet_df["root_id"]]
    flyid2name = {nid: f"sugar_{idx+1}" for idx, nid in enumerate(neu_sugar)}

    # load spikes and compute firing rates
    df_spike = utl.load_exps([str(p) for p in file_paths])
    df_rate, _ = utl.get_rate(df_spike, t_run=params["t_run"], n_run=params["n_run"], flyid2name=flyid2name)

    # ensure MN9 rows present
    try:
        df_mn9 = df_rate.loc[mn9_ids]
    except KeyError as exc:
        raise KeyError("MN9 neuron id not found in data") from exc

    # build condition mapping: condition -> freq -> {mn: rate}
    cond_points: Dict[str, Dict[int, Dict[int, float]]] = {}
    for col in df_mn9.columns:
        if not col.endswith("Hz"):
            continue
        *cond_parts, freq_part = col.split("_")
        freq = int(freq_part.replace("Hz", ""))
        condition = "_".join(cond_parts)
        cond_points.setdefault(condition, {}).setdefault(freq, {})
        for mn in mn9_ids:
            cond_points[condition][freq][mn] = df_mn9.at[mn, col]

    if not cond_points:
        raise ValueError("No columns with frequency tag found in data")

    # plot
    plt.figure(figsize=(10, 7))
    for condition, freq_dict in sorted(cond_points.items()):
        freqs_sorted = sorted(freq_dict.keys())
        for mn in mn9_ids:
            y_vals = [freq_dict[f][mn] for f in freqs_sorted]
            label = condition if len(mn9_ids) == 1 else f"{condition} (MN9 {mn})"
            plt.plot(freqs_sorted, y_vals, marker="o", label=label)

    # plt.figure(figsize=(10, 7))
    # for condition, freq_dict in sorted(cond_points.items()):
    #     freqs_sorted = sorted(freq_dict.keys())
    #     for mn in mn9_ids:
    #         y_vals = [freq_dict[f][mn] for f in freqs_sorted]
    #         label = condition if len(mn9_ids) == 1 else f"{condition} (MN9 {mn})"
    #         plt.plot(freqs_sorted, y_vals, marker="o", label=label)
    #         # Annotate each point with its y-value
    #         for x, y in zip(freqs_sorted, y_vals):
    #             plt.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=7)

    plt.xlabel("Stimulus frequency (Hz)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("MN9 firing rate vs stimulus frequency")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    if args.save:
        out_name = f"mn9_plot_{args.mn9}_{args.subset}.png"
        out_path = DIR_PLOTS / out_name
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
    # plt.show()

if __name__ == "__main__":
    main()
