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

Additionally, you may specify the sugar and ORN frequencies via command-line parameters.

Example:
    python run_frequency_sweeps.py --subset controls --sugar-hz 20 40 --orn-hz 50 100 150
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from brian2 import Hz

from model import run_exp
from model import default_params as params
import utils as utl

# ---------------------------------------------------------------------------
# paths and environment
# ---------------------------------------------------------------------------
# os.environ["CC"] = "/opt/homebrew/bin/gcc-14"
# os.environ["CXX"] = "/opt/homebrew/bin/g++-14"

CONFIG = {
    "path_comp": "./Completeness_783.csv",
    "path_con" : "./Connectivity_783.parquet",
    "path_res" : Path("./sweet_results/GRN"),
    "n_proc"   : -1,
}
DEFAULT_SUGAR_FREQS = [40]
DEFAULT_ORN_FREQS   = list(range(20, 201, 10))

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GRN frequency sweeps")
    p.add_argument(
        "--subset",
        default="both",
        choices=["controls", "experiments", "both"],
        help="Which part of the sweep to run (default: both)",
    )
    p.add_argument(
        "--res-dir",
        type=Path,
        default=CONFIG["path_res"],
        help="Root directory to save results (default: ./sweet_results/GRN)",
    )
    p.add_argument(
        "--sugar-hz",
        type=float,
        nargs="+",
        default=DEFAULT_SUGAR_FREQS,
        help="List of sugar frequencies in Hz (default: [40])",
    )
    p.add_argument(
        "--orn-hz",
        type=float,
        nargs="+",
        default=DEFAULT_ORN_FREQS,
        help="List of ORN frequencies in Hz (default: 20 to 200 step 10)",
    )
    p.add_argument(
        "--orn-types",
        type=str,
        nargs="+",
        default=None,
        metavar="CELL_TYPE",
        help="List of ORN cell types to stimulate (default: all available)"
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # override save path
    res_root: Path = args.res_dir
    CONFIG["path_res"] = res_root

    # prepare output subdirectories
    dir_controls    = res_root / "controls"
    dir_experiments = res_root / "experiments"
    dir_controls.mkdir(parents=True, exist_ok=True)
    dir_experiments.mkdir(parents=True, exist_ok=True)

    # load neuron populations
    sweet_df  = pd.read_csv("Data/sweet.csv")
    neu_sugar = [int(i) for i in sweet_df["root_id"]]

    orn_df = pd.read_csv("Data/filtered_orn.csv")
    cell_type_dict: dict[str, list[int]] = defaultdict(list)
    for _, row in orn_df.iterrows():
        cell_type_dict[row["primary_type"].strip()].append(int(row["root_id"]))

    # Filtering Requested ORN's
    if args.orn_types is not None:
        requested = set(args.orn_types)
        available = set(cell_type_dict.keys())
        missing = requested - available
        if missing:
            raise ValueError(f"Unknown ORN types: {', '.join(missing)}")
        cell_type_dict = {ct: cell_type_dict[ct] for ct in requested}

    exp_names: list[str]     = []
    file_paths: list[Path]   = []

    # controls
    if args.subset in ("controls", "both"):
        for sugar_rate in args.sugar_hz:
            params["r_poi"]  = sugar_rate * Hz
            params["r_poi2"] = 0 * Hz

            exp_name = f"controls/sugar_only_{int(sugar_rate)}Hz"
            run_exp(
                exp_name=exp_name,
                neu_exc=neu_sugar,
                neu_exc2=[],
                params=params,
                **CONFIG,
                force_overwrite=False,
            )
            exp_names.append(exp_name)
            file_paths.append(res_root / f"{exp_name}.parquet")

    # experiments
    if args.subset in ("experiments", "both"):
        for sugar_rate in args.sugar_hz:
            params["r_poi"] = sugar_rate * Hz
            for cell_type, orn_ids in list(cell_type_dict.items()):
                for orn_rate in args.orn_hz:
                    params["r_poi2"] = orn_rate * Hz

                    exp_name = (
                        f"experiments/"
                        f"sugar{int(sugar_rate)}Hz_plus_{cell_type}_{int(orn_rate)}Hz"
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
                    file_paths.append(res_root / f"{exp_name}.parquet")

    print(f"Finished running {len(exp_names)} experiments")

    if not file_paths:
        print("No experiments selected; exiting without analysis")
        return

    # compute firing rates and stds
    flyid2name = {nid: f"sugar_{idx+1}" for idx, nid in enumerate(neu_sugar)}
    print("Computing firing rates…")
    df_spike = utl.load_exps([str(p) for p in file_paths])
    df_rate, df_std = utl.get_rate(
        df_spike,
        t_run=params["t_run"],
        n_run=params["n_run"],
        flyid2name=flyid2name,
    )

    # save summary CSVs next to res_dir
    root_out = res_root.parent
    df_rate.fillna(0).to_csv(root_out / "all_rates.csv")
    df_std.fillna(0).to_csv(root_out / "all_rates_std.csv")
    print(f"Saved rate matrices to {root_out.resolve()}")


if __name__ == "__main__":
    main()
