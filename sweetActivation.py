#!/usr/bin/env python3
"""
Run GRN frequency sweep experiments (ASCII only).

This script performs two sets of simulations:

1. Sugar-only controls           : sweet neurons excited at 20, 25, 30 Hz.
2. Sugar + ORN combinations      : the same sweet neurons plus each ORN
   cell-type population.  For every sugar rate above, ORNs are excited at
   rates from 20 to 200 Hz in 10 Hz steps.

Sweet neurons use the Poisson rate stored in ``params['r_poi']``.
ORN neurons use the second Poisson rate stored in ``params['r_poi2']``.

Plotting code has been removed.  Use the separate script
``plot_parquet_results.py`` to visualise the results once the simulations
have finished.
"""

from collections import defaultdict
from pathlib import Path

from brian2 import Hz
import pandas as pd

from model import run_exp
from model import default_params as params
import utils as utl
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
os.environ['CC']  = '/opt/homebrew/bin/gcc-14'
os.environ['CXX'] = '/opt/homebrew/bin/g++-14'

CONFIG = {
    "path_comp": "./Completeness_783.csv",
    "path_con" : "./Connectivity_783.parquet",
    "n_proc"   : -1,
    "path_res" : Path("./sweet_results/GRN"),
}

(CONFIG["path_res"] / "controls").mkdir(parents=True, exist_ok=True)
(CONFIG["path_res"] / "experiments").mkdir(parents=True, exist_ok=True)


# Frequency sweeps
FREQS_SUGAR = [20, 25, 30]                  # Hz for sweet neurons
FREQS_ORN   = list(range(20, 201, 10))      # Hz for ORNs

# ---------------------------------------------------------------------------
# Load neuron populations
# ---------------------------------------------------------------------------
# Sweet GRN IDs
sweet_df = pd.read_csv("Data/sweet.csv")
neu_sugar = [int(i) for i in sweet_df["root_id"]]

# ORN IDs grouped by primary cell type
orn_df = pd.read_csv("Data/filtered_orn.csv")
cell_type_dict = defaultdict(list)
for _, row in orn_df.iterrows():
    cell_type_dict[row["primary_type"].strip()].append(int(row["root_id"]))

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
exp_names  = []
file_paths = []

# 1) Sugar-only controls -----------------------------------------
for sugar_rate in FREQS_SUGAR:
    params["r_poi"]  = sugar_rate * Hz   # sweet neuron rate
    params["r_poi2"] = 0 * Hz            # ORNs inactive

    # exp_name = f"sugar_only_{sugar_rate}Hz"
    exp_name = f"controls/sugar_only_{sugar_rate}Hz"

    run_exp(
        exp_name=exp_name,
        neu_exc=neu_sugar,
        neu_exc2=[],
        params=params,
        **CONFIG,
        force_overwrite=False,
    )

    exp_names.append(exp_name)
    file_paths.append(f"{CONFIG['path_res']}/controls/{exp_name}.parquet")

# 2) Sugar + ORN combinations ------------------------------------
for sugar_rate in FREQS_SUGAR:
    params["r_poi"] = sugar_rate * Hz  # sweet neuron rate

    for cell_type, orn_ids in cell_type_dict.items():
        for orn_rate in FREQS_ORN:
            params["r_poi2"] = orn_rate * Hz  # ORN rate

            # exp_name = (
            #     f"sugar{str(sugar_rate)}Hz_"
            #     f"plus_{cell_type}_{orn_rate}Hz"
            # )
            exp_name = f"experiments/sugar{str(sugar_rate)}Hz_plus_{cell_type}_{orn_rate}Hz"

            run_exp(
                exp_name=exp_name,
                neu_exc=neu_sugar,
                neu_exc2=orn_ids,
                params=params,
                **CONFIG,
                force_overwrite=False,
            )

            exp_names.append(exp_name)
            file_paths.append(f"{CONFIG['path_res']}/experiments/{exp_name}.parquet")

print(f"Finished running {len(exp_names)} experiments.")

# ---------------------------------------------------------------------------
# Compute and store firing-rate matrices
# ---------------------------------------------------------------------------
flyid2name = {nid: f"sugar_{idx + 1}" for idx, nid in enumerate(neu_sugar)}

print("Loading spike data and computing firing rates...")
df_spike = utl.load_exps(file_paths)

df_rate, df_std = utl.get_rate(
    df_spike,
    t_run=params["t_run"],
    n_run=params["n_run"],
    flyid2name=flyid2name,
)

root_out = CONFIG["path_res"].parent
(df_rate.fillna(0)).to_csv(root_out / "all_rates.csv")
(df_std.fillna(0)).to_csv(root_out / "all_rates_std.csv")

print(f"Saved rate matrices to {root_out.resolve()}")
