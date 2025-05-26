#!/usr/bin/env python3
"""Two-way ANOVA on MN9 firing rates (trial level, Parquet source)."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from brian2 import second
from statsmodels.formula.api import ols
from constants import CELL_LOOKUP_REV, CELL_LOOKUP, build_flyid2name
from summarize_rate import filter_parquets          # already written
from model import default_params
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trial-level two-way ANOVA (ORN cell × ORN Hz)"
    )
    p.add_argument("-e", "--res-dir", type=Path, required=True,
                   help="Sweep directory containing controls/ and experiments/")
    p.add_argument("--mn9", default="MN9_LEFT", help="MN9 neuron label")
    p.add_argument("--sugar-hz", type=int, default=45)
    p.add_argument("--out", default="anova_results", type=Path,
                   help="Directory to store ANOVA & post-hoc outputs")
    return p.parse_args()

import pyarrow as pa
import pyarrow.dataset as ds

def iter_mn9_rates(files: list[Path], mn9_label: str, hz: int):
    # integer ID for the MN9 cell
    try:
        cell_id = CELL_LOOKUP[mn9_label]
    except KeyError:
        raise ValueError(f"Unknown MN9 label {mn9_label}")

    pat = re.compile(rf"sugar{hz}Hz_plus_(.+)_(\d+)Hz", re.I)

    for f in files:
        m = pat.search(f.name)
        if not m:
            continue          # controls or malformed
        orn_cell, orn_hz = m.group(1), int(m.group(2))

        # -------- Arrow scan with predicate ---------------------------------
        ds_file = ds.dataset(str(f), format="parquet")
        tbl = ds_file.to_table(
            # filter=ds.field("flywire_id") == pa.scalar(cell_id),
            columns=["trial", "t"]             # only the 2 columns we need
        )

        # convert to pandas (still tiny: ≤ (#spikes rows) × 2 cols)
        df = tbl.to_pandas()

        # spike count per trial  →  firing rate (Hz)
        dur_s = float(default_params["t_run"] / second)  # seconds as float

        rate_per_trial = (
            df.groupby("trial", sort=False)["t"]
            .count()
            .rename("Rate")
            .div(dur_s)  # safe scalar now
            .reset_index(drop=True)
        )

        yield pd.DataFrame({
            "Rate":     rate_per_trial,
            "ORN_Cell": orn_cell,
            "ORN_Hz":   orn_hz
        })


def build_trial_table_stream(res_dir: Path, sugar_hz: int, mn9_label: str) -> pd.DataFrame:
    controls, experiments = filter_parquets(res_dir, sugar_hz,
                                            orn_hz=None, type_sel=None)
    files = controls + experiments
    if not files:
        raise FileNotFoundError("No Parquet files found.")

    chunks = [df_chunk for df_chunk in iter_mn9_rates(files, mn9_label, sugar_hz)]
    return pd.concat(chunks, ignore_index=True)

def run_anova(df: pd.DataFrame, args) -> None:
    print(pd.crosstab(df["ORN_Cell"], df["ORN_Hz"]))  # quick balance check
    model = ols("Rate ~ C(ORN_Cell) * C(ORN_Hz)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=3)
    print("\nTwo-way ANOVA (interaction included)\n")
    print(anova)
    save_anova(args.out, anova, model)  # <-- new

def run_per_cell_anova(df: pd.DataFrame) -> None:
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    for cell in df["ORN_Cell"].unique():
        sub_df = df[df["ORN_Cell"] == cell]
        print(f"\n--- ANOVA for ORN_Cell: {cell} ---")
        if sub_df["ORN_Hz"].nunique() < 2:
            print("  Skipping: not enough frequency conditions.")
            continue

        model = ols("Rate ~ C(ORN_Hz)", data=sub_df).fit()
        anova = sm.stats.anova_lm(model, typ=3)  # or typ=2 if balanced
        print(anova)

def run_grouped_anova(df: pd.DataFrame, pattern: str) -> None:
    # Filter ORN_Cells matching the given regex pattern
    sub_df = df[df["ORN_Cell"].str.contains(pattern, flags=re.IGNORECASE)]
    print(f"\nSubset size: {len(sub_df)} entries matching '{pattern}'")

    if sub_df.empty:
        print("No matching data.")
        return

    model = ols("Rate ~ C(ORN_Cell) * C(ORN_Hz)", data=sub_df).fit()
    anova = sm.stats.anova_lm(model, typ=3)
    print(anova)



def save_anova(out_dir: Path, anova_tbl: pd.DataFrame, model):
    out_dir.mkdir(parents=True, exist_ok=True)
    anova_path = out_dir / "two_way_anova.csv"
    anova_tbl.to_csv(anova_path)
    print(f"Wrote ANOVA table → {anova_path}")

    # Pair-wise Tukey across ORN_Cell, collapsing over ORN_Hz
    tukey = pairwise_tukeyhsd(model.model.endog,
                              model.model.data.frame["ORN_Cell"])
    tukey_path = out_dir / "tukey_orn_cell.csv"
    pd.DataFrame(data=tukey.summary().data[1:],
                 columns=tukey.summary().data[0]).to_csv(tukey_path,
                                                         index=False)
    print(f"Wrote Tukey HSD → {tukey_path}")

    # Optional residual QQ-plot
    import matplotlib.pyplot as plt
    sm.qqplot(model.resid, line='45')
    qq_path = out_dir / "residuals_qq.png"
    plt.savefig(qq_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote residual QQ-plot → {qq_path}")

def main() -> None:
    args = parse_args()

    df = build_trial_table_stream(args.res_dir,
                                  args.sugar_hz,
                                  args.mn9)          # pass label unchanged
    print(f"Loaded {len(df)} trial observations into RAM (~{df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")

    if df.empty:
        raise RuntimeError("No MN9 rates found in the Parquet files.")
    print(f"Loaded {len(df)} trial-level observations.")
    run_anova(df, args)
    # run_grouped_anova(df, pattern="VM1")
    # run_per_cell_anova(df)

if __name__ == "__main__":
    main()