#!/usr/bin/env python3
"""
summarize_rates.py

Create CSV summaries from a directory of *.parquet spike-train files generated
by run_frequency_sweeps.py.  The module can

* be invoked directly from the command line; or
* be imported and its `summarize_rates()` function called from another script.

Outputs
-------
results_dir/
    ├── all_rates.csv
    ├── all_rates_std.csv
    ├── filtered_rates.csv
    ├── filtered_rates_std.csv
    ├── rates_<sugar>Hz_<cell>.csv
    ├── filtered_rates_<sugar>Hz_<cell>.csv
    └── … (one pair for every detected {sugar Hz, cell-type})
"""
from __future__ import annotations

import argparse, re
import warnings
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple, List
import pandas as pd

# --------------------------------------------------------------------------
# project-level helpers
# --------------------------------------------------------------------------
from scripts.constants import build_flyid2name, get_sugar_ids
from model import default_params
import utils as utl

# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------
DEFAULT_SUGAR_F = 45
DEFAULT_ORN_F   = list(range(20, 201, 10))
CONFIG              = {"path_res": Path("./sweet_results/GRN")}

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize rates per cell type")
    p.add_argument(
        "--res-dir",
        type=Path,
        default=CONFIG["path_res"],
        help="Directory that holds *.parquet outputs (default: ./sweet_results/GRN)",
    )
    p.add_argument(
        "--sugar-hz",
        type=float,
        default=45,
        help="Sugar frequency to keep (default: 45)",
    )
    p.add_argument(
        "--orn-hz",
        type=float,
        nargs="+",
        default=DEFAULT_ORN_F,
        help="ORN frequencies to keep (default: 20 … 200 by 10)",
    )
    p.add_argument(
        "--orn-types",
        type=str,
        nargs="+",
        default=None,
        metavar="CELL_TYPE",
        help="Restrict to these ORN cell types (default: all detected)",
    )
    p.add_argument(                     # convenience single-value alias
        "--cell-type",
        type=str,
        help="Shortcut for a single ORN cell type (overrides --orn-types)",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Return a filesystem-safe, lower-snakecase slug."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()

def _csv_name(kind: str,
              sugar: int | str = "ALL",
              cell : str       = "ALL",
              filtered: bool   = False) -> str:
    """
    Build a systematic filename.

    kind      – "rates" | "rate_std" (or any descriptor you like)
    sugar     – integer Hz or "ALL"
    cell      – cell type (already slugged) or "control" / "ALL"
    filtered  – True → add 'filtered' suffix
    """
    suffix = "_filtered" if filtered else ""
    return f"{kind}_{sugar}Hz_{cell}{suffix}.csv"

def _extract_tags(col: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """
    Return (sugar_Hz, cell_type, orn_Hz) parsed from experiment/column name.
    control columns have cell_type = orn_Hz = None.
    """
    m = re.search(r"sugar(?:_only_)?(\d+)Hz", col)
    if not m:
        return None, None, None
    sugar = int(m.group(1))

    m_exp = re.search(r"plus_([^_]+)_(\d+)Hz", col)
    if m_exp:
        return sugar, m_exp.group(1), int(m_exp.group(2))
    return sugar, None, None

# ------------------------------------------------------------------
#  helper: keep only requested parquets
# ------------------------------------------------------------------
def _discover_parquets(root: Path) -> List[Path]:
    controls     = root / "controls"
    experiments  = root / "experiments"
    missing_dirs = [d for d in (controls, experiments) if not d.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Expected sub-directories not found: "
            + ", ".join(str(d) for d in missing_dirs)
        )

    files: list[Path] = []
    for d in (controls, experiments):
        files.extend(d.rglob("*.parquet"))
    return files

def filter_parquets(path: Path,
                    sugar_hz: float,
                    orn_hz: Optional[set[int]],
                    type_sel: Optional[set[str]]) -> Tuple[list[Path], list[Path]]:
    """
    Return the subset of `files` that matches the requested sugar Hz,
    ORN cell-types, and ORN Hz values.
    """
    control_dir = path / "controls"
    exp_dir     = path / "experiments"
    sugar_hz      = int(sugar_hz)

    control_pattern = re.compile(rf"sugar_only_{sugar_hz}Hz\.parquet$")
    exp_pattern = re.compile(rf"sugar{sugar_hz}Hz_plus_(.+)_(\d+)Hz\.parquet")

    controls: List[Path]    = []
    experiments: List[Path] = []

    for file in control_dir.glob(f"sugar_only_{sugar_hz}Hz.parquet"):
        if control_pattern.match(file.name):
            controls.append(file)
        else:
            warnings.warn(f"Skipping unrecognized control file: {file.name}")

    for file in exp_dir.glob(f"sugar{sugar_hz}Hz_plus_*_[0-9]*Hz.parquet"):
        m = exp_pattern.match(file.name)
        if not m:
            warnings.warn(f"Skipping unrecognized experiment file: {file.name}")
            continue

        cell_type = m.group(1)
        freq      = int(m.group(2))

        if (orn_hz is None or freq in orn_hz) and \
           (type_sel is None or cell_type in type_sel):
            experiments.append(file)

    return controls, experiments

def sort_columns_by_orn_freq(df: pd.DataFrame) -> pd.DataFrame:
    # Separate "name" column
    name_col = ["name"] if "name" in df.columns else []

    # Define a sorting key function that extracts the ORN frequency
    def orn_hz_key(col: str) -> int:
        match = re.search(r"_ORN_[A-Z]+_(\d+)Hz", col)
        return int(match.group(1)) if match else float('inf')

    # Get all columns except "name", then sort them by ORN Hz
    other_cols = [col for col in df.columns if col != "name"]
    sorted_cols = sorted(other_cols, key=orn_hz_key)

    # Reassemble ordered columns
    return df[name_col + sorted_cols]



# ------------------------------------------------------------------
#  public API
# ------------------------------------------------------------------
def summarize_rates(
    res_dir  : Path,
    orn_types: Optional[Iterable[str]] = None,
    sugar_hz : float          = DEFAULT_SUGAR_F,
    orn_hz   : Iterable[float] = DEFAULT_ORN_F,
) -> None:
    """
    Build CSV summaries for the given sweep directory.

    Parameters
    ----------
    res_dir   : Path
        Directory that contains controls/ and experiments/ subdirs
    orn_types : iterable[str] | None
        Restrict to these ORN cell types.  None means “all”.
    sugar_hz  : float
        Sugar frequency to keep.
    orn_hz    : iterable[float]
        ORN frequencies to keep.
    """
    # prepare filters
    orn_sel  = {int(o) for o in orn_hz}
    type_sel = {t.strip() for t in orn_types} if orn_types else None

    # discover parquet files (filenames only)
    print("Discovering Parquet files …")
    # all_files = _discover_parquets(res_dir)   # no longer needed for filtering
    controls, experiments = filter_parquets(
        res_dir,
        sugar_hz,
        orn_sel,
        type_sel
    )
    parquet_files = controls + experiments

    if not parquet_files:
        raise FileNotFoundError(
            "No parquet files matching the requested "
            f"sugar={int(sugar_hz)}Hz, orn_types={type_sel or 'ALL'}, "
            f"orn_hz={sorted(orn_sel)} under {res_dir}"
        )

    print(f"  → {len(parquet_files)} file(s) selected")

    # ----------------------------------------------------------------
    # build spike-rate matrices
    # ----------------------------------------------------------------
    sugar_ids  = get_sugar_ids()
    flyid2name = build_flyid2name(sugar_ids)

    print("Processing rates …")
    df_spike = utl.load_exps([str(p) for p in parquet_files])
    df_rate, df_std = utl.get_rate(
        df_spike,
        t_run        = default_params["t_run"],
        n_run        = default_params["n_run"],
        flyid2name   = flyid2name,
    )

    # build tags
    sugar_tag = f"{int(sugar_hz)}Hz"
    if type_sel:
        cell_tag = "-".join(_slug(c) for c in sorted(type_sel))
    else:
        cell_tag = "ALLcells"

    # filter out any blank names and write CSVs
    df_rate = sort_columns_by_orn_freq(df_rate)
    df_std = sort_columns_by_orn_freq(df_std)

    df_rate_filtered    = df_rate[df_rate["name"] != ""].fillna(0)
    df_std_filtered     = df_std[df_std["name"] != ""].fillna(0)

    df_rate_filtered.to_csv(
        res_dir / f"filtered_rates_{sugar_tag}_{cell_tag}.csv",
        index=False,
    )
    df_std_filtered.to_csv(
        res_dir / f"filtered_rates_std_{sugar_tag}_{cell_tag}.csv",
        index=False,
    )

    df_rate.to_csv(
        res_dir / f"all_rates_{sugar_tag}_{cell_tag}.csv",
        index=False,
    )
    df_std.to_csv(
        res_dir / f"all_rates_std_{sugar_tag}_{cell_tag}.csv",
        index=False,
    )

    print(f"CSV summaries written to {res_dir.resolve()}")


# --------------------------------------------------------------------------
# entry-point
# --------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.cell_type:
        args.orn_types = [args.cell_type]

    print("Summarizing rates...")
    summarize_rates(
        res_dir   = args.res_dir,
        orn_types = args.orn_types,
        sugar_hz  = args.sugar_hz,
        orn_hz    = args.orn_hz,
    )

if __name__ == "__main__":
    main()