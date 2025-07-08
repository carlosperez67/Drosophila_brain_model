import argparse
import json
import os
from pathlib import Path
from typing import Optional
import re

import pandas as pd
from brian2 import Hz
from matplotlib import pyplot as plt

import utils as utl
from ExperimentPlotter import ExperimentPlotterBuilder
from constants import read_ids
from model import default_params, run_exp

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

parser = argparse.ArgumentParser(description="Load run config from a JSON file")
parser.add_argument("config", type=str, help="Path to run config")
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)
    DEFAULT_RES_DIR = Path(config['DEFAULT_RES_DIR'])
    DEFAULT_GROUP_1_HZ = config['DEFAULT_GROUP_1_HZ']
    DEFAULT_GROUP_2_HZ = config.get('DEFAULT_GROUP_2_HZ')
    DEFAULT_GROUP_1_CSV = Path(config['DEFAULT_GROUP_1_CSV'])
    DEFAULT_GROUP_2_CSV = Path(config['DEFAULT_GROUP_2_CSV']) if config.get('DEFAULT_GROUP_2_CSV') else None
    DEFAULT_GROUP_1_NAME = config.get('DEFAULT_GROUP_1_NAME') or "group1"
    DEFAULT_GROUP_2_NAME = config.get('DEFAULT_GROUP_2_NAME') or "group2"
    DEFAULT_MONITORED_NEURONS_CSV = Path(config['DEFAULT_MONITORED_NEURONS_CSV'])

os.makedirs(DEFAULT_RES_DIR, exist_ok=True)
with open(DEFAULT_RES_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=4)

CONFIG = {
    "path_comp": "./Completeness_783.csv",
    "path_con": "./Connectivity_783.parquet",
    "path_res": DEFAULT_RES_DIR,
    "n_proc": 10,
}

class AnalysisConfig:
    def __init__(
            self,
            res_dir: Path,
            group_1_csv: Path,
            group_1_hz: list[int],
            group_2_csv: Path | None = None,
            group_2_hz: Optional[list[int]] = None,
            group_1_name: Optional[str] = None,
            group_2_name: Optional[str] = None,
            glob_pat: Optional[str] = "*",
    ):
        if group_2_csv is not None and group_2_hz is None:
            raise ValueError("group_2_hz must be provided when group_2_csv is given.")

        self.res_dir = res_dir
        self.group_1_csv = group_1_csv
        self.group_1_hz = group_1_hz
        self.group_1_name = group_1_name
        self.group_2_csv = group_2_csv
        self.group_2_hz = group_2_hz
        self.group_2_name = group_2_name
        self.glob_pat = glob_pat
        self.use_group_2 = group_2_csv is not None

        self.group_1_neu_ids = None
        self.group_2_neu_ids = None
        self.df_rate = None
        self.df_std = None




    def get_ids(self) -> tuple[list[int], Optional[list[int]]]:
        """
        Returns:
            A tuple containing:
              - group_1_neu_ids (list[int])
              - group_2_neu_ids (list[int] or None)
        """
        self.group_1_neu_ids = read_ids(self.group_1_csv)
        self.group_2_neu_ids = (
            read_ids(self.group_2_csv) if self.use_group_2 else None
        )
        return self.group_1_neu_ids, self.group_2_neu_ids

    def get_parquet_files(self) -> list[Path]:
        parquet_files = [p for p in self.res_dir.rglob(self.glob_pat) if p.suffix == ".parquet"]
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet matched pattern '{self.glob_pat}' under {self.res_dir}")
        return parquet_files

    def run_one_group(self, group: int, exp_names, file_paths):
        self.get_ids()  # Ensure neuron IDs are loaded

        match group:
            case 1:
                group_hz = self.group_1_hz
                group_neu_ids = self.group_1_neu_ids
                group_name = self.group_1_name or "group1"
            case 2:
                if not self.use_group_2:
                    raise ValueError("Group 2 is not configured (no CSV provided).")
                group_hz = self.group_2_hz
                group_neu_ids = self.group_2_neu_ids
                group_name = self.group_2_name or "group2"
            case _:
                raise ValueError(f"Unsupported group {group}. Only group 1 and 2 are valid.")

        if group_hz is None or group_neu_ids is None:
            raise ValueError(f"Missing Hz or neuron ID list for group {group}.")

        for rate in group_hz:
            params = default_params.copy()
            params["r_poi"] = rate * Hz
            exp_name = f"{group_name}_{int(rate)}Hz"
            run_exp(
                exp_name=exp_name,
                neu_exc=group_neu_ids,
                params=params,
                **CONFIG,
                force_overwrite=False,
            )
            exp_names.append(exp_name)
            file_paths.append(self.res_dir / f"{exp_name}.parquet")
        return exp_names, file_paths

    def run_two_groups(self, group_a, group_b, exp_names, file_paths):
        self.get_ids()

        def get_group_config(group: int):
            if group == 1:
                if self.group_1_neu_ids is None or self.group_1_hz is None:
                    raise ValueError("Group 1 neuron IDs or Hz not set.")
                return self.group_1_neu_ids, self.group_1_hz, self.group_1_name or "group1"
            elif group == 2:
                if not self.use_group_2 or self.group_2_neu_ids is None or self.group_2_hz is None:
                    raise ValueError("Group 2 is not properly configured.")
                return self.group_2_neu_ids, self.group_2_hz, self.group_2_name or "group2"
            else:
                raise ValueError(f"Unsupported group number: {group}")

        # Resolve configuration for both groups
        neu_a, hz_a, name_a = get_group_config(group_a)
        neu_b, hz_b, name_b = get_group_config(group_b)

        for rate_a in hz_a:
            for rate_b in hz_b:
                params = default_params.copy()
                params["r_poi"] = rate_a * Hz
                params["r_poi2"] = rate_b * Hz

                exp_name = f"{name_a}_{int(rate_a)}Hz_AND_{name_b}_{int(rate_b)}Hz"
                run_exp(
                    exp_name=exp_name,
                    neu_exc=neu_a,
                    neu_exc2=neu_b,
                    params=params,
                    **CONFIG,
                    force_overwrite=False,
                )
                exp_names.append(exp_name)
                file_paths.append(self.res_dir / f"{exp_name}.parquet")

        return exp_names, file_paths

    def neuron_activations(self):
        self.res_dir.mkdir(parents=True, exist_ok=True)
        exp_names: list[str] = []
        file_paths: list[Path] = []

        self.run_one_group(1, exp_names, file_paths)
        if self.use_group_2:
            self.run_two_groups(1, 2, exp_names, file_paths)

        print(f"Finished running {len(exp_names)} experiments")

    @staticmethod
    def _sort_by_hz_key(col_name: str) -> tuple[int, int]:
        """Extract up to two numeric Hz values from a string for sorting."""
        matches = re.findall(r"(\d+)Hz", col_name)
        hz1 = int(matches[0]) if len(matches) > 0 else float("inf")
        hz2 = int(matches[1]) if len(matches) > 1 else float("inf")
        return hz1, hz2

    def summarize_rates(self):
        parquet_files = self.get_parquet_files()
        # TODO: Add a good naming method
        # flyid2name = build_flyid2name(sugar_ids)
        df_spike = utl.load_exps([str(p) for p in parquet_files])

        df_rate, df_std = utl.get_rate(
            df_spike,
            t_run=default_params["t_run"],
            n_run=default_params["n_run"],
            # TODO: Add a good naming method
            # flyid2name = flyid2name,
        )
        # df_rate.sort_values(df_spike)
        sorted_columns = sorted(df_rate.columns, key=self._sort_by_hz_key)

        df_rate = df_rate[sorted_columns]
        df_std = df_std[sorted_columns]

        df_rate.to_csv(self.res_dir / "rates.csv")
        df_std.to_csv(self.res_dir / "std_rates.csv")

        self.df_rate = df_rate
        self.df_std = df_std
        print(f"CSV summaries written to {self.res_dir.resolve()}")

    @staticmethod
    def _filter_columns(df: pd.DataFrame, pattern: str) -> list[str]:
        """Return columns whose names match *pattern* (regex)."""
        return [c for c in df.columns if re.search(pattern, c)]

    def _ensure_plot_dir(self) -> Path:
        plot_dir = self.res_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    def _load_summary_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return mean and std DataFrames indexed by `root_id`."""
        if not (self.df_rate and self.df_std):
            rate_csv = self.res_dir / "rates.csv"
            std_csv = self.res_dir / "std_rates.csv"
            if not rate_csv.exists() or not std_csv.exists():
                raise FileNotFoundError(
                    "rates.csv / std_rates.csv not found – run `summarize_rates()` first"
                )
            self.df_rate = pd.read_csv(rate_csv, index_col=0)
            self.df_std = pd.read_csv(std_csv, index_col=0)
        return self.df_rate, self.df_std

    def _ensure_plot_dir(self) -> Path:
        plot_dir = self.res_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    @staticmethod
    def _neuron_labels(
            self,
            neu_ids: list[int],
            meta_csv: Optional[Path] = None,
    ) -> list[str]:
        """
        Return display labels for the chosen neurons.
        If *meta_csv* is supplied, look up a 'cell_type' column; otherwise use root_id.
        """
        if meta_csv is None:
            return [str(rid) for rid in neu_ids]

        meta = pd.read_csv(meta_csv)
        if "root_id" not in meta.columns:
            raise ValueError("'root_id' column missing in meta CSV.")
        meta = meta.set_index("root_id")
        labels = []
        for rid in neu_ids:
            try:
                ctype = meta.loc[rid, "cell_type"]
                labels.append(f"{ctype} ({rid})")
            except KeyError:
                labels.append(str(rid))
        return labels

    def _subplot_per_neuron(
            self,
            df_mean: pd.DataFrame,
            df_std: pd.DataFrame,
            neu_ids: list[int],
            cols: list[str],
            title: str,
            meta_csv: Optional[Path],
            file_out: Path,
    ) -> None:
        """Create one bar-chart subplot per neuron (± s.d.)."""
        labels = self._neuron_labels(neu_ids, meta_csv)
        n = len(neu_ids)
        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(14, 4 * n),
            sharex=True,
        )
        if n == 1:  # axes becomes a single Axes when n==1
            axes = [axes]

        for ax, rid, lbl in zip(axes, neu_ids, labels):
            means = df_mean.loc[rid, cols]
            errs = df_std.loc[rid, cols]
            means.plot.bar(
                yerr=errs,
                ax=ax,
                capsize=3,
                legend=False,
                rot=0,
            )
            ax.set_ylabel("Spikes / s")
            ax.set_title(lbl, loc="left", pad=4)

        axes[-1].set_xticklabels(cols, rotation=45, ha="right")
        fig.suptitle(title, fontsize=16, y=1.02)
        fig.tight_layout()
        file_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_out, dpi=300)
        plt.close(fig)

    def plot_single_group_per_neuron(
            self,
            group: int = 1,
            meta_csv: Optional[Path] = None,
    ) -> Path:
        """
        Plot *each* monitored neuron’s response for single-group activations.

        Parameters
        ----------
        group     : 1 or 2
        meta_csv  : optional CSV with columns ['root_id','cell_type', …]
        """
        if group not in {1, 2}:
            raise ValueError("group must be 1 or 2")

        # resolve IDs, names, columns
        neu_ids, g_name = (
            (self.group_1_neu_ids, self.group_1_name or "group1")
            if group == 1 else
            (self.group_2_neu_ids, self.group_2_name or "group2")
        )
        if neu_ids is None:
            raise RuntimeError("Call `get_ids()` before plotting.")

        df_rate, df_std = self._load_summary_frames()
        cols = self._filter_columns(df_rate, rf"^{g_name}_[0-9]+Hz$")
        if not cols:
            raise RuntimeError(f"No single-group columns found for {g_name}.")

        out_png = self._ensure_plot_dir() / f"{g_name}_per_neuron.png"
        self._subplot_per_neuron(
            df_mean=df_rate,
            df_std=df_std,
            neu_ids=neu_ids,
            cols=cols,
            title=f"{g_name}: response per neuron (no aggregation)",
            meta_csv=meta_csv,
            file_out=out_png,
        )
        return out_png

    def plot_pairwise_per_neuron(
            self,
            meta_csv: Optional[Path] = None,
    ) -> Path:
        """
        Plot each monitored neuron’s response for pairwise (group1 AND group2) activations.
        """
        if not self.use_group_2:
            raise RuntimeError("Pairwise plot requested but group-2 is not configured.")

        df_rate, df_std = self._load_summary_frames()
        cols = self._filter_columns(df_rate, r"_AND_")
        if not cols:
            raise RuntimeError("No pairwise columns found (_AND_).")

        all_neu_ids = list({*(self.group_1_neu_ids or []), *(self.group_2_neu_ids or [])})
        out_png = self._ensure_plot_dir() / "pairwise_per_neuron.png"
        self._subplot_per_neuron(
            df_mean=df_rate,
            df_std=df_std,
            neu_ids=all_neu_ids,
            cols=cols,
            title="Pairwise activation: response per neuron (no aggregation)",
            meta_csv=meta_csv,
            file_out=out_png,
        )
        return out_png


class AnalysisConfigBuilder:
    def __init__(self, res_dir: Path, group_1_csv: Path, group_1_hz: list[int]):
        self._res_dir = res_dir
        self._group_1_csv = group_1_csv
        self._group_1_hz = group_1_hz

        self._group_1_name: Optional[str] = None
        self._group_2_csv: Optional[Path] = None
        self._group_2_name: Optional[str] = None
        self._group_2_hz: Optional[list[int]] = None
        self._glob_pat: Optional[str] = "*"

    def group_1_name(self, name: str) -> "AnalysisConfigBuilder":
        self._group_1_name = name
        return self

    def group_2_csv(self, path: Path | None = None) -> "AnalysisConfigBuilder":
        self._group_2_csv = path
        return self

    def group_2_name(self, name: str) -> "AnalysisConfigBuilder":
        self._group_2_name = name
        return self

    def group_2_hz(self, hz: list[int]) -> "AnalysisConfigBuilder":
        self._group_2_hz = hz
        return self

    def glob_pat(self, pattern: str) -> "AnalysisConfigBuilder":
        self._glob_pat = pattern
        return self

    def build(self) -> AnalysisConfig:
        if self._group_2_csv is not None and self._group_2_hz is None:
            raise ValueError("group_2_hz must be set if group_2_csv is provided.")

        return AnalysisConfig(
            res_dir=self._res_dir,
            group_1_csv=self._group_1_csv,
            group_1_hz=self._group_1_hz,
            group_1_name=self._group_1_name,
            group_2_csv=self._group_2_csv,
            group_2_name=self._group_2_name,
            group_2_hz=self._group_2_hz,
            glob_pat=self._glob_pat
        )

def main() -> None:


    experiment = (AnalysisConfigBuilder(
        res_dir=DEFAULT_RES_DIR,
        group_1_csv=DEFAULT_GROUP_1_CSV,
        group_1_hz=DEFAULT_GROUP_1_HZ
    )
                  .group_1_name(DEFAULT_GROUP_1_NAME)
                  .group_2_csv(DEFAULT_GROUP_2_CSV)
                .group_2_hz(DEFAULT_GROUP_2_HZ)
                  .group_2_name(DEFAULT_GROUP_2_NAME).build())

    experiment.neuron_activations()
    experiment.summarize_rates()

    plotter = (
        ExperimentPlotterBuilder(res_dir=Path(DEFAULT_RES_DIR))
        .group_name(DEFAULT_GROUP_1_NAME)
        .group_hz(DEFAULT_GROUP_1_HZ)
        .mon_csv(DEFAULT_MONITORED_NEURONS_CSV)
        .lines_per_subplot(5)
    ).build()

    plotter.line_per_neuron()
    plotter.multiline_by_neuron()


if __name__ == "__main__":
    main()
