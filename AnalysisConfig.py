from pathlib import Path
from typing import Optional
import re

import pandas as pd
from brian2 import Hz
from matplotlib import pyplot as plt

import utils as utl
from model import default_params, run_exp

# ----------------------
# Global default config
#
# Change here to run script with different parameters
# ----------------------
DEFAULT_RES_DIR        = Path("/Volumes/T7/GordonLab/EGG_LAYING/oviEn")
DEFAULT_GROUP_1_HZ     = [20]
DEFAULT_GROUP_2_HZ     = None
DEFAULT_GROUP_1_CSV    = Path("./Data/oviEN.csv")
DEFAULT_GROUP_2_CSV    = None                     # Set to None if not using
DEFAULT_GROUP_1_NAME   = "group-1"
DEFAULT_GROUP_2_NAME   = "group-2"

CONFIG = {
    "path_comp": "./Completeness_783.csv",
    "path_con" : "./Connectivity_783.parquet",
    "path_res" : DEFAULT_RES_DIR,
    "n_proc"   : 10,
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

    @staticmethod
    def _read_ids(csv_path: Path) -> list[int]:
        df = pd.read_csv(csv_path)
        if "root_id" not in df.columns:
            raise ValueError(f"'root_id' column not found in {csv_path}")
        return df["root_id"].astype(int).tolist()

    def get_ids(self) -> tuple[list[int], Optional[list[int]]]:
        """
        Returns:
            A tuple containing:
              - group_1_neu_ids (list[int])
              - group_2_neu_ids (list[int] or None)
        """
        self.group_1_neu_ids = self._read_ids(self.group_1_csv)
        self.group_2_neu_ids = (
            self._read_ids(self.group_2_csv) if self.use_group_2 else None
        )
        return self.group_1_neu_ids, self.group_2_neu_ids


    def get_parquet_files(self) -> list[Path]:
        parquet_files = [p for p in self.res_dir.rglob(self.glob_pat) if p.suffix == ".parquet"]
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet matched pattern '{self.glob_pat}' under {self.res_dir}")
        return parquet_files

    def run_one_group(self, group: int, exp_names, file_paths):
        self.get_ids()          # Ensure neuron IDs are loaded

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
        #TODO: Add a good naming method
        # flyid2name = build_flyid2name(sugar_ids)
        df_spike = utl.load_exps([str(p) for p in parquet_files])

        df_rate, df_std = utl.get_rate(
            df_spike,
            t_run      = default_params["t_run"],
            n_run      = default_params["n_run"],
            #TODO: Add a good naming method
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


    def _load_summary_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return mean and std DataFrames indexed by `root_id`."""
        if not(self.df_rate and self.df_std):
            rate_csv = self.res_dir / "rates.csv"
            std_csv  = self.res_dir / "std_rates.csv"
            if not rate_csv.exists() or not std_csv.exists():
                raise FileNotFoundError(
                    "rates.csv / std_rates.csv not found – run `summarize_rates()` first"
                )
            self.df_rate = pd.read_csv(rate_csv, index_col=0)
            self.df_std  = pd.read_csv(std_csv,  index_col=0)
        return self.df_rate, self.df_std

    def _ensure_plot_dir(self) -> Path:
        plot_dir = self.res_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

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

    def group_2_csv(self, path: Path) -> "AnalysisConfigBuilder":
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
    experiment = AnalysisConfigBuilder(
        res_dir=DEFAULT_RES_DIR,
        group_1_csv=DEFAULT_GROUP_1_CSV,
        group_1_hz=DEFAULT_GROUP_1_HZ
    ).group_1_name(DEFAULT_GROUP_1_NAME).group_2_csv(DEFAULT_GROUP_2_CSV).group_2_name(DEFAULT_GROUP_2_NAME).build()

    experiment.neuron_activations()
    experiment.summarize_rates()

if __name__ == "__main__":
    main()



