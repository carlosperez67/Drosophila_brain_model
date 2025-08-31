from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import re
import math
import pandas as pd
import matplotlib.pyplot as plt

from constants import read_ids


class ExperimentPlotter:
    """
    Light-weight helper that builds line plots directly from the pre-computed
    `rates.csv` / `std_rates.csv` produced by AnalysisConfig.summarize_rates().
    """

    # ── construction via builder ──────────────────────────────────────────
    def __init__(
        self,
        res_dir: Path,
        group_name: str,
        group_hz: List[int],
        mon_csv: Path,
        mon_name: str,
        lines_per_subplot: int = 5,
    ):
        self.res_dir = res_dir
        self.neuron_ids = read_ids(mon_csv)
        self.group_name = group_name
        self.mon_name = mon_name
        self.group_hz = group_hz
        self.meta_csv = mon_csv
        self.lines_per_subplot = max(lines_per_subplot, 1)

        self.rate_csv = self.res_dir / "rates.csv"
        self.std_csv = self.res_dir / "std_rates.csv"

        if not (self.rate_csv.exists() and self.std_csv.exists()):
            raise FileNotFoundError(
                "Pre-computed rates not found – run summarize_rates() first."
            )

        # Lazily loaded frames
        self._df_mean: Optional[pd.DataFrame] = None
        self._df_std: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────
    @property
    def df_mean(self) -> pd.DataFrame:
        if self._df_mean is None:
            self._df_mean = pd.read_csv(self.rate_csv, index_col=0)
        return self._df_mean

    @property
    def df_std(self) -> pd.DataFrame:
        if self._df_std is None:
            self._df_std = pd.read_csv(self.std_csv, index_col=0)
        return self._df_std

    @staticmethod
    def _hz_sort(col_name: str) -> int:
        """Extract first ‹number›Hz in a column name."""
        m = re.search(r"(\d+)Hz", col_name)
        return int(m.group(1)) if m else math.inf

    def _columns_for_group(self) -> List[str]:
        pat = rf"^{re.escape(self.group_name)}_[0-9]+Hz$"
        cols = [c for c in self.df_mean.columns if re.match(pat, c)]
        return sorted(cols, key=self._hz_sort)

    def _labels_for_neurons(self) -> List[str]:
        if self.meta_csv is None:
            return [str(rid) for rid in self.neuron_ids]

        meta = pd.read_csv(self.meta_csv)
        if "root_id" not in meta.columns:
            raise ValueError("'root_id' column missing in meta CSV.")
        meta = meta.set_index("root_id")
        labels = []
        for rid in self.neuron_ids:
            try:
                labels.append(f"{meta.loc[rid, 'nickname']} ({rid})")
            except KeyError:
                labels.append(str(rid))
        return labels

    def _plot_dir(self) -> Path:
        p = self.res_dir / "plots"
        p.mkdir(exist_ok=True)
        return p

    # ──────────────────────────────────────────────────────────────────────
    # public plotting API
    # ──────────────────────────────────────────────────────────────────────
    def line_per_neuron(self) -> Path:
        cols = self._columns_for_group()
        if not cols:
            raise RuntimeError("No single-group columns detected.")

        labels = self._labels_for_neurons()

        # --- draw ---------------------------------------------------------
        for rid, lbl in zip(self.neuron_ids, labels):
            try:
                y = self.df_mean.loc[rid, cols]
                err = self.df_std.loc[rid, cols]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.errorbar(
                    self.group_hz, y, yerr=err, marker="o", linestyle="-", capsize=3
                )
                ax.set_title(f"{lbl} – response vs. {self.group_name} stimulation")
                ax.set_xlabel("Stimulation frequency (Hz)")
                ax.set_ylabel("Activation frequency (Hz)")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                out = self._plot_dir() / f"{lbl.replace(' ', '_')}_line.png"
                fig.savefig(out, dpi=300)
                print(f"Saved plot to {out}")
                plt.close(fig)
            except KeyError:
                Warning(f"{rid} not found")

        return self._plot_dir()

    def multiline_by_neuron(self) -> Path:
        """
        **All monitored neurons together** across frequencies.
        To avoid clutter, subplots are generated with up to
        `lines_per_subplot` neurons each.
        """
        cols = self._columns_for_group()
        if not cols:
            raise RuntimeError("No single-group columns detected.")

        n_neu = len(self.neuron_ids)
        chunk = self.lines_per_subplot
        n_fig = math.ceil(n_neu / chunk)
        labels = self._labels_for_neurons()

        for f_idx in range(n_fig):
            idx_start = f_idx * chunk
            idx_end = min(idx_start + chunk, n_neu)
            subset_ids = self.neuron_ids[idx_start:idx_end]
            subset_lbl = labels[idx_start:idx_end]

            fig, ax = plt.subplots(figsize=(8, 5))
            for rid, lbl in zip(subset_ids, subset_lbl):
                try:
                    y = self.df_mean.loc[rid, cols]
                    err = self.df_std.loc[rid, cols]
                    ax.errorbar(
                        self.group_hz,
                        y,
                        yerr=err,
                        marker="o",
                        linestyle="-",
                        capsize=3,
                        label=lbl,
                    )
                except KeyError:
                    Warning(f"{rid} not found")

            ax.set_title(
                f"{self.group_name} Activated: Monitored {self.mon_name} neuron responses "
                f"({idx_start + 1}-{idx_end} of {n_neu})"
            )
            ax.set_xlabel("Stimulation frequency (Hz)")
            ax.set_ylabel("Activated frequency (Hz)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="small")
            fig.tight_layout()

            out = self._plot_dir() / (
                f"{self.group_name}_chunk_{f_idx+1}.png"
            )
            print(f"Saved multiline plot to {out}")
            fig.savefig(out, dpi=300)
            plt.close(fig)

        return self._plot_dir()


# ─────────────────────────────────────────────────────────────────────────
#               B U I L D E R
# ─────────────────────────────────────────────────────────────────────────
class ExperimentPlotterBuilder:
    """
    Fluent builder for ExperimentPlotter.
    Only *res_dir* is mandatory; everything else can be filled in later.
    """

    def __init__(self, res_dir: Path):
        self._res_dir = res_dir
        self._neuron_ids: Optional[List[int]] = None
        self._group_name: Optional[str] = None
        self._group_hz: Optional[List[int]] = None
        self._mon_csv: Optional[Path] = None
        self._mon_name: Optional[str] = None
        self._lines_per_subplot: int = 5

    # def neuron_ids(self, ids: List[int]) -> "ExperimentPlotterBuilder":
    #     self._neuron_ids = ids
    #     return self

    def mon_csv(self, path: Path) -> "ExperimentPlotterBuilder":
        self._mon_csv = path
        return self

    def mon_name(self, name: str) -> "ExperimentPlotterBuilder":
        self._mon_name = name
        return self

    def group_name(self, name: str) -> "ExperimentPlotterBuilder":
        self._group_name = name
        return self

    def group_hz(self, hz_list: List[int]) -> "ExperimentPlotterBuilder":
        self._group_hz = hz_list
        return self

    def lines_per_subplot(self, n: int) -> "ExperimentPlotterBuilder":
        self._lines_per_subplot = n
        return self

    def build(self) -> ExperimentPlotter:
        # Minimal checks
        if self._mon_csv is None:
            raise ValueError("mon_csv() must be supplied.")
        if self._group_name is None:
            raise ValueError("group_name() must be supplied.")
        if self._group_hz is None:
            raise ValueError("group_hz() must be supplied.")

        return ExperimentPlotter(
            res_dir=self._res_dir,
            group_name=self._group_name,
            group_hz=self._group_hz,
            mon_csv=self._mon_csv,
            mon_name=self._mon_name,
            lines_per_subplot=self._lines_per_subplot,
        )