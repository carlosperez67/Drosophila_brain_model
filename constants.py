from typing import Iterable, Dict
from pathlib import Path
import pandas as pd

CELL_LOOKUP = {
    "MN9_RIGHT": 720575940660219265,
    "MN9_LEFT": 720575940618238523,
    "ORN_D_DOWNSTREAM": 720575940603985952,
    "SECOND_DOWNSTREAM": 720575940634298975,
}

CELL_LOOKUP_REV = {720575940660219265: "MN9_RIGHT",
                     720575940618238523: "MN9_LEFT",
                     720575940603985952: "ORN_D_DOWNSTREAM",
                   720575940634298975: "SECOND_DOWNSTREAM",
                   720575940606428850: "Inhibitory DM1 (1)",
                   720575940622500172: "Inhibitory DM1 (2)",
                   720575940620170119: "Inhibitory DM1 (3)",
                   720575940622147873: "Inhibitory DM1 (4)",
                   720575940612311987: "Excitation DM1 (5)",
                   720575940637612974: "Inhibitory DM1 (6)",
                   720575940620625880: "oviDN (1)",
                   720575940621257340: "oviDN (2)",
                   720575940632512156: "oviDN (3)",
                   720575940603980256: "oviDN (4)",
                   720575940646173748: "oviDN (5)",
                   720575940641518733: "oviDN (6)",
                   720575940642312136: "oviDN (7)",
                   720575940627921182: "oviDN (8)",
                   720575940612153041: "oviDN (9)",
                   720575940610760306:"oviDN (10)",}

def build_flyid2name(sugars: Iterable[int]) -> Dict[int, str]:
    mapping = {nid: f"sugar_{i+1}" for i, nid in enumerate(sugars)}
    mapping.update(CELL_LOOKUP_REV)
    return mapping

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Data" / "sweet.csv"

def get_sugar_ids():
    return pd.read_csv(DATA_PATH)["root_id"].astype(int).tolist()

def read_ids(csv_path: Path) -> list[int]:
    df = pd.read_csv(csv_path)
    if "root_id" not in df.columns:
        raise ValueError(f"'root_id' column not found in {csv_path}")
    return df["root_id"].astype(int).tolist()