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
                   720575940634298975: "SECOND_DOWNSTREAM"}

def build_flyid2name(sugars: Iterable[int]) -> Dict[int, str]:
    mapping = {nid: f"sugar_{i+1}" for i, nid in enumerate(sugars)}
    mapping.update(CELL_LOOKUP_REV)
    return mapping

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Data" / "sweet.csv"

def get_sugar_ids():
    return pd.read_csv(DATA_PATH)["root_id"].astype(int).tolist()