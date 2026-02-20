from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = (ROOT / "data").resolve()


DATASETS = {
    "DS02": "DS02.csv",
    "DS04": "DS04.csv",
    "DS05": "DS05.csv",
    "DS07": "DS07.csv",
    "DS08": "DS08.csv",
    "DS10": "DS10.csv",
}
