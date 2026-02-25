import pandas as pd
from pathlib import Path

def _strip_embedded_header(boston_dt: pd.DataFrame) -> pd.DataFrame:
    if boston_dt.shape[0] < 2:
        return boston_dt
    col_labels = [str(col) for col in boston_dt.columns]
    first_row = boston_dt.iloc[0].astype(str).tolist()
    if first_row == col_labels:
        return boston_dt.iloc[1:].reset_index(drop=True)
    return boston_dt

def load_boston_csv(path):
    for args in [
        {"sep": ","},
        {"sep": r"\s+", "header": None},
        {"sep": None, "engine": "python"},
    ]:
        try:
            boston_dt = pd.read_csv(path, **args)
        except Exception:
            continue
        if boston_dt.shape[1] > 1:
            boston_dt = _strip_embedded_header(boston_dt)
            if boston_dt.shape[1] > 1:
                return boston_dt
    raise FileNotFoundError(f"Could not parse file: {path}")

project_root = Path(__file__).resolve().parent.parent
csv_candidates = [
    Path.cwd() / "data" / "housing.csv",
    project_root / "data" / "housing.csv",
]

for csv_path in csv_candidates:
    if csv_path.exists():
        boston_dt = load_boston_csv(str(csv_path))

        break
else:
    raise FileNotFoundError("housing.csv not found!")

boston_cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS",
               "RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

if boston_dt.shape[1] == 1:
    boston_dt = boston_dt.iloc[:, 0].str.split(expand=True)

if boston_dt.shape[1] == len(boston_cols):
    boston_dt.columns = boston_cols

boston_dt = boston_dt.apply(pd.to_numeric, errors='coerce')
