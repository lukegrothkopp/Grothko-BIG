import pandas as pd

REQUIRED_COLUMNS = [
    "date","region","product","customer_id","age","gender","quantity","sales","discount_pct"
]

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
