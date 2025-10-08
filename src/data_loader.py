import pandas as pd
import re

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

# ---------- Flexible upload support ----------

SYNONYMS = {
    "date": ["date","order_date","transaction_date","timestamp","datetime","order_dt"],
    "region": ["region","market","area","country","state","territory"],
    "product": ["product","sku","item","product_name","product_id","name"],
    "customer_id": ["customer_id","customer","user_id","account_id","cid","shopper_id","client_id"],
    "age": ["age","customer_age"],
    "gender": ["gender","sex"],
    "quantity": ["quantity","qty","units","unit_qty","count"],
    "sales": ["sales","revenue","amount","total","total_sales","net_sales","gmv","order_total"],
    "discount_pct": ["discount_pct","discount_percent","discount_rate","pct_discount","percent_off","discount","promo_pct","promo_percent"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, keys: list[str]) -> str | None:
    for k in keys:
        if k in df.columns:
            return k
    return None

def _coerce_numeric(s: pd.Series) -> pd.Series:
    # remove $, commas, %, spaces etc.; handle NaNs
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

def prepare_any_sales_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Try best-effort normalization to REQUIRED_COLUMNS.
    - rename synonyms to canonical names
    - coerce dates / numerics
    - compute sales if price*quantity is available
    - compute/normalize discount_pct (0-100 scale)
    """
    df = _normalize_columns(df_in)

    mapping: dict[str, str] = {}
    for canon, keys in SYNONYMS.items():
        col = _find_col(df, keys)
        if col:
            mapping[canon] = col

    # Compute sales if missing but price * quantity exists
    if "sales" not in mapping and "price" in df.columns:
        qcol = mapping.get("quantity")
        if qcol:
            df["__price__"] = _coerce_numeric(df["price"])
            df[qcol] = _coerce_numeric(df[qcol])
            df["sales"] = (df["__price__"] * df[qcol]).round(2)
            mapping["sales"] = "sales"

    # Discount normalization: if only "discount" exists, interpret as 0-1 or 0-100
    if "discount_pct" not in mapping:
        if "discount" in df.columns:
            s = _coerce_numeric(df["discount"])
            s = s.where(s <= 1, s / 100.0)        # values > 1 => assume already percent
            df["discount_pct"] = (s * 100).round(2)
            mapping["discount_pct"] = "discount_pct"
        else:
            # default to 0 if discount missing
            df["discount_pct"] = 0.0
            mapping["discount_pct"] = "discount_pct"

    # Coerce/parse types
    if "date" in mapping:
        df[mapping["date"]] = pd.to_datetime(df[mapping["date"]], errors="coerce")

    for k in ["quantity","sales","discount_pct","age"]:
        col = mapping.get(k)
        if col and col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Rename to canonical
    rename = {mapping[k]: k for k in mapping}
    df = df.rename(columns=rename)

    # Verify required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # Select canonical order
    return df[REQUIRED_COLUMNS]
