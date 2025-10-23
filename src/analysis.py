from __future__ import annotations
import pandas as pd

def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.count().values,
        "Null Count": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

def numeric_corr(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True)
