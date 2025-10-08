from __future__ import annotations
import pandas as pd
from typing import Dict, Any

class StatsRetriever:
    """A lightweight, pandas-based retriever that returns key statistics
    based on a user question and context about the dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

def _sales_by_period(self, freq: str = "ME"):  # also silences the "M is deprecated" warning
    grp = (
        self.df.set_index("date")
        .groupby(pd.Grouper(freq=freq))["sales"]
        .sum()
        .reset_index()
    )
    grp = grp.rename(columns={"sales": "total_sales"})
    if "date" in grp.columns:
        grp["date"] = grp["date"].dt.strftime("%Y-%m-%d")  # <-- make JSON-safe strings
    return grp

def retrieve(self, question: str) -> Dict[str, Any]:
    return {
        "sales_by_month": self._sales_by_period("ME").to_dict(orient="records"),
        "product_performance": self._product_performance().to_dict(orient="records"),
        "regional_performance": self._regional_performance().to_dict(orient="records"),
        "customer_segments": self._customer_segments().to_dict(orient="records"),
        "basic_stats": (
            self._basic_stats()
            .reset_index()
            .rename(columns={"index": "metric"})
            .to_dict(orient="records")
        ),
        "note": "All figures aggregated from the current dataframe."
    }
