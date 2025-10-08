from __future__ import annotations
import pandas as pd
from typing import Dict, Any

class StatsRetriever:
    """A lightweight, pandas-based retriever that returns key statistics
    based on a user question and context about the dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _sales_by_period(self, freq: str = "ME"):
        grp = self.df.set_index("date").groupby(pd.Grouper(freq=freq))["sales"].sum().reset_index()
        grp = grp.rename(columns={"sales":"total_sales"})
        return grp

    def _product_performance(self):
        return (
            self.df.groupby("product")[["sales","quantity"]]
            .sum().sort_values("sales", ascending=False)
            .reset_index()
        )

    def _regional_performance(self):
        return (
            self.df.groupby("region")[["sales","quantity"]]
            .sum().sort_values("sales", ascending=False)
            .reset_index()
        )

    def _customer_segments(self):
        bins = [0, 24, 34, 44, 54, 64, 150]
        labels = ["18-24","25-34","35-44","45-54","55-64","65+"]
        seg = self.df.copy()
        seg["age_band"] = pd.cut(seg["age"], bins=bins, labels=labels, right=True)
        out = seg.groupby(["age_band","gender"])[["sales","quantity"]].sum().reset_index()
        return out.sort_values("sales", ascending=False)

    def _basic_stats(self):
        return self.df[["sales","quantity","discount_pct","age"]].describe().T

    def retrieve(self, question: str) -> Dict[str, Any]:
        """Return a dict of frames/statistics that the LLM can read from."""
        return {
            "sales_by_month": self._sales_by_period("M").to_dict(orient="records"),
            "product_performance": self._product_performance().to_dict(orient="records"),
            "regional_performance": self._regional_performance().to_dict(orient="records"),
            "customer_segments": self._customer_segments().to_dict(orient="records"),
            "basic_stats": self._basic_stats().reset_index().rename(columns={"index":"metric"}).to_dict(orient="records"),
            "note": "All figures aggregated from the current dataframe."
        }
