# src/viz.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sales_trend_plot(df: pd.DataFrame):
    """Line chart of total sales by month (month-end)."""
    monthly = (
        df.set_index("date")
          .groupby(pd.Grouper(freq="ME"))["sales"]
          .sum()
    )
    fig, ax = plt.subplots()
    monthly.plot(ax=ax)
    ax.set_title("Sales Trend (Month End)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    fig.tight_layout()
    return fig


def product_performance_plot(df: pd.DataFrame, top_n: int = 5):
    """Bar chart: top-N products by total sales."""
    prod = (
        df.groupby("product")["sales"]
          .sum()
          .sort_values(ascending=False)
          .head(top_n)
    )
    fig, ax = plt.subplots()
    prod.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {top_n} Products by Sales")
    ax.set_xlabel("Product")
    ax.set_ylabel("Total Sales")
    fig.tight_layout()
    return fig


def regional_performance_plot(df: pd.DataFrame):
    """Bar chart: sales by region."""
    reg = (
        df.groupby("region")["sales"]
          .sum()
          .sort_values(ascending=False)
    )
    fig, ax = plt.subplots()
    reg.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales")
    fig.tight_layout()
    return fig


def cohort_heatmap(
    df: pd.DataFrame,
    value: str = "retention",      # "retention" (% of cohort), or "customers"
    last_n_cohorts: int = 12,
    max_months: int = 12,
    annotate: bool = False
):
    """
    Readable cohort heatmap: rows = cohort month, cols = months since first purchase.

    Requires columns: ['customer_id', 'date'].
    """
    if "customer_id" not in df.columns or "date" not in df.columns:
        raise ValueError("cohort_heatmap requires 'customer_id' and 'date' columns")

    tmp = df.copy()
    tmp["order_month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    first = tmp.groupby("customer_id")["order_month"].min().rename("cohort")
    tmp = tmp.join(first, on="customer_id")
    tmp["cohort_index"] = (
        (tmp["order_month"].dt.year - tmp["cohort"].dt.year) * 12
        + (tmp["order_month"].dt.month - tmp["cohort"].dt.month)
    )

    cust_counts = (
        tmp.groupby(["cohort", "cohort_index"])["customer_id"]
           .nunique()
           .reset_index(name="customers")
    )
    pivot = cust_counts.pivot(index="cohort", columns="cohort_index", values="customers").sort_index()

    # keep last N cohorts & first M months
    pivot = pivot.tail(last_n_cohorts)
    pivot = pivot.reindex(columns=sorted([c for c in pivot.columns if isinstance(c, (int, np.integer)) and c <= max_months - 1]))

    if pivot.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data to plot cohorts", ha="center", va="center")
        ax.axis("off")
        return fig

    if value == "retention":
        denom = pivot.iloc[:, 0].replace(0, np.nan)
        mat = (pivot.divide(denom, axis=0) * 100.0).round(1)
        title = "Customer Retention by Cohort (%)"
        fmt = lambda v: f"{v:.0f}%" if np.isfinite(v) else ""
        cbar_label = "% of cohort"
    else:
        mat = pivot.fillna(0.0)
        title = "Unique Customers by Cohort"
        fmt = lambda v: f"{int(v):d}" if v >= 1 else ""
        cbar_label = "Customers"

    r, c = mat.shape
    fig_w = max(6, c * 0.6)
    fig_h = max(4, r * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)

    ax.set_xticks(range(c))
    ax.set_xticklabels([str(i) for i in mat.columns])
    ax.set_yticks(range(r))
    ax.set_yticklabels([idx.strftime("%Y-%m") for idx in mat.index])
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort (first purchase month)")
    ax.set_title(title)

    if annotate or (r <= 12 and c <= 12):
        for i in range(r):
            for j in range(c):
                v = mat.iat[i, j]
                if np.isfinite(v) and (v > 0):
                    ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig
