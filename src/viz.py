import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cohort_heatmap(
    df: pd.DataFrame,
    value: str = "retention",      # "retention" (% of cohort), or "customers"
    last_n_cohorts: int = 12,
    max_months: int = 12,
    annotate: bool = False
):
    """Readable cohort heatmap: rows=cohort month, cols=months since first purchase."""
    if "customer_id" not in df.columns or "date" not in df.columns:
        raise ValueError("cohort_heatmap requires 'customer_id' and 'date' columns")

    tmp = df.copy()
    tmp["order_month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    first = tmp.groupby("customer_id")["order_month"].min().rename("cohort")
    tmp = tmp.join(first, on="customer_id")
    # months since first purchase
    tmp["cohort_index"] = ((tmp["order_month"].dt.year - tmp["cohort"].dt.year) * 12 +
                           (tmp["order_month"].dt.month - tmp["cohort"].dt.month))

    # aggregate
    cust_counts = (
        tmp.groupby(["cohort","cohort_index"])["customer_id"]
        .nunique()
        .reset_index(name="customers")
    )
    pivot = cust_counts.pivot(index="cohort", columns="cohort_index", values="customers").sort_index()

    # keep last N cohorts & first M months
    pivot = pivot.tail(last_n_cohorts)
    pivot = pivot.reindex(columns=sorted([c for c in pivot.columns if (isinstance(c, (int,np.integer)) and c <= max_months-1)]))

    # compute retention %
    if value == "retention":
        denom = pivot.iloc[:, 0].replace(0, np.nan)
        mat = (pivot.divide(denom, axis=0) * 100.0).round(1)
        title = "Customer Retention by Cohort (%)"
        fmt = lambda v: f"{v:.0f}%" if np.isfinite(v) else ""
    else:
        mat = pivot.fillna(0.0)
        title = "Unique Customers by Cohort"
        fmt = lambda v: f"{int(v):d}" if v >= 1 else ""

    # plot
    r, c = mat.shape if mat.size else (0, 0)
    if r == 0 or c == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data to plot cohorts", ha="center", va="center")
        ax.axis("off")
        return fig

    fig_w = max(6, c * 0.6)   # width scales with months
    fig_h = max(4, r * 0.35)  # height scales with cohorts
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% of cohort" if value == "retention" else "Customers")

    # ticks & labels
    ax.set_xticks(range(c))
    ax.set_xticklabels([str(i) for i in mat.columns])
    ax.set_yticks(range(r))
    ax.set_yticklabels([idx.strftime("%Y-%m") for idx in mat.index])
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort (first purchase month)")
    ax.set_title(title)

    # optional annotations when small
    if annotate or (r <= 12 and c <= 12):
        for i in range(r):
            for j in range(c):
                v = mat.iat[i, j]
                if np.isfinite(v) and (v > 0):
                    ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig
