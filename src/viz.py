from __future__ import annotations
import pandas as pd
import plotly.express as px

def missing_plot(df: pd.DataFrame):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None
    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        labels={"x": "Missing", "y": "Column"},
        title="Missing values by column"
    )
    return fig
