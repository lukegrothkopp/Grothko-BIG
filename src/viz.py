import pandas as pd
import matplotlib.pyplot as plt

def sales_trend_plot(df: pd.DataFrame):
    monthly = df.set_index("date").groupby(pd.Grouper(freq="ME"))["sales"].sum()
    fig, ax = plt.subplots()
    monthly.plot(ax=ax)
    ax.set_title("Sales Trend (Monthly)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    return fig

def product_performance_plot(df: pd.DataFrame, top_n: int = 5):
    prod = df.groupby("product")["sales"].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots()
    prod.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {top_n} Products by Sales")
    ax.set_xlabel("Product")
    ax.set_ylabel("Total Sales")
    return fig

def regional_performance_plot(df: pd.DataFrame):
    reg = df.groupby("region")["sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    reg.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales")
    return fig
