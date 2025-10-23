import plotly.express as px

def missing_values_bar(missing_series):
    fig = px.bar(
        x=missing_series.values,
        y=missing_series.index,
        orientation="h",
        title="Missing Values by Column",
        labels={"x": "Missing", "y": "Column"},
    )
    return fig

def corr_heatmap(corr_df):
    fig = px.imshow(
        corr_df, text_auto=True, aspect="auto",
        title="Correlation Heatmap", color_continuous_scale="RdBu"
    )
    return fig
