import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import settings
from src.data_loader import load_dataframe
from src.chains import InsightForgeAssistant
from src.viz import sales_trend_plot, product_performance_plot, regional_performance_plot
from src.memory import reset_memory
from src.doc_index import build_index

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge — BI Assistant")

# Sidebar: user/session controls
st.sidebar.header("Session & Data")
user_id = st.sidebar.text_input("User ID (for persistent memory)", value="default")
if st.sidebar.button("Reset memory", type="secondary"):
    reset_memory(user_id)
    st.sidebar.success(f"Memory reset for user_id={user_id}")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
rebuild = st.sidebar.button("Rebuild doc index")
if rebuild:
    with st.spinner("Building FAISS index from docs/..."):
        build_index()
    st.sidebar.success("Rebuilt document index.")

# Load data
if uploaded is not None:
    try:
        # strict path: requires canonical columns already present
        df = load_dataframe(uploaded)
    except Exception:
        # IMPORTANT: reset pointer before reading again
        try:
            uploaded.seek(0)
        except Exception:
            pass
        # robust read: infer delimiter + strip BOM
        raw = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")

        # try flexible normalization to canonical schema
        from src.data_loader import prepare_any_sales_dataframe
        try:
            df = prepare_any_sales_dataframe(raw)
            st.sidebar.success("Auto-mapped your CSV to InsightForge’s schema.")
        except Exception as e2:
            st.sidebar.error(
                "Your CSV is missing required fields for InsightForge.\n\n"
                f"Details: {e2}\n\n"
                "Required columns: date, region, product, customer_id, age, gender, quantity, sales, discount_pct"
            )
            st.stop()
else:
    st.sidebar.info("Using sample dataset.")
    df = load_dataframe(settings.data_path)

# Filters
st.sidebar.header("Filters")
min_date = pd.to_datetime(df["date"].min()) if "date" in df.columns else None
max_date = pd.to_datetime(df["date"].max()) if "date" in df.columns else None
if min_date is not None and max_date is not None:
    date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
else:
    date_range = None

regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
products = sorted(df["product"].dropna().unique().tolist()) if "product" in df.columns else []

sel_regions = st.sidebar.multiselect("Region", regions, default=regions[:])
sel_products = st.sidebar.multiselect("Product", products, default=products[:])

# Apply filters
df_f = df.copy()
if date_range and len(date_range) == 2 and "date" in df_f.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["date"] >= start) & (df_f["date"] <= end)]
if sel_regions and "region" in df_f.columns:
    df_f = df_f[df_f["region"].isin(sel_regions)]
if sel_products and "product" in df_f.columns:
    df_f = df_f[df_f["product"].isin(sel_products)]

# Overview
st.subheader("Quick Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df_f):,}")
col2.metric("Total Sales", f"${df_f['sales'].sum():,.0f}")
col3.metric("Avg. Discount", f"{df_f['discount_pct'].mean():.1f}%")
col4.metric("Unique Customers", f"{df_f['customer_id'].nunique():,}")

# Visualizations
st.subheader("Visualizations")
c1, c2 = st.columns([2, 1])
with c1:
    st.pyplot(sales_trend_plot(df_f), clear_figure=True)
with c2:
    st.pyplot(product_performance_plot(df_f), clear_figure=True)
st.pyplot(regional_performance_plot(df_f), clear_figure=True)

# Advanced Visuals
st.subheader("Advanced Visuals")
# Cohort heatmap
if "customer_id" in df_f.columns and "date" in df_f.columns:
    with st.expander("Cohort heatmap settings", expanded=False):
        metric = st.selectbox("Metric", ["Retention %", "Customers"], index=0)
        last_n = st.slider("Show last N cohorts", min_value=6, max_value=24, value=12, step=1)
        max_months = st.slider("Max months since first purchase", min_value=6, max_value=24, value=12, step=1)
        annotate = st.checkbox("Annotate cells (auto when small)", value=False)

    try:
        fig = cohort_heatmap(
            df_f,
            value="retention" if metric == "Retention %" else "customers",
            last_n_cohorts=last_n,
            max_months=max_months,
            annotate=annotate,
        )
        st.caption("Heatmap shows either % of cohort retained over time or unique customers, row-normalized for readability.")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Cohort heatmap unavailable: {e}")

# Discount vs AOV
if "discount_pct" in df_f.columns and "sales" in df_f.columns:
    disc = df_f.groupby("discount_pct")["sales"].agg(["mean", "count"]).reset_index()
    import matplotlib.pyplot as plt
    st.caption("Discount vs. average order value (AOV)")
    fig2, ax2 = plt.subplots()
    ax2.scatter(disc["discount_pct"], disc["mean"], s=np.clip(disc["count"], 10, 300))
    ax2.set_xlabel("Discount %")
    ax2.set_ylabel("Average Order Value")
    ax2.set_title("Discount vs AOV (bubble size ~ order count)")
    st.pyplot(fig2, clear_figure=True)

# Product mix treemap
try:
    import plotly.express as px
    pmix = df_f.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    st.caption("Product mix treemap by sales")
    fig3 = px.treemap(pmix, path=["product"], values="sales")
    st.plotly_chart(fig3, use_container_width=True)
except Exception:
    st.info("Plotly treemap unavailable; install plotly to enable.")

# Chat Assistant
st.subheader("Ask InsightForge")
question = st.text_input("Your question (e.g., 'What are the top 3 products this quarter and where should we focus?')")

if "assistant" not in st.session_state or st.session_state.get("assistant_user") != user_id:
    st.session_state.assistant = InsightForgeAssistant(df_f, user_id=user_id)
    st.session_state.assistant_user = user_id

if st.button("Generate Insight") and question:
    try:
        answer = st.session_state.assistant.answer(question)
        st.markdown(answer)
        st.success("Response generated using stats + vector RAG. Memory saved per user.")
    except Exception as e:
        st.exception(e)
