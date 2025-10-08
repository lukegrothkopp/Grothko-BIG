import os
import streamlit as st
import pandas as pd
from src.config import settings
from src.data_loader import load_dataframe
from src.chains import InsightForgeAssistant
from src.viz import sales_trend_plot, product_performance_plot, regional_performance_plot

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge — BI Assistant")

# Sidebar
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    st.sidebar.info("Using sample dataset.")
    df = load_dataframe(settings.data_path)

# Overview
st.subheader("Quick Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Total Sales", f"${df['sales'].sum():,.0f}")
col3.metric("Avg. Discount", f"{df['discount_pct'].mean():.1f}%")
col4.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

# Visualizations
st.subheader("Visualizations")
c1, c2 = st.columns([2, 1])
with c1:
    st.pyplot(sales_trend_plot(df))
with c2:
    st.pyplot(product_performance_plot(df))
st.pyplot(regional_performance_plot(df))

# Chat Assistant
st.subheader("Ask InsightForge")
question = st.text_input("Your question (e.g., 'What are the top 3 products this quarter and where should we focus?')")

if "assistant" not in st.session_state:
    st.session_state.assistant = InsightForgeAssistant(df)

if st.button("Generate Insight") and question:
    answer = st.session_state.assistant.answer(question)
    st.markdown(answer)
    st.success("Response generated using RAG + LLM.")
