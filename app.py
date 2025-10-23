import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import get_openai, get_settings
from src.utils.session import init_state
from src.data.loader import load_csv
from src.viz.plots import missing_values_bar, corr_heatmap
from src.ai.rag import build_rag_index, retrieve, answer_with_context

st.set_page_config(page_title="InsightForge - AI Business Intelligence", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
init_state()
cfg = get_settings()
client = get_openai()

# ---- Sidebar ----
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=InsightForge", width=150)
    st.title("Navigation")
    page = st.radio("Select Page:", ["Dashboard", "Data Analysis", "AI Assistant", "Visualizations"])
    st.divider()
    st.subheader("📁 Data Upload")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = load_csv(uploaded)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

            if st.session_state.rag_index is None:
                with st.spinner("Building AI index (embeddings + FAISS)…"):
                    idx, texts = build_rag_index(client, cfg["embed"], df)
                    st.session_state.rag_index = idx
                    st.session_state.rag_texts = texts
                    st.success("AI index ready.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

st.markdown(f"<h1 style='text-align:center;'>{cfg['app_name']} – AI-Powered Business Intelligence</h1>", unsafe_allow_html=True)

# ---- Pages ----
if page == "Dashboard":
    st.header("📈 Business Intelligence Dashboard")

    if st.session_state.data_loaded:
        df = st.session_state.df

        c1
