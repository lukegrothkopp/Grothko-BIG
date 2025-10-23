import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import set_secrets_env
from src.llm import chat
from src.rag import build_index, retrieve
from src.analysis import column_overview, numeric_corr
from src.viz import missing_plot

st.set_page_config(
    page_title="InsightForge (OpenAI) – BI Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Secrets -> env for OpenAI SDK ----
set_secrets_env()

# ---- CSS (light) ----
st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: .5rem; }
.subtle { color: #6b7280; }
.metric-card { background: #f8fafc; padding: 10px 14px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---- Session state ----
for k, v in {
    "df": None,
    "rag_index": None,
    "chat_history": [],
    "model_name": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- Sidebar ----
with st.sidebar:
    st.image("assets/logo.svg", width=160)
    st.title("InsightForge")
    st.caption("AI-Powered Business Intelligence")

    st.subheader("🔑 API")
    key_present = "OPENAI_API_KEY" in st.secrets and bool(st.secrets["OPENAI_API_KEY"])
    st.write("OpenAI key:", "✅ set" if key_present else "❌ missing")

    st.divider()
    page = st.radio("Navigate", ["Dashboard", "Data Analysis", "AI Assistant", "Visualizations"])

    st.divider()
    st.subheader("📁 Upload CSV")
    up = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.session_state.df = df
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Load error: {e}")

    if st.session_state.df is not None and st.button("⚙️ Build/Refresh AI Index"):
        with st.spinner("Building local RAG index..."):
            try:
                st.session_state.rag_index = build_index(st.session_state.df)
                st.success("RAG index ready.")
            except Exception as e:
                st.error(f"RAG build failed: {e}")

st.markdown('<div class="main-header">📊 InsightForge – AI Business Intelligence</div>', unsafe_allow_html=True)
st.caption("OpenAI + Streamlit + FAISS • Local RAG over your data profile")

# ---- Pages ----
if page == "Dashboard":
    st.header("📈 Business Intelligence Dashboard")
    if st.session_state.df is None:
        st.info("Upload a CSV to begin.")
    else:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Records", f"{len(df):,}")
        with col2: st.metric("Columns", df.shape[1])
        with col3:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols):
                st.metric(f"Avg {num_cols[0]}", f"{df[num_cols[0]].mean():.2f}")
            else:
                st.metric("Numeric Cols", "None")
        with col4:
            completeness = (1 - df.isna().sum().sum() / (df.shape[0]*df.shape[1])) * 100
            st.metric("Data Quality", f"{completeness:.1f}%")

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Quick Stats")
        try:
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        except Exception:
            st.dataframe(df.describe().transpose(), use_container_width=True)

elif page == "Data Analysis":
    st.header("🔍 Advanced Data Analysis")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df
        t1, t2, t3 = st.tabs(["Column Analysis", "Missing Data", "Correlations"])
        with t1:
            st.subheader("Column Information")
            st.dataframe(column_overview(df), use_container_width=True)
        with t2:
            st.subheader("Missing Data")
            fig = missing_plot(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values 🎉")
        with t3:
            st.subheader("Correlation Heatmap")
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] > 1:
                corr = numeric_corr(df)
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns.")

elif page == "AI Assistant":
    st.header("🤖 BI Assistant (RAG over your data summary)")
    if st.session_state.df is None:
        st.warning("Upload a dataset to enable the assistant.")
    elif st.session_state.rag_index is None:
        st.info("Click **Build/Refresh AI Index** in the sidebar.")
    else:
        # render history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask a question about your data…")
        if user_q:
            with st.chat_message("user"):
                st.write(user_q)
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            # retrieve
            top_chunks = retrieve(st.session_state.rag_index, user_q, k=4)
            context = "\n\n".join([c for c, _ in top_chunks])

            sys = (
                "You are a precise data analyst. "
                "Use the provided CONTEXT to ground your answer. "
                "If a value is not in CONTEXT, say you cannot determine it without more data. "
                "Format lists and tables clearly."
            )
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Question: {user_q}\n\nCONTEXT:\n{context}"}
            ]

            with st.chat_message("assistant"):
                try:
                    ans = chat(messages, model=st.session_state["model_name"], temperature=0.2)
                    st.write(ans)
                    # show expandable context
                    with st.expander("📚 View retrieved context"):
                        for i, (c, score) in enumerate(top_chunks, 1):
                            st.markdown(f"**Chunk {i} (score {score:.4f})**")
                            st.text(c[:1200] + ("..." if len(c) > 1200 else ""))
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"LLM error: {e}")

elif page == "Visualizations":
    st.header("📊 Visualizations")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df
        viz_type = st.selectbox("Type", ["Line", "Bar", "Scatter", "Pie", "Histogram", "Box"])
        cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
        col1, col2 = st.columns(2)

        with col1:
            if viz_type in ["Line", "Bar"]:
                x_col = st.selectbox("X-axis", df.columns)
                y_col = st.selectbox("Y-axis (numeric)", cols_num) if cols_num else None
            elif viz_type == "Scatter":
                x_col = st.selectbox("X-axis (numeric)", cols_num) if cols_num else None
                y_col = st.selectbox("Y-axis (numeric)", cols_num) if cols_num else None
            elif viz_type == "Pie":
                x_col = st.selectbox("Category", cols_cat or df.columns.tolist())
                y_col = st.selectbox("Value (numeric)", cols_num) if cols_num else None
            else:  # Histogram / Box
                x_col = st.selectbox("Column (numeric)", cols_num or df.columns.tolist())
                y_col = None

        with col2:
            color_by = st.selectbox("Color (optional)", ["None"] + df.columns.tolist())
            color_by = None if color_by == "None" else color_by

        st.subheader("Chart")
        try:
            if viz_type == "Line" and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_by, title=f"{y_col} over {x_col}")
            elif viz_type == "Bar" and y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_by, title=f"{y_col} by {x_col}")
            elif viz_type == "Scatter" and (x_col and y_col):
                fig = px.scatter(df, x=x_col, y=y_col, color=color_by, title=f"{y_col} vs {x_col}")
            elif viz_type == "Pie" and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} distribution by {x_col}")
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_by, title=f"Distribution of {x_col}")
            elif viz_type == "Box":
                fig = px.box(df, y=x_col, color=color_by, title=f"Box plot of {x_col}")
            else:
                fig = None
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select appropriate fields to plot.")
        except Exception as e:
            st.error(f"Visualization error: {e}")

# ---- Footer ----
st.divider()
st.markdown(
    "<div style='text-align:center;color:#6b7280'>InsightForge • OpenAI • Streamlit • FAISS</div>",
    unsafe_allow_html=True,
)
