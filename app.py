# app.py

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightForge - AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Secrets → env (OpenAI)
# --------------------------------------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = st.secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data_summary" not in st.session_state:
    st.session_state.data_summary = ""

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def create_data_summary(df: pd.DataFrame) -> str:
    """Create a concise dataset summary for RAG context."""
    parts = []
    parts.append(
        f"Dataset Overview: The dataset contains {len(df)} records and {df.shape[1]} columns."
    )
    parts.append(f"Columns: {', '.join(df.columns.tolist())}")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        parts.append("\nNumeric Columns Analysis:")
        for col in numeric_cols:
            try:
                parts.append(f"\n{col}:")
                parts.append(f"  - Mean: {df[col].mean():.2f}")
                parts.append(f"  - Median: {df[col].median():.2f}")
                parts.append(f"  - Std Dev: {df[col].std():.2f}")
                parts.append(f"  - Min: {df[col].min():.2f}")
                parts.append(f"  - Max: {df[col].max():.2f}")
            except Exception:
                pass

    # Categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        parts.append("\nCategorical Columns Analysis:")
        for col in categorical_cols:
            try:
                unique_vals = df[col].nunique()
                parts.append(f"\n{col}:")
                parts.append(f"  - Unique values: {unique_vals}")
                if unique_vals <= 10:
                    value_counts = df[col].value_counts(dropna=False)
                    parts.append(f"  - Distribution: {value_counts.to_dict()}")
            except Exception:
                pass

    # Missing data
    missing = df.isna().sum()
    if missing.sum() > 0:
        parts.append("\nMissing Data:")
        for col, cnt in missing[missing > 0].items():
            parts.append(f"  - {col}: {int(cnt)} missing values")

    # Overall completeness
    completeness = (1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    parts.append("\nKey Insights:")
    parts.append(f"  - Total rows: {len(df)}")
    parts.append(f"  - Data completeness: {completeness:.1f}%")

    return "\n".join(parts)


def setup_rag_system(df: pd.DataFrame) -> bool:
    """Build vector store and conversation chain (OpenAI + FAISS)."""
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing in Streamlit secrets.")

        # Create summary + enriched context
        summary = create_data_summary(df)
        st.session_state.data_summary = summary

        data_insights = f"""
{summary}

Sample Data Records:
{df.head(20).to_string()}

Column Data Types and Info:
{df.dtypes.to_string()}
"""

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = splitter.split_text(data_insights)

        # Embeddings (OpenAI)
        embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBED_MODEL,
            api_key=OPENAI_API_KEY,
        )

        # Vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

        # LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
        )

        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        # Conversational chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        st.session_state.conversation_chain = conversation_chain
        return True
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        return False


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.title("Navigation")

    has_openai = bool(OPENAI_API_KEY)

    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file containing your business data",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            if has_openai and st.session_state.vectorstore is None:
                with st.spinner("Setting up AI system..."):
                    if setup_rag_system(df):
                        st.success("🤖 AI system ready!")
            elif not has_openai:
                st.warning(
                    "⚠️ Add OPENAI_API_KEY to Streamlit secrets (App → Settings → Secrets) to enable the AI Assistant."
                )
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()
    page = st.radio(
        "Select Page:",
        ["Dashboard", "Data Analysis", "AI Assistant", "Visualizations"],
    )

# --------------------------------------------------------------------------------------
# Main header
# --------------------------------------------------------------------------------------
st.markdown(
    '<h1 class="main-header">📊 InsightForge - AI-Powered Business Intelligence</h1>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
if page == "Dashboard":
    st.header("📈 Business Intelligence Dashboard")

    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Records", value=f"{len(df):,}", delta="Active")
        with col2:
            st.metric(label="Data Columns", value=df.shape[1], delta="Features")
        with col3:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                st.metric(
                    label=f"Avg {numeric_cols[0]}",
                    value=f"{df[numeric_cols[0]].mean():.2f}",
                )
        with col4:
            completeness = (1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric(label="Data Quality", value=f"{completeness:.1f}%", delta="Complete")

        st.divider()

        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📊 Quick Statistics")
        try:
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        except Exception:
            st.dataframe(df.describe().transpose(), use_container_width=True)
    else:
        st.info("👈 Please upload a dataset to get started.")
        st.markdown(
            """
            ### Getting Started
            1. Add your `OPENAI_API_KEY` in **App → Settings → Secrets** (optional for data viewing; required for AI features)  
            2. Upload your CSV file  
            3. Explore AI-powered insights!
            """
        )

elif page == "Data Analysis":
    st.header("🔍 Advanced Data Analysis")

    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df

        tab1, tab2, tab3 = st.tabs(["Column Analysis", "Missing Data", "Correlations"])

        with tab1:
            st.subheader("Column Information")
            col_info = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null Count": df.count(),
                    "Null Count": df.isna().sum(),
                    "Unique Values": df.nunique(),
                }
            )
            st.dataframe(col_info, use_container_width=True)

        with tab2:
            st.subheader("Missing Data Analysis")
            missing_data = df.isna().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_data) > 0:
                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation="h",
                    title="Missing Values by Column",
                    labels={"x": "Number of Missing Values", "y": "Column"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing data found! 🎉")

        with tab3:
            st.subheader("Correlation Analysis")
            numeric_df = df.select_dtypes(include=["number"])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr(numeric_only=True)
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
    else:
        st.warning("Please upload a dataset first.")

elif page == "AI Assistant":
    st.header("🤖 AI-Powered Business Intelligence Assistant")

    has_openai = bool(OPENAI_API_KEY)

    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    elif not has_openai:
        st.warning("⚠️ Add OPENAI_API_KEY to Streamlit secrets.")
    elif st.session_state.conversation_chain is None:
        st.warning("⚠️ AI system is not ready yet. Try reloading the page after uploading your CSV.")
    else:
        st.info("💡 Ask questions about your data and get AI-powered insights with RAG!")

        # Render history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask a question about your data…")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    try:
                        resp = st.session_state.conversation_chain({"question": user_q})
                        ans = resp["answer"]
                        st.write(ans)

                        if "source_documents" in resp and resp["source_documents"]:
                            with st.expander("📚 View Source Context"):
                                for i, doc in enumerate(resp["source_documents"]):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content[:300] + "...")

                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    except Exception as e:
                        err = f"Error generating response: {e}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

elif page == "Visualizations":
    st.header("📊 Data Visualizations")

    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df

        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"],
        )

        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if viz_type in ["Line Chart", "Bar Chart"]:
                x_col = st.selectbox("Select X-axis:", df.columns.tolist())
                y_col = st.selectbox("Select Y-axis:", numeric_cols) if numeric_cols else None
            elif viz_type == "Scatter Plot":
                x_col = st.selectbox("Select X-axis:", numeric_cols) if numeric_cols else None
                y_col = st.selectbox("Select Y-axis:", numeric_cols) if numeric_cols else None
            elif viz_type == "Pie Chart":
                x_col = st.selectbox("Select Category:", categorical_cols or df.columns.tolist())
                y_col = st.selectbox("Select Value:", numeric_cols) if numeric_cols else None
            elif viz_type in ["Histogram", "Box Plot"]:
                x_col = st.selectbox("Select Column:", numeric_cols or df.columns.tolist())
                y_col = None

        with col2:
            color_col = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist())
            color_col = None if color_col == "None" else color_col

        st.subheader(f"{viz_type} Visualization")

        try:
            fig = None
            if viz_type == "Line Chart" and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
            elif viz_type == "Bar Chart" and y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            elif viz_type == "Scatter Plot" and x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            elif viz_type == "Pie Chart" and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution of {y_col}")
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
            elif viz_type == "Box Plot":
                fig = px.box(df, y=x_col, color=color_col, title=f"Box Plot of {x_col}")

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select appropriate fields to plot.")
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.warning("Please upload a dataset first.")

# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
st.divider()
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>InsightForge — AI-Powered Business Intelligence | Built with Streamlit, LangChain, OpenAI</p>
</div>
""",
    unsafe_allow_html=True,
)
