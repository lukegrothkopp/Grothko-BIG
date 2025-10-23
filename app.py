# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="InsightForge - AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = ""

# Function to create data summary
def create_data_summary(df):
    """Create a comprehensive text summary of the dataset for RAG"""
    summary_parts = []
    summary_parts.append(f"Dataset Overview: The dataset contains {len(df)} records and {df.shape[1]} columns.")
    summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        summary_parts.append("\nNumeric Columns Analysis:")
        for col in numeric_cols:
            summary_parts.append(f"\n{col}:")
            summary_parts.append(f"  - Mean: {df[col].mean():.2f}")
            summary_parts.append(f"  - Median: {df[col].median():.2f}")
            summary_parts.append(f"  - Std Dev: {df[col].std():.2f}")
            summary_parts.append(f"  - Min: {df[col].min():.2f}")
            summary_parts.append(f"  - Max: {df[col].max():.2f}")

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary_parts.append("\nCategorical Columns Analysis:")
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            summary_parts.append(f"\n{col}:")
            summary_parts.append(f"  - Unique values: {unique_vals}")
            if unique_vals <= 10:
                value_counts = df[col].value_counts()
                summary_parts.append(f"  - Distribution: {value_counts.to_dict()}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary_parts.append("\nMissing Data:")
        for col in missing[missing > 0].index:
            summary_parts.append(f"  - {col}: {missing[col]} missing values")

    summary_parts.append("\nKey Insights:")
    summary_parts.append(f"  - Total rows: {len(df)}")
    summary_parts.append(f"  - Data completeness: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")
    return "\n".join(summary_parts)

# Function to setup RAG system
def setup_rag_system(df, api_key):
    """Setup RAG system with OpenAI chat model and OpenAI embeddings"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key

        data_summary = create_data_summary(df)
        st.session_state.data_summary = data_summary

        data_insights = f"""
        {data_summary}

        Sample Data Records:
        {df.head(20).to_string()}

        Column Data Types and Info:
        {df.dtypes.to_string()}
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(data_insights)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )

        st.session_state.conversation_chain = conversation_chain
        return True
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        return False

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=InsightForge", width=150)
    st.title("Navigation")

    st.subheader("🔑 API Configuration")
    api_key = st.text_input(
        "Enter OpenAI API Key:",
        type="password",
        help="Create a key at https://platform.openai.com/api-keys"
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key configured")

    st.divider()

    page = st.radio(
        "Select Page:",
        ["Dashboard", "Data Analysis", "AI Assistant", "Visualizations"]
    )

    st.divider()

    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)",
        type=['csv'],
        help="Upload a CSV file containing your business data"
    )

    if uploaded_file is not None and api_key:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            if st.session_state.vectorstore is None:
                with st.spinner("Setting up AI system..."):
                    if setup_rag_system(df, api_key):
                        st.success("🤖 AI system ready!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif uploaded_file is not None and not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key first")

# Main content
st.markdown('<h1 class="main-header">📊 InsightForge - AI-Powered Business Intelligence</h1>', unsafe_allow_html=True)

# Dashboard Page
if page == "Dashboard":
    st.header("📈 Business Intelligence Dashboard")

    if st.session_state.data_loaded:
        df = st.session_state.df

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Total Records", value=f"{len(df):,}", delta="Active")

        with col2:
            st.metric(label="Data Columns", value=df.shape[1], delta="Features")

        with col3:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                st.metric(label=f"Avg {numeric_cols[0]}", value=f"{df[numeric_cols[0]].mean():.2f}")

        with col4:
            st.metric(
                label="Data Quality",
                value=f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%",
                delta="Complete"
            )

        st.divider()

        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📊 Quick Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    else:
        st.info("👈 Please upload a dataset and enter your OpenAI API Key to get started.")
        st.markdown("""
        ### Getting Started
        1. Create an API key at the OpenAI Platform
        2. Enter the API key in the sidebar
        3. Upload your CSV file
        4. Explore AI-powered insights!
        """)

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🔍 Advanced Data Analysis")

    if st.session_state.data_loaded:
        df = st.session_state.df

        tab1, tab2, tab3 = st.tabs(["Column Analysis", "Missing Data", "Correlations"])

        with tab1:
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)

        with tab2:
            st.subheader("Missing Data Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

            if len(missing_data) > 0:
                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Number of Missing Values', 'y': 'Column'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing data found! 🎉")

        with tab3:
            st.subheader("Correlation Analysis")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])

            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
    else:
        st.warning("Please upload a dataset first.")

# AI Assistant Page
elif page == "AI Assistant":
    st.header("🤖 AI-Powered Business Intelligence Assistant")

    if st.session_state.data_loaded and api_key and st.session_state.conversation_chain:
        st.info("💡 Ask questions about your data and get AI-powered insights with RAG!")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input("Ask a question about your data...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    try:
                        response = st.session_state.conversation_chain({"question": user_question})
                        ai_response = response['answer']
                        st.write(ai_response)

                        if 'source_documents' in response and response['source_documents']:
                            with st.expander("📚 View Source Context"):
                                for i, doc in enumerate(response['source_documents']):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content[:300] + "...")

                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    else:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API Key in the sidebar.")
        elif not st.session_state.data_loaded:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            st.warning("⚠️ AI system is not ready. Please reload the page.")

# Visualizations Page
elif page == "Visualizations":
    st.header("📊 Data Visualizations")

    if st.session_state.data_loaded:
        df = st.session_state.df

        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"]
        )

        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if viz_type in ["Line Chart", "Bar Chart"]:
                x_col = st.selectbox("Select X-axis:", df.columns.tolist())
                y_col = st.selectbox("Select Y-axis:", numeric_cols) if numeric_cols else None
            elif viz_type == "Scatter Plot":
                x_col = st.selectbox("Select X-axis:", numeric_cols)
                y_col = st.selectbox("Select Y-axis:", numeric_cols)
            elif viz_type == "Pie Chart":
                x_col = st.selectbox("Select Category:", categorical_cols if categorical_cols else df.columns.tolist())
                y_col = st.selectbox("Select Value:", numeric_cols) if numeric_cols else None
            elif viz_type in ["Histogram", "Box Plot"]:
                x_col = st.selectbox("Select Column:", numeric_cols if numeric_cols else df.columns.tolist())
                y_col = None

        with col2:
            color_col = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist())
            color_col = None if color_col == "None" else color_col

        st.subheader(f"{viz_type} Visualization")

        try:
            if viz_type == "Line Chart" and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
            elif viz_type == "Bar Chart" and y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            elif viz_type == "Scatter Plot":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            elif viz_type == "Pie Chart" and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution of {y_col}")
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
            elif viz_type == "Box Plot":
                fig = px.box(df, y=x_col, color=color_col, title=f"Box Plot of {x_col}")

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.warning("Please upload a dataset first.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>InsightForge - AI-Powered Business Intelligence | Built with Streamlit & LangChain (OpenAI)</p>
</div>
""", unsafe_allow_html=True)
