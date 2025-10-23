# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re
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

# -------------------------
# Session State
# -------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"assistant", "content": str}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = ""

# -------------------------
# Utilities
# -------------------------
def _norm(s: str) -> str:
    """Normalize column names for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _find_col(df: pd.DataFrame, name: str):
    """
    Fuzzy / case-insensitive column matcher with semantic aliases.
    - Normalizes strings (lowercase, strip non-alnum).
    - Handles percent/percentage/perc <-> pct.
    - Maps common business terms (price, revenue, qty, date, etc.) to your actual columns if present.
    Returns the actual column name from df or None.
    """
    import re

    def _norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    # --- percent/percentage alias generator ---
    def _alias_set(s: str):
        alts = {s}
        if 'percentage' in s:
            alts.update({s.replace('percentage', 'pct'),
                         s.replace('percentage', 'percent')})
        if 'percent' in s:
            alts.update({s.replace('percent', 'pct'),
                         s.replace('percent', 'percentage')})
        if 'perc' in s:
            alts.update({s.replace('perc', 'pct'),
                         s.replace('perc', 'percent'),
                         s.replace('perc', 'percentage')})
        if 'pct' in s:
            alts.update({s.replace('pct', 'percent'),
                         s.replace('pct', 'percentage'),
                         s.replace('pct', 'perc')})
        return alts

    # --- semantic aliases: concept -> preferred df column candidates (normalized) ---
    # Add/adjust any you like; the function will only return ones that actually exist in df.
    semantic_aliases = {
        # pricing / revenue
        'price':        ['unit_price', 'price', 'avg_price', 'averageprice'],
        'revenue':      ['net_revenue', 'revenue', 'sales', 'totalrevenue', 'netsales'],

        # quantities / counts
        'quantity':     ['units', 'quantity', 'qty', 'unitssold', 'unit_sold', 'units_sold'],
        'qty':          ['units', 'quantity', 'qty', 'unitssold', 'units_sold'],

        # dates
        'date':         ['purchase_date', 'order_date', 'date', 'purchasedate', 'orderdate'],

        # customer identifiers
        'customer':     ['customer_id', 'customerid', 'cust_id', 'custid', 'id'],

        # geography / demographics
        'city':         ['city', 'town'],
        'state':        ['state_province', 'state', 'province', 'region', 'stateprovince'],
        'province':     ['state_province', 'province', 'state'],
        'country':      ['country', 'nation'],
        'income':       ['household_income', 'income', 'householdincome'],
        'maritalstatus':['marital_status', 'maritalstatus'],
        'children':     ['number_children', 'children', 'numchildren', 'childcount'],
        'childrenages': ['children_ages', 'childrenages', 'kidsages'],

        # membership / marketing
        'subscription': ['subscription_member', 'member', 'ismember', 'loyalty'],
        'marketing':    ['marketing_source', 'marketingsource', 'utm_source', 'source'],
        'acquisition':  ['acquisition_channel', 'acquisitionchannel', 'channel', 'utm_medium'],
        'format':       ['preferred_format', 'format'],

        # discounts / coupons / gifting
        'discount':     ['discount_pct', 'discount', 'discountpercent', 'discountpercentage', 'pctdiscount'],
        'coupon':       ['coupon_used', 'coupon', 'promocode', 'discountcode'],
        'gift':         ['gifting', 'gift', 'gifted'],

        # feedback / recommendation
        'feedbackscore':['feedback_score', 'rating', 'score', 'satisfaction'],
        'feedback':     ['feedback_text', 'comment', 'review', 'feedback'],
        'recommend':    ['would_recommend', 'recommend', 'nps', 'promoter', 'wouldrecommend'],
        'nps':          ['would_recommend', 'nps'],

        # behavior / lifecycle
        'repeat':       ['repeat_buyer', 'repeat', 'returningcustomer', 'returning'],
        'timetofinish': ['time_to_finish_days', 'timetofinish', 'completiontimedays', 'completiontime'],
        'occupation':   ['occupation', 'job', 'profession'],
    }

    # Build fast lookup for actual df columns (normalized -> actual)
    col_norm_to_actual = {}
    for c in df.columns:
        col_norm_to_actual[_norm(c)] = c

    target = _norm(name)

    # 1) Direct + percent aliases: try exact normalized or alias-equivalents
    target_alts = _alias_set(target)
    for ta in ([target] + list(target_alts)):
        if ta in col_norm_to_actual:
            return col_norm_to_actual[ta]

    # 2) Contains match on normalized names (helpful for "discount percentage" -> "...pct")
    for ta in ([target] + list(target_alts)):
        for cnorm, actual in col_norm_to_actual.items():
            if ta in cnorm or cnorm in ta:
                return actual

    # 3) Semantic matching: map the target to a concept key and then to preferred candidates
    #    Choose the first candidate that exists in the dataframe.
    def _concept_keys(t: str):
        # try exact concept name, then substring hits
        keys = []
        # common concept normalizations
        concept_map = {
            'price': ['price', 'unitprice', 'avgprice'],
            'revenue': ['revenue', 'netsales', 'sales', 'totalrevenue'],
            'quantity': ['quantity', 'qty', 'units'],
            'qty': ['qty', 'quantity', 'units'],
            'date': ['date', 'orderdate', 'purchasedate'],
            'customer': ['customer', 'customerid', 'custid'],
            'state': ['state', 'province', 'stateprovince', 'region'],
            'province': ['province', 'state'],
            'country': ['country', 'nation'],
            'income': ['income', 'householdincome'],
            'maritalstatus': ['maritalstatus', 'marital'],
            'children': ['children', 'numchildren', 'childcount'],
            'childrenages': ['childrenages', 'kidsages'],
            'subscription': ['subscription', 'member', 'loyalty'],
            'marketing': ['marketing', 'utm', 'source'],
            'acquisition': ['acquisition', 'channel', 'utm'],
            'format': ['format', 'preferredformat'],
            'discount': ['discount', 'discountpercent', 'discountpercentage', 'pctdiscount'],
            'coupon': ['coupon', 'promocode', 'discountcode'],
            'gift': ['gift', 'gifting', 'gifted'],
            'feedbackscore': ['feedbackscore', 'rating', 'score', 'satisfaction'],
            'feedback': ['feedback', 'comment', 'review'],
            'recommend': ['recommend', 'wouldrecommend', 'nps', 'promoter'],
            'nps': ['nps', 'recommend'],
            'repeat': ['repeat', 'returning'],
            'timetofinish': ['timetofinish', 'completiontime', 'completiontimedays'],
            'occupation': ['occupation', 'job', 'profession'],
        }
        for key, hints in concept_map.items():
            if t == key or any(h in t for h in hints):
                keys.append(key)
        return keys

    concept_hits = _concept_keys(target)
    # Also consider alias-expanded variants (e.g., percentage -> pct)
    for ta in target_alts:
        concept_hits += _concept_keys(ta)
    # keep order but unique
    seen = set()
    concept_hits = [x for x in concept_hits if not (x in seen or seen.add(x))]

    for concept in concept_hits:
        if concept in semantic_aliases:
            for cand in semantic_aliases[concept]:
                cn = _norm(cand)
                # direct match
                if cn in col_norm_to_actual:
                    return col_norm_to_actual[cn]
                # contains fallback (handles e.g., 'orderdate' vs 'purchase_date')
                for cnorm, actual in col_norm_to_actual.items():
                    if cn in cnorm or cnorm in cn:
                        return actual

    # 4) Last resort: soft contains across all columns
    for cnorm, actual in col_norm_to_actual.items():
        if any(token in cnorm for token in [target] + list(target_alts)):
            return actual

    return None

# -------------------------
# Data Summary for RAG
# -------------------------
def create_data_summary(df: pd.DataFrame) -> str:
    """Create a comprehensive text summary of the dataset for RAG"""
    summary_parts = []
    summary_parts.append(f"Dataset Overview: The dataset contains {len(df)} records and {df.shape[1]} columns.")
    summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")

    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    if len(numeric_cols) > 0:
        summary_parts.append("\nNumeric Columns Analysis:")
        for col in numeric_cols:
            try:
                summary_parts.append(f"\n{col}:")
                summary_parts.append(f"  - Mean: {df[col].mean():.2f}")
                summary_parts.append(f"  - Median: {df[col].median():.2f}")
                summary_parts.append(f"  - Std Dev: {df[col].std():.2f}")
                summary_parts.append(f"  - Min: {df[col].min():.2f}")
                summary_parts.append(f"  - Max: {df[col].max():.2f}")
            except Exception:
                pass

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary_parts.append("\nCategorical Columns Analysis:")
        for col in categorical_cols:
            try:
                unique_vals = df[col].nunique()
                summary_parts.append(f"\n{col}:")
                summary_parts.append(f"  - Unique values: {unique_vals}")
                if unique_vals <= 10:
                    value_counts = df[col].value_counts()
                    summary_parts.append(f"  - Distribution: {value_counts.to_dict()}")
            except Exception:
                pass

    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary_parts.append("\nMissing Data:")
        for col in missing[missing > 0].index:
            summary_parts.append(f"  - {col}: {missing[col]} missing values")

    summary_parts.append("\nKey Insights:")
    completeness = (1 - df.isnull().sum().sum() / max(1, (df.shape[0] * df.shape[1]))) * 100
    summary_parts.append(f"  - Total rows: {len(df)}")
    summary_parts.append(f"  - Data completeness: {completeness:.1f}%")
    return "\n".join(summary_parts)

# -------------------------
# RAG Setup (no chains/memory imports)
# -------------------------
def setup_rag_system(df: pd.DataFrame, api_key: str) -> bool:
    """Setup RAG using FAISS retriever + OpenAI embeddings + ChatOpenAI"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key

        data_summary = create_data_summary(df)
        st.session_state.data_summary = data_summary

        data_insights = f"""
{data_summary}

Sample Data Records (first 20):
{df.head(20).to_string()}

Column Data Types and Info:
{df.dtypes.to_string()}
"""

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(data_insights)

        # Embeddings + Vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # Save to session
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        return True
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        return False

def answer_with_rag(question: str):
    """Minimal RAG loop: retrieve → prompt → call LLM. Returns (answer, source_docs)"""
    retriever = st.session_state.retriever
    llm = st.session_state.llm
    source_docs = []
    context = ""

    if retriever is not None:
        try:
            # LANGCHAIN v0.2+ retrievers are Runnables → use .invoke()
            source_docs = retriever.invoke(question)
            context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in source_docs)
        except Exception as e:
            context = ""
            source_docs = []
            st.error(f"Retrieval error: {e}")

    # brief history
    history_tail = st.session_state.chat_history[-6:]
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history_tail])

    system_instructions = (
        "You are a helpful business data analyst. Use the provided context (from the uploaded CSV) "
        "to answer the user's question. If the answer is not in the context, say you don't have enough "
        "information and suggest what to compute or inspect."
    )

    prompt = (
        f"{system_instructions}\n\n"
        f"Chat history (may be empty):\n{history_text}\n\n"
        f"Context from data-derived knowledge base:\n{context}\n\n"
        f"User question: {question}\n\n"
        f"Answer clearly and concisely:"
    )

    try:
        resp = st.session_state.llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
        return answer, source_docs
    except Exception as e:
        return f"Error generating response: {e}", source_docs

# -------------------------
# Structured Analytics Engine
# -------------------------
def handle_analytics_query(question: str, df: pd.DataFrame):
    """
    Detects and executes common analytics requests on the *actual* dataframe.
    Returns (handled: bool, message: str)
    """
    q = question.lower().strip()

    # --- Correlation between X and Y ---
    m = re.search(r'correlat\w*\s+.*\bbetween\b\s+(.+?)\s+\b(and|&)\b\s+(.+)', q)
    if m:
        raw_x = m.group(1)
        raw_y = m.group(3)
        col_x = _find_col(df, raw_x)
        col_y = _find_col(df, raw_y)

        if not col_x or not col_y:
            return True, ("I couldn't match both columns in your dataset.\n"
                          f"Matched X: `{col_x or 'None'}` | Matched Y: `{col_y or 'None'}`.\n"
                          f"Available columns: {list(df.columns)}")

        # Coerce to numeric
        x = pd.to_numeric(df[col_x], errors='coerce')
        y = pd.to_numeric(df[col_y], errors='coerce')
        valid = x.notna() & y.notna()

        if valid.sum() < 3:
            return True, f"Not enough overlapping numeric values to compute correlation between `{col_x}` and `{col_y}`."

        r = x[valid].corr(y[valid], method='pearson')
        msg = f"**Pearson correlation (r) between `{col_x}` and `{col_y}`:** **{r:.3f}** (n={valid.sum()})"
        st.write(msg)

        # Optional scatter chart
        try:
            fig = px.scatter(
                pd.DataFrame({col_x: x[valid], col_y: y[valid]}),
                x=col_x, y=col_y,
                title=f"Scatter: {col_x} vs {col_y}"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        return True, msg

    # --- Explicit request to "calculate Pearson correlation between 'x' and 'y'" ---
    m2 = re.search(r'(pearson|corr(?!elation)\b).*?(between|for)\s+[\'"]?([\w %_]+)[\'"]?\s+(and|,)\s+[\'"]?([\w %_]+)[\'"]?', q)
    if m2:
        raw_x = m2.group(3)
        raw_y = m2.group(5)
        col_x = _find_col(df, raw_x)
        col_y = _find_col(df, raw_y)
        if not col_x or not col_y:
            return True, ("I couldn't match both columns in your dataset.\n"
                          f"Matched X: `{col_x or 'None'}` | Matched Y: `{col_y or 'None'}`.\n"
                          f"Available columns: {list(df.columns)}")

        x = pd.to_numeric(df[col_x], errors='coerce')
        y = pd.to_numeric(df[col_y], errors='coerce')
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            return True, f"Not enough overlapping numeric values to compute correlation between `{col_x}` and `{col_y}`."

        r = x[valid].corr(y[valid], method='pearson')
        msg = f"**Pearson correlation (r) between `{col_x}` and `{col_y}`:** **{r:.3f}** (n={valid.sum()})"
        st.write(msg)
        try:
            fig = px.scatter(
                pd.DataFrame({col_x: x[valid], col_y: y[valid]}),
                x=col_x, y=col_y,
                title=f"Scatter: {col_x} vs {col_y}"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
        return True, msg

    # --- "top 3 insights" ---
    if re.search(r'\btop\s*3\s*insight', q):
        insights = []

        # 1) Strongest numeric correlation pair
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_pair = divmod(corr.values.argmax(), corr.shape[1])
            col_a, col_b = num_df.columns[max_pair[0]], num_df.columns[max_pair[1]]
            insights.append(f"Strongest numeric relationship: **{col_a}** vs **{col_b}** (|r| ≈ **{corr.values.max():.3f}**).")

        # 2) Top category in first categorical column
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            vc = df[cat_cols[0]].value_counts(dropna=True).head(3)
            insights.append(f"Top categories in **{cat_cols[0]}**: " + ", ".join([f"{k} ({v})" for k, v in vc.items()]))

        # 3) Highest mean in first numeric column (by first categorical if available)
        if num_df.shape[1] >= 1 and cat_cols:
            grp = df.groupby(cat_cols[0])[num_df.columns[0]].mean().sort_values(ascending=False).head(3)
            insights.append(f"Highest average **{num_df.columns[0]}** by **{cat_cols[0]}**: " +
                            ", ".join([f"{idx} ({val:.2f})" for idx, val in grp.items()]))

        if not insights:
            return True, "I couldn't derive quick insights—dataset might lack numeric/categorical variety."

        msg = "### Top 3 Insights\n" + "\n".join([f"1. {insights[0]}" if i == 0 else f"{i+1}. {insight}"
                                                  for i, insight in enumerate(insights[:3])])
        return True, msg

    # --- "trends or patterns" ---
    # (Fixed try/except structure to avoid SyntaxError)
    if re.search(r'\btrends?\b|\bpatterns?\b', q):
        bullets = []
        num_df = df.select_dtypes(include=[np.number])
        date_col = _first_datetime_col(df)
        if date_col:
            try:
                df2 = df.copy()
                df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
                if num_df.shape[1] >= 1:
                    ycol = num_df.columns[0]
                    ts = df2.dropna(subset=[date_col, ycol]).sort_values(date_col)
                    if len(ts) >= 3:
                        ts['__period'] = ts[date_col].dt.to_period('M').dt.to_timestamp()
                        agg = ts.groupby('__period')[ycol].mean()
                        direction = "increasing" if agg.iloc[-1] > agg.iloc[0] else "decreasing"
                        bullets.append(f"**Time trend** in **{ycol}**: appears {direction} from first to last observed month.")
                        try:
                            fig = px.line(agg.reset_index(), x='__period', y=ycol, title=f"Trend of {ycol} over time")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
            except Exception:
                pass

        # correlation highlight
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            corr_abs = corr.abs()
            np.fill_diagonal(corr_abs.values, 0)
            max_pair = divmod(corr_abs.values.argmax(), corr_abs.shape[1])
            col_a, col_b = num_df.columns[max_pair[0]], num_df.columns[max_pair[1]]
            bullets.append(f"**Strongest numeric relationship**: {col_a} vs {col_b} (|r| ≈ {corr_abs.values.max():.3f}).")

        if not bullets:
            bullets.append("No clear trends/patterns detected automatically. Try asking about specific columns or groups.")

        msg = "### Detected Trends & Patterns\n" + "\n".join([f"- {b}" for b in bullets])
        return True, msg

    # Not handled here
    return False, ""

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=InsightForge", width=150)
    st.title("Navigation")

    # API Key input
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

    # Data upload section
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

            # Setup RAG system
            if st.session_state.vectorstore is None:
                with st.spinner("Setting up AI system..."):
                    if setup_rag_system(df, api_key):
                        st.success("🤖 AI system ready!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif uploaded_file is not None and not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key first")

# -------------------------
# Main content
# -------------------------
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
            completeness = (1 - df.isnull().sum().sum() / max(1, (df.shape[0] * df.shape[1]))) * 100
            st.metric(label="Data Quality", value=f"{completeness:.1f}%", delta="Complete")

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

    if st.session_state.data_loaded and api_key and st.session_state.retriever and st.session_state.llm:
        st.info("💡 Ask questions about your data and get AI-powered insights with RAG! I can also run common analyses directly (correlations, top insights, basic trends).")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input
        user_question = st.chat_input("Ask a question about your data...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            handled, msg = handle_analytics_query(user_question, st.session_state.df)

            # If handled by analytics engine, show result; else fall back to RAG
            if handled:
                with st.chat_message("assistant"):
                    st.write(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your data..."):
                        answer, source_docs = answer_with_rag(user_question)
                        st.write(answer)

                        if source_docs:
                            with st.expander("📚 View Source Context"):
                                for i, doc in enumerate(source_docs):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(getattr(doc, "page_content", str(doc))[:300] + "...")

                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Clear chat button
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
