from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np
from io import StringIO
from dataclasses import dataclass
from src.llm import embed_text
from langchain_community.vectorstores import FAISS  # local, fast
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

# ---- Data profiling -> a single text corpus we can chunk ----

def dataframe_profile(df: pd.DataFrame) -> str:
    parts = []
    parts.append(f"Dataset: {len(df)} rows x {df.shape[1]} columns.")
    parts.append("Columns: " + ", ".join(map(str, df.columns.tolist())))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        parts.append("\nNumeric summary:")
        desc = df[num_cols].describe().round(3)
        buf = StringIO()
        desc.to_string(buf)
        parts.append(buf.getvalue())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        parts.append("\nCategoricals:")
        for c in cat_cols[:12]:  # cap for brevity
            vc = df[c].value_counts(dropna=False).head(10)
            parts.append(f"- {c}: top values: {vc.to_dict()}")

    missing = df.isna().sum()
    miss = missing[missing > 0]
    if not miss.empty:
        parts.append("\nMissing values:")
        parts.extend([f"- {c}: {int(v)}" for c, v in miss.items()])

    # include small sample for RAG grounding
    head_buf = StringIO()
    df.head(20).to_string(head_buf)
    parts.append("\nSample rows:\n" + head_buf.getvalue())

    dtypes_buf = StringIO()
    df.dtypes.to_string(dtypes_buf)
    parts.append("\nDtypes:\n" + dtypes_buf.getvalue())

    return "\n".join(parts)

# ---- Chunking ----

def chunk_text(txt: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    chunks = []
    start = 0
    while start < len(txt):
        end = min(len(txt), start + chunk_size)
        chunks.append(txt[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(txt):
            break
    return chunks

@dataclass
class RAGIndex:
    vectorstore: FAISS
    chunks: List[str]

def build_index(df: pd.DataFrame) -> RAGIndex:
    corpus = dataframe_profile(df)
    chunks = chunk_text(corpus)
    embeddings = embed_text(chunks)  # OpenAI embeddings
    # Build FAISS
    vs = FAISS.from_embeddings(
        text_embeddings=list(zip(chunks, embeddings)),
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))],
        distance_strategy=DistanceStrategy.COSINE
    )
    return RAGIndex(vectorstore=vs, chunks=chunks)

def retrieve(rag: RAGIndex, query: str, k: int = 4) -> List[Tuple[str, float]]:
    docs_and_scores = rag.vectorstore.similarity_search_with_score(query, k=k)
    return [(d.page_content, float(score)) for d, score in docs_and_scores]
