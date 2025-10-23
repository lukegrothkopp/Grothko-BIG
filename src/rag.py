# src/rag.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from io import StringIO
from dataclasses import dataclass
import faiss

from src.llm import embed_text

# ---------- Data profiling -> corpus we can chunk/index ----------

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

    # include a small sample for RAG grounding
    head_buf = StringIO()
    df.head(20).to_string(head_buf)
    parts.append("\nSample rows:\n" + head_buf.getvalue())

    dtypes_buf = StringIO()
    df.dtypes.to_string(dtypes_buf)
    parts.append("\nDtypes:\n" + dtypes_buf.getvalue())

    return "\n".join(parts)

# ---------- Chunking ----------

def chunk_text(txt: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    chunks = []
    start = 0
    while start < len(txt):
        end = min(len(txt), start + chunk_size)
        chunks.append(txt[start:end])
        if end == len(txt):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- Simple FAISS-backed index (cosine similarity via normalized vectors) ----------

@dataclass
class RAGIndex:
    index: faiss.IndexFlatIP              # inner product index on normalized vectors = cosine sim
    embeddings: np.ndarray                # shape (n_chunks, dim)
    chunks: List[str]

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def build_index(df: pd.DataFrame) -> RAGIndex:
    corpus = dataframe_profile(df)
    chunks = chunk_text(corpus)

    # Get OpenAI embeddings
    vecs = np.array(embed_text(chunks), dtype="float32")
    vecs = _normalize(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine when vectors are normalized
    index.add(vecs)

    return RAGIndex(index=index, embeddings=vecs, chunks=chunks)

def retrieve(rag: RAGIndex, query: str, k: int = 4) -> List[Tuple[str, float]]:
    qvec = np.array(embed_text([query])[0], dtype="float32").reshape(1, -1)
    qvec = _normalize(qvec)

    scores, ids = rag.index.search(qvec, k)
    out: List[Tuple[str, float]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        out.append((rag.chunks[int(idx)], float(score)))
    return out
