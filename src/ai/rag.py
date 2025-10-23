from __future__ import annotations
import numpy as np
import pandas as pd
import faiss
from typing import List, Tuple
from openai import OpenAI

def _cosine_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def summarize_dataframe(df: pd.DataFrame) -> str:
    parts = []
    parts.append(f"Dataset has {len(df)} rows and {df.shape[1]} columns.")
    parts.append("Columns: " + ", ".join(df.columns.tolist()))
    num_cols = df.select_dtypes(include=["float64","int64"]).columns
    if len(num_cols):
        parts.append("\nNumeric column stats:")
        desc = df[num_cols].describe(include="all").round(2)
        parts.append(desc.to_string())
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols):
        parts.append("\nCategorical columns (unique counts):")
        uniq = {c: int(df[c].nunique()) for c in cat_cols}
        parts.append(str(uniq))
    missing = df.isnull().sum()
    if missing.sum() > 0:
        parts.append("\nMissing values:")
        parts.append(missing[missing > 0].sort_values(ascending=False).to_string())
    parts.append(f"\nData completeness: { (1 - df.isnull().to_numpy().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%")
    return "\n".join(parts)

def chunk_text(txt: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    chunks, start = [], 0
    while start < len(txt):
        end = min(len(txt), start + chunk_size)
        chunks.append(txt[start:end])
        start = end - overlap
        if start < 0: start = 0
        if end == len(txt): break
    return chunks

def build_rag_index(client: OpenAI, embed_model: str, df: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, List[str]]:
    summary = summarize_dataframe(df)
    head = df.head(20).to_string()
    types = df.dtypes.to_string()

    corpus = f"""{summary}

Sample records:
{head}

Column dtypes:
{types}
"""
    texts = chunk_text(corpus, 900, 180)
    embs = client.embeddings.create(model=embed_model, input=texts)
    vecs = np.array([e.embedding for e in embs.data], dtype=np.float32)
    vecs = _cosine_normalize(vecs)

    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx, texts

def retrieve(client: OpenAI, embed_model: str, idx: faiss.IndexFlatIP, texts: List[str], query: str, k: int = 3) -> List[str]:
    q = client.embeddings.create(model=embed_model, input=[query]).data[0].embedding
    qv = _cosine_normalize(np.array([q], dtype=np.float32))
    D, I = idx.search(qv, k)
    return [texts[i] for i in I[0]]

def answer_with_context(client: OpenAI, model: str, question: str, contexts: List[str], chat_history: List[dict]) -> str:
    sys = (
        "You are InsightForge, a precise BI analyst. "
        "Use the provided context from the user's dataset to answer clearly with numbers, caveats, and steps. "
        "If the context is insufficient, say what extra columns or filters are needed."
    )
    context_block = "\n\n---\n".join(contexts)
    messages = [{"role": "system", "content": sys}]
    for m in chat_history[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context_block}"} )

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return resp.choices[0].message.content
