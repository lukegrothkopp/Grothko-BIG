from __future__ import annotations
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_DIR = Path(os.getenv("INSIGHTFORGE_DOCS_DIR", "docs"))
INDEX_DIR = Path(os.getenv("INSIGHTFORGE_INDEX_DIR", "storage/faiss_index"))

def _load_docs() -> List:
    docs = []
    if not DOCS_DIR.exists():
        return docs
    pdf_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs += pdf_loader.load()
    for pattern in ["**/*.txt", "**/*.md", "**/*.csv"]:
        tl = DirectoryLoader(str(DOCS_DIR), glob=pattern, loader_cls=TextLoader, show_progress=True)
        try:
            docs += tl.load()
        except Exception:
            pass
    return docs

def build_index(embedding_model: str = "text-embedding-3-small"):
    docs = _load_docs()
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embed = OpenAIEmbeddings(model=embedding_model)
    vs = FAISS.from_documents(chunks, embed)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    return vs

def load_index():
    if not INDEX_DIR.exists():
        return None
    try:
        embed = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(str(INDEX_DIR), embed, allow_dangerous_deserialization=True)
    except Exception:
        return None

def retrieve(query: str, k: int = 4) -> List[str]:
    vs = load_index()
    if vs is None:
        return []
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]
