from __future__ import annotations
from typing import List, Dict, Any
import os
from openai import OpenAI

def get_client() -> OpenAI:
    # OPENAI_API_KEY is read by the SDK from env or override below
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # allow Streamlit to set env from secrets before import
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

def chat(messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.3) -> str:
    client = get_client()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def embed_text(texts: List[str], model: str | None = None) -> List[List[float]]:
    client = get_client()
    model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
