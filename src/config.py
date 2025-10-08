from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

def _get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

load_dotenv()

@dataclass
class Settings:
    openai_api_key: str = _get_secret("OPENAI_API_KEY", "")
    data_path: str = _get_secret("INSIGHTFORGE_DATA", "./data/sample_sales.csv")

settings = Settings()

if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
