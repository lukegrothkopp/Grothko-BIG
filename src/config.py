import streamlit as st
from openai import OpenAI

def get_settings():
    s = st.secrets
    return {
        "app_name": s.get("APP_NAME", "InsightForge"),
        "model":   s.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "embed":   s.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        "api_key": s["OPENAI_API_KEY"],
    }

@st.cache_resource(show_spinner=False)
def get_openai():
    cfg = get_settings()
    return OpenAI(api_key=cfg["api_key"])
