from __future__ import annotations
import streamlit as st

def set_secrets_env():
    # Map Streamlit secrets into environment for the OpenAI SDK
    import os
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in st.secrets:
        os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]
    if "OPENAI_EMBED_MODEL" in st.secrets:
        os.environ["OPENAI_EMBED_MODEL"] = st.secrets["OPENAI_EMBED_MODEL"]
