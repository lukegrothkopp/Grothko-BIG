import streamlit as st

def init_state():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("data_loaded", False)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("rag_index", None)
    st.session_state.setdefault("rag_texts", [])
    st.session_state.setdefault("data_summary", "")
