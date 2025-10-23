# InsightForge

A fresh, working build with:
- Persistent chat memory per user
- Vector RAG over docs/ via FAISS
- Robust JSON-safe serialization
- Filters + advanced visuals
- Logging of Q/A + retrieved context

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
streamlit run app.py
```

## Deploy
- Keep OPENAI_API_KEY in Streamlit **Secrets**.
- This code reads secrets first
