# InsightForge — AI-Powered Business Intelligence Assistant

InsightForge turns raw business data into insights and visuals using LLMs + a lightweight RAG pipeline.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
streamlit run app.py
```

## Inside
- `data/sample_sales.csv` — synthetic data
- `app.py` — Streamlit UI
- `src/retriever.py` — pandas-based stats retriever
- `src/prompts.py` — system + answer prompts
- `src/chains.py` — LangChain chat with memory + RAG
- `src/viz.py` — plotting helpers
- `src/evaluator.py` — Q/A eval with QAEvalChain
- `tests/test_basic.py` — basic sanity test
