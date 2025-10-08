from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path(os.getenv("INSIGHTFORGE_LOG_DIR", "storage/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_interaction(user_id: str, question: str, answer: str, retrieved_stats: Dict[str, Any], retrieved_docs: list):
    record = {
        "ts": int(time.time()),
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "retrieved_stats": retrieved_stats,
        "retrieved_docs": retrieved_docs,
    }
    fpath = LOG_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
