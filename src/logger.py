from __future__ import annotations
import os, json, time, math, datetime as dt
from pathlib import Path
from typing import Any, Dict
import numpy as np

LOG_DIR = Path(os.getenv("INSIGHTFORGE_LOG_DIR", "storage/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _json_default(o):
    import pandas as pd
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)): return o.isoformat()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating):
        x = float(o)
        if math.isnan(x) or math.isinf(x): return None
        return x
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

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
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
