import math, datetime as dt, numpy as np, pandas as pd

def _json_default(o):
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)): return o.isoformat()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating):
        x = float(o)
        if math.isnan(x) or math.isinf(x): return None
        return x
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def log_interaction(...):
    ...
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
