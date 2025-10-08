import math
import datetime as dt
import numpy as np
import pandas as pd

from __future__ import annotations
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .prompts import SYSTEM_PROMPT, QUESTION_TO_RETRIEVAL_PROMPT, ANSWER_PROMPT
from .retriever import StatsRetriever
from .memory import load_memory, save_memory
from .logger import log_interaction
from .doc_index import retrieve as doc_retrieve

def _json_default(o):
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        x = float(o)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)  # last-resort fallback

class InsightForgeAssistant:
    def __init__(self, df, user_id: str = "default", model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.memory = load_memory(user_id)  # memory_key='history'
        self.user_id = user_id
        self.retriever = StatsRetriever(df)

        self.plan_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(QUESTION_TO_RETRIEVAL_PROMPT),
            memory=self.memory,
            verbose=False,
        )
        self.answer_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(ANSWER_PROMPT),
            memory=self.memory,
            verbose=False,
        )

    def _plan(self, question: str) -> Dict[str, Any]:
        out = self.plan_chain.invoke({"question": question})
        plan_text = out.get("text", str(out))
        try:
            plan = json.loads(plan_text)
            if not isinstance(plan, dict):
                raise ValueError("Plan is not a JSON object")
        except Exception:
            plan = {"frames_needed": ["sales_by_month","product_performance","regional_performance","customer_segments"]}
        return plan

    def _retrieve_stats(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return self.retriever.retrieve(question="")

    def _retrieve_docs(self, question: str, k: int = 4) -> List[str]:
        return doc_retrieve(question, k=k)

    from .logger import log_interaction
    try:
        safe_stats = json.loads(safe_stats_str)   # round-trip to ensure purity
    except Exception:
        safe_stats = stats
        log_interaction(self.user_id, question, answer, safe_stats, docs)

    def answer(self, question: str) -> str:
        plan = self._plan(question)
        stats = self._retrieve_stats(plan)
        docs = self._retrieve_docs(question, k=4)
        out = self.answer_chain.invoke({
        safe_stats_str = json.dumps(stats, default=_json_default)
        out = self.answer_chain.invoke({
            "question": question,
            "retrieved_stats": safe_stats_str,
            "doc_snippets": "\n\n---\n".join(docs) if docs else "None",
        })
        answer = out.get("text", str(out))
        save_memory(self.user_id, self.memory)
        log_interaction(self.user_id, question, answer, stats, docs)
        return answer

    
