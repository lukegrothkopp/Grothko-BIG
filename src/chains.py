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
        # Explicitly pass inputs; get text safely
        out = self.plan_chain.invoke({"question": question})
        plan_text = out["text"] if isinstance(out, dict) and "text" in out else str(out)
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

    def answer(self, question: str) -> str:
        plan = self._plan(question)
        stats = self._retrieve_stats(plan)
        docs = self._retrieve_docs(question, k=4)
        out = self.answer_chain.invoke({
            "question": question,
            "retrieved_stats": json.dumps(stats),
            "doc_snippets": "\n\n---\n".join(docs) if docs else "None",
        })
        answer = out["text"] if isinstance(out, dict) and "text" in out else str(out)
        save_memory(self.user_id, self.memory)
        log_interaction(self.user_id, question, answer, stats, docs)
        return answer
