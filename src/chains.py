from __future__ import annotations
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from .prompts import SYSTEM_PROMPT, QUESTION_TO_RETRIEVAL_PROMPT, ANSWER_PROMPT
from .retriever import StatsRetriever

class InsightForgeAssistant:
    def __init__(self, df, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.retriever = StatsRetriever(df)

        self.plan_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(QUESTION_TO_RETRIEVAL_PROMPT)
        )
        self.answer_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(ANSWER_PROMPT)
        )

    def _plan(self, question: str) -> Dict[str, Any]:
        plan_text = self.plan_chain.run(question=question)
        try:
            plan = json.loads(plan_text)
        except Exception:
            plan = {"frames_needed": ["sales_by_month","product_performance","regional_performance","customer_segments"]}
        return plan

    def _retrieve(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # In this simple version we ignore the plan and return all frames.
        return self.retriever.retrieve(question="")

    def answer(self, question: str) -> str:
        plan = self._plan(question)
        stats = self._retrieve(plan)
        answer = self.answer_chain.run(question=question, retrieved_stats=json.dumps(stats))
        return answer
