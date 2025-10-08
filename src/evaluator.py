from __future__ import annotations
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain

def evaluate_qa(examples: List[Dict[str,str]], predictions: List[Dict[str,str]]):
    """Minimal evaluation wrapper.
    examples: list of {'question':..., 'answer':...} gold references
    predictions: list of {'question':..., 'answer':...} model outputs
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_chain = QAEvalChain.from_llm(llm)
    graded = eval_chain.evaluate(examples, predictions, question_key="question", prediction_key="answer", answer_key="answer")
    return graded
