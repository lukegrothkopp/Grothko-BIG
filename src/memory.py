from __future__ import annotations
import json, os
from pathlib import Path
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MEMORY_DIR = Path(os.getenv("INSIGHTFORGE_MEMORY_DIR", "storage/memory"))

def _serialize_messages(messages):
    out = []
    for m in messages:
        role = "system"
        if hasattr(m, "type"):
            if m.type == "human":
                role = "human"
            elif m.type == "ai":
                role = "ai"
        elif m.__class__.__name__ == "HumanMessage":
            role = "human"
        elif m.__class__.__name__ == "AIMessage":
            role = "ai"
        out.append({"role": role, "content": m.content})
    return out

def _deserialize_messages(msgs: List[dict]):
    out = []
    for d in msgs:
        role = d.get("role", "system")
        content = d.get("content", "")
        if role == "human":
            out.append(HumanMessage(content=content))
        elif role == "ai":
            out.append(AIMessage(content=content))
        else:
            out.append(SystemMessage(content=content))
    return out

def load_memory(user_id: str) -> ConversationBufferMemory:
    mem = ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        output_key="text",
        return_messages=True,
    )
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{user_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            messages = _deserialize_messages(data.get("messages", []))
            for m in messages:
                if isinstance(m, HumanMessage):
                    mem.chat_memory.add_user_message(m.content)
                elif isinstance(m, AIMessage):
                    mem.chat_memory.add_ai_message(m.content)
        except Exception:
            pass
    return mem

def save_memory(user_id: str, memory: ConversationBufferMemory) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{user_id}.json"
    data = {"messages": _serialize_messages(memory.chat_memory.messages)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def reset_memory(user_id: str) -> None:
    path = MEMORY_DIR / f"{user_id}.json"
    if path.exists():
        path.unlink()
