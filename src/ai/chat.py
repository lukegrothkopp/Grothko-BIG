from openai import OpenAI

def plain_chat(client: OpenAI, model: str, prompt: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful BI assistant."},
        {"role": "user", "content": prompt},
    ]
    out = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)
    return out.choices[0].message.content
