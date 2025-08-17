import os
import json
import logging
import requests

log = logging.getLogger("llm")

OPENAI_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

def chat_complete(messages, model, api_key, timeout=20) -> str:
    """
    Minimal client using HTTP to avoid SDK version mismatches.
    Returns content string from the first choice.
    """
    try:
        url = f"{OPENAI_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        log.error("chat_complete failed: %s", e)
        # Return empty so caller can fabricate
        return ""
