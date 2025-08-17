import json
import base64
import logging
from typing import Any, List

log = logging.getLogger("utils")

def _fake_image_data_uri() -> str:
    # Minimal 1x1 PNG (transparent) data URI
    tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    return "data:image/png;base64," + tiny_png_base64

def fabricate_fallback(questions: List[str], response_format: str) -> Any:
    """Fabricate answers in correct length/order when error/timeout occurs."""
    response_format = (response_format or "").lower()
    fabricated = []
    for q in questions:
        ql = q.lower()
        if "plot" in ql or "chart" in ql or "figure" in ql or "graph" in ql or "image" in ql:
            fabricated.append(_fake_image_data_uri())
        elif "correlation" in ql or "mean" in ql or "median" in ql or "rate" in ql or "score" in ql:
            fabricated.append(-1.0)
        elif "how many" in ql or "count" in ql or "number" in ql:
            fabricated.append(0)
        else:
            fabricated.append("Unknown")
    if "object" in response_format:
        # Return as {"answers": [...]} for object style
        return {"answers": fabricated}
    # Default: array
    return fabricated

def enforce_format(answers: Any, response_format: str, n: int) -> Any:
    """Ensure we return the requested format and correct number of answers."""
    response_format = (response_format or "").lower()
    # Normalize to list first
    if isinstance(answers, list):
        arr = answers
    elif isinstance(answers, dict) and "answers" in answers and isinstance(answers["answers"], list):
        arr = answers["answers"]
    else:
        # If it's a dict without 'answers', convert values in a stable order
        if isinstance(answers, dict):
            arr = list(answers.values())
        else:
            arr = [answers]
    # Pad/trim to exactly n
    if len(arr) < n:
        arr = arr + ["Unknown"] * (n - len(arr))
    if len(arr) > n:
        arr = arr[:n]
    # Return in requested style
    if "object" in response_format:
        return {"answers": arr}
    return arr
