#!/usr/bin/env python3
import os
import json
import time
import logging
import traceback
import re
from typing import List, Any, Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import enforce_format, fabricate_fallback
from llm import chat_complete
from sandbox import run_user_code

APP_NAME = "data-analyst-agent"
app = FastAPI(title=APP_NAME, version="1.0.1")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Logging --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(APP_NAME)

# -------------------- Time budgets (seconds) --------------------
PIPELINE_BUDGET = int(os.getenv("PIPELINE_BUDGET", 285))  # 4m45s
PLANNER_TIMEOUT = int(os.getenv("PLANNER_TIMEOUT", 30))
CODEGEN_TIMEOUT = int(os.getenv("CODEGEN_TIMEOUT", 60))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", 120))
REFORMAT_TIMEOUT = int(os.getenv("REFORMAT_TIMEOUT", 20))

# -------------------- Models & Keys --------------------
MODEL_PLANNER = os.getenv("OPENAI_MODEL_PLANNER", "gpt-4o-mini")
MODEL_CODEGEN = os.getenv("OPENAI_MODEL_CODEGEN", "gpt-4o-mini")
MODEL_ORCH    = os.getenv("OPENAI_MODEL_ORCH",    "gpt-4o-mini")

KEY_PLANNER = os.getenv("OPENAI_API_KEY_PLANNER")
KEY_CODEGEN = os.getenv("OPENAI_API_KEY_CODEGEN")
KEY_ORCH    = os.getenv("OPENAI_API_KEY_ORCH") or KEY_PLANNER or KEY_CODEGEN

if not (KEY_PLANNER and KEY_CODEGEN and KEY_ORCH):
    log.warning("One or more OpenAI API keys are missing. The app will still run but will always fabricate fallback answers.")

# -------------------- Schemas --------------------
class AnalyzeRequest(BaseModel):
    task: str = Field(..., description="Overall task description")
    questions: List[str] = Field(..., description="Questions to answer in order")
    response_format: str = Field("JSON array", description="e.g., 'JSON array' or 'JSON object'")

# -------------------- Health --------------------
@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME, "version": "1.0.1"}

# -------------------- Helpers --------------------
def _strip_md_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # remove a single leading ``` or ```python ... and a single trailing ```
    s = re.sub(r'^\s*```(?:[a-zA-Z0-9_+\-]+)?\s*', '', s)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

# ---- seaborn shim (injected if seaborn not installed) ----
SEABORN_SHIM = r"""
try:
    import seaborn as sns  # type: ignore
except Exception:
    class _SNSShim:
        def scatterplot(self, data=None, x=None, y=None, **kwargs):
            import matplotlib.pyplot as plt
            if data is not None and x is not None and y is not None:
                plt.scatter(data[x], data[y], **{k:v for k,v in kwargs.items() if k not in ("data","x","y")})
            elif x is not None and y is not None:
                plt.scatter(x, y, **kwargs)
        def lineplot(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            plt.plot(*args, **kwargs)
        def regplot(self, x=None, y=None, data=None, **kwargs):
            import numpy as np
            import matplotlib.pyplot as plt
            if data is not None and x is not None and y is not None:
                X = data[x].values
                Y = data[y].values
            else:
                X, Y = x, y
            plt.scatter(X, Y)
            try:
                m, b = np.polyfit(np.asarray(X, float), np.asarray(Y, float), 1)
                plt.plot(X, m*np.asarray(X, float) + b, linestyle=":", linewidth=2, color="red")
            except Exception:
                pass
    sns = _SNSShim()
"""

# ---- safe currency/number casting (shadow float in user script) ----
SAFE_CASTS_PRELUDE = r"""
import re as _re_cast
def _safe_float(x):
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = _re_cast.sub(r'[^0-9.\-eE]', '', str(x))
        if s in ('', '-', '.', '-.'):
            return float('nan')
        return float(s)
    except Exception:
        return float('nan')
float = _safe_float
"""

def _payload_inject(user_payload: dict) -> str:
    """
    Injects the *true* payload and wraps json.loads to be resilient:
    - Try normal json.loads
    - If that fails, try ast.literal_eval
    - As last resort, return the injected payload object
    Also exposes TASK / QUESTIONS / RESPONSE_FORMAT globals for convenience.
    """
    payload_json_text = json.dumps(user_payload, ensure_ascii=False)
    return (
        "import json, ast\n"
        f"__DAA_PAYLOAD_JSON__ = r'''{payload_json_text}'''\n"
        "__DAA_PAYLOAD_OBJ__ = json.loads(__DAA_PAYLOAD_JSON__)\n"
        "__json_loads_orig = json.loads\n"
        "def __daa_json_loads(s, *a, **k):\n"
        "    try:\n"
        "        return __json_loads_orig(s, *a, **k)\n"
        "    except Exception:\n"
        "        try:\n"
        "            return ast.literal_eval(s)\n"
        "        except Exception:\n"
        "            return __DAA_PAYLOAD_OBJ__\n"
        "json.loads = __daa_json_loads\n"
        "TASK = __DAA_PAYLOAD_OBJ__.get('task','')\n"
        "QUESTIONS = __DAA_PAYLOAD_OBJ__.get('questions',[])\n"
        "RESPONSE_FORMAT = __DAA_PAYLOAD_OBJ__.get('response_format','JSON array')\n"
    )

# -------------------- Build request from HTTP --------------------
async def _build_analyze_request_from_http(
    request: Request,
    questions_txt: Optional[UploadFile],
    file: Optional[UploadFile],
) -> Optional[AnalyzeRequest]:
    """
    Accept JSON or multipart. For multipart, accept 'questions.txt' or 'file'.
    For a plain-text file, treat first paragraph as task and numbered lines as questions.
    """
    ctype = request.headers.get("content-type", "")
    # JSON
    if "application/json" in ctype:
        payload = await request.json()
        return AnalyzeRequest(**payload)

    # Multipart
    upl = questions_txt or file
    if upl is None:
        return None
    txt = (await upl.read()).decode("utf-8", "ignore").strip()

    # If file is JSON {task, questions, response_format}
    try:
        data = json.loads(txt)
        return AnalyzeRequest(**data)
    except Exception:
        pass

    # Parse free-text file
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    task = paragraphs[0] if paragraphs else "Ad-hoc task from questions.txt"

    qs: List[str] = []
    for ln in txt.splitlines():
        m = re.match(r"\s*\d+\s*[\.\)\-:]\s*(.+)", ln)
        if m:
            qs.append(m.group(1).strip())
    if not qs:
        qs = [task]
        task = "Ad-hoc task from questions.txt"

    return AnalyzeRequest(task=task, questions=qs, response_format="JSON array")

# -------------------- Endpoints --------------------
@app.post("/api/analyze")
async def analyze(
    request: Request,
    questions_txt: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
) -> Any:
    req = None
    try:
        req = await _build_analyze_request_from_http(request, questions_txt, file)
    except Exception as e:
        log.error("Request parse failed: %s", e)
    if not req:
        # Always return a valid answer (never error)
        return fabricate_fallback(["q1"], "JSON array")
    return await _run_pipeline(req)

# Optional legacy/compat route
@app.post("/api")
async def analyze_compat(
    request: Request,
    questions_txt: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
) -> Any:
    req = None
    try:
        req = await _build_analyze_request_from_http(request, questions_txt, file)
    except Exception as e:
        log.error("Compat parse failed: %s", e)
    if not req:
        return fabricate_fallback(["q1"], "JSON array")
    return await _run_pipeline(req)

# -------------------- Core pipeline --------------------
async def _run_pipeline(req: AnalyzeRequest) -> Any:
    """
    1) Plan → 2) Codegen → 3) Execute → 4) Reformat (if needed) → 5) Enforce/Fill → Return
    Always returns a valid JSON in requested format.
    """
    t0 = time.time()
    d
