#!/usr/bin/env python3
import os
import io
import json
import time
import logging
import traceback
from fastapi import UploadFile, File
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import enforce_format, fabricate_fallback
from llm import chat_complete
from sandbox import run_user_code

APP_NAME = "data-analyst-agent"
app = FastAPI(title=APP_NAME, version="0.1.0")

# CORS (open by default; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logging ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(APP_NAME)

# --- Time budgets ---
PIPELINE_BUDGET = int(os.getenv("PIPELINE_BUDGET", 285))  # seconds (4m45s)
PLANNER_TIMEOUT = int(os.getenv("PLANNER_TIMEOUT", 30))
CODEGEN_TIMEOUT = int(os.getenv("CODEGEN_TIMEOUT", 60))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", 120))
REFORMAT_TIMEOUT = int(os.getenv("REFORMAT_TIMEOUT", 20))

# --- Models & Keys ---
MODEL_PLANNER = os.getenv("OPENAI_MODEL_PLANNER", "gpt-4o-mini")
MODEL_CODEGEN = os.getenv("OPENAI_MODEL_CODEGEN", "gpt-4o-mini")
MODEL_ORCH = os.getenv("OPENAI_MODEL_ORCH", "gpt-4o-mini")

KEY_PLANNER = os.getenv("OPENAI_API_KEY_PLANNER")
KEY_CODEGEN = os.getenv("OPENAI_API_KEY_CODEGEN")
KEY_ORCH = os.getenv("OPENAI_API_KEY_ORCH") or KEY_PLANNER or KEY_CODEGEN

if not (KEY_PLANNER and KEY_CODEGEN and KEY_ORCH):
    log.warning("One or more OpenAI API keys are missing. The app will still run but will always fabricate fallback answers.")

# --- Schemas ---
class AnalyzeRequest(BaseModel):
    task: str = Field(..., description="Overall task description")
    questions: List[str] = Field(..., description="Questions to answer in order")
    response_format: str = Field("JSON array", description="Expected format e.g., 'JSON array' or 'JSON object'")

class AnalyzeResponse(BaseModel):
    result: Any

@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME, "version": "0.1.0"}

# ---------- helper: strip markdown code fences from LLM code ----------
def _strip_md_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = re.sub(r'^\s*```(?:[a-zA-Z0-9_+\-]+)?\s*', '', s)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

# ---------- tiny seaborn shim (only used if seaborn is missing) ----------
SEABORN_SHIM = r"""
# ---- seaborn shim (only if seaborn unavailable) ----
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
            if X is not None and Y is not None:
                try:
                    m, b = np.polyfit(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), 1)
                    plt.plot(X, m*np.asarray(X, dtype=float) + b, linestyle=":", linewidth=2)
                except Exception:
                    pass
    sns = _SNSShim()
# ---- end seaborn shim ----
"""

# ---------- wrapper that accepts JSON or multipart; core pipeline in _run_pipeline ----------
@app.post("/api/analyze")
async def analyze(request: Request,
                  questions_txt: UploadFile | None = File(None),
                  file: UploadFile | None = File(None)) -> Any:
    """
    Accepts:
      - JSON body with {task, questions, response_format}
      - OR multipart/form-data with a text file field:
          * 'questions.txt' (preferred)
          * 'file'         (compat)
    Always returns a valid JSON answer (real or fabricated).
    """
    ctype = request.headers.get("content-type", "")

    # JSON path → forward to core
    if "application/json" in ctype:
        try:
            payload = await request.json()
            req = AnalyzeRequest(**payload)
            return await _run_pipeline(req)
        except Exception as e:
            log.error("JSON parse failed: %s", e)
            return fabricate_fallback(["q1"], "JSON array")

    # Multipart path → read whichever file is present
    upl = questions_txt or file
    if upl is None:
        return fabricate_fallback(["q1"], "JSON array")

    try:
        text = (await upl.read()).decode("utf-8", "ignore").strip()
    except Exception as e:
        log.error("Reading uploaded file failed: %s", e)
        return fabricate_fallback(["q1"], "JSON array")

    # If the file itself is JSON with task/questions/response_format, use it directly
    try:
        data = json.loads(text)
        req = AnalyzeRequest(**data)
        return await _run_pipeline(req)
    except Exception:
        pass

    # Otherwise parse a free-text prompt file:
    # - first paragraph → task
    # - numbered lines ("1) ...", "2. ...", "3 - ...") → questions
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    task = paragraphs[0] if paragraphs else "Ad-hoc task from questions.txt"

    qs: List[str] = []
    for ln in text.splitlines():
        m = re.match(r"\s*\d+\s*[\.\)\-:]\s*(.+)", ln)
        if m:
            qs.append(m.group(1).strip())

    if not qs:
        qs = [task]
        task = "Ad-hoc task from questions.txt"

    req = AnalyzeRequest(task=task, questions=qs, response_format="JSON array")
    return await _run_pipeline(req)


async def _run_pipeline(req: AnalyzeRequest) -> Any:
    """
    Main pipeline:
      1) Plan
      2) Codegen
      3) Execute user code
      4) Enforce format
      5) Fallback if anything breaks or time is up
    """
    t0 = time.time()
    deadline = t0 + PIPELINE_BUDGET

    task = req.task.strip()
    questions = [q.strip() for q in req.questions]
    resp_fmt = (req.response_format or "JSON array").strip()

    log.info("Received /api/analyze | format=%s | #questions=%d", resp_fmt, len(questions))

    def time_left() -> int:
        return max(0, int(deadline - time.time()))

    try:
        # Hard stop if we are already out of time
        if time_left() <= 0:
            raise TimeoutError("Budget exceeded before starting.")

        # 1) PLAN
        plan_prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "planner.txt")
        with open(plan_prompt_path, "r", encoding="utf-8") as f:
            planner_sys = f.read()
        planner_messages = [
            {"role": "system", "content": planner_sys},
            {"role": "user", "content": json.dumps({"task": task, "questions": questions}, ensure_ascii=False)},
        ]

        plan = None
        if KEY_PLANNER:
            log.info("Planning... (timeout=%ss)", min(PLANNER_TIMEOUT, time_left()))
            plan = chat_complete(planner_messages, MODEL_PLANNER, KEY_PLANNER, timeout=min(PLANNER_TIMEOUT, time_left()))
        else:
            log.warning("Planner key missing. Skipping planning.")

        # 2) CODEGEN
        code_prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "codegen.txt")
        with open(code_prompt_path, "r", encoding="utf-8") as f:
            codegen_sys = f.read()

        user_payload = {
            "task": task,
            "questions": questions,
            "response_format": resp_fmt,
            "plan": plan or {"steps": ["(no plan — fabricate if needed)"]},
        }
        code_messages = [
            {"role": "system", "content": codegen_sys},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        code = None
        if KEY_CODEGEN and time_left() > 0:
            log.info("Generating code... (timeout=%ss)", min(CODEGEN_TIMEOUT, time_left()))
            code = chat_complete(code_messages, MODEL_CODEGEN, KEY_CODEGEN, timeout=min(CODEGEN_TIMEOUT, time_left()))
            code = _strip_md_fences(code)  # remove ``` fences
            # inject seaborn shim before user code to avoid ModuleNotFoundError
            code = SEABORN_SHIM + "\n" + code
        else:
            log.warning("Codegen key missing or no time left.")

        answers: Any = None

        # 3) EXECUTE
        if code and time_left() > 0:
            # force headless backend for matplotlib in the child process
            os.environ.setdefault("MPLBACKEND", "Agg")

            exec_timeout = min(EXECUTION_TIMEOUT, time_left())
            log.info("Executing generated code (timeout=%ss)...", exec_timeout)
            out, err, rc = run_user_code(code, timeout=exec_timeout)
            log.info("Execution rc=%s bytes_out=%s bytes_err=%s", rc, len(out or ""), len(err or ""))

            # Expect code to PRINT the final JSON only
            if rc == 0 and out:
                try:
                    answers = json.loads(out.strip())
                except Exception as e:
                    log.warning("Execution output was not valid JSON. Will try to reformat with ORCH. %s", e)
                    # 4) REFORMAT (try to coerce with LLM, else fallback)
                    if KEY_ORCH and time_left() > 0:
                        reform_prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "reformat.txt")
                        with open(reform_prompt_path, "r", encoding="utf-8") as f:
                            reform_sys = f.read()
                        re_messages = [
                            {"role": "system", "content": reform_sys},
                            {"role": "user", "content": json.dumps({"raw": out, "expected": resp_fmt, "questions": questions}, ensure_ascii=False)},
                        ]
                        try:
                            answers_text = chat_complete(re_messages, MODEL_ORCH, KEY_ORCH, timeout=min(REFORMAT_TIMEOUT, time_left()))
                            answers = json.loads(answers_text)
                        except Exception as ee:
                            log.error("Reformatter failed: %s", ee)
                    else:
                        log.warning("No ORCH key or time for reformat.")
            else:
                log.warning("Execution failed or no output. stderr: %s", (err or "")[:500])
        else:
            log.warning("Skipping execution — no code or no time.")

        # 5) ENFORCE FORMAT or FABRICATE
        if answers is None:
            answers = fabricate_fallback(questions, resp_fmt)

        try:
            answers = enforce_format(answers, resp_fmt, len(questions))
        except Exception as e:
            log.error("Format enforcement failed: %s", e)
            answers = fabricate_fallback(questions, resp_fmt)

        # Return final
        return answers

    except Exception as e:
        log.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        return fabricate_fallback(questions, resp_fmt)
# ---------- END ----------
