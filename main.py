"""
mse_ai_api v2 - DuckDuckGo AI Backend
- No login, no API key; uses headless Chromium (Playwright) for DDG's JS challenge
- OpenAI-compatible API (/v1/chat/completions, /v1/responses)
- Web dashboard at /dashboard
- Works on VPS, Docker (install Chromium via Playwright; see Dockerfile)
"""
import os
import uuid
import time
import json
import re
import asyncio
import secrets
from typing import Optional

from fastapi import FastAPI, Request, Response, Cookie
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import chatgpt_client as ai
import stats
import keepalive

# ── Config ────────────────────────────────────────────────────────────────────
API_SECRET_KEY    = os.getenv("API_SECRET_KEY", "change-secret-key-2026")
DASHBOARD_PIN     = os.getenv("DASHBOARD_PIN", "281020")

# In-memory session store  {token: expiry_timestamp}
_sessions: dict[str, float] = {}
SESSION_TTL = 8 * 3600  # 8 hours

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(keepalive.start_keepalive())
    yield
    await ai.shutdown_playwright()


app = FastAPI(title="mse_ai_api", version="2.0.0", docs_url="/docs", redoc_url=None, lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ══════════════════════════════════════════════════════════════════════════════
# Auth & Prompt Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _check_auth(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    return auth.replace("Bearer ", "").strip() == API_SECRET_KEY


# ── Session helpers ───────────────────────────────────────────────────────────

def _create_session() -> str:
    token = secrets.token_hex(32)
    _sessions[token] = time.time() + SESSION_TTL
    return token


def _is_valid_session(token: str | None) -> bool:
    if not token:
        return False
    expiry = _sessions.get(token)
    if not expiry or time.time() > expiry:
        _sessions.pop(token, None)
        return False
    return True


def _require_dashboard_auth(request: Request):
    """Returns True if request has a valid dashboard session cookie."""
    token = request.cookies.get("dash_session")
    return _is_valid_session(token)


def _messages_to_prompt(messages: list) -> tuple[str, list]:
    """
    Convert OpenAI-style messages list into:
    - A single prompt string (last user message)
    - A history list for multi-turn context
    """
    history = []
    prompt = ""
    system_prefix = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Flatten list content
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )

        if role == "system":
            system_prefix = content
        elif role == "user":
            prompt = content
            if history or system_prefix:
                history.append({"role": "user", "content": content})
        elif role == "assistant":
            history.append({"role": "assistant", "content": content})

    # Prepend system message to first user message if present
    if system_prefix and history:
        for h in history:
            if h["role"] == "user":
                h["content"] = f"[System: {system_prefix}]\n\n{h['content']}"
                break

    # Final prompt = last user message (possibly with system prefix if no history)
    if system_prefix and not history:
        prompt = f"[System: {system_prefix}]\n\n{prompt}"

    return prompt, history[:-1] if history else []


def _effective_tools(data: dict) -> Optional[list]:
    """
    Only a non-empty JSON array counts as \"tools\". Empty [] is ignored so behaviour matches
    the dashboard (plain chat). If n8n sends real tools, tool instructions are appended.
    """
    raw = data.get("tools")
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    return raw


def _strip_degenerate_tool_json(text: str) -> str:
    """If the entire reply is only an empty tool_calls JSON, treat as no content."""
    if not text or not text.strip():
        return text
    raw = text.strip()
    t = raw
    if "```" in t:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    try:
        o = json.loads(t)
        if isinstance(o, dict) and "tool_calls" in o and o.get("tool_calls") == []:
            if set(o.keys()) <= {"tool_calls"}:
                return ""
    except (json.JSONDecodeError, TypeError):
        pass
    if re.fullmatch(r'\{\s*"tool_calls"\s*:\s*\[\s*\]\s*\}', t):
        return ""
    return raw


def _was_degenerate_tool_only(text: str) -> bool:
    s = text.strip()
    return bool(s) and _strip_degenerate_tool_json(text) == ""


async def _ask_with_tool_fallback(
    base_prompt: str,
    history: list,
    model: str,
    tools_effective: Optional[list],
) -> str:
    """Match dashboard behaviour: plain chat unless real tools are bound."""
    prompt = base_prompt + (_format_tools_instruction(tools_effective) if tools_effective else "")
    text = await ai.ask(prompt, model=model, history=history)
    if _was_degenerate_tool_only(text):
        text = await ai.ask(base_prompt, model=model, history=history)
    out = _strip_degenerate_tool_json(text)
    if not out.strip():
        text = await ai.ask(base_prompt, model=model, history=history)
        out = _strip_degenerate_tool_json(text) or text.strip()
    return out.strip() or "(Empty reply from model; try disabling Tools in the n8n OpenAI node.)"


def _format_tools_instruction(tools: list) -> str:
    out = "\n=== TOOL USAGE (when applicable) ===\n"
    out += "If a listed tool clearly applies, respond with ONLY valid JSON:\n"
    out += '{"tool_calls": [{"name": "TOOL_NAME", "arguments": {"param": "value"}}]}\n'
    out += "If no tool applies, reply with normal plain text only (do NOT output JSON with an empty tool_calls array).\n\n"
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        out += f"Tool: {name}\nDescription: {desc}\n"
        if params.get("properties"):
            req = params.get("required", [])
            for pn, pi in params["properties"].items():
                out += f"  - {pn} ({'required' if pn in req else 'optional'}): {pi.get('description','')}\n"
        out += "\n"
    out += "=== END OF TOOLS ===\n\n"
    return out


def _parse_tool_calls(text: str) -> Optional[list]:
    cleaned = text.strip()
    if "```" in cleaned:
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    candidates = [cleaned]
    m = re.search(r'\{[\s\S]*"tool_calls"[\s\S]*\}', cleaned)
    if m:
        candidates.append(m.group(0))

    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                raw = parsed["tool_calls"]
                if isinstance(raw, list) and raw:
                    return [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": call.get("name", ""),
                                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)
                                if isinstance(call.get("arguments"), dict)
                                else str(call.get("arguments", "{}"))
                            }
                        }
                        for call in raw
                    ]
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return None


def _build_completion(data: dict, text: str, start: float, tool_calls=None) -> dict:
    p_tok = max(1, len(text.split()) // 2)
    c_tok = max(1, len(text.split()))
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(start),
        "model": data.get("model", "gpt-4o-mini"),
        "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
    }
    if tool_calls:
        base["choices"] = [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": tool_calls}, "finish_reason": "tool_calls"}]
    else:
        base["choices"] = [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Core API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root(request: Request):
    # Browsers get redirected to dashboard, API clients get JSON
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/dashboard", status_code=302)
    return {"status": "running", "message": "mse_ai_api v2 (DDG backend)", "dashboard": "/dashboard"}


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "DuckDuckGo AI (no login required)"}


@app.get("/v1/models")
async def list_models():
    models = ai.get_available_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": m.get("created", 1735689600),
                "owned_by": m["provider"],
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _check_auth(request):
        return JSONResponse(status_code=401, content={"error": {"message": "Invalid API Key"}})
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON"}})

    messages = data.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages required"}})

    tools_effective = _effective_tools(data)
    model = data.get("model", "auto")
    start = time.time()

    try:
        base_prompt, history = _messages_to_prompt(messages)

        response_text = await _ask_with_tool_fallback(
            base_prompt, history, model, tools_effective
        )

        tool_calls = _parse_tool_calls(response_text) if tools_effective else None
        result = _build_completion(data, response_text, start, tool_calls)
        u = result["usage"]
        stats.record(True, u["prompt_tokens"], u["completion_tokens"], model)
        return result
    except Exception as e:
        stats.record(False, error=str(e), model=model)
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


@app.post("/v1/responses")
async def responses_api(request: Request):
    if not _check_auth(request):
        return JSONResponse(status_code=401, content={"error": {"message": "Invalid API Key"}})
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON"}})

    input_data = data.get("input", "")
    if isinstance(input_data, str):
        messages = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        messages = input_data
    else:
        messages = data.get("messages", [])

    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "input required"}})

    instructions = data.get("instructions", "")
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    tools_effective = _effective_tools(data)
    model = data.get("model", "auto")
    start = time.time()

    try:
        base_prompt, history = _messages_to_prompt(messages)

        response_text = await _ask_with_tool_fallback(
            base_prompt, history, model, tools_effective
        )

        tool_calls = _parse_tool_calls(response_text) if tools_effective else None
        p_tok = max(1, len((base_prompt + (_format_tools_instruction(tools_effective) if tools_effective else "")).split()))
        c_tok = max(1, len(response_text.split()))
        stats.record(True, p_tok, c_tok, model)

        if tool_calls:
            output = [
                {"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                 "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"}
                for tc in tool_calls
            ]
        else:
            output = [{"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": response_text}]}]

        return {
            "id": f"resp-{uuid.uuid4().hex[:29]}",
            "object": "response", "created_at": int(start),
            "model": model, "status": "completed", "output": output,
            "usage": {"input_tokens": p_tok, "output_tokens": c_tok, "total_tokens": p_tok + c_tok}
        }
    except Exception as e:
        stats.record(False, error=str(e), model=model)
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard Login
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    pin = form.get("pin", "").strip()
    if pin == DASHBOARD_PIN:
        token = _create_session()
        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(
            key="dash_session", value=token,
            max_age=SESSION_TTL, httponly=True, samesite="lax"
        )
        return response
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "رقم سري خاطئ، حاول مرة أخرى"},
        status_code=401
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("dash_session")
    return response


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not _require_dashboard_auth(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/dashboard/stats")
async def dashboard_stats(request: Request):
    if not _require_dashboard_auth(request):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return stats.get_stats()


@app.post("/dashboard/credentials")
async def dashboard_save_credentials(request: Request):
    if not _require_dashboard_auth(request):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})

    api_key = data.get("api_key", "").strip()
    if api_key:
        global API_SECRET_KEY
        API_SECRET_KEY = api_key
        os.environ["API_SECRET_KEY"] = api_key

    return {"ok": True, "message": "Saved"}


@app.post("/dashboard/test")
async def dashboard_test(request: Request):
    if not _require_dashboard_auth(request):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    prompt = data.get("prompt", "").strip()
    model  = data.get("model", "gpt-4o-mini")
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "prompt required"})

    try:
        response = await ai.ask(prompt, model=model)
        return {"response": response, "model": model}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7777))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
