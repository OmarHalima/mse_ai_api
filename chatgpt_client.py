"""
DuckDuckGo AI Chat Client
- Calls DDG API directly with httpx (no external library needed)
- Handles x-vqd-4 AND x-vqd-hash-1 tokens automatically
- Retry logic for rate limits and token errors
"""
import asyncio
import json
import httpx

# ── Model name mapping ────────────────────────────────────────────────────────
MODEL_MAP = {
    "gpt-5-mini":               "gpt-5-mini",
    "gpt5-mini":                "gpt-5-mini",
    "auto":                     "gpt-5-mini",
    "gpt-4o-mini":              "gpt-4o-mini",
    "gpt-4o":                   "gpt-4o-mini",
    "gpt-oss-120b":             "gpt-oss-120b",
    "gpt-oss":                  "gpt-oss-120b",
    "llama-4-scout":            "llama-4-scout",
    "llama4":                   "llama-4-scout",
    "llama":                    "llama-4-scout",
    "llama-3.3-70b":            "llama-4-scout",
    "claude-haiku-4.5":         "claude-3-haiku-20240307",
    "claude-haiku":             "claude-3-haiku-20240307",
    "claude-3-haiku":           "claude-3-haiku-20240307",
    "claude-3-haiku-20240307":  "claude-3-haiku-20240307",
    "mistral-small-3":          "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistral":                  "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral-8x7b":             "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral":                  "mistralai/Mistral-Small-24B-Instruct-2501",
}

MAX_RETRIES = 3

DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_CHAT_URL   = "https://duckduckgo.com/duckchat/v1/chat"

# Updated headers matching current Chrome browser requests
HEADERS_BASE = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept":          "text/event-stream",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control":   "no-cache",
    "Pragma":          "no-cache",
    "Origin":          "https://duckduckgo.com",
    "Referer":         "https://duckduckgo.com/",
}


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


def _build_messages(prompt: str, history: list | None) -> list:
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _get_vqd_tokens(client: httpx.AsyncClient) -> tuple[str, str]:
    """Fetch x-vqd-4 token and x-vqd-hash-1 from DDG status endpoint."""
    resp = await client.get(
        DDG_STATUS_URL,
        headers={
            **HEADERS_BASE,
            "Content-Type": "application/json",
            "x-vqd-accept": "1",
        },
        timeout=15,
    )
    vqd_token = resp.headers.get("x-vqd-4", "")
    vqd_hash  = resp.headers.get("x-vqd-hash-1", "")

    if not vqd_token:
        raise ValueError(
            f"Could not get x-vqd-4 token from DuckDuckGo "
            f"(status={resp.status_code}, headers={dict(resp.headers)})"
        )
    return vqd_token, vqd_hash


async def _do_chat(client: httpx.AsyncClient, model: str, messages: list) -> str:
    """Perform one chat request, returns the full response text."""
    vqd_token, vqd_hash = await _get_vqd_tokens(client)

    chat_headers = {
        **HEADERS_BASE,
        "Content-Type": "application/json",
        "x-vqd-4":      vqd_token,
    }
    if vqd_hash:
        chat_headers["x-vqd-hash-1"] = vqd_hash

    payload = {"model": model, "messages": messages}

    resp = await client.post(
        DDG_CHAT_URL,
        headers=chat_headers,
        content=json.dumps(payload),
        timeout=60,
    )

    if resp.status_code == 429:
        raise ValueError("RateLimit: DuckDuckGo rate limit hit")
    if resp.status_code != 200:
        raise ValueError(f"DDG returned HTTP {resp.status_code}: {resp.text[:300]}")

    # Parse SSE stream: data: {"message": "..."}
    result_parts = []
    for line in resp.text.splitlines():
        if not line.startswith("data: "):
            continue
        chunk = line[6:].strip()
        if chunk in ("[DONE]", ""):
            continue
        try:
            obj = json.loads(chunk)
            msg = obj.get("message", "")
            if msg:
                result_parts.append(msg)
        except json.JSONDecodeError:
            continue

    full = "".join(result_parts).strip()
    if not full:
        raise ValueError("Empty response from DuckDuckGo")
    return full


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    """
    Send a prompt to DuckDuckGo AI Chat and return the full response.
    Calls the DDG API directly — no external library required.
    """
    ddg_model = _resolve_model(model)
    messages  = _build_messages(prompt, history)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient() as client:
                return await _do_chat(client, ddg_model, messages)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "ratelimit" in err_str or "429" in err_str or "vqd" in err_str:
                await asyncio.sleep(2 ** attempt)   # 1s, 2s, 4s
                continue
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            break

    raise ValueError(f"DuckDuckGo AI failed after {MAX_RETRIES} attempts: {last_error}")


def get_available_models() -> list[dict]:
    return [
        {"id": "gpt-5-mini",       "name": "GPT-5 mini",        "provider": "OpenAI via DDG"},
        {"id": "gpt-4o-mini",      "name": "GPT-4o mini",       "provider": "OpenAI via DDG"},
        {"id": "gpt-oss-120b",     "name": "GPT-OSS 120B",      "provider": "OpenAI via DDG"},
        {"id": "llama-4-scout",    "name": "Llama 4 Scout",     "provider": "Meta via DDG"},
        {"id": "claude-haiku-4.5", "name": "Claude Haiku 4.5",  "provider": "Anthropic via DDG"},
        {"id": "mistral-small-3",  "name": "Mistral Small 3",   "provider": "Mistral via DDG"},
    ]
