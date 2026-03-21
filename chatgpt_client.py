"""
DuckDuckGo AI Chat Client — direct httpx implementation
- Fetches x-vqd-4 token from status endpoint
- Sends chat request with proper headers
- No Playwright, no browser, no external AI libraries
"""
import asyncio
import json
import random
import httpx

# ── Model name mapping ────────────────────────────────────────────────────────
MODEL_MAP = {
    "gpt-5-mini":               "gpt-4o-mini",
    "gpt5-mini":                "gpt-4o-mini",
    "auto":                     "gpt-4o-mini",
    "gpt-4o-mini":              "gpt-4o-mini",
    "gpt-4o":                   "gpt-4o-mini",
    "gpt-oss-120b":             "o3-mini",
    "gpt-oss":                  "o3-mini",
    "o3-mini":                  "o3-mini",
    "llama-4-scout":            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama4":                   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama":                    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama-3.3-70b":            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "claude-haiku-4.5":         "claude-3-haiku-20240307",
    "claude-haiku":             "claude-3-haiku-20240307",
    "claude-3-haiku":           "claude-3-haiku-20240307",
    "claude-3-haiku-20240307":  "claude-3-haiku-20240307",
    "mistral-small-3":          "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistral":                  "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral-8x7b":             "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral":                  "mistralai/Mistral-Small-24B-Instruct-2501",
}

MAX_RETRIES    = 3
DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_CHAT_URL   = "https://duckduckgo.com/duckchat/v1/chat"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
]


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


def _build_messages(prompt: str, history: list | None) -> list:
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _fetch_vqd(client: httpx.AsyncClient, ua: str) -> str:
    """Get x-vqd-4 token from DDG status endpoint."""
    resp = await client.get(
        DDG_STATUS_URL,
        headers={
            "User-Agent":    ua,
            "x-vqd-accept":  "1",
            "Accept":        "application/json",
            "Referer":       "https://duckduckgo.com/",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise ValueError(f"Status endpoint returned {resp.status_code}")
    vqd = resp.headers.get("x-vqd-4", "")
    if not vqd:
        raise ValueError("x-vqd-4 header missing from status response")
    return vqd


async def _do_chat(model: str, messages: list) -> str:
    ua  = random.choice(USER_AGENTS)

    async with httpx.AsyncClient() as client:
        vqd = await _fetch_vqd(client, ua)
        print(f"[ddg] vqd obtained: {vqd[:20]}...")

        headers = {
            "User-Agent":      ua,
            "Accept":          "text/event-stream",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type":    "application/json",
            "Origin":          "https://duckduckgo.com",
            "Referer":         "https://duckduckgo.com/",
            "x-vqd-4":         vqd,
        }

        resp = await client.post(
            DDG_CHAT_URL,
            headers=headers,
            content=json.dumps({"model": model, "messages": messages}),
            timeout=60,
        )

    if resp.status_code in (418, 403):
        raise ValueError(f"CHALLENGE:{resp.status_code}")
    if resp.status_code == 429:
        raise ValueError("RateLimit:429")
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    parts = []
    for line in resp.text.splitlines():
        if not line.startswith("data: "):
            continue
        chunk = line[6:].strip()
        if chunk in ("[DONE]", ""):
            continue
        try:
            obj = json.loads(chunk)
            if obj.get("message"):
                parts.append(obj["message"])
        except json.JSONDecodeError:
            continue

    result = "".join(parts).strip()
    if not result:
        raise ValueError("Empty response from DDG")
    return result


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    ddg_model = _resolve_model(model)
    messages  = _build_messages(prompt, history)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            result = await _do_chat(ddg_model, messages)
            print(f"[ddg] attempt {attempt+1} success — model={ddg_model}")
            return result
        except Exception as e:
            last_error = e
            err = str(e).lower()
            print(f"[ddg] attempt {attempt+1} failed: {e}")
            if "ratelimit" in err or "429" in err:
                await asyncio.sleep(5 * (attempt + 1))
            elif attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2)

    raise ValueError(f"DuckDuckGo AI failed after {MAX_RETRIES} attempts: {last_error}")


def get_available_models() -> list[dict]:
    return [
        {"id": "gpt-4o-mini",      "name": "GPT-4o mini",       "provider": "OpenAI via DDG"},
        {"id": "o3-mini",          "name": "o3-mini",            "provider": "OpenAI via DDG"},
        {"id": "llama-3.3-70b",    "name": "Llama 3.3 70B",     "provider": "Meta via DDG"},
        {"id": "claude-3-haiku",   "name": "Claude 3 Haiku",    "provider": "Anthropic via DDG"},
        {"id": "mistral-small-3",  "name": "Mistral Small 3",   "provider": "Mistral via DDG"},
    ]
