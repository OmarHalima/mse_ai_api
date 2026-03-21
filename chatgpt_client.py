"""
DuckDuckGo AI Chat Client
- No login required
- No browser needed
- Works on PythonAnywhere and any VPS
- Models: gpt-4o-mini, claude-3-haiku, llama-3.3-70b, mixtral-8x7b, o3-mini
"""
import json
import httpx

# ── Model name mapping (friendly names → DDG internal names) ─────────────────
MODEL_MAP = {
    # GPT-5 mini (default / newest)
    "gpt-5-mini":           "gpt-5-mini",
    "gpt5-mini":            "gpt-5-mini",
    "auto":                 "gpt-5-mini",            # default

    # GPT-4o mini
    "gpt-4o-mini":          "gpt-4o-mini",
    "gpt-4o":               "gpt-4o-mini",

    # GPT-OSS 120B (open source)
    "gpt-oss-120b":         "openai/gpt-oss-120b",
    "gpt-oss":              "openai/gpt-oss-120b",

    # Llama 4 Scout
    "llama-4-scout":        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama4":               "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama":                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # legacy llama 3
    "llama-3.3-70b":        "meta-llama/Llama-4-Scout-17B-16E-Instruct",

    # Claude Haiku 4.5
    "claude-haiku-4.5":     "claude-3-5-haiku-latest",
    "claude-haiku":         "claude-3-5-haiku-latest",
    "claude-3-haiku":       "claude-3-5-haiku-latest",
    "claude-3-haiku-20240307": "claude-3-5-haiku-latest",

    # Mistral Small 3
    "mistral-small-3":      "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistral":              "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral-8x7b":         "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral":              "mistralai/Mistral-Small-24B-Instruct-2501",
}

DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_CHAT_URL   = "https://duckduckgo.com/duckchat/v1/chat"

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://duckduckgo.com",
    "Referer": "https://duckduckgo.com/",
}


def _resolve_model(model: str) -> str:
    """Map any model name to a DDG-supported model string."""
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


async def _get_vqd_token(client: httpx.AsyncClient) -> str:
    """Fetch the x-vqd-4 token required for chat requests."""
    resp = await client.get(
        DDG_STATUS_URL,
        headers={**BASE_HEADERS, "x-vqd-accept": "1"},
        timeout=15,
    )
    if resp.status_code != 200:
        raise ValueError(f"DDG status check failed: {resp.status_code}")
    token = resp.headers.get("x-vqd-4")
    if not token:
        raise ValueError("Could not obtain x-vqd-4 token from DuckDuckGo")
    return token


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    """
    Send a prompt to DuckDuckGo AI Chat and return the full response text.

    Args:
        prompt:  The user message.
        model:   Model name (gpt-4o-mini, claude-3-haiku, llama-3.3-70b,
                 mixtral-8x7b, o3-mini, auto).
        history: Optional list of previous messages for multi-turn conversation.
                 Each item: {"role": "user"|"assistant", "content": "..."}

    Returns:
        The assistant's response as a plain string.

    Raises:
        ValueError: On auth / rate-limit / API errors.
    """
    ddg_model = _resolve_model(model)

    messages = list(history) if history else []
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=60) as client:
        vqd = await _get_vqd_token(client)

        payload = {"model": ddg_model, "messages": messages}
        headers = {
            **BASE_HEADERS,
            "x-vqd-4": vqd,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        full_text = ""
        async with client.stream("POST", DDG_CHAT_URL, headers=headers, json=payload) as resp:
            if resp.status_code == 429:
                raise ValueError("DuckDuckGo rate limit reached. Please wait and retry.")
            if resp.status_code != 200:
                body = await resp.aread()
                raise ValueError(f"DuckDuckGo returned {resp.status_code}: {body[:200]}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    full_text += chunk.get("message", "")
                except (json.JSONDecodeError, KeyError):
                    continue

    return full_text.strip()


def get_available_models() -> list[dict]:
    """Return list of available models in OpenAI format."""
    return [
        {"id": "gpt-5-mini",      "name": "GPT-5 mini",        "provider": "OpenAI via DDG"},
        {"id": "gpt-4o-mini",     "name": "GPT-4o mini",       "provider": "OpenAI via DDG"},
        {"id": "gpt-oss-120b",    "name": "GPT-OSS 120B",      "provider": "OpenAI via DDG"},
        {"id": "llama-4-scout",   "name": "Llama 4 Scout",     "provider": "Meta via DDG"},
        {"id": "claude-haiku-4.5","name": "Claude Haiku 4.5",  "provider": "Anthropic via DDG"},
        {"id": "mistral-small-3", "name": "Mistral Small 3",   "provider": "Mistral via DDG"},
    ]
