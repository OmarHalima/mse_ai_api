"""
DuckDuckGo AI Chat Client
- Uses duckduckgo_search library (handles x-vqd-4 token automatically)
- No login required, no browser needed
- Retry logic for rate limits and token errors
"""
import asyncio

from ddgs import DDGS  # pip install ddgs>=7.0.0

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


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


def _build_prompt(prompt: str, history: list | None) -> str:
    """Flatten history + prompt into a single string for DDGS.chat()"""
    if not history:
        return prompt
    parts = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "User" if role == "user" else "Assistant"
        parts.append(f"{prefix}: {content}")
    parts.append(f"User: {prompt}")
    return "\n".join(parts)


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    """
    Send a prompt to DuckDuckGo AI Chat and return the full response.
    Uses duckduckgo_search library which handles token management automatically.
    Retries up to MAX_RETRIES times on failure.
    """
    ddg_model = _resolve_model(model)
    full_prompt = _build_prompt(prompt, history)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            # DDGS.chat() is synchronous — run in thread pool to avoid blocking
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: DDGS().chat(full_prompt, model=ddg_model)
            )
            if result:
                return result.strip()
            raise ValueError("Empty response from DuckDuckGo")
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # Rate limit or token error → wait and retry
            if "ratelimit" in err_str or "429" in err_str or "vqd" in err_str:
                wait = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait)
                continue
            # Other errors → retry once then raise
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
