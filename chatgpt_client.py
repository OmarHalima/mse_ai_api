"""
DuckDuckGo AI Chat Client — using duckai library
- No Playwright, no browser, no challenge issues
- duckai handles x-vqd-4 / x-vqd-hash-1 tokens automatically
"""
import asyncio
from duckai import DuckAI

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
    "llama-4-scout":            "llama-3.3-70b",
    "llama4":                   "llama-3.3-70b",
    "llama":                    "llama-3.3-70b",
    "llama-3.3-70b":            "llama-3.3-70b",
    "claude-haiku-4.5":         "claude-3-haiku",
    "claude-haiku":             "claude-3-haiku",
    "claude-3-haiku":           "claude-3-haiku",
    "claude-3-haiku-20240307":  "claude-3-haiku",
    "mistral-small-3":          "mistral-small-3",
    "mistral":                  "mistral-small-3",
    "mixtral-8x7b":             "mistral-small-3",
    "mixtral":                  "mistral-small-3",
}

MAX_RETRIES = 3

# Shared DuckAI instance
_duck: DuckAI | None = None
_duck_lock = asyncio.Lock()


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


def _build_messages(prompt: str, history: list | None) -> list:
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


def _messages_to_single_prompt(messages: list) -> str:
    """Flatten multi-turn history into a single prompt string for duckai."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System]: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts)


async def _get_duck() -> DuckAI:
    global _duck
    async with _duck_lock:
        if _duck is None:
            _duck = DuckAI()
    return _duck


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    ddg_model = _resolve_model(model)
    messages  = _build_messages(prompt, history)

    # Build final prompt (single string for duckai)
    if len(messages) == 1:
        final_prompt = messages[0]["content"]
    else:
        final_prompt = _messages_to_single_prompt(messages)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            duck = await _get_duck()
            # duckai.chat is synchronous — run in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: duck.chat(final_prompt, model=ddg_model, timeout=60)
            )
            if not result or not result.strip():
                raise ValueError("Empty response")
            print(f"[ddg] attempt {attempt+1} success — model={ddg_model}")
            return result.strip()
        except Exception as e:
            last_error = e
            err = str(e).lower()
            print(f"[ddg] attempt {attempt+1} failed: {e}")
            # Always reset instance so next attempt gets a fresh session
            async with _duck_lock:
                _duck = None
            if "ratelimit" in err or "429" in err or "conversationlimit" in err:
                # DDG rate limit: 1 req/15s — wait before retry
                wait = 15 * (attempt + 1)
                print(f"[ddg] rate limited — waiting {wait}s")
                await asyncio.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                await asyncio.sleep(3)

    raise ValueError(f"DuckDuckGo AI failed after {MAX_RETRIES} attempts: {last_error}")


def get_available_models() -> list[dict]:
    return [
        {"id": "gpt-4o-mini",      "name": "GPT-4o mini",       "provider": "OpenAI via DDG"},
        {"id": "o3-mini",          "name": "o3-mini",            "provider": "OpenAI via DDG"},
        {"id": "llama-3.3-70b",    "name": "Llama 3.3 70B",     "provider": "Meta via DDG"},
        {"id": "claude-3-haiku",   "name": "Claude 3 Haiku",    "provider": "Anthropic via DDG"},
        {"id": "mistral-small-3",  "name": "Mistral Small 3",   "provider": "Mistral via DDG"},
    ]
