"""
DuckDuckGo AI Chat — Playwright evaluates x-vqd-hash-1 challenge, then posts via browser context.
DDG removed x-vqd-4; the status endpoint now returns obfuscated JS in x-vqd-hash-1 that must run in Chromium.
"""
import asyncio
import json
import random
from typing import Optional

from playwright.async_api import Browser, Playwright, async_playwright

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

MAX_RETRIES = 3
MAX_VQD_EVAL_RETRIES = 6
DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_CHAT_URL = "https://duckduckgo.com/duckchat/v1/chat"
DDG_ENTRY_REF = "https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
]

_EVAL_HASH_HDR = """
async (hb64) => {
  const src = atob(hb64);
  const fn = new Function("window", "navigator", "document", "return " + src);
  const vqd = await Promise.resolve(fn(window, navigator, document));
  const enc = new TextEncoder();
  const ch = await Promise.all(
    vqd.client_hashes.map(async (c) => {
      const n = await crypto.subtle.digest("SHA-256", enc.encode(c));
      const buff = new Uint8Array(n);
      return btoa(buff.reduce((e, t) => e + String.fromCharCode(t), ""));
    })
  );
  return btoa(JSON.stringify({ ...vqd, client_hashes: ch }));
}
"""

_pw: Playwright | None = None
_browser: Browser | None = None
_browser_lock = asyncio.Lock()


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


def _build_messages(prompt: str, history: list | None) -> list:
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


def _tool_metadata() -> dict:
    return {
        "toolChoice": {
            "LocalSearch": False,
            "NewsSearch": False,
            "VideoSearch": False,
            "WeatherForecast": False,
        }
    }


async def _ensure_browser() -> Browser:
    global _pw, _browser
    async with _browser_lock:
        if _browser is not None:
            return _browser
        _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        return _browser


async def shutdown_playwright() -> None:
    """Stop Chromium (call from app lifespan shutdown)."""
    global _pw, _browser
    async with _browser_lock:
        if _browser is not None:
            await _browser.close()
            _browser = None
        if _pw is not None:
            await _pw.stop()
            _pw = None


def _status_headers() -> dict:
    return {
        "Accept": "application/json",
        "Referer": DDG_ENTRY_REF,
        "Origin": "https://duckduckgo.com",
        "x-vqd-accept": "1",
    }


async def _solve_vqd_hash_hdr(ctx, page) -> str:
    """
    DDG's x-vqd-hash-1 script touches the live DOM (querySelector + getAttribute).
    If the page is not ready, evaluate throws. Always pair a fresh status request
    with evaluate, and retry with reload between attempts.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(MAX_VQD_EVAL_RETRIES):
        st = await ctx.request.get(DDG_STATUS_URL, headers=_status_headers())
        if st.status != 200:
            raise ValueError(f"Status endpoint returned {st.status}")
        hb64 = st.headers.get("x-vqd-hash-1") or ""
        if not hb64:
            raise ValueError("x-vqd-hash-1 header missing from status response (DDG API changed?)")
        try:
            return await page.evaluate(_EVAL_HASH_HDR, hb64)
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            if attempt >= MAX_VQD_EVAL_RETRIES - 1:
                break
            if "getattribute" not in err_s and "null" not in err_s and "cannot read properties" not in err_s:
                raise
            await page.goto(DDG_ENTRY_REF, wait_until="load", timeout=45_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=12_000)
            except Exception:
                pass
            await asyncio.sleep(0.35 + random.random() * 0.45 + attempt * 0.15)
    assert last_err is not None
    raise last_err


async def _do_chat(model: str, messages: list) -> str:
    ua = random.choice(USER_AGENTS)
    browser = await _ensure_browser()
    ctx = await browser.new_context(
        user_agent=ua,
        viewport={"width": 1280, "height": 720},
        locale="en-US",
    )
    page = await ctx.new_page()
    try:
        await ctx.request.get(
            "https://duckduckgo.com/",
            headers={"Referer": DDG_ENTRY_REF},
        )
        await page.goto(DDG_ENTRY_REF, wait_until="load", timeout=45_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        await asyncio.sleep(0.25 + random.random() * 0.35)

        hash_hdr = await _solve_vqd_hash_hdr(ctx, page)

        body = {
            "model": model,
            "messages": messages,
            "canUseTools": True,
            "metadata": _tool_metadata(),
        }
        chat = await ctx.request.post(
            DDG_CHAT_URL,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Referer": DDG_ENTRY_REF,
                "Origin": "https://duckduckgo.com",
                "X-Vqd-Hash-1": hash_hdr,
            },
            data=json.dumps(body, separators=(",", ":")),
            timeout=60_000,
        )

        text = await chat.text()
        if chat.status in (418, 403):
            raise ValueError(f"CHALLENGE:{chat.status}")
        if chat.status == 429:
            raise ValueError("RateLimit:429")
        if chat.status != 200:
            raise ValueError(f"HTTP {chat.status}: {text[:300]}")

    finally:
        await ctx.close()

    parts = []
    for line in text.splitlines():
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
    messages = _build_messages(prompt, history)
    last_error: Optional[BaseException] = None

    for attempt in range(MAX_RETRIES):
        try:
            result = await _do_chat(ddg_model, messages)
            print(f"[ddg] attempt {attempt + 1} success — model={ddg_model}")
            return result
        except Exception as e:
            last_error = e
            err = str(e).lower()
            print(f"[ddg] attempt {attempt + 1} failed: {e}")
            if "ratelimit" in err or "429" in err:
                await asyncio.sleep(5 * (attempt + 1))
            elif attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2)

    raise ValueError(f"DuckDuckGo AI failed after {MAX_RETRIES} attempts: {last_error}")


def get_available_models() -> list[dict]:
    return [
        {"id": "gpt-4o-mini",      "name": "GPT-4o mini",       "provider": "OpenAI via DDG"},
        {"id": "o3-mini",          "name": "o3-mini",           "provider": "OpenAI via DDG"},
        {"id": "llama-3.3-70b",    "name": "Llama 3.3 70B",     "provider": "Meta via DDG"},
        {"id": "claude-3-haiku",   "name": "Claude 3 Haiku",    "provider": "Anthropic via DDG"},
        {"id": "mistral-small-3",  "name": "Mistral Small 3",   "provider": "Mistral via DDG"},
    ]
