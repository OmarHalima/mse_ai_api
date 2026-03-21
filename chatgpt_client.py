"""
DuckDuckGo AI Chat Client — Playwright Backend
- Uses headless Chromium to bypass DDG's JS challenge
- Intercepts the real x-vqd-4 token from browser network requests
- Caches the token and reuses it until it expires
- Falls back to re-fetching token on 418/403 errors
"""
import asyncio
import json
import re
import time
import httpx
from playwright.async_api import async_playwright, Browser, BrowserContext

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

MAX_RETRIES   = 3
TOKEN_TTL     = 55 * 60   # re-fetch token every 55 minutes

DDG_CHAT_URL   = "https://duckduckgo.com/duckchat/v1/chat"
DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_HOME_URL   = "https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat"

HEADERS_BASE = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept":          "text/event-stream",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control":   "no-cache",
    "Pragma":          "no-cache",
    "Origin":          "https://duckduckgo.com",
    "Referer":         "https://duckduckgo.com/",
    "Content-Type":    "application/json",
}

# ── Global token cache ────────────────────────────────────────────────────────
_token_cache: dict = {"vqd": None, "hash": None, "expires": 0}
_token_lock = asyncio.Lock()
_playwright_instance = None
_browser: Browser | None = None


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, MODEL_MAP["auto"])


def _build_messages(prompt: str, history: list | None) -> list:
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _get_browser() -> Browser:
    """Get or create a shared Playwright browser instance."""
    global _playwright_instance, _browser
    if _browser is None or not _browser.is_connected():
        _playwright_instance = await async_playwright().start()
        _browser = await _playwright_instance.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-first-run",
                "--no-zygote",
                "--single-process",
            ],
        )
        print("[ddg] Playwright browser started")
    return _browser


async def _fetch_token_via_browser() -> tuple[str, str]:
    """
    Open DDG in headless browser, intercept the status request,
    and extract x-vqd-4 and x-vqd-hash-1 tokens.
    """
    browser = await _get_browser()
    context: BrowserContext = await browser.new_context(
        user_agent=HEADERS_BASE["User-Agent"],
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    vqd_token = None
    vqd_hash  = None
    event     = asyncio.Event()

    async def on_response(response):
        nonlocal vqd_token, vqd_hash
        if "duckchat/v1/status" in response.url:
            headers = response.headers
            vqd_token = headers.get("x-vqd-4", "")
            vqd_hash  = headers.get("x-vqd-hash-1", "")
            print(f"[ddg] Token intercepted — vqd-4: {'✓' if vqd_token else '✗'}, hash-1: {'✓' if vqd_hash else '✗'}")
            event.set()

    page = await context.new_page()
    page.on("response", on_response)

    try:
        # Navigate to DDG AI Chat page — this triggers the status request
        await page.goto(DDG_HOME_URL, wait_until="domcontentloaded", timeout=30000)

        # Wait up to 10s for the status request to be intercepted
        try:
            await asyncio.wait_for(event.wait(), timeout=10)
        except asyncio.TimeoutError:
            # Manually trigger the status request via JS if page didn't do it
            print("[ddg] Triggering status request manually via JS...")
            await page.evaluate("""
                fetch('https://duckduckgo.com/duckchat/v1/status', {
                    headers: {'x-vqd-accept': '1'}
                });
            """)
            await asyncio.wait_for(event.wait(), timeout=10)

        if not vqd_token and not vqd_hash:
            raise ValueError("Browser intercepted status response but found no VQD tokens")

        return vqd_token or "", vqd_hash or ""

    finally:
        await page.close()
        await context.close()


async def _get_tokens(force_refresh: bool = False) -> tuple[str, str]:
    """Return cached tokens, or fetch fresh ones if expired."""
    async with _token_lock:
        now = time.time()
        if not force_refresh and _token_cache["expires"] > now and (
            _token_cache["vqd"] or _token_cache["hash"]
        ):
            return _token_cache["vqd"], _token_cache["hash"]

        print("[ddg] Fetching fresh VQD tokens via browser...")
        vqd, vqd_hash = await _fetch_token_via_browser()
        _token_cache.update({"vqd": vqd, "hash": vqd_hash, "expires": now + TOKEN_TTL})
        return vqd, vqd_hash


async def _do_chat(model: str, messages: list, force_refresh: bool = False) -> str:
    """Perform one chat request using cached tokens."""
    vqd, vqd_hash = await _get_tokens(force_refresh=force_refresh)

    headers = {**HEADERS_BASE}
    if vqd:
        headers["x-vqd-4"] = vqd
    if vqd_hash:
        headers["x-vqd-hash-1"] = vqd_hash

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            DDG_CHAT_URL,
            headers=headers,
            content=json.dumps({"model": model, "messages": messages}),
            timeout=60,
        )

    if resp.status_code in (418, 403):
        raise ValueError(f"CHALLENGE:{resp.status_code}")   # triggers token refresh

    if resp.status_code == 429:
        raise ValueError("RateLimit: DuckDuckGo rate limit hit")

    if resp.status_code != 200:
        raise ValueError(f"DDG chat returned HTTP {resp.status_code}: {resp.text[:300]}")

    # Parse SSE stream
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
    ddg_model = _resolve_model(model)
    messages  = _build_messages(prompt, history)

    last_error  = None
    force_refresh = False

    for attempt in range(MAX_RETRIES):
        try:
            return await _do_chat(ddg_model, messages, force_refresh=force_refresh)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()

            if "challenge" in err_str or "418" in err_str or "403" in err_str:
                # Token rejected — force browser refresh
                print(f"[ddg] Token rejected (attempt {attempt+1}), refreshing via browser...")
                force_refresh = True
                await asyncio.sleep(2)
                continue

            if "ratelimit" in err_str or "429" in err_str:
                await asyncio.sleep(2 ** attempt)
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
