"""
DuckDuckGo AI Chat Client — Playwright Cookie Extraction + httpx
- Browser visits DDG to solve JS challenge and get valid cookies/headers
- Python (httpx) sends the actual chat request using those cookies
- Avoids "Failed to fetch" issue in restricted environments (HuggingFace)
"""
import asyncio
import json
import random
import httpx
from playwright.async_api import async_playwright, Browser

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

MAX_RETRIES  = 3
DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_CHAT_URL   = "https://duckduckgo.com/duckchat/v1/chat"
DDG_HOME_URL   = "https://duck.ai/"

# ── User-Agent pool ───────────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

# ── Shared state ──────────────────────────────────────────────────────────────
_playwright_inst = None
_browser: Browser | None = None
_browser_lock = asyncio.Lock()

# Cached session: {ua, cookies, vqd, hash, expires}
_session: dict = {}


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
    global _playwright_inst, _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            _playwright_inst = await async_playwright().start()
            _browser = await _playwright_inst.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox",
                      "--disable-dev-shm-usage", "--disable-gpu",
                      "--no-first-run", "--single-process"],
            )
            print("[ddg] Playwright browser started")
    return _browser


async def _refresh_session() -> dict:
    """
    Visit duck.ai in a real browser to:
    1. Solve JS challenge and get valid cookies
    2. Intercept the status request to get x-vqd-4 / x-vqd-hash-1
    Returns a session dict with cookies + tokens.
    """
    import time
    ua      = random.choice(USER_AGENTS)
    browser = await _get_browser()
    context = await browser.new_context(
        user_agent=ua,
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )

    vqd_token = ""
    vqd_hash  = ""
    got_token = asyncio.Event()

    async def on_response(resp):
        nonlocal vqd_token, vqd_hash
        if "duckchat/v1/status" in resp.url:
            h = resp.headers
            vqd_token = h.get("x-vqd-4", "")
            vqd_hash  = h.get("x-vqd-hash-1", "")
            print(f"[ddg] Status intercepted — vqd4={'✓' if vqd_token else '✗'} hash={'✓' if vqd_hash else '✗'}")
            got_token.set()

    page = await context.new_page()
    page.on("response", on_response)

    try:
        await page.goto(DDG_HOME_URL, wait_until="domcontentloaded", timeout=30000)

        # Wait for status request; if not triggered, force it
        try:
            await asyncio.wait_for(got_token.wait(), timeout=8)
        except asyncio.TimeoutError:
            print("[ddg] Forcing status request via CDP fetch...")
            await page.evaluate("""
                fetch('https://duckduckgo.com/duckchat/v1/status',
                      {headers:{'x-vqd-accept':'1','cache-control':'no-cache'}})
            """)
            await asyncio.wait_for(got_token.wait(), timeout=8)

        # Extract cookies from browser context
        raw_cookies = await context.cookies("https://duckduckgo.com")
        cookie_str  = "; ".join(f"{c['name']}={c['value']}" for c in raw_cookies)

        session = {
            "ua":      ua,
            "cookies": cookie_str,
            "vqd":     vqd_token,
            "hash":    vqd_hash,
            "expires": time.time() + 50 * 60,  # 50 min
        }
        print(f"[ddg] Session refreshed — cookies={len(raw_cookies)}, ua=...{ua[-30:]}")
        return session

    finally:
        await page.close()
        await context.close()


async def _get_session(force: bool = False) -> dict:
    import time
    global _session
    if not force and _session.get("expires", 0) > time.time():
        return _session
    _session = await _refresh_session()
    return _session


async def _do_chat(model: str, messages: list, force_refresh: bool = False) -> str:
    session = await _get_session(force=force_refresh)

    headers = {
        "User-Agent":      session["ua"],
        "Accept":          "text/event-stream",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control":   "no-cache",
        "Pragma":          "no-cache",
        "Content-Type":    "application/json",
        "Origin":          "https://duckduckgo.com",
        "Referer":         "https://duckduckgo.com/",
    }
    if session["cookies"]:
        headers["Cookie"] = session["cookies"]
    if session["vqd"]:
        headers["x-vqd-4"] = session["vqd"]
    if session["hash"]:
        headers["x-vqd-hash-1"] = session["hash"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            DDG_CHAT_URL,
            headers=headers,
            content=json.dumps({"model": model, "messages": messages}),
            timeout=60,
        )

    if resp.status_code in (418, 403):
        raise ValueError(f"CHALLENGE:{resp.status_code}")
    if resp.status_code == 429:
        raise ValueError("RateLimit")
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")

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
        raise ValueError("Empty response")
    return result


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    ddg_model     = _resolve_model(model)
    messages      = _build_messages(prompt, history)
    force_refresh = False
    last_error    = None

    for attempt in range(MAX_RETRIES):
        try:
            return await _do_chat(ddg_model, messages, force_refresh=force_refresh)
        except Exception as e:
            last_error = e
            err = str(e).lower()
            print(f"[ddg] attempt {attempt+1} failed: {e}")
            if "challenge" in err or "418" in err or "403" in err:
                force_refresh = True
                await asyncio.sleep(2)
            elif "ratelimit" in err or "429" in err:
                await asyncio.sleep(2 ** attempt)
            elif attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)

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
