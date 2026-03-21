"""
DuckDuckGo AI Chat Client — Full Playwright Backend
- Runs the entire chat request inside a real browser via Playwright
- Bypasses DDG's JS challenge completely by executing in real Chromium
- Intercepts the SSE response directly from the browser network
"""
import asyncio
import json
from playwright.async_api import async_playwright, Browser, Page

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
DDG_CHAT_URL = "https://duckduckgo.com/duckchat/v1/chat"
DDG_HOME_URL = "https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat"

# ── Shared browser instance ───────────────────────────────────────────────────
_playwright = None
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


async def _get_browser() -> Browser:
    global _playwright, _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-first-run",
                    "--single-process",
                ],
            )
            print("[ddg] Playwright browser started")
    return _browser


async def _chat_in_browser(model: str, messages: list) -> str:
    """
    Run the entire DDG chat inside a real browser page.
    Uses fetch() from within the page context so JS challenge is solved natively.
    Intercepts the response body directly.
    """
    browser = await _get_browser()
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    )
    page: Page = await context.new_page()

    try:
        # Navigate to DDG first so we're on the right origin
        await page.goto(DDG_HOME_URL, wait_until="domcontentloaded", timeout=30000)

        # Execute chat request via fetch() inside the browser — 
        # this means DDG's JS challenge runs natively
        result = await page.evaluate("""
            async ([chatUrl, model, messages]) => {
                // Step 1: get VQD token + hash via status endpoint
                const statusResp = await fetch('https://duckduckgo.com/duckchat/v1/status', {
                    headers: {
                        'accept': 'text/event-stream',
                        'accept-language': 'en-US,en;q=0.9',
                        'cache-control': 'no-cache',
                        'content-type': 'application/json',
                        'pragma': 'no-cache',
                        'x-vqd-accept': '1'
                    }
                });

                const vqd4   = statusResp.headers.get('x-vqd-4') || '';
                const hash1  = statusResp.headers.get('x-vqd-hash-1') || '';

                if (!vqd4 && !hash1) {
                    return {error: 'No VQD token from status endpoint'};
                }

                // Step 2: send chat request
                const chatHeaders = {
                    'accept': 'text/event-stream',
                    'accept-language': 'en-US,en;q=0.9',
                    'cache-control': 'no-cache',
                    'content-type': 'application/json',
                    'pragma': 'no-cache',
                    'origin': 'https://duckduckgo.com',
                    'referer': 'https://duckduckgo.com/'
                };
                if (vqd4)  chatHeaders['x-vqd-4']      = vqd4;
                if (hash1) chatHeaders['x-vqd-hash-1'] = hash1;

                const chatResp = await fetch(chatUrl, {
                    method: 'POST',
                    headers: chatHeaders,
                    body: JSON.stringify({model, messages})
                });

                if (!chatResp.ok) {
                    const txt = await chatResp.text();
                    return {error: `HTTP ${chatResp.status}: ${txt.slice(0, 300)}`};
                }

                // Step 3: read SSE stream and collect message chunks
                const text = await chatResp.text();
                const parts = [];
                for (const line of text.split('\\n')) {
                    if (!line.startsWith('data: ')) continue;
                    const chunk = line.slice(6).trim();
                    if (chunk === '[DONE]' || chunk === '') continue;
                    try {
                        const obj = JSON.parse(chunk);
                        if (obj.message) parts.push(obj.message);
                    } catch(e) {}
                }
                return {result: parts.join('')};
            }
        """, [DDG_CHAT_URL, model, messages])

        if result.get("error"):
            raise ValueError(result["error"])

        text = result.get("result", "").strip()
        if not text:
            raise ValueError("Empty response from DuckDuckGo")
        return text

    finally:
        await page.close()
        await context.close()


async def ask(prompt: str, model: str = "auto", history: list | None = None) -> str:
    ddg_model = _resolve_model(model)
    messages  = _build_messages(prompt, history)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await _chat_in_browser(ddg_model, messages)
        except Exception as e:
            last_error = e
            print(f"[ddg] attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            continue

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
