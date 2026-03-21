"""
Keep-alive background task.
Pings the /health endpoint every 25 minutes to prevent HuggingFace Space from sleeping.
HF free spaces sleep after ~48h of inactivity — this keeps them awake indefinitely.
"""
import asyncio
import os
import httpx

PING_INTERVAL = 25 * 60  # 25 minutes


async def _ping(url: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            print(f"[keepalive] ping {url} → {r.status_code}")
    except Exception as e:
        print(f"[keepalive] ping failed: {e}")


async def start_keepalive():
    """
    Start the keep-alive loop.
    Detects the Space URL automatically from HF env vars,
    falls back to localhost if not on HuggingFace.
    """
    # HuggingFace injects SPACE_HOST for Docker spaces
    space_host = os.getenv("SPACE_HOST", "")
    port = os.getenv("PORT", "7860")

    if space_host:
        url = f"https://{space_host}/health"
    else:
        url = f"http://localhost:{port}/health"

    print(f"[keepalive] started — pinging {url} every {PING_INTERVAL//60} min")

    # Wait for server to fully start before first ping
    await asyncio.sleep(30)

    while True:
        await _ping(url)
        await asyncio.sleep(PING_INTERVAL)
