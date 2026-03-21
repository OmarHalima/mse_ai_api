"""Simple in-memory stats tracker."""
import time
from collections import deque
from threading import Lock

_lock = Lock()
_stats = {"total_requests": 0, "success": 0, "errors": 0, "total_tokens": 0, "start_time": time.time()}
_recent: deque = deque(maxlen=100)


def record(success: bool, prompt_tokens: int = 0, completion_tokens: int = 0,
           model: str = "auto", error: str = ""):
    with _lock:
        _stats["total_requests"] += 1
        _stats["total_tokens"] += prompt_tokens + completion_tokens
        if success:
            _stats["success"] += 1
        else:
            _stats["errors"] += 1
        _recent.appendleft({
            "time": time.strftime("%H:%M:%S"),
            "model": model,
            "p_tokens": prompt_tokens,
            "c_tokens": completion_tokens,
            "status": "✅" if success else "❌",
            "error": error[:80] if error else "",
        })


def get_stats() -> dict:
    with _lock:
        uptime_sec = int(time.time() - _stats["start_time"])
        h, rem = divmod(uptime_sec, 3600)
        m, s = divmod(rem, 60)
        return {**_stats, "uptime": f"{h:02d}:{m:02d}:{s:02d}", "recent": list(_recent)}
