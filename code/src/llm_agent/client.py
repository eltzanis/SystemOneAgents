"""One chat completion from llama-server's OpenAI-compatible endpoint. llama.cpp's own response fields (timings) are kept."""
import time

import httpx


def complete(url, body, timeout=3600.0, retries=2):
    """-> {"response": <JSON>, "seconds"} or {"error": <text>, "seconds"}. Connection failures are retried (a server
    that restarts); an HTTP error (e.g. a prompt longer than the context) is returned as it is, never retried."""
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
        except httpx.TransportError as e:
            if attempt == retries:
                return {"error": f"{type(e).__name__}: {e}", "seconds": time.perf_counter() - t0}
            time.sleep(10)
            continue
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:4000]}", "seconds": dt}
        return {"response": r.json(), "seconds": dt}
