"""Optional production API security controls."""
from __future__ import annotations
import hashlib
import os
import time
from collections import defaultdict, deque
from threading import Lock
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

class SecurityMiddleware(BaseHTTPMiddleware):
    """API-key auth, lightweight rate limiting, and security headers."""
    def __init__(self, app):
        super().__init__(app)
        self.api_key = os.getenv("SA_ZD_NIDS_API_KEY", "").strip()
        self.limit = max(0, int(os.getenv("SA_ZD_NIDS_RATE_LIMIT_PER_MINUTE", "0")))
        self.window = 60.0
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def _authorized(self, request: Request) -> bool:
        if not self.api_key or not request.url.path.startswith("/api/"):
            return True
        supplied = request.headers.get("X-API-Key", "")
        return bool(supplied) and hashlib.compare_digest(supplied, self.api_key)

    def _rate_limited(self, client_id: str) -> bool:
        if self.limit <= 0:
            return False
        now = time.monotonic()
        with self.lock:
            bucket = self.hits[client_id]
            while bucket and now - bucket[0] >= self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return True
            bucket.append(now)
        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._authorized(request):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        client_id = request.client.host if request.client else "unknown"
        if request.url.path.startswith("/api/") and self._rate_limited(client_id):
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
