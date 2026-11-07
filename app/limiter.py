"""Rate limiter setup for HealHer API."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import RATE_LIMIT

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


def get_user_rate_limit_key(request) -> str:
    """Rate limit key based on user authorization header, x-user-ref, or client IP."""
    auth = request.headers.get("Authorization")
    if auth:
        return auth.strip()
    x_user = request.headers.get("x-user-ref")
    if x_user:
        return x_user.strip()
    return get_remote_address(request)

