"""Authentication and user identity service for HealHer.

Privacy & Security:
- Validates Supabase JWT using SUPABASE_JWT_SECRET.
- Extracts 'sub' claim and converts it to a SHA-256 hash (user_ref).
- Never stores real user IDs or raw tokens.
- In 'development' mode only, allows an 'x-user-ref' header fallback for local testing.
  In any non-development environment, this fallback is strictly disabled.
"""

import hashlib
import jwt
from fastapi import Header, HTTPException, Request, status

from app.config import APP_ENV, SUPABASE_JWT_SECRET


def hash_user_id(raw_id: str) -> str:
    """Compute deterministic SHA-256 hash of a user identifier."""
    raw = raw_id.strip()
    if len(raw) == 64 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return raw.lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()



async def get_current_user_ref(
    request: Request,
    authorization: str | None = Header(None, alias="Authorization"),
    x_user_ref: str | None = Header(None, alias="x-user-ref"),
) -> str:
    """FastAPI dependency to extract and verify the user_ref.

    Returns:
        64-character SHA-256 hash of the user identity.

    Raises:
        HTTPException(401) on missing or invalid credentials.
    """
    import logging
    logger = logging.getLogger("healher_backend.auth")

    is_dev = APP_ENV.strip().lower() == "development"
    is_placeholder_secret = (
        not SUPABASE_JWT_SECRET
        or SUPABASE_JWT_SECRET.strip().lower() in [
            "paste_from_supabase_project_settings",
            "your_supabase_jwt_secret_here",
        ]
    )

    # 1. Check for Bearer token
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            try:
                # In development with placeholder secret, skip signature check so dev tokens or mock tokens work
                verify_sig = not (is_dev and is_placeholder_secret)
                payload = jwt.decode(
                    token,
                    SUPABASE_JWT_SECRET if verify_sig else None,
                    algorithms=["HS256"],
                    options={"verify_signature": verify_sig, "verify_aud": False},
                )
                sub = payload.get("sub")
                if not sub:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token: missing subject claim ('sub').",
                    )
                return hash_user_id(str(sub))
            except jwt.PyJWTError as exc:
                logger.warning("JWT verification failed: %s (is_dev=%s)", exc, is_dev)
                # In dev mode, if JWT verification failed but x-user-ref was provided, use it
                if is_dev and x_user_ref:
                    return hash_user_id(x_user_ref)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid or expired authentication token: {str(exc)}",
                )
        else:
            if is_dev and x_user_ref:
                return hash_user_id(x_user_ref)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
            )

    # 2. Check for development fallback header
    is_dev = APP_ENV.strip().lower() == "development"
    if is_dev and x_user_ref:
        return hash_user_id(x_user_ref)

    # 3. Unauthorized
    if is_dev:
        detail_msg = "Authentication required. Provide an 'Authorization: Bearer <token>' header or 'x-user-ref' for local dev."
    else:
        detail_msg = "Authentication required. Provide an 'Authorization: Bearer <token>' header."

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail_msg,
    )
