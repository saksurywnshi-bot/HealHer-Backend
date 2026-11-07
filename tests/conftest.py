"""Pytest fixtures and configuration.

NEVER calls real Gemini API in tests.
"""

import os
import tempfile
import jwt
import pytest
from fastapi.testclient import TestClient

# Ensure test mode flag is present before loading modules
os.environ["MOCK_LLM_FOR_TESTS"] = "1"
os.environ["GEMINI_API_KEY"] = "mock-key-never-used"
if not os.environ.get("ADMIN_TOKEN"):
    os.environ["ADMIN_TOKEN"] = "test-admin-secret-token-123456789012345678901234"
if not os.environ.get("SUPABASE_JWT_SECRET"):
    os.environ["SUPABASE_JWT_SECRET"] = "test-supabase-secret-123456789012345678901234"

import app.config as config
from app.services.cache import TranslationCache
import app.services.cache as cache_module
from app.services.sentiment_db import init_sentiment_db
from app.main import app

TEST_JWT_SECRET = "test-supabase-secret-123456789012345678901234"


def create_test_jwt(sub: str = "user_test_123", secret: str = TEST_JWT_SECRET) -> str:
    """Helper to generate signed JWT for tests."""
    return jwt.encode(
        {"sub": sub, "aud": "authenticated", "exp": 9999999999},
        secret,
        algorithm="HS256",
    )


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Provide a fresh isolated SQLite database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    test_cache = TranslationCache(db_path=db_path)
    monkeypatch.setattr(cache_module, "cache_service", test_cache)
    monkeypatch.setattr("app.graphs.translate_graph.cache_service", test_cache)
    monkeypatch.setattr("app.routes.admin.cache_service", test_cache)

    monkeypatch.setattr("app.config.DATABASE_PATH", db_path)
    monkeypatch.setattr("app.services.sentiment_db.DATABASE_PATH", db_path)
    monkeypatch.setattr("app.config.SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
    monkeypatch.setattr("app.services.auth.SUPABASE_JWT_SECRET", TEST_JWT_SECRET)

    from app.limiter import limiter
    limiter.reset()
    init_sentiment_db()
    from app.services.caregiver_db import init_caregiver_db
    init_caregiver_db()

    yield test_cache

    limiter.reset()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_headers():
    """Valid admin authorization headers."""
    return {
        "x-admin-token": os.environ.get(
            "ADMIN_TOKEN", "test-admin-secret-token-123456789012345678901234"
        )
    }


@pytest.fixture
def auth_headers():
    """Valid user Supabase JWT Bearer header."""
    token = create_test_jwt("test_user_sub_999")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def dev_user_headers():
    """Development x-user-ref header."""
    return {"x-user-ref": "dev_user_local_456"}
