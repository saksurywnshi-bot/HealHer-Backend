"""Tests for JWT verification, security constraints, startup checks, and privacy guarantees."""

import os
import sqlite3
import pytest
from app.config import DATABASE_PATH, validate_startup_env
from app.services.auth import hash_user_id
from tests.conftest import create_test_jwt, TEST_JWT_SECRET


def test_jwt_authentication_success(client, auth_headers):
    """Valid Bearer JWT allows access."""
    res = client.get("/api/sentiment/patterns", headers=auth_headers)
    assert res.status_code == 200


def test_invalid_jwt_returns_401(client):
    """Invalid token signature or malformed token returns 401."""
    # Invalid signature (signed with different secret)
    bad_token = create_test_jwt("user_123", secret="wrong_secret_key_12345678901234567890")
    res = client.get("/api/sentiment/patterns", headers={"Authorization": f"Bearer {bad_token}"})
    assert res.status_code == 401

    # Malformed header
    res_malformed = client.get("/api/sentiment/patterns", headers={"Authorization": "NotBearer xyz"})
    assert res_malformed.status_code == 401


def test_dev_fallback_x_user_ref(client, dev_user_headers, monkeypatch):
    """When APP_ENV == 'development', x-user-ref is accepted if no Bearer token is sent."""
    monkeypatch.setattr("app.config.APP_ENV", "development")
    monkeypatch.setattr("app.services.auth.APP_ENV", "development")

    res = client.get("/api/sentiment/patterns", headers=dev_user_headers)
    assert res.status_code == 200

    # In production/staging, x-user-ref fallback MUST be disabled
    monkeypatch.setattr("app.config.APP_ENV", "production")
    monkeypatch.setattr("app.services.auth.APP_ENV", "production")

    res_prod = client.get("/api/sentiment/patterns", headers=dev_user_headers)
    assert res_prod.status_code == 401


def test_startup_check_missing_gemini_key_exits(monkeypatch, capsys):
    """Startup check exits with code 1 if GEMINI_API_KEY is missing or placeholder."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(SystemExit) as exc_info:
        validate_startup_env()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY is missing in .env" in captured.out

    monkeypatch.setenv("GEMINI_API_KEY", "paste_your_key_here")
    with pytest.raises(SystemExit) as exc_info2:
        validate_startup_env()
    assert exc_info2.value.code == 1


def test_startup_check_placeholder_jwt_secret_in_production_exits(monkeypatch, capsys):
    """Startup check exits if SUPABASE_JWT_SECRET is placeholder when APP_ENV != 'development'."""
    monkeypatch.setenv("GEMINI_API_KEY", "valid_real_gemini_api_key_12345")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "paste_from_supabase_project_settings")

    with pytest.raises(SystemExit) as exc_info:
        validate_startup_env()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SUPABASE_JWT_SECRET is a placeholder and APP_ENV is not 'development'" in captured.out


def test_privacy_no_note_text_in_database(client, auth_headers):
    """Verify that note text is NEVER stored in the database."""
    unique_secret_text = "MySuperSecretPersonalDiaryTextThatMustNeverBePersistedInDatabase"

    async def mock_llm(*args, **kwargs):
        return {
            "tone": "neutral",
            "intensity": "low",
            "themes": ["sleep"],
            "risk_language": False,
            "reflection": "You are reflecting on your sleep habits today.",
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": f"This note contains 10 words {unique_secret_text} checking storage privacy safeguards now.",
        "language": "en",
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200

    import app.config as app_config

    # Inspect SQLite database directly
    conn = sqlite3.connect(app_config.DATABASE_PATH)
    cursor = conn.cursor()

    # Verify column names of sentiment_results
    cursor.execute("PRAGMA table_info(sentiment_results);")
    columns = [col[1] for col in cursor.fetchall()]
    assert "note_text" not in columns
    assert "text" not in columns
    assert "reflection" not in columns

    # Verify raw database dump does not contain the secret note text
    cursor.execute("SELECT * FROM sentiment_results;")
    rows = cursor.fetchall()
    assert len(rows) > 0
    for row in rows:
        row_str = " ".join(str(val) for val in row)
        assert unique_secret_text not in row_str

    conn.close()


def test_admin_suggestions_management(client, admin_headers):
    """Test GET and PATCH on /api/admin/suggestions."""
    # List suggestions
    res = client.get("/api/admin/suggestions", headers=admin_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 12
    first_id = data["suggestions"][0]["id"]

    # Patch status to approved
    patch_res = client.patch(
        f"/api/admin/suggestions/{first_id}",
        json={"status": "approved"},
        headers=admin_headers,
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["suggestion"]["status"] == "approved"

    # Reset back to draft
    patch_res2 = client.patch(
        f"/api/admin/suggestions/{first_id}",
        json={"status": "draft"},
        headers=admin_headers,
    )
    assert patch_res2.status_code == 200
    assert patch_res2.json()["suggestion"]["status"] == "draft"
