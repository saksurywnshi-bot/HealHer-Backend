"""Tests for Admin API authentication and operations."""

import pytest


def test_admin_routes_require_valid_token(client):
    # Missing token
    resp = client.get("/api/admin/translations")
    assert resp.status_code == 401

    # Invalid token
    resp = client.get("/api/admin/translations", headers={"x-admin-token": "wrong-token"})
    assert resp.status_code == 401


def test_admin_list_translations_with_filters(client, setup_test_db, admin_headers):
    cache = setup_test_db
    cache.save_machine_translation("k1", "hi", "Hello", "नमस्ते")
    cache.save_machine_translation("k2", "bn", "Hello", "হ্যালো")
    cache.save_machine_translation("k3", "hi", "Bye", "अलविदा")
    cache.update_reviewed_translation("k3", "hi", "शुभ अलविदा")

    # Filter: status=machine and language=hi
    resp = client.get(
        "/api/admin/translations?status=machine&language=hi",
        headers=admin_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["translations"][0]["key"] == "k1"
    assert data["translations"][0]["status"] == "machine"

    # Filter: status=reviewed and language=hi
    resp = client.get(
        "/api/admin/translations?status=reviewed&language=hi",
        headers=admin_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["translations"][0]["key"] == "k3"
    assert data["translations"][0]["status"] == "reviewed"


def test_admin_patch_translation(client, setup_test_db, admin_headers):
    cache = setup_test_db
    cache.save_machine_translation("home.btn", "ta", "Next", "அடுத்து")

    payload = {
        "key": "home.btn",
        "language": "ta",
        "translated_text": "அடுத்த படி (மனித மதிப்பாய்வு செய்யப்பட்டது)",
    }
    resp = client.patch(
        "/api/admin/translations",
        json=payload,
        headers=admin_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["translation"]["status"] == "reviewed"
    assert data["translation"]["translated_text"] == payload["translated_text"]

    # Verify in DB
    record = cache.get_translation("home.btn", "ta")
    assert record["status"] == "reviewed"
    assert record["translated_text"] == payload["translated_text"]


def test_admin_export_csv(client, setup_test_db, admin_headers):
    cache = setup_test_db
    cache.save_machine_translation("nav.home", "hi", "Home", "होम")

    resp = client.get("/api/admin/export.csv", headers=admin_headers)
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert 'attachment; filename="translations_export.csv"' in resp.headers["content-disposition"]

    content = resp.text
    assert "key,language,source_text,source_hash,translated_text,status,updated_at" in content
    assert "nav.home,hi,Home" in content
