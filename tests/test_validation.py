"""Tests for input validation node and endpoint."""

import pytest


def test_unsupported_language_returns_400(client):
    payload = {
        "language": "es",
        "items": [{"key": "home.welcome", "text": "Welcome"}],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unsupported language 'es'" in detail["errors"][0]


def test_empty_items_returns_400(client):
    payload = {
        "language": "hi",
        "items": [],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("cannot be empty" in err for err in detail["errors"])


def test_exceeding_max_items_returns_400(client):
    items = [{"key": f"item.{i}", "text": f"Text {i}"} for i in range(101)]
    payload = {
        "language": "hi",
        "items": items,
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("Maximum items per batch is 100" in err for err in detail["errors"])


def test_empty_key_returns_400(client):
    payload = {
        "language": "hi",
        "items": [{"key": "   ", "text": "Some text"}],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("empty or missing key" in err for err in detail["errors"])


def test_empty_text_returns_400(client):
    payload = {
        "language": "hi",
        "items": [{"key": "greeting", "text": "   "}],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("empty text" in err for err in detail["errors"])
