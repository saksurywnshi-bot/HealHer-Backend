"""Tests for security middlewares and limits (body limit, Mermaid diagram export)."""

import pytest


def test_body_size_exceeding_200kb_rejected(client):
    large_payload = {
        "language": "hi",
        "items": [
            {"key": f"key_{i}", "text": "x" * 2500}
            for i in range(90)
        ],  # > 225 KB
    }
    response = client.post("/api/translate", json=large_payload)
    assert response.status_code == 413
    assert "Request body too large" in response.json()["detail"]


def test_mermaid_diagram_endpoint(client):
    response = client.get("/api/graph/mermaid")
    assert response.status_code == 200
    assert "graph TD" in response.text
    assert "validate_input" in response.text
    assert "validate_placeholders" in response.text
