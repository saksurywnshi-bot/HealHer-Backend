"""Tests for SQLite translation caching and 'reviewed' precedence."""

import pytest
from app.services.cache import compute_source_hash


def test_cache_hit_for_reviewed_row(client, setup_test_db, monkeypatch):
    cache = setup_test_db
    # Pre-populate a reviewed translation
    cache.save_machine_translation("home.header", "hi", "Welcome Mama", "स्वागतम् माँ")
    cache.update_reviewed_translation("home.header", "hi", "नमस्ते माँ (सत्यापित)")

    # Ensure LLM is NOT called
    llm_called = False

    async def mock_llm(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return {}

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "language": "hi",
        "items": [{"key": "home.header", "text": "Welcome Mama"}],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert not llm_called
    assert data["translations"]["home.header"]["status"] == "reviewed"
    assert data["translations"]["home.header"]["text"] == "नमस्ते माँ (सत्यापित)"


def test_cache_hit_for_machine_row_with_matching_hash(client, setup_test_db):
    cache = setup_test_db
    source_text = "Take a deep breath"
    translated_text = "गहरी सांस लें"
    cache.save_machine_translation("breathe.prompt", "hi", source_text, translated_text)

    llm_called = False

    async def mock_llm(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return {}

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "language": "hi",
        "items": [{"key": "breathe.prompt", "text": source_text}],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert not llm_called
    assert data["translations"]["breathe.prompt"]["status"] == "machine"
    assert data["translations"]["breathe.prompt"]["text"] == translated_text


def test_cache_miss_when_source_text_changes(client, setup_test_db):
    cache = setup_test_db
    old_text = "Take a breath"
    cache.save_machine_translation("breathe.prompt", "hi", old_text, "पुराना अनुवाद")

    new_text = "Take two deep breaths slowly"
    llm_called = False

    async def mock_llm(items, lang, attempt):
        nonlocal llm_called
        llm_called = True
        return {"breathe.prompt": "धीरे-धीरे दो गहरी सांसें लें"}

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "language": "hi",
        "items": [{"key": "breathe.prompt", "text": new_text}],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert llm_called
    assert data["translations"]["breathe.prompt"]["status"] == "machine"
    assert data["translations"]["breathe.prompt"]["text"] == "धीरे-धीरे दो गहरी सांसें लें"

    # Verify cache updated with new hash
    record = cache.get_translation("breathe.prompt", "hi")
    assert record["source_hash"] == compute_source_hash(new_text)


def test_reviewed_row_always_wins_over_machine(setup_test_db):
    cache = setup_test_db
    key = "critical.label"
    lang = "ta"
    source = "Contact your midwife"

    # Save as reviewed
    cache.save_machine_translation(key, lang, source, "முதல் மொழிபெயர்ப்பு")
    cache.update_reviewed_translation(key, lang, "மருத்துவரைத் தொடர்பு கொள்ளவும்")

    # Attempt to overwrite with save_machine_translation
    cache.save_machine_translation(key, lang, source, "புதிய இயந்திர மொழிபெயர்ப்பு")

    # Verify reviewed text persists
    record = cache.get_translation(key, lang)
    assert record["status"] == "reviewed"
    assert record["translated_text"] == "மருத்துவரைத் தொடர்பு கொள்ளவும்"
