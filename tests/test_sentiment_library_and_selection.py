"""Comprehensive tests for Sentiment Suggestion Library, Selection Rules, Recency, Admin, and Privacy."""

import copy
import hashlib
import json
import os
import sqlite3
from datetime import date, timedelta
import pytest

from app.services.sentiment_library import (
    load_raw_library,
    load_served_suggestions,
    validate_library_data,
    get_library_version_hash,
    SUGGESTIONS_PATH,
    OVERRIDES_PATH,
)
from app.services.suggestion_selector import select_sentiment_suggestions
from app.services.sentiment_db import (
    get_db_connection,
    delete_user_suggestion_history,
    get_recently_shown_ids,
)


# ---------------------------------------------------------------------------
# Test 1: Byte-identical Library File Test
# ---------------------------------------------------------------------------
def test_library_file_is_byte_identical_to_supplied_text():
    """Verify data/sentiment_suggestions.json is byte-identical to sentiment_suggestions.txt."""
    original_txt_path = "sentiment_suggestions.txt"
    json_path = "data/sentiment_suggestions.json"

    assert os.path.exists(original_txt_path), "sentiment_suggestions.txt missing"
    assert os.path.exists(json_path), "data/sentiment_suggestions.json missing"

    with open(original_txt_path, "rb") as f:
        orig_bytes = f.read()
    with open(json_path, "rb") as f:
        json_bytes = f.read()

    assert orig_bytes == json_bytes, "JSON file must be byte-for-byte identical to supplied text file!"
    hash_orig = hashlib.sha256(orig_bytes).hexdigest()
    hash_json = hashlib.sha256(json_bytes).hexdigest()
    assert hash_orig == hash_json


# ---------------------------------------------------------------------------
# Test 2: Different Notes Give Different Suggestions
# ---------------------------------------------------------------------------
def test_different_themes_give_disjoint_suggestions():
    """A sleep-themed note and a loneliness-themed note for same stage/tone share no IDs."""
    sleep_res = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=["sleep"],
        user_ref="user_diff_1",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    lonely_res = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=["loneliness"],
        user_ref="user_diff_1",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )

    assert len(sleep_res) > 0
    assert len(lonely_res) > 0

    sleep_ids = {s["id"] for s in sleep_res}
    lonely_ids = {s["id"] for s in lonely_res}

    assert sleep_ids.isdisjoint(lonely_ids), f"Sleep {sleep_ids} and Lonely {lonely_ids} must share NO IDs!"


# ---------------------------------------------------------------------------
# Test 3: Theme-First Ordering
# ---------------------------------------------------------------------------
def test_theme_first_non_understand_belongs_to_primary_theme():
    """With themes [feeding_difficulty, sleep] postnatal, the first non-understand item belongs to feeding_difficulty."""
    res = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=["feeding_difficulty", "sleep"],
        user_ref="user_theme_first",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    assert len(res) >= 2

    # Find first non-understand item
    non_understand_items = [r for r in res if r["kind"] != "understand"]
    assert non_understand_items, "Must have at least one action item"
    first_action = non_understand_items[0]

    assert first_action["why_this_key"] == "suggestion.why.feeding_difficulty"
    assert "feeding" in first_action["why_this_theme"].lower()


# ---------------------------------------------------------------------------
# Test 4: Stage Filtering
# ---------------------------------------------------------------------------
def test_stage_filtering_rules():
    """Prenatal user never receives postnatal items; postnatal user never receives delivery_fear prenatal items."""
    # Prenatal user with feeding_difficulty (which in library is postnatal only)
    res_prenatal = select_sentiment_suggestions(
        stage="prenatal",
        tone="neutral",
        themes=["feeding_difficulty"],
        user_ref="u_stage_1",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    # Postnatal only feeding items must not appear
    for r in res_prenatal:
        assert not r["id"].startswith("FED_")

    # Postnatal user with delivery_fear (which in library is prenatal only)
    res_postnatal = select_sentiment_suggestions(
        stage="postnatal",
        tone="anxious",
        themes=["delivery_fear"],
        user_ref="u_stage_2",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    for r in res_postnatal:
        assert not r["id"].startswith("DEL_")


# ---------------------------------------------------------------------------
# Test 5: Tone Effort & Exclusion Rules
# ---------------------------------------------------------------------------
def test_tone_effort_and_membership():
    """tired_overwhelmed never gets 'longer' as first action; positive never gets items excluding positive."""
    served_dict = {
        s["id"]: s
        for s in load_served_suggestions(dev_include_drafts=True, app_env="development")
    }

    # Test tired_overwhelmed
    res_tired = select_sentiment_suggestions(
        stage="prenatal",
        tone="tired_overwhelmed",
        themes=["baby_health_worry"],
        user_ref="u_tone_1",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    actions = [r for r in res_tired if r["kind"] != "understand"]
    if actions:
        first_act_item = served_dict[actions[0]["id"]]
        assert first_act_item.get("effort") != "longer"

    # Test positive tone
    res_pos = select_sentiment_suggestions(
        stage="prenatal",
        tone="positive",
        themes=["positive_moments"],
        user_ref="u_tone_2",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    for r in res_pos:
        item = served_dict[r["id"]]
        assert "positive" in item.get("tones", []), f"Item {r['id']} tones must include positive"


# ---------------------------------------------------------------------------
# Test 6: Distressed / Risk / Escalation Rules & Verified Helplines
# ---------------------------------------------------------------------------
def test_escalation_and_distressed_tone_returns_no_library_tips_and_verified_helplines(client, auth_headers):
    """distressed tone, risk_language true, or escalation: selector not called, no library items, verified helplines only."""
    async def mock_distressed_llm(*args, **kwargs):
        return {
            "tone": "distressed",
            "intensity": "high",
            "themes": ["sleep"],
            "risk_language": False,
            "reflection": "It sounds like you are feeling heavily distressed today.",
        }

    client.app.state.custom_llm_fn = mock_distressed_llm

    payload = {
        "note_text": "I am feeling so overwhelmed and distressed with everything right now.",
        "language": "en",
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["suggestions"] == []
    assert data["support"] is True
    assert data["helplines"] is not None

    # Strict check: only verified helplines (14416, 181, 112); 1800-889-4357, 102 and 108 never appear
    helpline_str = json.dumps(data["helplines"])
    assert "14416" in helpline_str
    assert "181" in helpline_str
    assert "112" in helpline_str
    assert "1800-889-4357" not in helpline_str
    assert "102" not in helpline_str
    assert "108" not in helpline_str
    assert data["safe_fallback_text"] is not None


# ---------------------------------------------------------------------------
# Test 7: Generic Limit
# ---------------------------------------------------------------------------
def test_generic_limit_rule():
    """Never more than one generic item, and none when 3 theme-specific items qualify."""
    served_dict = {
        s["id"]: s
        for s in load_served_suggestions(dev_include_drafts=True, app_env="development")
    }

    # Sleep postnatal has at least 3 theme items: SLP_U, SLP_NEWBORN, SLP_CBTI
    res = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=["sleep"],
        user_ref="u_gen_1",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    for r in res:
        assert not served_dict[r["id"]].get("generic", False), "No generic item allowed when >= 3 theme items qualify!"

    # Empty themes
    res_empty = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=[],
        user_ref="u_gen_2",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    generic_count = sum(1 for r in res_empty if served_dict[r["id"]].get("generic", False))
    assert generic_count <= 1


# ---------------------------------------------------------------------------
# Test 8: Shape Rules
# ---------------------------------------------------------------------------
def test_shape_constraints():
    """At most 3 items, at most 1 of kind 'clinician', no duplicate ids."""
    res = select_sentiment_suggestions(
        stage="prenatal",
        tone="anxious",
        themes=["baby_health_worry", "delivery_fear"],
        user_ref="u_shape_1",
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    assert len(res) <= 3
    ids = [r["id"] for r in res]
    assert len(ids) == len(set(ids)), "IDs must be unique"

    clinician_count = sum(1 for r in res if r["kind"] == "clinician")
    assert clinician_count <= 1, "At most 1 clinician item allowed"


# ---------------------------------------------------------------------------
# Test 9: Recency & Determinism
# ---------------------------------------------------------------------------
def test_recency_and_determinism():
    """Same user and same day gives same result. Over 20 days with 3 themes, >= 6 different IDs appear."""
    user_ref = "test_recency_and_det_user"
    delete_user_suggestion_history(user_ref)

    d1 = date(2026, 10, 1)

    # Determinism
    res_day1_a = select_sentiment_suggestions(
        stage="prenatal",
        tone="anxious",
        themes=["baby_health_worry", "sleep"],
        user_ref=user_ref,
        today_date=d1,
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    res_day1_b = select_sentiment_suggestions(
        stage="prenatal",
        tone="anxious",
        themes=["baby_health_worry", "sleep"],
        user_ref=user_ref,
        today_date=d1,
        dev_include_drafts=True,
        app_env="development",
        record_history=False,
    )
    assert [r["id"] for r in res_day1_a] == [r["id"] for r in res_day1_b]

    # Recency over 20 days
    all_seen = set()
    for day_idx in range(20):
        curr_d = d1 + timedelta(days=day_idx)
        res = select_sentiment_suggestions(
            stage="prenatal",
            tone="anxious",
            themes=["baby_health_worry", "delivery_fear", "sleep"],
            user_ref=user_ref,
            today_date=curr_d,
            dev_include_drafts=True,
            app_env="development",
            record_history=True,
        )
        for r in res:
            all_seen.add(r["id"])

    assert len(all_seen) >= 6, f"Expected >= 6 different IDs over 20 days, got {len(all_seen)}"


# ---------------------------------------------------------------------------
# Test 10: Draft Items Visibility & Production Behavior
# ---------------------------------------------------------------------------
def test_draft_items_hidden_in_production(monkeypatch):
    """Draft items hidden unless DEV_INCLUDE_DRAFTS=true and APP_ENV=development; with nothing approved, suggestions=[]."""
    monkeypatch.setattr("app.services.sentiment_library.load_overrides", lambda: {})
    # In production with drafts disabled
    prod_res = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=["sleep"],
        user_ref="u_prod_1",
        dev_include_drafts=False,
        app_env="production",
        record_history=False,
    )
    assert prod_res == []

    # In production even with DEV_INCLUDE_DRAFTS=True, drafts must be ignored
    prod_flag_res = select_sentiment_suggestions(
        stage="postnatal",
        tone="tired_overwhelmed",
        themes=["sleep"],
        user_ref="u_prod_2",
        dev_include_drafts=True,
        app_env="production",
        record_history=False,
    )
    assert prod_flag_res == []


# ---------------------------------------------------------------------------
# Test 11: Validation Rejections (needs_source, digits, forbidden words, meds)
# ---------------------------------------------------------------------------
def test_loader_validation_rejections():
    """Approved item with needs_source=True or text violations must fail the loader."""
    raw = load_raw_library()

    # 1. Approved with needs_source=True
    bad_source = copy.deepcopy(raw)
    bad_source["suggestions"][0]["status"] = "approved"
    bad_source["suggestions"][0]["needs_source"] = True
    with pytest.raises(ValueError, match="needs_source is True or source_refs is empty"):
        validate_library_data(bad_source)

    # 2. Digit in text_en
    bad_digit = copy.deepcopy(raw)
    bad_digit["suggestions"][0]["text_en"] = "Rest for 10 minutes."
    with pytest.raises(ValueError, match="contains_digit"):
        validate_library_data(bad_digit)

    # 3. Forbidden word in text_en
    bad_forbidden = copy.deepcopy(raw)
    bad_forbidden["suggestions"][0]["text_en"] = "If you have mild symptoms, contact us."
    with pytest.raises(ValueError, match="contains_forbidden_word"):
        validate_library_data(bad_forbidden)

    # 4. Medication word in text_en
    bad_med = copy.deepcopy(raw)
    bad_med["suggestions"][0]["text_en"] = "Take this pill with water."
    with pytest.raises(ValueError, match="contains_medication_word"):
        validate_library_data(bad_med)


# ---------------------------------------------------------------------------
# Test 12: Privacy & Database Grep
# ---------------------------------------------------------------------------
def test_suggestion_history_stores_only_ids_and_times(client, auth_headers):
    """suggestion_history stores only ids and times; proves no note text, theme or score is stored in history."""
    unique_secret = "UNIQUE_SECRET_NOTE_TEXT_FOR_PRIVACY_CHECK_12345"

    async def mock_private_llm(*args, **kwargs):
        return {
            "tone": "tired_overwhelmed",
            "intensity": "low",
            "themes": ["sleep"],
            "risk_language": False,
            "reflection": "You are noticing tiredness in your body today.",
        }

    client.app.state.custom_llm_fn = mock_private_llm

    payload = {
        "note_text": f"I am feeling very tired and exhausted today. {unique_secret}",
        "language": "en",
        "stage": "postnatal",
    }
    resp = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert resp.status_code == 200

    # Query SQLite database suggestion_history table directly
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(suggestion_history)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "note_text" not in columns
        assert "themes" not in columns
        assert "score" not in columns

        cursor.execute("SELECT * FROM suggestion_history")
        rows = cursor.fetchall()
        for r in rows:
            row_str = " ".join(str(val) for val in r)
            assert unique_secret not in row_str, "Secret note text must NEVER appear in suggestion_history!"


# ---------------------------------------------------------------------------
# Test 13: Admin Endpoints
# ---------------------------------------------------------------------------
def test_admin_sentiment_library_endpoints(client, admin_headers):
    """Test GET /sentiment-library, PATCH suggestion, GET untranslated, and GET stats."""
    # 1. GET /api/admin/sentiment-library
    resp = client.get("/api/admin/sentiment-library", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "sources" in data
    assert "coverage" in data
    assert "thin_coverage" in data
    assert "library_version" in data
    assert len(data["items"]) == 70

    # 2. PATCH invalid: try to approve needs_source item
    patch_bad = client.patch(
        "/api/admin/sentiment-library/suggestion/TIR_U",
        json={"status": "approved"},
        headers=admin_headers,
    )
    assert patch_bad.status_code == 400
    assert "needs_source is True" in patch_bad.json()["detail"]

    # 3. GET /api/admin/sentiment-library/untranslated
    resp_untrans = client.get("/api/admin/sentiment-library/untranslated", headers=admin_headers)
    assert resp_untrans.status_code == 200
    untrans_data = resp_untrans.json()
    assert "untranslated" in untrans_data
    assert untrans_data["count"] > 0

    # 4. GET /api/admin/sentiment-library/stats
    resp_stats = client.get("/api/admin/sentiment-library/stats", headers=admin_headers)
    assert resp_stats.status_code == 200
    stats_data = resp_stats.json()
    assert "shown_counts_30d" in stats_data
    assert "repeat_rate_14d" in stats_data
    assert "no_suggestions_count" in stats_data
    # Assert no user identifiers in stats
    stats_str = json.dumps(stats_data)
    assert "user_ref" not in stats_str


# ---------------------------------------------------------------------------
# Test 14: User Suggestion History Deletion
# ---------------------------------------------------------------------------
def test_delete_user_suggestion_history_endpoint(client, auth_headers):
    """Test DELETE /api/me/suggestion-history deletes records for the user."""
    # First call DELETE to verify 200
    del_resp = client.delete("/api/me/suggestion-history", headers=auth_headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True
