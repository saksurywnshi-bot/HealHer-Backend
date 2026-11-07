"""Comprehensive Tests for Questionnaire Guidance AI Personalisation.

Tests cover all clinical safety guardrails, prompt boundaries, deterministic fallbacks,
reword validation rules, cache invalidation, ownership verification, and drafting script.
All model calls are strictly mocked (zero real API calls).
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.config import (
    ADMIN_TOKEN,
    GUIDANCE_FIXED_THEMES,
)
from app.main import app
from app.services.guidance_db import (
    init_guidance_db,
    clear_guidance_cache,
    save_questionnaire_result,
    get_guidance_stats,
)
from app.limiter import limiter
from app.services.guidance_library import (
    get_library_version_hash,
    get_verified_helplines,
    SUGGESTIONS_PATH,
    OVERRIDES_PATH,
)
from app.services.guidance_validator import (
    validate_reword_text,
    validate_selection_payload,
)
from app.graphs.guidance_graph import build_select_prompt_payload

client = TestClient(app)

DEV_HEADERS = {"x-user-ref": "guidance_user_test_1"}
OTHER_USER_HEADERS = {"x-user-ref": "guidance_user_other_9"}
ADMIN_HEADERS = {"x-admin-token": ADMIN_TOKEN or "test_admin_token"}


@pytest.fixture(autouse=True)
def setup_db():
    init_guidance_db()
    clear_guidance_cache()
    limiter.reset()


# ---------------------------------------------------------------------------
# Test 1: GUIDANCE_AI_ENABLED=false -> Deterministic result & model mock called 0 times
# ---------------------------------------------------------------------------
def test_guidance_ai_disabled_returns_deterministic(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "false")
    monkeypatch.setenv("GUIDANCE_AI_REWORD", "false")

    # Score an assessment
    score_res = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "phq9",
            "answers": {"1": 1, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0},
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert score_res.status_code == 200
    res_id = score_res.json()["result_id"]

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        guidance_res = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": res_id, "language": "en", "themes": ["sleep"]},
        )
        assert guidance_res.status_code == 200
        data = guidance_res.json()
        assert data["personalised"] is False
        assert data["mode"] == "none"
        assert len(data["suggestions"]) > 0
        mock_llm.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: safety_triggered, see_clinician_soon, see_clinician_urgent -> model called 0 times
# ---------------------------------------------------------------------------
def test_safety_and_urgent_escalation_gates_model_completely(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")

    # Case A: Safety triggered (PHQ-9 Q9 > 0)
    score_res = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "phq9",
            "answers": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 1},
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert score_res.status_code == 200
    res_id_safety = score_res.json()["result_id"]
    assert score_res.json()["safety_triggered"] is True

    # Case B: see_clinician_soon (score 16)
    score_res_soon = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "phq9",
            "answers": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2, "9": 0},
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert score_res_soon.status_code == 200
    res_id_soon = score_res_soon.json()["result_id"]
    assert score_res_soon.json()["escalation"] == "see_clinician_soon"

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        for rid in (res_id_safety, res_id_soon):
            resp = client.post(
                "/api/questionnaire/guidance",
                headers=DEV_HEADERS,
                json={"result_id": rid, "language": "en"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["personalised"] is False
            assert data["mode"] == "none"
            assert "Tele-MANAS" in data["safe_fallback_text"]
            assert data["helplines"]["emergency"] == "112"
            assert data["helplines"]["tele_manas"] == "14416"

        mock_llm.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Malformed Model Outputs (unknown id, duplicate, >4 ids, empty, extra keys, free text)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "malformed_json",
    [
        '{"selected_ids": ["SUG_NONEXISTENT_999"], "reason_tags": []}',
        '{"selected_ids": ["SUG_SLEEP_PROTECT", "SUG_SLEEP_PROTECT"], "reason_tags": []}',
        '{"selected_ids": ["SUG_SLEEP_PROTECT", "SUG_TALK_TRUSTED", "SUG_TIME_FOR_SELF", "SUG_ONE_WORRY_STEP", "SUG_BE_GENTLE"], "reason_tags": []}',
        '{"selected_ids": [], "reason_tags": []}',
        '{"selected_ids": ["SUG_SLEEP_PROTECT"], "reason_tags": [], "extra_clinical_advice": "You have depression"}',
        'Here is my recommendation: You should sleep more and take medication.',
    ],
)
def test_model_invalid_output_falls_back_to_deterministic(monkeypatch, malformed_json):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")
    monkeypatch.setenv("GUIDANCE_AI_REWORD", "false")

    rid = save_questionnaire_result(
        user_ref="guidance_user_test_1",
        instrument="phq9",
        total_score=6,
        band_id="mild",
        escalation="monitor",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1},
        result_dict={},
    )

    mock_resp = MagicMock()
    mock_resp.content = malformed_json

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        resp = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "en", "themes": ["sleep"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["personalised"] is False
        assert data["mode"] == "none"
        assert len(data["suggestions"]) > 0


# ---------------------------------------------------------------------------
# Test 4: Moderate-band keeps clinician-contact suggestion first even if ordered last
# ---------------------------------------------------------------------------
def test_moderate_band_keeps_clinician_contact_first(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")
    monkeypatch.setenv("GUIDANCE_AI_REWORD", "false")

    rid = save_questionnaire_result(
        user_ref="guidance_user_test_1",
        instrument="phq9",
        total_score=11,
        band_id="moderate",
        escalation="see_clinician",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 1},
        result_dict={},
    )

    # Model purposely returns clinician contact last
    model_response = {
        "selected_ids": ["SUG_SLEEP_PROTECT", "SUG_TALK_TRUSTED", "SUG_CLINICIAN_CONTACT"],
        "reason_tags": ["sleep"],
    }
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(model_response)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        resp = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "en", "themes": ["sleep"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["personalised"] is True
        # Clinician contact must be at index 0!
        assert data["suggestions"][0]["id"] == "SUG_CLINICIAN_CONTACT"


# ---------------------------------------------------------------------------
# Test 5: Prompt contains ONLY allowed fields (leakage test)
# ---------------------------------------------------------------------------
def test_prompt_contains_only_allowed_fields_no_pii_or_scores():
    fake_state = {
        "instrument": "phq9",
        "stage": "postnatal",
        "language": "hi",
        "band_id": "mild",
        "escalation": "monitor",
        "themes": ["sleep", "invalid_random_theme"],
        "candidates": [
            {"id": "SUG_SLEEP_PROTECT", "category": "sleep", "text": {"en": "Prioritise rest."}},
        ],
        # Surrounding PII and score data that must NEVER reach the prompt
        "user_name": "Pooja Sharma",
        "user_email": "pooja.test@example.com",
        "user_ref": "sha256_secret_hash_value",
        "score": 12,
        "total_score": 12,
        "note_text": "I feel very sad and cried today",
        "free_text": "Secret user journal note",
    }

    prompt_data = build_select_prompt_payload(fake_state)

    # Must contain exactly the 7 permitted keys
    allowed_keys = {"instrument", "stage", "language", "band_id", "escalation", "themes", "candidates"}
    assert set(prompt_data.keys()) == allowed_keys

    # Must not contain PII or score anywhere in serialized JSON
    serialized = json.dumps(prompt_data)
    assert "Pooja" not in serialized
    assert "pooja.test@example.com" not in serialized
    assert "sha256_secret_hash_value" not in serialized
    assert "cried today" not in serialized
    assert "Secret user journal" not in serialized
    assert "12" not in serialized
    assert "invalid_random_theme" not in serialized
    assert "sleep" in prompt_data["themes"]


# ---------------------------------------------------------------------------
# Test 6: Mixed evidence item (calming practice) NEVER appears in moderate or higher
# ---------------------------------------------------------------------------
def test_mixed_evidence_item_never_in_moderate_or_higher_band(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")

    rid = save_questionnaire_result(
        user_ref="guidance_user_test_1",
        instrument="phq9",
        total_score=12,
        band_id="moderate",
        escalation="see_clinician",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2},
        result_dict={},
    )

    # Even if model mock tries to return SUG_CALMING_PRACTICE (mixed evidence)
    model_response = {
        "selected_ids": ["SUG_CALMING_PRACTICE", "SUG_CLINICIAN_CONTACT"],
        "reason_tags": [],
    }
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(model_response)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        resp = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        sug_ids = [s["id"] for s in data["suggestions"]]
        assert "SUG_CALMING_PRACTICE" not in sug_ids


# ---------------------------------------------------------------------------
# Test 7: Reword validation strict checks
# ---------------------------------------------------------------------------
def test_reword_validation_rejections():
    orig_en = "Take a few minutes just for yourself each day to pause and breathe."

    # 1. Extra digit not in original
    valid, err = validate_reword_text("Take 5 minutes just for yourself each day.", orig_en, "en")
    assert valid is False
    assert err == "reword_digit"

    # 2. Phone number not in original
    valid, err = validate_reword_text("Call 9876543210 for help.", orig_en, "en")
    assert valid is False
    assert err == "reword_phone"

    # 3. URL not in original
    valid, err = validate_reword_text("Visit https://example.com for advice.", orig_en, "en")
    assert valid is False
    assert err == "reword_url"

    # 4. Forbidden diagnostic word
    valid, err = validate_reword_text("This helps with your mild depression.", orig_en, "en")
    assert valid is False
    assert err == "reword_forbidden_word"

    # 5. Medication word
    valid, err = validate_reword_text("Do not forget your daily medicine tablet.", orig_en, "en")
    assert valid is False
    assert err == "reword_medication_word"

    # 6. Wrong script (Hindi requested, but returned Latin script)
    valid, err = validate_reword_text("Roz apne liye thoda samay nikalein.", orig_en, "hi")
    assert valid is False
    assert err == "reword_script_mismatch"

    # 7. Length > 1.3x original
    long_text = "Take a few minutes just for yourself each day to pause and breathe. " * 3
    valid, err = validate_reword_text(long_text, orig_en, "en")
    assert valid is False
    assert err == "reword_length"

    # 8. Valid Hindi translation
    valid_hi = "हर दिन अपने लिए कुछ शांत मिनट निकालें।"
    valid, err = validate_reword_text(valid_hi, orig_en, "hi")
    assert valid is True
    assert err is None


# ---------------------------------------------------------------------------
# Test 8: Rewording is never used when GUIDANCE_AI_REWORD=false
# ---------------------------------------------------------------------------
def test_rewording_disabled_by_default(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")
    monkeypatch.setenv("GUIDANCE_AI_REWORD", "false")

    rid = save_questionnaire_result(
        user_ref="guidance_user_test_1",
        instrument="phq9",
        total_score=2,
        band_id="minimal",
        escalation="routine",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 1, "2": 1},
        result_dict={},
    )

    mock_resp = MagicMock()
    mock_resp.content = json.dumps({"selected_ids": ["SUG_SLEEP_PROTECT"], "reason_tags": []})

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        resp = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "hi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["personalised"] is True
        assert data["mode"] == "select"
        # Since reword is disabled, uses approved library text for Hindi
        assert "आराम" in data["suggestions"][0]["text"]


# ---------------------------------------------------------------------------
# Test 9: Cache lookup & invalidation on library / override changes
# ---------------------------------------------------------------------------
def test_cache_lookup_and_invalidation(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")
    monkeypatch.setenv("GUIDANCE_AI_REWORD", "false")

    rid = save_questionnaire_result(
        user_ref="guidance_user_test_1",
        instrument="phq9",
        total_score=3,
        band_id="minimal",
        escalation="routine",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 1, "2": 2},
        result_dict={},
    )

    mock_resp = MagicMock()
    mock_resp.content = json.dumps({"selected_ids": ["SUG_SLEEP_PROTECT"], "reason_tags": []})

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        # Call 1: Misses cache, calls model
        r1 = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "en", "themes": ["sleep"]},
        )
        assert r1.status_code == 200
        assert mock_llm.ainvoke.call_count == 1

        # Call 2: Hits cache, model mock NOT called again
        r2 = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "en", "themes": ["sleep"]},
        )
        assert r2.status_code == 200
        assert mock_llm.ainvoke.call_count == 1  # Still 1!

        # Modify library override file to simulate update
        with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps({"SUG_TEST": {"category": "test"}}))

        try:
            # Call 3: Hash changed, cache invalidated -> calls model again
            r3 = client.post(
                "/api/questionnaire/guidance",
                headers=DEV_HEADERS,
                json={"result_id": rid, "language": "en", "themes": ["sleep"]},
            )
            assert r3.status_code == 200
            assert mock_llm.ainvoke.call_count == 2
        finally:
            # Clean up override file
            with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
                f.write("{}")


# ---------------------------------------------------------------------------
# Test 10: Ownership: result_id belonging to another user returns 404
# ---------------------------------------------------------------------------
def test_guidance_ownership_check_404():
    rid = save_questionnaire_result(
        user_ref="user_alice_secret",
        instrument="phq9",
        total_score=4,
        band_id="minimal",
        escalation="routine",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 2, "2": 2},
        result_dict={},
    )

    # Bob attempts to request guidance for Alice's result
    resp = client.post(
        "/api/questionnaire/guidance",
        headers={"x-user-ref": "user_bob_other"},
        json={"result_id": rid, "language": "en"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 11: Timeout / 429 / Exception -> deterministic result with personalised=false
# ---------------------------------------------------------------------------
def test_model_exception_graceful_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AI_ENABLED", "true")

    rid = save_questionnaire_result(
        user_ref="guidance_user_test_1",
        instrument="phq9",
        total_score=3,
        band_id="minimal",
        escalation="routine",
        safety_triggered=False,
        stage="postnatal",
        answers={"1": 1, "2": 2},
        result_dict={},
    )

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("Model connection timed out after 10 seconds"))

    with patch("app.graphs.guidance_graph.ChatGoogleGenerativeAI", return_value=mock_llm):
        resp = client.post(
            "/api/questionnaire/guidance",
            headers=DEV_HEADERS,
            json={"result_id": rid, "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["personalised"] is False
        assert data["mode"] == "none"
        assert len(data["suggestions"]) > 0


# ---------------------------------------------------------------------------
# Test 12: Drafting script always writes 'draft', never touches main library
# ---------------------------------------------------------------------------
def test_drafting_script_safeguards(tmp_path):
    from scripts.draft_library_item import draft_suggestion_item

    mock_draft_output = {
        "id": "SUG_HYDRATION_REST",
        "category": "self_care",
        "text": {"en": "Drink adequate water throughout the day and rest when possible."},
        "evidence": "guideline",
        "evidence_note": "ACOG highlights perinatal hydration and rest.",
        "source_refs": ["ACOG_2023"],
        "status": "approved",  # Model might mistakenly try to return 'approved'
    }

    mock_resp = MagicMock()
    mock_resp.content = json.dumps(mock_draft_output)

    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_resp):
        res = draft_suggestion_item("Adequate hydration is advised during lactation.", "ACOG_2023")
        # Script must enforce 'draft' status regardless of model output
        assert res["status"] == "draft"
        assert res["id"].startswith("SUG_")


def test_drafting_script_refuses_empty_source():
    import subprocess
    cmd = ["python3", "scripts/draft_library_item.py"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "required" in res.stderr.lower() or "error" in res.stderr.lower()


# ---------------------------------------------------------------------------
# Test 13: Responses NEVER contain 1800-889-4357, 102, or 108; contain verified helplines
# ---------------------------------------------------------------------------
def test_helplines_strictly_verified():
    helplines = get_verified_helplines()

    # Verified helpline numbers
    assert helplines["tele_manas"] == "14416"
    assert helplines["emergency"] == "112"
    assert helplines["women_helpline"] == "181"

    # Strictly forbidden helpline numbers
    all_values = " ".join(helplines.values())
    assert "1800-889-4357" not in all_values
    assert "102" not in all_values
    assert "108" not in all_values


# ---------------------------------------------------------------------------
# Test 14: Admin Guidance Stats & Flags Endpoints
# ---------------------------------------------------------------------------
def test_admin_guidance_stats_and_flags():
    # 1. Flags endpoint
    flags_res = client.get("/api/admin/guidance/flags", headers={"x-admin-token": ADMIN_TOKEN})
    assert flags_res.status_code == 200
    flags_data = flags_res.json()
    assert "guidance_ai_enabled" in flags_data
    assert "guidance_ai_reword" in flags_data
    assert flags_data["guidance_cache_days"] >= 1

    # 2. Stats endpoint
    stats_res = client.get("/api/admin/guidance/stats", headers={"x-admin-token": ADMIN_TOKEN})
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "ai_selections_used" in stats_data
    assert "fallbacks" in stats_data
    assert "cache_hits" in stats_data
    assert "validation_failures_by_type" in stats_data


# ---------------------------------------------------------------------------
# Test 15: EPDS Scoring with Prefixed Keys & Guidance Validation
# ---------------------------------------------------------------------------
def test_epds_scoring_with_prefixed_keys_and_guidance():
    # Case A: Score 16 with 'epds10': 1 ("Hardly ever" -> NO safety override)
    score_res = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "epds",
            "answers": {
                "epds1": 2, "epds2": 2, "epds3": 1, "epds4": 2, "epds5": 1,
                "epds6": 2, "epds7": 2, "epds8": 1, "epds9": 2, "epds10": 1,
            },
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert score_res.status_code == 200
    data = score_res.json()
    assert data["total_score"] == 16
    assert data["band_id"] in ("possible_depression", "moderately_severe")
    # epds10 is 1 ("Hardly ever") -> NOT a crisis trigger
    assert data["safety_triggered"] is False
    assert data["escalation"] == "see_clinician"

    # Case B: 'epds10': 2 ("Sometimes" -> active self-harm ideation triggers urgent escalation)
    score_res_crisis = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "epds",
            "answers": {
                "epds1": 2, "epds2": 2, "epds3": 1, "epds4": 2, "epds5": 1,
                "epds6": 2, "epds7": 2, "epds8": 1, "epds9": 2, "epds10": 2,
            },
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert score_res_crisis.status_code == 200
    data_crisis = score_res_crisis.json()
    assert data_crisis["safety_triggered"] is True
    assert data_crisis["escalation"] == "see_clinician_urgent"
    assert len(data_crisis["suggestions"]) > 0
    # First suggestion must be clinical contact for urgent escalation
    assert data_crisis["suggestions"][0]["id"] == "SUG_CLINICIAN_CONTACT"
    assert "tele_manas" in data_crisis["helplines"]
    assert data_crisis["helplines"]["tele_manas"] == "14416"


def test_pbq_scoring_and_storage(client):
    """Test PBQ scoring, band determination, and backend persistence."""
    # 1. PBQ State P1 (Usual range): all forward items 0, all reverse items 5 (scores sum to 0)
    p1_answers = {str(i): 5 for i in range(1, 26)}
    for fwd in [1, 4, 8, 9, 11, 16, 22, 25]:
        p1_answers[str(fwd)] = 0

    res_p1 = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "pbq",
            "answers": p1_answers,
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert res_p1.status_code == 200
    p1_data = res_p1.json()
    assert p1_data["total_score"] == 0
    assert p1_data["band_id"] == "pbq_p1"
    assert p1_data["safety_triggered"] is False
    assert len(p1_data["suggestions"]) == 2
    assert p1_data["result_id"].startswith("qres_")

    # 2. PBQ State P2 (Some strain): total score >= 12
    p2_answers = dict(p1_answers)
    p2_answers["1"] = 3
    p2_answers["4"] = 3
    p2_answers["8"] = 3
    p2_answers["9"] = 3  # total = 12

    res_p2 = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "pbq",
            "answers": p2_answers,
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert res_p2.status_code == 200
    p2_data = res_p2.json()
    assert p2_data["total_score"] == 12
    assert p2_data["band_id"] == "pbq_p2"
    assert len(p2_data["suggestions"]) == 5

    # 3. PBQ State P0 (Safety trigger): Item 18 or 24 != 5 (e.g. option index 3 -> score > 0)
    p0_answers = dict(p1_answers)
    p0_answers["18"] = 3

    res_p0 = client.post(
        "/api/questionnaire/score",
        headers=DEV_HEADERS,
        json={
            "instrument": "pbq",
            "answers": p0_answers,
            "stage": "postnatal",
            "language": "en",
        },
    )
    assert res_p0.status_code == 200
    p0_data = res_p0.json()
    assert p0_data["safety_triggered"] is True
    assert p0_data["band_id"] == "pbq_p0"
    assert p0_data["escalation"] == "see_clinician_urgent"
    assert "14416" in p0_data["helplines"].values()

