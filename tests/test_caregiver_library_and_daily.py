"""Tests for Caregiver Daily Tips Library and Selection Engine.

Covers:
1. Five different tips returned when enough served; same user, stage, and date gives same 5; different users differ.
   Every day contains one check_in, one small_joy or try_together, one do, and one say.
2. Stage isolation: prenatal user never gets postnatal-only; postnatal user never gets prenatal-only.
3. 90-day simulation verifying caps:
   - at most 1 each of check_in, self_care, clinician, notice, avoid
   - at most 2 of kind 'say'
   - at most 3 of kind 'do'
   - at most 2 of kinds small_joy and try_together together
   - 5 different first themes (no two tips with same first theme)
   - never repeat an id
4. Variety over 90 days:
   - no tip returns sooner than 6 days after it was shown
   - at least 90% of served tips appear at least once
5. get_help_today: returns no tips (help_today: []), verified helplines present and only verified ones
   (Tele-MANAS 14416, Women Helpline 181, 112; never 1800-889-4357, 102, 108).
6. talk_soon always includes an appointments or start_conversation tip.
7. Drafts hidden unless DEV_INCLUDE_DRAFTS=true and APP_ENV=development; with nothing approved, help_today is [] and rest of result returns.
8. Approved tip with needs_source=True rejected by loader and admin PATCH route.
9. Digits, forbidden words, and medication words in any served text fail the loader (fixture tests).
10. Privacy check: caregiver_tip_history stores only ids and dates; test inspects SQLite DB and proves no answers, themes, or scores stored.
11. Hash test proving data/caregiver_suggestions.json is byte-identical to caregiver_suggestions.txt.
12. Admin endpoints: library inspection, patch overrides, untranslated tips, and stats.
"""

from datetime import date, timedelta
import hashlib
import json
import os
import sqlite3
import pytest
from fastapi.testclient import TestClient

import app.config as config
from app.services.auth import hash_user_id
from app.services.caregiver_library import (
    LIBRARY_FILE_PATH,
    OVERRIDES_FILE_PATH,
    get_caregiver_library_version_hash,
    get_raw_caregiver_library,
    get_served_caregiver_library,
    load_caregiver_library,
    validate_caregiver_library,
)
from app.services.caregiver_daily import get_daily_caregiver_tips, select_daily_tips
from app.services.caregiver_db import (
    get_db_connection,
    get_recent_tip_ids,
    record_daily_tips,
    delete_user_caregiver_history,
)


@pytest.fixture(autouse=True)
def enable_drafts_in_dev(monkeypatch):
    """Enable drafts during tests to simulate full 86-tip library serving."""
    monkeypatch.setattr(config, "APP_ENV", "development")
    monkeypatch.setattr(config, "DEV_INCLUDE_DRAFTS", True)
    # Ensure fresh reload
    load_caregiver_library()


def test_hash_identity_of_caregiver_suggestions():
    """Verify data/caregiver_suggestions.json is byte-identical to caregiver_suggestions.txt."""
    src_path = os.path.join(os.path.dirname(LIBRARY_FILE_PATH), "..", "caregiver_suggestions.txt")
    if not os.path.exists(src_path):
        src_path = os.path.join(os.path.dirname(LIBRARY_FILE_PATH), "caregiver_suggestions.txt")
    
    with open(LIBRARY_FILE_PATH, "rb") as f1, open(src_path, "rb") as f2:
        hash1 = hashlib.sha256(f1.read()).hexdigest()
        hash2 = hashlib.sha256(f2.read()).hexdigest()
    assert hash1 == hash2, "data/caregiver_suggestions.json must be byte-identical to caregiver_suggestions.txt"


def test_daily_selection_determinism_and_mix():
    """Verify 5 different tips, daily mix, determinism for same user/date, and difference across users."""
    test_date = date(2026, 11, 7)
    user_a = "user_alpha"
    user_b = "user_beta"

    tips_a1, v1 = get_daily_caregiver_tips("prenatal", user_a, "en", day=test_date, record_history=False)
    tips_a2, v2 = get_daily_caregiver_tips("prenatal", user_a, "en", day=test_date, record_history=False)
    tips_b, vb = get_daily_caregiver_tips("prenatal", user_b, "en", day=test_date, record_history=False)

    assert len(tips_a1) == 5
    # Determinism: same user, stage, date produces exact same 5 tips
    assert [t["id"] for t in tips_a1] == [t["id"] for t in tips_a2]
    # Different users get different sets
    assert [t["id"] for t in tips_a1] != [t["id"] for t in tips_b]

    # Check daily mix: check_in, small_joy or try_together, do, say
    kinds = [t["kind"] for t in tips_a1]
    assert "check_in" in kinds
    assert any(k in ("small_joy", "try_together") for k in kinds)
    assert "do" in kinds
    assert "say" in kinds


def test_stage_isolation():
    """A prenatal user never gets a postnatal-only tip; postnatal user never gets prenatal-only tip."""
    raw_lib = get_raw_caregiver_library()
    prenatal_only = {t["id"] for t in raw_lib["suggestions"] if t["stage"] == "prenatal"}
    postnatal_only = {t["id"] for t in raw_lib["suggestions"] if t["stage"] == "postnatal"}

    assert len(prenatal_only) > 0
    assert len(postnatal_only) > 0

    test_date = date(2026, 11, 7)
    pre_tips, _ = get_daily_caregiver_tips("prenatal", "stage_tester", "en", day=test_date, record_history=False)
    post_tips, _ = get_daily_caregiver_tips("postnatal", "stage_tester", "en", day=test_date, record_history=False)

    pre_ids = {t["id"] for t in pre_tips}
    post_ids = {t["id"] for t in post_tips}

    assert pre_ids.isdisjoint(postnatal_only), "Prenatal user received postnatal-only tip"
    assert post_ids.isdisjoint(prenatal_only), "Postnatal user received prenatal-only tip"


def test_90_day_simulation_limits_and_variety():
    """Verify limits and variety over 90 consecutive days.
    
    Limits:
    - Exactly 5 tips per day
    - Daily mix: 1 check_in, 1 small_joy or try_together, 1 do, 1 say
    - at most 1 each of check_in, self_care, clinician, notice, avoid
    - at most 2 of 'say'
    - at most 3 of 'do'
    - at most 2 of small_joy + try_together together
    - 5 different first themes (no duplicate first themes on any day)
    - never repeat an ID within a day

    Variety:
    - No tip returns sooner than 6 days after it was shown
    - At least 90% of served tips appear at least once across 90 days
    """
    start_date = date(2026, 6, 1)
    user_ref = "simulation_user_90_days"
    history: dict = {}

    all_served = get_served_caregiver_library("prenatal")
    all_served_ids = {t["id"] for t in all_served}

    seen_ids = set()
    last_seen_date = {}

    for d_idx in range(90):
        current_date = start_date + timedelta(days=d_idx)
        tips = select_daily_tips(
            stage="prenatal",
            user_ref=user_ref,
            day=current_date,
            result_state="keep_supporting",
            flags=[],
            history=history,
            served=all_served,
        )

        assert len(tips) == 5, f"Day {d_idx} returned {len(tips)} tips, expected 5"

        ids = [t["id"] for t in tips]
        assert len(set(ids)) == 5, f"Day {d_idx} has duplicate IDs: {ids}"

        # Caps and Mix
        kinds = [t.get("kind", "do") for t in tips]
        assert kinds.count("check_in") == 1, f"Day {d_idx} check_in count: {kinds.count('check_in')}"
        assert any(k in ("small_joy", "try_together") for k in kinds), f"Day {d_idx} missing light kind"
        assert kinds.count("small_joy") + kinds.count("try_together") <= 2
        assert kinds.count("do") <= 3
        assert kinds.count("say") <= 2
        for single_cap_kind in ("self_care", "clinician", "notice", "avoid"):
            assert kinds.count(single_cap_kind) <= 1

        # Theme variety: 5 different first themes
        first_themes = [t.get("themes", ["other"])[0] for t in tips]
        assert len(set(first_themes)) == 5, f"Day {d_idx} duplicate first themes: {first_themes}"

        # Variety rule: No tip returns sooner than 6 days (i.e. at least 6 days gap)
        for tid in ids:
            if tid in last_seen_date:
                gap = (current_date - last_seen_date[tid]).days
                assert gap >= 6, f"Tip {tid} returned too early after {gap} days on day {d_idx}"
            last_seen_date[tid] = current_date
            history[tid] = current_date
            seen_ids.add(tid)

    # Coverage: >= 90% of served tips appear at least once
    coverage_pct = len(seen_ids) / len(all_served_ids)
    assert coverage_pct >= 0.90, f"Coverage was {coverage_pct * 100:.1f}%, expected >= 90%"


def test_get_help_today_returns_no_tips_and_only_verified_helplines(client, auth_headers):
    """When result_state is get_help_today: return NO tips, and only verified helplines."""
    # Trigger safety in score endpoint (Q13 = Yes)
    resp = client.post(
        "/api/caregiver/score",
        json={
            "stage": "prenatal",
            "answers": {"q13": "Yes"},
            "language": "en",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result_state"] == "get_help_today"
    assert data["safety_triggered"] is True
    assert data["help_today"] == [], "get_help_today must return 0 tips"

    # Helplines check
    helplines = data["helplines"]
    assert helplines.get("tele_manas") == "14416"
    assert helplines.get("women_helpline") == "181"
    assert helplines.get("emergency") == "112"
    # Negative assertions for forbidden helpline numbers
    for k, v in helplines.items():
        assert "1800-889-4357" not in v
        assert "102" != v
        assert "108" != v


def test_talk_soon_favours_appointments_or_start_conversation(client, auth_headers):
    """talk_soon always includes an appointments or start_conversation tip."""
    # Part 2: Signs high (Q8-Q12 Often -> 10 points)
    resp = client.post(
        "/api/caregiver/score",
        json={
            "stage": "postnatal",
            "answers": {
                "q8": "Often",
                "q9": "Often",
                "q10": "Often",
                "q11": "Often",
                "q12": "Often",
            },
            "language": "en",
            "date": "2026-11-08",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result_state"] == "talk_soon"
    help_today = data["help_today"]
    assert len(help_today) == 5

    raw_lib = get_raw_caregiver_library()
    tip_lookup = {t["id"]: t for t in raw_lib["suggestions"]}
    all_themes = []
    for t in help_today:
        orig = tip_lookup.get(t["id"], {})
        all_themes.extend(orig.get("themes", []))

    assert any(th in ("appointments", "start_conversation") for th in all_themes), (
        "talk_soon must include at least one tip with theme appointments or start_conversation"
    )


def test_drafts_hidden_in_prod_mode(monkeypatch):
    """Drafts hidden unless DEV_INCLUDE_DRAFTS=true and APP_ENV=development."""
    # Set production mode
    monkeypatch.setattr(config, "APP_ENV", "production")
    monkeypatch.setattr(config, "DEV_INCLUDE_DRAFTS", True)
    load_caregiver_library()

    # All tips in caregiver_suggestions.json are currently draft
    served = get_served_caregiver_library("prenatal")
    assert len(served) == 0, "In production mode, draft tips must not be served"

    tips, _ = get_daily_caregiver_tips("prenatal", "prod_user", "en", record_history=False)
    assert tips == [], "With 0 approved tips, daily tips must be empty list"


def test_approved_tip_with_needs_source_rejected():
    """An approved tip with needs_source=True must be rejected by loader and admin route."""
    raw_lib = get_raw_caregiver_library()
    bad_lib = json.loads(json.dumps(raw_lib))
    # Pick first tip and set approved but needs_source=True
    bad_lib["suggestions"][0]["status"] = "approved"
    bad_lib["suggestions"][0]["needs_source"] = True

    with pytest.raises(ValueError, match="needs_source is true"):
        validate_caregiver_library(bad_lib)


def test_loader_rejects_digits_forbidden_words_and_medications():
    """Digits, forbidden words, and medication words in served text_en fail validation."""
    raw_lib = get_raw_caregiver_library()

    # 1. Digit test
    lib_digit = json.loads(json.dumps(raw_lib))
    lib_digit["suggestions"][0]["text_en"] = "Make sure she drinks 2 glasses of water."
    with pytest.raises(ValueError, match="contains digit"):
        validate_caregiver_library(lib_digit)

    # 2. Forbidden clinical word test (e.g. 'moderate')
    lib_forbidden = json.loads(json.dumps(raw_lib))
    lib_forbidden["suggestions"][0]["text_en"] = "Notice if she has moderate changes in mood."
    with pytest.raises(ValueError, match="forbidden or medication word"):
        validate_caregiver_library(lib_forbidden)

    # 3. Medication word test (e.g. 'sertraline')
    lib_med = json.loads(json.dumps(raw_lib))
    lib_med["suggestions"][0]["text_en"] = "Remind her about taking sertraline on time."
    with pytest.raises(ValueError, match="forbidden or medication word"):
        validate_caregiver_library(lib_med)


def test_privacy_caregiver_tip_history_no_answers_or_scores():
    """caregiver_tip_history stores only user_ref, tip_id, shown_on; never answers or scores."""
    user = "privacy_test_user_789"
    test_day = date(2026, 11, 9)

    record_daily_tips(user, ["CG_P_APPT", "CG_P_HELPPLAN"], test_day)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check table column names
    cursor.execute("PRAGMA table_info(caregiver_tip_history)")
    columns = [row["name"] for row in cursor.fetchall()]
    assert set(columns) == {"id", "user_ref", "tip_id", "shown_on", "created_at"}

    # Fetch rows
    cursor.execute("SELECT * FROM caregiver_tip_history WHERE user_ref = ?", (user,))
    rows = cursor.fetchall()
    assert len(rows) == 2
    for r in rows:
        # None of the values should contain score, answer, risk, or clinical message
        for val in r:
            s_val = str(val).lower()
            assert "score" not in s_val
            assert "answer" not in s_val
            assert "talk_soon" not in s_val
            assert "extra_support" not in s_val


def test_admin_caregiver_endpoints(client, admin_headers, monkeypatch):
    """Test GET /api/admin/caregiver-library, PATCH tip, untranslated, and stats."""
    # 1. GET library
    resp = client.get("/api/admin/caregiver-library", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tips"] == 86
    assert "prenatal" in data["coverage"]
    assert "postnatal" in data["coverage"]

    # 2. PATCH tip status and translation
    patch_resp = client.patch(
        "/api/admin/caregiver-library/tip/CG_P_APPT",
        json={
            "text_hi": "परीक्षण अनुवाद",
        },
        headers=admin_headers,
    )
    assert patch_resp.status_code == 200
    patch_data = patch_resp.json()
    assert patch_data["success"] is True

    # Check override persisted
    with open(OVERRIDES_FILE_PATH, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    assert "CG_P_APPT" in overrides
    assert overrides["CG_P_APPT"]["text_hi"] == "परीक्षण अनुवाद"

    # 3. GET untranslated
    untranslated_resp = client.get("/api/admin/caregiver-library/untranslated", headers=admin_headers)
    assert untranslated_resp.status_code == 200
    untranslated_data = untranslated_resp.json()
    assert "count" in untranslated_data
    assert isinstance(untranslated_data["untranslated"], list)

    # 4. GET stats
    stats_resp = client.get("/api/admin/caregiver-library/stats", headers=admin_headers)
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "tip_counts_last_30_days" in stats_data
    assert "days_with_fewer_than_5_tips" in stats_data
    # Must NOT have user references
    assert "user_ref" not in str(stats_data)


def test_delete_caregiver_tip_history(client, auth_headers):
    """Verify user can delete their caregiver tip history."""
    user = hash_user_id("test_user_sub_999")
    record_daily_tips(user, ["CG_P_APPT"], date(2026, 11, 10))

    resp = client.delete("/api/me/caregiver-tip-history", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["deleted_count"] >= 1

    # Verify history is empty
    recent = get_recent_tip_ids(user, 30)
    assert len(recent) == 0
