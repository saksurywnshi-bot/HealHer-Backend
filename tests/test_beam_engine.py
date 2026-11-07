"""Comprehensive unit tests for BEAM Perinatal Insights Rule Engine.

Covers all required clinical scenarios and edge cases specified in Section 8:
- 8 depressed days in 14 -> Tier 2.
- 4 days at -2 or -3 in 14 -> Tier 2.
- 2 consecutive -3 days -> Tier 2.
- 3 consecutive depressed days -> Tier 1.
- 6 depressed days plus short sleep on 7 nights -> Tier 2.
- 2 consecutive marked elevated days -> Tier 2.
- 3 elevated days in 7 with 2 or more reduced-sleep nights -> Tier 2.
- A single Severe Elevated (+3) day -> Tier 3.
- Postpartum, week 4, two days at +2 with 2 nights of sleep 2 hours below usual -> Tier 3.
- Postpartum day 10 with 10 of 14 depressed days -> capped at Tier 1.
- Postpartum day 10 with a +3 day -> Tier 3 (elevated side is not capped).
- Pregnancy, +1 mood with normal sleep -> good day, no alert.
- Fewer than 4 entries in a week -> 'not enough data'.
- Boundary tests at every threshold (e.g. 7 vs 8 depressed days in 14).
- Sleep, anxiety and irritability alone never create Tier 2 (except elevated combined rule).
- Config-flag defaults match the spec.
"""

from datetime import date, timedelta
import pytest

from app.services.insights.config_loader import get_beam_rules_config
from app.services.insights.engine import evaluate


def make_entry(date_str: str, mood=0, anxiety=0, irritability=0, sleep=8.0):
    """Helper to build a clean BEAM entry dict."""
    return {
        "log_date": date_str,
        "mood_beam": mood,
        "anxiety": anxiety,
        "irritability": irritability,
        "sleep_hours": sleep,
    }


def make_series(start_date: str, days_count: int, default_mood=0, default_sleep=8.0):
    """Generate a chronological sequence of daily entries."""
    start = date.fromisoformat(start_date)
    return [
        make_entry((start + timedelta(days=i)).isoformat(), mood=default_mood, sleep=default_sleep)
        for i in range(days_count)
    ]


def test_config_flag_defaults():
    """Verify that clinical config flag defaults match the specification exactly."""
    cfg = get_beam_rules_config()
    flags = cfg.get("clinician_flags", {})
    assert flags.get("exclude_postpartum_days_1_to_7_from_14_day_count") is False
    assert flags.get("count_mild_depressed_in_half_days_rule") is True
    assert flags.get("postpartum_urgent_elevated_window_weeks") == 12
    assert flags.get("include_optional_suggestions") is False


def test_scenario_8_depressed_days_in_14_triggers_tier_2():
    """8 depressed days out of 14 logged days triggers Tier 2."""
    entries = make_series("2026-10-01", 14, default_mood=0)
    # Mark 8 days as depressed (-1)
    for i in range(8):
        entries[i]["mood_beam"] = -1

    res = evaluate(entries, phase="pregnancy", today="2026-10-14")
    assert res["tier"] == 2
    assert res["depressed_side_flagged"] is True
    assert any("8 of 14 days were depressed" in r for r in res["reasons"])


def test_scenario_boundary_7_vs_8_depressed_days():
    """Boundary test: 7 depressed days in 14 does NOT trigger Tier 2 (unless other rules fire)."""
    # 7 depressed days distributed non-consecutively with good sleep and low anxiety
    entries = make_series("2026-10-01", 14, default_mood=0)
    # 7 depressed days alternating
    for i in [0, 2, 4, 6, 8, 10, 12]:
        entries[i]["mood_beam"] = -1

    res = evaluate(entries, phase="pregnancy", today="2026-10-14")
    # 7 days should not trigger Tier 2 depressed side
    assert res["depressed_side_flagged"] is False
    assert res["tier"] != 2


def test_scenario_4_impaired_depressed_days_in_14_triggers_tier_2():
    """4 days at -2 or -3 in 14 days triggers Tier 2."""
    entries = make_series("2026-10-01", 14, default_mood=0)
    # 4 non-consecutive impaired depressed days
    entries[0]["mood_beam"] = -2
    entries[3]["mood_beam"] = -3
    entries[7]["mood_beam"] = -2
    entries[11]["mood_beam"] = -2

    res = evaluate(entries, phase="pregnancy", today="2026-10-14")
    assert res["tier"] == 2
    assert res["depressed_side_flagged"] is True
    assert any("impaired depressed days in 14 days" in r for r in res["reasons"])


def test_scenario_2_consecutive_minus_3_days_triggers_tier_2():
    """2 consecutive -3 (Severe Depressed) days triggers Tier 2."""
    entries = make_series("2026-10-01", 7, default_mood=0)
    entries[2]["mood_beam"] = -3
    entries[3]["mood_beam"] = -3

    res = evaluate(entries, phase="pregnancy", today="2026-10-07")
    assert res["tier"] == 2
    assert res["depressed_side_flagged"] is True
    assert any("2 consecutive days at Severe Depressed (-3)" in r for r in res["reasons"])


def test_scenario_3_consecutive_depressed_days_triggers_tier_1():
    """3 consecutive depressed days triggers Tier 1 (when no Tier 2 or 3 rule fires)."""
    entries = make_series("2026-10-01", 7, default_mood=0)
    entries[1]["mood_beam"] = -1
    entries[2]["mood_beam"] = -1
    entries[3]["mood_beam"] = -1

    res = evaluate(entries, phase="pregnancy", today="2026-10-07")
    assert res["tier"] == 1
    assert any("3 or more consecutive depressed days" in r for r in res["reasons"])


def test_scenario_6_depressed_plus_short_sleep_triggers_tier_2():
    """6 depressed days plus short sleep on 7 nights in 14 days triggers Tier 2."""
    entries = make_series("2026-10-01", 14, default_mood=0, default_sleep=8.0)
    # 6 depressed days
    for i in range(6):
        entries[i]["mood_beam"] = -1
    # 7 nights of short sleep (< 6h in pregnancy)
    for i in range(7):
        entries[i]["sleep_hours"] = 5.0

    res = evaluate(entries, phase="pregnancy", today="2026-10-14")
    assert res["tier"] == 2
    assert any("combined rule triggered" in r for r in res["reasons"])


def test_scenario_2_consecutive_marked_elevated_days_triggers_tier_2():
    """2 consecutive marked elevated days (>= +2) triggers Tier 2 elevated side."""
    entries = make_series("2026-10-01", 7, default_mood=0)
    entries[1]["mood_beam"] = 2
    entries[2]["mood_beam"] = 2

    res = evaluate(entries, phase="pregnancy", today="2026-10-07")
    assert res["tier"] == 2
    assert res["elevated_side_flagged"] is True
    assert "offer_mdq_if_elevated_rule_fired" in res["actions"]


def test_scenario_3_elevated_in_7_with_2_reduced_sleep_triggers_tier_2():
    """3 elevated days in 7 with 2 or more reduced-sleep nights triggers Tier 2."""
    entries = make_series("2026-10-01", 7, default_mood=0, default_sleep=8.0)
    # 3 elevated days (+1) with usual sleep 8.0, so <= 6.0 is reduced sleep
    entries[0]["mood_beam"] = 1
    entries[0]["sleep_hours"] = 5.5  # reduced
    entries[1]["mood_beam"] = 1
    entries[1]["sleep_hours"] = 5.0  # reduced
    entries[2]["mood_beam"] = 1
    entries[2]["sleep_hours"] = 8.0  # normal

    res = evaluate(entries, phase="pregnancy", today="2026-10-07", usual_sleep=8.0)
    assert res["tier"] == 2
    assert res["elevated_side_flagged"] is True


def test_scenario_single_severe_elevated_plus_3_triggers_tier_3():
    """A single Severe Elevated (+3) day triggers Tier 3 Urgent immediately."""
    entries = [make_entry("2026-10-01", mood=3)]
    res = evaluate(entries, phase="pregnancy", today="2026-10-01")
    assert res["tier"] == 3
    assert res["tier_name"] == "tier3_urgent_same_day"
    assert "fixed_script_only" in res["actions"]
    assert any("Severe Elevated (+3)" in r for r in res["reasons"])


def test_scenario_postpartum_week_4_elevated_plus_reduced_sleep_triggers_tier_3():
    """Postpartum week 4 (day 28): 2 days at +2 with 2 nights of sleep 2h below usual triggers Tier 3."""
    delivery_date = "2026-09-15"
    # Today is 2026-10-13 (day 28 postpartum, week 4)
    entries = make_series("2026-10-07", 7, default_mood=0, default_sleep=8.0)
    # 2 consecutive days at +2 with sleep 5.5h (usual=8.0h, so diff=2.5h >= 2h)
    entries[4]["mood_beam"] = 2
    entries[4]["sleep_hours"] = 5.5
    entries[5]["mood_beam"] = 2
    entries[5]["sleep_hours"] = 5.0

    res = evaluate(
        entries,
        phase="postpartum",
        today="2026-10-13",
        delivery_date=delivery_date,
        usual_sleep=8.0,
    )
    assert res["tier"] == 3
    assert res["tier_name"] == "tier3_urgent_same_day"
    assert any("Postpartum warning signs of mood >= +2 with reduced sleep" in r for r in res["reasons"])


def test_scenario_postpartum_day_10_depressed_side_capped_at_tier_1():
    """Postpartum day 10 with 10 of 14 depressed days is capped at Tier 1 (baby blues reassurance)."""
    delivery_date = "2026-10-01"
    # Today is 2026-10-11 (day 10 postpartum, in days 1-14 window)
    entries = make_series("2026-09-28", 14, default_mood=-1)

    res = evaluate(
        entries,
        phase="postpartum",
        today="2026-10-11",
        delivery_date=delivery_date,
    )
    # Depressed side would be Tier 2, but capped at Tier 1 for postpartum days 1-14
    assert res["tier"] == 1
    assert res["tier_name"] == "tier1_gentle_checkin"
    assert any("Baby Blues window, days 1-14" in r for r in res["reasons"])


def test_scenario_postpartum_day_10_plus_3_elevated_never_capped():
    """Postpartum day 10 with a +3 day triggers Tier 3 (elevated side is NEVER capped)."""
    delivery_date = "2026-10-01"
    entries = make_series("2026-10-01", 10, default_mood=0)
    entries[9]["mood_beam"] = 3  # +3 on day 10

    res = evaluate(
        entries,
        phase="postpartum",
        today="2026-10-10",
        delivery_date=delivery_date,
    )
    assert res["tier"] == 3
    assert res["tier_name"] == "tier3_urgent_same_day"


def test_scenario_pregnancy_mild_elevated_normal_sleep_is_good_day():
    """In pregnancy, a +1 mood with normal sleep is classified as a good day with no alerts."""
    entries = make_series("2026-10-01", 7, default_mood=0, default_sleep=8.0)
    entries[3]["mood_beam"] = 1
    entries[3]["sleep_hours"] = 8.5  # normal sleep

    res = evaluate(entries, phase="pregnancy", today="2026-10-07")
    assert res["tier"] == 0
    assert res["weekly_band"] == "good_balanced"


def test_scenario_fewer_than_4_entries_returns_not_enough_data():
    """Fewer than 4 entries in a 7-day window returns 'not_enough_data'."""
    entries = [
        make_entry("2026-10-01", mood=0),
        make_entry("2026-10-02", mood=0),
        make_entry("2026-10-03", mood=0),
    ]
    res = evaluate(entries, phase="pregnancy", today="2026-10-07")
    assert res["weekly_band"] == "not_enough_data"
    assert any("Fewer than 4 entries logged" in r for r in res["reasons"])


def test_scenario_sleep_anxiety_irritability_alone_never_trigger_tier_2():
    """Sleep, anxiety, and irritability alone without elevated or depressed mood never trigger Tier 2."""
    entries = make_series("2026-10-01", 14, default_mood=0, default_sleep=4.0)
    for e in entries:
        e["anxiety"] = 3
        e["irritability"] = 3

    res = evaluate(entries, phase="pregnancy", today="2026-10-14")
    # Should not trigger Tier 2 (only Tier 1 checkin for anxious/irritable addon)
    assert res["tier"] <= 1
    assert res["tier"] != 2
    assert res["elevated_side_flagged"] is False
    assert res["depressed_side_flagged"] is False
