"""Tests for GET /api/sentiment/patterns (longitudinal mood and theme metrics)."""

from datetime import datetime, timedelta, timezone
import pytest
from app.services.auth import hash_user_id
from app.services.sentiment_db import save_sentiment_result


def test_patterns_not_enough_data_when_below_minimums(client, auth_headers):
    """Returns status: 'not_enough_data' if < 3 notes for week or < 8 for month."""
    # Empty DB -> 0 notes
    res_week = client.get("/api/sentiment/patterns?period=week", headers=auth_headers)
    assert res_week.status_code == 200
    assert res_week.json()["status"] == "not_enough_data"
    assert res_week.json()["notes_count"] == 0
    assert res_week.json()["minimum_required"] == 3

    # Add 2 notes
    user_ref = hash_user_id("test_user_sub_999")
    now = datetime.now(timezone.utc)
    save_sentiment_result(user_ref, "prenatal", "positive", "low", ["sleep"], False, False, False, created_at=(now - timedelta(days=1)).isoformat())
    save_sentiment_result(user_ref, "prenatal", "neutral", "low", ["sleep"], False, False, False, created_at=(now - timedelta(days=2)).isoformat())

    res_week_2 = client.get("/api/sentiment/patterns?period=week", headers=auth_headers)
    assert res_week_2.json()["status"] == "not_enough_data"
    assert res_week_2.json()["notes_count"] == 2

    res_month = client.get("/api/sentiment/patterns?period=month", headers=auth_headers)
    assert res_month.json()["status"] == "not_enough_data"
    assert res_month.json()["minimum_required"] == 8


def test_patterns_metrics_with_sufficient_data(client, auth_headers):
    """Calculates theme frequencies, consecutive low streaks, and sleep comparisons."""
    user_ref = hash_user_id("test_user_sub_999")
    now = datetime.now(timezone.utc)

    # Populate 5 daily notes over the last 5 days
    # Day -5: positive, sleep 8.0
    save_sentiment_result(
        user_ref, "prenatal", "positive", "low", ["sleep", "positive_moments"],
        False, False, False, sleep_hours=8.0,
        created_at=(now - timedelta(days=5)).isoformat()
    )
    # Day -4: neutral, sleep 7.5
    save_sentiment_result(
        user_ref, "prenatal", "neutral", "low", ["sleep"],
        False, False, False, sleep_hours=7.5,
        created_at=(now - timedelta(days=4)).isoformat()
    )
    # Day -3: anxious, sleep 5.0 (Low/anxious streak start)
    save_sentiment_result(
        user_ref, "prenatal", "anxious", "medium", ["baby_health_worry", "sleep"],
        False, False, False, sleep_hours=5.0,
        created_at=(now - timedelta(days=3)).isoformat()
    )
    # Day -2: low, sleep 5.5 (Low/anxious streak day 2)
    save_sentiment_result(
        user_ref, "prenatal", "low", "medium", ["baby_health_worry", "tiredness"],
        False, False, False, sleep_hours=5.5,
        created_at=(now - timedelta(days=2)).isoformat()
    )
    # Day -1: anxious, sleep 4.5 (Low/anxious streak day 3)
    save_sentiment_result(
        user_ref, "prenatal", "anxious", "high", ["baby_health_worry"],
        False, True, False, sleep_hours=4.5,
        created_at=(now - timedelta(days=1)).isoformat()
    )

    response = client.get("/api/sentiment/patterns?period=week", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["notes_count"] == 5
    assert data["difficult_tones_count"] == 3  # 3 low/anxious notes
    assert data["longest_streak_low_anxious_days"] == 3  # 3 consecutive days

    # Check top themes: baby_health_worry was present in 3 notes, sleep in 3 notes
    themes = {t["theme"]: t["count"] for t in data["top_themes"]}
    assert themes["baby_health_worry"] == 3
    assert themes["sleep"] == 3

    # Sleep link check:
    # low/anxious group: [5.0, 5.5, 4.5] (3 days >= 3)
    # other group: [8.0, 7.5] (2 days < 3) -> should be None because other group < 3 days!
    assert data["sleep_link"] is None

    # Now add one more 'other' day with sleep hours to reach 3 days for other group
    save_sentiment_result(
        user_ref, "prenatal", "positive", "low", ["positive_moments"],
        False, False, False, sleep_hours=7.0,
        created_at=(now - timedelta(days=6)).isoformat()
    )

    response2 = client.get("/api/sentiment/patterns?period=week", headers=auth_headers)
    data2 = response2.json()
    assert data2["sleep_link"] is not None
    # avg low sleep = (5.0 + 5.5 + 4.5) / 3 = 5.0
    # avg other sleep = (8.0 + 7.5 + 7.0) / 3 = 7.5
    assert data2["sleep_link"]["avg_sleep_low_anxious_days"] == 5.0
    assert data2["sleep_link"]["avg_sleep_other_days"] == 7.5
    assert data2["sleep_link"]["difference_hours"] == 2.5


def test_suggest_talk_to_someone_triggers(client, auth_headers):
    """Triggers suggest_talk_to_someone=True if distressed occurs or >= 4 low/anxious out of 7."""
    user_ref = hash_user_id("test_user_sub_999")
    now = datetime.now(timezone.utc)

    # 1. Test distressed triggers it immediately
    save_sentiment_result(
        user_ref, "prenatal", "positive", "low", ["sleep"], False, False, False,
        created_at=(now - timedelta(days=3)).isoformat()
    )
    save_sentiment_result(
        user_ref, "prenatal", "positive", "low", ["sleep"], False, False, False,
        created_at=(now - timedelta(days=2)).isoformat()
    )
    save_sentiment_result(
        user_ref, "prenatal", "distressed", "high", ["loneliness"], False, True, False,
        created_at=(now - timedelta(days=1)).isoformat()
    )

    res = client.get("/api/sentiment/patterns?period=week", headers=auth_headers)
    assert res.json()["status"] == "ok"
    assert res.json()["suggest_talk_to_someone"] is True


def test_seed_demo_and_distinct_weekly_monthly_patterns(client, auth_headers):
    """Verifies that seed-demo generates distinct 7-day and 30-day pattern metrics."""
    # Seed tougher scenario
    seed_res = client.post(
        "/api/sentiment/seed-demo",
        headers=auth_headers,
        json={"scenario": "tougher", "days_count": 30, "stage": "postnatal"},
    )
    assert seed_res.status_code == 200
    seed_data = seed_res.json()
    assert seed_data["status"] == "ok"
    assert seed_data["inserted_notes"] > 15
    assert seed_data["week_notes_count"] == 6

    # 7-day window
    res_week = client.get("/api/sentiment/patterns?period=week", headers=auth_headers)
    assert res_week.status_code == 200
    week_data = res_week.json()
    assert week_data["status"] == "ok"
    assert week_data["period"] == "week"
    assert week_data["notes_count"] == 6
    assert week_data["difficult_tone_count"] == 5

    # 30-day window
    res_month = client.get("/api/sentiment/patterns?period=month", headers=auth_headers)
    assert res_month.status_code == 200
    month_data = res_month.json()
    assert month_data["status"] == "ok"
    assert month_data["period"] == "month"
    assert month_data["notes_count"] > 18
    assert month_data["notes_count"] != week_data["notes_count"]
    assert month_data["difficult_tone_count"] != week_data["difficult_tone_count"]
    # Check frontend compatibility aliases are present
    assert "difficult_tone_count" in month_data
    assert "longest_streak_days" in month_data
    assert "pattern_sentences" in month_data

