"""Service for computing user sentiment patterns over time (plain code, no AI).

Metrics:
- Filtered by period: 'week' (7 days, min 3 notes) or 'month' (30 days, min 8 notes).
- Top 3 themes with frequency counts.
- Count of difficult-tone notes (low, anxious, tired_overwhelmed, distressed).
- Longest consecutive day streak with low or anxious tone.
- Sleep correlation (when >= 3 days exist in both groups).
- Change compared to previous period.
- Suggest talk to someone flag.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from app.services.sentiment_db import (
    get_sentiment_results_for_period,
    get_sentiment_results_for_range,
)


def compute_sentiment_patterns(
    user_ref: str,
    period: str = "week",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Compute longitudinal sentiment patterns for a user."""
    ref_time = now or datetime.now(timezone.utc)
    days = 7 if period == "week" else 30
    min_notes = 3 if period == "week" else 8

    # Fetch notes for current period
    notes = get_sentiment_results_for_period(user_ref, days=days, reference_time=ref_time)
    notes_count = len(notes)

    # 1. Check minimum notes threshold
    if notes_count < min_notes:
        return {
            "status": "not_enough_data",
            "period": period,
            "notes_count": notes_count,
            "minimum_required": min_notes,
        }

    # 2. Top 3 themes with counts
    THEME_ALIASES = {
        "sleep_rest": "sleep",
        "no_need_sleep": "sleep",
        "joy_connection": "positive_moments",
        "self_care": "positive_moments",
        "guilt_worthlessness": "self_criticism_guilt",
        "lack_of_support": "family_support",
        "restless_energy": "body_physical_symptoms",
        "racing_thoughts": "tiredness",
    }
    theme_counter = Counter()
    for n in notes:
        for t in n.get("themes", []):
            canonical = THEME_ALIASES.get(t, t)
            theme_counter[canonical] += 1

    top_themes = []
    for theme, count in theme_counter.most_common(3):
        pct = int(round((count / max(1, notes_count)) * 100))
        top_themes.append(
            {
                "theme": theme,
                "theme_key": f"sentiment.theme.{theme}",
                "count": count,
                "total": notes_count,
                "total_notes": notes_count,
                "pct": pct,
            }
        )

    # 3. How many notes had tone low, anxious, tired_overwhelmed or distressed
    difficult_tones = {"low", "anxious", "tired_overwhelmed", "distressed"}
    difficult_tones_count = sum(1 for n in notes if n.get("tone") in difficult_tones)

    # 4. Longest run of consecutive days with a low or anxious tone
    # Group by date string (YYYY-MM-DD)
    days_with_low_anxious = set()
    sleep_by_low_group: List[float] = []
    sleep_by_other_group: List[float] = []

    for n in notes:
        dt_str = n.get("created_at", "")[:10]
        tone = n.get("tone")
        sleep = n.get("sleep_hours")

        is_low_anxious = tone in ("low", "anxious")
        if is_low_anxious:
            days_with_low_anxious.add(dt_str)
            if sleep is not None:
                sleep_by_low_group.append(float(sleep))
        else:
            if sleep is not None:
                sleep_by_other_group.append(float(sleep))

    # Calculate longest streak
    longest_streak = 0
    if days_with_low_anxious:
        sorted_dates = sorted(
            [datetime.strptime(d, "%Y-%m-%d").date() for d in days_with_low_anxious]
        )
        current_streak = 1
        longest_streak = 1
        for i in range(1, len(sorted_dates)):
            if sorted_dates[i] - sorted_dates[i - 1] == timedelta(days=1):
                current_streak += 1
                if current_streak > longest_streak:
                    longest_streak = current_streak
            elif sorted_dates[i] != sorted_dates[i - 1]:
                current_streak = 1

    # 5. Sleep link: only when >= 3 days exist in each group
    sleep_link = None
    if len(sleep_by_low_group) >= 3 and len(sleep_by_other_group) >= 3:
        avg_low = round(sum(sleep_by_low_group) / len(sleep_by_low_group), 1)
        avg_other = round(sum(sleep_by_other_group) / len(sleep_by_other_group), 1)
        sleep_link = {
            "avg_sleep_low_anxious_days": avg_low,
            "avg_sleep_other_days": avg_other,
            "difference_hours": round(avg_other - avg_low, 1),
        }

    # 6. Change versus previous period
    prev_end = ref_time - timedelta(days=days)
    prev_start = ref_time - timedelta(days=days * 2)
    prev_notes = get_sentiment_results_for_range(
        user_ref, prev_start.isoformat(), prev_end.isoformat()
    )

    prev_low_days = {
        n.get("created_at", "")[:10]
        for n in prev_notes
        if n.get("tone") in ("low", "anxious")
    }
    cur_low_days_count = len(days_with_low_anxious)
    prev_low_days_count = len(prev_low_days)
    diff_days = cur_low_days_count - prev_low_days_count

    if diff_days > 0:
        change_direction = "more"
    elif diff_days < 0:
        change_direction = "fewer"
    else:
        change_direction = "same"

    period_comparison = {
        "current_low_anxious_days": cur_low_days_count,
        "previous_low_anxious_days": prev_low_days_count,
        "change_direction": change_direction,
        "difference": abs(diff_days),
    }

    # 7. suggest_talk_to_someone:
    # Low or anxious tone on 4 or more of the last 7 analysed notes, OR distressed appears at least once
    has_distressed = any(n.get("tone") == "distressed" for n in notes)
    # Check last 7 notes (notes are sorted ascending by created_at, so take notes[-7:])
    last_7_notes = notes[-7:]
    last_7_low_anxious_count = sum(
        1 for n in last_7_notes if n.get("tone") in ("low", "anxious")
    )

    suggest_talk_to_someone = has_distressed or (last_7_low_anxious_count >= 4)

    # 8. Pattern sentences using i18n keys
    patterns = []
    if top_themes:
        primary = top_themes[0]
        patterns.append(
            {
                "template_key": "pattern.theme_frequent",
                "placeholders": {
                    "theme": primary["theme"],
                    "count": primary["count"],
                    "total": primary["total_notes"],
                },
            }
        )

    if longest_streak >= 2:
        patterns.append(
            {
                "template_key": "pattern.consecutive_low",
                "placeholders": {"streak": longest_streak},
            }
        )

    if sleep_link:
        patterns.append(
            {
                "template_key": "pattern.sleep_impact",
                "placeholders": {
                    "low_sleep": sleep_link["avg_sleep_low_anxious_days"],
                    "other_sleep": sleep_link["avg_sleep_other_days"],
                },
            }
        )

    patterns.append(
        {
            "template_key": "pattern.period_comparison",
            "placeholders": {
                "change": change_direction,
                "difference": abs(diff_days),
            },
        }
    )

    if suggest_talk_to_someone:
        patterns.append(
            {
                "template_key": "pattern.talk_to_someone",
                "placeholders": {},
            }
        )

    return {
        "status": "ok",
        "period": period,
        "notes_count": notes_count,
        "top_themes": top_themes,
        "difficult_tones_count": difficult_tones_count,
        "difficult_tone_count": difficult_tones_count,
        "longest_streak_low_anxious_days": longest_streak,
        "longest_streak_days": longest_streak,
        "sleep_link": sleep_link,
        "sleep_comparison": (
            {
                "low_tone_avg_sleep": sleep_link["avg_sleep_low_anxious_days"],
                "other_avg_sleep": sleep_link["avg_sleep_other_days"],
                "difference_hours": sleep_link["difference_hours"],
                "low_tone_days_count": len(sleep_by_low_group),
                "other_days_count": len(sleep_by_other_group),
            }
            if sleep_link
            else None
        ),
        "period_comparison": period_comparison,
        "change_vs_previous": {
            "difference": abs(diff_days),
            "trend": "more" if diff_days > 0 else ("fewer" if diff_days < 0 else "equal"),
        },
        "suggest_talk_to_someone": suggest_talk_to_someone,
        "patterns": patterns,
        "pattern_sentences": [
            {
                "template_key": p["template_key"],
                "params": p.get("placeholders", {}),
            }
            for p in patterns
        ],
    }

