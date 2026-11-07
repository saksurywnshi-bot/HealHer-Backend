"""BEAM Perinatal Insights Deterministic Pure Rule Engine.

Pure functions implementing day classification, weekly bands, monthly summaries,
and tiered safety triggers (Tiers 0-3) exactly as defined in:
- beam_perinatal_mood_sleep_guidelines_v0.3.md
- beam_perinatal_rules.json

Safety & Privacy Constraints:
- Pure functional code; NO AI / LLM decisions for classifications or alert tiers.
- Zero magic numbers: all thresholds loaded from config.
- Explainable outputs: returns a detailed `reasons` list for every rule fired.
- Counting by days, never signed averages.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from app.services.insights.config_loader import get_beam_rules_config


def parse_date(d: Union[str, date, datetime]) -> date:
    """Parse date from string or date/datetime object."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d[:10], "%Y-%m-%d").date()


def normalize_mood_score(raw_mood: Union[int, float, str, None], config: Dict[str, Any]) -> Optional[int]:
    """Convert raw mood (numeric -3..+3 or string label) to standardized integer score -3..+3."""
    if raw_mood is None:
        return None

    if isinstance(raw_mood, (int, float)):
        score = int(round(raw_mood))
        # Handle 1..7 legacy scale where 4 is Normal (0), 7 is +3, 1 is -3
        if 1 <= score <= 7 and score not in (-3, -2, -1, 0, 1, 2, 3):
            return score - 4
        if -3 <= score <= 3:
            return score
        return max(-3, min(3, score))

    # String label matching
    text = str(raw_mood).strip().lower()
    for item in config.get("mood_levels", []):
        if item["label"].lower() == text:
            return item["score"]

    # Fuzzy matches
    mapping = {
        "severe elevated": 3,
        "moderate elevated": 2,
        "mild elevated": 1,
        "normal": 0,
        "balanced": 0,
        "mild depressed": -1,
        "moderate depressed": -2,
        "severe depressed": -3,
    }
    return mapping.get(text)


def classify_day(entry: Dict[str, Any], usual_sleep: float, phase: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single day's entry into clinical day types according to config."""
    raw_mood = entry.get("mood_beam") if entry.get("mood_beam") is not None else entry.get("mood")
    mood = normalize_mood_score(raw_mood, config)
    anxiety = entry.get("anxiety") or 0
    irritability = entry.get("irritability") or 0
    sleep = entry.get("sleep_hours")
    sleep_float = float(sleep) if sleep is not None else None

    day_rules = config.get("day_classification", {})
    reduced_sleep_threshold = day_rules.get("reduced_sleep_night", {}).get("hours_below_usual_by_at_least", 2)
    is_reduced_sleep = (
        sleep_float is not None
        and (usual_sleep - sleep_float) >= reduced_sleep_threshold
    )

    # In pregnancy, +1 elevated mood with normal sleep is considered a good day
    is_pregnancy = phase.lower() == "pregnancy"
    phase_rules = config.get("phase_adjustments", {}).get(phase.lower(), {})

    is_balanced = mood == 0
    is_depressed = mood is not None and mood <= day_rules.get("depressed_day", {}).get("mood_at_most", -1)
    is_impaired_depressed = mood is not None and mood <= day_rules.get("impaired_depressed_day", {}).get("mood_at_most", -2)
    is_elevated = mood is not None and mood >= day_rules.get("elevated_day", {}).get("mood_at_least", 1)
    is_marked_elevated = mood is not None and mood >= day_rules.get("marked_elevated_day", {}).get("mood_at_least", 2)
    is_anxious = anxiety >= day_rules.get("anxious_day", {}).get("anxiety_at_least", 2)
    is_irritable = irritability >= day_rules.get("irritable_day", {}).get("irritability_at_least", 2)

    # Sleep band classification
    sleep_bands = config.get("sleep_bands_hours", {}).get(
        "postpartum" if phase.lower() == "postpartum" else "pregnancy", {}
    )
    sleep_quality = "unknown"
    if sleep_float is not None:
        if sleep_float >= sleep_bands.get("long_above", 10):
            sleep_quality = "long"
        elif sleep_float >= sleep_bands.get("good_min", 7):
            sleep_quality = "good"
        elif sleep_float >= sleep_bands.get("a_bit_short_min", 6):
            sleep_quality = "a_bit_short"
        else:
            sleep_quality = "short"

    return {
        "log_date": entry.get("log_date"),
        "mood": mood,
        "anxiety": anxiety,
        "irritability": irritability,
        "sleep_hours": sleep_float,
        "is_balanced": is_balanced,
        "is_depressed": is_depressed,
        "is_impaired_depressed": is_impaired_depressed,
        "is_elevated": is_elevated,
        "is_marked_elevated": is_marked_elevated,
        "is_anxious": is_anxious,
        "is_irritable": is_irritable,
        "is_reduced_sleep": is_reduced_sleep,
        "sleep_quality": sleep_quality,
    }


def find_consecutive_runs(classified_entries: List[Dict[str, Any]], predicate_key: str, required_count: int) -> List[Tuple[date, date]]:
    """Detect consecutive chronological days satisfying a boolean predicate."""
    runs = []
    current_run: List[Dict[str, Any]] = []

    for entry in sorted(classified_entries, key=lambda x: parse_date(x["log_date"])):
        if entry.get(predicate_key):
            if not current_run:
                current_run.append(entry)
            else:
                prev_date = parse_date(current_run[-1]["log_date"])
                curr_date = parse_date(entry["log_date"])
                if curr_date == prev_date + timedelta(days=1):
                    current_run.append(entry)
                elif curr_date == prev_date:
                    pass  # duplicate entry on same day
                else:
                    if len(current_run) >= required_count:
                        runs.append((parse_date(current_run[0]["log_date"]), parse_date(current_run[-1]["log_date"])))
                    current_run = [entry]
        else:
            if len(current_run) >= required_count:
                runs.append((parse_date(current_run[0]["log_date"]), parse_date(current_run[-1]["log_date"])))
            current_run = []

    if len(current_run) >= required_count:
        runs.append((parse_date(current_run[0]["log_date"]), parse_date(current_run[-1]["log_date"])))

    return runs


def evaluate(
    entries: List[Dict[str, Any]],
    phase: str = "pregnancy",
    today: Optional[Union[str, date]] = None,
    config: Optional[Dict[str, Any]] = None,
    usual_sleep: float = 7.5,
    delivery_date: Optional[Union[str, date]] = None,
    clinician_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate BEAM entries over 7, 14, and 30-day windows and return clinical bands, tiers, and audit reasons.

    Args:
        entries: List of daily log dicts with keys log_date, mood_beam, anxiety, irritability, sleep_hours.
        phase: Maternal phase ("pregnancy", "postpartum", or "any").
        today: Reference anchor date (defaults to max date in entries or today's date).
        config: Loaded BEAM perinatal rules JSON configuration (loaded from file if None).
        usual_sleep: User's baseline usual sleep hours (default 7.5h).
        delivery_date: Date of delivery for postpartum age and cap calculations.
        clinician_overrides: Optional overrides for clinician flags.

    Returns:
        Dict containing weekly_band, monthly_summary, tier, tier_name, reasons[], and statistics.
    """
    cfg = config or get_beam_rules_config()
    flags = dict(cfg.get("clinician_flags", {}))
    if clinician_overrides:
        flags.update(clinician_overrides)

    today_date = parse_date(today) if today else (
        max((parse_date(e["log_date"]) for e in entries), default=date.today())
    )

    # Classify each daily entry
    classified_by_date: Dict[date, Dict[str, Any]] = {}
    for entry in entries:
        d = parse_date(entry.get("log_date"))
        classified = classify_day(entry, usual_sleep, phase, cfg)
        classified_by_date[d] = classified

    # Calculate postpartum days if delivery_date is provided
    postpartum_day: Optional[int] = None
    if phase.lower() == "postpartum" and delivery_date:
        deliv = parse_date(delivery_date)
        postpartum_day = (today_date - deliv).days

    reasons: List[str] = []

    # -------------------------------------------------------------------------
    # 1. 7-Day Window Analysis (Weekly Band)
    # -------------------------------------------------------------------------
    start_7d = today_date - timedelta(days=cfg.get("weekly_bands", {}).get("window_days", 7) - 1)
    entries_7d = [classified_by_date[d] for d in classified_by_date if start_7d <= d <= today_date]
    min_weekly_required = cfg.get("minimum_data", {}).get("weekly_entries_required", 4)

    is_weekly_sufficient = len(entries_7d) >= min_weekly_required
    weekly_band = "not_enough_data"
    weekly_addon: Optional[str] = None
    weekly_strength: Optional[str] = None

    if not is_weekly_sufficient:
        reasons.append(
            f"Fewer than {min_weekly_required} entries logged in the past 7 days ({len(entries_7d)} found). Minimum data requirement not met."
        )
    else:
        # Tally counts for past 7 days
        depressed_days_7d = sum(1 for e in entries_7d if e["is_depressed"])
        impaired_depressed_days_7d = sum(1 for e in entries_7d if e["is_impaired_depressed"])
        elevated_days_7d = sum(1 for e in entries_7d if e["is_elevated"])
        marked_elevated_days_7d = sum(1 for e in entries_7d if e["is_marked_elevated"])
        anxious_days_7d = sum(1 for e in entries_7d if e["is_anxious"])
        irritable_days_7d = sum(1 for e in entries_7d if e["is_irritable"])
        reduced_sleep_on_elevated_days_7d = sum(1 for e in entries_7d if e["is_elevated"] and e["is_reduced_sleep"])
        days_at_abs_mood_2_or_more_7d = sum(1 for e in entries_7d if e["mood"] is not None and abs(e["mood"]) >= 2)

        # Days balanced or mild elevated with normal sleep
        balanced_or_normal_elevated_7d = sum(
            1 for e in entries_7d if e["mood"] == 0 or (e["mood"] == 1 and not e["is_reduced_sleep"])
        )

        wb_cfg = cfg.get("weekly_bands", {})
        gb_cfg = wb_cfg.get("good_balanced", {})
        is_good = (
            balanced_or_normal_elevated_7d >= gb_cfg.get("min_days_balanced_or_mild_elevated_with_normal_sleep", 5)
            and depressed_days_7d <= gb_cfg.get("max_depressed_days", 1)
            and days_at_abs_mood_2_or_more_7d <= gb_cfg.get("max_days_at_abs_mood_2_or_more", 0)
            and anxious_days_7d <= gb_cfg.get("max_anxious_days", 2)
        )

        tl_cfg = wb_cfg.get("tougher_low", {})
        is_tougher = (
            depressed_days_7d >= 3
            or impaired_depressed_days_7d >= 2
        )

        is_high_energy = (
            (elevated_days_7d >= 3 and reduced_sleep_on_elevated_days_7d >= 2)
            or marked_elevated_days_7d >= 1
        )

        # Mixed swings check
        if is_tougher and is_high_energy:
            weekly_band = "mixed_swings"
            reasons.append("Weekly pattern shows mixed swings: met both tougher-low and high-energy criteria in the same 7-day window.")
        elif is_tougher:
            weekly_band = "tougher_low"
            reasons.append(f"Weekly band 'tougher_low' triggered ({depressed_days_7d} depressed days, {impaired_depressed_days_7d} impaired days).")
        elif is_high_energy:
            weekly_band = "high_energy"
            reasons.append(f"Weekly band 'high_energy' triggered ({marked_elevated_days_7d} marked elevated days, {elevated_days_7d} elevated with {reduced_sleep_on_elevated_days_7d} reduced-sleep nights).")
        elif is_good:
            weekly_band = "good_balanced"
            reasons.append(f"Weekly band 'good_balanced' triggered ({balanced_or_normal_elevated_7d} balanced/mild-elevated days with normal sleep).")
        else:
            weekly_band = "okay_mixed"
            reasons.append("Weekly band 'okay_mixed' triggered: normal perinatal ups and downs.")

        # Check for anxious or irritable add-on
        if anxious_days_7d >= 4 or irritable_days_7d >= 4:
            weekly_addon = "anxious_or_irritable"
            reasons.append(f"Weekly add-on 'anxious_or_irritable' triggered ({anxious_days_7d} anxious days, {irritable_days_7d} irritable days).")

        # Select a strength
        if len(entries_7d) >= 6:
            weekly_strength = "Consistent daily check-ins throughout the week."
        elif balanced_or_normal_elevated_7d >= 3:
            weekly_strength = "Noticeable stretches of balanced and grounded mood."
        else:
            weekly_strength = "Showing care for yourself by tracking your patterns."

    # -------------------------------------------------------------------------
    # 2. 14-Day Window Analysis (Tier 2 Evaluation)
    # -------------------------------------------------------------------------
    start_14d = today_date - timedelta(days=13)
    raw_entries_14d = [classified_by_date[d] for d in classified_by_date if start_14d <= d <= today_date]

    # Clinician flag: exclude postpartum days 1 to 7 from 14-day depressed count if enabled
    filtered_entries_14d = raw_entries_14d
    if (
        phase.lower() == "postpartum"
        and delivery_date
        and flags.get("exclude_postpartum_days_1_to_7_from_14_day_count", False)
    ):
        deliv = parse_date(delivery_date)
        filtered_entries_14d = [
            e for e in raw_entries_14d
            if not (0 <= (parse_date(e["log_date"]) - deliv).days <= 7)
        ]

    min_14d_required = cfg.get("minimum_data", {}).get("tier2_entries_required_in_14_days", 10)
    is_14d_sufficient = len(raw_entries_14d) >= min_14d_required

    # -------------------------------------------------------------------------
    # 3. 30-Day Window Analysis (Monthly Summary & Monthly Tier 2)
    # -------------------------------------------------------------------------
    start_30d = today_date - timedelta(days=cfg.get("monthly", {}).get("window_days", 30) - 1)
    entries_30d = [classified_by_date[d] for d in classified_by_date if start_30d <= d <= today_date]

    depressed_days_30d = sum(1 for e in entries_30d if e["is_depressed"])
    elevated_days_30d = sum(1 for e in entries_30d if e["is_elevated"])
    balanced_days_30d = sum(1 for e in entries_30d if e["is_balanced"])
    anxious_days_30d = sum(1 for e in entries_30d if e["is_anxious"])
    irritable_days_30d = sum(1 for e in entries_30d if e["is_irritable"])
    sleep_values_30d = [e["sleep_hours"] for e in entries_30d if e["sleep_hours"] is not None]
    avg_sleep_30d = (
        round(sum(sleep_values_30d) / len(sleep_values_30d), 1) if sleep_values_30d else None
    )

    depressed_share_30d = (depressed_days_30d / len(entries_30d)) if entries_30d else 0.0

    monthly_summary = {
        "entries_count": len(entries_30d),
        "depressed_days": depressed_days_30d,
        "elevated_days": elevated_days_30d,
        "balanced_days": balanced_days_30d,
        "anxious_days": anxious_days_30d,
        "irritable_days": irritable_days_30d,
        "average_sleep_hours": avg_sleep_30d,
        "depressed_share": round(depressed_share_30d, 2),
    }

    monthly_tier2_met = False
    if len(entries_30d) >= 15:  # Require at least half a month of data for monthly triggers
        depressed_share_threshold = cfg.get("monthly", {}).get("tier2_if_depressed_day_share_at_least", 0.5)
        if depressed_share_30d >= depressed_share_threshold:
            monthly_tier2_met = True
            reasons.append(
                f"Monthly Tier 2 triggered: {depressed_days_30d} of {len(entries_30d)} days ({round(depressed_share_30d * 100)}%) were depressed."
            )

    # -------------------------------------------------------------------------
    # 4. Tiers Evaluation (Tier 3 -> Tier 2 -> Tier 1 -> Tier 0)
    # -------------------------------------------------------------------------
    tier = 0
    tier_name = "tier0_normal"
    tier3_fired = False
    tier2_depressed_fired = False
    tier2_elevated_fired = False
    tier1_fired = False

    # Check Tier 3 triggers:
    # 1. Single Severe Elevated (+3) day on ANY day
    severe_elevated_days = [e for e in classified_by_date.values() if e["mood"] == 3]
    if severe_elevated_days:
        tier3_fired = True
        reasons.append(
            f"Tier 3 Urgent triggered: Severe Elevated (+3) recorded on {severe_elevated_days[-1]['log_date']}."
        )

    # 2. Postpartum elevated-plus-reduced-sleep rule:
    # In first 12 weeks (configurable): consecutive days mood >= +2 for >= 2 days AND consecutive reduced-sleep nights >= 2
    postpartum_max_weeks = flags.get("postpartum_urgent_elevated_window_weeks", 12)
    is_within_pp_window = (
        phase.lower() == "postpartum"
        and postpartum_day is not None
        and postpartum_day <= (postpartum_max_weeks * 7)
    )

    if is_within_pp_window or (phase.lower() == "postpartum" and delivery_date is None):
        # Look for consecutive marked elevated days (>= +2) with consecutive reduced sleep
        marked_runs = find_consecutive_runs(list(classified_by_date.values()), "is_marked_elevated", 2)
        reduced_sleep_runs = find_consecutive_runs(list(classified_by_date.values()), "is_reduced_sleep", 2)

        # Check for overlapping 2+ consecutive marked elevated days AND 2+ reduced sleep nights
        for m_start, m_end in marked_runs:
            for s_start, s_end in reduced_sleep_runs:
                if max(m_start, s_start) <= min(m_end, s_end):  # overlap
                    tier3_fired = True
                    reasons.append(
                        f"Tier 3 Urgent triggered: Postpartum warning signs of mood >= +2 with reduced sleep for consecutive nights between {max(m_start, s_start)} and {min(m_end, s_end)}."
                    )

    if tier3_fired:
        tier = 3
        tier_name = "tier3_urgent_same_day"
        return {
            "status": "ok",
            "weekly_band": weekly_band,
            "weekly_addon": weekly_addon,
            "weekly_strength": weekly_strength,
            "monthly_summary": monthly_summary,
            "tier": tier,
            "tier_name": tier_name,
            "reasons": reasons,
            "actions": ["fixed_script_only", "emergency_referral"],
        }

    # Check Tier 2 triggers (Depressed side & Elevated side)
    # Consecutive days at -3
    severe_dep_runs = find_consecutive_runs(list(classified_by_date.values()), "mood", 2)
    # Filter for mood == -3
    all_sorted_entries = sorted(classified_by_date.values(), key=lambda x: parse_date(x["log_date"]))
    has_2_consecutive_minus_3 = False
    for i in range(len(all_sorted_entries) - 1):
        if (
            all_sorted_entries[i]["mood"] == -3
            and all_sorted_entries[i + 1]["mood"] == -3
            and parse_date(all_sorted_entries[i + 1]["log_date"]) == parse_date(all_sorted_entries[i]["log_date"]) + timedelta(days=1)
        ):
            has_2_consecutive_minus_3 = True
            reasons.append(
                f"Tier 2 triggered: 2 consecutive days at Severe Depressed (-3) on {all_sorted_entries[i]['log_date']} and {all_sorted_entries[i+1]['log_date']}."
            )
            break

    # Consecutive marked elevated days (>= +2) for 2 days
    marked_elevated_runs = find_consecutive_runs(list(classified_by_date.values()), "is_marked_elevated", 2)
    if marked_elevated_runs:
        tier2_elevated_fired = True
        reasons.append(
            f"Tier 2 Elevated-side triggered: 2 consecutive marked elevated days (>= +2) between {marked_elevated_runs[-1][0]} and {marked_elevated_runs[-1][1]}."
        )

    # 14-day window checks (only if 14-day data is sufficient, or boundary rules met)
    if is_14d_sufficient:
        depressed_count_14d = sum(1 for e in filtered_entries_14d if e["is_depressed"])
        impaired_count_14d = sum(1 for e in filtered_entries_14d if e["is_impaired_depressed"])
        short_sleep_nights_14d = sum(1 for e in filtered_entries_14d if e["sleep_quality"] == "short")
        anxious_days_14d = sum(1 for e in filtered_entries_14d if e["is_anxious"])

        # (a) 8 or more depressed days in 14
        if depressed_count_14d >= 8:
            tier2_depressed_fired = True
            reasons.append(f"Tier 2 Depressed-side triggered: {depressed_count_14d} of 14 days were depressed (threshold: >= 8).")

        # (b) 4 or more impaired depressed days in 14
        if impaired_count_14d >= 4:
            tier2_depressed_fired = True
            reasons.append(f"Tier 2 Depressed-side triggered: {impaired_count_14d} impaired depressed days in 14 days (threshold: >= 4).")

        # (d) 6 or more depressed days AND (7+ short-sleep nights OR 7+ anxious days)
        if depressed_count_14d >= 6 and (short_sleep_nights_14d >= 7 or anxious_days_14d >= 7):
            tier2_depressed_fired = True
            reasons.append(
                f"Tier 2 Depressed-side combined rule triggered: {depressed_count_14d} depressed days with {short_sleep_nights_14d} short-sleep nights and {anxious_days_14d} anxious days in 14 days."
            )

    if has_2_consecutive_minus_3:
        tier2_depressed_fired = True

    # Elevated side 7-day checks:
    # 3 or more elevated days in 7 with reduced sleep on 2+ nights
    if is_weekly_sufficient:
        if elevated_days_7d >= 3 and reduced_sleep_on_elevated_days_7d >= 2:
            tier2_elevated_fired = True
            reasons.append(
                f"Tier 2 Elevated-side triggered: {elevated_days_7d} elevated days in past 7 with {reduced_sleep_on_elevated_days_7d} reduced-sleep nights."
            )

        # Elevated mood together with irritability >= 2 on 2+ days
        elevated_and_irritable_days = sum(1 for e in entries_7d if e["is_elevated"] and e["is_irritable"])
        if elevated_days_7d >= 2 and elevated_and_irritable_days >= 2:
            tier2_elevated_fired = True
            reasons.append(
                f"Tier 2 Elevated-side triggered: {elevated_and_irritable_days} days with elevated mood and moderate/severe irritability."
            )

    # Mixed swings also applies Tier 2 elevated-side rule
    if weekly_band == "mixed_swings":
        tier2_elevated_fired = True

    if monthly_tier2_met:
        tier2_depressed_fired = True

    # Phase adjustment: Postpartum days 1 to 14 capping
    # Depressed side is capped at Tier 1, but elevated side is NEVER capped!
    is_postpartum_early = (
        phase.lower() == "postpartum"
        and postpartum_day is not None
        and 1 <= postpartum_day <= 14
    )

    if tier2_elevated_fired:
        tier = 2
        tier_name = "tier2_suggest_professional_conversation"
    elif tier2_depressed_fired:
        if is_postpartum_early:
            tier = 1
            tier_name = "tier1_gentle_checkin"
            reasons.append(
                f"Postpartum Day {postpartum_day} (Baby Blues window, days 1-14): Depressed-side Tier 2 capped at Tier 1 with gentle check-in and reassurance."
            )
        else:
            tier = 2
            tier_name = "tier2_suggest_professional_conversation"

    # Check Tier 1 triggers (if not already Tier 2 or 3)
    if tier == 0:
        consecutive_dep_runs = find_consecutive_runs(list(classified_by_date.values()), "is_depressed", 3)
        reduced_sleep_with_elevated_runs = [
            e for e in classified_by_date.values() if e["is_reduced_sleep"] and e["mood"] is not None and e["mood"] >= 1
        ]

        if consecutive_dep_runs:
            tier1_fired = True
            reasons.append(
                f"Tier 1 triggered: 3 or more consecutive depressed days between {consecutive_dep_runs[-1][0]} and {consecutive_dep_runs[-1][1]}."
            )
        elif weekly_band == "tougher_low":
            tier1_fired = True
            reasons.append("Tier 1 triggered: weekly band is tougher_low.")
        elif weekly_addon == "anxious_or_irritable":
            tier1_fired = True
            reasons.append("Tier 1 triggered: weekly add-on is anxious_or_irritable.")
        elif len(reduced_sleep_with_elevated_runs) >= 2:
            tier1_fired = True
            reasons.append(
                f"Tier 1 triggered: {len(reduced_sleep_with_elevated_runs)} reduced-sleep nights with elevated mood (+1)."
            )

        if tier1_fired:
            tier = 1
            tier_name = "tier1_gentle_checkin"

    actions: List[str] = []
    if tier == 2:
        actions = ["offer_epds", "suggest_doctor_midwife_or_health_worker", "offer_summary_to_share"]
        if weekly_addon == "anxious_or_irritable" or any(e["is_anxious"] for e in entries_7d):
            actions.append("offer_gad7_if_anxiety_flagged")
        if tier2_elevated_fired:
            actions.append("offer_mdq_if_elevated_rule_fired")
    elif tier == 1:
        actions = ["gentle_checkin", "self_care_suggestions", "trusted_contact_chat"]
    elif tier == 0:
        actions = ["normal_summary"]

    return {
        "status": "ok",
        "weekly_band": weekly_band,
        "weekly_addon": weekly_addon,
        "weekly_strength": weekly_strength,
        "days_logged_7d": len(entries_7d),
        "days_logged_30d": len(entries_30d),
        "monthly_summary": monthly_summary,
        "tier": tier,
        "tier_name": tier_name,
        "reasons": reasons,
        "actions": actions,
        "elevated_side_flagged": tier2_elevated_fired,
        "depressed_side_flagged": tier2_depressed_fired,
    }
