"""BEAM Suggestions Selection and Rotation Engine.

Selects gentle, evidence-based suggestions from beam_suggestions.json based on:
- Weekly band (good, okay, tougher, high_energy, anxious_addon, irritable_addon, sleep bands).
- User phase (pregnancy, postpartum, or any).
- Condition checks (only_if flags).
- Non-repeating rotation: prevents the same suggestion IDs from appearing in consecutive summaries.
- Evidence filters: omits items marked optional (evidence 'L') unless include_optional_suggestions is True.
"""

from typing import Any, Dict, List, Optional, Set
from app.services.insights.config_loader import get_beam_suggestions_config, get_beam_rules_config


def evaluate_only_if_condition(condition_str: str, context: Dict[str, Any]) -> bool:
    """Evaluate clinical conditional predicates specified in suggestions JSON."""
    if not condition_str:
        return True

    if condition_str == "impaired_depressed_days_at_least_2_or_tier2":
        impaired = context.get("impaired_depressed_days", 0)
        tier = context.get("tier", 0)
        return impaired >= 2 or tier == 2

    if condition_str == "anxiety_at_least_2_on_7_or_more_of_14_days":
        anxious_14d = context.get("anxious_days_14d", 0)
        return anxious_14d >= 7

    return False


def select_suggestions(
    band: str,
    phase: str,
    context: Optional[Dict[str, Any]] = None,
    previous_suggestion_ids: Optional[List[str]] = None,
    count: int = 3,
    include_optional: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Select 2 to 3 appropriate suggestions from the approved library.

    Args:
        band: The user's weekly band ('good_balanced', 'okay_mixed', 'tougher_low', 'high_energy', 'mixed_swings').
        phase: User phase ('pregnancy' or 'postpartum').
        context: Calculated metrics dict (impaired_depressed_days, tier, anxious_days_14d, sleep_quality, etc.).
        previous_suggestion_ids: List of suggestion IDs shown in the previous summary to prevent repeats.
        count: Target number of suggestions (default 3, minimum 2).
        include_optional: Whether to include evidence 'L' optional items (default from config clinician_flags).
        config: Loaded beam_suggestions_config (loaded if None).

    Returns:
        List of chosen suggestion dicts with id, band, phase, text, evidence, sources.
    """
    sugg_cfg = config or get_beam_suggestions_config()
    all_suggestions = sugg_cfg.get("suggestions", [])
    rules_cfg = get_beam_rules_config()

    if include_optional is None:
        include_optional = rules_cfg.get("clinician_flags", {}).get("include_optional_suggestions", False)

    ctx = context or {}
    prev_ids: Set[str] = set(previous_suggestion_ids or [])
    user_phase = phase.lower()

    # Map engine band name to suggestions band name
    band_key = band
    if band == "good_balanced":
        band_key = "good"
    elif band == "okay_mixed":
        band_key = "okay"
    elif band == "tougher_low":
        band_key = "tougher"
    elif band in ("high_energy", "mixed_swings"):
        band_key = "high_energy"

    # Eligible pool from primary band
    eligible = []
    for item in all_suggestions:
        # Check band
        if item["band"] != band_key:
            continue

        # Check phase compatibility
        item_phase = item.get("phase", "any")
        if item_phase != "any" and item_phase != user_phase:
            continue

        # Check optional flag
        if item.get("optional", False) and not include_optional:
            continue

        # Check only_if condition
        only_if = item.get("only_if")
        if only_if and not evaluate_only_if_condition(only_if, ctx):
            continue

        eligible.append(item)

    # Add sleep or anxiety / irritable addon if flagged in context
    addon_pool = []
    if ctx.get("weekly_addon") == "anxious_or_irritable":
        for item in all_suggestions:
            if item["band"] in ("anxious_addon", "irritable_addon"):
                item_phase = item.get("phase", "any")
                if item_phase != "any" and item_phase != user_phase:
                    continue
                if item.get("optional", False) and not include_optional:
                    continue
                only_if = item.get("only_if")
                if only_if and not evaluate_only_if_condition(only_if, ctx):
                    continue
                addon_pool.append(item)

    # Prefer non-repeated suggestions first
    fresh_candidates = [s for s in eligible if s["id"] not in prev_ids]
    repeat_candidates = [s for s in eligible if s["id"] in prev_ids]

    chosen: List[Dict[str, Any]] = []

    # Pick up to count - 1 from fresh primary candidates
    target_primary = count - 1 if addon_pool else count
    for s in fresh_candidates:
        if len(chosen) >= target_primary:
            break
        chosen.append(s)

    # If addon pool exists, pick 1 relevant addon
    if addon_pool and len(chosen) < count:
        fresh_addons = [a for a in addon_pool if a["id"] not in prev_ids]
        addon_to_add = fresh_addons[0] if fresh_addons else addon_pool[0]
        chosen.append(addon_to_add)

    # If still below minimum (2), backfill from repeat candidates
    if len(chosen) < 2:
        for s in repeat_candidates:
            if s["id"] not in [c["id"] for c in chosen]:
                chosen.append(s)
            if len(chosen) >= count:
                break

    return chosen
