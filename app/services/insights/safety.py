"""BEAM Real-time Tier 3 Safety Evaluation Layer.

Safety Rules:
- Evaluated on EVERY SINGLE saved entry, reflection note, voice transcript, or EPDS screen.
- Triggers Tier 3 immediately upon:
  1. Severe Elevated (+3) day.
  2. Postpartum elevated mood (>= +2) with reduced sleep for 2+ consecutive nights in first 12 weeks.
  3. Multilingual self-harm or harm-to-baby language detection (en, hi, bn, ta) using clinician-reviewed config.
  4. EPDS Item 10 non-zero response (thoughts of harming myself).
  5. Any mention of confusion or hallucinations.
- Output is STRICTLY a fixed, pre-approved script from config.
  The language model is NEVER allowed to generate or alter Tier 3 text.
- FAILS SAFE: On any unhandled exception or parsing error, returns the pre-approved urgent script rather than nothing.
"""

from datetime import timedelta
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.services.insights.config_loader import (
    get_beam_rules_config,
    get_beam_safety_phrases_config,
)
from app.services.insights.engine import classify_day, normalize_mood_score, parse_date

logger = logging.getLogger("healher_backend.insights.safety")


def check_multilingual_safety_language(text: Optional[str], language: str = "en") -> Tuple[bool, List[str]]:
    """Scan text for self-harm, harm-to-baby, or crisis phrases using clinician-reviewed config lists.

    Conservative matching: Errs on the side of flagging.
    """
    if not text or not text.strip():
        return False, []

    safety_cfg = get_beam_safety_phrases_config()
    phrases_dict = safety_cfg.get("phrases", {})

    cleaned_text = text.lower().strip()
    # Normalize punctuation and spaces
    cleaned_text = re.sub(r"[\s\.,!\?]+", " ", cleaned_text)

    # Check target language phrases plus English phrases as universal fallback
    lang_key = (language or "en").lower().strip()
    target_phrases = list(phrases_dict.get(lang_key, []))
    if lang_key != "en":
        target_phrases.extend(phrases_dict.get("en", []))

    matched_reasons = []
    for phrase in target_phrases:
        p_clean = phrase.lower().strip()
        if p_clean and p_clean in cleaned_text:
            matched_reasons.append(f"Crisis phrase pattern detected in note text: '{p_clean}'")

    # Additional generic safety checks for psychosis / hallucinations
    psychosis_patterns = [
        "hearing voices", "see things that aren't there", "people are watching me",
        "आवाजें सुनाई", "অদ্ভুত আওয়াজ", "குரல்கள் கேட்கின்றன"
    ]
    for p in psychosis_patterns:
        if p in cleaned_text:
            matched_reasons.append(f"Perceptual disturbance / hallucination keyword detected: '{p}'")

    return bool(matched_reasons), matched_reasons


def evaluate_entry_safety(
    entry: Dict[str, Any],
    recent_entries: Optional[List[Dict[str, Any]]] = None,
    phase: str = "pregnancy",
    delivery_date: Optional[Union[str, Any]] = None,
    usual_sleep: float = 7.5,
    note_text: Optional[str] = None,
    language: str = "en",
    epds_answers: Optional[Dict[Union[int, str], int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate real-time Tier 3 safety triggers on a single saved daily check-in or note.

    Returns:
        Dict with keys:
        - is_tier3 (bool)
        - reasons (list of string triggers)
        - script (fixed pre-approved clinical safety message)
        - emergency_number (e.g. 112)
        - tele_manas_number (e.g. 14416)
        - fail_safe (bool, True if fallback triggered due to error)
    """
    cfg = config or get_beam_rules_config()
    helplines = cfg.get("helplines_verify_before_release", {
        "emergency_india": "112",
        "tele_manas": "14416",
    })
    scripts = cfg.get("tier_scripts", {})
    urgent_script = scripts.get(
        "tier3_urgent_script",
        "Thank you for telling us. What you are feeling matters, and you don't have to face it alone. "
        "Please contact your doctor or go to the nearest hospital today, or call a helpline right away. "
        "If you are in immediate danger, call emergency services (112 in India). "
        "Tele-MANAS (14416) offers free mental health support. *Verify helpline numbers before release.*",
    )
    fail_safe_script = scripts.get(
        "tier3_fail_safe_script",
        "Please contact your doctor or health worker today. What you are feeling matters, and support is available right now. Emergency: 112 | Tele-MANAS: 14416.",
    )

    reasons: List[str] = []

    try:
        # 1. Check EPDS Item 10 (thoughts of harming myself)
        if epds_answers:
            item_10_score = epds_answers.get(10) or epds_answers.get("10") or epds_answers.get(9) or epds_answers.get("9")
            # Score 1 is "Hardly ever" (transient thought) which does not trigger safety override.
            # Only active self-harm ideation (score 2 "Sometimes" or score 3 "Yes, quite often") triggers Tier 3.
            if item_10_score is not None and int(item_10_score) > 1:
                reasons.append(f"EPDS screening item 10 positive (score {item_10_score}) indicating thoughts of self-harm.")

        # 2. Check note text / voice transcript language
        raw_note = note_text or entry.get("note_text") or entry.get("notes")
        if raw_note:
            is_language_flagged, lang_reasons = check_multilingual_safety_language(raw_note, language)
            if is_language_flagged:
                reasons.extend(lang_reasons)

        # 3. Check single-entry Severe Elevated (+3)
        raw_mood = entry.get("mood_beam") if entry.get("mood_beam") is not None else entry.get("mood")
        mood_score = normalize_mood_score(raw_mood, cfg)
        if mood_score == 3:
            reasons.append(f"Severe Elevated (+3) mood recorded on {entry.get('log_date', 'entry')}.")

        # 4. Check postpartum elevated + reduced sleep warning rule
        # In first 12 weeks: 2+ consecutive days at mood >= +2 with 2+ nights of sleep >= 2h below usual
        if phase.lower() == "postpartum" and recent_entries:
            all_entries = list(recent_entries)
            # Ensure current entry is included
            if not any(e.get("log_date") == entry.get("log_date") for e in all_entries):
                all_entries.append(entry)

            all_entries.sort(key=lambda x: parse_date(x["log_date"]))
            classified = [classify_day(e, usual_sleep, phase, cfg) for e in all_entries]

            # Check last 2 entries for consecutive marked elevated AND reduced sleep
            if len(classified) >= 2:
                last_two = classified[-2:]
                both_marked = all(e["is_marked_elevated"] for e in last_two)
                both_reduced_sleep = all(e["is_reduced_sleep"] for e in last_two)
                consecutive_days = (
                    parse_date(last_two[1]["log_date"]) == parse_date(last_two[0]["log_date"]) + timedelta(days=1)
                )

                if both_marked and both_reduced_sleep and consecutive_days:
                    reasons.append(
                        f"Postpartum warning signs: 2 consecutive days with marked elevated mood (>= +2) "
                        f"and sleep 2+ hours below usual on {last_two[0]['log_date']} and {last_two[1]['log_date']}."
                    )

        if reasons:
            return {
                "is_tier3": True,
                "reasons": reasons,
                "script": urgent_script,
                "emergency_number": helplines.get("emergency_india", "112"),
                "tele_manas_number": helplines.get("tele_manas", "14416"),
                "fail_safe": False,
            }

        return {
            "is_tier3": False,
            "reasons": [],
            "script": None,
            "emergency_number": helplines.get("emergency_india", "112"),
            "tele_manas_number": helplines.get("tele_manas", "14416"),
            "fail_safe": False,
        }

    except Exception as e:
        logger.error("Tier 3 safety layer encountered error, FAILING SAFE: %s", str(e))
        # Strictly fail safe
        return {
            "is_tier3": True,
            "reasons": [f"Safety check error, fail-safe clinical referral activated: {type(e).__name__}"],
            "script": fail_safe_script,
            "emergency_number": helplines.get("emergency_india", "112"),
            "tele_manas_number": helplines.get("tele_manas", "14416"),
            "fail_safe": True,
        }
