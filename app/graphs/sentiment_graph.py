"""LangGraph workflow for sentiment analysis in HealHer.

Steps:
1. crisis_rules (keyword check, skips LLM if crisis matched)
2. validate_input (word/char check; eligible if words >= 10 or chars >= 50)
3. analyze_with_llm (structured JSON output from Gemini/provider)
4. validate_output (enums, stage-valid themes, non-diagnostic reflection < 35 words; retries once)
5. escalation_and_mismatch (plain code)
6. pick_suggestions (plain code matching data/suggestions.json)
7. save_result (SQLite sentiment_results, no note text)
8. respond (i18n keys and approved suggestions)
"""

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, StateGraph

from app.config.crisis_keywords import contains_crisis_keywords
from app.config.diagnostic_terms import contains_diagnostic_terms
from app.services.llm_provider import (
    ALLOWED_THEMES,
    POSTNATAL_PROHIBITED_THEMES,
    PRENATAL_PROHIBITED_THEMES,
    get_llm_provider,
)
from app.services.sentiment_db import save_sentiment_result
from app.services.suggestion_selector import select_sentiment_suggestions
from app.services.sentiment_library import get_library_version_hash
from app.services.guidance_library import get_safe_fallback_text, get_verified_helplines

logger = logging.getLogger("healher_backend.sentiment_graph")

VALID_TONES = {
    "positive",
    "neutral",
    "tired_overwhelmed",
    "anxious",
    "low",
    "distressed",
}
VALID_INTENSITIES = {"low", "medium", "high"}


class SentimentState(TypedDict, total=False):
    # Inputs
    note_text: str
    language: str  # 'en' | 'hi' | 'bn' | 'ta'
    stage: str  # 'prenatal' | 'postnatal'
    checkin_id: Optional[str]
    mood_band: Optional[str]
    anxiety: Optional[int]
    sleep_hours: Optional[float]
    user_ref: str
    custom_llm_fn: Optional[Any]  # for testing/mocking

    # Internal pipeline state
    truncated_note: str
    attempt_count: int
    validation_error: Optional[str]
    raw_llm_output: Optional[Dict[str, Any]]

    # Outputs
    status: str  # 'ok' | 'too_short' | 'crisis' | 'unavailable'
    tone: Optional[str]
    tone_phrase_key: Optional[str]
    intensity: Optional[str]
    themes: List[str]
    theme_keys: List[str]
    reflection: Optional[str]
    risk_language: bool
    support: bool
    mismatch: bool
    suggestions: List[Dict[str, Any]]
    library_version: Optional[str]
    helplines: Optional[Dict[str, str]]
    safe_fallback_text: Optional[str]
    disclaimer_key: str


# ---------------------------------------------------------------------------
# Node 1: Crisis Rules
# ---------------------------------------------------------------------------
def check_crisis_rules(state: SentimentState) -> Dict[str, Any]:
    note = state.get("note_text", "")
    lang = state.get("language", "en")

    if contains_crisis_keywords(note, lang):
        return {
            "status": "crisis",
            "support": True,
            "mismatch": False,
            "tone": None,
            "tone_phrase_key": None,
            "intensity": None,
            "themes": [],
            "theme_keys": [],
            "reflection": None,
            "risk_language": True,
            "suggestions": [],
            "library_version": get_library_version_hash(),
            "helplines": get_verified_helplines(),
            "safe_fallback_text": get_safe_fallback_text(lang),
            "disclaimer_key": "sentiment.disclaimer",
        }

    return {"attempt_count": 0}


def detect_script_language(text: str, fallback_language: str = "en") -> str:
    """Detect if note text contains Indic script (Hindi, Bengali, Tamil).

    If 3 or more characters of an Indic script are found, returns that language code ('hi', 'bn', 'ta').
    Otherwise returns fallback_language.
    """
    if not text:
        return fallback_language
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    bengali = len(re.findall(r"[\u0980-\u09FF]", text))
    tamil = len(re.findall(r"[\u0B80-\u0BFF]", text))

    counts = [("hi", devanagari), ("bn", bengali), ("ta", tamil)]
    counts.sort(key=lambda x: x[1], reverse=True)
    best_lang, best_count = counts[0]

    if best_count >= 3:
        return best_lang
    return fallback_language


# ---------------------------------------------------------------------------
# Node 2: Validate Input
# ---------------------------------------------------------------------------
def validate_input(state: SentimentState) -> Dict[str, Any]:
    note = state.get("note_text", "").strip()
    words = note.split()
    chars_no_space = len(note.replace(" ", ""))

    # Eligible only if words >= 10 OR characters >= 50
    if len(words) < 10 and chars_no_space < 50:
        return {
            "status": "too_short",
            "support": False,
            "mismatch": False,
            "tone": None,
            "tone_phrase_key": None,
            "intensity": None,
            "themes": [],
            "theme_keys": [],
            "reflection": None,
            "risk_language": False,
            "suggestions": [],
            "disclaimer_key": "sentiment.disclaimer",
        }

    # Automatically detect if note was written in Hindi, Bengali, or Tamil script
    resolved_lang = detect_script_language(note, state.get("language", "en"))

    # Truncate to first 300 words
    truncated = " ".join(words[:300])
    return {
        "truncated_note": truncated,
        "language": resolved_lang,
        "status": "pending",
    }


# ---------------------------------------------------------------------------
# Node 3: Analyze with LLM
# ---------------------------------------------------------------------------
async def analyze_with_llm(state: SentimentState) -> Dict[str, Any]:
    note = state.get("truncated_note", "")
    lang = state.get("language", "en")
    stage = state.get("stage", "prenatal")
    attempts = state.get("attempt_count", 0) + 1
    retry_instruction = state.get("validation_error")

    custom_fn = state.get("custom_llm_fn")
    try:
        if custom_fn:
            output = await custom_fn(note, lang, stage, retry_instruction)
        else:
            provider = get_llm_provider(lang)
            output = await provider.analyze_note(
                note, lang, stage, retry_instruction=retry_instruction
            )
        return {
            "attempt_count": attempts,
            "raw_llm_output": output,
        }
    except Exception as exc:
        logger.error(
            "LLM analysis failed on attempt %d: %s", attempts, exc.__class__.__name__
        )
        return {
            "attempt_count": attempts,
            "raw_llm_output": None,
        }


# ---------------------------------------------------------------------------
# Node 4: Validate Output
# ---------------------------------------------------------------------------
def validate_output(state: SentimentState) -> Dict[str, Any]:
    raw = state.get("raw_llm_output")
    stage = state.get("stage", "prenatal")
    attempts = state.get("attempt_count", 1)

    if not isinstance(raw, dict):
        error_msg = "Model returned invalid non-dictionary JSON."
        if attempts >= 2:
            return {"status": "unavailable", "validation_error": error_msg}
        return {"status": "pending", "validation_error": error_msg}

    tone = raw.get("tone")
    intensity = raw.get("intensity")
    themes = raw.get("themes", [])
    reflection = raw.get("reflection", "")

    errors = []

    # 1. Enums valid
    if tone not in VALID_TONES:
        errors.append(f"Invalid tone '{tone}'. Must be one of {VALID_TONES}.")
    if intensity not in VALID_INTENSITIES:
        errors.append(
            f"Invalid intensity '{intensity}'. Must be one of {VALID_INTENSITIES}."
        )

    # 2. Themes validation
    if not isinstance(themes, list) or len(themes) > 3:
        errors.append("Themes must be a list of up to 3 items.")
    else:
        for t in themes:
            if t not in ALLOWED_THEMES:
                errors.append(f"Theme '{t}' is not in allowed list.")
            if stage == "prenatal" and t in PRENATAL_PROHIBITED_THEMES:
                errors.append(
                    f"Theme '{t}' is postpartum-only and prohibited in prenatal stage."
                )
            if stage == "postnatal" and t in POSTNATAL_PROHIBITED_THEMES:
                errors.append(
                    f"Theme '{t}' is prenatal-only and prohibited in postnatal stage."
                )

    # 3. Reflection validation
    reflection_words = reflection.split() if isinstance(reflection, str) else []
    if len(reflection_words) >= 35:
        errors.append("Reflection must be under 35 words.")
    if contains_diagnostic_terms(reflection):
        errors.append(
            "Reflection contains prohibited diagnostic/clinical wording."
        )

    if errors:
        error_msg = "; ".join(errors)
        logger.warning(
            "Validation failed on attempt %d: %s", attempts, error_msg
        )
        if attempts >= 2:
            return {"status": "unavailable", "validation_error": error_msg}
        return {"status": "pending", "validation_error": error_msg}

    # All valid
    return {
        "status": "ok",
        "tone": tone,
        "tone_phrase_key": f"sentiment.tone.{tone}",
        "intensity": intensity,
        "themes": themes,
        "theme_keys": [f"sentiment.theme.{t}" for t in themes],
        "reflection": reflection,
        "risk_language": bool(raw.get("risk_language", False)),
        "validation_error": None,
    }


# ---------------------------------------------------------------------------
# Node 5: Escalation & Mismatch
# ---------------------------------------------------------------------------
def compute_escalation_and_mismatch(state: SentimentState) -> Dict[str, Any]:
    tone = state.get("tone", "neutral")
    intensity = state.get("intensity", "low")
    risk_language = state.get("risk_language", False)
    mood_band = state.get("mood_band")

    # Escalation rule
    support = (
        risk_language
        or tone == "distressed"
        or (tone in ("low", "anxious") and intensity == "high")
    )

    # Mismatch rule
    mismatch = False
    if mood_band:
        band_lower = mood_band.lower()
        is_normal_or_elevated = (
            "normal" in band_lower
            or "elevated" in band_lower
            or "balanced" in band_lower
        )
        is_depressed = "depress" in band_lower or "low" in band_lower

        if tone in ("low", "anxious", "distressed") and is_normal_or_elevated:
            mismatch = True
        elif tone == "positive" and is_depressed:
            mismatch = True

    return {
        "support": support,
        "mismatch": mismatch,
    }


# ---------------------------------------------------------------------------
# Node 6: Pick Suggestions
# ---------------------------------------------------------------------------
def select_suggestions(state: SentimentState) -> Dict[str, Any]:
    tone = state.get("tone", "neutral")
    themes = state.get("themes", [])
    stage = state.get("stage", "prenatal")
    language = state.get("language", "en")
    user_ref = state.get("user_ref", "")
    risk_language = state.get("risk_language", False)
    support = state.get("support", False)
    escalation = "see_clinician" if support else "none"

    # Rule 1: If tone is 'distressed', risk_language is true, or escalation is anything
    # other than 'none': return NO library tips. Do not call the selector at all.
    if tone == "distressed" or risk_language or escalation != "none":
        return {"suggestions": []}

    suggestions = select_sentiment_suggestions(
        stage=stage,
        tone=tone,
        themes=themes,
        escalation=escalation,
        risk_language=risk_language,
        user_ref=user_ref,
        language=language,
        record_history=True,
    )
    return {"suggestions": suggestions}


# ---------------------------------------------------------------------------
# Node 7: Save Result
# ---------------------------------------------------------------------------
def save_result_to_db(state: SentimentState) -> Dict[str, Any]:
    user_ref = state.get("user_ref", "")
    if user_ref:
        save_sentiment_result(
            user_ref=user_ref,
            stage=state.get("stage", "prenatal"),
            tone=state.get("tone", "neutral"),
            intensity=state.get("intensity", "low"),
            themes=state.get("themes", []),
            risk_language=state.get("risk_language", False),
            support=state.get("support", False),
            mismatch=state.get("mismatch", False),
            checkin_id=state.get("checkin_id"),
            sleep_hours=state.get("sleep_hours"),
            mood_band=state.get("mood_band"),
        )
    return {"disclaimer_key": "sentiment.disclaimer"}


# ---------------------------------------------------------------------------
# Node 8: Final Respond
# ---------------------------------------------------------------------------
def respond(state: SentimentState) -> Dict[str, Any]:
    status = state.get("status", "unavailable")
    lib_version = get_library_version_hash()
    language = state.get("language", "en")
    support = state.get("support", False)
    risk_language = state.get("risk_language", False)
    tone = state.get("tone")

    needs_support = bool(support or risk_language or tone == "distressed" or status == "crisis")
    helplines = get_verified_helplines() if needs_support else None
    safe_text = get_safe_fallback_text(language) if needs_support else None

    if status != "ok":
        return {
            "status": status if status in ("crisis", "too_short") else "unavailable",
            "tone": None,
            "tone_phrase_key": None,
            "intensity": None,
            "themes": [],
            "theme_keys": [],
            "reflection": None,
            "suggestions": [],
            "library_version": lib_version,
            "support": needs_support,
            "mismatch": False,
            "disclaimer_key": "sentiment.disclaimer",
            "helplines": helplines,
            "safe_fallback_text": safe_text,
        }

    return {
        "status": "ok",
        "tone": state.get("tone"),
        "tone_phrase_key": state.get("tone_phrase_key"),
        "intensity": state.get("intensity"),
        "themes": state.get("themes", []),
        "theme_keys": state.get("theme_keys", []),
        "reflection": state.get("reflection"),
        "suggestions": state.get("suggestions", []),
        "library_version": lib_version,
        "support": needs_support,
        "mismatch": state.get("mismatch", False),
        "disclaimer_key": "sentiment.disclaimer",
        "helplines": helplines,
        "safe_fallback_text": safe_text,
    }


# ---------------------------------------------------------------------------
# Routing Conditions
# ---------------------------------------------------------------------------
def route_after_crisis(state: SentimentState) -> str:
    if state.get("status") == "crisis":
        return "respond"
    return "validate_input"


def route_after_input_validation(state: SentimentState) -> str:
    if state.get("status") == "too_short":
        return "respond"
    return "analyze_with_llm"


def route_after_output_validation(state: SentimentState) -> str:
    if state.get("status") == "ok":
        return "escalation_and_mismatch"
    if state.get("attempt_count", 0) < 2 and state.get("status") != "unavailable":
        return "analyze_with_llm"
    return "respond"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------
def build_sentiment_graph():
    builder = StateGraph(SentimentState)

    builder.add_node("check_crisis_rules", check_crisis_rules)
    builder.add_node("validate_input", validate_input)
    builder.add_node("analyze_with_llm", analyze_with_llm)
    builder.add_node("validate_output", validate_output)
    builder.add_node(
        "escalation_and_mismatch", compute_escalation_and_mismatch
    )
    builder.add_node("select_suggestions", select_suggestions)
    builder.add_node("save_result_to_db", save_result_to_db)
    builder.add_node("respond", respond)

    builder.set_entry_point("check_crisis_rules")

    builder.add_conditional_edges(
        "check_crisis_rules",
        route_after_crisis,
        {
            "respond": "respond",
            "validate_input": "validate_input",
        },
    )

    builder.add_conditional_edges(
        "validate_input",
        route_after_input_validation,
        {
            "respond": "respond",
            "analyze_with_llm": "analyze_with_llm",
        },
    )

    builder.add_edge("analyze_with_llm", "validate_output")

    builder.add_conditional_edges(
        "validate_output",
        route_after_output_validation,
        {
            "escalation_and_mismatch": "escalation_and_mismatch",
            "analyze_with_llm": "analyze_with_llm",
            "respond": "respond",
        },
    )

    builder.add_edge("escalation_and_mismatch", "select_suggestions")
    builder.add_edge("select_suggestions", "save_result_to_db")
    builder.add_edge("save_result_to_db", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


sentiment_graph = build_sentiment_graph()


def get_sentiment_graph_mermaid() -> str:
    """Return the Mermaid graph visualization."""
    return sentiment_graph.get_graph().draw_mermaid()
