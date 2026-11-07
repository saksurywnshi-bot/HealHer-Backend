"""FastAPI Endpoints for BEAM Perinatal Insights, Screenings, and Real-time Safety.

Endpoints:
- POST /api/insights/check-entry: Real-time Tier 3 safety evaluation on saving an entry or note.
- POST /api/insights/evaluate: 7-day and 30-day BEAM chart evaluation and insight phrasing.
- GET  /api/insights/latest: Retrieve most recently generated insights for the user.
- GET  /api/insights/profile: Retrieve user phase, due date, delivery date, baseline sleep.
- POST /api/insights/profile: Upsert user phase and sleep baseline.
- GET  /api/insights/consent: Check if user has consented to insights analysis.
- POST /api/insights/consent: Record user informed consent.
- GET  /api/insights/questionnaires: Retrieve clinical questionnaire structures (EPDS, GAD-7, MDQ).
- POST /api/insights/questionnaires/submit: Score questionnaire, evaluate Tier 3 trigger, and save.
- GET  /api/insights/doctor-summary: Printable / PDF doctor-facing summary.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.config import SENTIMENT_RATE_LIMIT
from app.graphs.translate_graph import translation_graph
from app.limiter import limiter
from app.services.auth import get_current_user_ref
from app.services.insights.config_loader import (
    get_beam_questionnaires_config,
    get_beam_rules_config,
)
from app.services.insights.engine import evaluate
from app.services.insights.phrasing import generate_phrased_insights
from app.services.insights.safety import evaluate_entry_safety
from app.services.insights.storage import (
    get_recent_insights,
    get_user_consent,
    get_user_profile,
    save_generated_insight,
    save_screening,
    save_user_consent,
    save_user_profile,
)
from app.services.insights.suggestions import select_suggestions

logger = logging.getLogger("healher_backend.routes.insights")

router = APIRouter(prefix="/api/insights", tags=["insights"])


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------
class CheckEntryRequest(BaseModel):
    entry: Dict[str, Any] = Field(..., description="Daily entry dict with log_date, mood_beam, sleep_hours, etc.")
    recent_entries: Optional[List[Dict[str, Any]]] = Field(None, description="Recent 1-7 days entries for consecutive rules.")
    note_text: Optional[str] = Field(None, description="Optional accompanying journal or reflection note text.")
    language: Literal["en", "hi", "bn", "ta"] = Field("en", description="Language code.")


class CheckEntryResponse(BaseModel):
    is_tier3: bool
    reasons: List[str]
    script: Optional[str]
    emergency_number: str
    tele_manas_number: str
    fail_safe: bool


class DailyEntryModel(BaseModel):
    log_date: str
    mood_beam: Optional[Union[int, str]] = None
    anxiety: Optional[int] = 0
    irritability: Optional[int] = 0
    sleep_hours: Optional[float] = None
    notes: Optional[str] = None


class EvaluateInsightsRequest(BaseModel):
    entries: List[DailyEntryModel] = Field(..., description="List of logged daily check-in entries.")
    phase: Literal["pregnancy", "postpartum", "any"] = Field("pregnancy", description="Maternal phase.")
    today: Optional[str] = Field(None, description="Reference anchor date YYYY-MM-DD.")
    usual_sleep: Optional[float] = Field(None, description="Baseline sleep hours.")
    delivery_date: Optional[str] = Field(None, description="Postpartum delivery date YYYY-MM-DD.")
    language: Literal["en", "hi", "bn", "ta"] = Field("en", description="Target language for phrasing.")
    use_llm: bool = Field(True, description="Whether to query local Ollama model for phrasing or use template.")


class ProfileRequest(BaseModel):
    phase: Literal["pregnancy", "postpartum"] = Field("pregnancy")
    due_date: Optional[str] = None
    delivery_date: Optional[str] = None
    usual_sleep_hours: float = Field(7.5, ge=3.0, le=14.0)
    history_bipolar_psychosis: bool = False


class ConsentRequest(BaseModel):
    consented: bool = True


class QuestionnaireSubmitRequest(BaseModel):
    screening_type: Literal["epds", "gad7", "mdq", "phq9"]
    answers: Dict[str, int] = Field(..., description="Question index/id to integer score map.")


# ---------------------------------------------------------------------------
# Helper: Translate Phrased Output
# ---------------------------------------------------------------------------
async def translate_phrased_output(phrased: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Translate user-facing phrasing strings into target Indic language."""
    if language == "en":
        return phrased

    items_to_translate = []
    if phrased.get("title"):
        items_to_translate.append({"key": "insights.title", "text": phrased["title"]})
    if phrased.get("opening"):
        items_to_translate.append({"key": "insights.opening", "text": phrased["opening"]})
    if phrased.get("strength"):
        items_to_translate.append({"key": "insights.strength", "text": phrased["strength"]})
    if phrased.get("sleep_note"):
        items_to_translate.append({"key": "insights.sleep_note", "text": phrased["sleep_note"]})
    if phrased.get("pattern_note"):
        items_to_translate.append({"key": "insights.pattern_note", "text": phrased["pattern_note"]})
    if phrased.get("tier_message"):
        items_to_translate.append({"key": "insights.tier_message", "text": phrased["tier_message"]})
    if phrased.get("disclaimer"):
        items_to_translate.append({"key": "insights.disclaimer", "text": phrased["disclaimer"]})

    for i, s in enumerate(phrased.get("suggestions", [])):
        items_to_translate.append({"key": f"insights.sugg_{i}", "text": s})

    try:
        graph_res = await translation_graph.ainvoke({
            "language": language,
            "items": items_to_translate,
        })
        translations = graph_res.get("translations", {})

        def get_t(k: str, default: Any):
            return translations.get(k, {}).get("text", default)

        translated_suggs = [
            get_t(f"insights.sugg_{i}", phrased["suggestions"][i])
            for i in range(len(phrased.get("suggestions", [])))
        ]

        return {
            "title": get_t("insights.title", phrased.get("title")),
            "opening": get_t("insights.opening", phrased.get("opening")),
            "strength": get_t("insights.strength", phrased.get("strength")),
            "sleep_note": get_t("insights.sleep_note", phrased.get("sleep_note")),
            "pattern_note": get_t("insights.pattern_note", phrased.get("pattern_note")),
            "suggestions": translated_suggs,
            "tier_message": get_t("insights.tier_message", phrased.get("tier_message")),
            "tier": phrased.get("tier"),
            "disclaimer": get_t("insights.disclaimer", phrased.get("disclaimer")),
            "source": phrased.get("source"),
        }
    except Exception as e:
        logger.warning("Translation graph failed for insights (%s), returning English phrasing.", str(e))
        return phrased


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/check-entry", response_model=CheckEntryResponse, summary="Evaluate real-time Tier 3 safety on single entry")
@limiter.limit(SENTIMENT_RATE_LIMIT)
async def check_entry(
    request: Request,
    body: CheckEntryRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Real-time safety endpoint called every time an entry, voice note, or reflection is saved."""
    profile = get_user_profile(user_ref) or {}
    phase = profile.get("phase", "pregnancy")
    usual_sleep = profile.get("usual_sleep_hours", 7.5)
    delivery_date = profile.get("delivery_date")

    safety_result = evaluate_entry_safety(
        entry=body.entry,
        recent_entries=body.recent_entries,
        phase=phase,
        delivery_date=delivery_date,
        usual_sleep=usual_sleep,
        note_text=body.note_text,
        language=body.language,
    )
    return CheckEntryResponse(**safety_result)


@router.post("/evaluate", summary="Run full BEAM insights evaluation and phrasing")
@limiter.limit(SENTIMENT_RATE_LIMIT)
async def evaluate_insights(
    request: Request,
    body: EvaluateInsightsRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Classifies 7, 14, and 30-day chart entries, selects suggestions, and generates phrased insights."""
    profile = get_user_profile(user_ref) or {}
    phase = body.phase or profile.get("phase", "pregnancy")
    usual_sleep = body.usual_sleep or profile.get("usual_sleep_hours", 7.5)
    delivery_date = body.delivery_date or profile.get("delivery_date")

    dict_entries = [e.model_dump() for e in body.entries]

    # Pure rule engine evaluation
    eval_result = evaluate(
        entries=dict_entries,
        phase=phase,
        today=body.today,
        usual_sleep=usual_sleep,
        delivery_date=delivery_date,
    )

    # Suggestion selection
    chosen_suggestions = select_suggestions(
        band=eval_result["weekly_band"],
        phase=phase,
        context={
            "tier": eval_result["tier"],
            "weekly_addon": eval_result.get("weekly_addon"),
            "impaired_depressed_days": sum(
                1 for e in dict_entries if e.get("mood_beam") in (-2, -3, "Moderate Depressed", "Severe Depressed")
            ),
        },
    )

    # Phrasing layer
    phrased = generate_phrased_insights(
        evaluation_result=eval_result,
        suggestions=chosen_suggestions,
        phase=phase,
        use_llm=body.use_llm,
    )

    # Translate if requested Indic language
    translated_phrased = await translate_phrased_output(phrased, body.language)

    # Save generated insight for audit
    insight_id = save_generated_insight(
        user_ref=user_ref,
        insight_type="weekly",
        anchor_date=body.today or str(dict_entries[-1]["log_date"] if dict_entries else "now"),
        weekly_band=eval_result.get("weekly_band"),
        weekly_addon=eval_result.get("weekly_addon"),
        tier=eval_result["tier"],
        tier_name=eval_result["tier_name"],
        reasons=eval_result.get("reasons", []),
        chosen_suggestion_ids=[s["id"] for s in chosen_suggestions],
        phrased_text=translated_phrased,
    )

    return {
        "insight_id": insight_id,
        "evaluation": eval_result,
        "suggestions": chosen_suggestions,
        "phrased": translated_phrased,
        "language": body.language,
    }


@router.get("/latest", summary="Get user's latest insight and active tier")
async def get_latest_insights(user_ref: str = Depends(get_current_user_ref)):
    recent = get_recent_insights(user_ref, limit=1)
    if not recent:
        return {"has_insight": False, "insight": None}
    return {"has_insight": True, "insight": recent[0]}


@router.get("/profile", summary="Get user phase and sleep baseline")
async def get_profile(user_ref: str = Depends(get_current_user_ref)):
    profile = get_user_profile(user_ref)
    if not profile:
        return {
            "user_ref": user_ref,
            "phase": "pregnancy",
            "due_date": None,
            "delivery_date": None,
            "usual_sleep_hours": 7.5,
            "history_bipolar_psychosis": False,
        }
    return profile


@router.post("/profile", summary="Update user phase and sleep baseline")
async def update_profile(body: ProfileRequest, user_ref: str = Depends(get_current_user_ref)):
    return save_user_profile(
        user_ref=user_ref,
        phase=body.phase,
        due_date=body.due_date,
        delivery_date=body.delivery_date,
        usual_sleep_hours=body.usual_sleep_hours,
        history_bipolar_psychosis=body.history_bipolar_psychosis,
    )


@router.get("/consent", summary="Check user informed consent status")
async def check_consent(user_ref: str = Depends(get_current_user_ref)):
    return {"consented": get_user_consent(user_ref)}


@router.post("/consent", summary="Record user informed consent")
async def record_consent(body: ConsentRequest, user_ref: str = Depends(get_current_user_ref)):
    return save_user_consent(user_ref, consented=body.consented)


@router.get("/questionnaires", summary="Get questionnaires metadata and items")
async def get_questionnaires():
    return get_beam_questionnaires_config()


@router.post("/questionnaires/submit", summary="Score questionnaire and evaluate safety triggers")
async def submit_questionnaire(
    body: QuestionnaireSubmitRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    q_cfg = get_beam_questionnaires_config().get(body.screening_type)
    if not q_cfg:
        raise HTTPException(status_code=400, detail=f"Unknown questionnaire type '{body.screening_type}'")

    total_score = sum(int(v) for v in body.answers.values())
    self_harm_flag = False
    positive_screen = False

    # Check EPDS
    if body.screening_type == "epds":
        cutoff = q_cfg.get("cutoff_positive", 10)
        positive_screen = total_score >= cutoff
        # Item 10 (index 9 or 10): score 1 is "Hardly ever" which does not trigger warning; scores >= 2 ("Sometimes", "Yes, quite often") trigger self-harm flag.
        item_10 = body.answers.get("10") or body.answers.get("9") or 0
        if int(item_10) > 1:
            self_harm_flag = True

    # Check GAD-7
    elif body.screening_type == "gad7":
        cutoff = q_cfg.get("cutoff_positive", 5)
        positive_screen = total_score >= cutoff

    # Check MDQ
    elif body.screening_type == "mdq":
        cutoff = q_cfg.get("cutoff_positive_yes_count", 7)
        positive_screen = total_score >= cutoff

    # Check PHQ-9
    elif body.screening_type == "phq9":
        cutoff = q_cfg.get("cutoff_positive", 10)
        positive_screen = total_score >= cutoff
        # Item 9 self-harm thoughts (index 8 or 9)
        item_9 = body.answers.get("9") or body.answers.get("8") or 0
        if int(item_9) > 0:
            self_harm_flag = True

    scr_id = save_screening(
        user_ref=user_ref,
        screening_type=body.screening_type,
        total_score=total_score,
        positive_screen=positive_screen,
        self_harm_flag=self_harm_flag,
        answers=body.answers,
    )

    rules_cfg = get_beam_rules_config()
    scripts = rules_cfg.get("tier_scripts", {})
    helplines = rules_cfg.get("helplines_verify_before_release", {})

    # If self-harm flagged, escalate to Tier 3 immediately
    tier = 3 if self_harm_flag else (2 if positive_screen else 0)
    tier_message = None
    if tier == 3:
        tier_message = scripts.get("tier3_urgent_script")
    elif tier == 2:
        tier_message = (
            "Your responses suggest it is worth discussing how you've been feeling with your doctor or health worker. "
            "You can share a summary of your recent charts with them."
        )

    return {
        "screening_id": scr_id,
        "screening_type": body.screening_type,
        "total_score": total_score,
        "positive_screen": positive_screen,
        "self_harm_flag": self_harm_flag,
        "tier": tier,
        "tier_message": tier_message,
        "helplines": helplines,
        "disclaimer": "This screening tool does not diagnose. Please speak to a healthcare professional.",
    }


@router.get("/doctor-summary", summary="Generate shareable doctor summary")
async def get_doctor_summary(user_ref: str = Depends(get_current_user_ref)):
    """Provides a structured clinical report according to mood, sleep hours, and reflection sentiment analysis."""
    from collections import Counter
    from datetime import datetime, timezone
    from app.services.patterns_service import compute_sentiment_patterns
    from app.services.sentiment_db import get_sentiment_results_for_period

    profile = get_user_profile(user_ref) or {}
    insights = get_recent_insights(user_ref, limit=4)
    latest_insight = insights[0] if insights else None

    # Compute longitudinal sentiment patterns (monthly and weekly fallback)
    month_patterns = compute_sentiment_patterns(user_ref, period="month")
    week_patterns = compute_sentiment_patterns(user_ref, period="week")

    sentiment_notes = get_sentiment_results_for_period(user_ref, days=30)
    tones = Counter(n.get("tone") for n in sentiment_notes if n.get("tone"))

    # Top recurring themes from maternal reflections
    raw_themes = month_patterns.get("top_themes") or week_patterns.get("top_themes") or []
    top_themes = []
    for t in raw_themes:
        tot = t.get("total") or t.get("total_notes") or len(sentiment_notes) or 1
        cnt = t.get("count", 0)
        pct = round((cnt / max(1, tot)) * 100)
        top_themes.append({
            "theme": t.get("theme", ""),
            "theme_key": t.get("theme_key", f"sentiment.theme.{t.get('theme', '')}"),
            "count": cnt,
            "total": tot,
            "pct": pct,
        })

    # Sleep impact analysis
    sleep_comp = month_patterns.get("sleep_comparison") or week_patterns.get("sleep_comparison")
    sleep_impact = None
    if sleep_comp:
        sleep_impact = {
            "low_mood_avg_sleep": sleep_comp.get("low_tone_avg_sleep", 0),
            "other_avg_sleep": sleep_comp.get("other_avg_sleep", 0),
            "difference": round(abs(sleep_comp.get("low_tone_avg_sleep", 0) - sleep_comp.get("other_avg_sleep", 0)), 1),
        }

    # Extract qualitative reflection pattern sentences
    raw_patterns = month_patterns.get("pattern_sentences") or week_patterns.get("pattern_sentences") or []
    pattern_sentences = []
    for p in raw_patterns:
        tpl = p.get("template_key", "")
        ph = p.get("params") or p.get("placeholders") or {}
        if "theme_frequent" in tpl and ph.get("theme"):
            pattern_sentences.append(f"Primary recurring theme is '{str(ph.get('theme')).replace('_', ' ')}' (expressed in {ph.get('count')}/{ph.get('total')} reflections).")
        elif "consecutive_low" in tpl and ph.get("streak"):
            pattern_sentences.append(f"Recorded a streak of {ph.get('streak')} consecutive days with low/anxious reflections.")
        elif "sleep_impact" in tpl and ph.get("low_sleep") and ph.get("other_sleep"):
            pattern_sentences.append(f"Recorded lower sleep on low-mood days ({ph.get('low_sleep')}h) compared to other days ({ph.get('other_sleep')}h).")

    # Attention flags
    flag_reasons = []
    if latest_insight:
        flag_reasons.extend(latest_insight.get("reasons", []))
    if month_patterns.get("suggest_talk_to_someone"):
        flag_reasons.append("Sustained low/anxious reflection pattern indicates patient could benefit from clinical discussion.")
    if sleep_impact and sleep_impact["difference"] >= 1.5:
        flag_reasons.append(f"Noticeable sleep vulnerability: patient sleeps {sleep_impact['difference']}h less on lower-mood days.")

    # Clinical consultation points synthesizing Mood + Sleep + Reflections
    primary_theme = top_themes[0]["theme"] if top_themes else "daily adjustment"
    weekly_band = latest_insight.get("weekly_band", "good_balanced") if latest_insight else "good_balanced"

    discussion_points = [
        f"Assess emotional stability in context of {profile.get('phase', 'perinatal')} stage.",
        f"Review nightly sleep patterns against baseline ({profile.get('usual_sleep_hours', 7.5)}h/night).",
    ]
    if top_themes:
        theme_names = ", ".join(t["theme"].replace("_", " ") for t in top_themes[:2])
        discussion_points.append(f"Explore frequent concerns expressed in maternal reflections: {theme_names}.")
    if sleep_impact and sleep_impact["low_mood_avg_sleep"] < 6.0:
        discussion_points.append("Discuss practical support for nighttime awakenings and sleep fragmentation.")

    # 30-Day symptom tallies
    depressed_days = sum(1 for n in sentiment_notes if n.get("tone") in ("low", "anxious", "tired_overwhelmed", "distressed"))
    impaired_days = sum(1 for n in sentiment_notes if n.get("tone") == "distressed")
    balanced_days = max(0, len(sentiment_notes) - depressed_days)
    logged_sleeps = [n.get("sleep_hours") for n in sentiment_notes if n.get("sleep_hours") is not None]
    avg_sleep = round(sum(logged_sleeps) / len(logged_sleeps), 1) if logged_sleeps else profile.get("usual_sleep_hours", 7.5)

    return {
        "user_ref": user_ref,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Clinical discussion summary prepared for healthcare providers (OB-GYN, midwife, perinatal psychiatrist). Not a diagnostic instrument.",
        "header": {
            "title": "BEAM Perinatal Mood & Sleep Clinical Summary",
            "phase": profile.get("phase", "pregnancy"),
            "usual_sleep_hours": profile.get("usual_sleep_hours", 7.5),
            "generated_date": str(latest_insight.get("anchor_date") if latest_insight else "N/A"),
            "notice": "Draft perinatal insights summary prepared for clinical discussion. Not a diagnostic report.",
        },
        "latest_classification": {
            "weekly_band": weekly_band,
            "tier": latest_insight.get("tier", 0) if latest_insight else 0,
            "tier_name": latest_insight.get("tier_name", "Tier 0") if latest_insight else "Tier 0",
            "reasons": flag_reasons,
            "chosen_suggestions": latest_insight.get("chosen_suggestion_ids", []) if latest_insight else [],
        },
        "history": [
            {
                "date": ins.get("anchor_date"),
                "band": ins.get("weekly_band"),
                "tier": ins.get("tier"),
            }
            for ins in insights
        ],
        "phase_profile": {
            "perinatal_phase": profile.get("phase", "pregnancy"),
            "gestational_week": profile.get("gestational_week"),
            "postpartum_week": profile.get("postpartum_week"),
            "usual_sleep_hours": profile.get("usual_sleep_hours", 7.5),
        },
        "last_30_days": {
            "days_logged": len(sentiment_notes) or 7,
            "depressed_days": depressed_days,
            "impaired_depressed_days": impaired_days,
            "elevated_days": 0,
            "marked_elevated_days": 0,
            "balanced_days": balanced_days,
            "average_sleep_hours": avg_sleep,
        },
        "current_tier": latest_insight.get("tier", 0) if latest_insight else 0,
        "current_weekly_band": weekly_band,
        "flag_reasons": flag_reasons,
        "sentiment_analysis": {
            "total_reflections": len(sentiment_notes),
            "difficult_reflections_count": month_patterns.get("difficult_tones_count", 0),
            "longest_low_streak_days": month_patterns.get("longest_streak_days", 0),
            "tone_distribution": dict(tones),
            "top_recurring_themes": top_themes,
            "sleep_impact": sleep_impact,
            "pattern_sentences": pattern_sentences,
        },
        "clinical_recommendations": {
            "summary": f"Patient's recent longitudinal trajectory reflects a {weekly_band.replace('_', ' ')} trend with primary reflection concerns centered around {primary_theme.replace('_', ' ')}.",
            "focus_areas": [
                "Perinatal Mood Trajectory",
                "Sleep Duration & Fragmentation",
                "Maternal Reflection Themes & Coping",
            ],
            "discussion_points": discussion_points,
        },
    }
