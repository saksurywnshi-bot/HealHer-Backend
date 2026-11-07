"""FastAPI Router for Caregiver 'How You Can Help Her Today' Track & Admin Management.

Endpoints:
- GET /api/caregiver/daily: Returns 5 stage-aware daily tips for caregiver.
- POST /api/caregiver/score: Evaluates caregiver observation log and returns guidance with help_today tips.
- DELETE /api/me/caregiver-tip-history: Deletes tip history for authenticated user.
- GET /api/admin/caregiver-library: All tips, sources, coverage per stage, and repeat warnings.
- PATCH /api/admin/caregiver-library/tip/{id}: Edit status, translations, wording in overrides file.
- GET /api/admin/caregiver-library/untranslated: Tips missing translations.
- GET /api/admin/caregiver-library/stats: 30-day tip frequency and days with fewer than 5 tips.
"""

from datetime import date
import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.config import (
    ADMIN_TOKEN,
    CAREGIVER_DAILY_COUNT,
    RATE_LIMIT,
)
from app.limiter import limiter, get_user_rate_limit_key
from app.routes.admin import verify_admin_token
from app.services.auth import get_current_user_ref
from app.services.caregiver_db import (
    delete_user_caregiver_history,
    get_caregiver_admin_stats,
)
from app.services.caregiver_daily import (
    get_daily_caregiver_tips,
    get_kolkata_today,
)
from app.services.caregiver_library import (
    OVERRIDES_FILE_PATH,
    get_caregiver_library_version_hash,
    get_raw_caregiver_library,
    get_served_caregiver_library,
    load_overrides,
    reload_caregiver_library,
    validate_caregiver_library,
)
from app.services.guidance_library import get_verified_helplines

logger = logging.getLogger("healher_backend.routes.caregiver")

router = APIRouter(tags=["caregiver"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CaregiverTipItem(BaseModel):
    id: str
    kind: str
    kind_label_key: str
    text: str
    text_language: str


class CaregiverDailyResponse(BaseModel):
    stage: Literal["prenatal", "postnatal"]
    result_state: str
    help_today: List[CaregiverTipItem]
    library_version: str


class CaregiverScoreRequest(BaseModel):
    stage: Literal["prenatal", "postnatal"] = Field(
        ..., description="Maternal stage: prenatal or postnatal."
    )
    answers: Dict[str, Any] = Field(
        ..., description="Map of question IDs (q1..q15) to response values."
    )
    language: Literal["en", "hi", "bn", "ta"] = Field(
        "en", description="Target language code."
    )
    date: Optional[str] = Field(
        None, description="Optional local date YYYY-MM-DD for deterministic testing."
    )


class CaregiverScoreResponse(BaseModel):
    result_state: Literal["get_help_today", "talk_soon", "extra_support", "keep_supporting"]
    result_message: str
    safety_triggered: bool
    helplines: Dict[str, str]
    help_today: List[CaregiverTipItem]
    library_version: str
    # Maintained for backward compatibility for one release
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)


class PatchCaregiverTipPayload(BaseModel):
    status: Optional[Literal["draft", "approved"]] = None
    text_hi: Optional[str] = None
    text_bn: Optional[str] = None
    text_ta: Optional[str] = None
    text_en: Optional[str] = None


# ---------------------------------------------------------------------------
# Caregiver Evaluation Logic
# ---------------------------------------------------------------------------
MESSAGES_EN = {
    "get_help_today": "Please stay close to her and get support today. Call Tele-MANAS 14416 (free, any time) or 112 now, and contact her doctor today. You are doing the right thing by acting.",
    "talk_soon": "Some of the changes you have noticed may mean she needs more support. A talk with her doctor or midwife can help. You could offer to go with her.",
    "extra_support": "Some things at home may be making this a harder time for her. Small practical steps from the family can make a real difference. Mention these at her next check-up.",
    "keep_supporting": "Thank you for checking in. Your support matters. Keep staying close, listening and sharing the load. We will ask again at the next check.",
}

# Verbatim translated messages for Hindi, Bengali, Tamil
MESSAGES_LOCALIZED: Dict[str, Dict[str, str]] = {
    "get_help_today": {
        "en": "Please stay close to her and get support today. Call Tele-MANAS 14416 (free, any time) or 112 now, and contact her doctor today. You are doing the right thing by acting.",
        "hi": "कृपया उनके साथ रहें और आज ही सहायता प्राप्त करें। टेली-मानस 14416 (निःशुल्क, 24 घंटे) या 112 पर कॉल करें, और आज ही उनके डॉक्टर से संपर्क करें। कदम उठाकर आप बिल्कुल सही कर रहे हैं।",
        "bn": "অনুগ্রহ করে তার পাশে থাকুন এবং আজই সাহায্য নিন। টেলি-মানস 14416 (বিনামূল্যে, যেকোনো সময়) বা 112 নম্বরে কল করুন এবং আজই তার ডাক্তারের সাথে যোগাযোগ করুন। পদক্ষেপ নিয়ে আপনি সঠিক কাজটি করছেন।",
        "ta": "தயவுசெய்து அவருடன் இருங்கள் மற்றும் இன்றே உதவி பெறுங்கள். டெலி-மானாஸ் 14416 அல்லது 112 ஐ அழைக்கவும், இன்றே மருத்துவரை அணுகவும். நடவடிக்கை எடுப்பதன் மூலம் நீங்கள் சரியானதைச் செய்கிறீர்கள்.",
    },
    "talk_soon": {
        "en": "Some of the changes you have noticed may mean she needs more support. A talk with her doctor or midwife can help. You could offer to go with her.",
        "hi": "आपने जो बदलाव देखे हैं उनका मतलब हो सकता है कि उन्हें अधिक मदद की जरूरत है। डॉक्टर या मिडवाइफ से बात करना मददगार होगा। आप साथ जाने की पेशकश कर सकते हैं।",
        "bn": "আপনি যে পরিবর্তনগুলি লক্ষ্য করেছেন তার অর্থ হতে পারে তার অতিরিক্ত সহায়তার প্রয়োজন। ডাক্তার বা মিডওয়াইফের সাথে কথা বলা সহায়ক হবে। আপনি সাথে যাওয়ার প্রস্তাব দিতে পারেন।",
        "ta": "நீங்கள் கவனித்த மாற்றங்கள் அவருக்கு கூடுதல் ஆதரவு தேவை என்பதைக் குறிக்கலாம். மருத்துவர் அல்லது செவிலியரிடம் பேசுவது உதவக்கூடும். நீங்கள் அவருடன் செல்ல முன்வரலாம்.",
    },
    "extra_support": {
        "en": "Some things at home may be making this a harder time for her. Small practical steps from the family can make a real difference. Mention these at her next check-up.",
        "hi": "घर की कुछ परिस्थितियां इस समय को उनके लिए अधिक कठिन बना सकती हैं। परिवार के छोटे व्यावहारिक कदम बड़ा बदलाव ला सकते हैं। अगली जांच में डॉक्टर को यह बताएं।",
        "bn": "বাড়ির কিছু পরিস্থিতি এই সময়টিকে তার জন্য আরও কঠিন করে তুলতে পারে। পরিবারের ছোট ছোট পদক্ষেপ বাস্তব পরিবর্তন আনতে পারে। পরবর্তী চেক-আপে ডাক্তারকে এটি জানান।",
        "ta": "வீட்டுச் சூழ்நிலைகள் அவருக்கு இந்தக் காலகட்டத்தை கடினமாக்கலாம். குடும்பத்தினரின் சிறிய நடைமுறை உதவிகள் நல்ல மாற்றத்தை ஏற்படுத்தும். அடுத்த பரிசோதனையில் இதைக் குறிப்பிடுங்கள்.",
    },
    "keep_supporting": {
        "en": "Thank you for checking in. Your support matters. Keep staying close, listening and sharing the load. We will ask again at the next check.",
        "hi": "यह जानकारी साझा करने के लिए धन्यवाद। आपका सहयोग अमूल्य है। उनके करीब रहें, सुनें और जिम्मेदारियां साझा करें। हम अगली जांच में पुनः पूछेंगे।",
        "bn": "চেক-ইন করার জন্য ধন্যবাদ। আপনার সহযোগিতা অত্যন্ত মূল্যবান। পাশে থাকুন, শুনুন এবং কাজের চাপ ভাগ করে নিন। আমরা পরবর্তী চেক-আপে আবার জিজ্ঞাসা করব।",
        "ta": "பகிர்ந்தமைக்கு நன்றி. உங்கள் ஆதரவு முக்கியமானது. தொடர்ந்து அருகில் இருங்கள், கேளுங்கள் மற்றும் பொறுப்புகளைப் பகிர்ந்து கொள்ளுங்கள்.",
    },
}


def evaluate_caregiver_answers(
    stage: str, answers: Dict[str, Any]
) -> Dict[str, Any]:
    """Score the 15-question caregiver log:
    - Part 1 (Q1-Q7): Risk score (0 to 7), Q5 reversed (No = 1).
    - Part 2 (Q8-Q12): Signs score (0 to 10), Often=2, Sometimes=1.
    - Part 3 (Q13-Q15): Safety questions (Yes or Not sure -> get_help_today).
    """
    clean_answers: Dict[str, str] = {}
    for k, v in answers.items():
        norm_k = str(k).lower().strip().replace("_", "")
        if not norm_k.startswith("q") and norm_k.isdigit():
            norm_k = f"q{norm_k}"
        clean_answers[norm_k] = str(v).strip()

    # Part 3: Safety Questions (Q13 to Q15)
    safety_triggered = False
    for q_num in (13, 14, 15):
        val = clean_answers.get(f"q{q_num}", "")
        if val in ("Yes", "Not sure", "हाँ", "হ্যাঁ", "ஆம்"):
            safety_triggered = True
            break

    if safety_triggered:
        return {
            "result_state": "get_help_today",
            "safety_triggered": True,
            "risk_count": 0,
            "signs_count": 0,
            "flagged_areas": [],
        }

    # Part 1: Risk (Q1-Q7)
    risk_count = 0
    flagged_areas: List[str] = []

    # Q1: History
    if clean_answers.get("q1") in ("Yes", "हाँ", "হ্যাঁ", "ஆம்"):
        risk_count += 1
    # Q2: Stress
    if clean_answers.get("q2") in ("Yes", "हाँ", "হ্যাঁ", "ஆம்"):
        risk_count += 1
        flagged_areas.append("money_food")
    # Q3: Money / food
    if clean_answers.get("q3") in ("Yes", "हाँ", "হ্যাঁ", "ஆம்"):
        risk_count += 1
        if "money_food" not in flagged_areas:
            flagged_areas.append("money_food")
    # Q4: Tension
    if clean_answers.get("q4") in ("Yes", "हाँ", "হ্যাঁ", "ஆம்"):
        risk_count += 1
        flagged_areas.append("tension")
    # Q5: Reversed (No / lack of support = 1)
    if clean_answers.get("q5") in ("No", "नहीं", "না", "இல்லை"):
        risk_count += 1
        flagged_areas.append("no_support")
        flagged_areas.append("isolated")
    # Q6: Unplanned or Baby health/settling
    if clean_answers.get("q6") in ("Yes", "हाँ", "হ্যাঁ", "ஆம்"):
        risk_count += 1
        flagged_areas.append("baby_hard")
    # Q7: Loss or Blues
    if clean_answers.get("q7") in ("Yes", "हाँ", "হ্যাঁ", "ஆம்"):
        risk_count += 1

    # Part 2: Signs (Q8-Q12)
    signs_count = 0
    for q_num in range(8, 13):
        val = clean_answers.get(f"q{q_num}", "")
        if val in ("Often", "अक्सर", "প্রায়ই", "அடிக்கடி"):
            signs_count += 2
        elif val in ("Sometimes", "कभी-कभी", "কখনও কখনও", "எப்போதாவது"):
            signs_count += 1

    if signs_count >= 5:
        flagged_areas.append("signs_high")

    # State decision
    if signs_count >= 5 or (signs_count >= 3 and risk_count >= 2):
        result_state = "talk_soon"
    elif risk_count >= 2:
        result_state = "extra_support"
    else:
        result_state = "keep_supporting"

    return {
        "result_state": result_state,
        "safety_triggered": False,
        "risk_count": risk_count,
        "signs_count": signs_count,
        "flagged_areas": flagged_areas,
    }


# ---------------------------------------------------------------------------
# Caregiver User Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/api/caregiver/daily",
    response_model=CaregiverDailyResponse,
    summary="Get 5 stage-aware daily tips for caregiver",
)
@limiter.limit(RATE_LIMIT, key_func=get_user_rate_limit_key)
def get_caregiver_daily_tips_endpoint(
    request: Request,
    stage: Literal["prenatal", "postnatal"] = Query(
        "prenatal", description="Stage: prenatal or postnatal."
    ),
    language: Literal["en", "hi", "bn", "ta"] = Query(
        "en", description="Preferred language."
    ),
    result_state: str = Query(
        "keep_supporting",
        description="Caregiver result state: get_help_today, talk_soon, extra_support, keep_supporting",
    ),
    flags: Optional[List[str]] = Query(None, description="Optional flagged risk area keys."),
    date_str: Optional[str] = Query(None, alias="date", description="Optional date override (YYYY-MM-DD)."),
    user_ref: str = Depends(get_current_user_ref),
):
    """Retrieve exactly five stage-appropriate daily tips for the caregiver.
    
    Fixed all day (Asia/Kolkata), shuffles after midnight.
    """
    local_day: Optional[date] = None
    if date_str:
        try:
            local_day = date.fromisoformat(date_str)
        except Exception:
            local_day = get_kolkata_today()

    help_today, version_hash = get_daily_caregiver_tips(
        stage=stage,
        user_ref=user_ref,
        language=language,
        result_state=result_state,
        flags=flags or [],
        day=local_day,
        record_history=True,
    )

    return CaregiverDailyResponse(
        stage=stage,
        result_state=result_state,
        help_today=[CaregiverTipItem(**t) for t in help_today],
        library_version=version_hash,
    )


@router.post(
    "/api/caregiver/score",
    response_model=CaregiverScoreResponse,
    summary="Score caregiver observation log and return daily help tips",
)
@limiter.limit(RATE_LIMIT, key_func=get_user_rate_limit_key)
def score_caregiver_log(
    request: Request,
    body: CaregiverScoreRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Evaluate 15-question caregiver observation log.
    
    Returns result_state, clinical message, verified helplines, and 5 daily help tips.
    Caregiver answers are NEVER stored with note text.
    """
    eval_res = evaluate_caregiver_answers(body.stage, body.answers)
    result_state = eval_res["result_state"]
    safety_triggered = eval_res["safety_triggered"]
    flagged_areas = eval_res["flagged_areas"]

    local_day: Optional[date] = None
    if body.date:
        try:
            local_day = date.fromisoformat(body.date)
        except Exception:
            local_day = get_kolkata_today()

    help_today, version_hash = get_daily_caregiver_tips(
        stage=body.stage,
        user_ref=user_ref,
        language=body.language,
        result_state=result_state,
        flags=flagged_areas,
        day=local_day,
        record_history=True,
    )

    # Localized message
    state_msgs = MESSAGES_LOCALIZED.get(result_state, MESSAGES_LOCALIZED["keep_supporting"])
    msg = state_msgs.get(body.language, state_msgs.get("en", ""))

    return CaregiverScoreResponse(
        result_state=result_state,
        result_message=msg,
        safety_triggered=safety_triggered,
        helplines=get_verified_helplines(),
        help_today=[CaregiverTipItem(**t) for t in help_today],
        library_version=version_hash,
        suggestions=[],
    )


@router.delete(
    "/api/me/caregiver-tip-history",
    summary="Delete caregiver tip history for authenticated user",
)
def delete_my_caregiver_tip_history(
    user_ref: str = Depends(get_current_user_ref),
):
    """Delete all recorded caregiver tip history for the authenticated user."""
    deleted_count = delete_user_caregiver_history(user_ref)
    return {"success": True, "deleted_count": deleted_count}


# ---------------------------------------------------------------------------
# Admin Endpoints (Protected by x-admin-token)
# ---------------------------------------------------------------------------
@router.get(
    "/api/admin/caregiver-library",
    summary="List all tips, sources, coverage per stage, and repeat warnings",
)
def get_caregiver_library_admin(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Admin inspect caregiver library with coverage and repeat warning check."""
    verify_admin_token(x_admin_token)

    raw_lib = get_raw_caregiver_library()
    suggestions = raw_lib.get("suggestions", [])
    sources = raw_lib.get("sources", {})

    # Compute coverage per stage (served tips)
    # Check coverage for prenatal and postnatal
    # A tip qualifies for stage if tip.stage in (stage, "both") AND status == "approved"
    coverage: Dict[str, Dict[str, int]] = {"prenatal": {}, "postnatal": {}}
    warnings: List[str] = []

    for stage in ("prenatal", "postnatal"):
        served_stage_tips = [
            t for t in suggestions
            if (t.get("status") == "approved" or t.get("_serve_draft")) and t.get("stage") in (stage, "both")
        ]
        kind_counts: Dict[str, int] = {}
        for t in served_stage_tips:
            k = t.get("kind", "do")
            kind_counts[k] = kind_counts.get(k, 0) + 1
        coverage[stage] = kind_counts

        total_stage = len(served_stage_tips)
        check_in_count = kind_counts.get("check_in", 0)
        light_count = kind_counts.get("small_joy", 0) + kind_counts.get("try_together", 0)
        say_count = kind_counts.get("say", 0)

        stage_warns = []
        if total_stage < 35:
            stage_warns.append(f"fewer than 35 served tips ({total_stage})")
        if check_in_count < 7:
            stage_warns.append(f"fewer than 7 check_in tips ({check_in_count})")
        if light_count < 14:
            stage_warns.append(f"fewer than 14 small_joy+try_together tips ({light_count})")
        if say_count < 7:
            stage_warns.append(f"fewer than 7 'say' tips ({say_count})")

        if stage_warns:
            warnings.append(
                f"Warning for {stage} stage: {', '.join(stage_warns)}. Below these thresholds, the daily mix will repeat within a week."
            )

    # Build tip list for admin
    tip_list = []
    for t in suggestions:
        tip_list.append({
            "id": t.get("id"),
            "status": t.get("status"),
            "evidence_level": t.get("evidence_level"),
            "source_refs": t.get("source_refs", []),
            "needs_source": t.get("needs_source", False),
            "flag_list": t.get("flag_list", []),
            "kind": t.get("kind"),
            "stage": t.get("stage"),
            "text_en": t.get("text_en"),
            "text_hi": t.get("text_hi", ""),
            "text_bn": t.get("text_bn", ""),
            "text_ta": t.get("text_ta", ""),
        })

    return {
        "version": raw_lib.get("version"),
        "version_hash": get_caregiver_library_version_hash(),
        "total_tips": len(suggestions),
        "coverage": coverage,
        "warning": " | ".join(warnings) if warnings else None,
        "sources": sources,
        "suggestions": tip_list,
    }


@router.patch(
    "/api/admin/caregiver-library/tip/{tip_id}",
    summary="Update tip status or edit translations/wording",
)
def patch_caregiver_tip_admin(
    tip_id: str,
    payload: PatchCaregiverTipPayload,
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Clinician edits to tip status, translations, or wording.
    
    Refuses approved if needs_source is true or source_refs is empty.
    Merges into data/caregiver_overrides.json and re-validates.
    """
    verify_admin_token(x_admin_token)

    raw_lib = get_raw_caregiver_library()
    suggestions = raw_lib.get("suggestions", [])
    target_tip = next((t for t in suggestions if t.get("id") == tip_id), None)

    if not target_tip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tip '{tip_id}' not found in caregiver library.",
        )

    # Validation: refuse approved if needs_source or source_refs is empty
    if payload.status == "approved":
        if target_tip.get("needs_source") is True:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot approve tip '{tip_id}': needs_source is True.",
            )
        if not target_tip.get("source_refs"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot approve tip '{tip_id}': source_refs is empty.",
            )

    # Validation: if text_en edited, check no digits or forbidden words
    if payload.text_en is not None:
        if not payload.text_en.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text_en cannot be empty.",
            )
        if re.search(r"\d", payload.text_en):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text_en must not contain digits.",
            )

    # Load existing overrides
    overrides = load_overrides()
    if tip_id not in overrides:
        overrides[tip_id] = {}

    patch_dict = payload.model_dump(exclude_unset=True)
    for k, v in patch_dict.items():
        if v is not None:
            overrides[tip_id][k] = v

    # Write to caregiver_overrides.json
    try:
        with open(OVERRIDES_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write overrides: {e}",
        )

    # Reload and re-validate
    try:
        reload_caregiver_library()
    except Exception as e:
        logger.error("Failed to re-validate library after override: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Library re-validation failed: {e}",
        )

    updated_lib = get_raw_caregiver_library()
    updated_tip = next((t for t in updated_lib["suggestions"] if t["id"] == tip_id), None)

    return {
        "success": True,
        "tip": updated_tip,
        "library_version": get_caregiver_library_version_hash(),
    }


@router.get(
    "/api/admin/caregiver-library/untranslated",
    summary="List tips missing Hindi, Bengali, or Tamil translations",
)
def get_untranslated_caregiver_tips(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Retrieve tips that are missing one or more Indic translations."""
    verify_admin_token(x_admin_token)

    raw_lib = get_raw_caregiver_library()
    untranslated = []

    for t in raw_lib.get("suggestions", []):
        missing = []
        if not t.get("text_hi", "").strip():
            missing.append("hi")
        if not t.get("text_bn", "").strip():
            missing.append("bn")
        if not t.get("text_ta", "").strip():
            missing.append("ta")

        if missing:
            untranslated.append({
                "id": t.get("id"),
                "text_en": t.get("text_en"),
                "missing_languages": missing,
                "status": t.get("status"),
            })

    return {
        "count": len(untranslated),
        "untranslated": untranslated,
    }


@router.get(
    "/api/admin/caregiver-library/stats",
    summary="Get 30-day tip frequency and days with fewer than 5 tips",
)
def get_caregiver_library_stats_admin(
    x_admin_token: Optional[str] = Header(None, alias="x-admin-token"),
):
    """Compute aggregate counts for the last 30 days. No user identifiers."""
    verify_admin_token(x_admin_token)
    stats = get_caregiver_admin_stats(days=30)
    return stats

