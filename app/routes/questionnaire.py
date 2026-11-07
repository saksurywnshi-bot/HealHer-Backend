"""FastAPI Router for Questionnaire Guidance & Scoring (PHQ-9 and EPDS).

Endpoints:
- POST /api/questionnaire/guidance: Optional AI personalisation layer on questionnaire results.
- POST /api/questionnaire/score: Standard clinical scoring endpoint returning baseline guidance.
"""

import logging
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import (
    GUIDANCE_RATE_LIMIT,
    GUIDANCE_FIXED_THEMES,
)
from app.graphs.guidance_graph import run_guidance_workflow
from app.limiter import limiter, get_user_rate_limit_key
from app.services.auth import get_current_user_ref
from app.services.guidance_db import save_questionnaire_result
from app.services.guidance_library import (
    build_candidates,
    get_library_version_hash,
    get_safe_fallback_text,
    get_verified_helplines,
    order_deterministic,
)
from app.services.llm import get_band_message, translate_text_with_gemini

logger = logging.getLogger("healher_backend.routes.questionnaire")

router = APIRouter(prefix="/api/questionnaire", tags=["questionnaire"])


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------
class GuidanceRequest(BaseModel):
    result_id: str = Field(..., description="ID of the stored questionnaire result.")
    language: Literal["en", "hi", "bn", "ta"] = Field("en", description="Target language code.")
    themes: Optional[List[str]] = Field(None, description="Optional theme tags from the fixed 18-theme list.")


class ScoreRequest(BaseModel):
    instrument: Literal["phq9", "epds", "pbq"] = Field(..., description="Screening instrument.")
    answers: Dict[str, int] = Field(..., description="Question ID/index to option score map.")
    stage: Literal["prenatal", "postnatal"] = Field("postnatal", description="Maternal stage.")
    language: Literal["en", "hi", "bn", "ta"] = Field("en", description="User interface language.")


class SuggestionItem(BaseModel):
    id: str
    category: str
    text: str


class GuidanceResponse(BaseModel):
    band_id: str
    band_message: str
    escalation: str
    suggestions: List[SuggestionItem]
    safety_triggered: bool
    helplines: Dict[str, str]
    safe_fallback_text: str
    disclaimer_key: str
    previous: Optional[Dict[str, Any]] = None
    personalised: bool
    mode: Literal["none", "select", "reword"]
    library_version: str


class ScoreResponse(GuidanceResponse):
    result_id: str
    instrument: str
    total_score: int


# ---------------------------------------------------------------------------
# Scoring Helper Logic
# ---------------------------------------------------------------------------
def compute_instrument_score(
    instrument: str,
    answers: Dict[str, int],
) -> Dict[str, Any]:
    """Compute score, band, escalation, and safety flag for PHQ-9 or EPDS."""
    # Normalize keys by lowercasing and stripping prefixes like 'epds', 'phq', 'q', '_'
    clean_answers: Dict[str, int] = {}
    for k, v in answers.items():
        clean_k = str(k).lower().replace("epds", "").replace("phq", "").replace("_", "").replace("-", "").strip()
        if clean_k.startswith("q"):
            clean_k = clean_k[1:].strip()
        clean_answers[clean_k] = int(v) if v is not None else 0

    total_score = sum(int(v) for v in answers.values())
    safety_triggered = False

    if instrument == "phq9":
        # Check PHQ-9 Q9 for self-harm thoughts (key '9' if 1-indexed, or '8' if 0-indexed)
        if "9" in clean_answers:
            q9_val = clean_answers["9"]
        elif "8" in clean_answers and "0" in clean_answers:
            q9_val = clean_answers["8"]
        else:
            q9_val = clean_answers.get("9", 0)
        q9_score = int(q9_val or 0)
        if q9_score > 0:
            safety_triggered = True

        if total_score >= 20:
            band_id = "severe"
            escalation = "see_clinician_urgent"
        elif total_score >= 15:
            band_id = "moderately_severe"
            escalation = "see_clinician_soon"
        elif total_score >= 10:
            band_id = "moderate"
            escalation = "see_clinician"
        elif total_score >= 5:
            band_id = "mild"
            escalation = "monitor"
        else:
            band_id = "minimal"
            escalation = "routine"

    elif instrument == "pbq":
        # Postpartum Bonding Questionnaire: 25 items (1 to 25).
        # Forward items: 1, 4, 8, 9, 11, 16, 22, 25 (0 to 5)
        # Reverse items: all others (5 - val)
        FORWARD_ITEMS = {1, 4, 8, 9, 11, 16, 22, 25}
        total_score = 0
        safety_triggered = False

        sub1_items = {1, 6, 7, 8, 9, 10, 12, 13, 15, 16, 19, 21}
        sub2_items = {2, 3, 4, 11, 17, 20, 23}
        sub3_items = {5, 14, 22, 25}
        sub1_score = 0
        sub2_score = 0
        sub3_score = 0

        for k, v in clean_answers.items():
            try:
                item_num = int(k)
            except ValueError:
                continue
            val = int(v) if v is not None else 0
            if item_num < 1:
                item_num = item_num + 1

            item_score = val if item_num in FORWARD_ITEMS else max(0, 5 - val)
            total_score += item_score

            if item_num in sub1_items:
                sub1_score += item_score
            if item_num in sub2_items:
                sub2_score += item_score
            if item_num in sub3_items:
                sub3_score += item_score

            # Safety override: Item 18 or 24 > 0 (State P0)
            if item_num in (18, 24) and item_score > 0:
                safety_triggered = True

        if safety_triggered:
            band_id = "pbq_p0"
            escalation = "see_clinician_urgent"
        elif total_score >= 40 or sub2_score >= 17:
            band_id = "pbq_p3"
            escalation = "see_clinician_soon"
        elif total_score >= 12 or sub1_score >= 12 or sub3_score >= 10:
            band_id = "pbq_p2"
            escalation = "see_clinician"
        else:
            band_id = "pbq_p1"
            escalation = "routine"

    else:  # epds
        # Check EPDS Q10 for self-harm thoughts (key '10' if 1-indexed, or '9' if 0-indexed)
        if "10" in clean_answers:
            q10_val = clean_answers["10"]
        elif "9" in clean_answers and "0" in clean_answers:
            q10_val = clean_answers["9"]
        else:
            q10_val = clean_answers.get("10", 0)
        q10_score = int(q10_val or 0)
        # EPDS item 10: Score >= 2 ("Sometimes" or "Yes, quite often") triggers active safety override
        if q10_score >= 2:
            safety_triggered = True

        # EPDS 4 finalized score bands from clinical guidance: 0-9, 10-12, 13-19, 20-30
        if total_score >= 20:
            band_id = "severe"
            escalation = "see_clinician_urgent"
        elif total_score >= 13:
            band_id = "moderately_severe"
            escalation = "see_clinician"
        elif total_score >= 10:
            band_id = "moderate"
            escalation = "see_clinician"
        else:
            band_id = "minimal"
            escalation = "routine"

    if safety_triggered:
        escalation = "see_clinician_urgent"

    return {
        "total_score": total_score,
        "band_id": band_id,
        "escalation": escalation,
        "safety_triggered": safety_triggered,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/score", response_model=ScoreResponse, summary="Score questionnaire and return deterministic result")
@limiter.limit(GUIDANCE_RATE_LIMIT, key_func=get_user_rate_limit_key)
async def score_questionnaire(
    request: Request,
    body: ScoreRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Score PHQ-9 or EPDS, evaluate safety, save result row, and return guidance."""
    scored = compute_instrument_score(body.instrument, body.answers)
    band_id = scored["band_id"]
    escalation = scored["escalation"]
    safety_triggered = scored["safety_triggered"]
    total_score = scored["total_score"]

    if body.instrument == "pbq":
        pbq_messages = {
            "pbq_p0": {
                "en": "Your safety and your baby's safety come first. Please reach out to someone who can help today.",
                "hi": "आपकी और आपके शिशु की सुरक्षा सबसे पहले है। कृपया आज ही किसी ऐसे व्यक्ति से संपर्क करें जो मदद कर सके।",
                "bn": "আপনার এবং আপনার শিশুর সুরক্ষা সবচেয়ে আগে। অনুগ্রহ করে আজই এমন কারো সাথে যোগাযোগ করুন যিনি সাহায্য করতে পারেন।",
                "ta": "உங்கள் மற்றும் உங்கள் குழந்தையின் பாதுகாப்பே முதன்மையானது. தயவுசெய்து இன்றே உதவக்கூடிய ஒருவரைத் தொடர்பு கொள்ளுங்கள்.",
            },
            "pbq_p1": {
                "en": "Your answers suggest bonding is developing within the usual range for this stage.",
                "hi": "आपके उत्तर बताते हैं कि शिशु के साथ जुड़ाव इस चरण के लिए सामान्य सीमा में विकसित हो रहा है।",
                "bn": "আপনার উত্তরগুলি থেকে বোঝা যায় যে শিশুর সাথে সংযোগ এই পর্যায়ের জন্য স্বাভাবিক সীমার মধ্যে তৈরি হচ্ছে।",
                "ta": "உங்கள் பதில்கள் இந்த கட்டத்திற்கான வழக்கமான வரம்பிற்குள் குழந்தையின் பிணைப்பு உருவாகி வருவதைக் காட்டுகின்றன.",
            },
            "pbq_p2": {
                "en": "Your answers suggest you may be feeling some strain in your relationship with your baby right now. Many mothers find this stage hard.",
                "hi": "आपके उत्तर बताते हैं कि आप इस समय अपने शिशु के साथ रिश्ते में कुछ तनाव महसूस कर रही होंगी। कई माताओं को यह चरण कठिन लगता है।",
                "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে আপনি হয়তো এই মুহূর্তে আপনার শিশুর সাথে সম্পর্কের ক্ষেত্রে কিছু চাপ অনুভব করছেন। অনেক মায়ের কাছেই এই পর্যায়টি কঠিন মনে হয়।",
                "ta": "உங்கள் பதில்கள் உங்கள் குழந்தையுடனான உறவில் இப்போது சில சிரமங்களை நீங்கள் உணரக்கூடும் என்பதைக் காட்டுகின்றன. பல தாய்மார்கள் இந்த கட்டத்தை கடினமாக உணர்கிறார்கள்.",
            },
            "pbq_p3": {
                "en": "Your answers show that this relationship feels especially difficult right now. Support from a healthcare professional can help you and your baby find your way.",
                "hi": "आपके उत्तर बताते हैं कि यह रिश्ता इस समय विशेष रूप से कठिन लग रहा है। एक स्वास्थ्य देखभाल पेशेवर की मदद आपको और आपके शिशु को सही रास्ता खोजने में मदद कर सकती है।",
                "bn": "আপনার উত্তরগুলি দেখায় যে এই সম্পর্কটি এই মুহূর্তে বিশেষভাবে কঠিন মনে হচ্ছে। একজন স্বাস্থ্যসেবা বিশেষজ্ঞের সহায়তা আপনাকে এবং আপনার শিশুকে সঠিক পথ খুঁজে পেতে সহায়তা করতে পারে।",
                "ta": "இந்த உறவு இப்போது மிகவும் கடினமாக இருப்பதாக உங்கள் பதில்கள் காட்டுகின்றன. ஒரு சுகாதார நிபுணரின் ஆதரவு உங்களுக்கும் உங்கள் குழந்தைக்கும் உதவக்கூடும்.",
            },
        }

        pbq_suggestions_dict = {
            "pbq_p0": [],
            "pbq_p1": [
                {
                    "id": "PBQ_P1_SKIN_TO_SKIN",
                    "category": "bonding",
                    "text": {
                        "en": "Keep up skin-to-skin and calm face-to-face time.",
                        "hi": "त्वचा से त्वचा का संपर्क और शांत आमने-सामने का समय बनाए रखें।",
                        "bn": "ত্বকের সাথে ত্বকের স্পর্শ এবং শান্ত মুখোমুখি সময় বজায় রাখুন।",
                        "ta": "தோலோடு தோல் தொடுதல் மற்றும் அமைதியான முகத்திற்கு முகம் பார்க்கும் நேரத்தைத் தொடருங்கள்.",
                    },
                },
                {
                    "id": "PBQ_P1_SHARE_LOAD",
                    "category": "support",
                    "text": {
                        "en": "Share the load with a partner or family member.",
                        "hi": "साथी या परिवार के सदस्य के साथ जिम्मेदारियों को साझा करें।",
                        "bn": "সঙ্গী বা পরিবারের সদস্যের সাথে কাজের চাপ ভাগ করে নিন।",
                        "ta": "துணைவர் அல்லது குடும்ப உறுப்பினருடன் வேலைப் பளுவைப் பகிர்ந்து கொள்ளுங்கள்.",
                    },
                },
            ],
            "pbq_p2": [
                {
                    "id": "PBQ_P2_HEALTH_WORKER",
                    "category": "clinical",
                    "text": {
                        "en": "Talk to your health worker or doctor about how you feel with the baby.",
                        "hi": "शिशु के साथ आपको कैसा महसूस हो रहा है, इस बारे में अपने स्वास्थ्य कार्यकर्ता या डॉक्टर से बात करें।",
                        "bn": "শিশুর সাথে আপনার কেমন অনুভূতি হচ্ছে সে সম্পর্কে আপনার স্বাস্থ্যকর্মী বা ডাক্তারের সাথে কথা বলুন।",
                        "ta": "குழந்தையுடன் நீங்கள் எப்படி உணர்கிறீர்கள் என்பதைப் பற்றி உங்கள் சுகாதாரப் பணியாளர் அல்லது மருத்துவரிடம் பேசுங்கள்.",
                    },
                },
                {
                    "id": "PBQ_P2_CHECK_MOOD",
                    "category": "wellbeing",
                    "text": {
                        "en": "Check your mood too.",
                        "hi": "अपने मूड की भी जाँच करें।",
                        "bn": "আপনার মেজাজ বা মনের অবস্থাও লক্ষ্য করুন।",
                        "ta": "உங்கள் மனநிலையையும் கவனித்துக் கொள்ளுங்கள்.",
                    },
                },
                {
                    "id": "PBQ_P2_FEEDING_HELP",
                    "category": "feeding",
                    "text": {
                        "en": "Ask about help with feeding if it is a struggle.",
                        "hi": "यदि स्तनपान या दूध पिलाने में कठिनाई हो रही है, तो सहायता के लिए पूछें।",
                        "bn": "যদি দুধ খাওয়ানো কঠিন মনে হয় তবে সহায়তার জন্য জিজ্ঞাসা করুন।",
                        "ta": "பாலூட்டுவது சிரமமாக இருந்தால் உதவி கேட்கவும்.",
                    },
                },
                {
                    "id": "PBQ_P2_PROTECT_SLEEP",
                    "category": "sleep",
                    "text": {
                        "en": "Protect a stretch of sleep (a partner takes one night feed).",
                        "hi": "लगातार कुछ घंटों की नींद सुरक्षित करें (रात में एक बार दूध पिलाने का काम साथी संभाल सकते हैं)।",
                        "bn": "টানা ঘুমের সময় সুরক্ষিত করুন (সঙ্গী রাতে একবার দুধ খাওয়ানোর দায়িত্ব নিতে পারেন)।",
                        "ta": "தொடர்ச்சியான தூக்கத்தைப் பாதுகாத்துக் கொள்ளுங்கள் (இரவில் ஒருமுறை பாலூட்டுவதை துணைவர் பார்த்துக் கொள்ளலாம்).",
                    },
                },
                {
                    "id": "PBQ_P2_SING_TALK",
                    "category": "bonding",
                    "text": {
                        "en": "Sing or talk to your baby during a calm moment.",
                        "hi": "शांत पलों में अपने शिशु को गाना गाकर सुनाएं या उससे बातें करें।",
                        "bn": "শান্ত মুহূর্তে আপনার শিশুর সাথে গান গান বা কথা বলুন।",
                        "ta": "அமைதியான தருணங்களில் உங்கள் குழந்தையிடம் பாடுங்கள் அல்லது பேசுங்கள்.",
                    },
                },
            ],
            "pbq_p3": [
                {
                    "id": "PBQ_P3_INTERVIEW",
                    "category": "clinical",
                    "text": {
                        "en": "See a professional for an interview about the relationship, anxiety, anger and obsessive thoughts.",
                        "hi": "रिश्ते, चिंता, गुस्से और अनचाहे विचारों के बारे में परामर्श के लिए किसी पेशेवर से मिलें।",
                        "bn": "সম্পর্ক, উদ্বেগ, রাগ এবং অতিরিক্ত চিন্তার বিষয়ে মূল্যায়নের জন্য একজন পেশাদারের সাথে দেখা করুন।",
                        "ta": "உறவு, பதட்டம், கோபம் மற்றும் தொடர் எண்ணங்கள் பற்றிய நேர்காணலுக்கு ஒரு தொழில்முறை நிபுணரை அணுகவும்.",
                    },
                },
                {
                    "id": "PBQ_P3_SESSIONS",
                    "category": "bonding",
                    "text": {
                        "en": "Ask about mother-and-baby sessions, where a trained person watches you play with your baby and gives feedback.",
                        "hi": "माता-शिशु सत्रों के बारे में पूछें, जहाँ एक प्रशिक्षित व्यक्ति शिशु के साथ आपके जुड़ाव को देखकर मार्गदर्शन देता है।",
                        "bn": "মা-শিশুর সেশন সম্পর্কে খোঁজ নিন, যেখানে একজন প্রশিক্ষিত ব্যক্তি শিশুর সাথে খেলা লক্ষ্য করে পরামর্শ দেন।",
                        "ta": "தாய்-சேய் அமர்வுகள் பற்றி கேளுங்கள், அங்கு பயிற்சியாளர் நீங்கள் குழந்தையுடன் பழகுவதைப் பார்த்து வழிகாட்டுவார்.",
                    },
                },
                {
                    "id": "PBQ_P3_ANXIETY",
                    "category": "clinical",
                    "text": {
                        "en": "Get support for low mood or anxiety.",
                        "hi": "उदास मन या चिंता के लिए पेशेवर सहायता प्राप्त करें।",
                        "bn": "মন খারাপ বা উদ্বেগের জন্য সহায়তা নিন।",
                        "ta": "மனச்சோர்வு அல்லது பதட்டத்திற்கு ஆதரவு பெறவும்.",
                    },
                },
                {
                    "id": "PBQ_P3_FAMILY_REST",
                    "category": "support",
                    "text": {
                        "en": "Ask whether a family member can take over some care so you can rest.",
                        "hi": "परिवार के किसी सदस्य से शिशु की देखभाल में मदद करने को कहें ताकि आप आराम कर सकें।",
                        "bn": "পরিবারের কোনো সদস্যকে শিশুর কিছু যত্ন নেওয়ার অনুরোধ করুন যাতে আপনি বিশ্রাম নিতে পারেন।",
                        "ta": "நீங்கள் ஓய்வெடுக்க ஏதுவாக குடும்ப உறுப்பினரிடம் குழந்தையின் சில பராமரிப்புகளைப் பார்த்துக் கொள்ளச் சொல்லுங்கள்.",
                    },
                },
            ],
        }

        band_msg_map = pbq_messages.get(band_id, pbq_messages["pbq_p1"])
        band_message = band_msg_map.get(body.language, band_msg_map.get("en", ""))
        raw_pbq_sugs = pbq_suggestions_dict.get(band_id, [])
        ordered = [
            {
                "id": s["id"],
                "category": s["category"],
                "text": s["text"].get(body.language, s["text"].get("en", "")),
            }
            for s in raw_pbq_sugs
        ]
    else:
        band_message = get_band_message(band_id, body.language)
        candidates = build_candidates(band_id=band_id, stage=body.stage, escalation=escalation)
        ordered = order_deterministic(candidates=candidates, escalation=escalation, language=body.language)

        # Translate any suggestions that were fallback in English using Gemini AI
        if body.language in ("hi", "bn", "ta"):
            for sug in ordered:
                txt = sug.get("text", "")
                # If the text is purely ASCII English, translate via Gemini AI
                if txt and all(ord(ch) < 128 for ch in txt):
                    sug["text"] = await translate_text_with_gemini(txt, body.language)

    result_payload = {
        "band_id": band_id,
        "band_message": band_message,
        "escalation": escalation,
        "suggestions": ordered,
        "safety_triggered": safety_triggered,
        "helplines": get_verified_helplines(),
        "safe_fallback_text": get_safe_fallback_text(body.language),
        "disclaimer_key": "sentiment.disclaimer",
        "previous": None,
        "personalised": False,
        "mode": "none",
        "library_version": get_library_version_hash(),
    }

    result_id = save_questionnaire_result(
        user_ref=user_ref,
        instrument=body.instrument,
        total_score=total_score,
        band_id=band_id,
        escalation=escalation,
        safety_triggered=safety_triggered,
        stage=body.stage,
        answers=body.answers,
        result_dict=result_payload,
    )

    return ScoreResponse(
        result_id=result_id,
        instrument=body.instrument,
        total_score=total_score,
        **result_payload,
    )


@router.post("/guidance", response_model=GuidanceResponse, summary="Optional AI personalisation layer on questionnaire guidance")
@limiter.limit(GUIDANCE_RATE_LIMIT, key_func=get_user_rate_limit_key)
async def get_questionnaire_guidance(
    request: Request,
    body: GuidanceRequest,
    user_ref: str = Depends(get_current_user_ref),
):
    """Retrieve personalised or deterministic guidance for a stored questionnaire result.

    Security & Privacy Rules:
    - Never trust client for band, score, or safety flag.
    - Verified strictly against stored questionnaire_results owned by user_ref (404 otherwise).
    - Model never sees user free text, names, emails, ids or scores.
    """
    clean_themes = [t for t in (body.themes or []) if t in GUIDANCE_FIXED_THEMES]

    guidance_res = await run_guidance_workflow(
        result_id=body.result_id,
        user_ref=user_ref,
        language=body.language,
        themes=clean_themes,
    )

    if guidance_res.get("_error") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Questionnaire result not found or does not belong to the authenticated user.",
        )

    return GuidanceResponse(**guidance_res)
