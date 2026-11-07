"""Gemini LLM Service using langchain-google-genai.

Privacy Rules:
- Third-party tracing is disabled.
- Prompts, response bodies, and API keys are NEVER logged.
- Model name configured via GEMINI_MODEL in app.config.
- 10-second timeout enforced.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import GEMINI_MODEL, LLM_TIMEOUT

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: Dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
}

SYSTEM_PROMPT = (
    "Translate for a mental health app for pregnant and postpartum women in India. "
    "Warm, simple, respectful everyday language, no clinical jargon. "
    "Keep placeholders exactly as they are ({{name}}, {{week}}, %s). "
    "Do not translate: HealHer, BEAM, Tele-MANAS, phone numbers. "
    "Return ONLY JSON mapping each key to its translation."
)

STRICT_PLACEHOLDER_INSTRUCTION = (
    "\n\nCRITICAL RETRY REQUIREMENT: The previous attempt failed placeholder validation. "
    "You MUST preserve every placeholder (e.g. {{name}}, {{week}}, %s, {0}) EXACTLY "
    "as it appears in the source text, character-for-character, without translating, "
    "reformatting, or dropping them."
)


def get_llm(timeout: float = LLM_TIMEOUT) -> ChatGoogleGenerativeAI:
    """Instantiate ChatGoogleGenerativeAI with project config and privacy constraints."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        timeout=timeout,
        temperature=0.1,
        max_retries=1,
        response_mime_type="application/json",
    )


def clean_json_response(content: str) -> str:
    """Clean markdown code fences if present in JSON response."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


async def translate_batch_with_llm(
    items: List[Dict[str, str]],
    target_language: str,
    attempt: int = 1,
    llm_instance: Optional[Any] = None,
) -> Dict[str, str]:
    """Call Gemini once for all missing items with structured JSON output.

    Returns a dict mapping key -> translated_text.
    Never logs request/response bodies or keys.
    """
    if not items:
        return {}

    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    system_text = SYSTEM_PROMPT
    if attempt > 1:
        system_text += STRICT_PLACEHOLDER_INSTRUCTION

    # Build input payload: key -> source text
    input_payload = {item["key"]: item["text"] for item in items}
    user_prompt = (
        f"Target Language: {lang_name} ({target_language})\n\n"
        f"Translate the following key-value pairs:\n{json.dumps(input_payload, ensure_ascii=False)}"
    )

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_prompt),
    ]

    try:
        llm = llm_instance or get_llm()
        response = await llm.ainvoke(messages)

        raw_content = response.content
        if isinstance(raw_content, list):
            # If parts list returned
            raw_text = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in raw_content
            )
        else:
            raw_text = str(raw_content)

        cleaned = clean_json_response(raw_text)
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            # Ensure all values are strings
            return {str(k): str(v) for k, v in parsed.items()}
        elif isinstance(parsed, list):
            # Sometimes models return [{"key": ..., "translation": ...}]
            result = {}
            for entry in parsed:
                if isinstance(entry, dict) and "key" in entry:
                    val = entry.get("translation") or entry.get("text") or ""
                    result[str(entry["key"])] = str(val)
            return result
        else:
            logger.warning("Gemini returned non-dict/list JSON structure.")
            return {}

    except Exception as exc:
        # Never log prompts, response bodies, or keys
        logger.error("LLM translation failed with exception: %s", type(exc).__name__)
        return {}


# ---------------------------------------------------------------------------
# Multilingual Clinical Band Messages (Hindi, Bengali, Tamil, English)
# ---------------------------------------------------------------------------
BAND_MESSAGES_TRANSLATED: Dict[str, Dict[str, str]] = {
    "minimal": {
        "en": "Your answers suggest you are doing okay right now.",
        "hi": "आपके उत्तरों से लगता है कि आप अभी ठीक हैं।",
        "bn": "আপনার উত্তরগুলি থেকে বোঝা যায় যে আপনি এখন ঠিক আছেন।",
        "ta": "உங்கள் பதில்கள் நீங்கள் தற்போது நலமாக இருக்கிறீர்கள் என்பதைக் காட்டுகின்றன.",
    },
    "well": {
        "en": "Your answers suggest you are doing okay right now.",
        "hi": "आपके उत्तरों से लगता है कि आप अभी ठीक हैं।",
        "bn": "আপনার উত্তরগুলি থেকে বোঝা যায় যে আপনি এখন ঠিক আছেন।",
        "ta": "உங்கள் பதில்கள் நீங்கள் தற்போது நலமாக இருக்கிறீர்கள் என்பதைக் காட்டுகின்றன.",
    },
    "mild": {
        "en": "Your answers suggest some things are weighing on you. Be gentle with yourself and keep checking in.",
        "hi": "आपके उत्तर बताते हैं कि कुछ बातें आप पर भारी पड़ रही हैं। अपने प्रति दयालु रहें और अपनी देखभाल करते रहें।",
        "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে কিছু বিষয় আপনার মনে চাপ ফেলছে। নিজের প্রতি যত্নবান হোন এবং লক্ষ্য রাখুন।",
        "ta": "சில விஷயங்கள் உங்களுக்கு சுமையாக இருப்பதை உங்கள் பதில்கள் காட்டுகின்றன. உங்களை கனிவாக கவனித்துக் கொள்ளுங்கள்.",
    },
    "possible_mild_depression": {
        "en": "Your answers suggest some things are weighing on you. Be gentle with yourself and keep checking in.",
        "hi": "आपके उत्तर बताते हैं कि कुछ बातें आप पर भारी पड़ रही हैं। अपने प्रति दयालु रहें और अपनी देखभाल करते रहें।",
        "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে कुछ বিষয় আপনার মনে চাপ ফেলছে। নিজের প্রতি যত্নবান হোন এবং লক্ষ্য রাখুন।",
        "ta": "சில விஷயங்கள் உங்களுக்கு சுமையாக இருப்பதை உங்கள் பதில்கள் காட்டுகின்றன. உங்களை கனிவாக கவனித்துக் கொள்ளுங்கள்.",
    },
    "moderate": {
        "en": "Your answers suggest you may be carrying a lot right now. Reaching out to a doctor, midwife or health worker can help.",
        "hi": "आपके उत्तर बताते हैं कि आप इस समय काफी मानसिक तनाव से गुजर रही हैं। डॉक्टर, मिडवाइफ या स्वास्थ्य कार्यकर्ता से बात करने से मदद मिल सकती है।",
        "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে আপনি হয়তো অনেক মানসিক চাপের মধ্যে আছেন। ডাক্তার বা স্বাস্থ্যকর্মীর সাথে কথা বললে উপকার হতে পারে।",
        "ta": "நீங்கள் தற்போது அதிக மன அழுத்தத்தை சுமந்து கொண்டிருக்கலாம் என்பதை உங்கள் பதில்கள் காட்டுகின்றன. மருத்துவரிடம் பேசுவது உதவும்.",
    },
    "possible_depression": {
        "en": "Your answers suggest you may be carrying a lot right now. Reaching out to a doctor, midwife or health worker can help.",
        "hi": "आपके उत्तर बताते हैं कि आप इस समय काफी मानसिक तनाव से गुजर रही हैं। डॉक्टर, मिडवाइफ या स्वास्थ्य कार्यकर्ता से बात करने से मदद मिल सकती है।",
        "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে আপনি হয়তো অনেক মানসিক চাপের মধ্যে আছেন। ডাক্তার বা স্বাস্থ্যকর্মীর সাথে কথা বললে উপকার হতে পারে।",
        "ta": "நீங்கள் தற்போது அதிக மன அழுத்தத்தை சுமந்து கொண்டிருக்கலாம் என்பதை உங்கள் பதில்கள் காட்டுகின்றன. மருத்துவரிடம் பேசுவது உதவும்.",
    },
    "moderately_severe": {
        "en": "Your answers suggest you are going through a difficult time. We strongly recommend speaking with a health professional.",
        "hi": "आपके उत्तर बताते हैं कि आप एक कठिन समय से गुजर रही हैं। हम स्वास्थ्य विशेषज्ञ से परामर्श करने की दृढ़ता से सलाह देते हैं।",
        "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে আপনি একটি কঠিন সময়ের মধ্য দিয়ে যাচ্ছেন। আমরা একজন স্বাস্থ্য বিশেষজ্ঞের সাথে কথা বলার জোরালো পরামর্শ দিচ্ছি।",
        "ta": "நீங்கள் ஒரு கடினமான காலகட்டத்தை கடந்து கொண்டிருப்பதை உங்கள் பதில்கள் காட்டுகின்றன. சுகாதார நிபுணருடன் பேசுமாறு பரிந்துரைக்கிறோம்.",
    },
    "severe": {
        "en": "Your answers indicate heavy distress. Please reach out to your doctor, health worker or free helpline today.",
        "hi": "आपके उत्तर भारी परेशानी का संकेत देते हैं। कृपया आज ही अपने डॉक्टर, स्वास्थ्य कार्यकर्ता या मुफ्त हेल्पलाइन से संपर्क करें।",
        "bn": "আপনার উত্তরগুলি তীব্র মানসিক কষ্টের ইঙ্গিত দেয়। অনুগ্রহ করে আজই আপনার ডাক্তার, স্বাস্থ্যকর্মী বা বিনামূল্যে হেল্পলাইনে যোগাযোগ করুন।",
        "ta": "உங்கள் பதில்கள் கடுமையான மன உளைச்சலைக் குறிக்கின்றன. தயவுசெய்து இன்றே உங்கள் மருத்துவர் அல்லது இலவச உதவி மையத்தைத் தொடர்பு கொள்ளுங்கள்.",
    },
}

_gemini_translation_cache: Dict[str, str] = {}


def get_band_message(band_id: str, language: str = "en") -> str:
    """Return clinical band message for the given band and language."""
    band_clean = band_id.lower()
    band_dict = BAND_MESSAGES_TRANSLATED.get(band_clean, BAND_MESSAGES_TRANSLATED.get("well", {}))
    return band_dict.get(language, band_dict.get("en", "Take gentle care of yourself."))


async def translate_text_with_gemini(
    text: str,
    target_language: str,
    source_language: str = "en",
) -> str:
    """Translate single string using normal Gemini translation with caching."""
    if not text or target_language == source_language or target_language == "en":
        return text

    cache_key = f"{source_language}:{target_language}:{text}"
    if cache_key in _gemini_translation_cache:
        return _gemini_translation_cache[cache_key]

    # Pre-validated clinical band dictionary check
    clean_text = text.strip()
    for band_key, translations in BAND_MESSAGES_TRANSLATED.items():
        if translations.get(source_language, "").strip() == clean_text:
            matched = translations.get(target_language)
            if matched:
                _gemini_translation_cache[cache_key] = matched
                return matched

    try:
        items = [{"key": "sug_0", "text": text}]
        result = await translate_batch_with_llm(items, target_language)
        translated = result.get("sug_0", "").strip()

        if translated:
            _gemini_translation_cache[cache_key] = translated
            return translated
    except Exception as exc:
        logger.warning("Gemini translation error: %s", type(exc).__name__)

    return text

