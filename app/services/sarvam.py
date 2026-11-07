"""Sarvam AI Translation Service for Indic Languages (Hindi, Bengali, Tamil).

Provides high-quality regional translation for perinatal clinical guidance,
questionnaire band messages, and self-care recommendations using Sarvam AI.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
import httpx

from app.config import SARVAM_API_KEY, LLM_TIMEOUT

logger = logging.getLogger("healher_backend.sarvam")

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Language code mapping to Sarvam BCP-47 codes
LANGUAGE_MAP: Dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
}

# Pre-validated clinical translations for questionnaire band messages
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
        "bn": "আপনার উত্তরগুলি ইঙ্গিত করে যে কিছু বিষয় আপনার মনে চাপ ফেলছে। নিজের প্রতি যত্নবান হোন এবং লক্ষ্য রাখুন।",
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

# In-memory translation cache to avoid duplicate API calls
_translation_cache: Dict[str, str] = {}


def get_sarvam_api_key() -> str:
    """Retrieve Sarvam API Key from config or environment."""
    return os.environ.get("SARVAM_API_KEY", SARVAM_API_KEY).strip()


def get_band_message(band_id: str, language: str = "en") -> str:
    """Return pre-validated clinical band message for the given band and language."""
    band_clean = band_id.lower()
    band_dict = BAND_MESSAGES_TRANSLATED.get(band_clean, BAND_MESSAGES_TRANSLATED.get("well", {}))
    return band_dict.get(language, band_dict.get("en", "Take gentle care of yourself."))


async def translate_with_sarvam(
    text: str,
    target_lang: str,
    source_lang: str = "en",
    timeout_sec: float = 8.0,
) -> str:
    """Translate single string into Indic language (Hindi, Bengali, Tamil).

    Uses Google Gemini AI as the primary engine for Bengali (bn) and Tamil (ta),
    and as an automatic fallback whenever Sarvam API fails.
    """
    if not text or target_lang == source_lang or target_lang == "en":
        return text

    cache_key = f"{source_lang}:{target_lang}:{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    # Check pre-validated clinical band dictionary first
    clean_text = text.strip()
    for band_key, translations in BAND_MESSAGES_TRANSLATED.items():
        if translations.get(source_lang, "").strip() == clean_text:
            matched = translations.get(target_lang)
            if matched:
                _translation_cache[cache_key] = matched
                return matched

    # For Bengali (bn) and Tamil (ta), use Gemini AI directly
    if target_lang in ("bn", "ta"):
        try:
            from app.services.llm import translate_text_with_gemini
            gemini_res = await translate_text_with_gemini(text, target_lang, source_lang)
            if gemini_res and gemini_res != text:
                _translation_cache[cache_key] = gemini_res
                return gemini_res
        except Exception as exc:
            logger.warning("Gemini translation error for %s: %s", target_lang, type(exc).__name__)

    target_code = LANGUAGE_MAP.get(target_lang)
    if not target_code:
        logger.debug("Unsupported target language for Sarvam: %s. Trying Gemini.", target_lang)
        try:
            from app.services.llm import translate_text_with_gemini
            return await translate_text_with_gemini(text, target_lang, source_lang)
        except Exception:
            return text

    api_key = get_sarvam_api_key()
    if not api_key or api_key in ("your_sarvam_api_key_here", "paste_your_key_here"):
        logger.debug("SARVAM_API_KEY not configured. Falling back to Gemini.")
        try:
            from app.services.llm import translate_text_with_gemini
            return await translate_text_with_gemini(text, target_lang, source_lang)
        except Exception:
            return text

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "input": text,
        "source_language_code": LANGUAGE_MAP.get(source_lang, "en-IN"),
        "target_language_code": target_code,
        "mode": "formal",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(SARVAM_TRANSLATE_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get("translated_text", "").strip()
                if translated:
                    _translation_cache[cache_key] = translated
                    return translated
            else:
                logger.warning(
                    "Sarvam API translation error: status=%d text=%s. Falling back to Gemini.",
                    resp.status_code,
                    resp.text[:100],
                )
    except Exception as exc:
        logger.warning("Sarvam API translation call failed: %s. Falling back to Gemini.", type(exc).__name__)

    # Fallback to Gemini on Sarvam failure
    try:
        from app.services.llm import translate_text_with_gemini
        return await translate_text_with_gemini(text, target_lang, source_lang)
    except Exception:
        return text


async def translate_batch_with_sarvam(
    texts: List[str],
    target_lang: str,
    source_lang: str = "en",
) -> List[str]:
    """Translate multiple texts concurrently into Indic languages.

    For Bengali (bn) and Tamil (ta), delegates to Gemini batch translation.
    """
    if not texts or target_lang == source_lang or target_lang == "en":
        return list(texts)

    # For Bengali and Tamil, prioritize Gemini batch translation
    if target_lang in ("bn", "ta"):
        try:
            from app.services.llm import translate_batch_with_llm
            items = [{"key": f"item_{i}", "text": t} for i, t in enumerate(texts)]
            res = await translate_batch_with_llm(items, target_lang)
            if res:
                out = []
                for i, t in enumerate(texts):
                    val = res.get(f"item_{i}")
                    out.append(val if val and val.strip() else t)
                if any(o != texts[i] for i, o in enumerate(out)):
                    return out
        except Exception as exc:
            logger.warning("Gemini batch translation error for %s: %s", target_lang, type(exc).__name__)

    tasks = [translate_with_sarvam(t, target_lang, source_lang) for t in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if isinstance(r, str) and r else texts[i]
        for i, r in enumerate(results)
    ]


async def synthesize_speech_with_sarvam(
    text: str,
    language: str = "hi",
    speaker: str = "ritu",
    pace: float = 1.0,
    timeout_sec: float = 15.0,
) -> Optional[str]:
    """Synthesize text to speech using Sarvam AI Bulbul TTS model (v3).

    Supports en-IN, hi-IN, bn-IN, ta-IN with natural Indic speakers.
    Returns base64 encoded audio string (WAV format) or None if synthesis fails.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return None

    api_key = get_sarvam_api_key()
    if not api_key or api_key in ("your_sarvam_api_key_here", "paste_your_key_here"):
        logger.debug("SARVAM_API_KEY not configured for TTS.")
        return None

    target_code = LANGUAGE_MAP.get(language, "en-IN")
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    # Map legacy/deprecated speakers to valid bulbul:v3 speaker (e.g. ritu)
    valid_speaker = "ritu" if not speaker or speaker.lower() in ("meera", "default") else speaker.lower()
    payload = {
        "inputs": [clean_text[:1000]],
        "target_language_code": target_code,
        "speaker": valid_speaker,
        "pitch": 0,
        "pace": max(0.5, min(2.0, float(pace))),
        "loudness": 1.0,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
        "model": "bulbul:v3",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(SARVAM_TTS_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                audios = data.get("audios", [])
                if audios and isinstance(audios, list) and audios[0]:
                    return audios[0]
            else:
                logger.warning(
                    "Sarvam TTS failed: status=%d text=%s",
                    resp.status_code,
                    resp.text[:150],
                )
    except Exception as exc:
        logger.warning("Sarvam TTS request failed: %s", type(exc).__name__)

    return None

