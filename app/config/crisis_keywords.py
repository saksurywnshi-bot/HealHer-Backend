"""Crisis Keywords Configuration for Maternal Mental Health Safety.

IMPORTANT CLINICAL NOTICE:
This file contains safety-critical screening keywords for self-harm, suicidal ideation,
and infanticide risk.
TODO(Clinician Review): A licensed clinical psychologist / psychiatrist must review,
refine, and clinically approve all keywords in each supported language prior to production.

Rules:
- All phrases are lower-case substrings matched against normalized user input.
- Any match on notes of ANY length must trigger an immediate crisis escalation
  ({status: 'crisis', support: true}) without sending data to any external LLM.
"""

from typing import Dict, List

# TODO(Clinician): Review and expand English crisis keywords
EN_CRISIS_KEYWORDS: List[str] = [
    # Suicidal ideation & self-harm
    "kill myself",
    "end my life",
    "want to die",
    "wish i were dead",
    "wish i was dead",
    "better off dead",
    "suicide",
    "suicidal",
    "take my own life",
    "hang myself",
    "cut myself",
    "slit my wrist",
    "overdose",
    "no reason to live",
    "can't go on living",
    "cant go on living",
    "don't want to wake up",
    "dont want to wake up",
    # Harm to baby / infanticide ideation
    "hurt the baby",
    "hurt my baby",
    "harm the baby",
    "harm my baby",
    "kill the baby",
    "kill my baby",
    "throw the baby",
    "shake the baby",
    "drown the baby",
    "suffocate the baby",
]

# TODO(Clinician): Review and expand Hindi crisis keywords
HI_CRISIS_KEYWORDS: List[str] = [
    # Suicidal ideation & self-harm
    "खुद को मारना",
    "खुद को खत्म",
    "मरना चाहती हूं",
    "मरना चाहती हूँ",
    "जीना नहीं चाहती",
    "मौत चाहती हूं",
    "आत्महत्या",
    "जान दे दूंगी",
    "जान देना चाहती",
    "खुदकुशी",
    "नुकसान पहुंचाना",
    # Harm to baby
    "बच्चे को मार",
    "बच्चे को नुकसान",
    "बच्चे को फेंक",
    "बच्चे को चोट",
]

# TODO(Clinician): Review and expand Bengali crisis keywords
BN_CRISIS_KEYWORDS: List[str] = [
    # Suicidal ideation & self-harm
    "নিজেকে শেষ",
    "নিজেকে মেরে",
    "মরতে চাই",
    "বাঁচতে চাই না",
    "আত্মহত্যা",
    "মরে যেতে চাই",
    "জীবন শেষ",
    # Harm to baby
    "শিশুকে আঘাত",
    "শিশুকে মেরে",
    "বাচ্চাকে মেরে",
    "বাচ্চাকে ক্ষতি",
]

# TODO(Clinician): Review and expand Tamil crisis keywords
TA_CRISIS_KEYWORDS: List[str] = [
    # Suicidal ideation & self-harm
    "தற்கொலை",
    "சாக வேண்டும்",
    "உயிரை விட",
    "வாழ பிடிக்கவில்லை",
    "சாக தோன்றுகிறது",
    "என்னை நானே அழித்து",
    # Harm to baby
    "குழந்தையைக் காயப்படுத்த",
    "குழந்தையை கொல்ல",
    "குழந்தைக்கு தீங்கு",
]

CRISIS_KEYWORDS_BY_LANG: Dict[str, List[str]] = {
    "en": EN_CRISIS_KEYWORDS,
    "hi": HI_CRISIS_KEYWORDS,
    "bn": BN_CRISIS_KEYWORDS,
    "ta": TA_CRISIS_KEYWORDS,
}


def contains_crisis_keywords(text: str, language: str = "en") -> bool:
    """Check if the given note text matches any known crisis keywords.

    Checks both the target language keyword list and the English list
    (to handle bilingual/code-switched input).
    """
    if not text:
        return False

    normalized = text.lower()

    # Always check the specified language
    target_keywords = CRISIS_KEYWORDS_BY_LANG.get(language, [])
    for kw in target_keywords:
        if kw.lower() in normalized:
            return True

    # If the language was not English, also check English (code-switching safety)
    if language != "en":
        for kw in EN_CRISIS_KEYWORDS:
            if kw.lower() in normalized:
                return True

    return False
