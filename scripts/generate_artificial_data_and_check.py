"""Generate Artificial Dataset for PHQ-9 and EPDS Score Ranges and Test Guidance AI.

This script constructs artificial screening submissions across all clinical score ranges:
- PHQ-9: Minimal (0-4), Mild (5-9), Moderate (10-14), Moderately Severe (15-19), Severe (20-27), and Safety Triggered (Q9 > 0).
- EPDS: Well (0-9), Possible Mild Depression (10-12), Possible Depression (13-30), and Safety Triggered (Q10 > 0).

For each scenario, it:
1. Scores the assessment via POST /api/questionnaire/score.
2. Requests guidance via POST /api/questionnaire/guidance with relevant theme tags.
3. Captures whether AI personalisation was applied (Mode A/B) or safely gated.
4. Validates that clinical safeguards (pinning clinician contact, mixed evidence exclusion, verified helplines) hold.
"""

import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# Ensure AI guidance is enabled for evaluation
os.environ["GUIDANCE_AI_ENABLED"] = "true"
os.environ["GUIDANCE_AI_REWORD"] = "false"

from app.main import app
from app.services.guidance_db import init_guidance_db, clear_guidance_cache
from app.limiter import limiter

client = TestClient(app)

# ---------------------------------------------------------------------------
# Artificial Dataset Definition
# ---------------------------------------------------------------------------
SCENARIOS: List[Dict[str, Any]] = [
    # ---------------- PHQ-9 SCENARIOS ----------------
    {
        "id": "PHQ9_RANGE_1_MINIMAL",
        "instrument": "phq9",
        "name": "PHQ-9 Minimal Distress (Score 0-4)",
        "expected_band": "minimal",
        "expected_escalation": "routine",
        "stage": "postnatal",
        "language": "en",
        "themes": ["sleep", "positive_moments"],
        "answers": {"1": 1, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0},
        "description": "Mother reports occasional minor tiredness, overall doing well. Self-care and sleep routine prioritized.",
    },
    {
        "id": "PHQ9_RANGE_2_MILD",
        "instrument": "phq9",
        "name": "PHQ-9 Mild Distress (Score 5-9)",
        "expected_band": "mild",
        "expected_escalation": "monitor",
        "stage": "postnatal",
        "language": "en",
        "themes": ["sleep", "tiredness", "body_physical_symptoms"],
        "answers": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 0, "9": 0},
        "description": "Mild fatigue and sleep disruption post-delivery. AI orders sleep protection, pacing rest, and self-care first.",
    },
    {
        "id": "PHQ9_RANGE_3_MODERATE",
        "instrument": "phq9",
        "name": "PHQ-9 Moderate Distress (Score 10-14)",
        "expected_band": "moderate",
        "expected_escalation": "see_clinician",
        "stage": "postnatal",
        "language": "en",
        "themes": ["loneliness", "family_support", "self_criticism_guilt"],
        "answers": {"1": 2, "2": 2, "3": 1, "4": 2, "5": 1, "6": 2, "7": 1, "8": 1, "9": 0},
        "description": "Moderate distress. Escalation is see_clinician: Clinician Contact item MUST be pinned at position #0. Calming practice (mixed evidence) excluded.",
    },
    {
        "id": "PHQ9_RANGE_4_MODERATELY_SEVERE",
        "instrument": "phq9",
        "name": "PHQ-9 Moderately Severe Distress (Score 15-19)",
        "expected_band": "moderately_severe",
        "expected_escalation": "see_clinician_soon",
        "stage": "postnatal",
        "language": "en",
        "themes": ["sleep", "baby_health_worry"],
        "answers": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 3, "9": 0},
        "description": "Elevated distress. AI model is strictly GATED (see_clinician_soon). Returns deterministic safety guidance and verified helplines.",
    },
    {
        "id": "PHQ9_RANGE_5_SEVERE",
        "instrument": "phq9",
        "name": "PHQ-9 Severe Distress (Score 20-27)",
        "expected_band": "severe",
        "expected_escalation": "see_clinician_urgent",
        "stage": "postnatal",
        "language": "en",
        "themes": ["sleep", "tiredness"],
        "answers": {"1": 3, "2": 3, "3": 3, "4": 3, "5": 3, "6": 3, "7": 2, "8": 3, "9": 0},
        "description": "Severe distress. AI model is strictly GATED. Urgent professional escalation, safe fallback text, and verified helplines.",
    },
    {
        "id": "PHQ9_RANGE_6_SAFETY_TRIGGERED",
        "instrument": "phq9",
        "name": "PHQ-9 Self-Harm Trigger (Question 9 > 0, Score 3)",
        "expected_band": "minimal",
        "expected_escalation": "see_clinician_urgent",
        "stage": "postnatal",
        "language": "en",
        "themes": ["loneliness"],
        "answers": {"1": 1, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 1},
        "description": "Low total score (3) but Q9 is 1 (self-harm thoughts). Immediately triggers safety gate, AI is NEVER called, Tele-MANAS (14416) displayed.",
    },

    # ---------------- EPDS SCENARIOS ----------------
    {
        "id": "EPDS_RANGE_1_WELL",
        "instrument": "epds",
        "name": "EPDS Well / Normal Emotional Range (Score 0-9)",
        "expected_band": "well",
        "expected_escalation": "routine",
        "stage": "postnatal",
        "language": "hi",
        "themes": ["bonding_worries", "positive_moments"],
        "answers": {"1": 1, "2": 1, "3": 0, "4": 1, "5": 1, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0},
        "description": "Mother adjusting well to postpartum routine. AI personalizes suggestions in Hindi for bonding and emotional balance.",
    },
    {
        "id": "EPDS_RANGE_2_POSSIBLE_MILD",
        "instrument": "epds",
        "name": "EPDS Possible Mild Distress (Score 10-12)",
        "expected_band": "possible_mild_depression",
        "expected_escalation": "monitor",
        "stage": "postnatal",
        "language": "en",
        "themes": ["tiredness", "feeding_difficulty", "sleep"],
        "answers": {"1": 1, "2": 2, "3": 1, "4": 2, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 0},
        "description": "Score 11 above screening cutoff. Supportive monitoring. AI orders rest, sleep routines, and partner/family support.",
    },
    {
        "id": "EPDS_RANGE_3_POSSIBLE_DEPRESSION",
        "instrument": "epds",
        "name": "EPDS Possible Depression (Score 13-30)",
        "expected_band": "possible_depression",
        "expected_escalation": "see_clinician",
        "stage": "postnatal",
        "language": "en",
        "themes": ["loneliness", "family_support", "self_criticism_guilt"],
        "answers": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 1, "8": 2, "9": 1, "10": 0},
        "description": "Score 16. Clinical band possible_depression. Clinician contact suggestion MUST be pinned at position #0. Calming practice excluded.",
    },
    {
        "id": "EPDS_RANGE_4_SAFETY_TRIGGERED",
        "instrument": "epds",
        "name": "EPDS Self-Harm Trigger (Question 10 > 0, Score 4)",
        "expected_band": "well",
        "expected_escalation": "see_clinician_urgent",
        "stage": "postnatal",
        "language": "en",
        "themes": ["sleep"],
        "answers": {"1": 1, "2": 1, "3": 1, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 1},
        "description": "Total score 4, but EPDS Item 10 is 1. AI is strictly GATED. Urgent clinician escalation, safety text, and emergency 112/14416.",
    },
]


def run_evaluation() -> Dict[str, Any]:
    """Execute all artificial test scenarios and compile results."""
    init_guidance_db()
    clear_guidance_cache()
    limiter.reset()

    results = []

    for idx, sc in enumerate(SCENARIOS, 1):
        test_user = f"synthetic_eval_user_{idx}"
        headers = {"x-user-ref": test_user}

        # Step 1: Score the assessment
        score_payload = {
            "instrument": sc["instrument"],
            "answers": sc["answers"],
            "stage": sc["stage"],
            "language": sc["language"],
        }
        score_res = client.post("/api/questionnaire/score", headers=headers, json=score_payload)
        assert score_res.status_code == 200, f"Scoring failed for {sc['id']}: {score_res.text}"
        score_data = score_res.json()
        result_id = score_data["result_id"]

        # Step 2: Request guidance with themes
        guidance_payload = {
            "result_id": result_id,
            "language": sc["language"],
            "themes": sc["themes"],
        }
        guidance_res = client.post("/api/questionnaire/guidance", headers=headers, json=guidance_payload)
        assert guidance_res.status_code == 200, f"Guidance failed for {sc['id']}: {guidance_res.text}"
        guidance_data = guidance_res.json()

        # Step 3: Verify safeguards
        sug_ids = [s["id"] for s in guidance_data.get("suggestions", [])]
        is_clinician_first = (sug_ids[0] == "SUG_CLINICIAN_CONTACT") if (
            guidance_data["escalation"] in ("see_clinician", "see_clinician_soon", "see_clinician_urgent") and sug_ids
        ) else None

        calming_practice_present = "SUG_CALMING_PRACTICE" in sug_ids

        outcome = {
            "scenario_id": sc["id"],
            "instrument": sc["instrument"].upper(),
            "name": sc["name"],
            "description": sc["description"],
            "submitted_answers": sc["answers"],
            "total_score": score_data["total_score"],
            "band_id": guidance_data["band_id"],
            "band_message": guidance_data["band_message"],
            "escalation": guidance_data["escalation"],
            "safety_triggered": guidance_data["safety_triggered"],
            "language": sc["language"],
            "themes_requested": sc["themes"],
            "ai_personalised": guidance_data["personalised"],
            "ai_mode": guidance_data["mode"],
            "suggestions_count": len(guidance_data["suggestions"]),
            "suggestions": guidance_data["suggestions"],
            "helplines": guidance_data["helplines"],
            "safe_fallback_text": guidance_data["safe_fallback_text"],
            "safeguards_verification": {
                "clinician_pinned_first": is_clinician_first,
                "calming_practice_excluded_in_elevated": (not calming_practice_present) if guidance_data["band_id"] in ("moderate", "moderately_severe", "severe", "possible_depression") else True,
                "only_verified_helplines": "14416" in str(guidance_data["helplines"]) and "112" in str(guidance_data["helplines"]) and "1800-889-4357" not in str(guidance_data),
            }
        }
        results.append(outcome)

    output_path = os.path.join("data", "artificial_guidance_scenarios.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return {"count": len(results), "scenarios": results, "saved_to": output_path}


if __name__ == "__main__":
    report = run_evaluation()
    print(f"\nSuccessfully evaluated {report['count']} artificial scenarios.")
    print(f"Results saved to: {report['saved_to']}\n")
    for r in report["scenarios"]:
        print(f"[{r['instrument']}] {r['name']}")
        print(f"  Score: {r['total_score']} | Band: {r['band_id']} | Escalation: {r['escalation']} | Safety Triggered: {r['safety_triggered']}")
        print(f"  AI Personalised: {r['ai_personalised']} (Mode: {r['ai_mode']})")
        print(f"  Suggestions ({r['suggestions_count']}): {[s['id'] for s in r['suggestions']]}")
        print(f"  Safeguards: {r['safeguards_verification']}")
        print("-" * 80)
