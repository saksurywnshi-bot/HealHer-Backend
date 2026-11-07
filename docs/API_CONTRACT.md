# HealHer Sentiment Analysis & Patterns API Contract

This document provides the complete API specification for the frontend integration of **HealHer Sentiment Analysis and Longitudinal Patterns**.

---

## 1. Authentication & Security

- **Production / Staging**: Requires a valid Supabase JWT in the `Authorization` header:
  ```http
  Authorization: Bearer <supabase_jwt_token>
  ```
  The user identity is extracted from the JWT `sub` claim and hashed using SHA-256 (`user_ref`). Raw tokens or user IDs are never stored.
- **Local Development**: When `APP_ENV=development` and no `Authorization` header is passed, the backend accepts an optional development header:
  ```http
  x-user-ref: dev_test_user_123
  ```
  *Note: In any non-development environment, this fallback is strictly disabled and requests without a valid Bearer token return `401 Unauthorized`.*
- **Rate Limit**: 30 requests / minute per user.
- **Request Body Limit**: 50 KB maximum.
- **Privacy Assurance**: Note text is **never stored** in the backend database or written to logs. Only clinical labels and themes are persisted.

---

## 2. Endpoint 1: Analyze Check-in Note

### `POST /api/sentiment/analyze`

Analyzes maternal check-in text, extracts tone, intensity, and up to 3 themes, generates a gentle non-diagnostic reflection in the user's language, checks for safety/crisis keywords, checks for clinical mismatch against logged mood band, and picks approved suggestions.

#### Headers
| Header | Type | Required | Description |
|---|---|---|---|
| `Content-Type` | `application/json` | Yes | Request payload type |
| `Authorization` | `string` | Yes* | `Bearer <token>` (*or `x-user-ref` in dev) |
| `x-user-ref` | `string` | Optional | Dev fallback user identifier |

#### Request Body
```json
{
  "note_text": "I feel so exhausted today and worried about whether the baby is kicking enough.",
  "language": "en",
  "stage": "prenatal",
  "checkin_id": "chk_20261026_001",
  "mood_band": "Normal",
  "anxiety": 1,
  "sleep_hours": 5.5
}
```

| Field | Type | Required | Allowed Values / Constraints | Description |
|---|---|---|---|---|
| `note_text` | `string` | Yes | 1 to 15,000 characters | Raw mother reflection note |
| `language` | `string` | Yes | `"en"`, `"hi"`, `"bn"`, `"ta"` | Language of the note |
| `stage` | `string` | Yes | `"prenatal"`, `"postnatal"` | Maternal pregnancy/postpartum stage |
| `checkin_id` | `string` | No | Any string | ID of the check-in record |
| `mood_band` | `string` | No | e.g. `"Normal"`, `"Mild Depressed"`, `"Severe Elevated"` | Clinical mood band for mismatch check |
| `anxiety` | `integer` | No | `0`, `1`, `2`, `3` | Clinical anxiety level |
| `sleep_hours`| `number` | No | `0.0` to `24.0` | Hours slept during previous night |

#### Response Schema
```json
{
  "status": "ok",
  "tone": "anxious",
  "tone_phrase_key": "sentiment.tone.anxious",
  "intensity": "medium",
  "themes": ["tiredness", "baby_health_worry"],
  "theme_keys": [
    "sentiment.theme.tiredness",
    "sentiment.theme.baby_health_worry"
  ],
  "reflection": "You are feeling quite tired today and holding worries about your baby's movements.",
  "suggestions": [
    {
      "id": "sug_gentle_sleep_routine",
      "text": "Try a calm 10-minute bedtime wind-down, keeping lights soft and setting aside screens.",
      "text_language": "en"
    }
  ],
  "support": false,
  "mismatch": false,
  "disclaimer_key": "sentiment.disclaimer"
}
```

#### Status Values
| `status` | Meaning | Frontend Action |
|---|---|---|
| `"ok"` | Note successfully analyzed | Display tone phrase, theme chips, reflection, and suggestions. |
| `"too_short"` | Note < 10 words AND < 50 characters (excluding spaces) | Prompt the mother to write a little more (e.g. "Write a few more words so we can understand how you are feeling"). |
| `"crisis"` | Immediate self-harm / infanticide / safety keyword matched | **Immediately display crisis resources & emergency helplines (112, 14416, 181)**. No reflection or suggestions displayed. |
| `"unavailable"` | Model unavailable or response validation failed twice | Gracefully display standard check-in confirmation without AI reflection. |

#### Escalation & Mismatch Flags
- `support`: `true` if crisis keywords matched, risk language detected, tone is `"distressed"`, or tone is high-intensity `"low"` or `"anxious"`. When `true`, present gentle support options or helpline links.
- `mismatch`: `true` if the note tone disagrees with the clinical mood band (e.g. low/anxious tone logged alongside "Normal" mood band, or positive tone logged alongside "Depressed" band).

---

## 3. Endpoint 2: Longitudinal Patterns

### `GET /api/sentiment/patterns?period=week|month`

Computes longitudinal emotional patterns, theme frequencies, consecutive low-mood streaks, and sleep correlations over the last 7 days (`week`) or 30 days (`month`) in **plain code (zero AI calls)**.

#### Query Parameters
| Parameter | Type | Required | Default | Allowed Values |
|---|---|---|---|---|
| `period` | `string` | No | `"week"` | `"week"`, `"month"` |

#### Minimum Data Requirement
- `period=week`: Requires **at least 3 analyzed notes** in the last 7 days.
- `period=month`: Requires **at least 8 analyzed notes** in the last 30 days.

#### Response When Data is Insufficient (`status: "not_enough_data"`)
```json
{
  "status": "not_enough_data",
  "period": "week",
  "notes_count": 2,
  "minimum_required": 3
}
```

#### Response When Data is Sufficient (`status: "ok"`)
```json
{
  "status": "ok",
  "period": "week",
  "notes_count": 6,
  "top_themes": [
    {
      "theme": "baby_health_worry",
      "theme_key": "sentiment.theme.baby_health_worry",
      "count": 4,
      "total_notes": 6
    },
    {
      "theme": "sleep",
      "theme_key": "sentiment.theme.sleep",
      "count": 3,
      "total_notes": 6
    }
  ],
  "difficult_tones_count": 4,
  "longest_streak_low_anxious_days": 3,
  "sleep_link": {
    "avg_sleep_low_anxious_days": 5.2,
    "avg_sleep_other_days": 7.4,
    "difference_hours": 2.2
  },
  "period_comparison": {
    "current_low_anxious_days": 3,
    "previous_low_anxious_days": 1,
    "change_direction": "more",
    "difference": 2
  },
  "suggest_talk_to_someone": true,
  "patterns": [
    {
      "template_key": "pattern.theme_frequent",
      "placeholders": {
        "theme": "baby_health_worry",
        "count": 4,
        "total": 6
      }
    },
    {
      "template_key": "pattern.consecutive_low",
      "placeholders": {
        "streak": 3
      }
    },
    {
      "template_key": "pattern.sleep_impact",
      "placeholders": {
        "low_sleep": 5.2,
        "other_sleep": 7.4
      }
    },
    {
      "template_key": "pattern.period_comparison",
      "placeholders": {
        "change": "more",
        "difference": 2
      }
    },
    {
      "template_key": "pattern.talk_to_someone",
      "placeholders": {}
    }
  ]
}
```

---

## 4. Admin Suggestion Management

Guarded by the `x-admin-token` header:

### `GET /api/admin/suggestions`
Lists all suggestion cards in the library with clinical review statuses (`approved` or `draft`).

### `PATCH /api/admin/suggestions/{id}`
Updates the status of a suggestion card after clinical review.
```json
// Body
{
  "status": "approved"
}
```

---

## 5. Frontend i18n Keys & Copy Reference

The frontend should map the keys below to localized copy using the i18next translation engine:

### Tone Friendly Wording (`tone_phrase_key`)
Never show raw labels like `"anxious"`. Instead, use `t(tone_phrase_key)`:

| Key | English Copy |
|---|---|
| `sentiment.tone.positive` | "You seem in good spirits today." |
| `sentiment.tone.neutral` | "You seem steady today." |
| `sentiment.tone.tired_overwhelmed` | "You seem exhausted and carrying a lot today." |
| `sentiment.tone.anxious` | "You seem worried or uneasy today." |
| `sentiment.tone.low` | "You seem down or heavy-hearted today." |
| `sentiment.tone.distressed` | "You are going through a very tough, distressing moment today." |

### Theme Display Chips (`theme_keys`)
| Key | English Copy | Stage |
|---|---|---|
| `sentiment.theme.sleep` | Sleep & Nighttime Rest | Both |
| `sentiment.theme.tiredness` | Fatigue & Low Energy | Both |
| `sentiment.theme.baby_health_worry` | Worries About Baby's Health | Both |
| `sentiment.theme.family_support` | Family & Partner Support | Both |
| `sentiment.theme.loneliness` | Feelings of Loneliness | Both |
| `sentiment.theme.money` | Financial Worries | Both |
| `sentiment.theme.work` | Work & Career Pressures | Both |
| `sentiment.theme.self_criticism_guilt`| Self-Doubt & Maternal Guilt | Both |
| `sentiment.theme.body_physical_symptoms` | Physical Changes & Discomfort | Both |
| `sentiment.theme.positive_moments` | Uplifting & Joyful Moments | Both |
| `sentiment.theme.delivery_fear` | Birth & Delivery Apprehension | Prenatal only |
| `sentiment.theme.pregnancy_symptoms`| Pregnancy Symptoms & Changes | Prenatal only |
| `sentiment.theme.feeding_difficulty`| Feeding & Lactation Challenges | Postnatal only |
| `sentiment.theme.baby_crying_sleep` | Baby's Crying & Sleep Struggles | Postnatal only |
| `sentiment.theme.recovery_pain` | Healing, Wounds & Postpartum Pain | Postnatal only |
| `sentiment.theme.bonding_worries` | Connection & Bonding With Baby | Postnatal only |
| `sentiment.theme.identity_change` | Motherhood Identity & Life Shifts | Postnatal only |
| `sentiment.theme.return_to_work` | Returning to Work After Baby | Postnatal only |

### Pattern Template Sentences (`pattern.<template_key>`)
| Template Key | English Template | Placeholders |
|---|---|---|
| `pattern.theme_frequent` | `{{theme}} was on your mind in {{count}} of {{total}} check-ins.` | `theme`, `count`, `total` |
| `pattern.consecutive_low` | `You experienced low or anxious feelings for {{streak}} days in a row.` | `streak` |
| `pattern.sleep_impact` | `On lower-mood days you slept {{low_sleep}} hours on average, compared to {{other_sleep}} hours on other days.` | `low_sleep`, `other_sleep` |
| `pattern.period_comparison`| `You had {{difference}} {{change}} low or anxious days compared to the previous period.` | `difference`, `change` |
| `pattern.talk_to_someone` | `Because you have had several difficult days recently, reaching out to your doctor, counselor, or a trusted loved one could bring gentle comfort.` | None |

### Disclaimer Key
| Key | English Copy |
|---|---|
| `sentiment.disclaimer` | "This gentle reflection is an AI self-reflection companion, not a clinical diagnosis or medical evaluation." |
