"""Translate API Route for HealHer.

Endpoint:
POST /api/translate
Body: { language: str, items: [{ key: str, text: str }] }
Response: { language: str, translations: { key: { text: str, status: 'reviewed' | 'machine' | 'fallback' } } }
"""

import logging
from typing import List, Dict, Literal
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel, Field

from app.graphs.translate_graph import translation_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["translate"])


class ItemIn(BaseModel):
    key: str = Field(description="Unique translation key", examples=["home.welcome"])
    text: str = Field(description="Source English text", examples=["Hello {{name}}, welcome to week {{week}}!"])


class TranslateRequest(BaseModel):
    language: str = Field(
        description="Target language code: 'hi' (Hindi), 'bn' (Bengali), or 'ta' (Tamil)",
        examples=["hi"],
    )
    items: List[ItemIn] = Field(
        description="List of items to translate (max 100)",
        examples=[
            [
                {
                    "key": "home.welcome",
                    "text": "Hello {{name}}, welcome to week {{week}} of your pregnancy!",
                }
            ]
        ],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "language": "hi",
                "items": [
                    {
                        "key": "home.welcome",
                        "text": "Hello {{name}}, welcome to week {{week}} of your pregnancy!",
                    }
                ],
            }
        }
    }


class ItemResultOut(BaseModel):
    text: str
    status: Literal["reviewed", "machine", "fallback"]


class TranslateResponse(BaseModel):
    language: str
    translations: Dict[str, ItemResultOut]


@router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate application text into Hindi, Bengali, or Tamil",
)
async def translate_content(request: Request, payload: TranslateRequest):
    """Translate batch of UI text into Hindi (hi), Bengali (bn), or Tamil (ta).

    Executes the LangGraph translation workflow.
    - Excluded keys (questionnaire.*, safety.*) are rejected with HTTP 400.
    - Valid items are served from reviewed/machine cache, or translated via Gemini.
    - Preserves placeholders like {{name}}, {{week}}, %s.
    - Falls back to English on LLM error/timeout with status 'fallback'.
    """
    custom_llm_fn = getattr(request.app.state, "custom_llm_fn", None)

    state_input = {
        "language": payload.language,
        "items": [{"key": it.key, "text": it.text} for it in payload.items],
        "results": {},
        "missing_items": [],
        "attempt": 1,
        "errors": [],
    }
    if custom_llm_fn is not None:
        state_input["custom_llm_fn"] = custom_llm_fn

    state_output = await translation_graph.ainvoke(state_input)

    errors = state_output.get("errors", [])
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors},
        )

    final_resp = state_output.get("final_response", {})
    return final_resp
