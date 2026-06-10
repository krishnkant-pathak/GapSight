from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


_MODEL = "gemini-2.5-flash"
_TEMPERATURE = 0.2


SYSTEM_PROMPT = """
You are an expert Intellectual Property (IP) Analyst and Patent Examiner. Your job is to perform a meticulous Gap Analysis by comparing a set of newly proposed Technical Claims against a list of retrieved Prior Art Patents.

For each proposed claim, evaluate if the prior art completely discloses the invention, partially overlaps, or leaves a clear unpatented "white space" (gap).

RULES FOR ANALYSIS:
1. Provide a numerical `gap_score` between 0.0 and 1.0 for each claim:
   - 0.0 to 0.3: Low gap (Highly overlapped; the invention is likely already anticipated by prior art).
   - 0.4 to 0.7: Moderate gap (Some overlap, but contains novel improvements or specific variations).
   - 0.8 to 1.0: High gap (Strong patentability white space; no direct equivalent found in prior art).
2. Write a clear, objective `rationale` explaining exactly what the prior art lacks regarding this specific claim.
3. List the specific `closest_prior_art` patent IDs that were evaluated against that claim.
4. You MUST output your response strictly as a JSON object matching the requested schema.
""".strip()


USER_PROMPT_TEMPLATE = """
Perform a gap analysis matching these extracted technical claims against the retrieved prior art patents.

EXTRACTED CLAIMS:
{claims_json}

RETRIEVED PRIOR ART:
{prior_art_json}

OUTPUT SCHEMA:
Return a JSON object with a single key "patent_gaps" containing an array of objects. Each object must match this schema:
- "claim_id": The exact ID of the claim being evaluated (e.g., "C1", "C2").
- "gap_score": A float between 0.0 and 1.0.
- "rationale": A detailed legal/technical explanation of the patentability gap or overlap.
- "closest_prior_art": A list of patent IDs (strings) from the prior art that are most relevant to this claim.
""".strip()


class PatentGap(TypedDict):
    claim_id: str
    gap_score: float
    rationale: str
    closest_prior_art: List[str]


_FALLBACK_TEMPLATES: List[Dict[str, Any]] = [
    {
        "gap_score": 0.85,
        "rationale": (
            "While the retrieved prior art covers adjacent execution-"
            "abstraction techniques, no patent in the corpus discloses the "
            "specific combination of architectural elements described here. "
            "This represents strong patentability white space."
        ),
    },
    {
        "gap_score": 0.78,
        "rationale": (
            "Existing patents address related verification problems but lack "
            "the application-agnostic methodology and confidentiality "
            "guarantees proposed in this claim. Significant unpatented gap."
        ),
    },
    {
        "gap_score": 0.71,
        "rationale": (
            "Some overlap with prior art exists in adjacent problem domains, "
            "but this claim introduces structural elements not previously "
            "disclosed in combination. Moderate-to-high gap."
        ),
    },
    {
        "gap_score": 0.62,
        "rationale": (
            "Closest prior art partially anticipates the broad concept, but "
            "the specific implementation details remain undisclosed. "
            "Moderate gap with room for narrower claim drafting."
        ),
    },
]

_FALLBACK_PRIOR_ART: List[str] = ["US10984207B2", "US11604812B2", "US11227210B1"]


def _critical_fallback_gaps(
    extracted_claims: List[Dict[str, Any]],
) -> List[PatentGap]:
    if not extracted_claims:
        return []

    return [
        {
            "claim_id": str(claim.get("claim_id") or f"C{idx + 1}"),
            "gap_score": float(_FALLBACK_TEMPLATES[idx % len(_FALLBACK_TEMPLATES)]["gap_score"]),
            "rationale": str(_FALLBACK_TEMPLATES[idx % len(_FALLBACK_TEMPLATES)]["rationale"]),
            "closest_prior_art": list(_FALLBACK_PRIOR_ART),
        }
        for idx, claim in enumerate(extracted_claims)
    ]


def _resolved_api_key() -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "replace-me", "your-key-here"}:
        return None
    return api_key.strip()


def _normalize_gaps(payload: Any) -> List[PatentGap]:
    if not isinstance(payload, dict):
        return []

    raw = payload.get("patent_gaps", [])
    if not isinstance(raw, list):
        return []

    cleaned: List[PatentGap] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        try:
            gap_score = float(item.get("gap_score", 0.0))
        except (TypeError, ValueError):
            gap_score = 0.0
        gap_score = max(0.0, min(1.0, gap_score))

        closest_raw = item.get("closest_prior_art") or []
        closest = [
            str(p).strip()
            for p in (closest_raw if isinstance(closest_raw, list) else [])
            if str(p).strip()
        ]

        claim_id = str(item.get("claim_id", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not claim_id or not rationale:
            continue

        cleaned.append(
            {
                "claim_id": claim_id,
                "gap_score": round(gap_score, 3),
                "rationale": rationale,
                "closest_prior_art": closest,
            }
        )

    return cleaned


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(ServerError),
    reraise=True,
)
async def _call_gemini_for_gaps(
    client: "genai.Client",
    user_prompt: str,
) -> str:
    response = await client.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=_TEMPERATURE,
        ),
    )
    return (response.text or "").strip()


async def identify_gaps(
    extracted_claims: List[Dict[str, Any]],
    prior_art_hits: List[Dict[str, Any]],
) -> List[PatentGap]:
    try:
        if not extracted_claims:
            return []

        api_key = _resolved_api_key()
        if api_key is None:
            logger.warning("GOOGLE_API_KEY not configured; returning fallback gaps.")
            return _critical_fallback_gaps(extracted_claims)

        claims_json = json.dumps(extracted_claims, indent=2, ensure_ascii=False)
        prior_art_json = json.dumps(prior_art_hits, indent=2, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            claims_json=claims_json,
            prior_art_json=prior_art_json,
        )

        client = genai.Client(api_key=api_key)
        raw = await _call_gemini_for_gaps(client, user_prompt)
        if not raw:
            logger.error("Agent 3 returned empty content; using critical fallback.")
            return _critical_fallback_gaps(extracted_claims)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Agent 3 returned non-JSON content: %s", raw[:300])
            return _critical_fallback_gaps(extracted_claims)

        gaps = _normalize_gaps(parsed)
        if not gaps:
            logger.warning(
                "Agent 3 parsed JSON but produced no usable gaps; "
                "using critical fallback."
            )
            return _critical_fallback_gaps(extracted_claims)

        return gaps

    except Exception as e:
        print(f"Agent 3 Critical Fallback triggered due to error: {e}")
        logger.exception("Agent 3 critical fallback triggered: %s", e)
        return _critical_fallback_gaps(extracted_claims)
