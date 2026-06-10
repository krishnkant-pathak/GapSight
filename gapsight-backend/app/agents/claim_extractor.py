from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from google import genai
from google.genai import types

from app.core.gemini_retry import GEMINI_RETRY
from app.core.pipeline_context import add_pipeline_warning

logger = logging.getLogger(__name__)


_MODEL = "gemini-2.5-flash"
_MAX_CHARS = 60_000


SYSTEM_PROMPT = """
You are an elite Patent Attorney and Deep Tech Technical Analyst. Your objective is to analyze academic research papers and extract the novel, patentable technical claims.
RULES FOR EXTRACTION:
1. Ignore standard background information, related works, and generic problem statements.
2. Focus strictly on the "Methodology", "Architecture", "Proposed System", and "Implementation" sections.
3. Formulate each extracted concept as a distinct, isolated "technical claim".
4. Ensure claims are highly technical, specific, and detailed enough to be searched against a patent database.
5. You MUST output your response strictly as a JSON object matching the requested schema.
""".strip()


USER_PROMPT_TEMPLATE = """
Below is the extracted text from a research paper. Analyze the text and extract the core novel technical claims.

RESEARCH PAPER TEXT:
{pdf_text}

OUTPUT SCHEMA:
Return a JSON object with a single key "extracted_claims" containing an array of objects. Each object should have:
- "claim_id": A unique identifier (e.g., "C1", "C2").
- "category": The type of claim (e.g., "Algorithm", "Hardware").
- "technical_description": The highly detailed explanation of the novel mechanism or process.
- "keywords": A list of 3-5 specific technical keywords to be used for querying prior art databases.
""".strip()


class ExtractedClaim(TypedDict):
    claim_id: str
    category: str
    technical_description: str
    keywords: List[str]


def _degraded_claims(reason: str) -> Dict[str, Any]:
    add_pipeline_warning(f"Agent 1 (Claim Extractor): {reason}")
    return {
        "extracted_claims": [
            {
                "claim_id": "C0",
                "category": "Pipeline Notice",
                "technical_description": (
                    f"Live claim extraction could not run: {reason}. "
                    "Your PDF was parsed successfully, but Gemini did not "
                    "return paper-specific claims. Check your API key, billing, "
                    "and daily quota at https://ai.dev/rate-limit, then retry."
                ),
                "keywords": ["pipeline-degraded", "retry-required"],
            }
        ]
    }


def _resolved_api_key() -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "replace-me", "your-key-here"}:
        return None
    return api_key.strip()


def _stub_payload(reason: str) -> Dict[str, Any]:
    add_pipeline_warning(f"Agent 1 (Claim Extractor): {reason}")
    return _degraded_claims(reason)


def _normalize_claims(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"extracted_claims": []}

    raw_claims = payload.get("extracted_claims", [])
    if not isinstance(raw_claims, list):
        return {"extracted_claims": []}

    cleaned: List[ExtractedClaim] = []
    for idx, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            continue
        description = str(item.get("technical_description", "")).strip()
        if not description:
            continue
        keywords_raw = item.get("keywords") or []
        keywords = [
            str(kw).strip()
            for kw in (keywords_raw if isinstance(keywords_raw, list) else [])
            if str(kw).strip()
        ]
        cleaned.append(
            {
                "claim_id": str(item.get("claim_id") or f"C{idx}"),
                "category": str(item.get("category") or "Uncategorized").strip(),
                "technical_description": description,
                "keywords": keywords,
            }
        )

    return {"extracted_claims": cleaned}


@GEMINI_RETRY
async def _call_gemini_for_claims(
    client: "genai.Client",
    user_prompt: str,
) -> str:
    response = await client.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return (response.text or "").strip()


async def extract_claims(pdf_text: str) -> Dict[str, Any]:
    try:
        if not pdf_text or not pdf_text.strip():
            return {"extracted_claims": []}

        api_key = _resolved_api_key()
        if api_key is None:
            logger.warning("GOOGLE_API_KEY not configured; returning stub claim.")
            return _stub_payload("GOOGLE_API_KEY is not set.")

        client = genai.Client(api_key=api_key)
        truncated = pdf_text[:_MAX_CHARS]
        user_prompt = USER_PROMPT_TEMPLATE.format(pdf_text=truncated)

        raw = await _call_gemini_for_claims(client, user_prompt)
        if not raw:
            logger.error("Gemini returned empty content for claim extraction.")
            return _degraded_claims("Gemini returned an empty response.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Claim extractor returned non-JSON content: %s", raw[:300])
            return _degraded_claims("Gemini returned invalid JSON for claim extraction.")

        normalized = _normalize_claims(parsed)
        if not normalized.get("extracted_claims"):
            logger.warning(
                "Claim extractor returned zero usable claims after "
                "normalization; falling back."
            )
            return _degraded_claims("No usable claims were extracted from the paper.")
        return normalized

    except Exception as e:
        reason = f"{e.__class__.__name__}: {e}"
        print(f"Agent 1 Critical Fallback triggered due to error: {e}")
        logger.exception("Agent 1 critical fallback triggered: %s", e)
        return _degraded_claims(reason)
