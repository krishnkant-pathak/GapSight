from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from app.core.gemini_retry import GEMINI_RETRY
from app.core.pipeline_context import add_pipeline_warning

logger = logging.getLogger(__name__)


_MODEL = "gemini-2.5-flash"
_TEMPERATURE = 0.4


SYSTEM_PROMPT = """
You are an elite Patent Attorney specializing in drafting provisional patent applications for deep tech, AI, and software. Your task is to take extracted technical features and draft formal provisional patent claims.

RULES FOR DRAFTING:
1. Use standard formal patent legalese (e.g., "1. A system comprising...", "2. The system of claim 1, further comprising...").
2. Draft one primary "independent claim" covering the core architecture/methodology.
3. Draft 2-3 "dependent claims" that expand on specific technical nuances, algorithms, or hardware configurations provided in the extracted data.
4. Output your response strictly as a cleanly formatted Markdown string, using bolding for key terms.
5. Do not include any conversational filler, introductory remarks, or markdown code blocks (like ```markdown). Just the draft.
""".strip()


USER_PROMPT_TEMPLATE = """
Draft formal provisional patent claims based on the following extracted technical features:

EXTRACTED CLAIMS:
{claims_json}
""".strip()


def _degraded_draft(reason: str) -> str:
    add_pipeline_warning(f"Agent 4 (Claim Drafter): {reason}")
    return f"""\
**Patent draft unavailable**

Live claim drafting could not run: {reason}

Your uploaded paper was processed, but Gemini did not generate paper-specific patent language. Check your API key, billing, and daily quota, then retry the analysis.

---

*GapSight — retry after resolving the Gemini API issue.*
"""


def _resolved_api_key() -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "replace-me", "your-key-here"}:
        return None
    return api_key.strip()


@GEMINI_RETRY
async def _call_gemini_for_draft(
    client: "genai.Client",
    user_prompt: str,
) -> str:
    response = await client.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=_TEMPERATURE,
        ),
    )
    return (response.text or "").strip()


async def draft_claims(extracted_claims: List[Dict[str, Any]]) -> str:
    try:
        if not extracted_claims:
            return _degraded_draft("No extracted claims were available to draft from.")

        api_key = _resolved_api_key()
        if api_key is None:
            logger.warning("GOOGLE_API_KEY not configured; returning fallback draft.")
            return _degraded_draft("GOOGLE_API_KEY is not set.")

        claims_json = json.dumps(extracted_claims, indent=2, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(claims_json=claims_json)

        client = genai.Client(api_key=api_key)
        content = await _call_gemini_for_draft(client, user_prompt)

        if not content:
            logger.warning("Agent 4 returned empty content; using critical fallback.")
            return _degraded_draft("Gemini returned an empty response.")

        return content

    except Exception as e:
        reason = f"{e.__class__.__name__}: {e}"
        print(f"Agent 4 Critical Fallback triggered due to error: {e}")
        logger.exception("Agent 4 critical fallback triggered: %s", e)
        return _degraded_draft(reason)
