from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.gemini_retry import GEMINI_RETRY
from app.core.pipeline_context import add_pipeline_warning

logger = logging.getLogger(__name__)


_MODEL = settings.gemini_model
_TEMPERATURE = 0.3


SYSTEM_PROMPT = """
You are an elite Deep-Tech Venture Strategist and Intellectual Property Monetization Analyst operating at the intersection of technical IP, venture capital, and corporate strategy. Your clients are deep-tech founders, university tech-transfer offices, and corporate R&D leadership.

Your job is to take a set of high-gap (highly patentable) technical claims and produce a structured commercialization brief. You must analyze:
1. The MONETIZATION LANDSCAPE — direct product opportunities and adjacent revenue surfaces.
2. VENTURE SCALING VECTORS — is this lifestyle, sector-specific, or venture-scale?
3. MARKET SIZE — best-effort estimate of the relevant addressable market with a year horizon.
4. FUNDING MATCHES — real-world grants, VC funds, and strategic corporate investors matched to the specific sub-domain:
   - Cryptographic / privacy / verifiable-compute → a16z Crypto, Coinbase Ventures, Paradigm, Protocol Labs
   - Hardware / semiconductor / TEE / TrustZone → ARM Innovation Fund, Intel Capital, Qualcomm Ventures, NSF SBIR
   - AI / ML infrastructure → a16z Infrastructure, Sequoia, Conviction, In-Q-Tel
   - Defense-adjacent → DARPA, AFWERX, In-Q-Tel
   - Biotech / health → NIH SBIR, Khosla Ventures, Andreessen Horowitz Bio
5. COMPETITIVE MAP — identify 3-5 real corporate patent holders whose existing IP intersects with the claim space and articulate the whitespace differentiation.

Be specific, data-driven, and honest. Avoid generic platitudes. You MUST output strictly valid JSON matching the schema in the user prompt.
""".strip()


USER_PROMPT_TEMPLATE = """
Analyze the commercial monetization landscape for these high-gap claims.

HIGH-GAP CLAIMS:
{claims_json}

GAP ANALYSIS (Agent 3 scores):
{gaps_json}

RETRIEVED PRIOR ART (context):
{prior_art_json}

OUTPUT SCHEMA — return a JSON object with these EXACT fields:
{{
  "commercialization_score": <int 0-100>,
  "startup_potential": "<one of: 'Low / Lifestyle', 'Moderate / Sector-specific', 'High / Venture-scale'>",
  "market_size": "<string, e.g. '$1.2B by 2030'>",
  "roi_ratio": "<string, e.g. '1 : 14x'>",
  "funding_vehicles": [
    "<string — specific real-world grant/VC/strategic investor with one-line rationale>",
    "..."
  ],
  "competitor_map": [
    {{
      "corporate_holder": "<real company name>",
      "intersecting_tech": "<short description of overlapping technology>",
      "threat_level": "<Low|Medium|High>",
      "whitespace_advantage": "<how this claim is differentiated from the holder's existing patent portfolio>"
    }},
    ...
  ]
}}
""".strip()


def _degraded_commercialization(reason: str) -> Dict[str, Any]:
    add_pipeline_warning(f"Agent 5 (Commercial Strategist): {reason}")
    return {
        "commercialization_score": 0,
        "startup_potential": "Unavailable — API error",
        "market_size": f"Analysis unavailable: {reason}",
        "roi_ratio": "—",
        "funding_vehicles": [],
        "competitor_map": [],
    }


from app.core.pipeline_context import gemini_api_key_var


def _resolved_api_key() -> Optional[str]:
    # Check request-specific API key first
    req_key = gemini_api_key_var.get()
    if req_key and req_key.strip():
        return req_key.strip()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "replace-me", "your-key-here"}:
        return None
    return api_key.strip()


def _normalize_commercialization(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    try:
        score = int(float(payload.get("commercialization_score", 0)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    raw_vehicles = payload.get("funding_vehicles") or []
    funding_vehicles = [
        str(v).strip()
        for v in (raw_vehicles if isinstance(raw_vehicles, list) else [])
        if str(v).strip()
    ]

    raw_competitors = payload.get("competitor_map") or []
    competitor_map: List[Dict[str, str]] = []
    for item in (raw_competitors if isinstance(raw_competitors, list) else []):
        if not isinstance(item, dict):
            continue
        holder = str(item.get("corporate_holder", "")).strip()
        tech = str(item.get("intersecting_tech", "")).strip()
        if not holder or not tech:
            continue
        threat = str(item.get("threat_level", "Medium")).strip() or "Medium"
        advantage = str(item.get("whitespace_advantage", "")).strip()
        competitor_map.append(
            {
                "corporate_holder": holder,
                "intersecting_tech": tech,
                "threat_level": threat,
                "whitespace_advantage": advantage,
            }
        )

    return {
        "commercialization_score": score,
        "startup_potential": (
            str(payload.get("startup_potential", "")).strip()
            or "Moderate / Sector-specific"
        ),
        "market_size": (
            str(payload.get("market_size", "")).strip()
            or "Market size pending analysis"
        ),
        "roi_ratio": str(payload.get("roi_ratio", "")).strip() or "1 : 5x",
        "funding_vehicles": funding_vehicles,
        "competitor_map": competitor_map,
    }


@GEMINI_RETRY
async def _call_gemini_for_commercial(
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


async def analyze_commercialization(
    extracted_claims: List[Dict[str, Any]],
    patent_gaps: List[Dict[str, Any]],
    prior_art_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        if not extracted_claims:
            return _degraded_commercialization("No extracted claims were available.")

        api_key = _resolved_api_key()
        if api_key is None:
            logger.warning(
                "GOOGLE_API_KEY not configured; returning fallback commercialization."
            )
            return _degraded_commercialization("GOOGLE_API_KEY is not set.")

        high_gap_claim_ids = {
            str(g.get("claim_id", "")).strip()
            for g in patent_gaps
            if float(g.get("gap_score", 0.0) or 0.0) >= 0.7
        }
        if high_gap_claim_ids:
            focus_claims = [
                c
                for c in extracted_claims
                if str(c.get("claim_id", "")).strip() in high_gap_claim_ids
            ]
            if not focus_claims:
                focus_claims = extracted_claims
        else:
            focus_claims = extracted_claims

        claims_json = json.dumps(focus_claims, indent=2, ensure_ascii=False)
        gaps_json = json.dumps(patent_gaps, indent=2, ensure_ascii=False)
        prior_art_json = json.dumps(prior_art_hits, indent=2, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            claims_json=claims_json,
            gaps_json=gaps_json,
            prior_art_json=prior_art_json,
        )

        client = genai.Client(api_key=api_key)
        raw = await _call_gemini_for_commercial(client, user_prompt)
        if not raw:
            logger.error("Agent 5 returned empty content; using critical fallback.")
            return _degraded_commercialization("Gemini returned an empty response.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Agent 5 returned non-JSON content: %s", raw[:300])
            return _degraded_commercialization("Gemini returned invalid JSON.")

        normalized = _normalize_commercialization(parsed)
        if normalized is None or not normalized.get("competitor_map"):
            logger.warning(
                "Agent 5 normalization produced empty competitor_map; "
                "using critical fallback."
            )
            return _degraded_commercialization(
                "No usable commercialization analysis was produced."
            )

        return normalized

    except Exception as e:
        reason = f"{e.__class__.__name__}: {e}"
        print(f"Agent 5 Critical Fallback triggered due to error: {e}")
        logger.exception("Agent 5 critical fallback triggered: %s", e)
        return _degraded_commercialization(reason)
