from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, TypedDict

from app.core.vector_db import search_prior_art as vector_search

logger = logging.getLogger(__name__)


_TOP_K_PER_CLAIM = 3


class PriorArtHit(TypedDict):
    patent_id: str
    title: str
    abstract: str
    similarity_score: float
    matched_claim_ids: List[str]


_STUB_HITS: List[PriorArtHit] = [
    {
        "patent_id": "US-STUB-0000",
        "title": "[STUB] Prior art search returned no results",
        "abstract": (
            "Vector DB is empty or the embedding service failed. Verify "
            "GOOGLE_API_KEY is set and that the seed corpus was loaded at "
            "startup (look for 'Seeded vector DB' in the uvicorn logs)."
        ),
        "similarity_score": 0.0,
        "matched_claim_ids": [],
    }
]


def _build_query(claim: Dict[str, Any]) -> str:
    description = str(claim.get("technical_description", "")).strip()
    keywords_raw = claim.get("keywords") or []
    keywords = " ".join(
        str(kw).strip()
        for kw in (keywords_raw if isinstance(keywords_raw, list) else [])
        if str(kw).strip()
    )
    if description and keywords:
        return f"{description}\n\nKeywords: {keywords}"
    return description or keywords


async def _safe_search(query: str) -> List[Dict[str, Any]]:
    if not query:
        return []
    try:
        return await vector_search(query, top_k=_TOP_K_PER_CLAIM)
    except Exception as exc:
        logger.warning("Per-claim prior-art search failed: %s", exc)
        return []


async def execute_prior_art_search(
    extracted_claims: List[Dict[str, Any]],
) -> List[PriorArtHit]:
    if not extracted_claims:
        return []

    queries = [_build_query(claim) for claim in extracted_claims]
    per_claim_hits = await asyncio.gather(*(_safe_search(q) for q in queries))

    by_patent: Dict[str, PriorArtHit] = {}
    for claim, hits in zip(extracted_claims, per_claim_hits):
        claim_id = str(claim.get("claim_id", "")).strip()
        for hit in hits:
            patent_id = str(hit.get("patent_id", "")).strip()
            if not patent_id:
                continue
            score = float(hit.get("similarity_score", 0.0))
            existing = by_patent.get(patent_id)
            if existing is None:
                by_patent[patent_id] = {
                    "patent_id": patent_id,
                    "title": str(hit.get("title", "")),
                    "abstract": str(hit.get("abstract", "")),
                    "similarity_score": score,
                    "matched_claim_ids": [claim_id] if claim_id else [],
                }
            else:
                if score > existing["similarity_score"]:
                    existing["similarity_score"] = score
                if claim_id and claim_id not in existing["matched_claim_ids"]:
                    existing["matched_claim_ids"].append(claim_id)

    if not by_patent:
        logger.warning("Prior art search produced no hits; returning stub.")
        return _STUB_HITS

    consolidated = sorted(
        by_patent.values(),
        key=lambda h: h["similarity_score"],
        reverse=True,
    )
    return consolidated
