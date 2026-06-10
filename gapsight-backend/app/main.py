from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from google.genai.errors import APIError as GeminiAPIError
from pydantic import BaseModel, Field

from app.agents.claim_drafter import draft_claims
from app.agents.claim_extractor import extract_claims
from app.agents.commercial_strategist import analyze_commercialization
from app.agents.gap_identifier import identify_gaps
from app.agents.prior_art_search import execute_prior_art_search
from app.core.config import settings
from app.core.seed_data import SEED_PATENTS
from app.core.vector_db import seed_vector_db
from app.services.pdf_parser import PDFParseError, extract_text_from_pdf


logger = logging.getLogger(__name__)

ALLOWED_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}

HIGH_GAP_THRESHOLD: float = 0.7


class ExtractedClaim(BaseModel):
    claim_id: str
    category: str
    technical_description: str
    keywords: List[str] = Field(default_factory=list)


class PatentGap(BaseModel):
    claim_id: str
    gap_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    closest_prior_art: List[str] = Field(default_factory=list)


class CompetitorEntry(BaseModel):
    corporate_holder: str
    intersecting_tech: str
    threat_level: str
    whitespace_advantage: str


class CommercializationAnalysis(BaseModel):
    commercialization_score: int = Field(ge=0, le=100)
    startup_potential: str
    market_size: str
    roi_ratio: str
    funding_vehicles: List[str] = Field(default_factory=list)
    competitor_map: List[CompetitorEntry] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    status: str
    filename: str
    characters_extracted: int
    extracted_claims: List[ExtractedClaim]
    patent_gaps: List[PatentGap]
    drafted_claims: str
    commercialization: CommercializationAnalysis


def _select_claims_for_drafting(
    extracted_claims: List[Dict[str, Any]],
    gap_analysis_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not extracted_claims:
        return []

    if not gap_analysis_results:
        logger.warning(
            "Agent 3 produced no gap analyses; passing all %d extracted "
            "claim(s) to Agent 4 as a fallback.",
            len(extracted_claims),
        )
        return list(extracted_claims)

    high_gap_claim_ids = {
        str(gap.get("claim_id", "")).strip()
        for gap in gap_analysis_results
        if float(gap.get("gap_score", 0.0)) >= HIGH_GAP_THRESHOLD
        and str(gap.get("claim_id", "")).strip()
    }

    if not high_gap_claim_ids:
        highest = max(
            gap_analysis_results,
            key=lambda g: float(g.get("gap_score", 0.0)),
        )
        highest_id = str(highest.get("claim_id", "")).strip()
        logger.warning(
            "No claims met the high-gap threshold (>= %.2f); falling back "
            "to the highest-scoring claim '%s' (gap_score=%.3f).",
            HIGH_GAP_THRESHOLD,
            highest_id or "<unknown>",
            float(highest.get("gap_score", 0.0)),
        )
        if not highest_id:
            return list(extracted_claims)
        high_gap_claim_ids = {highest_id}

    filtered = [
        claim
        for claim in extracted_claims
        if str(claim.get("claim_id", "")).strip() in high_gap_claim_ids
    ]

    if not filtered:
        logger.warning(
            "High-gap claim_ids %s did not match any extracted claim "
            "(possible LLM hallucination); falling back to all %d "
            "extracted claim(s).",
            sorted(high_gap_claim_ids),
            len(extracted_claims),
        )
        return list(extracted_claims)

    logger.info(
        "Filtered %d/%d claim(s) for Agent 4 drafting based on gap_score >= %.2f.",
        len(filtered),
        len(extracted_claims),
        HIGH_GAP_THRESHOLD,
    )
    return filtered


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("GapSight starting up; attempting to seed prior-art vector DB...")
    try:
        await seed_vector_db(SEED_PATENTS)
    except Exception as exc:
        logger.exception("Vector DB seeding failed at startup: %s", exc)
    yield
    logger.info("GapSight shutting down.")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.app_debug,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "env": settings.app_env}


@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    tags=["analysis"],
)
async def analyze_paper(file: UploadFile = File(...)) -> AnalysisResponse:
    if file.content_type not in ALLOWED_PDF_CONTENT_TYPES and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF uploads are supported.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
        )

    try:
        paper_text = await extract_text_from_pdf(file_bytes)
    except PDFParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        extracted_claims = await extract_claims(paper_text)
    except GeminiAPIError as exc:
        logger.exception("Agent 1 (Claim Extractor) failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Claim extraction failed while calling the Gemini API "
                f"({exc.__class__.__name__}). Please retry."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Agent 1 raised an unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during claim extraction.",
        ) from exc

    prior_art_hits = await execute_prior_art_search(
        extracted_claims["extracted_claims"]
    )

    try:
        gap_analysis_results = await identify_gaps(
            extracted_claims["extracted_claims"],
            prior_art_hits,
        )
    except GeminiAPIError as exc:
        logger.exception("Agent 3 (Gap Identifier) failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Gap analysis failed while calling the Gemini API "
                f"({exc.__class__.__name__}). Please retry."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Agent 3 raised an unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during gap analysis.",
        ) from exc

    claims_to_draft = _select_claims_for_drafting(
        extracted_claims["extracted_claims"],
        gap_analysis_results,
    )
    drafted_claims: str = await draft_claims(claims_to_draft)

    try:
        commercialization = await analyze_commercialization(
            extracted_claims["extracted_claims"],
            gap_analysis_results,
            prior_art_hits,
        )
    except Exception as exc:
        logger.exception("Agent 5 raised an unexpected error: %s", exc)
        commercialization = {
            "commercialization_score": 0,
            "startup_potential": "Analysis unavailable",
            "market_size": "Unable to estimate",
            "roi_ratio": "—",
            "funding_vehicles": [],
            "competitor_map": [],
        }

    return AnalysisResponse(
        status="ok",
        filename=file.filename or "uploaded.pdf",
        characters_extracted=len(paper_text),
        extracted_claims=extracted_claims["extracted_claims"],
        patent_gaps=gap_analysis_results,
        drafted_claims=drafted_claims,
        commercialization=commercialization,
    )
