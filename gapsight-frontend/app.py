from __future__ import annotations

import html
import json
import threading
import time
from typing import Any, Dict, List, Optional

import os
import requests
import streamlit as st


BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
ANALYZE_ENDPOINT: str = f"{BACKEND_URL}/api/v1/analyze"
HEALTH_ENDPOINT: str = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT_S: int = 240

AGENT_DURATIONS_S: List[float] = [5.0, 3.0, 22.0, 22.0, 22.0]

AGENTS_META: List[Dict[str, str]] = [
    {
        "id": "Agent 1",
        "role": "Claim Extractor",
        "processing": "Distilling technical claims...",
        "success_template": "{count} novel contribution profiles extracted.",
    },
    {
        "id": "Agent 2",
        "role": "Prior Art Searcher",
        "processing": "Querying local semantic vector indexes...",
        "success_template": "Historical prior art nodes mapped.",
    },
    {
        "id": "Agent 3",
        "role": "Gap Identifier",
        "processing": "Calculating patentability mathematical delta...",
        "success_template": "Maximum white-space gaps identified.",
    },
    {
        "id": "Agent 4",
        "role": "Claim Drafter",
        "processing": "Compiling legal-grade provisional patent text...",
        "success_template": "Standard claims draft written.",
    },
    {
        "id": "Agent 5",
        "role": "Commercial Strategist",
        "processing": "Evaluating market size and venture funding match arrays...",
        "success_template": "Commercialization analytics optimized.",
    },
]

st.set_page_config(
    page_title="GapSight — Research-to-Patent Gap Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .gs-header {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.25rem 0;
        }
        .gs-subheader {
            color: #475569;
            font-size: 1.05rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }
        .gs-section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0.5rem 0 0.25rem 0;
        }
        .gs-section-caption {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .gs-badge {
            display: inline-block;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-right: 0.4rem;
        }
        .gs-badge-green { background: #dcfce7; color: #166534; }
        .gs-badge-amber { background: #fef3c7; color: #92400e; }
        .gs-badge-red   { background: #fee2e2; color: #991b1b; }
        .gs-badge-blue  { background: #dbeafe; color: #1e3a8a; }
        .gs-chip {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            font-size: 0.78rem;
            font-family: 'SFMono-Regular', 'Menlo', 'Consolas', monospace;
            color: #1e293b;
            margin: 0.18rem 0.25rem 0.18rem 0;
        }
        .gs-tracker-row {
            font-family: 'SFMono-Regular', 'Menlo', 'Consolas', monospace;
            font-size: 0.92rem;
            padding: 0.45rem 0.8rem;
            margin: 0.2rem 0;
            border-radius: 6px;
            background: #0f172a;
            color: #e2e8f0;
            border-left: 4px solid #334155;
        }
        .gs-tracker-row.pending { opacity: 0.55; border-left-color: #475569; }
        .gs-tracker-row.processing { border-left-color: #3b82f6; background: #0f1b3b; }
        .gs-tracker-row.success { border-left-color: #22c55e; background: #0a1f17; }
        .gs-tracker-row.error { border-left-color: #ef4444; background: #2a0f10; }
        .gs-tracker-row.skipped { opacity: 0.4; border-left-color: #71717a; }
        .gs-metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            min-height: 132px;
            height: 100%;
            box-sizing: border-box;
        }
        .gs-metric-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.035em;
            line-height: 1.35;
            margin-bottom: 0.55rem;
            word-wrap: break-word;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .gs-metric-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.5;
            word-wrap: break-word;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .gs-metric-sub {
            font-size: 0.82rem;
            color: #475569;
            margin-top: 0.45rem;
            line-height: 1.4;
            word-wrap: break-word;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        section[data-testid="stSidebar"] .stButton button {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _gap_color_class(score: float) -> str:
    if score >= 0.7:
        return "gs-badge-green"
    if score >= 0.4:
        return "gs-badge-amber"
    return "gs-badge-red"


def _gap_label(score: float) -> str:
    if score >= 0.7:
        return "High white space"
    if score >= 0.4:
        return "Moderate gap"
    return "Highly overlapped"


def _render_commercial_metric_card(label: str, value: str, sub: str = "") -> None:
    label_safe = html.escape(label)
    value_safe = html.escape(value or "—")
    sub_html = ""
    if sub:
        sub_html = f'<div class="gs-metric-sub">{html.escape(sub)}</div>'
    st.markdown(
        f"""
        <div class="gs-metric-card">
            <div class="gs-metric-label">{label_safe}</div>
            <div class="gs-metric-value">{value_safe}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _check_backend_health() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        return None
    return None


def _call_analyze(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes, "application/pdf")}

    try:
        response = requests.post(
            ANALYZE_ENDPOINT,
            files=files,
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach the GapSight backend at `{BACKEND_URL}`.\n\n"
            "Make sure uvicorn is running. From `gapsight-backend/`:\n\n"
            "```powershell\n"
            ".\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload "
            "--host 0.0.0.0 --port 8000\n"
            "```"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"The pipeline did not respond within {REQUEST_TIMEOUT_S}s. "
            "Gemini may be rate-limited or slow — please retry."
        ) from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text or "(no body)"
        raise RuntimeError(
            f"Backend returned HTTP {response.status_code}.\n\n```\n{detail}\n```"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Backend returned a non-JSON response.") from exc


def _render_tracker_row(state: str, agent_id: str, role: str, message: str) -> str:
    icons = {
        "pending": "⚪",
        "processing": "🔵",
        "success": "🟢",
        "error": "🔴",
        "skipped": "⚫",
    }
    labels = {
        "pending": "Pending",
        "processing": "Processing",
        "success": "Success",
        "error": "Failed",
        "skipped": "Skipped",
    }
    icon = icons.get(state, "⚪")
    label = labels.get(state, "Pending")
    return (
        f'<div class="gs-tracker-row {state}">'
        f"{icon} <b>[{agent_id}]</b> <i>{role}</i> &middot; "
        f"<b>{label}:</b> {message}"
        f"</div>"
    )


with st.sidebar:
    st.markdown("### Upload Research Paper")
    st.caption(
        "Upload an academic PDF (arXiv preprint, conference paper, technical "
        "report, etc.). GapSight will run a 5-agent comprehensive analysis to "
        "identify patentable white space AND commercialization potential."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button(
        "Run Comprehensive Analysis",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state.pop("filename", None)
        st.session_state.pop("elapsed", None)
        st.rerun()

    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        """
        1. **Agent 1** — Claim Extractor  
        2. **Agent 2** — Prior Art Searcher  
        3. **Agent 3** — Gap Identifier  
        4. **Agent 4** — Claim Drafter *(high-gap only)*  
        5. **Agent 5** — Commercial Strategist
        """
    )

    st.divider()
    st.markdown("### Backend Status")
    health = _check_backend_health()
    if health is None:
        st.error("Backend offline at `localhost:8000`.")
    else:
        st.success(
            f"Online — `{health.get('service', 'service')}` "
            f"({health.get('env', 'env')})"
        )


st.markdown('<div class="gs-header">GapSight</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gs-subheader">Research-to-Patent Gap Finder &mdash; '
    "a 5-agent AI pipeline that extracts novel technical claims, surfaces "
    "prior art via semantic retrieval, scores patentability gaps, drafts "
    "provisional patent language, and analyzes the commercial monetization "
    "landscape end-to-end.</div>",
    unsafe_allow_html=True,
)


if analyze_clicked and uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    if not file_bytes:
        st.error("The uploaded file is empty.")
        st.stop()

    st.markdown(
        '<div class="gs-section-title">Orchestrator Live Telemetry</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="gs-section-caption">Real-time status of each agent as '
        "the FastAPI orchestrator executes the pipeline.</div>",
        unsafe_allow_html=True,
    )

    placeholders = [st.empty() for _ in AGENTS_META]
    for ph, meta in zip(placeholders, AGENTS_META):
        ph.markdown(
            _render_tracker_row("pending", meta["id"], meta["role"], "Awaiting orchestrator..."),
            unsafe_allow_html=True,
        )

    result_container: Dict[str, Any] = {"status": "running", "data": None, "error": None}

    def _run_api() -> None:
        try:
            data = _call_analyze(file_bytes, filename)
            result_container["status"] = "success"
            result_container["data"] = data
        except Exception as exc:
            result_container["status"] = "error"
            result_container["error"] = str(exc)

    thread = threading.Thread(target=_run_api, daemon=True)
    thread.start()

    start_time = time.perf_counter()

    for i, meta in enumerate(AGENTS_META):
        placeholders[i].markdown(
            _render_tracker_row("processing", meta["id"], meta["role"], meta["processing"]),
            unsafe_allow_html=True,
        )

        deadline = time.perf_counter() + AGENT_DURATIONS_S[i]
        while time.perf_counter() < deadline and thread.is_alive():
            time.sleep(0.25)

        if result_container["status"] == "error":
            placeholders[i].markdown(
                _render_tracker_row("error", meta["id"], meta["role"], "Backend error (see below)."),
                unsafe_allow_html=True,
            )
            for j in range(i + 1, len(AGENTS_META)):
                m = AGENTS_META[j]
                placeholders[j].markdown(
                    _render_tracker_row("skipped", m["id"], m["role"], "Pipeline aborted."),
                    unsafe_allow_html=True,
                )
            break

        success_msg = meta["success_template"]
        if (
            meta["id"] == "Agent 1"
            and result_container["status"] == "success"
            and isinstance(result_container["data"], dict)
        ):
            count = len(result_container["data"].get("extracted_claims", []) or [])
            success_msg = success_msg.format(count=count)
        else:
            success_msg = success_msg.replace("{count}", "")

        placeholders[i].markdown(
            _render_tracker_row("success", meta["id"], meta["role"], success_msg),
            unsafe_allow_html=True,
        )

        if not thread.is_alive() and result_container["status"] == "success":
            for j in range(i + 1, len(AGENTS_META)):
                m = AGENTS_META[j]
                msg = m["success_template"].replace("{count}", "")
                placeholders[j].markdown(
                    _render_tracker_row("success", m["id"], m["role"], msg),
                    unsafe_allow_html=True,
                )
            break

    thread.join(timeout=REQUEST_TIMEOUT_S)
    elapsed = time.perf_counter() - start_time

    if result_container["status"] == "error":
        st.error(result_container["error"])
        st.stop()
    if result_container["status"] != "success" or result_container["data"] is None:
        st.error("Backend did not return a response in time.")
        st.stop()

    st.session_state["result"] = result_container["data"]
    st.session_state["filename"] = filename
    st.session_state["elapsed"] = elapsed
    st.success(f"Pipeline complete in **{elapsed:.1f}s**.")


result: Optional[Dict[str, Any]] = st.session_state.get("result")
filename: Optional[str] = st.session_state.get("filename")
elapsed: Optional[float] = st.session_state.get("elapsed")

if result is None:
    st.info(
        "Upload a research PDF in the sidebar and click "
        "**Run Comprehensive Analysis** to begin."
    )

    with st.expander("How GapSight works", expanded=False):
        st.markdown(
            """
            GapSight runs a five-agent orchestration over your uploaded paper:

            1. **Claim Extractor (Agent 1).** Distills the paper's novel
               technical contributions into structured claims with categories
               and search keywords.
            2. **Prior Art Searcher (Agent 2).** Embeds each claim and runs
               cosine-similarity search over a local vector DB of patent
               abstracts to surface relevant prior art.
            3. **Gap Identifier (Agent 3).** LLM-scores how much patentable
               *white space* remains around each claim, with a numeric
               `gap_score` and a written rationale.
            4. **Claim Drafter (Agent 4).** Drafts formal provisional-patent
               language for the high-gap claims only (`gap_score >= 0.7`).
            5. **Commercial Strategist (Agent 5).** Analyzes the commercial
               monetization landscape: market size, venture potential,
               funding-vehicle matches, and the competitor patent map.

            The final response includes extracted claims, scored gaps, a
            Markdown patent draft, and a venture commercialization brief.
            """
        )
    st.stop()


extracted_claims: List[Dict[str, Any]] = result.get("extracted_claims", []) or []
patent_gaps: List[Dict[str, Any]] = result.get("patent_gaps", []) or []
drafted_claims: str = result.get("drafted_claims", "") or ""
characters_extracted: int = int(result.get("characters_extracted", 0) or 0)
commercialization: Dict[str, Any] = result.get("commercialization", {}) or {}
high_gap_count = sum(
    1 for g in patent_gaps if float(g.get("gap_score", 0.0) or 0.0) >= 0.7
)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Source PDF", filename or "—")
with col_m2:
    st.metric("Characters parsed", f"{characters_extracted:,}")
with col_m3:
    st.metric("Claims extracted", len(extracted_claims))
with col_m4:
    st.metric(
        "High-gap claims",
        f"{high_gap_count}/{len(patent_gaps)}" if patent_gaps else "—",
        help="Claims with gap_score >= 0.7 (genuine white space).",
    )

if elapsed is not None:
    st.caption(f"Last run: {elapsed:.1f}s end-to-end.")

st.divider()


st.markdown(
    '<div class="gs-section-title">1 &nbsp;&middot;&nbsp; Extracted Claims</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gs-section-caption">Novel technical contributions distilled '
    "from the paper by <b>Agent 1 (Claim Extractor)</b>.</div>",
    unsafe_allow_html=True,
)

if not extracted_claims:
    st.warning("No claims were extracted from this paper.")
else:
    for claim in extracted_claims:
        claim_id = str(claim.get("claim_id", "?"))
        category = str(claim.get("category", "Uncategorized"))
        description = str(claim.get("technical_description", ""))
        keywords = claim.get("keywords", []) or []
        preview = description[:90] + ("..." if len(description) > 90 else "")

        with st.expander(f"**{claim_id}** &middot; {category} — {preview}"):
            st.markdown("**Technical Description**")
            st.write(description)

            if keywords:
                st.markdown("**Keywords**")
                chips_html = "".join(
                    f'<span class="gs-chip">{kw}</span>' for kw in keywords
                )
                st.markdown(chips_html, unsafe_allow_html=True)

st.divider()


st.markdown(
    '<div class="gs-section-title">2 &nbsp;&middot;&nbsp; Prior Art &amp; Gap Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gs-section-caption"><b>Agent 2</b> retrieved similar patents '
    "via semantic vector search; <b>Agent 3</b> scored the patentability gap "
    "of each claim against that prior art.</div>",
    unsafe_allow_html=True,
)

if not patent_gaps:
    st.warning("No gap analysis available.")
else:
    sorted_gaps = sorted(
        patent_gaps,
        key=lambda g: float(g.get("gap_score", 0.0) or 0.0),
        reverse=True,
    )
    for gap in sorted_gaps:
        score = float(gap.get("gap_score", 0.0) or 0.0)
        score = max(0.0, min(1.0, score))
        badge_class = _gap_color_class(score)
        label = _gap_label(score)
        claim_id = str(gap.get("claim_id", "?"))
        rationale = str(gap.get("rationale", ""))
        prior_art = gap.get("closest_prior_art", []) or []

        with st.container(border=True):
            col_left, col_right = st.columns([1, 3], gap="large")

            with col_left:
                st.metric(label=f"Claim {claim_id}", value=f"{score:.2f}")
                st.progress(score)
                st.markdown(
                    f'<span class="gs-badge {badge_class}">{label}</span>',
                    unsafe_allow_html=True,
                )

            with col_right:
                st.markdown("**Rationale**")
                st.write(rationale or "_(no rationale provided)_")

                if prior_art:
                    st.markdown("**Closest Prior Art**")
                    chips_html = "".join(
                        f'<span class="gs-chip">{pid}</span>' for pid in prior_art
                    )
                    st.markdown(chips_html, unsafe_allow_html=True)

st.divider()


st.markdown(
    '<div class="gs-section-title">3 &nbsp;&middot;&nbsp; Provisional Patent Draft</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gs-section-caption"><b>Agent 4 (Claim Drafter)</b> generated '
    "formal patent language for the high-gap claims only. Always review with "
    "a licensed IP attorney before filing.</div>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    if drafted_claims.startswith("**[STUB"):
        st.warning(
            "Agent 4 returned a placeholder draft (see below). This usually "
            "means a missing API key or an upstream Gemini failure — check "
            "the backend logs."
        )
    if drafted_claims:
        st.markdown(drafted_claims)
    else:
        st.markdown("_(no draft available)_")

st.divider()


st.markdown(
    '<div class="gs-section-title">'
    "4 &nbsp;&middot;&nbsp; 💡 Venture Commercialization &amp; IP Monetization Analytics"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gs-section-caption"><b>Agent 5 (Commercial Strategist)</b> '
    "assessed monetization vectors, market size, venture funding routes, and "
    "the competitive patent landscape around the high-gap claims.</div>",
    unsafe_allow_html=True,
)

if not commercialization:
    st.warning("No commercialization analysis available.")
else:
    score = int(commercialization.get("commercialization_score", 0) or 0)
    startup_potential = str(commercialization.get("startup_potential", "—"))
    market_size = str(commercialization.get("market_size", "—"))
    roi_ratio = str(commercialization.get("roi_ratio", "—"))
    funding_vehicles = commercialization.get("funding_vehicles", []) or []
    competitor_map = commercialization.get("competitor_map", []) or []

    col_c1, col_c2, col_c3 = st.columns(3, gap="medium")
    with col_c1:
        _render_commercial_metric_card(
            "Commercialization Score",
            f"{score}/100",
            startup_potential,
        )
    with col_c2:
        _render_commercial_metric_card(
            "Core Addressable Market",
            market_size,
        )
    with col_c3:
        _render_commercial_metric_card(
            "ROI Estimation Ratio",
            roi_ratio,
        )

    st.markdown("")

    with st.container(border=True):
        st.markdown("**Recommended Funding Vehicles**")
        if funding_vehicles:
            for v in funding_vehicles:
                st.markdown(f"- {v}")
        else:
            st.markdown("_(no funding-vehicle suggestions available)_")

    with st.container(border=True):
        st.markdown("**Competitor Space Map**")
        if competitor_map:
            header = (
                "| Active Corporate Patent Holder | Intersecting Core Tech | "
                "Threat Level | White-Space Advantage |"
            )
            divider = "|---|---|---|---|"
            rows = []
            for entry in competitor_map:
                holder = str(entry.get("corporate_holder", "—")).replace("|", "/")
                tech = str(entry.get("intersecting_tech", "—")).replace("|", "/")
                threat = str(entry.get("threat_level", "—")).replace("|", "/")
                advantage = (
                    str(entry.get("whitespace_advantage", "—")).replace("|", "/")
                )
                rows.append(f"| {holder} | {tech} | **{threat}** | {advantage} |")
            st.markdown("\n".join([header, divider, *rows]))
        else:
            st.markdown("_(no competitor map available)_")

st.divider()


col_d1, col_d2 = st.columns(2)
base_name = (filename or "gapsight").rsplit(".", 1)[0]
with col_d1:
    st.download_button(
        "Download draft as Markdown",
        data=drafted_claims or "(empty)",
        file_name=f"{base_name}-provisional-claims.md",
        mime="text/markdown",
        use_container_width=True,
    )
with col_d2:
    st.download_button(
        "Download full analysis as JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"{base_name}-analysis.json",
        mime="application/json",
        use_container_width=True,
    )
