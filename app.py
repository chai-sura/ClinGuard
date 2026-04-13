"""
ClinGuard — Streamlit dashboard.

Ties together the full 3-agent pipeline, eval harness, and SQLite logger
into a single UI where users can paste AE reports and see live results.
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Allow "from graph.graph import ..." style imports from the clinguard package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "clinguard"))

from db.logger import get_recent_runs, init_db, log_decision, log_eval
from eval.judge import evaluate_decision
from graph.graph import build_graph, run_pipeline

load_dotenv()

# ------------------------------------------------------------------ #
# Page config — must be the first Streamlit call                      #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="ClinGuard",
    page_icon="🏥",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Custom CSS                                                          #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
/*
 * ClinGuard color palette:
 *   Background : #000000  — pure black
 *   White      : #ffffff  — all primary text and headings
 *   Blue       : #2563eb  — accents, badges, borders, dots
 *   Muted      : #94a3b8  — subtitles, descriptions
 * Status colors (functional only): red / amber / green for decisions & grades
 */

/* Global */
html, body, [data-testid="stAppViewContainer"], [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #000000 !important;
    color: #ffffff;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    background-color: #000000 !important;
}
[data-testid="stAppViewContainer"] > .main { background-color: #000000 !important; }
[data-testid="stHeader"] { background-color: #000000 !important; }

/* Cards */
.card {
    background: #0d0d0d;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Section headers */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.section-subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: -0.75rem;
    margin-bottom: 1rem;
}

/* Decision banners */
.banner-escalate { background: #450a0a; color: #fca5a5; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; border: 1px solid #991b1b; }
.banner-monitor  { background: #451a03; color: #fcd34d; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; border: 1px solid #92400e; }
.banner-dismiss  { background: #052e16; color: #86efac; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; border: 1px solid #166534; }
.banner-decision { font-size: 2rem; font-weight: 800; letter-spacing: 0.1em; }
.banner-meta     { font-size: 0.95rem; margin-top: 0.4rem; font-weight: 500; opacity: 0.9; }

/* Pills */
.pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.pill-blue   { background: #1e3a8a; color: #93c5fd; }
.pill-gray   { background: #1e293b; color: #cbd5e1; }
.pill-green  { background: #052e16; color: #86efac; }
.pill-orange { background: #431407; color: #fdba74; }
.pill-red    { background: #450a0a; color: #fca5a5; }

/* Grade rows */
.grade-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e293b;
}
.grade-row:last-child { border-bottom: none; }
.grade-symptom { font-size: 0.9rem; color: #ffffff; text-transform: capitalize; }

/* Reasoning trace */
.trace-step {
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    padding: 0.6rem 0.8rem 0.6rem 1.2rem;
    background: #0d0d0d;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.4rem;
    white-space: pre-wrap;
    word-break: break-word;
    color: #cbd5e1;
}

/* Pipeline timeline */
.timeline-step { display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; background: #2563eb; }
.agent-name { font-size: 0.9rem; font-weight: 500; color: #ffffff; flex: 1; }

/* Eval score rows */
.eval-row   { margin-bottom: 0.8rem; }
.eval-label { font-size: 0.85rem; font-weight: 600; color: #ffffff; }
.eval-desc  { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem; }

/* Overall score */
.overall-score { font-size: 3rem; font-weight: 800; text-align: center; line-height: 1; color: #ffffff; }
.overall-label { text-align: center; font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; margin-bottom: 1rem; }

/* Misc */
.symptom-pills { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.3rem; }
.ai-badge { display: inline-block; background: #1e293b; color: #94a3b8; font-size: 0.72rem; padding: 0.15rem 0.6rem; border-radius: 999px; margin-bottom: 0.75rem; }
.memo-text { font-size: 0.9rem; line-height: 1.8; color: #e2e8f0; }
.header-pills { display: flex; gap: 0.5rem; margin-top: 0.5rem; margin-bottom: 0.25rem; }

/* Streamlit native widget overrides */
[data-testid="stTextArea"] textarea {
    background-color: #0d0d0d !important;
    color: #ffffff !important;
    border: 1px solid #1e293b !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #94a3b8 !important; }
label[data-testid="stWidgetLabel"] p { color: #ffffff !important; }
[data-testid="metric-container"] { background: #0d0d0d; border: 1px solid #1e293b; border-radius: 8px; padding: 0.5rem; }
[data-testid="metric-container"] label { color: #94a3b8 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; }
[data-testid="stDataFrame"] { background: #0d0d0d !important; }
div[data-testid="stAlert"] { background: #0d0d0d !important; border-color: #1e293b !important; }

/* Sidebar */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background-color: #0d0d0d !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span { color: #e2e8f0 !important; }
.sidebar-logo    { font-size: 1.4rem; font-weight: 800; color: #ffffff; }
.sidebar-version { display: inline-block; background: #1e3a8a; color: #93c5fd; font-size: 0.72rem; padding: 0.15rem 0.6rem; border-radius: 999px; margin-top: 0.25rem; }
.sidebar-disclaimer { font-size: 0.72rem; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 0.75rem; margin-top: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #
_SAMPLE_REPORT = (
    "Patient PT-0042 reported severe chest pain and shortness of breath "
    "starting 3 days after dose 2. BP was 88/54, HR 115. "
    "Patient was hospitalized immediately. No prior cardiac history."
)

_EVAL_META = {
    "grounding":          ("Grounding",          "Claims traceable to source report"),
    "completeness":       ("Completeness",        "All symptoms considered"),
    "hallucination_risk": ("Hallucination Risk",  "Lower is better — fabricated facts"),
    "reasoning_depth":    ("Reasoning Depth",     "CTCAE lookup steps performed"),
    "agent_agreement":    ("Agent Agreement",     "Classifier ↔ Safety Officer coherence"),
}


@st.cache_resource
def get_app():
    """Initialise the database and compile the LangGraph pipeline once."""
    init_db()
    return build_graph()


get_app()

# ------------------------------------------------------------------ #
# Session state                                                       #
# ------------------------------------------------------------------ #
if "sample_loaded" not in st.session_state:
    st.session_state["sample_loaded"] = False
if "result" not in st.session_state:
    st.session_state["result"] = None
if "eval_scores" not in st.session_state:
    st.session_state["eval_scores"] = None

# ------------------------------------------------------------------ #
# Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏥 ClinGuard</div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-version">v1.0 | CTCAE v6.0</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**About**")
    st.write(
        "ClinGuard is a 3-agent LangGraph system that autonomously classifies "
        "clinical trial adverse events against CTCAE v6.0 severity grades using "
        "ReAct reasoning loops and an independent LLM-as-judge eval harness."
    )

    st.markdown("**Agent Pipeline**")
    st.markdown("1. **Extractor** — parses patient ID, symptoms, vitals, and timeline from raw text")
    st.markdown("2. **Classifier** — ReAct loop with CTCAE v6.0 lookup to assign grade 1–5")
    st.markdown("3. **Safety Officer** — reviews grading, applies protocol rules, writes memo")

    st.markdown("**Eval Dimensions**")
    st.markdown("- **Grounding** — are memo claims traceable to the report?")
    st.markdown("- **Completeness** — were all symptoms addressed?")
    st.markdown("- **Hallucination Risk** — fabricated facts (lower = better)")
    st.markdown("- **Reasoning Depth** — quality of ReAct steps")
    st.markdown("- **Agent Agreement** — classifier and officer logically aligned?")

    st.markdown(
        '<div class="sidebar-disclaimer">For portfolio demonstration only.<br>'
        'Not for clinical use.</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ #
# Header                                                              #
# ------------------------------------------------------------------ #
st.markdown(
    '<h1 style="font-size:2.5rem;font-weight:800;color:#ffffff;margin-bottom:0;">ClinGuard</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#94a3b8;font-size:1rem;margin-top:0.25rem;margin-bottom:0.5rem;">'
    'Autonomous Clinical Trial Adverse Event Monitor</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="header-pills">'
    '<span class="pill pill-blue">CTCAE v6.0</span>'
    '<span class="pill pill-blue">3-Agent Pipeline</span>'
    '<span class="pill pill-blue">LLM-as-Judge Eval</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

# ------------------------------------------------------------------ #
# Main layout                                                         #
# ------------------------------------------------------------------ #
left_col, right_col = st.columns([6, 4])

# ================================================================== #
# LEFT COLUMN                                                         #
# ================================================================== #
with left_col:

    # INPUT CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Adverse Event Report</div>'
        '<p style="font-size:0.82rem;color:#6b7280;margin-top:-0.5rem;margin-bottom:0.75rem;">'
        'Paste the raw report text from the clinical trial system</p>',
        unsafe_allow_html=True,
    )

    if st.session_state["sample_loaded"]:
        default_text = _SAMPLE_REPORT
    else:
        default_text = ""

    report_text = st.text_area(
        label="report_input",
        label_visibility="collapsed",
        value=default_text,
        height=180,
        placeholder="Patient PT-XXXX reported...",
    )

    btn_col1, btn_col2, _ = st.columns([2, 2, 3])
    with btn_col1:
        analyze_clicked = st.button("Analyze Report", type="primary", use_container_width=True)
    with btn_col2:
        sample_clicked = st.button("Load Sample", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if sample_clicked:
        st.session_state["sample_loaded"] = True
        st.rerun()

    if analyze_clicked and report_text.strip():
        with st.spinner("Running agent pipeline..."):
            result = run_pipeline(report_text)
            eval_scores = evaluate_decision(result)
            run_id = log_decision(result)
            log_eval(run_id, eval_scores)
            st.session_state["result"] = result
            st.session_state["eval_scores"] = eval_scores

    elif analyze_clicked:
        st.warning("Please paste an adverse event report before analyzing.")

    # RESULTS — shown when state has a result
    result = st.session_state.get("result")
    if result:
        decision = result.get("decision", "")
        escalation_grade = result.get("escalation_grade", "—")
        confidence_pct = f"{result.get('confidence', 0):.0%}"
        risk_pct = f"{result.get('risk_score', 0):.0%}"

        # 1. DECISION BANNER
        banner_class = {
            "escalate": "banner-escalate",
            "monitor":  "banner-monitor",
            "dismiss":  "banner-dismiss",
        }.get(decision, "banner-dismiss")

        icon = {"escalate": "⚠️", "monitor": "👁", "dismiss": "✅"}.get(decision, "")

        st.markdown(
            f'<div class="{banner_class}">'
            f'<div class="banner-decision">{icon} {decision.upper()}</div>'
            f'<div class="banner-meta">'
            f'Escalation Grade {escalation_grade} &nbsp;|&nbsp; '
            f'Confidence {confidence_pct} &nbsp;|&nbsp; '
            f'Risk Score {risk_pct}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # 2. PATIENT SUMMARY CARD
        symptoms = result.get("symptoms") or []
        symptom_pills = "".join(
            f'<span class="pill pill-gray" style="margin:2px;">{s}</span>'
            for s in symptoms
        )
        st.markdown(
            f'<div class="card">'
            f'<div class="section-title">Patient Summary</div>',
            unsafe_allow_html=True,
        )
        ps_left, ps_right = st.columns(2)
        with ps_left:
            st.markdown(
                f'<div style="font-size:0.8rem;color:#94a3b8;">Patient ID</div>'
                f'<div style="font-weight:600;color:#ffffff;margin-bottom:0.75rem;">'
                f'{result.get("patient_id", "—")}</div>'
                f'<div style="font-size:0.8rem;color:#94a3b8;">Symptoms</div>'
                f'<div class="symptom-pills">{symptom_pills}</div>',
                unsafe_allow_html=True,
            )
        with ps_right:
            st.markdown(
                f'<div style="font-size:0.8rem;color:#94a3b8;">Timeline</div>'
                f'<div style="font-weight:500;color:#e2e8f0;margin-bottom:0.75rem;">'
                f'{result.get("timeline") or "—"}</div>'
                f'<div style="font-size:0.8rem;color:#94a3b8;">Vitals</div>'
                f'<div style="font-weight:500;color:#e2e8f0;">'
                f'{result.get("vitals") or "—"}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. CTCAE GRADES CARD
        ctcae_grades = result.get("ctcae_grades") or {}
        st.markdown(
            '<div class="card">'
            '<div class="section-title">CTCAE Grade Assignment</div>'
            '<div class="section-subtitle">Per CTCAE v6.0 — National Cancer Institute</div>',
            unsafe_allow_html=True,
        )
        if ctcae_grades:
            for symptom, grade in ctcae_grades.items():
                if grade >= 4:
                    pill_cls = "pill-red"
                elif grade == 3:
                    pill_cls = "pill-orange"
                else:
                    pill_cls = "pill-green"
                st.markdown(
                    f'<div class="grade-row">'
                    f'<span class="grade-symptom">{symptom}</span>'
                    f'<span class="pill {pill_cls}">Grade {grade}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No grades assigned.")
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. DECISION MEMO CARD
        st.markdown(
            '<div class="card">'
            '<div class="section-title">Safety Officer Decision Memo</div>'
            '<span class="ai-badge">AI Generated | Requires Human Review</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="memo-text">{result.get("decision_memo", "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. REASONING TRACE CARD
        st.markdown(
            '<div class="card">'
            '<div class="section-title">Agent Reasoning Trace</div>'
            '<div class="section-subtitle">Step-by-step ReAct loop execution</div>',
            unsafe_allow_html=True,
        )
        for step in result.get("reasoning_trace") or []:
            step_lower = step.lower()
            if step_lower.startswith("extractor"):
                cls = "trace-extractor"
            elif step_lower.startswith("classifier"):
                cls = "trace-classifier"
            elif step_lower.startswith("safety officer"):
                cls = "trace-safety-officer"
            else:
                cls = "trace-other"
            safe_step = step.replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="trace-step {cls}">{safe_step}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================== #
# RIGHT COLUMN                                                        #
# ================================================================== #
with right_col:
    eval_scores = st.session_state.get("eval_scores")
    result = st.session_state.get("result")

    # EVAL SCORES CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Eval Harness Results</div>'
        '<div class="section-subtitle">LLM-as-judge scoring across 5 dimensions</div>',
        unsafe_allow_html=True,
    )

    if eval_scores:
        overall = eval_scores.get("overall_score", 0)

        st.markdown(
            f'<div class="overall-score">{overall:.2f}</div>'
            f'<div class="overall-label">Overall Quality Score</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        for key, (label, desc) in _EVAL_META.items():
            score = eval_scores.get(key, 0)
            # Invert hallucination_risk for the progress bar (lower risk = better)
            bar_val = (1 - score) if key == "hallucination_risk" else score
            st.markdown(
                f'<div class="eval-row">'
                f'<div class="eval-label">{label} <span style="float:right;font-weight:600;">{score:.2f}</span></div>'
                f'<div class="eval-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(bar_val))
    else:
        st.markdown(
            '<div style="color:#9ca3af;font-size:0.9rem;text-align:center;padding:1rem 0;">'
            'Run a report to see eval scores.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # PIPELINE STATUS CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pipeline Status</div>', unsafe_allow_html=True)

    agents = [
        ("Extractor",      "Completed", ""),
        ("Classifier",     "Completed",
         f'  (+{result.get("retry_count", 0)} retries)' if result and result.get("retry_count") else ""),
        ("Safety Officer", "Completed", ""),
    ]

    if result:
        for name, status, extra in agents:
            st.markdown(
                f'<div class="timeline-step">'
                f'<div class="dot"></div>'
                f'<span class="agent-name">{name}</span>'
                f'<span class="pill pill-green">{status}</span>'
                f'<span style="font-size:0.78rem;color:#6b7280;">{extra}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        latency_s = result.get("latency_ms", 0) / 1000
        st.markdown(
            f'<div style="margin-top:0.75rem;font-size:0.85rem;color:#94a3b8;">'
            f'Total: <strong style="color:#ffffff;">{latency_s:.1f}s</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#9ca3af;font-size:0.9rem;text-align:center;padding:0.5rem 0;">'
            'Awaiting analysis.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # RECENT RUNS CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Recent Decisions</div>'
        '<div class="section-subtitle">Last 10 runs logged to SQLite</div>',
        unsafe_allow_html=True,
    )
    runs = get_recent_runs(10)
    if runs:
        display_cols = {
            "patient_id":      "Patient",
            "decision":        "Decision",
            "escalation_grade":"Grade",
            "confidence":      "Confidence",
            "overall_score":   "Score",
            "created_at":      "Time",
        }
        filtered = [
            {display_cols[k]: r.get(k) for k in display_cols}
            for r in runs
        ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div style="color:#9ca3af;font-size:0.9rem;text-align:center;padding:0.5rem 0;">'
            'No runs logged yet.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
