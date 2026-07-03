"""
ClinGuard — Streamlit UI.

Shows the CURRENT system: a linear LangGraph pipeline
    Extractor → Grader (Claude) → deterministic Rules → cross-model Verifier (GPT-4o-mini) → Decision
The decision is made by the deterministic protocol-rules engine, NOT by an LLM.
The verifier is an independent second model that only gates confidence.

Two modes:
  • Demo (default): replays frozen real traces from clinguard/data/demo_traces.json.
    No API calls, instant, self-contained.
  • Live: runs the real pipeline on a pasted report (length-capped), persists to
    SQLite, and enqueues flagged runs for human review.

No LLM-as-judge here — the headline numbers shown are the ground-truth synthetic
eval results from RESULTS.md, not a self-score.
"""

import html
import json
import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "clinguard"))

from db.logger import (
    get_recent_runs,
    get_review_queue,
    init_db,
    log_decision,
    record_human_review,
)
from graph.graph import run_pipeline
from tools.ctcae_lookup import lookup_ctcae_grade
from tools.protocol_rules import check_protocol_rule

load_dotenv()

# Stage-8 input length cap — a single AE report is short; block oversized pastes
# before they reach any paid model call.
MAX_REPORT_CHARS = 4000

# Frozen ground-truth eval numbers (see RESULTS.md). Synthetic data only.
RESULTS = {
    "standard_grade": "81.6%",
    "standard_escalation": "94.4%",
    "hard_grade": "84.0%",
    "agreement": "94%",
    "review": "11%",
    "hard_breakdown": "boundary 57% · missing-vital 100% · rare-term 100% · multi-symptom 83%",
}

_DEMO_PATH = os.path.join(os.path.dirname(__file__), "clinguard", "data", "demo_traces.json")

st.set_page_config(page_title="ClinGuard", page_icon="🏥", layout="wide")

# ------------------------------------------------------------------ #
# Styling                                                             #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #000000 !important; color: #ffffff;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stHeader"] { background-color: #000000 !important; }

.card { background:#0d0d0d; border:1px solid #1e293b; border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:1rem; }
.stage-card { background:#0d0d0d; border:1px solid #1e293b; border-left:3px solid #2563eb; border-radius:10px; padding:1rem 1.25rem; margin-bottom:0.5rem; }
.stage-head { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.6rem; }
.stage-num { display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; background:#1e3a8a; color:#93c5fd; font-size:0.75rem; font-weight:700; flex-shrink:0; }
.stage-title { font-size:0.95rem; font-weight:700; color:#ffffff; }
.stage-model { margin-left:auto; font-size:0.7rem; color:#64748b; }
.flow-arrow { text-align:center; color:#334155; font-size:0.9rem; margin:-0.15rem 0 0.35rem 0; }

.section-title { font-size:1rem; font-weight:700; color:#ffffff; border-bottom:1px solid #1e293b; padding-bottom:0.5rem; margin-bottom:0.9rem; }
.section-subtitle { font-size:0.78rem; color:#94a3b8; margin-top:-0.6rem; margin-bottom:0.9rem; }
.muted { color:#94a3b8; font-size:0.82rem; }
.kv-label { font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em; }
.kv-val { font-weight:600; color:#ffffff; margin-bottom:0.6rem; }

.banner-escalate { background:#450a0a; color:#fca5a5; border:1px solid #991b1b; }
.banner-monitor  { background:#451a03; color:#fcd34d; border:1px solid #92400e; }
.banner-dismiss  { background:#052e16; color:#86efac; border:1px solid #166534; }
.banner { border-radius:12px; padding:1.25rem; text-align:center; margin-bottom:0.75rem; }
.banner-decision { font-size:2rem; font-weight:800; letter-spacing:0.1em; }
.banner-meta { font-size:0.9rem; margin-top:0.35rem; font-weight:500; opacity:0.92; }

.pill { display:inline-block; padding:0.18rem 0.65rem; border-radius:999px; font-size:0.78rem; font-weight:600; }
.pill-blue{background:#1e3a8a;color:#93c5fd;} .pill-gray{background:#1e293b;color:#cbd5e1;}
.pill-green{background:#052e16;color:#86efac;} .pill-orange{background:#431407;color:#fdba74;}
.pill-red{background:#450a0a;color:#fca5a5;} .pill-amber{background:#451a03;color:#fcd34d;}

.grade-block { border-bottom:1px solid #1e293b; padding:0.6rem 0; }
.grade-block:last-child { border-bottom:none; }
.grade-top { display:flex; justify-content:space-between; align-items:center; }
.grade-symptom { font-size:0.9rem; color:#ffffff; text-transform:capitalize; font-weight:600; }
.ctcae-term { font-size:0.72rem; color:#64748b; margin-top:0.15rem; }
.ctcae-def { font-family:"SF Mono","Fira Code",monospace; font-size:0.75rem; line-height:1.5; color:#cbd5e1;
             background:#111827; border-left:2px solid #334155; border-radius:0 6px 6px 0; padding:0.4rem 0.6rem; margin-top:0.35rem; white-space:pre-wrap; }
.rule-fired { background:#111827; border:1px solid #1e293b; border-radius:8px; padding:0.7rem 0.9rem; font-size:0.85rem; color:#e2e8f0; }
.trace-step { font-family:"SF Mono","Fira Code",monospace; font-size:0.75rem; line-height:1.7; padding:0.45rem 0.7rem; background:#0d0d0d; border-left:3px solid #2563eb; border-radius:0 6px 6px 0; margin-bottom:0.35rem; white-space:pre-wrap; color:#cbd5e1; }
.memo-text { font-size:0.88rem; line-height:1.7; color:#e2e8f0; }
.det-note { font-size:0.75rem; color:#93c5fd; background:#0b1220; border:1px dashed #1e3a8a; border-radius:8px; padding:0.5rem 0.7rem; margin-top:0.6rem; }

.metric-big { font-size:1.8rem; font-weight:800; color:#ffffff; line-height:1; }
.metric-cap { font-size:0.72rem; color:#94a3b8; margin-top:0.2rem; }

section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background-color:#0d0d0d !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] span { color:#e2e8f0 !important; }
[data-testid="stTextArea"] textarea { background-color:#0d0d0d !important; color:#ffffff !important; border:1px solid #1e293b !important; }
[data-testid="stDataFrame"] { background:#0d0d0d !important; }
.sidebar-logo { font-size:1.4rem; font-weight:800; color:#ffffff; }
.sidebar-version { display:inline-block; background:#1e3a8a; color:#93c5fd; font-size:0.72rem; padding:0.15rem 0.6rem; border-radius:999px; margin-top:0.25rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Setup (cached)                                                      #
# ------------------------------------------------------------------ #
@st.cache_resource
def _setup():
    init_db()
    return True


@st.cache_data
def _load_demo():
    with open(_DEMO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_setup()
DEMO = _load_demo()


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def esc(x) -> str:
    return html.escape(str(x))


_GRADE_PILL = {5: "pill-red", 4: "pill-red", 3: "pill-orange", 2: "pill-amber", 1: "pill-green"}


def _norm(k) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(k).lower()).strip()


def _banner(decision: str) -> tuple[str, str]:
    d = (decision or "").lower()
    icon = {"escalate": "⚠️", "monitor": "👁", "dismiss": "✅"}.get(d, "•")
    cls = {"escalate": "banner-escalate", "monitor": "banner-monitor", "dismiss": "banner-dismiss"}.get(d, "banner-dismiss")
    return icon, cls


def _reconstruct_trace(state: dict) -> list[str]:
    """Rebuild a faithful pipeline trace from persisted fields (demo mode has no
    stored trace). Deterministic — mirrors what the agents actually emit."""
    syms = state.get("symptoms") or []
    grades = state.get("ctcae_grades") or {}
    vg = state.get("verifier_grades") or {}
    trace = [f"Extractor: extracted {len(syms)} symptoms: {syms}"]
    for s in syms:
        lk = lookup_ctcae_grade(s)
        if lk.get("matched"):
            trace.append(f"Classifier lookup_ctcae_grade('{s}') -> '{lk['term']}' (match {lk['match_score']})")
        else:
            trace.append(f"Classifier lookup_ctcae_grade('{s}') -> no match; grade from severity")
    trace.append(f"Classifier: assigned grades {grades}")
    trace.append(f"Verifier (gpt-4o-mini): grades {vg} — "
                 f"{'AGREEMENT' if state.get('grade_agreement') else 'DISAGREEMENT'}")
    return trace


def _parse_vitals(text: str) -> str:
    import re
    pats = [r"BP[\s:]*\d{2,3}/\d{2,3}", r"HR[\s:]*\d{2,3}", r"SpO2[\s:]*\d{1,3}\s*%?",
            r"(?:glucose|BG)[\s:]*\d{2,3}", r"platelets?[\s:]*[\d,]+", r"temp(?:erature)?[\s:]*\d{2,3}(?:\.\d)?"]
    found = []
    for p in pats:
        found += re.findall(p, text, flags=re.I)
    return ", ".join(found) if found else "—"


# ------------------------------------------------------------------ #
# Render: the machinery                                               #
# ------------------------------------------------------------------ #
def render_pipeline(state: dict, report_text: str):
    grades = state.get("ctcae_grades") or {}
    vg = state.get("verifier_grades") or {}
    symptoms = state.get("symptoms") or []
    protocol = check_protocol_rule(grades)

    # ---- Stage 1: Extractor ----
    vitals = state.get("vitals") or _parse_vitals(report_text)
    sym_pills = "".join(f'<span class="pill pill-gray" style="margin:2px;">{esc(s)}</span>' for s in symptoms) or "—"
    st.markdown(
        f'<div class="stage-card"><div class="stage-head">'
        f'<span class="stage-num">1</span><span class="stage-title">Extractor</span>'
        f'<span class="stage-model">Claude Haiku</span></div>'
        f'<div class="kv-label">Patient</div><div class="kv-val">{esc(state.get("patient_id") or "—")}</div>'
        f'<div class="kv-label">Symptoms (canonical CTCAE terms)</div><div style="margin-bottom:0.6rem;">{sym_pills}</div>'
        f'<div class="kv-label">Vitals</div><div class="kv-val">{esc(vitals)}</div>'
        + (f'<div class="kv-label">Timeline</div><div class="kv-val">{esc(state.get("timeline"))}</div>' if state.get("timeline") else "")
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)

    # ---- Stage 2: Grader (grounded) ----
    st.markdown(
        '<div class="stage-card"><div class="stage-head">'
        '<span class="stage-num">2</span><span class="stage-title">Grader — grounded in CTCAE v6.0</span>'
        '<span class="stage-model">Claude Haiku</span></div>'
        '<div class="muted" style="margin-bottom:0.5rem;">Each grade is driven by the actual CTCAE criterion retrieved for that grade (the rulebook text below):</div>',
        unsafe_allow_html=True,
    )
    if grades:
        for sym, g in grades.items():
            lk = lookup_ctcae_grade(sym)
            pill = _GRADE_PILL.get(int(g), "pill-gray")
            if lk.get("matched"):
                crit = str(lk.get(f"grade_{int(g)}") or "").strip()
                crit = crit if crit and crit.lower() not in ("nan", "none", "not applicable", "") else "(no per-grade criterion text)"
                term_line = f'<div class="ctcae-term">CTCAE term: “{esc(lk["term"])}” · match {lk.get("match_score")}</div>'
                def_line = f'<div class="ctcae-def">Grade {int(g)} criterion: {esc(crit)}</div>'
            else:
                term_line = '<div class="ctcae-term">no CTCAE match — graded from described severity</div>'
                def_line = ""
            st.markdown(
                f'<div class="grade-block"><div class="grade-top">'
                f'<span class="grade-symptom">{esc(sym)}</span>'
                f'<span class="pill {pill}">Grade {int(g)}</span></div>'
                f'{term_line}{def_line}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No gradeable symptoms.</div>', unsafe_allow_html=True)
    st.markdown('</div><div class="flow-arrow">↓</div>', unsafe_allow_html=True)

    # ---- Stage 3: Deterministic rules ----
    st.markdown(
        f'<div class="stage-card" style="border-left-color:#7c3aed;"><div class="stage-head">'
        f'<span class="stage-num" style="background:#4c1d95;color:#c4b5fd;">3</span>'
        f'<span class="stage-title">Deterministic Protocol Rules</span>'
        f'<span class="stage-model">code, not an LLM</span></div>'
        f'<div class="rule-fired"><strong>Rule fired:</strong> {esc(protocol["reason"])}<br>'
        f'<span class="muted">urgency: {esc(protocol["urgency"])} · max grade: {esc(protocol["max_grade"])} · '
        f'decision: <strong style="color:#fff;">{esc(protocol["decision"]).upper()}</strong></span></div>'
        f'</div><div class="flow-arrow">↓</div>',
        unsafe_allow_html=True,
    )

    # ---- Stage 4: Cross-model verifier ----
    agree = state.get("grade_agreement")
    cg_norm = {_norm(k): v for k, v in grades.items()}
    rows = ""
    for k, v in (vg or {}).items():
        cgv = cg_norm.get(_norm(k))
        same = (cgv == v)
        badge = f'<span class="pill {"pill-green" if same else "pill-red"}">{"match" if same else f"grader {cgv} vs {v}"}</span>'
        rows += (f'<div class="grade-top" style="padding:0.25rem 0;">'
                 f'<span class="grade-symptom">{esc(k)}</span>{badge}</div>')
    overall = ('<span class="pill pill-green">AGREEMENT</span>' if agree
               else '<span class="pill pill-red">DISAGREEMENT → human review</span>')
    st.markdown(
        f'<div class="stage-card" style="border-left-color:#0891b2;"><div class="stage-head">'
        f'<span class="stage-num" style="background:#164e63;color:#67e8f9;">4</span>'
        f'<span class="stage-title">Cross-model Verifier</span>'
        f'<span class="stage-model">GPT-4o-mini · independent re-grade</span></div>'
        f'<div style="margin-bottom:0.5rem;">{overall}</div>{rows or "<div class=muted>—</div>"}'
        f'</div><div class="flow-arrow">↓</div>',
        unsafe_allow_html=True,
    )

    # ---- Stage 5: Decision ----
    icon, cls = _banner(state.get("decision"))
    st.markdown(
        f'<div class="banner {cls}"><div class="banner-decision">{icon} {esc(state.get("decision")).upper()}</div>'
        f'<div class="banner-meta">Escalation grade {esc(state.get("escalation_grade"))} · '
        f'grader confidence {float(state.get("confidence") or 0):.0%}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="card"><div class="section-title">Decision Memo</div>'
        f'<div class="memo-text">{esc(state.get("decision_memo") or "—")}</div>'
        f'<div class="det-note">The decision is produced by the deterministic protocol-rules engine '
        f'from the grader\'s CTCAE grades — not by any LLM. The verifier only flags confidence.</div></div>',
        unsafe_allow_html=True,
    )
    if state.get("needs_human_review"):
        st.warning("⚑ This run is flagged **needs human review** (grade disagreement or low confidence). "
                   "See the Human Review Queue below.")

    with st.expander("Reasoning trace (pipeline steps)"):
        for step in (state.get("reasoning_trace") or _reconstruct_trace(state)):
            st.markdown(f'<div class="trace-step">{esc(step)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏥 ClinGuard</div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-version">CTCAE v6.0 · cross-model</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. **Extractor** — Claude, canonical CTCAE terms + value inference")
    st.markdown("2. **Grader** — Claude, grounded in retrieved CTCAE criteria")
    st.markdown("3. **Rules** — deterministic protocol engine (the decision-maker)")
    st.markdown("4. **Verifier** — GPT-4o-mini, independent cross-model re-grade")
    st.markdown("5. **Decision** — escalate / monitor / dismiss + memo")
    st.divider()
    st.markdown(
        '<div class="muted">The decision comes from deterministic rules, not an LLM. '
        'Metrics are on synthetic data. For demonstration only — not for clinical use.</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ #
# Header + evidence strip                                             #
# ------------------------------------------------------------------ #
st.markdown('<h1 style="font-size:2.4rem;font-weight:800;margin-bottom:0;">ClinGuard</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;margin-top:0.2rem;">Agentic adverse-event triage — grounded grading, deterministic decisions, cross-model verification</p>', unsafe_allow_html=True)

ev = st.columns(5)
_cards = [
    (RESULTS["standard_grade"], "Standard grade accuracy"),
    (RESULTS["standard_escalation"], "Escalation accuracy"),
    (RESULTS["hard_grade"], "Hard-case grade accuracy"),
    (RESULTS["agreement"], "Cross-model agreement (decision)"),
    (RESULTS["review"], "Human-review rate (decision)"),
]
for col, (val, cap) in zip(ev, _cards):
    col.markdown(f'<div class="card" style="text-align:center;"><div class="metric-big">{val}</div>'
                 f'<div class="metric-cap">{cap}</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="muted" style="margin-top:-0.4rem;">Frozen results on 150 synthetic label-by-construction cases '
            f'(see RESULTS.md). Hard-case breakdown: {RESULTS["hard_breakdown"]}.</div>', unsafe_allow_html=True)
st.divider()

# ------------------------------------------------------------------ #
# Mode                                                                #
# ------------------------------------------------------------------ #
mode = st.radio("Mode", ["Demo (instant, no API)", "Live (runs the real pipeline)"],
                horizontal=True, label_visibility="collapsed")
LIVE = mode.startswith("Live")

main, side = st.columns([7, 5])

with main:
    if not LIVE:
        st.markdown('<div class="section-title">Example reports — pick one</div>', unsafe_allow_html=True)
        cols = st.columns(len(DEMO))
        if "demo_idx" not in st.session_state:
            st.session_state["demo_idx"] = 0
        labels = {"escalate": "pill-red", "monitor": "pill-amber", "dismiss": "pill-green"}
        for i, (col, case) in enumerate(zip(cols, DEMO)):
            if col.button(case["label"], key=f"demo_{i}", width="stretch"):
                st.session_state["demo_idx"] = i
        case = DEMO[st.session_state["demo_idx"]]
        st.markdown(f'<div class="card"><div class="kv-label">Report ({esc(case["id"])})</div>'
                    f'<div style="font-size:0.9rem;color:#e2e8f0;line-height:1.6;">{esc(case["report_text"])}</div></div>',
                    unsafe_allow_html=True)
        render_pipeline(case, case["report_text"])

    else:
        st.markdown('<div class="section-title">Live analysis</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted">Runs 3 real, capped model calls (2 Claude Haiku + 1 GPT-4o-mini). '
                    f'Typical cost ~$0.003 / report, 5–12s. Max input {MAX_REPORT_CHARS} chars.</div>',
                    unsafe_allow_html=True)
        report_text = st.text_area("report", height=180, label_visibility="collapsed",
                                   placeholder="Patient PT-XXXX reported ...")
        n = len(report_text)
        over = n > MAX_REPORT_CHARS
        st.markdown(f'<div class="muted">{n} / {MAX_REPORT_CHARS} chars'
                    + (' — <span style="color:#fca5a5;">too long, trim before running</span>' if over else '')
                    + '</div>', unsafe_allow_html=True)
        run = st.button("Run pipeline", type="primary", disabled=over)
        if run and report_text.strip():
            t0 = time.time()
            with st.spinner("Running Extractor → Grader → Rules → Verifier ..."):
                try:
                    result = run_pipeline(report_text.strip())
                    log_decision(result)  # persists + enqueues if flagged
                    st.session_state["live_result"] = result
                    st.session_state["live_report"] = report_text.strip()
                    st.session_state["live_latency"] = time.time() - t0
                except Exception as exc:  # noqa: BLE001
                    st.session_state["live_result"] = None
                    st.error(f"Pipeline error: {exc}")
        elif run:
            st.warning("Paste a report first.")

        if st.session_state.get("live_result"):
            res = st.session_state["live_result"]
            lat = st.session_state.get("live_latency", 0)
            st.markdown(f'<div class="muted">Completed in {lat:.1f}s · est. cost ~$0.003 · persisted to SQLite.</div>',
                        unsafe_allow_html=True)
            render_pipeline(res, st.session_state.get("live_report", ""))

with side:
    # ---- Human review queue (Stage-6 backend) ----
    st.markdown('<div class="section-title">Human Review Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Runs flagged needs_human_review — approve or override, persisted via record_human_review</div>', unsafe_allow_html=True)
    queue = get_review_queue(status="pending", limit=8)
    if not queue:
        st.markdown('<div class="muted">No pending reviews.</div>', unsafe_allow_html=True)
    for row in queue:
        rid = row["run_id"]
        st.markdown(
            f'<div class="card" style="padding:0.9rem 1rem;">'
            f'<div class="grade-top"><span class="grade-symptom">{esc(row.get("patient_id") or "—")}</span>'
            f'<span class="pill pill-amber">{esc(row.get("system_decision")).upper()}</span></div>'
            f'<div class="muted" style="margin:0.35rem 0;">{esc(row.get("review_reasons") or "")}</div>'
            f'<div class="ctcae-term">grader: {esc(row.get("ctcae_grades"))}</div>'
            f'<div class="ctcae-term">verifier: {esc(row.get("verifier_grades"))}</div></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([2, 2, 3])
        if c1.button("Approve", key=f"appr_{rid}", width="stretch"):
            record_human_review(rid, row.get("system_decision"), "approved via UI")
            st.rerun()
        override = c3.selectbox("override", ["escalate", "monitor", "dismiss"],
                                key=f"ov_{rid}", label_visibility="collapsed")
        if c2.button("Override", key=f"ovb_{rid}", width="stretch"):
            record_human_review(rid, override, f"overridden to {override} via UI")
            st.rerun()

    # ---- Audit log ----
    st.markdown('<div class="section-title" style="margin-top:1rem;">Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Recent runs persisted to SQLite (observability)</div>', unsafe_allow_html=True)
    runs = get_recent_runs(10)
    if runs:
        cols = {"patient_id": "Patient", "decision": "Decision", "escalation_grade": "Grade",
                "confidence": "Conf", "created_at": "Time"}
        st.dataframe([{v: r.get(k) for k, v in cols.items()} for r in runs],
                     width="stretch", hide_index=True)
    else:
        st.markdown('<div class="muted">No runs logged yet.</div>', unsafe_allow_html=True)
