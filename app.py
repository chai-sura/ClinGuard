"""
ClinGuard — Streamlit UI (presentation layer).

Pipeline:
    Extractor → Grader (Claude) → deterministic Rules → Verifier / LLM-as-Judge (gpt-4o-mini) → Decision
Only the final decision is made by fixed protocol rules; the extractor, grader,
and verifier are LLMs. Grading is grounded in the CTCAE v6.0 rulebook.

Tabs: Examples (frozen demo, no API) · Analyze (live) · Review (human-in-the-loop).
Presentation only — nothing in the pipeline, config, agents, or DB logic changes.
"""

from __future__ import annotations

import html
import json
import os
import re
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

MAX_REPORT_CHARS = 4000
CONFIDENCE_THRESHOLD = 0.7  # mirrors the verifier's gate (display only)

# Raw model names live only here / in the technical-details line — never as a primary label.
GRADER_MODEL = "Claude Haiku"
VERIFIER_MODEL = "gpt-4o-mini"

# Frozen ground-truth eval numbers (see RESULTS.md). Synthetic data only.
METRICS = [
    ("81.6%", "Standard grade accuracy"),
    ("94.4%", "Escalation accuracy"),
    ("84.0%", "Hard-case accuracy"),
    ("94%", "Model agreement"),
    ("11%", "Human-review rate"),
]
METRICS_FOOTNOTE = ("Measured on 150 synthetic label-by-construction cases · "
                    "hard cases by type: boundary 57% · missing-vital 100% · rare-term 100% · multi-symptom 83%")

DEMO_LABELS = {
    "Escalate":     "🔴 Life-threatening (escalate)",
    "Monitor":      "🟡 Needs monitoring",
    "Dismiss":      "🟢 Mild (dismiss)",
    "Human review": "⚠️ Ambiguous (human review)",
}

EXAMPLE_REPORT = (
    "Patient PT-0042 developed a widespread rash and elevated blood glucose "
    "(240 mg/dL) three days after the second dose. BP 150/95, HR 98. "
    "Managed at home with oral medication and continued on the study."
)

_DEMO_PATH = os.path.join(os.path.dirname(__file__), "clinguard", "data", "demo_traces.json")

st.set_page_config(page_title="ClinGuard", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ #
# Styling — warm, elevated, professional clinical palette             #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
:root {
  /* one charcoal background, two surfaces, one line */
  --bg:#15171d; --surface:#1e212a; --surface-2:#262a34; --line:#2d323d;
  /* one off-white text + one muted caption tone */
  --text:#e9ecf2; --text-2:#959dab; --muted:#959dab;
  /* one calm slate-blue accent — tabs / buttons / focus only */
  --accent:#7488a8; --accent-soft:#1d2532; --accent-line:#37435c;
  /* red / amber / green — severity + decision badges ONLY, never decorative */
  --esc:#e8756e; --esc-bg:#241619; --esc-line:#573034;
  --mon:#e0b45f; --mon-bg:#241f13; --mon-line:#544629;
  --dis:#5fc794; --dis-bg:#142019; --dis-line:#28503c;
  --shadow:0 1px 2px rgba(0,0,0,0.3), 0 10px 26px -18px rgba(0,0,0,0.5);
}
html, body, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
  color:var(--text);
}
[data-testid="stAppViewContainer"] { background:var(--bg) !important; }
.block-container { padding-top:2.2rem; padding-bottom:3.5rem; max-width:1240px; }
[data-testid="stHeader"] { background:transparent !important; }
hr { border-color:var(--line); }

.app-title { font-size:1.8rem; font-weight:750; letter-spacing:-0.015em; color:var(--text); margin:0; }
.app-sub { color:var(--text-2); font-size:0.96rem; margin-top:0.4rem; line-height:1.6; max-width:780px; }
.eyebrow { font-size:0.72rem; font-weight:650; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); }

.card { background:var(--surface); border:1px solid var(--line); border-radius:16px;
        padding:1.5rem 1.6rem; margin-bottom:1.2rem; box-shadow:var(--shadow); }
.card-title { font-size:1rem; font-weight:680; color:var(--text); margin-bottom:0.2rem; }
.card-sub { font-size:0.83rem; color:var(--text-2); margin-bottom:1rem; }

.metric { background:var(--surface); border:1px solid var(--line); border-radius:16px;
          padding:1.2rem 1rem; text-align:center; height:100%; box-shadow:var(--shadow); }
.metric-num { font-size:1.6rem; font-weight:750; color:var(--text); letter-spacing:-0.015em; }
.metric-lab { font-size:0.75rem; color:var(--text-2); margin-top:0.35rem; line-height:1.35; }
.footnote { font-size:0.74rem; color:var(--muted); margin-top:0.7rem; line-height:1.55; }

.stage { background:var(--surface); border:1px solid var(--line); border-radius:16px;
         padding:1.3rem 1.5rem; margin-bottom:0.5rem; box-shadow:var(--shadow); }
.stage-head { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.95rem; }
.stage-num { width:28px; height:28px; border-radius:9px; background:var(--accent-soft);
             color:var(--accent); font-size:0.82rem; font-weight:700; display:inline-flex;
             align-items:center; justify-content:center; flex-shrink:0; }
.stage-title { font-size:1rem; font-weight:680; color:var(--text); }
.stage-tag { margin-left:auto; font-size:0.72rem; color:var(--text-2); background:var(--surface-2);
             padding:0.2rem 0.65rem; border-radius:999px; border:1px solid var(--line); }
.connector { width:2px; height:20px; background:linear-gradient(var(--line),transparent); margin:0 auto 0.5rem 27px; }

.kv { margin-bottom:0.8rem; }
.kv-label { font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; }
.kv-val { font-size:0.95rem; font-weight:560; color:var(--text); margin-top:0.14rem; }

.chip { display:inline-block; padding:0.26rem 0.7rem; border-radius:9px; font-size:0.81rem;
        font-weight:620; margin:0.2rem 0.32rem 0.2rem 0; border:1px solid transparent; }
.chip.neutral { background:var(--surface-2); color:var(--text-2); border-color:var(--line); }
.chip.g1 { background:var(--dis-bg); color:var(--dis); border-color:var(--dis-line); }
.chip.g2 { background:var(--mon-bg); color:var(--mon); border-color:var(--mon-line); }
.chip.g3 { background:#271c10; color:#eca657; border-color:#634321; }
.chip.g4, .chip.g5 { background:var(--esc-bg); color:var(--esc); border-color:var(--esc-line); }
.chip.diff { border-style:dashed; }

.criterion { padding:0.65rem 0; border-bottom:1px solid var(--line); }
.criterion:last-child { border-bottom:none; }
.criterion-tag { font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.03em; }
.criterion-text { font-size:0.88rem; line-height:1.6; color:var(--text); margin-top:0.3rem; }

.rulebox { background:var(--surface-2); border:1px solid var(--line); border-radius:12px;
           padding:0.95rem 1.1rem; font-size:0.91rem; color:var(--text); line-height:1.55; }
.rulebox .lead { color:var(--muted); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.3rem; }

.decision { background:var(--surface); border:1px solid var(--line); border-top:3px solid var(--line);
            border-radius:18px; padding:1.7rem; text-align:center; margin-bottom:0.5rem; box-shadow:var(--shadow); }
.decision.esc { border-top-color:var(--esc); } .decision.mon { border-top-color:var(--mon); } .decision.dis { border-top-color:var(--dis); }
.decision-word { font-size:2.05rem; font-weight:770; letter-spacing:0.06em; color:var(--text); }
.decision.esc .decision-word { color:var(--esc); } .decision.mon .decision-word { color:var(--mon); } .decision.dis .decision-word { color:var(--dis); }
.decision-meta { font-size:0.87rem; margin-top:0.45rem; color:var(--text-2); }

.summary { font-size:1rem; line-height:1.6; color:var(--text); }
.trust { display:inline-flex; align-items:center; gap:0.45rem; font-size:0.76rem; color:var(--accent);
         background:var(--accent-soft); border:1px solid var(--accent-line); border-radius:999px;
         padding:0.34rem 0.82rem; margin-top:1rem; }
.flag { font-size:0.84rem; color:var(--text-2); background:var(--surface-2); border:1px solid var(--line);
        border-left:3px solid var(--accent); border-radius:11px; padding:0.7rem 0.95rem; margin-top:0.8rem; }

.queue-item { background:var(--surface); border:1px solid var(--line); border-radius:16px;
              padding:1.15rem 1.3rem; margin-bottom:0.9rem; box-shadow:var(--shadow); }
.queue-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:0.6rem; }
.queue-patient { font-size:0.94rem; font-weight:680; color:var(--text); }
.queue-reason { font-size:0.82rem; color:var(--text-2); background:var(--surface-2); border:1px solid var(--line);
                border-left:3px solid var(--accent); border-radius:9px; padding:0.4rem 0.65rem; margin-bottom:0.7rem; }
.queue-report { font-size:0.85rem; line-height:1.55; color:var(--text-2); background:var(--surface-2);
                border:1px solid var(--line); border-radius:11px; padding:0.7rem 0.85rem; margin-bottom:0.7rem; }
.report-label { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.25rem; }

.audit { width:100%; border-collapse:collapse; font-size:0.83rem; }
.audit th { text-align:left; color:var(--muted); font-weight:600; font-size:0.72rem;
            text-transform:uppercase; letter-spacing:0.03em; padding:0.5rem 0.5rem; border-bottom:1px solid var(--line); }
.audit td { padding:0.6rem 0.5rem; border-bottom:1px solid var(--line); color:var(--text-2); }
.audit td.pt { color:var(--text); font-weight:560; }
.tag { padding:0.14rem 0.55rem; border-radius:7px; font-size:0.72rem; font-weight:640; }
.tag.esc{background:var(--esc-bg);color:var(--esc);} .tag.mon{background:var(--mon-bg);color:var(--mon);}
.tag.dis{background:var(--dis-bg);color:var(--dis);}

.pill { padding:0.24rem 0.72rem; border-radius:999px; font-size:0.78rem; font-weight:640; }
.pill.esc{background:var(--esc-bg);color:var(--esc);border:1px solid var(--esc-line);}
.pill.mon{background:var(--mon-bg);color:var(--mon);border:1px solid var(--mon-line);}
.pill.dis{background:var(--dis-bg);color:var(--dis);border:1px solid var(--dis-line);}
.pill.neutral{background:var(--surface-2);color:var(--text-2);border:1px solid var(--line);}
.pill.accent{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent-line);}
.muted { color:var(--text-2); font-size:0.84rem; }
.tech { font-size:0.8rem; color:var(--text-2); line-height:1.6; }
.tech code { background:var(--surface-2); padding:0.1rem 0.35rem; border-radius:5px; color:var(--text); }

.stTabs [data-baseweb="tab-list"] { gap:0.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:2.7rem; padding:0 1.15rem; color:var(--text-2); font-weight:640; font-size:0.9rem; }
.stTabs [aria-selected="true"] { color:var(--text); border-bottom:2px solid var(--accent); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent) !important; }

section[data-testid="stSidebar"] > div { background:var(--surface) !important; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] span { color:var(--text-2) !important; }
[data-testid="stTextArea"] textarea { background:var(--surface) !important; color:var(--text) !important;
   border:1px solid var(--line) !important; border-radius:12px !important; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:var(--surface-2); }
div[data-testid="stExpander"] summary { font-size:0.82rem; color:var(--text-2); }
.stButton button { border-radius:10px !important; border:1px solid var(--line) !important; }
.stButton button[kind="primary"] { background:var(--accent) !important; border-color:var(--accent) !important; color:#12141a !important; }
[data-testid="stElementToolbar"] { display:none; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Setup                                                               #
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
# Helpers (presentation only)                                         #
# ------------------------------------------------------------------ #
def esc(x) -> str:
    return html.escape(str(x))


def _norm(k) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(k).lower()).strip()


def _gc(g) -> str:
    try:
        g = int(g)
    except (TypeError, ValueError):
        return "neutral"
    return f"g{g}" if 1 <= g <= 5 else "neutral"


def _decision_kind(decision: str) -> str:
    return {"escalate": "esc", "monitor": "mon", "dismiss": "dis"}.get((decision or "").lower(), "dis")


def _as_dict(v) -> dict:
    if isinstance(v, dict):
        return v
    try:
        d = json.loads(v or "{}")
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _grade_chips(grader: dict, verifier: dict | None = None) -> str:
    """Per-symptom grade chips. If a verifier grade differs, show 'G3 → G4 ⚠'."""
    vn = {_norm(k): v for k, v in (verifier or {}).items()}
    parts = []
    for sym, g in grader.items():
        vv = vn.get(_norm(sym))
        if verifier is not None and vv is not None and vv != g:
            parts.append(f'<span class="chip {_gc(g)} diff">{esc(sym)} · G{g} → G{vv} ⚠</span>')
        else:
            parts.append(f'<span class="chip {_gc(g)}">{esc(sym)} · G{g}</span>')
    return "".join(parts) or '<span class="muted">—</span>'


def _parse_vitals(text: str) -> str:
    pats = [r"BP[\s:]*\d{2,3}/\d{2,3}", r"HR[\s:]*\d{2,3}", r"SpO2[\s:]*\d{1,3}\s*%?",
            r"(?:glucose|BG)[\s:]*\d{2,3}", r"platelets?[\s:]*[\d,]+", r"temp(?:erature)?[\s:]*\d{2,3}(?:\.\d)?"]
    found = []
    for p in pats:
        found += re.findall(p, text, flags=re.I)
    return ", ".join(found) if found else "—"


def _flag_reasons(state: dict) -> list[str]:
    """Human-readable reasons a run was flagged (disagreement / low confidence)."""
    reasons = []
    if state.get("grade_agreement") is False:
        reasons.append("a grader–verifier disagreement")
    if float(state.get("confidence") or 0) < CONFIDENCE_THRESHOLD:
        reasons.append("low grader confidence")
    return reasons


def _decision_sentence(state: dict, protocol: dict) -> str:
    """Clean human sentence for the decision summary (no raw dicts)."""
    verb = {"escalate": "Escalated", "monitor": "Flagged for monitoring",
            "dismiss": "Dismissed"}.get((state.get("decision") or "").lower(), "Assessed")
    text = f"{verb} — {protocol['reason']}".strip()
    if state.get("needs_human_review"):
        reasons = _flag_reasons(state)
        if reasons:
            text += f" Flagged for human review due to {' and '.join(reasons)}."
    return text


def _story(state: dict) -> list[str]:
    syms = state.get("symptoms") or []
    grades = state.get("ctcae_grades") or {}
    matched = sum(1 for s in syms if lookup_ctcae_grade(s).get("matched"))
    return [
        f"Extracted {len(syms)} symptom(s) and vitals from the report.",
        f"Retrieved CTCAE v6.0 criteria for {matched} of {len(syms)} symptom(s) and graded each.",
        ("The Verifier independently re-graded the same symptoms and agreed."
         if state.get("grade_agreement") else
         "The Verifier disagreed on a grade, so the case was flagged for review."),
        f"Fixed safety rules were applied and returned: {check_protocol_rule(grades)['decision'].upper()}.",
    ]


def _fmt_time(iso) -> str:
    try:
        return str(iso)[11:16]
    except Exception:  # noqa: BLE001
        return "—"


# ------------------------------------------------------------------ #
# Pipeline renderer                                                   #
# ------------------------------------------------------------------ #
def render_pipeline(state: dict, report_text: str):
    grades = state.get("ctcae_grades") or {}
    vg = state.get("verifier_grades") or {}
    symptoms = state.get("symptoms") or []
    protocol = check_protocol_rule(grades)

    # Stage 1 — Extractor
    vitals = state.get("vitals") or _parse_vitals(report_text)
    sym_chips = "".join(f'<span class="chip neutral">{esc(s)}</span>' for s in symptoms) or '<span class="muted">—</span>'
    st.markdown(
        f'<div class="stage"><div class="stage-head"><span class="stage-num">1</span>'
        f'<span class="stage-title">Extractor</span><span class="stage-tag">reads the report</span></div>'
        f'<div class="kv"><div class="kv-label">Patient</div><div class="kv-val">{esc(state.get("patient_id") or "—")}</div></div>'
        f'<div class="kv"><div class="kv-label">Symptoms</div><div style="margin-top:0.2rem;">{sym_chips}</div></div>'
        f'<div class="kv"><div class="kv-label">Vitals</div><div class="kv-val">{esc(vitals)}</div></div>'
        + (f'<div class="kv"><div class="kv-label">Timeline</div><div class="kv-val">{esc(state.get("timeline"))}</div></div>' if state.get("timeline") else "")
        + '</div><div class="connector"></div>',
        unsafe_allow_html=True,
    )

    # Stage 2 — Grader (progressive disclosure)
    grade_badges = "".join(
        f'<span class="chip {_gc(g)}">{esc(sym)} · Grade {int(g)}</span>' for sym, g in grades.items()
    ) or '<span class="muted">No gradeable symptoms.</span>'
    st.markdown(
        f'<div class="stage"><div class="stage-head"><span class="stage-num">2</span>'
        f'<span class="stage-title">Grader</span><span class="stage-tag">grounded in CTCAE v6.0</span></div>'
        f'<div>{grade_badges}</div></div>',
        unsafe_allow_html=True,
    )
    if grades:
        with st.expander("View criteria — the exact CTCAE v6.0 text behind each grade"):
            for sym, g in grades.items():
                lk = lookup_ctcae_grade(sym)
                if lk.get("matched"):
                    crit = str(lk.get(f"grade_{int(g)}") or "").strip()
                    crit = crit if crit and crit.lower() not in ("nan", "none", "not applicable", "") else "No distinct criterion text at this grade."
                    st.markdown(
                        f'<div class="criterion"><div class="criterion-tag">{esc(sym)} · CTCAE v6.0 “{esc(lk["term"])}” · Grade {int(g)}</div>'
                        f'<div class="criterion-text">{esc(crit)}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="criterion"><div class="criterion-tag">{esc(sym)} · no CTCAE match</div>'
                        f'<div class="criterion-text">Graded from the described severity.</div></div>',
                        unsafe_allow_html=True,
                    )
    st.markdown('<div class="connector"></div>', unsafe_allow_html=True)

    # Stage 3 — Deterministic rules
    st.markdown(
        f'<div class="stage"><div class="stage-head"><span class="stage-num">3</span>'
        f'<span class="stage-title">Protocol Rules</span><span class="stage-tag">fixed rules · makes the decision</span></div>'
        f'<div class="rulebox"><div class="lead">Rule applied</div>{esc(protocol["reason"])}</div>'
        f'</div><div class="connector"></div>',
        unsafe_allow_html=True,
    )

    # Stage 4 — Verifier (LLM-as-Judge)
    agree = state.get("grade_agreement")
    verdict = ('<span class="pill neutral">Verifier agrees</span>' if agree
               else '<span class="pill accent">Verifier disagrees — flagged for review</span>')
    st.markdown(
        f'<div class="stage"><div class="stage-head"><span class="stage-num">4</span>'
        f'<span class="stage-title">Verifier</span><span class="stage-tag">independent LLM-as-Judge · cross-checks every grade</span></div>'
        f'<div style="margin-bottom:0.75rem;">{verdict}</div>'
        f'<div>{_grade_chips(grades, vg)}</div>'
        f'</div><div class="connector"></div>',
        unsafe_allow_html=True,
    )

    # Stage 5 — Decision
    kind = _decision_kind(state.get("decision"))
    st.markdown(
        f'<div class="decision {kind}"><div class="decision-word">{esc(state.get("decision")).upper()}</div>'
        f'<div class="decision-meta">Escalation grade {esc(state.get("escalation_grade"))} · '
        f'grader confidence {float(state.get("confidence") or 0):.0%}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="card" style="margin-top:0.5rem;"><div class="card-title">Decision summary</div>'
        f'<div class="summary">{esc(_decision_sentence(state, protocol))}</div>'
        f'<div class="trust">◆ This decision is set by fixed protocol rules — not by an AI model</div>'
        + ('<div class="flag">⚑ Flagged for human review — see the Review tab.</div>' if state.get("needs_human_review") else "")
        + '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Technical details"):
        st.markdown(
            f'<div class="tech">Grader: <code>{esc(GRADER_MODEL)}</code> · '
            f'Verifier (LLM-as-Judge): <code>{esc(VERIFIER_MODEL)}</code><br><br>'
            f'Raw decision record:<br><code>{esc(state.get("decision_memo") or "—")}</code></div>',
            unsafe_allow_html=True,
        )
    with st.expander("How this decision was reached"):
        for step in _story(state):
            st.markdown(f'<div class="muted" style="padding:0.2rem 0;">• {esc(step)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown('<div class="app-title" style="font-size:1.3rem;">🩺 ClinGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted" style="margin-bottom:1.2rem;">Adverse-event triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-bottom:0.8rem;">Pipeline</div>', unsafe_allow_html=True)
    stages = [
        ("1", "Extractor", "Extracts symptoms & vitals"),
        ("2", "Grader", "Grades severity against CTCAE v6.0"),
        ("3", "Rules", "Applies fixed protocol rules — makes the decision"),
        ("4", "Verifier", "Independent LLM-as-Judge cross-check"),
        ("5", "Decision", "Escalate, monitor, or dismiss"),
    ]
    for num, name, desc in stages:
        st.markdown(
            f'<div style="display:flex;gap:0.6rem;margin-bottom:0.85rem;">'
            f'<span class="stage-num">{num}</span>'
            f'<div><div style="color:var(--text);font-weight:640;font-size:0.86rem;">{name}</div>'
            f'<div class="muted" style="font-size:0.78rem;">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="footnote" style="border-top:1px solid var(--line);padding-top:0.9rem;margin-top:0.7rem;">'
        'Prototype for demonstration only — not a medical device, not for clinical use. '
        'Evaluated on synthetic data.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
# Header + explainer                                                  #
# ------------------------------------------------------------------ #
st.markdown('<div class="app-title">ClinGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">ClinGuard reviews a clinical-trial adverse-event report, grades how severe each '
    'symptom is against <b>CTCAE — the official medical severity scale</b> — and recommends whether to '
    'escalate, monitor, or dismiss. A separate <b>Verifier</b> model independently re-checks every grade, '
    'and the final call is made by fixed safety rules, not by an AI.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)

cols = st.columns(len(METRICS))
for col, (num, lab) in zip(cols, METRICS):
    col.markdown(f'<div class="metric"><div class="metric-num">{num}</div><div class="metric-lab">{lab}</div></div>',
                 unsafe_allow_html=True)
st.markdown(f'<div class="footnote">{METRICS_FOOTNOTE}</div>', unsafe_allow_html=True)
st.markdown('<div style="height:1.4rem;"></div>', unsafe_allow_html=True)

tab_examples, tab_analyze, tab_review = st.tabs(["  Examples  ", "  Analyze a report  ", "  Review queue  "])

# ---- Tab 1: Examples ----
with tab_examples:
    st.markdown('<div class="eyebrow" style="margin:0.5rem 0 0.8rem;">Pick a case to see how the system reasons</div>',
                unsafe_allow_html=True)
    if "demo_idx" not in st.session_state:
        st.session_state["demo_idx"] = 0
    bcols = st.columns(len(DEMO))
    for i, (col, case) in enumerate(zip(bcols, DEMO)):
        label = str(DEMO_LABELS.get(case["label"], case["label"]))
        if col.button(label, key=f"demo_{i}", width="stretch",
                      type="primary" if i == st.session_state["demo_idx"] else "secondary"):
            st.session_state["demo_idx"] = i
    case = DEMO[st.session_state["demo_idx"]]
    left, right = st.columns([7, 5], gap="large")
    with left:
        render_pipeline(case, case["report_text"])
    with right:
        st.markdown(f'<div class="card"><div class="card-title">Report</div>'
                    f'<div class="summary" style="font-size:0.9rem;">{esc(case["report_text"])}</div></div>',
                    unsafe_allow_html=True)

# ---- Tab 2: Analyze (live) ----
with tab_analyze:
    st.markdown('<div class="eyebrow" style="margin:0.5rem 0 0.5rem;">Analyze a real report</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted" style="margin-bottom:0.7rem;">Paste a report or edit the example below, then Run. '
                f'Runs the full pipeline (a few seconds). Maximum {MAX_REPORT_CHARS:,} characters.</div>',
                unsafe_allow_html=True)
    report_text = st.text_area("report", value=EXAMPLE_REPORT, height=160, label_visibility="collapsed")
    n = len(report_text)
    over = n > MAX_REPORT_CHARS
    st.markdown('<div class="muted">' + f'{n:,} / {MAX_REPORT_CHARS:,} characters'
                + ('  ·  <span style="color:var(--esc);">too long — please trim</span>' if over else '')
                + '</div>', unsafe_allow_html=True)
    run = st.button("Run analysis", type="primary", disabled=over)
    if run and report_text.strip():
        t0 = time.time()
        with st.spinner("Analyzing…"):
            try:
                result = run_pipeline(report_text.strip())
                log_decision(result)
                st.session_state["live_result"] = result
                st.session_state["live_report"] = report_text.strip()
                st.session_state["live_latency"] = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                st.session_state["live_result"] = None
                st.error(f"Could not complete the analysis: {exc}")
    elif run:
        st.warning("Please enter a report first.")

    if st.session_state.get("live_result"):
        lat = st.session_state.get("live_latency", 0)
        st.markdown(f'<div class="muted" style="margin:0.5rem 0 0.9rem;">Completed in {lat:.1f}s · saved to the audit log.</div>',
                    unsafe_allow_html=True)
        render_pipeline(st.session_state["live_result"], st.session_state.get("live_report", ""))

# ---- Tab 3: Review ----
with tab_review:
    if st.session_state.get("review_msg"):
        st.success(st.session_state.pop("review_msg"))

    q_left, q_right = st.columns([6, 5], gap="large")
    with q_left:
        st.markdown('<div class="card-title" style="margin-top:0.5rem;">Human review</div>'
                    '<div class="card-sub">Cases the Verifier flagged, awaiting your confirmation.</div>',
                    unsafe_allow_html=True)
        queue = get_review_queue(status="pending", limit=8)
        if not queue:
            st.markdown('<div class="muted">Nothing awaiting review.</div>', unsafe_allow_html=True)
        for row in queue:
            rid = row["run_id"]
            patient = row.get("patient_id") or "Unknown patient"
            kind = _decision_kind(row.get("system_decision"))
            reason = row.get("review_reasons") or "flagged for review"
            report = row.get("report_text") or "(report text unavailable)"
            chips = _grade_chips(_as_dict(row.get("ctcae_grades")), _as_dict(row.get("verifier_grades")))
            st.markdown(
                f'<div class="queue-item"><div class="queue-head">'
                f'<span class="queue-patient">{esc(patient)}</span>'
                f'<span class="pill {kind}">{esc(row.get("system_decision")).upper()}</span></div>'
                f'<div class="queue-reason">⚑ {esc(reason)}</div>'
                f'<div class="report-label">Original report</div>'
                f'<div class="queue-report">{esc(report)}</div>'
                f'<div>{chips}</div></div>',
                unsafe_allow_html=True,
            )
            a, b, c = st.columns([2, 2, 3])
            if a.button("Approve", key=f"appr_{rid}", width="stretch"):
                record_human_review(rid, row.get("system_decision"), "approved via UI")
                st.session_state["review_msg"] = f"Recorded — {patient} marked confirmed. Saved to the review log."
                st.rerun()
            override = c.selectbox("Override to", ["escalate", "monitor", "dismiss"],
                                   key=f"ov_{rid}", label_visibility="collapsed")
            if b.button("Override", key=f"ovb_{rid}", width="stretch"):
                record_human_review(rid, override, f"overridden to {override} via UI")
                st.session_state["review_msg"] = f"Recorded — {patient} overridden to {override}. Saved to the review log."
                st.rerun()
    with q_right:
        st.markdown('<div class="card-title" style="margin-top:0.5rem;">Recent activity</div>'
                    '<div class="card-sub">Latest decisions logged by the system.</div>', unsafe_allow_html=True)
        runs = get_recent_runs(10)
        if runs:
            body = ""
            for r in runs:
                k = _decision_kind(r.get("decision"))
                conf = r.get("confidence")
                conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
                body += (f'<tr><td class="pt">{esc(r.get("patient_id") or "—")}</td>'
                         f'<td><span class="tag {k}">{esc(r.get("decision")).upper()}</span></td>'
                         f'<td>G{esc(r.get("escalation_grade"))}</td><td>{conf_s}</td>'
                         f'<td>{_fmt_time(r.get("created_at"))}</td></tr>')
            st.markdown(
                '<table class="audit"><thead><tr><th>Patient</th><th>Decision</th><th>Grade</th>'
                f'<th>Confidence</th><th>Time</th></tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="muted">No activity yet.</div>', unsafe_allow_html=True)
