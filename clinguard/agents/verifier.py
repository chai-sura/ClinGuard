"""
ClinGuard Agent 3 — Cross-model Verifier.

Independently re-grades the same symptoms on a DIFFERENT model from the grader
(gpt-4o-mini vs the Claude classifier), against the same CTCAE criteria. It does
NOT make the decision — the deterministic protocol_rules engine does that on the
grader's grades. The verifier only gates confidence: if the two models agree the
run proceeds; if they disagree (or the verifier can't be parsed) the run is
flagged for human review and both grade sets are carried forward.

The decision memo is a template built from the deterministic result + agreement,
not a separate LLM agent. Parse errors fail safe to escalate.
"""

from __future__ import annotations

import json
import re
import time

from dotenv import load_dotenv

from clinguard.agents.classifier import _coerce_grade, _criteria_for
from clinguard.config import get_chat_model
from clinguard.graph.state import AgentState
from clinguard.tools.protocol_rules import check_protocol_rule

load_dotenv()

_CONFIDENCE_THRESHOLD = 0.7

_VERIFY_SYSTEM_PROMPT = (
    "You are an independent clinical safety grader, a second opinion. Assign each "
    "symptom an integer CTCAE v6.0 grade from 1 to 5 using ONLY the CTCAE criteria "
    "provided plus the reported severity and vitals. Return ONLY JSON, no prose:\n"
    '{"ctcae_grades": {"<symptom>": <grade 1-5>}}'
)


def _norm(key) -> str:
    # Compare on the semantic symptom, not formatting: lowercase and treat
    # underscores/punctuation as spaces so 'respiratory_distress' (Claude's JSON
    # key style) matches 'respiratory distress' (the verifier's).
    return re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()


def _parse_grades(raw: str) -> tuple[dict, bool]:
    """Parse the verifier's grade JSON. Returns (grades, parse_ok)."""
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        raw_grades = data.get("ctcae_grades", {})
        if not isinstance(raw_grades, dict):
            raise ValueError("ctcae_grades not a JSON object")
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return {}, False
    grades = {}
    for sym, val in raw_grades.items():
        g = _coerce_grade(val)
        if g is not None:
            grades[str(sym)] = g
    return grades, True


def verifier_node(state: AgentState) -> dict:
    """
    LangGraph node: independently re-grade on a second model, compare to the
    grader, gate confidence, and emit the deterministic decision + memo.
    """
    start = time.time()
    symptoms = state.get("symptoms") or []
    vitals = state.get("vitals") or "NONE"
    severity_description = state.get("severity_description") or ""
    classifier_grades = state.get("ctcae_grades") or {}
    classifier_confidence = float(state.get("confidence") or 0.0)

    # Independent second-model grading (skip the call if nothing to grade).
    if symptoms:
        criteria_block, _ = _criteria_for(symptoms)
        llm = get_chat_model("verifier")
        response = llm.invoke([
            {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Symptoms to grade: {symptoms}\n"
                f"Reported severity: {severity_description}\n"
                f"Vitals: {vitals}\n\n"
                f"CTCAE v6.0 criteria:\n{criteria_block}"
            )},
        ])
        verifier_grades, parse_ok = _parse_grades(str(response.content).strip())
    else:
        verifier_grades, parse_ok = {}, True

    # Agreement is exact match of the normalized grade maps.
    cg_norm = {_norm(k): v for k, v in classifier_grades.items()}
    vg_norm = {_norm(k): v for k, v in verifier_grades.items()}
    grade_agreement = parse_ok and cg_norm == vg_norm

    # Deterministic decision — the verifier does NOT decide, protocol_rules does.
    protocol = check_protocol_rule(classifier_grades)
    decision = protocol["decision"]
    escalation_grade = int(protocol["max_grade"])
    urgency, reason = protocol["urgency"], protocol["reason"]

    # Confidence gate → human review; parse failure fails safe to escalate.
    review_reasons: list[str] = []
    if not parse_ok:
        decision = "escalate"  # fail-safe: cannot verify -> escalate
        review_reasons.append("verifier response unparsable — escalated as fail-safe")
    elif not grade_agreement:
        review_reasons.append("grader/verifier grade disagreement")
    if classifier_confidence < _CONFIDENCE_THRESHOLD:
        review_reasons.append(
            f"low grader confidence ({classifier_confidence:.2f} < {_CONFIDENCE_THRESHOLD})")
    needs_human_review = bool(review_reasons)

    # Decision memo as a template/output field (no LLM).
    memo = (
        f"[{decision.upper()} — {urgency}] {reason} "
        f"CTCAE grades (grader/Claude): {classifier_grades or '{}'}. "
        f"Cross-model check (verifier/gpt-4o-mini): "
        f"{verifier_grades if verifier_grades else 'unavailable'} — "
        f"{'AGREEMENT' if grade_agreement else 'DISAGREEMENT'}."
    )
    if needs_human_review:
        memo += " [FLAGGED FOR HUMAN REVIEW: " + "; ".join(review_reasons) + "]"

    trace = list(state.get("reasoning_trace") or [])
    trace.append(
        f"Verifier: decision={decision}, grade={escalation_grade}, "
        f"agreement={grade_agreement}, human_review={needs_human_review}, "
        f"verifier_grades={verifier_grades}"
    )

    return {
        "decision": decision,
        "decision_memo": memo,
        "escalation_grade": escalation_grade,
        "verifier_grades": verifier_grades,
        "grade_agreement": grade_agreement,
        "needs_human_review": needs_human_review,
        "reasoning_trace": trace,
        "latency_ms": (time.time() - start) * 1000,
    }


if __name__ == "__main__":
    import pprint

    test_state = {
        "report_text": "Patient PT-0023 severe chest pain and hypotension.",
        "patient_id": "PT-0023",
        "symptoms": ["chest pain", "hypotension"],
        "severity_description": "severe, unable to stand",
        "vitals": "BP 90/60, HR 110",
        "ctcae_grades": {"chest pain": 3, "hypotension": 3},
        "confidence": 0.88,
        "risk_score": 0.85,
        "reasoning_trace": [],
    }
    pprint.pprint(verifier_node(test_state))
