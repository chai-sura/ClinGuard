"""
ClinGuard Agent 2 — Classifier (grader).

Grounds grading in the real CTCAE lookup: for each extracted symptom it fetches
the CTCAE v6.0 grade criteria and passes them into a single grading prompt, so
grades come from the published criteria rather than model memory. Runs on Claude
(see config role "classifier"). Then applies the deterministic protocol rules.

Output shape is unchanged (ctcae_grades, confidence, risk_score, protocol_breach)
so nothing downstream breaks.
"""

from __future__ import annotations

import json
import time

from dotenv import load_dotenv

from clinguard.config import get_chat_model
from clinguard.graph.state import AgentState
from clinguard.tools.ctcae_lookup import lookup_ctcae_grade
from clinguard.tools.protocol_rules import check_protocol_rule

load_dotenv()

_GRADE_SYSTEM_PROMPT = (
    "You are a clinical trial safety grader. Assign each symptom an integer "
    "CTCAE v6.0 grade from 1 to 5, using ONLY the CTCAE grade criteria provided "
    "for that symptom plus the reported severity and vitals. Do not rely on prior "
    "memory of CTCAE — ground every grade in the criteria text given. If no "
    "criteria are provided for a symptom, grade it from the described severity.\n\n"
    "Return ONLY JSON, no prose, in exactly this shape:\n"
    '{"ctcae_grades": {"<symptom>": <grade 1-5>}, '
    '"confidence": <0.0-1.0>, "risk_score": <0.0-1.0>}\n'
    "confidence = how well the symptoms map onto the provided criteria; "
    "risk_score = overall clinical risk of the case."
)

_NA = ("not applicable", "nan", "none", "")


def _coerce_grade(value) -> int | None:
    """Coerce an LLM grade value to an int in 1..5, or None if unusable."""
    try:
        grade = int(value)
    except (TypeError, ValueError):
        return None
    return grade if 1 <= grade <= 5 else None


def _criteria_for(symptoms: list[str]) -> tuple[str, list[str]]:
    """
    Build the CTCAE criteria block for the grading prompt and matching trace
    notes, by calling the fixed lookup for each symptom (result is USED now).
    """
    blocks: list[str] = []
    notes: list[str] = []
    for sym in symptoms:
        lk = lookup_ctcae_grade(sym)
        if lk.get("matched"):
            grade_lines = []
            for g in range(1, 6):
                text = str(lk.get(f"grade_{g}") or "").strip()
                if text.lower() not in _NA:
                    grade_lines.append(f"  Grade {g}: {text}")
            block = (f"- {sym} (CTCAE term '{lk['term']}', match {lk['match_score']}):\n"
                     + ("\n".join(grade_lines) if grade_lines else "  (no per-grade text)"))
            blocks.append(block)
            notes.append(f"lookup_ctcae_grade('{sym}') -> '{lk['term']}' "
                         f"({len(grade_lines)} grade criteria)")
        else:
            blocks.append(f"- {sym}: no CTCAE criteria found; grade from severity.")
            notes.append(f"lookup_ctcae_grade('{sym}') -> no match")
    return "\n".join(blocks), notes


def classifier_node(state: AgentState) -> dict:
    """
    LangGraph node: grade each symptom against real CTCAE criteria, score risk,
    and apply the deterministic protocol rules.
    """
    symptoms = state.get("symptoms") or []
    vitals = state.get("vitals") or "NONE"
    severity_description = state.get("severity_description") or ""
    start = time.time()

    llm = get_chat_model("classifier")

    criteria_block, lookup_notes = _criteria_for(symptoms)

    user_message = (
        f"Symptoms to grade: {symptoms}\n"
        f"Reported severity: {severity_description}\n"
        f"Vitals: {vitals}\n\n"
        f"CTCAE v6.0 criteria for these symptoms:\n{criteria_block}"
    )

    response = llm.invoke([
        {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    raw = str(response.content).strip()

    # Strip markdown code fences the model may add despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        grading = json.loads(raw)
        if not isinstance(grading, dict):
            raise ValueError("grading response was not a JSON object")
        raw_grades = grading.get("ctcae_grades", {})
        if not isinstance(raw_grades, dict):
            raise ValueError("ctcae_grades was not a JSON object")
        ctcae_grades = {}
        for sym, val in raw_grades.items():
            g = _coerce_grade(val)
            if g is not None:
                ctcae_grades[str(sym)] = g
        confidence = float(grading.get("confidence", 0.5))
        risk_score = float(grading.get("risk_score", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        # Safe defaults keep the pipeline running if the LLM misbehaves
        # (covers non-dict arrays/scalars, not just decode errors).
        ctcae_grades = {}
        confidence = 0.5
        risk_score = 0.5

    # Deterministic protocol rule check
    protocol_result = check_protocol_rule(ctcae_grades)
    protocol_breach = protocol_result["decision"] == "escalate"

    # Reasoning trace — record the grounded lookups actually used
    trace = list(state.get("reasoning_trace") or [])
    trace.extend(f"Classifier {n}" for n in lookup_notes)
    trace.append(
        f"Classifier: assigned grades {ctcae_grades}, "
        f"risk_score={risk_score:.2f}, confidence={confidence:.2f}, "
        f"protocol_breach={protocol_breach}"
    )

    return {
        "ctcae_grades": ctcae_grades,
        "risk_score": risk_score,
        "confidence": confidence,
        "protocol_breach": protocol_breach,
        "reasoning_trace": trace,
        "latency_ms": (time.time() - start) * 1000,
    }


if __name__ == "__main__":
    import pprint

    test_state = {
        "report_text": (
            "Patient PT-0023 reported severe chest pain and hypotension "
            "starting 2 days after dose 3. BP was 90/60, HR 110. "
            "Patient was unable to stand without assistance."
        ),
        "patient_id": "PT-0023",
        "symptoms": ["chest pain", "hypotension"],
        "severity_description": "severe, unable to stand",
        "timeline": "2 days after dose 3",
        "vitals": "BP 90/60, HR 110",
        "reasoning_trace": [],
    }

    result = classifier_node(test_state)
    pprint.pprint(result)
