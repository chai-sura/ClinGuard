"""
ClinGuard — CLI entrypoint.

Runs a single adverse event report through the pipeline
    Extractor → Grader (Claude) → deterministic Rules → cross-model Verifier (GPT-4o-mini)
and prints a structured summary. The decision comes from the deterministic
protocol-rules engine, not an LLM. Persists the run to SQLite (and enqueues it
for human review if flagged).

Usage:
    python main.py '<report text>'
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "clinguard"))

from db.logger import init_db, log_decision
from graph.graph import run_pipeline

_SEP = "=" * 60
_SEC = "-" * 60


def main(report_text: str) -> None:
    print(_SEP)
    print("  ClinGuard — Clinical Trial Adverse Event Triage")
    print(_SEP)
    print(f"\nRunning pipeline...  [{time.strftime('%H:%M:%S')}]\n")

    result = run_pipeline(report_text)

    init_db()
    run_id = log_decision(result)  # persists + enqueues if needs_human_review

    patient_id       = result.get("patient_id", "UNKNOWN")
    symptoms         = result.get("symptoms") or []
    vitals           = result.get("vitals") or "NONE"
    timeline         = result.get("timeline") or "—"
    ctcae_grades     = result.get("ctcae_grades") or {}
    verifier_grades  = result.get("verifier_grades") or {}
    grade_agreement  = result.get("grade_agreement")
    needs_review     = result.get("needs_human_review")
    decision         = result.get("decision", "")
    escalation_grade = result.get("escalation_grade", 0)
    confidence       = result.get("confidence") or 0.0
    risk_score       = result.get("risk_score") or 0.0
    latency_ms       = result.get("latency_ms") or 0.0
    decision_memo    = result.get("decision_memo", "")
    reasoning_trace  = result.get("reasoning_trace") or []

    # Section 1 — Patient
    print("PATIENT")
    print(_SEC)
    print(f"Patient ID   : {patient_id}")
    print(f"Symptoms     : {', '.join(symptoms)}")
    print(f"Vitals       : {vitals}")
    print(f"Timeline     : {timeline}")

    # Section 2 — CTCAE grades (grader, Claude)
    print("\nCTCAE GRADES  (grader / Claude, grounded in CTCAE v6.0)")
    print(_SEC)
    if ctcae_grades:
        for symptom, grade in ctcae_grades.items():
            print(f"  {symptom:<25} Grade {grade}")
    else:
        print("  No grades assigned.")

    # Section 3 — Cross-model verifier (GPT-4o-mini)
    print("\nCROSS-MODEL VERIFIER  (GPT-4o-mini, independent re-grade)")
    print(_SEC)
    if verifier_grades:
        for symptom, grade in verifier_grades.items():
            print(f"  {symptom:<25} Grade {grade}")
    else:
        print("  (no verifier grades)")
    agree_str = {True: "AGREEMENT", False: "DISAGREEMENT"}.get(grade_agreement, "—")
    print(f"  Agreement        : {agree_str}")

    # Section 4 — Decision (deterministic)
    print("\nDECISION  (deterministic protocol rules — not an LLM)")
    print(_SEC)
    if decision == "escalate":
        print("!! ESCALATE — Immediate action required !!")
    elif decision == "monitor":
        print(">> MONITOR — Flag for next review")
    elif decision == "dismiss":
        print("-- DISMISS — Continue standard monitoring")

    print(f"Escalation Grade : {escalation_grade}")
    print(f"Confidence       : {confidence:.0%}")
    print(f"Risk Score       : {risk_score:.2f}")
    print(f"Latency          : {latency_ms / 1000:.1f}s")
    if needs_review:
        print("FLAGGED          : needs human review (queued in SQLite)")

    # Section 5 — Decision memo
    print("\nDECISION MEMO")
    print(_SEC)
    print(decision_memo)

    # Section 6 — Reasoning trace
    print("\nREASONING TRACE")
    print(_SEC)
    for i, step in enumerate(reasoning_trace):
        print(f"  [{i + 1}] {step}")

    print()
    print(_SEP)
    print(f"Run ID: {run_id} | Logged to SQLite")
    print(_SEP)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py '<report text>'")
        print()
        print("Example:")
        print("  python main.py 'Patient PT-0042 reported")
        print("  severe chest pain 3 days after dose 2.'")
        sys.exit(1)

    main(sys.argv[1])
