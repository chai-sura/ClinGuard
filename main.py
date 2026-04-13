"""
ClinGuard — CLI entrypoint.

Runs a single adverse event report through the full 3-agent pipeline
from the terminal and prints a structured results summary.

Usage:
    python main.py '<report text>'
"""

import sys
import time

from dotenv import load_dotenv

load_dotenv()

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "clinguard"))

from db.logger import init_db, log_decision, log_eval
from eval.judge import evaluate_decision
from graph.graph import run_pipeline

_SEP = "=" * 60
_SEC = "-" * 60


def main(report_text: str) -> None:
    # Header
    print(_SEP)
    print("  ClinGuard — AI Clinical Trial Safety Monitor")
    print(_SEP)
    print(f"\nRunning agent pipeline...  [{time.strftime('%H:%M:%S')}]\n")

    # Run pipeline
    result = run_pipeline(report_text)

    # Eval
    eval_scores = evaluate_decision(result)

    # Persist
    init_db()
    run_id = log_decision(result)
    log_eval(run_id, eval_scores)

    # Unpack fields
    patient_id       = result.get("patient_id", "UNKNOWN")
    symptoms         = result.get("symptoms") or []
    vitals           = result.get("vitals") or "NONE"
    timeline         = result.get("timeline") or "—"
    ctcae_grades     = result.get("ctcae_grades") or {}
    decision         = result.get("decision", "")
    escalation_grade = result.get("escalation_grade", 0)
    confidence       = result.get("confidence") or 0.0
    risk_score       = result.get("risk_score") or 0.0
    latency_ms       = result.get("latency_ms") or 0.0
    decision_memo    = result.get("decision_memo", "")
    reasoning_trace  = result.get("reasoning_trace") or []

    grounding          = eval_scores.get("grounding", 0.0)
    completeness       = eval_scores.get("completeness", 0.0)
    hallucination_risk = eval_scores.get("hallucination_risk", 0.0)
    reasoning_depth    = eval_scores.get("reasoning_depth", 0.0)
    agent_agreement    = eval_scores.get("agent_agreement", 0.0)
    overall_score      = eval_scores.get("overall_score", 0.0)

    # ------------------------------------------------------------------ #
    # Section 1 — Patient                                                 #
    # ------------------------------------------------------------------ #
    print("PATIENT")
    print(_SEC)
    print(f"Patient ID   : {patient_id}")
    print(f"Symptoms     : {', '.join(symptoms)}")
    print(f"Vitals       : {vitals}")
    print(f"Timeline     : {timeline}")

    # ------------------------------------------------------------------ #
    # Section 2 — CTCAE Grades                                           #
    # ------------------------------------------------------------------ #
    print("\nCTCAE GRADES")
    print(_SEC)
    if ctcae_grades:
        for symptom, grade in ctcae_grades.items():
            print(f"  {symptom:<25} Grade {grade}")
    else:
        print("  No grades assigned.")

    # ------------------------------------------------------------------ #
    # Section 3 — Decision                                               #
    # ------------------------------------------------------------------ #
    print("\nDECISION")
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

    # ------------------------------------------------------------------ #
    # Section 4 — Decision Memo                                          #
    # ------------------------------------------------------------------ #
    print("\nDECISION MEMO")
    print(_SEC)
    print(decision_memo)

    # ------------------------------------------------------------------ #
    # Section 5 — Eval Scores                                            #
    # ------------------------------------------------------------------ #
    print("\nEVAL SCORES  (LLM-as-Judge)")
    print(_SEC)
    print(f"  Grounding          : {grounding:.2f}")
    print(f"  Completeness       : {completeness:.2f}")
    print(f"  Hallucination Risk : {hallucination_risk:.2f}")
    print(f"  Reasoning Depth    : {reasoning_depth:.2f}")
    print(f"  Agent Agreement    : {agent_agreement:.2f}")
    print(f"  Overall Score      : {overall_score:.2f}")

    # ------------------------------------------------------------------ #
    # Section 6 — Reasoning Trace                                        #
    # ------------------------------------------------------------------ #
    print("\nREASONING TRACE")
    print(_SEC)
    for i, step in enumerate(reasoning_trace):
        print(f"  [{i + 1}] {step}")

    # Footer
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

    report_text = sys.argv[1]
    main(report_text)
