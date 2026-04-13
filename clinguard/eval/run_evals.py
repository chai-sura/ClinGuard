"""
ClinGuard batch evaluation runner.

Runs all 50 synthetic AE fixtures through the full 3-agent pipeline,
scores each run with the LLM-as-judge harness, persists results to
SQLite, and prints aggregated portfolio-ready metrics.

Usage (from project root, with cenv activated):
    python -m eval.run_evals
    # or
    python clinguard/eval/run_evals.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# Allow running as a top-level script from the project root
_CLINGUARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_CLINGUARD_DIR, ".."))
sys.path.insert(0, _CLINGUARD_DIR)   # allows: from graph.graph import ...
sys.path.insert(0, _PROJECT_ROOT)    # allows: from clinguard.agents import ... (used inside graph.py)

from graph.graph import run_pipeline
from eval.judge import evaluate_decision
from db.logger import init_db, log_decision, log_eval

_FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ae_fixtures.json")

_SEP = "=" * 60
_SEC = "-" * 60


def _expected_decision(grade: int) -> str:
    """Map a fixture's CTCAE grade to the expected safety decision."""
    if grade in (1, 2):
        return "dismiss_or_monitor"
    return "escalate"  # grade 3, 4, 5


def run_batch_eval() -> dict:
    """
    Run all 50 AE fixtures through the full pipeline.

    Returns a summary dict with mean scores across all runs plus
    escalation accuracy.
    """
    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    with open(_FIXTURES_PATH, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    init_db()

    # Score accumulators
    all_grounding    = []
    all_completeness = []
    all_hallucination = []
    all_reasoning    = []
    all_agreement    = []
    all_overall      = []
    all_latency      = []

    escalation_correct = 0
    total              = 0

    print(_SEP)
    print("  ClinGuard — Batch Eval Runner")
    print(_SEP)
    print(f"  Fixtures loaded : {len(fixtures)}")
    print(f"  Started at      : {time.strftime('%H:%M:%S')}")
    print(_SEC)

    # Main loop                                                            
    for i, fixture in enumerate(fixtures):
        grade       = fixture.get("grade", 0)
        report_text = fixture.get("report_text", "")

        # Best-effort patient ID extraction — fixtures may or may not have
        # an explicit "patient_id" key; fall back to the fixture id field.
        patient_id = fixture.get("patient_id") or fixture.get("id", f"fixture_{i+1:03d}")

        print(f"[{i+1:02d}/{len(fixtures)}] Grade {grade} — {patient_id}...", end=" ", flush=True)
        run_start = time.time()

        try:
            result     = run_pipeline(report_text)
            eval_scores = evaluate_decision(result)
            run_id     = log_decision(result)
            log_eval(run_id, eval_scores)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            continue

        elapsed = time.time() - run_start

        # Accumulate scores                                           
        all_grounding.append(eval_scores.get("grounding", 0.0))
        all_completeness.append(eval_scores.get("completeness", 0.0))
        all_hallucination.append(eval_scores.get("hallucination_risk", 0.0))
        all_reasoning.append(eval_scores.get("reasoning_depth", 0.0))
        all_agreement.append(eval_scores.get("agent_agreement", 0.0))
        all_overall.append(eval_scores.get("overall_score", 0.0))
        all_latency.append(result.get("latency_ms") or elapsed * 1000)

        # Escalation correctness                                      
        actual   = result.get("decision", "")
        expected = _expected_decision(grade)

        if expected == "escalate":
            correct = actual == "escalate"
        else:
            # grade 1-2 → either "dismiss" or "monitor" is acceptable
            correct = actual in ("dismiss", "monitor")

        if correct:
            escalation_correct += 1
        total += 1

        status = "OK" if correct else "WRONG"
        print(f"{actual:<10} [{status}]  overall={eval_scores.get('overall_score', 0):.2f}  {elapsed:.1f}s")

    # Aggregate                                                            
    def _mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    n = len(all_overall)
    summary = {
        "runs_completed":       n,
        "escalation_accuracy":  escalation_correct / total if total else 0.0,
        "mean_grounding":       _mean(all_grounding),
        "mean_completeness":    _mean(all_completeness),
        "mean_hallucination":   _mean(all_hallucination),
        "mean_reasoning":       _mean(all_reasoning),
        "mean_agreement":       _mean(all_agreement),
        "mean_overall":         _mean(all_overall),
        "mean_latency_ms":      _mean(all_latency),
        "p95_latency_ms":       sorted(all_latency)[int(0.95 * n) - 1] if n >= 20 else max(all_latency, default=0.0),
    }

    # Print report                                                        
    print()
    print(_SEP)
    print("  BATCH EVAL RESULTS")
    print(_SEP)
    print(f"  Runs completed        : {summary['runs_completed']}")
    print(f"  Escalation accuracy   : {summary['escalation_accuracy']:.1%}")
    print(_SEC)
    print("  Eval Scores  (mean across all runs)")
    print(_SEC)
    print(f"  Grounding             : {summary['mean_grounding']:.3f}")
    print(f"  Completeness          : {summary['mean_completeness']:.3f}")
    print(f"  Hallucination Risk    : {summary['mean_hallucination']:.3f}  (lower is better)")
    print(f"  Reasoning Depth       : {summary['mean_reasoning']:.3f}")
    print(f"  Agent Agreement       : {summary['mean_agreement']:.3f}")
    print(f"  Overall Score         : {summary['mean_overall']:.3f}")
    print(_SEC)
    print("  Latency")
    print(_SEC)
    print(f"  Mean latency          : {summary['mean_latency_ms'] / 1000:.1f}s")
    print(f"  P95 latency           : {summary['p95_latency_ms'] / 1000:.1f}s")
    print()
    print(_SEP)
    print(f"  Completed at : {time.strftime('%H:%M:%S')}")
    print(f"  All runs persisted to SQLite")
    print(_SEP)

    return summary


if __name__ == "__main__":
    run_batch_eval()
