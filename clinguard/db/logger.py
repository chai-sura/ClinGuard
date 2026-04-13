"""
ClinGuard database logger.

Handles SQLite persistence for pipeline runs and evaluation scores.
Call init_db() once at application startup to ensure tables exist.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone


# Database file sits at the project root, one level above clinguard/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(_PROJECT_ROOT, "clinguard.db")

# Schema file lives alongside this module
_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def _connect() -> sqlite3.Connection:
    """Open a connection with row_factory so queries return dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the database and both tables if they don't already exist.

    Reads DDL from schema.sql and executes it as a script so the function
    is idempotent — safe to call on every startup.
    """
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with _connect() as conn:
        conn.executescript(schema_sql)


def log_decision(state: dict) -> str:
    """
    Insert one pipeline run into the decisions table.

    Serializes list/dict fields to JSON strings before insertion.
    Returns the generated run_id so it can be passed to log_eval().
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO decisions (
                run_id, patient_id, report_text, symptoms, ctcae_grades,
                decision, escalation_grade, confidence, risk_score,
                protocol_breach, decision_memo, latency_ms, retry_count,
                created_at
            ) VALUES (
                :run_id, :patient_id, :report_text, :symptoms, :ctcae_grades,
                :decision, :escalation_grade, :confidence, :risk_score,
                :protocol_breach, :decision_memo, :latency_ms, :retry_count,
                :created_at
            )
            """,
            {
                "run_id": run_id,
                "patient_id": state.get("patient_id"),
                "report_text": state.get("report_text"),
                "symptoms": json.dumps(state.get("symptoms") or []),
                "ctcae_grades": json.dumps(state.get("ctcae_grades") or {}),
                "decision": state.get("decision"),
                "escalation_grade": state.get("escalation_grade"),
                "confidence": state.get("confidence"),
                "risk_score": state.get("risk_score"),
                "protocol_breach": 1 if state.get("protocol_breach") else 0,
                "decision_memo": state.get("decision_memo"),
                "latency_ms": state.get("latency_ms"),
                "retry_count": state.get("retry_count", 0),
                "created_at": now,
            },
        )

    return run_id


def log_eval(run_id: str, eval_scores: dict) -> None:
    """
    Insert evaluation scores for a completed run into eval_scores.

    Calculates overall_score as the mean of the five quality dimensions.
    """
    dimensions = [
        "grounding",
        "completeness",
        "hallucination_risk",
        "reasoning_depth",
        "agent_agreement",
    ]
    scores = [float(eval_scores.get(d, 0.0)) for d in dimensions]
    overall_score = sum(scores) / len(scores)
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_scores (
                run_id, grounding, completeness, hallucination_risk,
                reasoning_depth, agent_agreement, overall_score, created_at
            ) VALUES (
                :run_id, :grounding, :completeness, :hallucination_risk,
                :reasoning_depth, :agent_agreement, :overall_score, :created_at
            )
            """,
            {
                "run_id": run_id,
                "grounding": scores[0],
                "completeness": scores[1],
                "hallucination_risk": scores[2],
                "reasoning_depth": scores[3],
                "agent_agreement": scores[4],
                "overall_score": overall_score,
                "created_at": now,
            },
        )


def get_recent_runs(limit: int = 10) -> list:
    """
    Return the most recent pipeline runs with their eval scores.

    Joins decisions with eval_scores on run_id. Runs without eval scores
    are still included (LEFT JOIN), with overall_score as None.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                d.run_id, d.patient_id, d.decision, d.escalation_grade,
                d.confidence, d.risk_score, d.protocol_breach, d.latency_ms,
                d.retry_count, d.created_at,
                e.overall_score
            FROM decisions d
            LEFT JOIN eval_scores e ON d.run_id = e.run_id
            ORDER BY d.created_at DESC
            LIMIT :limit
            """,
            {"limit": limit},
        ).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    import pprint

    # Initialise tables (idempotent)
    init_db()
    print(f"Database initialised at: {DB_PATH}")

    # Log a sample pipeline run
    sample_state = {
        "patient_id": "PT-0099",
        "report_text": "Patient reported severe nausea and fatigue after dose 1.",
        "symptoms": ["nausea", "fatigue"],
        "ctcae_grades": {"nausea": 2, "fatigue": 1},
        "decision": "monitor",
        "escalation_grade": 2,
        "confidence": 0.88,
        "risk_score": 0.35,
        "protocol_breach": False,
        "decision_memo": "Grade 2 nausea flagged for next safety review.",
        "latency_ms": 5200.0,
        "retry_count": 0,
    }
    run_id = log_decision(sample_state)
    print(f"Logged decision with run_id: {run_id}")

    # Log evaluation scores for that run
    sample_eval = {
        "grounding": 0.90,
        "completeness": 0.85,
        "hallucination_risk": 0.10,
        "reasoning_depth": 0.80,
        "agent_agreement": 0.95,
    }
    log_eval(run_id, sample_eval)
    print(f"Logged eval scores for run_id: {run_id}")

    # Retrieve and display recent runs
    print("\nRecent runs:")
    pprint.pprint(get_recent_runs(limit=5))
