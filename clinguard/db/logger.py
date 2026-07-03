"""
ClinGuard database logger.

Handles SQLite persistence for pipeline runs and evaluation scores.
Call init_db() once at application startup to ensure tables exist.
"""

import json
import os
import re
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


def _extract_review_reasons(memo) -> str:
    """
    Pull the human-review reasons out of the verifier's decision memo.

    The verifier appends '[FLAGGED FOR HUMAN REVIEW: <reasons>]' to the memo;
    lift that substring so the queue row carries why it was flagged. Returns
    '' if the memo is missing or unflagged.
    """
    if not memo:
        return ""
    m = re.search(r"FLAGGED FOR HUMAN REVIEW:\s*(.*?)\]", memo, re.S)
    return m.group(1).strip() if m else ""


# Columns added to `decisions` after the table first shipped. CREATE TABLE
# IF NOT EXISTS won't add them to a pre-existing DB, so init_db() ALTERs them
# in for older databases. (name, SQL type)
_DECISION_MIGRATIONS = [
    ("verifier_grades", "TEXT"),
    ("grade_agreement", "INTEGER"),
    ("needs_human_review", "INTEGER"),
]


def _migrate_decisions(conn: sqlite3.Connection) -> None:
    """
    Additively bring an existing `decisions` table up to the current schema.

    SQLite has no ADD COLUMN IF NOT EXISTS, so we diff against PRAGMA
    table_info and ALTER in only the missing columns. Additive + nullable —
    existing rows are preserved (older runs get NULL for the new fields).
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(decisions)")}
    for name, sql_type in _DECISION_MIGRATIONS:
        if name not in existing:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {name} {sql_type}")


def init_db() -> None:
    """
    Create the tables if they don't exist, then migrate older databases.

    Reads DDL from schema.sql and executes it as a script (idempotent — safe
    on every startup), then ALTERs any columns missing from a pre-existing
    `decisions` table so old and new databases converge on one schema.
    """
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with _connect() as conn:
        conn.executescript(schema_sql)
        _migrate_decisions(conn)


def log_decision(state: dict) -> str:
    """
    Insert one pipeline run into the decisions table.

    Serializes list/dict fields to JSON strings before insertion.
    Returns the generated run_id so it can be passed to log_eval().
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Normalize the verifier outputs (Stage 6). grade_agreement is tri-state:
    # 1 agreed / 0 disagreed / NULL if the field was never set.
    agreement = state.get("grade_agreement")
    grade_agreement = None if agreement is None else (1 if agreement else 0)
    needs_review = 1 if state.get("needs_human_review") else 0

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO decisions (
                run_id, patient_id, report_text, symptoms, ctcae_grades,
                decision, escalation_grade, confidence, risk_score,
                protocol_breach, decision_memo, latency_ms, retry_count,
                verifier_grades, grade_agreement, needs_human_review,
                created_at
            ) VALUES (
                :run_id, :patient_id, :report_text, :symptoms, :ctcae_grades,
                :decision, :escalation_grade, :confidence, :risk_score,
                :protocol_breach, :decision_memo, :latency_ms, :retry_count,
                :verifier_grades, :grade_agreement, :needs_human_review,
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
                "verifier_grades": json.dumps(state.get("verifier_grades") or {}),
                "grade_agreement": grade_agreement,
                "needs_human_review": needs_review,
                "created_at": now,
            },
        )

        # HITL: enqueue flagged runs for human review. OR IGNORE keeps the
        # UNIQUE(run_id) constraint idempotent if log_decision is ever retried.
        if needs_review:
            conn.execute(
                """
                INSERT OR IGNORE INTO review_queue (
                    run_id, status, system_decision, review_reasons, created_at
                ) VALUES (:run_id, 'pending', :system_decision, :review_reasons, :created_at)
                """,
                {
                    "run_id": run_id,
                    "system_decision": state.get("decision"),
                    "review_reasons": _extract_review_reasons(state.get("decision_memo")),
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


def get_review_queue(status: str = "pending", limit: int = 50) -> list:
    """
    Return runs in the human-review queue, newest first.

    Joins review_queue to decisions so each row carries the clinical context a
    reviewer needs (patient, grades, decision, memo). Pass status=None to list
    the whole queue regardless of state.
    """
    query = """
        SELECT
            q.run_id, q.status, q.system_decision, q.review_reasons,
            q.reviewer_decision, q.reviewer_note, q.reviewed_at, q.created_at,
            d.patient_id, d.report_text, d.ctcae_grades, d.verifier_grades,
            d.grade_agreement, d.confidence, d.decision, d.decision_memo
        FROM review_queue q
        JOIN decisions d ON q.run_id = d.run_id
        {where}
        ORDER BY q.created_at DESC
        LIMIT :limit
    """
    params = {"limit": limit}
    if status is None:
        query = query.format(where="")
    else:
        query = query.format(where="WHERE q.status = :status")
        params["status"] = status

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def record_human_review(
    run_id: str, reviewer_decision: str, reviewer_note: str = ""
) -> bool:
    """
    Record a human's verdict on a flagged run and close out its queue entry.

    status becomes 'confirmed' if the human agrees with the pipeline's
    decision, else 'overridden'. Returns True if a pending row was updated,
    False if the run_id wasn't in the queue.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT system_decision FROM review_queue WHERE run_id = :run_id",
            {"run_id": run_id},
        ).fetchone()
        if row is None:
            return False
        status = "confirmed" if reviewer_decision == row["system_decision"] else "overridden"
        conn.execute(
            """
            UPDATE review_queue
               SET status = :status,
                   reviewer_decision = :reviewer_decision,
                   reviewer_note = :reviewer_note,
                   reviewed_at = :reviewed_at
             WHERE run_id = :run_id
            """,
            {
                "status": status,
                "reviewer_decision": reviewer_decision,
                "reviewer_note": reviewer_note,
                "reviewed_at": now,
                "run_id": run_id,
            },
        )
    return True


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
