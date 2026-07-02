-- ClinGuard database schema

-- Stores one row per pipeline run with all agent outputs
CREATE TABLE IF NOT EXISTS decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT    NOT NULL UNIQUE,   -- UUID identifying this pipeline run
    patient_id       TEXT,                       -- Extracted patient identifier
    report_text      TEXT,                       -- Raw AE report paragraph
    symptoms         TEXT,                       -- JSON array of symptom strings
    ctcae_grades     TEXT,                       -- JSON object: symptom → grade number
    decision         TEXT,                       -- Final disposition: escalate/monitor/dismiss
    escalation_grade INTEGER,                    -- Highest grade that triggered escalation
    confidence       REAL,                       -- Classifier confidence (0.0–1.0)
    risk_score       REAL,                       -- Composite risk score (0.0–1.0)
    protocol_breach  INTEGER,                    -- 1 if protocol was breached, 0 otherwise
    decision_memo    TEXT,                       -- Full cross-model decision memo
    latency_ms       REAL,                       -- End-to-end pipeline latency in ms
    retry_count      INTEGER,                    -- Number of classifier retries in this run
    -- Cross-model verifier outputs (Stage 6 — previously dropped, audit flaw c)
    verifier_grades    TEXT,                     -- JSON object: verifier (gpt-4o-mini) symptom → grade
    grade_agreement    INTEGER,                  -- 1 if grader/verifier agreed, 0 if not, NULL if unknown
    needs_human_review INTEGER,                  -- 1 if run was flagged for human review, else 0
    created_at       TEXT                        -- ISO 8601 timestamp of insert
);


-- Human-in-the-loop review queue / feedback table (Stage 6).
-- One row per run that was flagged needs_human_review; a reviewer later
-- confirms or overrides the deterministic decision and the verdict is
-- recorded back here. status: pending → confirmed | overridden.
CREATE TABLE IF NOT EXISTS review_queue (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL UNIQUE,    -- References decisions.run_id (one review per run)
    status            TEXT    NOT NULL,           -- pending | confirmed | overridden
    system_decision   TEXT,                       -- Decision the pipeline produced (for the reviewer's context)
    review_reasons    TEXT,                       -- Why it was flagged (disagreement / low confidence / parse fail)
    reviewer_decision TEXT,                        -- Human's final decision (NULL until reviewed)
    reviewer_note     TEXT,                        -- Free-text reviewer note (NULL until reviewed)
    reviewed_at       TEXT,                        -- ISO 8601 timestamp the human acted (NULL until reviewed)
    created_at        TEXT,                        -- ISO 8601 timestamp the run was enqueued
    FOREIGN KEY (run_id) REFERENCES decisions (run_id)
);


-- Stores evaluation scores per run, keyed by run_id
CREATE TABLE IF NOT EXISTS eval_scores (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,          -- References decisions.run_id
    grounding         REAL,                      -- How well grades are grounded in CTCAE
    completeness      REAL,                      -- Whether all symptoms were addressed
    hallucination_risk REAL,                     -- Risk of fabricated clinical detail
    reasoning_depth   REAL,                      -- Quality of the ReAct reasoning steps
    agent_agreement   REAL,                      -- Agreement between classifier and safety officer
    overall_score     REAL,                      -- Average of all five dimensions
    created_at        TEXT,                      -- ISO 8601 timestamp of insert
    FOREIGN KEY (run_id) REFERENCES decisions (run_id)
);