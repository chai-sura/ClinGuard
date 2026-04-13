from __future__ import annotations
from typing import Dict, List, Optional, TypedDict


class AgentState(TypedDict):

    # Input
    report_text: str                # Raw adverse event report paragraph submitted for classification

    # Extractor agent outputs
    patient_id: Optional[str]                 # De-identified patient identifier extracted from the report
    symptoms: Optional[List[str]]             # List of adverse event symptom strings parsed from the report
    severity_description: Optional[str]       # Free-text severity description as written in the report
    timeline: Optional[str]                   # Onset, duration, and progression of the adverse event
    vitals: Optional[str]                     # Relevant vital signs mentioned in the report


    # Classifier agent outputs
    ctcae_grades: Optional[Dict[str, int]]    # Mapping of symptom → CTCAE grade number, e.g. {"chest pain": 3}
    risk_score: Optional[float]               # Composite risk score from grades and patient context (0.0–1.0)
    protocol_breach: Optional[bool]           # Whether the event constitutes a protocol deviation or breach
    confidence: Optional[float]               # Classifier confidence in the assigned grades (0.0–1.0)


    # Safety officer outputs
    decision: Optional[str]                   # Final disposition: one of "escalate", "monitor", or "dismiss"
    decision_memo: Optional[str]              # Human-readable justification for the safety officer's decision
    escalation_grade: Optional[int]           # Highest CTCAE grade that triggered escalation (0 if not escalated)


    # Eval + observability
    eval_scores: Optional[Dict[str, float]]   # Per-metric evaluation scores, e.g. {"accuracy": 0.91, "f1": 0.88}
    reasoning_trace: Optional[List[str]]      # Ordered log of reasoning steps emitted by each agent
    latency_ms: Optional[float]               # End-to-end pipeline latency in milliseconds
    retry_count: Optional[int]                # Number of LLM retries performed across all agents in this run
