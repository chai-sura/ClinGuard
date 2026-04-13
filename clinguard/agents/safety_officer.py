"""
ClinGuard Agent 3 — Safety Officer.

Acts as the critic agent. Reads all extractor and classifier outputs,
evaluates reasoning quality, and produces a final decision memo.
"""

import json
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from clinguard.graph.state import AgentState
from clinguard.tools.protocol_rules import check_protocol_rule

load_dotenv()

# Initialise once at module level so all invocations share the same client
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_SYSTEM_PROMPT = (
    "You are a senior clinical trial safety officer reviewing "
    "an AI-generated adverse event classification. Your job is to:\n"
    "1. Verify the grading makes sense given the symptoms described\n"
    "2. Confirm or override the protocol decision\n"
    "3. Write a clear decision memo for the clinical team\n\n"
    "Be concise, clinical, and decisive. Always cite the specific "
    "CTCAE grades in your memo."
)

# Low-confidence threshold — cases below this get a FLAG annotation
_CONFIDENCE_THRESHOLD = 0.7


def safety_officer_node(state: AgentState) -> dict:
    """
    LangGraph node: review classifier output and produce a final decision.

    Calls the protocol rule engine for a deterministic baseline decision,
    then asks the LLM to verify grading quality and write the decision memo.
    """
    start = time.time()

    # Step 1 — deterministic protocol decision as grounding context       
    ctcae_grades = state.get("ctcae_grades") or {}
    protocol_result = check_protocol_rule(ctcae_grades)
    decision_proto = protocol_result["decision"]
    urgency = protocol_result["urgency"]
    reason = protocol_result["reason"]

    # Step 2 — LLM reviews evidence and writes the decision memo         
    reasoning_trace = state.get("reasoning_trace") or []

    user_message = (
        f"Patient ID: {state.get('patient_id')}\n\n"
        f"Original report:\n{state.get('report_text')}\n\n"
        f"Extracted symptoms: {state.get('symptoms')}\n"
        f"Vitals: {state.get('vitals')}\n"
        f"Timeline: {state.get('timeline')}\n\n"
        f"CTCAE grades assigned: {ctcae_grades}\n"
        f"Risk score: {state.get('risk_score')}\n"
        f"Confidence: {state.get('confidence')}\n\n"
        f"Protocol rule triggered: {reason}\n"
        f"Recommended decision: {decision_proto}\n"
        f"Urgency: {urgency}\n\n"
        f"Reasoning trace from classifier:\n"
        f"{chr(10).join(reasoning_trace)}\n\n"
        f"Return JSON only in this format:\n"
        f"{{\n"
        f"  \"decision\": \"escalate\" or \"monitor\" or \"dismiss\",\n"
        f"  \"decision_memo\": \"full memo text here\",\n"
        f"  \"escalation_grade\": highest grade number as integer,\n"
        f"  \"confidence_check\": \"PASS\" or \"FLAG\",\n"
        f"  \"confidence_note\": \"one sentence on reasoning quality\"\n"
        f"}}"
    )

    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    raw = response.content.strip()

    # Strip markdown code fences the model may add despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
        decision = result.get("decision", decision_proto)
        decision_memo = result.get("decision_memo", "")
        escalation_grade = int(result.get("escalation_grade", 0))
        confidence_check = result.get("confidence_check", "PASS")
        confidence_note = result.get("confidence_note", "")
    except (json.JSONDecodeError, ValueError):
        # Always escalate on parse error — safer than silently dismissing
        decision = "escalate"
        decision_memo = "Parse error - defaulting to escalate for safety"
        escalation_grade = 0
        confidence_check = "FLAG"
        confidence_note = "LLM response could not be parsed"

    # Step 3 — flag low-confidence classifications for human review       
    classifier_confidence = float(state.get("confidence") or 0.0)
    if classifier_confidence < _CONFIDENCE_THRESHOLD:
        confidence_check = "FLAG"
        decision_memo += (
            " [FLAGGED: Low classifier confidence - human review recommended]"
        )

    # Update reasoning trace                                              
    trace = list(reasoning_trace)
    trace.append(
        f"Safety Officer: decision={decision}, "
        f"grade={escalation_grade}, "
        f"confidence_check={confidence_check}"
    )

    return {
        "decision": decision,
        "decision_memo": decision_memo,
        "escalation_grade": escalation_grade,
        "reasoning_trace": trace,
        "latency_ms": (time.time() - start) * 1000,
    }


if __name__ == "__main__":
    import pprint

    test_state = {
        "report_text": (
            "Patient PT-0023 reported severe chest pain and dizziness "
            "2 days after dose 3. BP 90/60, HR 110."
        ),
        "patient_id": "PT-0023",
        "symptoms": ["chest pain", "hypotension"],
        "severity_description": "severe, unable to stand",
        "timeline": "2 days after dose 3",
        "vitals": "BP 90/60, HR 110",
        "ctcae_grades": {"chest pain": 3, "hypotension": 3},
        "risk_score": 0.85,
        "confidence": 0.91,
        "protocol_breach": True,
        "reasoning_trace": [
            "Extractor: extracted 2 symptoms: chest pain, hypotension",
            "Classifier THOUGHT: both symptoms are Grade 3",
            "Classifier ACTION: lookup_ctcae_grade(chest pain)",
            "Classifier OBSERVATION: Grade 3 - hospitalization indicated",
        ],
    }

    result = safety_officer_node(test_state)
    pprint.pprint(result)
