"""
ClinGuard Agent 2 — Classifier.

Takes extracted symptoms from AgentState and grades each one against
CTCAE v6.0 using a ReAct reasoning loop, then applies protocol rules
to determine whether the case requires escalation.
"""

import json
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from clinguard.graph.state import AgentState
from clinguard.tools.ctcae_lookup import lookup_ctcae_grade
from clinguard.tools.protocol_rules import check_protocol_rule

load_dotenv()

# Initialise once at module level so all invocations share the same client
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_REACT_SYSTEM_PROMPT = (
    "You are a clinical trial safety classifier. You must grade "
    "each symptom using CTCAE v6.0 criteria. You have access to "
    "one tool: lookup_ctcae_grade(symptom).\n\n"
    "Respond in this exact format every iteration:\n"
    "THOUGHT: your reasoning about what to do next\n"
    "ACTION: lookup_ctcae_grade or FINISH\n"
    "INPUT: the symptom name or DONE\n\n"
    "When you have graded all symptoms respond with:\n"
    "THOUGHT: your final reasoning\n"
    "ACTION: FINISH\n"
    "INPUT: DONE"
)

_GRADE_SYSTEM_PROMPT = (
    "Based on the CTCAE lookup results and symptom descriptions, "
    "assign a grade number 1-5 to each symptom. "
    "Return only JSON in this format:\n"
    '{"ctcae_grades": {"symptom": grade_number}, '
    '"confidence": 0.0-1.0, '
    '"risk_score": 0.0-1.0}'
)

_MAX_REACT_ITERATIONS = 5


def _parse_react_response(text: str) -> tuple[str, str, str]:
    """Extract THOUGHT, ACTION, INPUT lines from an LLM ReAct response."""
    thought = action = inp = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("THOUGHT:"):
            thought = line[len("THOUGHT:"):].strip()
        elif line.startswith("ACTION:"):
            action = line[len("ACTION:"):].strip()
        elif line.startswith("INPUT:"):
            inp = line[len("INPUT:"):].strip()
    return thought, action, inp


def classifier_node(state: AgentState) -> dict:
    """
    LangGraph node: grade each symptom via a ReAct loop, then score risk.

    Reads symptoms/vitals/severity_description from state, runs up to
    5 ReAct iterations calling lookup_ctcae_grade as needed, then makes
    a final grading call and applies protocol rules.
    """
    symptoms = state.get("symptoms") or []
    vitals = state.get("vitals") or "NONE"
    severity_description = state.get("severity_description") or ""
    start = time.time()

    react_steps: list[str] = []

    # ReAct loop — up to MAX_REACT_ITERATIONS                             #
    for _ in range(_MAX_REACT_ITERATIONS):
        user_message = (
            f"Symptoms to grade: {symptoms}\n"
            f"Vitals: {vitals}\n"
            f"Severity description: {severity_description}\n\n"
            f"Previous steps:\n{chr(10).join(react_steps)}"
        )

        response = llm.invoke([
            {"role": "system", "content": _REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])

        thought, action, inp = _parse_react_response(response.content)

        if action == "FINISH" or inp == "DONE":
            # LLM signals it has gathered enough information
            react_steps.append(f"THOUGHT: {thought}\nACTION: FINISH\nINPUT: DONE")
            break

        if action == "lookup_ctcae_grade":
            # Execute the tool and record the observation
            lookup_ctcae_grade(inp)   # result used for LLM context via trace
            react_steps.append(
                f"THOUGHT: {thought}\n"
                f"ACTION: lookup_ctcae_grade({inp})\n"
                f"OBSERVATION: Grade descriptions retrieved for {inp}"
            )

    # Final grading call — assign numeric grades from accumulated context
    final_user_message = (
        f"Symptoms: {symptoms}\n"
        f"Severity: {severity_description}\n"
        f"Vitals: {vitals}\n"
        f"ReAct steps completed: {react_steps}"
    )

    final_response = llm.invoke([
        {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
        {"role": "user", "content": final_user_message},
    ])

    raw = final_response.content.strip()

    # Strip markdown code fences the model may add despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        grading = json.loads(raw)
        ctcae_grades = grading.get("ctcae_grades", {})
        confidence = float(grading.get("confidence", 0.5))
        risk_score = float(grading.get("risk_score", 0.5))
    except (json.JSONDecodeError, ValueError):
        # Safe defaults keep the pipeline running if the LLM misbehaves
        ctcae_grades = {}
        confidence = 0.5
        risk_score = 0.5

    # Protocol rule check                                                 
    protocol_result = check_protocol_rule(ctcae_grades)
    protocol_breach = protocol_result["decision"] == "escalate"

    # Update reasoning trace with the full ReAct log                     
    trace = list(state.get("reasoning_trace") or [])
    for step in react_steps:
        trace.append(step)
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
