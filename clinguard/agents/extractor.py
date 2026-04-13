"""
ClinGuard Agent 1 — Extractor.

Reads a raw adverse event report from AgentState and extracts
structured clinical fields via GPT-4o-mini.
"""

import json
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from clinguard.graph.state import AgentState

load_dotenv()

# Initialise once at module level so all invocations share the same client
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_SYSTEM_PROMPT = (
    "You are a clinical trial data extractor. Read the adverse "
    "event report and extract the following fields as JSON:\n"
    "- patient_id: string (format PT-XXXX, or UNKNOWN if not found)\n"
    "- symptoms: list of strings (each symptom as a clean term)\n"
    "- severity_description: string (how patient described severity)\n"
    "- timeline: string (when symptoms started relative to dose)\n"
    "- vitals: string (any vitals mentioned, or NONE if not present)\n"
    "Return only valid JSON, no explanation, no markdown."
)


def extractor_node(state: AgentState) -> dict:
    """
    LangGraph node: extract structured fields from the raw AE report.

    Reads state["report_text"], calls the LLM, parses the JSON response,
    and returns a partial state dict with all extracted fields.
    """
    report_text = state["report_text"]
    start = time.time()

    # Call the LLM with the extraction prompt
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": report_text},
    ])

    raw = response.content.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        extracted = json.loads(raw)
        patient_id = extracted.get("patient_id", "UNKNOWN")
        symptoms = extracted.get("symptoms", [])
        severity_description = extracted.get("severity_description", "")
        timeline = extracted.get("timeline", "")
        vitals = extracted.get("vitals", "NONE")
    except json.JSONDecodeError:
        # Return safe defaults so the pipeline can continue
        patient_id = "UNKNOWN"
        symptoms = []
        severity_description = ""
        timeline = ""
        vitals = "NONE"

    # Append this agent's step to the shared reasoning trace
    trace = list(state.get("reasoning_trace") or [])
    trace.append(
        f"Extractor: extracted {len(symptoms)} symptoms: {symptoms}"
    )

    return {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "severity_description": severity_description,
        "timeline": timeline,
        "vitals": vitals,
        "reasoning_trace": trace,
        "latency_ms": (time.time() - start) * 1000,
    }


if __name__ == "__main__":
    test_report = (
        "Patient PT-0023 reported severe chest pain and dizziness "
        "starting 2 days after dose 3. BP was 90/60, HR 110. "
        "Patient was unable to stand without assistance."
    )

    result = extractor_node({"report_text": test_report, "reasoning_trace": []})

    import pprint
    pprint.pprint(result)
