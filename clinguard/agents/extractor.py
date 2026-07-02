"""
ClinGuard Agent 1 — Extractor.

Reads a raw adverse event report from AgentState and extracts
structured clinical fields via the configured Claude model.
"""

import json
import time

from dotenv import load_dotenv

from clinguard.config import get_chat_model
from clinguard.graph.state import AgentState

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a clinical trial data extractor. Read the adverse "
    "event report and extract the following fields as JSON:\n"
    "- patient_id: string (format PT-XXXX, or UNKNOWN if not found)\n"
    "- symptoms: list of strings. Name each adverse event using the standard "
    "CTCAE clinical term, NOT the patient's lay wording — e.g. "
    "'increased thirst and urination' -> 'hyperglycemia'; "
    "'couldn't catch my breath' -> 'dyspnea'; 'throwing up' -> 'vomiting'. "
    "Also infer the underlying condition implied by abnormal lab values or "
    "vital signs and list it as a symptom: elevated blood glucose -> "
    "'hyperglycemia', high blood pressure -> 'hypertension', low blood "
    "pressure -> 'hypotension', low neutrophils -> 'neutropenia', low "
    "platelets -> 'thrombocytopenia', low potassium -> 'hypokalemia'. "
    "Only include conditions the report actually supports — do NOT invent "
    "findings from thin or absent evidence, and do NOT put a severity word "
    "or grade in the term (name the condition only). "
    "If the report explicitly states the patient died or expired, include "
    "'death' as a symptom (a CTCAE grade-5 event). Fire this ONLY on an "
    "explicit death/expiration statement — do NOT infer death from "
    "severe-but-survived language such as 'nearly died', 'life-threatening', "
    "'coded but was revived', or 'stabilized after'.\n"
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

    llm = get_chat_model("extractor")

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
