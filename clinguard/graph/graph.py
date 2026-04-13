"""
ClinGuard LangGraph pipeline.

Connects the three agents (extractor → classifier → safety_officer)
with conditional routing that retries the classifier when confidence
is low, up to a maximum of 3 retries.
"""

import time

from langgraph.graph import END, StateGraph

from clinguard.agents.classifier import classifier_node
from clinguard.agents.extractor import extractor_node
from clinguard.agents.safety_officer import safety_officer_node
from clinguard.graph.state import AgentState

_MAX_RETRIES = 3
_CONFIDENCE_THRESHOLD = 0.7


def route_after_classifier(state: AgentState) -> str:
    """
    Conditional router executed after the classifier node.

    Routes to safety_officer when confidence is sufficient or retries
    are exhausted; routes back to classifier for a low-confidence retry.
    """
    confidence = float(state.get("confidence") or 0.0)
    retry_count = int(state.get("retry_count") or 0)

    if confidence >= _CONFIDENCE_THRESHOLD or retry_count >= _MAX_RETRIES:
        return "safety_officer"

    # Increment retry counter in state before looping back
    state["retry_count"] = retry_count + 1
    return "classifier"


def build_graph():
    """
    Build and compile the ClinGuard LangGraph state graph.

    Graph topology:
        extractor → classifier →(conditional)→ safety_officer → END
                         ↑______________(retry)__|
    """
    graph = StateGraph(AgentState)

    # Register all three agent nodes
    graph.add_node("extractor", extractor_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("safety_officer", safety_officer_node)

    # Entry point — always start with extraction
    graph.set_entry_point("extractor")

    # Extractor always hands off to classifier
    graph.add_edge("extractor", "classifier")

    # Classifier routes conditionally based on confidence and retry count
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "safety_officer": "safety_officer",
            "classifier": "classifier",
        },
    )

    # Safety officer is always the terminal node
    graph.add_edge("safety_officer", END)

    return graph.compile()


def run_pipeline(report_text: str) -> dict:
    """
    Run the full ClinGuard pipeline on a single AE report.

    Returns the final AgentState dict with a top-level latency_ms field
    reflecting total wall-clock time for the entire pipeline.
    """
    app = build_graph()
    start = time.time()

    initial_state = {
        "report_text": report_text,
        "retry_count": 0,
        "reasoning_trace": [],
    }

    result = app.invoke(initial_state)

    result["latency_ms"] = (time.time() - start) * 1000
    return result


if __name__ == "__main__":
    test_report = (
        "Patient PT-0042 reported severe chest pain and shortness of breath "
        "starting 3 days after dose 2. BP was 88/54, HR 115. "
        "Patient was hospitalized immediately. No prior cardiac history."
    )

    result = run_pipeline(test_report)

    print(f"patient_id      : {result.get('patient_id')}")
    print(f"symptoms        : {result.get('symptoms')}")
    print(f"ctcae_grades    : {result.get('ctcae_grades')}")
    print(f"decision        : {result.get('decision')}")
    print(f"escalation_grade: {result.get('escalation_grade')}")
    print(f"confidence      : {result.get('confidence')}")
    print(f"latency_ms      : {result.get('latency_ms'):.0f}")
    print(f"\ndecision_memo:\n{result.get('decision_memo')}")
    print("\nreasoning_trace:")
    for step in result.get("reasoning_trace") or []:
        print(f"  {step}")