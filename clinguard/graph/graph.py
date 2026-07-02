"""
ClinGuard LangGraph pipeline.

Linear flow: extractor → classifier → verifier → END. There is no retry loop —
retrying an identical prompt at temperature 0 reproduces the same answer, so it
added latency without value and risked an unbounded classifier→classifier loop.
Low grader confidence (<0.7) is now handled downstream by the verifier, which
flags the run for human review rather than looping.
"""

import time

from langgraph.graph import END, StateGraph

from clinguard.agents.classifier import classifier_node
from clinguard.agents.extractor import extractor_node
from clinguard.agents.verifier import verifier_node
from clinguard.graph.state import AgentState


def build_graph():
    """
    Build and compile the ClinGuard LangGraph state graph.

    Graph topology (acyclic — cannot loop):
        extractor → classifier → verifier → END
    """
    graph = StateGraph(AgentState)

    # Register all three agent nodes
    graph.add_node("extractor", extractor_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("verifier", verifier_node)

    # Straight-line edges — every node hands off exactly once, no cycles.
    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "classifier")
    graph.add_edge("classifier", "verifier")
    graph.add_edge("verifier", END)

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