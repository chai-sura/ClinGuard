"""
ClinGuard protocol rule engine.

Evaluates CTCAE grades against clinical trial safety rules and returns
a structured decision for the safety officer agent.
"""

# Cardiac symptoms that trigger immediate escalation at Grade 3
_CARDIAC_TERMS = {"chest pain", "cardiac arrest", "palpitations", "arrhythmia"}


def check_protocol_rule(grades: dict) -> dict:
    """
    Evaluate a symptom→grade mapping against ClinGuard safety rules.

    Rules are checked in priority order (highest severity first).
    Returns a dict with keys: decision, reason, urgency, max_grade.
    """
    max_grade = max(grades.values(), default=0)

    # Rule 1 — Grade 5 anywhere: patient death, FDA report required
    if max_grade == 5:
        return {
            "decision": "escalate",
            "reason": (
                "Grade 5 event detected - patient death reported. "
                "FDA MedWatch report required within 7 days."
            ),
            "urgency": "immediate",
            "max_grade": max_grade,
        }

    # Rule 2 — Grade 4 anywhere: life-threatening, immediate intervention
    if max_grade == 4:
        return {
            "decision": "escalate",
            "reason": (
                "Grade 4 life-threatening event detected. "
                "Immediate physician intervention required."
            ),
            "urgency": "immediate",
            "max_grade": max_grade,
        }

    # Collect all Grade 3 symptoms for Rules 3–5
    grade_3_symptoms = [sym for sym, g in grades.items() if g == 3]

    # Rule 3 — Grade 3 cardiac event: cardiac safety protocol
    if any(sym.lower() in _CARDIAC_TERMS for sym in grade_3_symptoms):
        return {
            "decision": "escalate",
            "reason": (
                "Grade 3 cardiac event detected. "
                "Immediate escalation per cardiac safety protocol."
            ),
            "urgency": "immediate",
            "max_grade": max_grade,
        }

    # Rule 4 — Two or more concurrent Grade 3 events: co-occurrence protocol
    if len(grade_3_symptoms) >= 2:
        return {
            "decision": "escalate",
            "reason": (
                "Multiple concurrent Grade 3 events detected. "
                "Immediate escalation per co-occurrence protocol."
            ),
            "urgency": "immediate",
            "max_grade": max_grade,
        }

    # Rule 5 — Single Grade 3 event: physician review within 24 hours
    if len(grade_3_symptoms) == 1:
        return {
            "decision": "escalate",
            "reason": (
                "Grade 3 event detected. "
                "Physician review required within 24 hours."
            ),
            "urgency": "24h",
            "max_grade": max_grade,
        }

    # Rule 6 — Grade 2 anywhere: flag for next safety review
    if max_grade == 2:
        return {
            "decision": "monitor",
            "reason": (
                "Grade 2 event detected. "
                "Flag for next safety review meeting."
            ),
            "urgency": "routine",
            "max_grade": max_grade,
        }

    # Rule 7 — Grade 1 only: continue standard monitoring
    if max_grade == 1:
        return {
            "decision": "dismiss",
            "reason": (
                "Grade 1 event only. "
                "Continue monitoring per standard protocol."
            ),
            "urgency": "none",
            "max_grade": max_grade,
        }

    # Rule 8 — No grades present
    return {
        "decision": "dismiss",
        "reason": "No gradeable adverse events detected.",
        "urgency": "none",
        "max_grade": max_grade,
    }


if __name__ == "__main__":
    import json

    test_cases = [
        {"chest pain": 3},
        {"nausea": 3, "hypotension": 3},
        {"fatigue": 1},
        {"vomiting": 4},
    ]

    for grades in test_cases:
        print(f"\nInput: {grades}")
        print(json.dumps(check_protocol_rule(grades), indent=2))
