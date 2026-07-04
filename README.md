# ClinGuard

**Agentic triage for clinical-trial adverse events — grounded severity grading, a deterministic safety decision, and cross-model verification with a human in the loop.**

🔗 **Live demo:** [clinicalguard.streamlit.app](https://clinicalguard.streamlit.app)
🎥 **Demo video:** [▶ Watch the walkthrough](Demo-app.mp4)

> Prototype for demonstration only — not a medical device, and not for clinical use. All metrics below are on **synthetic** data.

---

## The problem

Clinical trials generate a constant stream of free-text adverse-event (AE) reports — messy, abbreviated, and inconsistent. Each one has to be graded for severity and triaged (escalate, monitor, or dismiss), and getting it wrong in either direction is costly: a missed escalation is a safety risk, and over-escalation drowns reviewers in noise. Today this is slow, manual, and high-stakes.

## What it does

ClinGuard reads an AE report and runs it through a short, auditable pipeline:

1. **Extracts** the symptoms and vitals from the raw text.
2. **Grades** each symptom's severity, **grounded in the retrieved CTCAE v6.0 criteria** (the official medical severity scale) rather than model memory.
3. **Decides** escalate / monitor / dismiss with a **deterministic rules engine** — not an LLM.
4. **Verifies** the grades with a **different model** to catch disagreement.
5. **Routes** low-confidence or disagreement cases to a **human-review queue**, and logs every run to a full **audit trail**.

## Architecture

```
Report ─► Extractor ─► Grader ─────► Rules engine ─────► Verifier ─────► Human review
          (Claude)     (Claude,        (deterministic       (gpt-4o-mini,   (queue + audit log)
                        CTCAE v6.0      code — makes the     independent
                        grounded)       decision)            cross-check)
```

Two deliberate choices define the design:

- **The decision is made by deterministic code, not an LLM.** The rules engine maps CTCAE grades to a disposition through fixed clinical-safety rules. The LLMs extract and grade; they never make the call. This keeps the safety-critical step transparent and testable.
- **A different model verifies the primary grader.** The grader runs on **Claude**; an independent **OpenAI gpt-4o-mini** re-grades the same symptoms. Using a separate model avoids self-evaluation bias — agreement is a genuine cross-check, not a model grading its own work. Disagreement (or low grader confidence) flags the case for a human.

## Results

Measured on **150 synthetic, label-by-construction cases** (target grade chosen first, then observable text generated to match — no severity leakage). See [RESULTS.md](RESULTS.md).

| Metric | Result |
| --- | --- |
| Standard grade accuracy | **81.6%** (102/125) |
| Escalation accuracy | **94.4%** (118/125) |
| Hard-case grade accuracy | **84.0%** (21/25) |
| Cross-model agreement (decision-level) | **94%** |
| Human-review rate | **~11%** |

**Safety framing.** On the standard cases there were **zero dangerous misses** — no should-escalate event was ever under-decided. Across all 150 cases there was exactly **one** under-escalation, on a deliberately ambiguous *boundary* case (a grade-3 that read as grade-2). The cross-model verifier disagreed on it and it was **routed to human review** — the safety net working as designed. Grade errors skew toward *over*-grading (the safe direction), which is why escalation accuracy stays high even where raw grade accuracy is middling.

## Key engineering decisions

- **Retrieval-grounded grading.** Grades are driven by the actual CTCAE v6.0 criterion text retrieved per symptom, not the model's memory of the rulebook — and the retrieved text is shown in the UI as the grounding proof.
- **Deterministic safety guardrail.** The escalate/monitor/dismiss decision is fixed code with explicit protocol rules, so the critical step is auditable and can't hallucinate.
- **Cross-model verification.** A second, different model re-grades every case, so the confidence signal isn't self-referential.
- **Human-in-the-loop.** Uncertain cases (disagreement or low confidence) are persisted to a review queue where a reviewer confirms or overrides, and the action is logged.
- **Honest evaluation.** The harness scores against ground-truth labels on a purpose-built synthetic set — not an LLM grading its own family's output.

## Known limitations

Framed as understood trade-offs (see [AUDIT.md](AUDIT.md)):

- **Synthetic data, not real PHI.** Every case is constructed; the system has not been evaluated on real patient reports.
- **~82% live grade accuracy**, with **numeric-threshold cases a known miss source** — grade-3 boundary calls that hinge on exact lab/vital cutoffs are the stubbornest band and don't improve with better extraction; they're inherent grade-boundary ambiguity.
- **Borderline grade-1/2 cases can flip between runs** — genuinely two-way calls, non-dangerous.
- These uncertain cases are exactly what the **verifier and confidence gate are there to catch** and hand to a human, rather than silently deciding.

## Tech stack

Python · [LangGraph](https://github.com/langchain-ai/langgraph) · Claude (Haiku) · OpenAI gpt-4o-mini · Streamlit · SQLite · CTCAE v6.0

## Future work

- **OCR intake for field use:** photo of a handwritten note → OCR → **human verification of the transcription** → into the pipeline. This would let field staff capture AE reports from paper without manual data entry, while keeping a human checkpoint on the error-prone OCR step.

## Run locally

```bash
git clone https://github.com/chai-sura/ClinGuard.git
cd ClinGuard
python3 -m venv cenv && source cenv/bin/activate
pip install -r requirements.txt

# Live analysis needs both keys (Demo/Examples work without them):
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EOF

streamlit run app.py
```

Open http://localhost:8501. The **Examples** tab replays real pre-computed traces with no API calls; **Analyze a report** runs the live pipeline (requires the keys above).
