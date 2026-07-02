# ClinGuard — Audit

## What it does
- Takes a free-text clinical-trial adverse-event report and extracts structured fields (patient ID, symptoms, vitals, timeline) via an LLM.
- Grades each symptom against CTCAE v6.0, applies deterministic protocol rules, and outputs an escalate/monitor/dismiss decision with a memo.
- Scores each run with an LLM-as-judge, persists everything to SQLite, and surfaces it in a Streamlit dashboard.

## What works
| Component | Status | Reason |
|---|---|---|
| extractor | WORKS | Single LLM call parses report → JSON fields with fence-stripping and safe defaults on parse failure. |
| classifier | PARTIAL | Produces grades/risk/confidence, but the CTCAE lookup result is thrown away (see flaw a); grading is ungrounded LLM guessing. |
| ctcae_lookup | WORKS | Loads the Excel sheet once, does exact→substring term matching, returns grade text or a not-found dict. |
| protocol_rules | WORKS | Deterministic priority-ordered rule engine over the grade map; fully testable, no LLM. |
| judge | PARTIAL | Runs and scores 5 dimensions, but uses the same model/temp as the pipeline agents (see flaw b) — not truly independent. |
| run_evals | WORKS | Batch-runs fixtures, scores, persists, prints escalation accuracy + latency; wraps each run in try/except. |
| db | PARTIAL | decisions + eval_scores insert/join/read all work, but there is no feedback/human-review table (see flaw c). |
| app | WORKS | Streamlit UI runs the pipeline, shows decision/grades/memo/trace/eval, logs runs, lists recent decisions. |

## Known flaws
- **(a) CTCAE lookup discarded — CONFIRMED.** In [classifier.py:102](clinguard/agents/classifier.py#L102) `lookup_ctcae_grade(inp)` is called but its return value is dropped; only a static `"Grade descriptions retrieved"` string enters the trace, so the final grading call never sees actual CTCAE criteria — grades are LLM-invented, not grounded.
- **(b) Self-evaluation — CONFIRMED.** [judge.py:16](clinguard/eval/judge.py#L16) instantiates `ChatOpenAI(model="gpt-4o-mini", temperature=0)`, the identical model used by every agent ([extractor.py:19](clinguard/agents/extractor.py#L19), [classifier.py:22](clinguard/agents/classifier.py#L22), [safety_officer.py:19](clinguard/agents/safety_officer.py#L19)); the comment itself admits "same model," so the judge grades its own family's output.
- **(c) No human-in-the-loop persistence — CONFIRMED.** [schema.sql](clinguard/db/schema.sql) defines only `decisions` and `eval_scores`; the UI shows a "Requires Human Review" badge and a low-confidence FLAG, but no feedback table or write path exists to capture human corrections.
- **(d) ReAct loop unnecessary — CONFIRMED.** The loop in [classifier.py:80-107](clinguard/agents/classifier.py#L80-L107) discards tool output and is always followed by a single fixed grading call ([classifier.py:117](clinguard/agents/classifier.py#L117)); for a fixed grade-each-symptom task a direct lookup + grade would give the same result with less latency and cost.

## Additional flaws

### Correctness
- [graph.py:36](clinguard/graph/graph.py#L36) — HIGH — router mutates `state["retry_count"]` in place, but LangGraph does not persist mutations from conditional-edge functions (only node return values); the classifier never returns `retry_count`, so it stays 0. Any report with confidence < 0.7 loops classifier→classifier until the recursion limit → `GraphRecursionError` crash (or, best case, retries do nothing).
- [logger.py:103-104](clinguard/db/logger.py#L103-L104) — MED — stored `overall_score` averages the **raw** `hallucination_risk` without inverting it, contradicting [judge.py:113-119](clinguard/eval/judge.py#L113-L119) which uses `(1 - hallucination_risk)`; the dashboard's "Score" column is therefore wrong/inconsistent with the judge.
- [ctcae_lookup.py:69-71](clinguard/tools/ctcae_lookup.py#L69-L71) — MED — substring match tests `query in term` and returns `iloc[0]` (first row); short queries ("pain") match many terms and return an arbitrary/wrong one, and the match is directional (won't match "chest pain, radiating" → "chest pain"). Moot today since the result is discarded (flaw a), but wrong if ever used.
- [protocol_rules.py:49](clinguard/tools/protocol_rules.py#L49) — LOW — cardiac trigger uses exact set membership on `sym.lower()`, so any symptom string not identical to the four hardcoded terms silently bypasses the cardiac protocol.

### Error handling
- [ctcae_lookup.py:51](clinguard/tools/ctcae_lookup.py#L51) — MED — `_load_data()` runs at import with no try/except; a missing/renamed Excel file (name is hardcoded with a date, [line 14](clinguard/tools/ctcae_lookup.py#L14)) crashes import of the whole app.
- [extractor.py:58-65](clinguard/agents/extractor.py#L58-L65) — MED — `except json.JSONDecodeError` only; if the LLM returns a JSON array/scalar, `json.loads` succeeds and `.get()` raises an uncaught `AttributeError`. Same pattern risk in [classifier.py:131-136](clinguard/agents/classifier.py#L131-L136).
- [logger.py:54](clinguard/db/logger.py#L54) — LOW — no try/except around DB writes; concurrent Streamlit sessions can hit SQLite `database is locked`. `eval_scores.run_id` has no UNIQUE constraint, so double-logging fans out the LEFT JOIN in [get_recent_runs](clinguard/db/logger.py#L131).

### Input validation / security
- [app.py:421](clinguard/app.py#L421) — MED — LLM-generated `decision_memo` (and `patient_id`/symptom pills) rendered via `unsafe_allow_html=True` without escaping (unlike the trace at [line 443](clinguard/app.py#L443)) → HTML/script injection from model output.
- [app.py:313](clinguard/app.py#L313) — MED — only `report_text.strip()` is validated; no length cap, so an arbitrarily large paste is sent to every agent + judge (uncontrolled token cost).

### Cost / token controls
- [classifier.py:80-107](clinguard/agents/classifier.py#L80-L107) — MED — up to 5 paid ReAct LLM calls per run whose outputs are discarded, plus `react_steps` is echoed back into each prompt (growing context), then a 6th grading call — pure wasted spend.
- [extractor.py:19](clinguard/agents/extractor.py#L19) — LOW — no `max_tokens` on any `ChatOpenAI` (all 4 agents/judge); no output cap.

### Testability
- [extractor.py:19](clinguard/agents/extractor.py#L19), [classifier.py:22](clinguard/agents/classifier.py#L22), [safety_officer.py:19](clinguard/agents/safety_officer.py#L19), [judge.py:16](clinguard/eval/judge.py#L16) — MED — `ChatOpenAI` instantiated at module import; importing any module requires `OPENAI_API_KEY` and hits config eagerly, blocking unit tests / mocking.
- [app.py:15](clinguard/app.py#L15), [main.py:19](main.py#L19), [run_evals.py:26-27](clinguard/eval/run_evals.py#L26-L27) — LOW — fragile `sys.path.insert` hacks and dual import styles (`graph.graph` vs `clinguard.agents`) instead of a proper installable package.
- repo-wide — MED — no tests exist; only ad-hoc `__main__` blocks that all require a live API key.

### Config / dependencies
- [requirements.txt:1-8](requirements.txt#L1-L8) — MED — zero version pins; `langgraph`/`langchain-openai` drift can break the graph API silently. Reproducibility risk.
- [requirements.txt:8](requirements.txt#L8) — MED — `sentence-transformers` (pulls torch, large) is declared but never imported anywhere in the code — dead heavyweight dependency.
- [extractor.py:19](clinguard/agents/extractor.py#L19) et al. — LOW — model id `"gpt-4o-mini"` hardcoded in 4 places; no central config/env override.

### Secrets
- [.gitignore:4,7](.gitignore#L4-L7) — LOW (OK) — `.env` and `*.db` are gitignored and not tracked (`git ls-files` confirms); `.env.example` ships only a placeholder key. No secret leak found — noted for completeness.

### Grounding / extraction (found via 150-case eval)
- [extractor.py](clinguard/agents/extractor.py) — MED — **extractor terminology gap: the grounded CTCAE lookup never fires on the real events because the extractor emits lay descriptions instead of CTCAE-canonical terms, and does not infer conditions from lab/vital values.** Example: `syn_0043` (target g2). Report states glucose 210 mg/dL with oral metformin started, and BP 150/95. The extractor produced `increased thirst / increased urination / persistent cough`; the classifier graded those (all g1) and dismissed. The actual gradeable events — `hyperglycemia` (glucose 210 + drug initiated → CTCAE g2) and `hypertension` (150/95 + intervention) — were never surfaced, so [ctcae_lookup.lookup_ctcae_grade](clinguard/tools/ctcae_lookup.py) was never called on them. Root cause is upstream of grading: the grounding fix (Stage 3) is only as good as the symptom strings it is handed. Direction is non-dangerous here (g2 under-graded to dismiss, not a missed escalation) and the low-confidence gate flagged it for review (conf 0.35), but the mechanism would equally suppress a higher-acuity event whose only signal is a lab/vital value.
  - **RESOLVED (Option A applied + measured).** The extractor prompt now (1) names events with CTCAE-canonical terms, (2) infers conditions from lab/vital values (glucose → `hyperglycemia`, BP → `hypertension`, etc.), and (3) extracts `death` (grade 5) on explicit death/expiration statements only. Measured before/after on the full 150-case eval: standard grade accuracy **70.4% → 81.6% (+11.2pp)**, hard **48% → 84% (+36pp)**, standard g4 **56% → 92%** and g5 **72% → 92%**, escalation held at 94.4%. See [RESULTS.md](RESULTS.md). Option B (normalization component) was NOT needed and remains unbuilt.

### Known limitations (measured on the 150-case eval, accepted — not extraction-fixable)
- **Grade-3 band ambiguity** — MED — standard grade-3 accuracy sits at **52%** and did **not** move with the extractor fix (52% before and after). This is inherent grade-boundary ambiguity (CTCAE g2↔g3↔g4 hinge on judgment calls the report often underdetermines), not a terminology/extraction problem, so canonicalization can't touch it. Accepted as a floor; would need grading-side work (few-shot boundary exemplars or a grade-specific rubric), not extraction.
- **Borderline g1/g2 instability** — LOW — some cases sit exactly on the CTCAE g1/g2 line and flip between runs at temperature 0. Example: `syn_0046` (nausea + ondansetron, patient home and eating) graded g2/monitor in the targeted sample but g1/dismiss in the full re-run. This is case-inherent ambiguity (a genuinely two-way call), **not** temperature-driven and not a bug; the deterministic gate keeps the outcome non-dangerous (dismiss↔monitor, never a missed escalation). Accepted.
- **One dangerous miss in the boundary set** — noted — across all 150, exactly one should-escalate case was under-decided: `syn_0132` (hard/boundary, target g3 read as g2 → monitor instead of escalate). Boundary cases are deliberately constructed to sit on the escalate/monitor line; standard cases had **zero** dangerous misses. Tracked, not a regression from the extractor change.

## Data flow
```
run_pipeline(report_text)
        │  initial state: {report_text, retry_count=0, reasoning_trace=[]}
        ▼
   ┌──────────┐   report_text
   │ extractor│ ─────────────► sets: patient_id, symptoms,
   └──────────┘                severity_description, timeline, vitals
        │
        ▼
   ┌──────────┐   symptoms/vitals/severity
   │classifier│ ─── ReAct loop → lookup_ctcae_grade() [result dropped]
   └──────────┘ ─── final LLM grade → ctcae_grades, risk_score,
        │            confidence, protocol_breach (via check_protocol_rule)
        │
        ▼
   route_after_classifier
     confidence < 0.7 and retry_count < 3 ──► back to classifier (retry++)
     else ─────────────────────────────────► safety_officer
        │
        ▼
   ┌──────────────┐  ctcae_grades + protocol rule
   │safety_officer│ ─► decision, decision_memo, escalation_grade,
   └──────────────┘    confidence_check (FLAG if confidence < 0.7)
        │
        ▼  END → final AgentState (+ total latency_ms)
        │
        ├─► evaluate_decision(state)  → eval_scores (judge LLM)
        ├─► log_decision(state)       → decisions table (run_id)
        └─► log_eval(run_id, scores)  → eval_scores table
                                         ▼
                              Streamlit dashboard / get_recent_runs()
```
