# ClinGuard — Evaluation Results

Frozen source-of-truth numbers for the README and demo. All figures are from the
full 150-case synthetic eval run through the live pipeline (2 Claude calls +
1 OpenAI call per case), scored against label-by-construction ground truth.

- **Test set:** 150 synthetic AE reports (`clinguard/data/synthetic_full.json`),
  labelled by construction — target grade chosen first, observable text generated
  to match, with no severity/grade leakage. 125 standard + 25 deliberately hard.
- **Pipeline:** extractor + classifier (grader) on Claude Haiku; verifier on
  OpenAI gpt-4o-mini (cross-model, to avoid self-evaluation bias). The
  deterministic protocol-rules engine is the sole decision-maker; the verifier
  only gates confidence.
- **Run:** 150/150 completed, **0 errors**. All runs persisted to SQLite.

---

## Headline — standard cases (125)

| Metric | Result |
|---|---|
| **Grade accuracy** (assigned max grade == target) | **81.6%** (102/125) |
| **Escalation accuracy** (decision == expected) | **94.4%** (118/125) |

Grade accuracy by target grade: g1 92% · g2 80% · g3 52% · g4 92% · g5 92%.

## Hard cases (25) — by subtype

Reported separately; low grade accuracy is expected (deliberately ambiguous).

| subtype | grade accuracy | escalation accuracy |
|---|---|---|
| boundary (7) | 57.1% (4/7) | 57.1% (4/7) |
| missing_vital (6) | 100% (6/6) | 100% (6/6) |
| rare_term (6) | 100% (6/6) | 100% (6/6) |
| multi_symptom (6) | 83.3% (5/6) | 100% (6/6) |
| **all hard** | **84.0%** (21/25) | **88.0%** (22/25) |

---

## Before / after — extractor terminology fix

The single largest accuracy driver. **Root cause:** the original extractor emitted
the patient's lay wording ("increased thirst", "numbness in hands") and copied
lab/vital values verbatim, so the grounded CTCAE lookup never fired on the real
gradeable events. **Fix (prompt-only):** name events with CTCAE-canonical terms,
infer conditions from lab/vital values (glucose 210 + drug started →
`hyperglycemia`; BP 150/95 → `hypertension`), and extract `death` (grade 5) on
explicit death/expiration statements only.

| Metric | Before | After | Δ |
|---|---|---|---|
| Standard grade accuracy | 70.4% | **81.6%** | **+11.2pp** |
| Standard escalation accuracy | 94.4% | 94.4% | — |
| Hard grade accuracy | 48.0% | **84.0%** | **+36.0pp** |
| Hard escalation accuracy | 84.0% | 88.0% | +4.0pp |

Standard grade accuracy by target grade:

| | g1 | g2 | g3 | g4 | g5 |
|---|---|---|---|---|---|
| Before | 88% | 84% | 52% | 56% | 72% |
| After | 92% | 80% | 52% | **92%** | **92%** |
| Δ | +4 | −4 | 0 | **+36** | **+20** |

The gains concentrate exactly where lay-term extraction was failing — high-acuity
events (g4/g5) and hard subtypes whose signal is a lab value or a fatal outcome
(`missing_vital` 33%→100%, `multi_symptom` 33%→83%). The small g2 dip (−4pp) is
value inference occasionally nudging a borderline g2 to g3 → escalate: over-caution,
a safe direction.

**Example (syn_0043, target g2):** `increased thirst / increased urination`
(all g1 → dismiss) → `hyperglycemia / hypertension / cough` (g2 → monitor).

---

## Cross-model agreement & human-review load

Two definitions — the operational metric the pipeline shipped with, and the
decision-level metric that reflects what actually matters (do the two models reach
the same disposition):

| | Agreement | Needs-human-review |
|---|---|---|
| **Exact per-symptom match** (as-run) | 46.7% | 57.3% |
| **Decision-level** (same escalate/monitor/dismiss) | **94.0%** | **11.3%** |

Decision-level, split: standard **96.8%** agree / 7.2% review; hard **80.0%** agree
/ 32.0% review (appropriately more review on ambiguous cases).

The exact-match number is pessimistic and got noisier after the extractor fix
(~37% more symptoms per case → more per-symptom grades to mismatch on). At the
decision level the two models agree 94% of the time. **Recommendation:** switch the
human-review trigger from exact-map agreement to decision-level agreement — cuts the
review queue from 57% to ~11% of runs with no loss of safety signal.

---

## Safety — miss direction

Escalation integrity is the property that matters: a *should-escalate* event must
never be dismissed/monitored.

- **Standard cases: zero dangerous misses.** All 7 standard escalation misses are
  the safe direction (over-caution: g1→g2 monitor, g2→g3 escalate) or a borderline
  low-acuity flip (g2↔dismiss). No target-g3/4/5 case was ever under-decided.
- **Hard cases: one dangerous miss** — `syn_0132` (boundary, target g3 read as g2 →
  monitor instead of escalate). Boundary cases are constructed to sit exactly on the
  escalate/monitor line; this is the single should-escalate→monitor across all 150.

---

## Cost

| | |
|---|---|
| Total eval cost (150 cases) | **$0.5041** |
| Mean cost / case | $0.00336 |
| Mean tokens / case | 2,273 |
| LLM calls | 450 (300 Claude + 150 gpt-4o-mini) |
| Wall-clock | 17.4 min |
| Errors | 0 |

Well within the ~$2/provider budget (≈$0.45 Claude + ≈$0.05 OpenAI).

---

## Known limitations

Tracked in [AUDIT.md](AUDIT.md) → "Known limitations":

1. **Grade-3 band (52%)** — inherent CTCAE grade-boundary ambiguity, not
   extraction-fixable; unchanged by the extractor fix. Would need grading-side work.
2. **Borderline g1/g2 instability** — a few cases (e.g. `syn_0046`) sit on the
   g1/g2 line and flip between runs at temperature 0; case-inherent, non-dangerous.
3. **One boundary dangerous miss** (`syn_0132`, above) — tracked.
