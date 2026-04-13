# ClinGuard — AI Clinical Trial Safety Monitor

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-00B4D8?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat-square&logo=sqlite&logoColor=white)
![CTCAE](https://img.shields.io/badge/CTCAE-v6.0-2D6A4F?style=flat-square)

A 3-agent LangGraph system that classifies clinical trial adverse event reports against CTCAE v6.0 severity grades using ReAct reasoning loops, with a 5-dimension LLM-as-judge eval harness and full SQLite observability logging.

---

## The Problem

Clinical trials generate hundreds of adverse event (AE) reports per week, each requiring manual classification using the NCI's Common Terminology Criteria for Adverse Events (CTCAE) grading system. This review process is time-consuming, inconsistent across reviewers, and high-stakes — a missed Grade 4 or 5 event can result in patient harm or FDA trial suspension. ClinGuard assists reviewers with AI-powered severity classification, deterministic protocol rule enforcement, and a full audit trail for every decision.

---

## Architecture

```
Input Report
     │
     ▼
┌────────────────────────────────────┐
│         LangGraph State Graph      │
│                                    │
│  ┌──────────┐   ┌──────────────┐   │
│  │Extractor │──▶│  Classifier  │   │
│  │ Agent 1  │   │   Agent 2    │   │
│  └──────────┘   │  ReAct Loop  │   │
│                 └──────┬───────┘   │
│                        │           │
│              confidence >= 0.7?    │
│                        │           │
│                 ┌──────▼───────┐   │
│                 │Safety Officer│   │
│                 │   Agent 3    │   │
│                 └──────┬───────┘   │
└────────────────────────┼───────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         Decision    Eval Harness  SQLite
           Memo      5 dimensions   Log
```

---

## Agent Pipeline

| Agent | Role | Tools Used |
|---|---|---|
| **Extractor** | Parses raw AE report into structured fields (patient ID, symptoms, vitals, timeline) | GPT-4o-mini |
| **Classifier** | ReAct loop — grades each symptom against CTCAE v6.0; retries up to 3× if confidence < 0.7 | `lookup_ctcae_grade()`, `check_protocol_rule()` |
| **Safety Officer** | Critic — writes a decision memo, confirms or overrides the classifier's grade, flags low-confidence runs | GPT-4o-mini, protocol rules |

---

## Eval Harness

Each pipeline run is scored by an independent LLM-as-judge call (separate from the pipeline agents to avoid self-evaluation bias).

| Dimension | What It Measures | Type |
|---|---|---|
| **Grounding** | Are all memo claims traceable to the source report? | LLM-as-judge |
| **Completeness** | Were all reported symptoms considered? | LLM-as-judge |
| **Hallucination Risk** | Did the system fabricate clinical facts? | LLM-as-judge |
| **Reasoning Depth** | Did the classifier use sufficient ReAct steps before grading? | LLM-as-judge |
| **Agent Agreement** | Do the CTCAE grades and final decision logically agree? | LLM-as-judge |

---

## Results

```
Eval Results — 50 synthetic AE fixtures
─────────────────────────────────────────
Overall Quality Score   : 0.997
Grounding               : 1.000
Completeness            : 0.990
Hallucination Risk      : 0.000  (lower is better)
Reasoning Depth         : 0.996
Agent Agreement         : 1.000
Mean Latency            : 11.6s
─────────────────────────────────────────
Note: Evaluated on synthetic data. Real-world performance
requires validation on de-identified clinical trial data.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| ReAct reasoning | Custom prompt loop with tool binding |
| CTCAE lookup | pandas + CTCAE v6.0 Excel (NCI) |
| Memory & logging | SQLite |
| Eval harness | LLM-as-judge (GPT-4o-mini, independent call) |
| Dashboard | Streamlit |

---

## Project Structure

```
clinguard/
├── agents/
│   ├── extractor.py          # Agent 1 — structured extraction
│   ├── classifier.py         # Agent 2 — ReAct grading loop
│   └── safety_officer.py     # Agent 3 — critic & decision memo
├── tools/
│   ├── ctcae_lookup.py       # CTCAE v6.0 term lookup (exact + substring)
│   └── protocol_rules.py     # 8-rule deterministic protocol engine
├── graph/
│   ├── state.py              # AgentState TypedDict
│   └── graph.py              # LangGraph state graph + run_pipeline()
├── eval/
│   ├── judge.py              # LLM-as-judge eval harness
│   └── run_evals.py          # Batch runner — 50 fixtures, aggregate metrics
├── db/
│   ├── schema.sql            # DDL for decisions + eval_scores tables
│   └── logger.py             # init_db, log_decision, log_eval, get_recent_runs
├── data/
│   ├── CTCAE v6.0 Final_Jan2026.xlsx   # NCI CTCAE source data
│   └── ae_fixtures.json                # 50 synthetic AE reports (10 per grade)
app.py                        # Streamlit dashboard
main.py                       # CLI entrypoint
generate_fixtures.py          # Synthetic fixture generator
requirements.txt
```

---

## Setup & Installation

**1. Clone the repo**
```bash
git clone https://github.com/chai-sura/ClinGuard.git
cd ClinGuard
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

**5. Download CTCAE v6.0 data**

Download the CTCAE v6.0 Excel file from the NCI CTEP website:
[https://ctep.cancer.gov](https://ctep.cancer.gov)

Save it as:
```
data/CTCAE v6.0 Final_Jan2026.xlsx
```

**6. Generate synthetic test fixtures** *(one-time)*
```bash
python generate_fixtures.py
```

**7. Run the Streamlit dashboard**
```bash
streamlit run app.py
```

**8. Run a single report via CLI**
```bash
python main.py "Patient PT-0042 reported severe chest pain and shortness of breath 3 days after dose 2. BP 88/54, HR 115."
```

**9. Run the full eval harness**
```bash
python clinguard/eval/run_evals.py
```

---

## Sample Output

```
============================================================
  ClinGuard — AI Clinical Trial Safety Monitor
============================================================

Patient ID   : PT-0042
Symptoms     : chest pain, shortness of breath
Vitals       : BP 88/54, HR 115
Timeline     : 3 days after dose 2

CTCAE Grade Assignment:
  chest pain                Grade 4
  shortness of breath       Grade 4

!! ESCALATE — Immediate action required !!
Escalation Grade : 4
Confidence       : 90%
Risk Score       : 0.95
Latency          : 14.2s

Eval Scores:
  Grounding          : 1.00
  Completeness       : 1.00
  Hallucination Risk : 0.00
  Reasoning Depth    : 1.00
  Agent Agreement    : 1.00
  Overall Score      : 1.00
============================================================
```

---

## Limitations & Future Work

- Evaluated on synthetic data generated by GPT-4o-mini; real-world use requires validation on de-identified clinical trial data
- Production deployment requires HIPAA compliance, data governance review, and clinician validation before any patient-facing use
- Future: fine-tune on real AE reports, add multi-language support, integrate with EDC systems (e.g., Medidata Rave, Veeva Vault)
- The human-in-the-loop gate is currently simulated; production would require a clinician sign-off UI and audit workflow

---

