"""
Synthetic adverse-event report generator — label-by-construction test set.

We pick target_grade FIRST, then ask OpenAI gpt-4o-mini to write messy report
text that conveys that severity WITHOUT ever stating the grade. The label is
ground truth because we specified it, never because a model judged the text.

Generation is deliberately OpenAI-only: Claude is the downstream grader and must
not recognize its own phrasing. Symptoms are drawn from the real CTCAE table via
ctcae_lookup, and expected_decision is derived from target_grade using the SAME
protocol_rules logic the evaluator uses, so grade-accuracy stays consistent.

Usage (from project root):
    python -m clinguard.data.generate_synthetic --tier dev --dry-run   # no API
    python -m clinguard.data.generate_synthetic --tier dev             # live
    python -m clinguard.data.generate_synthetic --tier full            # live
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

# Allow both `python -m clinguard.data.generate_synthetic` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from clinguard.config import get_chat_model
from clinguard.tools.ctcae_lookup import all_terms, lookup_ctcae_grade
from clinguard.tools.protocol_rules import check_protocol_rule

_DATA_DIR = os.path.dirname(__file__)

# ~8 plausible (fictional) trial drug names.
DRUGS = [
    "Veltorib", "Naxolimab", "Cardexanib", "Zolpretin",
    "Emvatuzumab", "Trelagliptor", "Osurentib", "Panvorexin",
]

MESSINESS = ["clean", "abbreviated", "sloppy"]

# Candidate common symptoms; each is validated against CTCAE at startup so only
# real terms survive into the test set.
_COMMON_CANDIDATES = [
    "nausea", "vomiting", "diarrhea", "fatigue", "headache", "fever",
    "anemia", "hypertension", "hypotension", "rash", "chest pain", "dyspnea",
    "cough", "constipation", "neutropenia", "thrombocytopenia", "dizziness",
    "pruritus", "mucositis", "peripheral sensory neuropathy", "arthralgia",
    "hyperglycemia", "hypokalemia", "edema limbs",
]

# Grade -> OBSERVABLE signals to show (vitals, care setting, treatment, function,
# outcome). Deliberately no severity adjectives or CTCAE definition language:
# the grade must be inferable from what happened, never announced.
GRADE_SIGNALS = {
    1: "An incidental, barely-noticeable finding. Vitals essentially normal. The "
       "patient keeps doing all usual activities and work. No treatment given — "
       "often noticed only in passing or on a routine check.",
    2: "Noticeable but the patient stays out of hospital. Vitals or labs a little "
       "off but stable. The patient scales back some errands, chores, or hobbies. "
       "Managed with a simple oral medication, or the study dose is briefly held.",
    3: "The patient is admitted to a hospital ward (or spends hours in the ED). "
       "Vitals or labs are clearly abnormal. They cannot manage self-care, work, "
       "or get out of bed easily. Treated with IV medication, fluids, oxygen, or "
       "a transfusion.",
    4: "The patient is rushed to the ICU. Vitals are dangerously deranged (e.g. "
       "SpO2 in the low 80s, systolic BP under 80, or a collapse). Emergency "
       "measures such as vasopressors, intubation, or an urgent procedure are "
       "started; there are signs of failing organs.",
    5: "The patient dies. Record the event, the timeline to death, and that it was "
       "considered related to the study drug.",
}

# Deliberately in-between course for grade-2-vs-3 boundary hard cases: a single
# report that could be defended either way. Used instead of GRADE_SIGNALS.
BOUNDARY_SIGNAL = (
    "This case must read as genuinely ambiguous — a reviewer should be able to "
    "argue it either way. Give BORDERLINE vitals (mildly off, not dramatic — e.g. "
    "BP around 95-102/60-66, HR high-90s to ~105, SpO2 91-93%). Give an EQUIVOCAL "
    "disposition: the patient was held in the emergency department or an "
    "observation unit for a few hours or overnight and then discharged, or was "
    "watched closely without a formal ward admission. Keep treatment low-intensity "
    "(oral meds, or at most a single bag of IV fluids). Do NOT include ICU, "
    "transfusion, intubation, vasopressors, or more than low-flow oxygen. The "
    "functional impact should be mixed — some difficulty but not clearly bed-bound."
)

# Words that announce (rather than show) severity — used both in the prompt's ban
# list and as a post-generation validator so no leaky report ships.
_LEAK_PATTERNS = [
    r"\bgrade", r"ctcae", r"life[\s-]?threat", r"\bmild\b", r"\bmoderate\b",
    r"\bsevere\b", r"\bserious\b", r"\bcritical\b", r"medically significant",
    r"adverse event", r"escalat", r"\bdismiss", r"\bmonitor",
    r"instrumental activit", r"urgent intervention",
]
_LEAK_RE = re.compile("|".join(_LEAK_PATTERNS), re.IGNORECASE)


def find_leaks(text: str) -> list[str]:
    """Return the distinct forbidden tokens that leak/telegraph the grade."""
    return sorted({m.group(0).lower() for m in _LEAK_RE.finditer(text)})

MESSINESS_STYLE = {
    "clean": "Clear, complete clinical prose. Include full vital signs.",
    "abbreviated": "Terse clinical shorthand and abbreviations (pt, c/o, SOB, "
                   "n/v, HR, BP, VS). Fragmentary sentences are fine.",
    "sloppy": "Informal notes with a few typos and hedging language such as "
              "'possibly dose-related' or 'seems'. Some vitals may be omitted.",
}


# --------------------------------------------------------------------------- #
# Symptom pools (validated against real CTCAE terms)                          #
# --------------------------------------------------------------------------- #
def _build_symptom_pools() -> tuple[list[str], list[str]]:
    """Return (common validated symptoms, rare CTCAE terms)."""
    common = [s for s in _COMMON_CANDIDATES if lookup_ctcae_grade(s).get("matched")]
    if not common:
        # CTCAE file unavailable — fall back to the raw candidates so the
        # generator still runs (labels remain construction-based).
        common = list(_COMMON_CANDIDATES)

    terms = all_terms()
    common_norm = {c.lower() for c in common}
    rare = [t for t in terms if t.lower() not in common_norm and 3 <= len(t) <= 40]
    return common, rare


# --------------------------------------------------------------------------- #
# Label derivation — SAME logic as protocol_rules                             #
# --------------------------------------------------------------------------- #
def expected_decision(target_grade: int) -> str:
    """escalate/monitor/dismiss derived from target_grade via protocol_rules."""
    return check_protocol_rule({"adverse_event": target_grade})["decision"]


# --------------------------------------------------------------------------- #
# Plan construction (metadata only — no API needed)                           #
# --------------------------------------------------------------------------- #
def _spec(rng, grade, common, rare, messiness=None, hard=False, note="",
          rare_terms=False, n_sym=None, hard_kind="grid"):
    pool = rare if rare_terms and rare else common
    if n_sym is None:
        n_sym = rng.choice([1, 1, 2, 3])  # bias toward single-symptom
    n_sym = min(n_sym, len(pool))
    symptoms = rng.sample(pool, n_sym)
    return {
        "target_grade": grade,
        "drug": rng.choice(DRUGS),
        "symptoms": symptoms,
        "messiness": messiness or rng.choice(MESSINESS),
        "hard": hard,
        "note": note,
        "hard_kind": hard_kind,  # written to the JSON record for hard cases (see make_record)
    }


def build_plan(tier: str, rng, common, rare) -> list[dict]:
    specs: list[dict] = []

    if tier == "dev":
        # ~30: 6 per grade, cycling messiness and single/multi profiles.
        for grade in range(1, 6):
            for k in range(6):
                specs.append(_spec(
                    rng, grade, common, rare,
                    messiness=MESSINESS[k % 3],
                    n_sym=1 if k % 2 == 0 else rng.choice([2, 3]),
                ))
        return specs

    # full: 25 per grade across the grid (125) ...
    for grade in range(1, 6):
        for k in range(25):
            specs.append(_spec(
                rng, grade, common, rare,
                messiness=MESSINESS[k % 3],
                n_sym=1 if k % 2 == 0 else rng.choice([2, 3]),
            ))

    # ... plus ~25 deliberately hard cases across four categories.
    hard: list[dict] = []
    # (a) grade 2-vs-3 boundary
    for _ in range(7):
        g = rng.choice([2, 3])
        hard.append(_spec(rng, g, common, rare, hard=True, n_sym=1, hard_kind="boundary",
                          note="Make the picture genuinely borderline between a "
                               "stayed-home course and a hospital-ward course; keep "
                               "the vitals and treatment ambiguous, not decisive."))
    # (b) missing the vital you'd need to grade
    for _ in range(6):
        g = rng.choice([3, 4])
        hard.append(_spec(rng, g, common, rare, hard=True, messiness="sloppy", n_sym=1,
                          hard_kind="missing_vital",
                          note="Do NOT report any vital signs or lab values — omit "
                               "the measurement a clinician would need to grade this."))
    # (c) rare CTCAE terms
    for _ in range(6):
        g = rng.randint(1, 5)
        hard.append(_spec(rng, g, common, rare, hard=True, rare_terms=True, n_sym=1,
                          hard_kind="rare_term"))
    # (d) multi-symptom with conflicting severities
    for _ in range(6):
        g = rng.choice([3, 4])
        hard.append(_spec(rng, g, common, rare, hard=True, n_sym=rng.choice([2, 3]),
                          hard_kind="multi_symptom",
                          note="Portray the symptoms with conflicting courses — one "
                               "clearly needing hospital-level care, another trivial "
                               "— so the overall picture is ambiguous."))
    specs.extend(hard)
    return specs


# --------------------------------------------------------------------------- #
# Report text: model prompt + offline stub                                    #
# --------------------------------------------------------------------------- #
def build_messages(spec: dict) -> list[dict]:
    grade = spec["target_grade"]

    system = (
        "You are a clinical trial site coordinator writing a raw, unpolished "
        "adverse event note for the trial safety system. Show what happened "
        "through concrete observations — never characterize the severity in "
        "words. A reviewer should be able to infer how serious it is purely from "
        "the vitals, care setting, treatments, and functional impact you record."
    )
    happened = BOUNDARY_SIGNAL if spec.get("hard_kind") == "boundary" else GRADE_SIGNALS[grade]
    lines = [
        f"Study drug: {spec['drug']}",
        f"Adverse events to feature: {', '.join(spec['symptoms'])}",
        f"What actually happened (turn these facts into a note — do NOT copy this "
        f"wording): {happened}",
        f"Writing style: {MESSINESS_STYLE[spec['messiness']]}",
    ]
    if spec.get("note"):
        lines.append(f"Extra instruction: {spec['note']}")
    lines.append(
        "Write ONLY the free-text note (2-4 sentences). Include a patient id like "
        "PT-0042 and roughly when symptoms began relative to a dose.\n"
        "HARD RULES — never use any of these words or you fail: grade, CTCAE, "
        "mild, moderate, severe, serious, critical, life-threatening, "
        "'adverse event', 'medically significant', escalate, monitor, dismiss, "
        "'urgent intervention', 'instrumental activities'. Do not state a severity "
        "score or a recommended action. Convey severity ONLY through specifics: "
        "actual vital-sign numbers, whether the patient stayed home / went to the "
        "ward / went to the ICU, what treatment was given, what they could or "
        "couldn't do, and the outcome."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


# Offline observable snippets per grade (no severity adjectives) so the stub
# demonstrates the leak-free format the live prompt targets.
_STUB_SIGNAL = {
    1: ("stayed fully active, no treatment given", "BP 120/78 HR 74 SpO2 98%"),
    2: ("skipped errands for a day, given an oral med, dose held", "BP 132/85 HR 92"),
    3: ("admitted to the ward, started on IV fluids, could not self-care", "BP 100/60 HR 108 SpO2 92%"),
    4: ("rushed to ICU, started on pressors and intubated", "BP 76/48 HR 128 SpO2 83%"),
    5: ("could not be resuscitated and died", "BP unrecordable prior"),
}


def stub_report(spec: dict, rng) -> str:
    """Deterministic offline placeholder so --dry-run can show the format."""
    pid = f"PT-{rng.randint(1, 9999):04d}"
    drug, syms, m = spec["drug"], spec["symptoms"], spec["messiness"]
    course, vitals = _STUB_SIGNAL[spec["target_grade"]]
    joined = ", ".join(syms)
    no_vitals = "omit" in spec.get("note", "").lower()
    vs = "" if no_vitals else f" {vitals}."
    if m == "clean":
        return (f"Patient {pid} on {drug} developed {joined} about two days after "
                f"the most recent dose; {course}.{vs}")
    if m == "abbreviated":
        return (f"{pid} on {drug}. pt c/o {joined} ~48h post dose 2; {course}.{vs}")
    return (f"pt {pid} ({drug}) - {joined} started sometime after a recent dose, "
            f"possibly dose-related; {course}.{vs}")


# --------------------------------------------------------------------------- #
# Record assembly                                                             #
# --------------------------------------------------------------------------- #
def _generate_one(llm, spec: dict, max_attempts: int = 3) -> tuple[str, list[str]]:
    """
    Generate one report, retrying with a correction if leak words appear.
    Returns (text, residual_leaks). Keeps the least-leaky attempt if none clean.
    """
    messages = build_messages(spec)
    best_text, best_leaks = "", None
    for _ in range(max_attempts):
        text = str(llm.invoke(messages).content).strip()
        leaks = find_leaks(text)
        if not leaks:
            return text, []
        if best_leaks is None or len(leaks) < len(best_leaks):
            best_text, best_leaks = text, leaks
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                "You used forbidden words: " + ", ".join(leaks) + ". Rewrite the "
                "note without any of them — show severity only through vitals, "
                "care setting (home/ward/ICU), treatments given, functional impact, "
                "and outcome. Do not name a severity or an action.")},
        ]
    return best_text, best_leaks or []


def make_record(idx: int, spec: dict, report_text: str) -> dict:
    rec = {
        "id": f"syn_{idx:04d}",
        "report_text": report_text.strip(),
        "drug": spec["drug"],
        "symptoms": spec["symptoms"],
        "target_grade": spec["target_grade"],
        "expected_decision": expected_decision(spec["target_grade"]),
        "messiness": spec["messiness"],
        "hard": spec["hard"],
    }
    # Only hard cases carry a subtype; the eval buckets on this field directly
    # (boundary / missing_vital / rare_term / multi_symptom) rather than
    # inferring it from id ranges. Standard cases omit it entirely.
    if spec["hard"]:
        rec["hard_kind"] = spec["hard_kind"]
    return rec


def print_summary(specs: list[dict], title: str) -> None:
    by_grade = Counter(s["target_grade"] for s in specs)
    by_mess = Counter(s["messiness"] for s in specs)
    hard = sum(1 for s in specs if s["hard"])
    print(f"\n{'=' * 52}\n  {title}  (total: {len(specs)})\n{'=' * 52}")
    print("  Per grade:")
    for g in range(1, 6):
        dec = expected_decision(g)
        print(f"    grade {g}  ({dec:<8}): {by_grade.get(g, 0)}")
    print("  Per messiness:")
    for m in MESSINESS:
        print(f"    {m:<12}: {by_mess.get(m, 0)}")
    print(f"  Hard cases       : {hard}")
    print("=" * 52)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic AE report generator")
    parser.add_argument("--tier", choices=["dev", "full"], default="dev")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 3 stubbed samples + summary; no API calls.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    common, rare = _build_symptom_pools()
    specs = build_plan(args.tier, rng, common, rare)

    if args.dry_run:
        print(f"\n[DRY RUN] tier={args.tier} — NO API CALLS. "
              f"Symptom pool: {len(common)} common / {len(rare)} rare CTCAE terms.")
        # 3 stubbed samples covering the three messiness styles.
        sample_rng = random.Random(args.seed + 1)
        picks = []
        for style in MESSINESS:
            cands = [s for s in specs if s["messiness"] == style]
            if cands:
                picks.append(sample_rng.choice(cands))
        print("\n--- 3 STUBBED SAMPLES (not model output) ---")
        for i, spec in enumerate(picks, 1):
            rec = make_record(i, spec, stub_report(spec, sample_rng))
            print(f"\n[{i}] {json.dumps(rec, indent=2)}")
        print_summary(specs, f"PLANNED DISTRIBUTION ({args.tier})")
        print("\nApprove and re-run without --dry-run to generate live.")
        return

    # ---- Live generation (only when explicitly requested) ----
    llm = get_chat_model("generator", temperature=0.9)
    records = []
    residual_leaks = []
    print(f"Generating {len(specs)} reports (tier={args.tier}) with gpt-4o-mini...")
    for i, spec in enumerate(specs, 1):
        text, leaks = _generate_one(llm, spec)
        if leaks:
            residual_leaks.append((f"syn_{i:04d}", leaks))
        records.append(make_record(i, spec, text))
        print(f"  [{i:03d}/{len(specs)}] grade {spec['target_grade']} "
              f"{spec['messiness']:<11} {'HARD' if spec['hard'] else '':<4} "
              f"{('LEAK:' + ','.join(leaks)) if leaks else 'ok':<14} "
              f"{', '.join(spec['symptoms'])[:40]}")

    out_path = os.path.join(_DATA_DIR, f"synthetic_{args.tier}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {out_path}")
    print_summary(specs, f"GENERATED ({args.tier})")
    if residual_leaks:
        print(f"\n  WARNING: {len(residual_leaks)} report(s) still contain leak "
              f"words after retries:")
        for rid, lk in residual_leaks:
            print(f"    {rid}: {lk}")
    else:
        print("\n  Leak check: clean — no forbidden severity words in any report.")


if __name__ == "__main__":
    main()
