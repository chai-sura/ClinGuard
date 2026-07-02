"""
CTCAE v6.0 lookup tool.

Loads the CTCAE Excel file and exposes lookup_ctcae_grade() for robust term
matching (exact -> normalized -> best bidirectional/token-overlap) against NCI
CTCAE v6.0 adverse event definitions. Returns the single best match with a
score, and degrades cleanly when the data file is missing or a term is unknown.
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "CTCAE v6.0 Final_Jan2026.xlsx"
)
_SHEET_NAME = "CTCAE v6.0 Clean Copy"

# Minimum fuzzy score for a non-exact match to count as found.
_MATCH_THRESHOLD = 0.5

_GRADE_KEYS = ("grade_1", "grade_2", "grade_3", "grade_4", "grade_5")


def _normalize(text: str) -> str:
    """Lowercase, replace non-alphanumerics with spaces, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _load_rows() -> list[dict]:
    """
    Read the CTCAE sheet into a list of row dicts, each pre-normalized with a
    token set for fuzzy matching. Returns [] (never raises) if the file is
    missing or unreadable, so importing this module can't crash the app.
    """
    df = pd.read_excel(_DATA_PATH, sheet_name=_SHEET_NAME, engine="openpyxl")

    # Strip regular and non-breaking whitespace (\xa0) from all column names
    df.columns = [c.strip().replace("\xa0", "").strip() for c in df.columns]

    col_map = {
        "CTCAE v6.0 MedDRA 28.0 Term": "term",
        "Grade 1": "grade_1",
        "Grade 2": "grade_2",
        "Grade 3": "grade_3",
        "Grade 4": "grade_4",
        "Grade 5": "grade_5",
        "Definition": "definition",
    }

    df = df[list(col_map.keys())].rename(columns=col_map)
    df["term"] = df["term"].astype(str).str.strip().str.lower()
    df = df.replace("-", "Not applicable")
    df = df.dropna(how="all")
    df = df[df["term"].notna() & (df["term"] != "nan")]

    rows: list[dict] = []
    for _, r in df.iterrows():
        norm = _normalize(r["term"])
        if not norm:
            continue
        row = {k: r.get(k) for k in ("term", *_GRADE_KEYS, "definition")}
        row["_norm"] = norm
        row["_tokens"] = frozenset(norm.split())
        rows.append(row)
    return rows


# Load once at import time; failure degrades to an empty table, not a crash.
_LOAD_ERROR: str | None = None
try:
    _ROWS = _load_rows()
except Exception as exc:  # noqa: BLE001 — any load failure must not crash import
    _ROWS = []
    _LOAD_ERROR = f"CTCAE data unavailable: {type(exc).__name__}: {exc}"


def _score(q_norm: str, q_tokens: frozenset, row: dict) -> float:
    """
    Similarity in [0, 1] between a normalized query and a CTCAE row.

    Uses token-overlap F1 (recall against the query, precision against the
    term) plus a small bidirectional-substring bonus. F1 keeps multi-token
    matches ahead of short generic terms — "chest pain, radiating" scores
    "chest pain - cardiac" above the bare term "pain" — while the substring
    bonus rewards clean containment ("chest pain" inside "chest pain cardiac").
    """
    t_norm = row["_norm"]
    if q_norm == t_norm:
        return 1.0

    t_tokens = row["_tokens"]
    inter = len(q_tokens & t_tokens)
    if not inter:
        return 0.0

    recall = inter / len(q_tokens)      # how much of the query is covered
    precision = inter / len(t_tokens)   # how much of the term is covered
    f1 = 2 * precision * recall / (precision + recall)

    if q_norm in t_norm or t_norm in q_norm:
        f1 += 0.15

    return min(1.0, f1)


def all_terms() -> list[str]:
    """Every CTCAE term currently loaded (empty if the data file is missing)."""
    return [row["term"] for row in _ROWS]


def lookup_ctcae_grade(symptom: str) -> dict:
    """
    Robust lookup of a CTCAE adverse event term.

    Matching order: exact -> normalized exact -> best fuzzy (bidirectional
    substring + token overlap). Returns the single best-scoring row with a
    match_score, or a clean not-found result.

    Found:     {term, grade_1..grade_5, definition, match_score, matched: True, query}
    Not found: {matched: False, match_score: 0.0, term, query, error}
    """
    query = str(symptom)
    q_norm = _normalize(query)

    if not _ROWS:
        return {
            "matched": False,
            "match_score": 0.0,
            "term": query,
            "query": query,
            "error": _LOAD_ERROR or "CTCAE data unavailable",
        }

    if not q_norm:
        return {
            "matched": False,
            "match_score": 0.0,
            "term": query,
            "query": query,
            "error": "empty query",
        }

    q_tokens = frozenset(q_norm.split())

    best_row = None
    best_score = 0.0
    for row in _ROWS:
        s = _score(q_norm, q_tokens, row)
        if s > best_score:  # strictly greater -> first row wins ties, deterministic
            best_score, best_row = s, row
            if s == 1.0:
                break

    if best_row is None or best_score < _MATCH_THRESHOLD:
        return {
            "matched": False,
            "match_score": round(best_score, 3),
            "term": query,
            "query": query,
            "error": "term not found",
        }

    result = {k: best_row.get(k) for k in ("term", *_GRADE_KEYS, "definition")}
    result["match_score"] = round(best_score, 3)
    result["matched"] = True
    result["query"] = query
    return result


if __name__ == "__main__":
    for test_term in ("nausea", "chest pain", "chest pain, radiating", "wibble xyz"):
        print(f"\n=== {test_term} ===")
        res = lookup_ctcae_grade(test_term)
        print(f"matched={res['matched']} score={res['match_score']} term={res.get('term')!r}")
        print(json.dumps(res, indent=2, default=str))
