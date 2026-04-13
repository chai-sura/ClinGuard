"""
CTCAE v6.0 lookup tool.

Loads the CTCAE Excel file and exposes lookup_ctcae_grade() for fuzzy
term matching against NCI CTCAE v6.0 adverse event definitions.
"""

import json
import os

import pandas as pd

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "CTCAE v6.0 Final_Jan2026.xlsx"
)
_SHEET_NAME = "CTCAE v6.0 Clean Copy"


def _load_data() -> pd.DataFrame:
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

    # Normalize term to lowercase for case-insensitive matching
    df["term"] = df["term"].str.strip().str.lower()

    # Replace bare "-" entries with a readable label
    df = df.replace("-", "Not applicable")

    # Drop rows that are entirely empty or have no term
    df = df.dropna(how="all")
    df = df[df["term"].notna()]

    return df.reset_index(drop=True)


# Load once at module import time
_df = _load_data()


def lookup_ctcae_grade(symptom: str) -> dict:
    """
    Case-insensitive fuzzy lookup of a CTCAE adverse event term.

    Returns a dict with keys: term, grade_1..grade_5, definition.
    Falls back to {"error": "term not found", "term": symptom} if no match.
    """
    query = symptom.strip().lower()

    # 1. Exact match
    exact = _df[_df["term"] == query]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # 2. Substring match (query appears anywhere in a term)
    substr = _df[_df["term"].str.contains(query, regex=False, na=False)]
    if not substr.empty:
        return substr.iloc[0].to_dict()

    return {"error": "term not found", "term": symptom}


if __name__ == "__main__":
    for test_term in ("nausea", "chest pain"):
        print(f"\n=== {test_term} ===")
        print(json.dumps(lookup_ctcae_grade(test_term), indent=2, default=str))
