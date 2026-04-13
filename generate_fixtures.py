"""
Generate 50 synthetic adverse event report fixtures for ClinGuard.
Calls GPT-4o-mini once per CTCAE grade (5 calls total) and saves
all results to clinguard/data/ae_fixtures.json.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env file
load_dotenv()

# Initialise the OpenAI client using the key from the environment
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Human-readable severity label for each CTCAE grade (used in the prompt)
GRADE_PROMPTS = {
    1: "mild",
    2: "moderate",
    3: "requiring hospitalization",
    4: "life-threatening",
    5: "fatal",
}

# Destination file for all generated fixtures
OUTPUT_PATH = os.path.join("clinguard", "data", "ae_fixtures.json")


def generate_reports_for_grade(grade: int) -> list[str]:
    """Call GPT-4o-mini and return 10 AE report strings for the given grade."""

    # severity_label is referenced in the prompt to steer narrative tone
    severity_label = GRADE_PROMPTS[grade]

    # Instruct the model to return a bare JSON array so we can parse it
    # directly without any extra text stripping
    prompt = (
        f"Generate 10 realistic clinical trial adverse event reports "
        f"for Grade {grade} severity. Each report should be a paragraph "
        f"written by a nurse or patient. Include patient ID (format PT-XXXX), "
        f"symptoms, vitals where relevant, and timeline since last dose. "
        f"Make Grade 1 mild, Grade 2 moderate, Grade 3 requiring "
        f"hospitalization, Grade 4 life-threatening, Grade 5 fatal. "
        f"Return a JSON array of 10 strings, each string is one report. "
        f"Return only the JSON array, nothing else."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,          # High temperature for variety across reports
        messages=[{"role": "user", "content": prompt}],
    )

    # Strip whitespace and markdown code fences (```json ... ```) that
    # the model sometimes adds despite being told not to
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]          # drop opening fence
        if raw.startswith("json"):
            raw = raw[4:]                      # drop language tag
        raw = raw.rsplit("```", 1)[0].strip()  # drop closing fence
    reports = json.loads(raw)
    return reports


def main() -> None:
    fixtures = []
    counter = 1  # Global counter so IDs are sequential across all grades

    for grade in range(1, 6):  # Grades 1–5 inclusive
        print(f"Generating Grade {grade} reports...")
        reports = generate_reports_for_grade(grade)

        # Wrap each raw string in the fixture envelope before saving
        for report_text in reports:
            fixtures.append({
                "id": f"fixture_{counter:03d}",   # e.g. fixture_001
                "grade": grade,
                "report_text": report_text,
            })
            counter += 1

    # Write all 50 fixtures as a pretty-printed JSON array
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(fixtures)} fixtures to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
