"""A second user's corpus, and the check that Kivi learns from it.

This exists because of a specific failure. The assignment's stage-two
evaluation imports the reviewer's own ~500 dictations from a user who is not
the author, and the first version of this system had regexes matching this
project by name - "golden goose", "sarvam", "predictive maintenance". Run
against a different person it learned 8 memories from 500 records and abstained
on nearly every question, while still scoring 11/11 on its own evaluation.

So the evaluation could not see its own biggest failure. This corpus is the
control: a product designer in Pune, different job, tools, city, manager, and
one belief that changes over time. Nothing in it overlaps the seed corpus.

    python3 -m tests.foreign_corpus            # generate + ingest + report
    python3 -m tests.foreign_corpus --write out.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kivi_memory.config import KiviConfig  # noqa: E402
from kivi_memory.engine import KiviMemoryEngine  # noqa: E402


APPS = ["whatsapp", "gmail", "figma", "jira", "calendar", "notes"]
PEOPLE = ["Neha", "Karthik", "Sanjana", "Rohit", "Divya"]

ANCHORS = [
    "I am a product designer at Zerodha and I work out of the Pune office.",
    "My manager is Neha and she wants design reviews every Thursday.",
    "I like long detailed written feedback, not one-line comments, when someone reviews my work.",
    "For the onboarding redesign I am prototyping in Figma with variables instead of Framer.",
    "Actually we moved the onboarding redesign off Figma variables to a coded prototype in React.",
    "I need to ship the accessibility audit before the 14th.",
    "Remind me that Neha prefers Loom walkthroughs over live design reviews.",
    "The retention experiment is the main thing on my plate this quarter.",
    "Call with the growth team about the paywall copy tomorrow morning.",
    "Draft for Neha: the retention experiment should focus on day-7 drop-off, the paywall wording, "
    "and why we need qualitative interviews before shipping.",
]

WORK = [
    "Working on the onboarding redesign and the handoff notes for engineering.",
    "The accessibility audit needs proper contrast ratios documented.",
    "For the retention experiment, keep the writeup detailed and evidence-backed.",
    "Sync with {p} about the design system tokens.",
    "I am rewriting the paywall copy for the growth experiment.",
    "Follow up with {p} on the usability test recruitment.",
    "Design system work should be easy to revert if it breaks the app.",
    "Do not make me approve every change; just show me an audit trail.",
]

NOISE = [
    "The dosa at the canteen was decent today.",
    "Auto rickshaw took forever near Baner.",
    "My headphones ran out of battery mid-call.",
    "It rained all evening and I stayed in.",
    "Random note: someone left a birthday cake in the pantry.",
]

# Questions a reviewer would reasonably ask of THIS user's history.
QUESTIONS = [
    ("employer", "Where do I work?", ["zerodha"]),
    ("manager", "Who is my manager?", ["neha"]),
    ("location", "Which office do I work out of?", ["pune"]),
    ("feedback", "What kind of feedback do I prefer?", ["detailed"]),
    ("tooling", "What am I using for the onboarding redesign?", ["react"]),
    ("commitment", "What do I need to ship?", ["accessibility"]),
    ("unknown", "What is my employee ID?", None),  # must abstain
]


def _asr(text: str) -> str:
    return (text.lower().replace(".", "").replace(",", "")
            .replace("figma", "fig ma").replace("zerodha", "zero dha"))


def generate(count: int = 500, seed: int = 7) -> list[dict]:
    random.seed(seed)
    base = datetime(2026, 7, 5, 9, 0)
    records: list[dict] = []
    for index, text in enumerate(ANCHORS):
        created = base + timedelta(days=index * 2, hours=random.randint(0, 8))
        records.append({
            "id": f"x_{index:04d}", "created_at": created.replace(microsecond=0).isoformat(),
            "app": random.choice(APPS), "raw_asr": _asr(text), "formatted_text": text,
            "metadata": {"kind": "anchor", "synthetic": True},
        })
    while len(records) < count:
        created = base + timedelta(days=random.randint(0, 20), hours=random.randint(0, 10),
                                   minutes=random.randint(0, 59))
        noise = random.random() < 0.34
        text = random.choice(NOISE) if noise else random.choice(WORK).format(p=random.choice(PEOPLE))
        records.append({
            "id": f"x_{len(records):04d}", "created_at": created.replace(microsecond=0).isoformat(),
            "app": random.choice(APPS), "raw_asr": _asr(text), "formatted_text": text,
            "metadata": {"kind": "noise" if noise else "work", "synthetic": True},
        })
    records.sort(key=lambda record: record["created_at"])
    return records[:count]


def run(records: int = 500, config: KiviConfig | None = None) -> dict:
    corpus = generate(records)
    tmp = Path(tempfile.mkdtemp(prefix="kivi_foreign_"))
    engine = KiviMemoryEngine(str(tmp / "foreign.db"),
                              config or KiviConfig().apply_overrides(
                                  extractor="rule", nli="off", retrieval="lexical", answerer="template"))
    ingest = engine.ingest_records(corpus)

    cases = []
    answered = 0
    for case_id, query, expected in QUESTIONS:
        result = engine.ask(query)
        text = result["answer"].lower()
        if expected is None:
            ok = bool(result["abstained"])
        else:
            ok = not result["abstained"] and all(term in text for term in expected)
        answered += int(ok)
        cases.append({"id": case_id, "query": query, "expected": expected,
                      "abstained": result["abstained"], "answer": result["answer"],
                      "passed": ok,
                      "sources": [item["id"] for item in result.get("source_transcripts", [])][:3]})

    stats = engine.stats()
    return {
        "records": len(corpus),
        "memories": stats["counts"]["memories"],
        "decisions_by_action": stats["decisions_by_action"],
        "growth": {key: value for key, value in engine.growth().items() if key != "samples"},
        "answered": answered,
        "total": len(QUESTIONS),
        "ingest": ingest,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kivi on a different user's corpus")
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--write", help="Write the corpus to this path and exit")
    args = parser.parse_args()

    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in generate(args.records):
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"wrote {args.records} records to {path}")
        return 0

    result = run(args.records)
    print(f"corpus            : {result['records']} records from a different user")
    print(f"memories learned  : {result['memories']}")
    print(f"decisions         : {result['decisions_by_action']}")
    print(f"selectivity       : {result['growth'].get('selectivity')}")
    print(f"questions answered: {result['answered']}/{result['total']}\n")
    for case in result["cases"]:
        mark = "PASS" if case["passed"] else "FAIL"
        answer = case["answer"].replace("\n", " ")[:64]
        print(f"  {mark}  {case['query']:<42} -> {answer}")

    # A run that learns almost nothing from a foreign corpus is the failure this
    # file exists to catch, so it is an error, not a note.
    if result["memories"] < 10 or result["answered"] < 4:
        print("\nFAILED: the system does not generalise to a corpus it was not written against.")
        return 1
    print("\nOK: learns and answers on a corpus it was not written against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
