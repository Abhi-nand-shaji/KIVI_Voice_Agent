from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


APPS = ["slack", "gmail", "notion", "docs", "calendar", "linear", "dictation"]
NOISE = [
    "Had chai after lunch and the weather was humid again.",
    "Traffic was slow near the metro, nothing important for work.",
    "Random note: the cafe playlist was too loud today.",
    "I walked for ten minutes after dinner.",
    "Coffee was colder than usual and I need to remember nothing from this.",
    "The office chair squeaks sometimes.",
]
PEOPLE = ["Ravi", "Asha", "Meera", "Nikhil", "Priya", "Arjun"]


def asr_noise(text: str) -> str:
    replacements = {
        "Kivi": "kiwi",
        "Sarvam": "servum",
        "Golden Goose": "golden goos",
        "semantic": "seman tic",
        "concise": "con size",
        "Python": "pie thin",
    }
    raw = text
    for src, dst in replacements.items():
        raw = raw.replace(src, dst)
    return raw.lower().replace(".", "").replace(",", "")


def make_record(idx: int, created_at: datetime, app: str, text: str, kind: str = "seed") -> dict[str, Any]:
    return {
        "id": f"seed_{idx:04d}",
        "created_at": created_at.replace(microsecond=0).isoformat(),
        "app": app,
        "raw_asr": asr_noise(text),
        "formatted_text": text,
        "metadata": {"kind": kind, "synthetic": True},
    }


def generate_corpus(count: int = 500) -> list[dict[str, Any]]:
    random.seed(42)
    base = datetime(2026, 8, 18, 9, 0)
    records: list[dict[str, Any]] = []

    anchors = [
        (datetime(2026, 8, 18, 10, 15), "docs", "I have a Sarvam AI Golden Goose internship task. It is about semantic memory for Kivi."),
        (datetime(2026, 8, 18, 14, 20), "dictation", "I prefer concise technical explanations. Keep the answers short when I am reviewing implementation choices."),
        (datetime(2026, 8, 19, 11, 5), "notion", "I prefer Python for quick backend prototypes because it is easier to inspect."),
        (datetime(2026, 8, 20, 16, 40), "docs", "For the Golden Goose prototype I am using Rust for now because I want to test a stricter controller."),
        (datetime(2026, 8, 22, 12, 5), "notion", "Actually I switched the Golden Goose prototype to Python so the local reviewer can run it without extra setup."),
        (datetime(2026, 8, 23, 9, 45), "dictation", "Use bullet summaries when you review my notes, especially for memory controller tradeoffs."),
        (datetime(2026, 8, 25, 18, 30), "calendar", "Meeting with Ravi tomorrow about the predictive maintenance presentation."),
        (datetime(2026, 8, 27, 13, 10), "slack", "The predictive maintenance thread needs a cleaner executive summary before the meeting."),
        (datetime(2026, 8, 31, 17, 7), "slack", "Draft for Ravi: the predictive maintenance meeting should focus on sensor drift, failure windows, and why the model needs evidence before action."),
        (datetime(2026, 9, 1, 9, 5), "dictation", "Today I need to finish the prototype README and run the evaluation before sending the Sarvam AI task."),
    ]
    for idx, (created, app, text) in enumerate(anchors):
        records.append(make_record(idx, created, app, text, "anchor"))

    templates = [
        "Working on the Golden Goose memory controller. The useful part is deciding what to ignore, not storing every transcript.",
        "Kivi should keep provenance for any important memory so I can inspect the source later.",
        "For semantic memory, confidence should be separate from importance.",
        "I need to polish the Sarvam AI demo flow tomorrow.",
        "The Golden Goose project focuses on selective semantic memory and Hey Kivi tool use.",
        "For the predictive maintenance notes, keep the explanation concise and grounded in evidence.",
        "Meeting with {person} about memory retrieval and update behavior.",
        "Follow up with {person} about the project evaluation cases.",
        "I am working on the Kivi memory prototype and the backend should be easy to reset.",
        "Do not make me approve every memory; give me control through an audit trail.",
    ]
    while len(records) < count:
        idx = len(records)
        created = base + timedelta(days=random.randint(0, 14), hours=random.randint(0, 10), minutes=random.randint(0, 59))
        if random.random() < 0.34:
            text = random.choice(NOISE)
            kind = "noise"
        else:
            text = random.choice(templates).format(person=random.choice(PEOPLE))
            kind = "work"
        app = random.choice(APPS)
        records.append(make_record(idx, created, app, text, kind))

    records.sort(key=lambda item: item["created_at"])
    for idx, record in enumerate(records):
        record["id"] = f"seed_{idx:04d}"
    return records[:count]


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a corpus. Accepts JSONL, a JSON array, or {"records": [...]}.

    The import format is documented as JSONL, but a reviewer translating their
    own corpus should not have a run fail over a wrapping array.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return list(json.loads(text))
    if text[0] == "{" and '"records"' in text[:200]:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return payload["records"]
        except json.JSONDecodeError:
            pass
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not valid JSON ({exc.msg})") from exc
    return records
