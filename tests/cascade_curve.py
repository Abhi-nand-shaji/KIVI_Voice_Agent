"""How much does the model actually cost, and where does it earn its place?

    python3 -m tests.cascade_curve                 # the seed corpus
    python3 -m tests.cascade_curve --records 300

`--extractor hybrid` spends one model call on every transcript. `--extractor
cascade` spends one only where the frames demonstrably fell short. This traces
the curve between them by replaying a corpus through the escalation policy with
a stub sensor in place of the model, so it costs nothing to run and answers the
question the README would otherwise have to assert:

    at the default thresholds, what fraction of transcripts need the model?

No model is called. The escalation decision depends only on what the frames
produced, so the rate is exact even with the model absent - which is the whole
point of putting that decision in the controller rather than in a prompt.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kivi_memory.config import KiviConfig  # noqa: E402
from kivi_memory.core import normalize_text, strip_disfluency  # noqa: E402
from kivi_memory.extractors import CascadeExtractor, RuleExtractor  # noqa: E402
from kivi_memory.seed import generate_corpus, read_jsonl  # noqa: E402


THRESHOLDS = (0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 1.0)


class NoModel:
    """Stands in for the LLM sensor. Never consulted - only counted."""

    def extract(self, record):
        return [], []

    def describe(self):
        return {"backend": "not called"}


def classify(reason: str | None) -> str:
    if reason is None:
        return "settled by frames"
    if "no frame" in reason:
        return "no frame matched"
    if "confidence" in reason:
        return "low confidence"
    return "low coverage"


def run(records: list[dict]) -> None:
    rule = RuleExtractor()
    config = KiviConfig()
    total = len(records)

    # Pre-compute the frame output once; only the thresholds change per row.
    parsed = []
    for record in records:
        text = strip_disfluency(normalize_text(
            record.get("formatted_text") or record.get("raw_asr") or ""))
        candidates, ignored = rule.extract(record)
        parsed.append((text, candidates, ignored))

    print(f"\n{total} records, thresholds swept against a fixed corpus\n")
    print(f"  {'escalate_below':>14}  {'model calls':>11}  {'of total':>8}  "
          f"{'saved vs hybrid':>15}   why the model was called")
    print("  " + "-" * 96)

    for threshold in THRESHOLDS:
        config.escalate_below = threshold
        cascade = CascadeExtractor(rule, NoModel(), config)
        reasons = collections.Counter(
            classify(cascade._escalation_reason(text, candidates, ignored))
            for text, candidates, ignored in parsed
        )
        escalated = total - reasons["settled by frames"]
        mix = ", ".join(
            f"{name} {count}" for name, count in reasons.most_common()
            if name != "settled by frames"
        )
        print(f"  {threshold:>14.2f}  {escalated:>11}  {escalated / total:>7.0%}  "
              f"{1 - escalated / total:>14.0%}   {mix}")

    print("\n  hybrid, for comparison: 1 model call per record "
          f"({total} calls, 0% saved).")
    print("  'settled by frames' includes transcripts the person explicitly asked")
    print("  not to remember; escalating those would let the model overturn an")
    print("  instruction the frames already honoured.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cascade cost curve")
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--corpus", default=None,
                        help="A JSONL corpus to sweep instead of the generated seed corpus.")
    args = parser.parse_args()

    records = read_jsonl(args.corpus) if args.corpus else generate_corpus(args.records)
    run(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
