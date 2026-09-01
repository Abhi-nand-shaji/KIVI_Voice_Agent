"""Show what NLI catches that the structural rule cannot.

    python3 -m tests.nli_demo                 # uses whatever backend is available
    python3 -m tests.nli_demo --nli off       # force the structural rule only
    python3 -m tests.nli_demo --nli llm       # force NLI through the local model

The structural exclusivity rule compares memories that share a predicate:
`manager_is` against `manager_is`. That covers most conflicts, because the
grammatical frames name a relation the same way every time.

It does not cover the case this file demonstrates. When the LLM extractor reads
two transcripts it may name the *same* relation two different ways -
`home_city_is` in one, `residence_city_is` in the next. Nothing structural can
link those, so without inference both are stored and the person now has two
contradictory memories about where they live, with no record that they clash.

Two scenarios, run under both settings so the difference is visible:

  1. CONTRADICTION - same claim, different relation names, incompatible values.
     Expect: NLI archives the older one. Structural rule keeps both.
  2. PARAPHRASE    - same claim worded differently, different relation names.
     Expect: NLI merges them into one reinforced memory. Structural rule
     stores two.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kivi_memory.config import KiviConfig  # noqa: E402
from kivi_memory.core import Candidate  # noqa: E402
from kivi_memory.db import connect, migrate  # noqa: E402
from kivi_memory.engine import KiviMemoryEngine  # noqa: E402


SCENARIOS = {
    "contradiction": {
        "blurb": "Same relation, named two ways, with incompatible values.",
        "pair": [
            ("home_city_is", "bangalore", "User currently lives in Bangalore.",
             "I currently live in Bangalore.", "2026-06-01T10:00:00"),
            ("residence_city_is", "chennai", "User currently lives in Chennai.",
             "I currently live in Chennai.", "2026-08-01T10:00:00"),
        ],
        "want": "the older memory archived, the newer one active",
    },
    "paraphrase": {
        "blurb": "Same preference, worded two ways, named two ways.",
        "pair": [
            ("prefers_reply", "short replies", "User prefers short replies.",
             "I prefer short replies.", "2026-06-01T10:00:00"),
            ("likes_response", "replies that are short", "User prefers replies that are short.",
             "I like replies that are short.", "2026-08-01T10:00:00"),
        ],
        "want": "one memory, reinforced - not two",
    },
}


def build(predicate: str, object_: str, canonical: str, evidence: str) -> Candidate:
    return Candidate(
        memory_type="fact" if predicate.endswith("_is") else "preference",
        subject="user", predicate=predicate, object=object_, scope="profile",
        canonical_text=canonical, evidence=evidence,
        importance=0.88, confidence=0.9, utility=0.88, decay_rate=0.005, source="demo",
    )


def run_scenario(name: str, config: KiviConfig) -> dict:
    scenario = SCENARIOS[name]
    tmp = Path(tempfile.mkdtemp(prefix=f"kivi_nli_{name}_"))
    engine = KiviMemoryEngine(str(tmp / "demo.db"), config)

    actions, verdicts = [], []
    with connect(engine.db_path) as conn:
        migrate(conn)
        for index, (predicate, object_, canonical, evidence, created) in enumerate(scenario["pair"], start=1):
            record = {"id": f"d{index}", "created_at": created, "app": "dictation",
                      "raw_asr": evidence.lower(), "formatted_text": evidence, "metadata": {}}
            engine._upsert_transcript(conn, record)
            decision = engine._apply_candidate(conn, record, build(predicate, object_, canonical, evidence))
            actions.append(decision["action"])
            verdicts.append(decision.get("nli"))
        conn.commit()
        rows = [dict(row) for row in conn.execute(
            "SELECT canonical_text, status, recurrence FROM memories ORDER BY created_at")]

    return {
        "actions": actions,
        "verdicts": [v for v in verdicts if v],
        "memories": rows,
        "active": sum(1 for row in rows if row["status"] != "archived"),
        "backend": engine.nli.describe().get("backend"),
    }


def show(name: str, result: dict) -> None:
    scenario = SCENARIOS[name]
    print(f"\n  {name.upper()} - {scenario['blurb']}")
    print(f"  wanted: {scenario['want']}")
    print(f"  actions: {' then '.join(result['actions'])}")
    for row in result["memories"]:
        mark = "archived" if row["status"] == "archived" else "active  "
        seen = f"  (seen {row['recurrence']}x)" if row["recurrence"] > 1 else ""
        print(f"    [{mark}] {row['canonical_text']}{seen}")
    for verdict in result["verdicts"]:
        print(f"    NLI said: {verdict['label']} p={verdict['probability']} via {verdict['backend']}")
    if not result["verdicts"]:
        print("    NLI said: nothing - it was not consulted")


def main() -> int:
    parser = argparse.ArgumentParser(description="What NLI catches that the structural rule cannot")
    parser.add_argument("--nli", choices=["off", "llm", "cross-encoder", "auto", "both"], default="both")
    args = parser.parse_args()

    modes = ["off", "llm"] if args.nli == "both" else [args.nli]
    summary = {}

    for mode in modes:
        config = KiviConfig.from_env().apply_overrides(
            extractor="rule", nli=mode,
            retrieval="lexical" if mode == "off" else "hybrid",
            answerer="template",
        )
        print("\n" + "=" * 72)
        print(f"NLI = {mode}")
        print("=" * 72)
        for name in SCENARIOS:
            result = run_scenario(name, config)
            show(name, result)
            summary[(mode, name)] = result["active"]

    if args.nli == "both":
        print("\n" + "=" * 72)
        print("ACTIVE MEMORIES AFTER EACH PAIR")
        print("=" * 72)
        print(f"  {'scenario':<16}{'nli off':>10}{'nli on':>10}   wanted")
        for name in SCENARIOS:
            off, on = summary[("off", name)], summary[("llm", name)]
            print(f"  {name:<16}{off:>10}{on:>10}   1")
        print("\n  Two active memories means the system holds a contradiction it")
        print("  cannot see. One means it noticed. The structural rule cannot")
        print("  reach either case, because the two relations are named")
        print("  differently and it only compares like with like.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
