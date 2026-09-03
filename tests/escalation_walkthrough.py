"""The cascade, end to end: frames first, model only where they fell short.

    python3 -m tests.escalation_walkthrough
    python3 -m tests.escalation_walkthrough --quiet    # just the checks

This is the test that shows the whole proposition working on one short
transcript sequence, rather than testing each layer in isolation. Six
dictations go in. For each one it prints what the frames found, whether the
controller escalated and why, what the model added if it did, and what the
controller then decided - with the NLI verdict where inference was consulted.
Four questions are asked at the end.

It runs against `tests/fake_ollama.py`, so it needs no model installed and
takes about a second. The escalation decision itself never depends on the
model: it is computed from frame output alone, which is the point of putting it
in the controller.

What each record is here to demonstrate:

  R1  frames are enough                    -> no model call at all
  R2  no frame matched                     -> escalate; model supplies the memory
  R3  frames matched but missed most of it -> escalate; both sensors contribute
  R4  a contradiction the structural rule cannot see, because the model named
      the relation `uses_language` and nobody enumerated that
                                           -> NLI catches it, controller archives
  R5  escalated, and still rejected        -> escalation is not admission
  R6  "do not remember this"               -> never escalated, zero model calls
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kivi_memory.config import KiviConfig  # noqa: E402
from kivi_memory.core import is_exclusive  # noqa: E402
from kivi_memory.engine import KiviMemoryEngine  # noqa: E402
from tests import fake_ollama  # noqa: E402


PORT = 11603

TRANSCRIPTS = [
    ("R1", "frames are enough",
     "My manager is Neha and I prefer detailed written feedback on drafts."),
    ("R2", "no frame matches this shape",
     "Reviewing the quarterly notes, bullet summaries land better than prose for me."),
    ("R3", "a frame matches, but most of the sentence is unparsed",
     "For the Golden Goose task I am using Python for the semantic memory work."),
    ("R4", "a change of mind, named a way no rule anticipated",
     "We switched the Golden Goose prototype to Rust last week."),
    ("R5", "nothing worth keeping, and both sensors agree",
     "I walked to the cafe and had chai, the weather was nice."),
    ("R6", "an explicit retraction",
     "Do not remember any of this, just thinking out loud about reorganising things."),
]

QUESTIONS = [
    ("What kind of feedback do I prefer?", "written feedback"),
    ("What are we building the Golden Goose prototype in?", "rust"),
    ("How do I like summaries?", "bulleted"),
    ("What is my passport number?", None),          # None = must abstain
]

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cascade escalation, end to end")
    parser.add_argument("--quiet", action="store_true", help="Skip the narrative, print only checks.")
    args = parser.parse_args()
    loud = not args.quiet

    tmp = Path(tempfile.mkdtemp(prefix="kivi_walk_"))
    server, _ = fake_ollama.start(PORT)
    try:
        config = KiviConfig()
        config.ollama_host = f"http://127.0.0.1:{PORT}"
        config.cache_path = str(tmp / "llm_cache.db")
        config.health_timeout = 2.0
        config.request_timeout = 10.0
        config.apply_overrides(
            extractor="cascade", nli="llm", retrieval="hybrid", answerer="llm")

        engine = KiviMemoryEngine(str(tmp / "walkthrough.db"), config)
        history: dict[str, dict] = {}

        if loud:
            print("\n" + "=" * 78)
            print("INGEST - six dictations through the cascade")
            print("=" * 78)
            print(f"  extractor={config.extractor}  nli={config.nli}  "
                  f"escalate_below={config.escalate_below}  "
                  f"escalate_coverage={config.escalate_coverage}")

        for index, (tag, why, text) in enumerate(TRANSCRIPTS, start=1):
            before = engine.extractor.escalated
            result = engine.ingest_record({
                "id": f"rec_{index}", "created_at": f"2026-08-0{index}T10:00:00",
                "app": "dictation", "raw_asr": text.lower(),
                "formatted_text": text, "metadata": {},
            })
            escalated = engine.extractor.escalated > before
            history[tag] = {"escalated": escalated, "decisions": result["decisions"],
                            "ignored": result.get("ignored") or []}

            if loud:
                mark = "ESCALATED to the model" if escalated else "frames only, 0 model calls"
                print(f"\n  {tag}  {why}")
                print(f"      \"{text}\"")
                print(f"      -> {mark}")
                for decision in result["decisions"]:
                    candidate = decision.get("candidate") or {}
                    predicate = candidate.get("predicate") or "-"
                    object_ = str(candidate.get("object") or "-")[:34]
                    sensor = decision.get("extractor", "?")
                    line = (f"      {decision['action']:<14} {predicate:<24} = "
                            f"{object_:<36} [{sensor}]")
                    if decision.get("nli"):
                        line += f"  NLI={decision['nli']['label']} p={decision['nli']['probability']}"
                    print(line)
                for note in history[tag]["ignored"]:
                    print(f"      ignored: {note}")

        # -- what the sequence proves ------------------------------------------
        if loud:
            print("\n" + "=" * 78)
            print("CHECKS")
            print("=" * 78)

        def actions(tag: str) -> list[str]:
            return [d["action"] for d in history[tag]["decisions"]]

        def sensors(tag: str) -> set[str]:
            return {d.get("extractor") for d in history[tag]["decisions"]
                    if d["action"] not in ("NO_OP",)}

        check("R1  frames sufficed, so the model was never called",
              not history["R1"]["escalated"] and actions("R1").count("ADD") == 2,
              f"escalated={history['R1']['escalated']}, {actions('R1')}")

        check("R2  no frame matched, so it escalated and the model supplied the memory",
              history["R2"]["escalated"] and "llm" in sensors("R2") and "ADD" in actions("R2"),
              f"sensors={sensors('R2')}, {actions('R2')}")

        check("R3  a confident frame still escalated, on coverage",
              history["R3"]["escalated"] and sensors("R3") == {"rule", "llm"},
              f"sensors={sensors('R3')} - both sensors contributed to one record")

        # The heart of it. `uses_language` is not in the exclusive-predicate
        # list, does not end in `_is` and does not start with `prefers_`, so the
        # structural rule has no way to know a second value replaces the first.
        conflict = next((d for d in history["R4"]["decisions"] if d["action"] == "UPDATE"), None)
        check("R4  the structural rule alone could NOT have caught this conflict",
              is_exclusive("uses_language") is False,
              "is_exclusive('uses_language') = False")
        check("R4  NLI caught it and the controller archived the old memory",
              conflict is not None
              and conflict.get("nli", {}).get("label") == "contradiction"
              and conflict["candidate"]["object"] == "rust",
              f"{conflict['action']} on NLI {conflict['nli']['label']} "
              f"p={conflict['nli']['probability']}" if conflict else "no UPDATE decision")

        check("R5  escalating is not admitting - the controller still said no",
              history["R5"]["escalated"] and "ADD" not in actions("R5"),
              f"escalated={history['R5']['escalated']}, {actions('R5')}")

        check("R6  an explicit retraction is never escalated",
              not history["R6"]["escalated"] and "ADD" not in actions("R6"),
              f"escalated={history['R6']['escalated']}, {actions('R6')}")

        check("the cascade skipped the model on 2 of 6 records",
              engine.extractor.escalated == 4 and engine.extractor.seen == 6,
              f"{engine.extractor.escalated}/{engine.extractor.seen} escalated "
              f"(hybrid would be 6/6)")

        # -- the read path ------------------------------------------------------
        if loud:
            print("\n" + "=" * 78)
            print("MEMORY STATE")
            print("=" * 78)
            for memory in engine.memories(50):
                flag = "archived" if memory["status"] == "archived" else "active  "
                print(f"  [{flag}] {memory['predicate']:<24} = "
                      f"{str(memory['object'])[:32]:<34} {memory['scope']}")

            print("\n" + "=" * 78)
            print("ASK")
            print("=" * 78)

        for question, expected in QUESTIONS:
            answer = engine.ask(question)
            if loud:
                print(f"\n  Q  {question}")
                print(f"  A  {answer['answer']}")
                if not answer["abstained"]:
                    cited = [m["canonical_text"] for m in answer.get("used_memories", [])]
                    for claim in cited[:3]:
                        print(f"     cites: {claim}")

            if expected is None:
                check(f"abstains: {question!r}", answer["abstained"], answer["answer"][:60])
            else:
                check(f"answers correctly: {question!r}",
                      not answer["abstained"] and expected in answer["answer"].lower(),
                      answer["answer"][:70])

        # The superseded claim must not resurface in an answer.
        rust = engine.ask("What are we building the Golden Goose prototype in?")
        check("the archived Python memory is not cited after the switch",
              "python" not in rust["answer"].lower(),
              rust["answer"][:70])

        if loud:
            print("\n" + "=" * 78)
            print("NOTE - a real limitation this run exposes")
            print("=" * 78)
            print("  The frames also stored uses_tool=python under scope")
            print("  'project:golden_goose_task' and uses_tool=rust under")
            print("  'project:golden_goose_prototype'. Those scope slugs come from two")
            print("  different phrasings of the same project, so neither the structural")
            print("  rule nor NLI links them - conflict detection is scoped to the same")
            print("  (subject, scope). The stale one survives. Scope unification is not")
            print("  implemented; it is listed in the README limitations.")

    finally:
        server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
