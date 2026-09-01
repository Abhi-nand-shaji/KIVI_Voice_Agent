"""End-to-end checks for the model-backed paths, against `tests/fake_ollama.py`.

    python3 -m tests.run_llm_path

Covers, in order:
  1. deterministic path still passes after the refactor
  2. LLM extractor -> same Candidate objects -> same controller
  3. NLI catches a contradiction on a predicate that is NOT in the
     exclusive-predicate whitelist (the thing the old controller could not do)
  4. NLI merges a paraphrase instead of storing a duplicate
  5. the grounded answerer refuses to cite memories it was not given
  6. a dead daemon degrades to rules instead of silently forgetting everything
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kivi_memory.config import KiviConfig  # noqa: E402
from kivi_memory.engine import KiviMemoryEngine  # noqa: E402
from kivi_memory.evaluation import run_evaluation  # noqa: E402
from tests import fake_ollama  # noqa: E402


PORT = 11599
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def make_config(tmp: Path, **overrides) -> KiviConfig:
    config = KiviConfig()
    config.ollama_host = f"http://127.0.0.1:{PORT}"
    config.cache_path = str(tmp / "llm_cache.db")
    config.health_timeout = 2.0
    config.request_timeout = 10.0
    return config.apply_overrides(**overrides)


def record(idx: int, text: str, created: str, app: str = "dictation") -> dict:
    return {"id": f"t_{idx:03d}", "created_at": created, "app": app,
            "raw_asr": text.lower(), "formatted_text": text, "metadata": {}}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kivi_test_"))
    server, _ = fake_ollama.start(PORT)
    try:
        # 1 -- deterministic path -------------------------------------------
        det = run_evaluation(
            db_path=str(tmp / "det.db"),
            corpus_path=str(tmp / "corpus.jsonl"),
            out_path=str(tmp / "det.json"),
            records=500,
            config=KiviConfig().apply_overrides(
                extractor="rule", nli="off", retrieval="lexical", answerer="template"
            ),
        )
        # The deterministic path is the no-install fallback: all controller and
        # admission checks must hold, and it must answer most questions. The two
        # known misses are lexical paraphrase gaps documented in the README.
        check("deterministic: controller + admission policy fully holds",
              det["summary"]["controller_checks"]["accuracy"] == 1.0
              and det["summary"]["admission_checks"]["accuracy"] == 1.0,
              f"controller {det['summary']['controller_checks']['passed']}/"
              f"{det['summary']['controller_checks']['total']}, admission "
              f"{det['summary']['admission_checks']['passed']}/"
              f"{det['summary']['admission_checks']['total']}")
        check("deterministic: answers most questions",
              det["summary"]["answer_checks"]["passed"] >= 4,
              f"{det['summary']['answer_checks']['passed']}/{det['summary']['answer_checks']['total']}")

        # 2 -- full local-model path ----------------------------------------
        llm = run_evaluation(
            db_path=str(tmp / "llm.db"),
            corpus_path=str(tmp / "corpus.jsonl"),
            out_path=str(tmp / "llm.json"),
            records=500,
            config=make_config(tmp, extractor="llm", nli="llm", retrieval="hybrid", answerer="llm"),
        )
        summary = llm["summary"]
        check("llm path: extractor is the model, not the rules",
              summary["backends"]["extractor"]["backend"].startswith("ollama"),
              summary["backends"]["extractor"]["backend"])
        check("llm path: embeddings are real vectors",
              summary["backends"]["embedder"]["semantic"] is True,
              str(summary["backends"]["embedder"]))
        check("llm path: memories are attributed to the llm sensor",
              summary["database"]["memories_by_source"].get("llm", 0) > 0,
              str(summary["database"]["memories_by_source"]))
        check("llm path: controller invariants hold unchanged",
              summary["controller_checks"]["accuracy"] == 1.0,
              f"{summary['controller_checks']['passed']}/{summary['controller_checks']['total']}")
        check("llm path: admission policy unchanged by the sensor swap",
              summary["admission_checks"]["accuracy"] == 1.0,
              f"{summary['admission_checks']['passed']}/{summary['admission_checks']['total']}")
        check("llm path: answers are model-generated",
              any(case["answer"].get("reason", "").find("ollama") >= 0
                  for case in llm["cases"] if not case["answer"]["abstained"]))
        check("llm path: NLI vetoes answers that contradict the reference claim",
              any("nli-contradiction-veto" in str((case.get("grade") or {}).get("grader"))
                  for case in llm["cases"]),
              str({case["id"]: (case.get("grade") or {}).get("entailment", {}).get("label")
                   for case in llm["cases"] if case.get("grade", {}).get("entailment")}))
        check("llm path: intent routed by the model",
              all(str((case["answer"].get("intent") or {}).get("backend", "")).startswith("llm")
                  for case in llm["cases"]),
              str({case["id"]: (case["answer"].get("intent") or {}).get("intent") for case in llm["cases"]}))
        check("llm path: still abstains on unknown facts",
              next(c for c in llm["cases"] if c["id"] == "abstain_unknown")["passed"])
        check("llm path: NLI actually ran",
              summary["database"]["decisions_with_nli"] > 0,
              f"{summary['database']['decisions_with_nli']} decisions carry an NLI label")

        # 3 -- contradiction outside the whitelist ---------------------------
        # `based_in` IS whitelisted, so use a predicate that is not: the fake
        # model emits `topic`/`focuses_on`, and we feed a city conflict through
        # a free-form predicate to prove NLI, not the whitelist, caught it.
        engine = KiviMemoryEngine(str(tmp / "nli.db"),
                                  make_config(tmp, extractor="rule", nli="llm",
                                              retrieval="hybrid", answerer="template"))
        from kivi_memory.core import Candidate  # noqa: PLC0415

        def candidate(city: str, confidence: float) -> Candidate:
            return Candidate(
                memory_type="fact", subject="user", predicate="home_base",  # NOT in EXCLUSIVE_PREDICATES
                object=city, scope="profile",
                canonical_text=f"User currently lives in {city.title()}.",
                evidence=f"I live in {city.title()}.", importance=0.85, confidence=confidence,
                utility=0.8, decay_rate=0.005, source="test",
            )

        from kivi_memory.db import connect, migrate  # noqa: PLC0415
        with connect(engine.db_path) as conn:
            migrate(conn)
            engine._upsert_transcript(conn, record(1, "I live in Bangalore.", "2026-08-01T10:00:00"))
            engine._upsert_transcript(conn, record(2, "I live in Chennai.", "2026-08-20T10:00:00"))
            first = engine._apply_candidate(conn, record(1, "I live in Bangalore.", "2026-08-01T10:00:00"),
                                            candidate("bangalore", 0.86))
            second = engine._apply_candidate(conn, record(2, "I live in Chennai.", "2026-08-20T10:00:00"),
                                             candidate("chennai", 0.88))
            conn.commit()
            statuses = {row["object"]: row["status"] for row in conn.execute(
                "SELECT object, status FROM memories WHERE predicate='home_base'")}

        check("NLI supersedes a conflict on a non-whitelisted predicate",
              first["action"] == "ADD" and second["action"] == "UPDATE",
              f"{first['action']} then {second['action']}")
        check("superseded memory is archived, not deleted",
              statuses.get("bangalore") == "archived" and statuses.get("chennai") == "active",
              str(statuses))
        check("the NLI verdict is written to the audit trail",
              second.get("nli") is not None and second["nli"]["label"] == "contradiction",
              json.dumps(second.get("nli")))

        # 4 -- paraphrase merge ---------------------------------------------
        engine2 = KiviMemoryEngine(str(tmp / "dup.db"),
                                   make_config(tmp, extractor="rule", nli="llm",
                                               retrieval="hybrid", answerer="template"))
        para_a = Candidate(memory_type="preference", subject="user", predicate="prefers_tone",
                           object="short and technical answers", scope="communication",
                           canonical_text="User prefers short and technical answers.",
                           evidence="short and technical answers", importance=0.8, confidence=0.85,
                           utility=0.8, decay_rate=0.01, source="test")
        para_b = Candidate(memory_type="preference", subject="user", predicate="prefers_tone",
                           object="technical answers that are short", scope="communication",
                           canonical_text="User prefers technical answers that are short.",
                           evidence="technical short answers", importance=0.8, confidence=0.85,
                           utility=0.8, decay_rate=0.01, source="test")
        with connect(engine2.db_path) as conn:
            migrate(conn)
            engine2._upsert_transcript(conn, record(1, "short technical answers", "2026-08-01T10:00:00"))
            engine2._upsert_transcript(conn, record(2, "technical short answers", "2026-08-02T10:00:00"))
            engine2._apply_candidate(conn, record(1, "a", "2026-08-01T10:00:00"), para_a)
            merged = engine2._apply_candidate(conn, record(2, "b", "2026-08-02T10:00:00"), para_b)
            conn.commit()
            count = conn.execute(
                "SELECT count(*) AS n FROM memories WHERE predicate='prefers_tone' AND status<>'archived'"
            ).fetchone()["n"]
        check("paraphrase is merged rather than duplicated",
              merged["action"] == "REINFORCE" and count == 1,
              f"action={merged['action']} active_rows={count}")

        # 5 -- grounding guard ----------------------------------------------
        from kivi_memory.answerer import LlmAnswerer  # noqa: PLC0415
        from kivi_memory.llm import OllamaClient  # noqa: PLC0415

        class Liar(OllamaClient):
            def chat_json(self, *args, **kwargs):
                return {"answer": "You live on Mars.", "abstain": False,
                        "used_memory_ids": ["mem_not_supplied"]}

        config = make_config(tmp, answerer="llm")
        answerer = LlmAnswerer(Liar(config), config)
        row = {"id": "mem_real", "predicate": "based_in", "object": "bangalore", "scope": "profile",
               "canonical_text": "User is based in Bangalore.", "memory_type": "fact",
               "status": "active", "confidence": 0.9, "last_seen_at": "2026-08-01T10:00:00"}
        composed = answerer.compose("Where do I live?", [row])
        check("ungrounded answer citing an unknown memory id is rejected",
              "mars" not in composed["answer"].lower() and "rejected" in composed["backend"],
              f"{composed['backend']}: {composed['answer']}")

        # 5b -- generalisation to a foreign corpus ---------------------------
        from tests import foreign_corpus  # noqa: PLC0415
        foreign = foreign_corpus.run(300)
        check("learns from a corpus it was not written against",
              foreign["memories"] >= 10 and foreign["answered"] >= 4,
              f"{foreign['memories']} memories, {foreign['answered']}/{foreign['total']} questions answered")

        # 5c -- dictation mode ------------------------------------------------
        dict_engine = KiviMemoryEngine(str(tmp / "dictate.db"),
                                       make_config(tmp, extractor="rule", nli="off",
                                                   retrieval="lexical", answerer="template"))
        spoken = dict_engine.dictate(
            "um so I prefer detailed written feedback you know and I need to send the audit by friday")
        check("dictation returns polished text and learns quietly",
              spoken["formatted"] != spoken["raw"] and len(spoken["learned"]) >= 1
              and "um" not in spoken["formatted"].lower().split(),
              f"learned {len(spoken['learned'])}: {[l['claim'][:34] for l in spoken['learned']]}")

        forgotten = dict_engine.forget(spoken["learned"][0]["id"])
        check("a learned memory can be dropped from the interface",
              forgotten["ok"] and forgotten["memory"]["status"] == "archived",
              str(forgotten.get("memory", {}).get("status")))

        # 6 -- daemon down ---------------------------------------------------
        dead = KiviConfig().apply_overrides(extractor="llm", nli="off", retrieval="lexical",
                                            answerer="template")
        dead.ollama_host = "http://127.0.0.1:1"   # nothing listening
        dead.cache_path = str(tmp / "dead_cache.db")
        dead.health_timeout = 0.5
        dead_engine = KiviMemoryEngine(str(tmp / "dead.db"), dead)
        dead_engine.ingest_record(record(
            1, "I prefer concise technical explanations. Keep the answers short.", "2026-08-01T10:00:00"))
        remembered = dead_engine.stats()["counts"]["memories"]
        check("dead daemon degrades to rules instead of forgetting everything",
              remembered > 0, f"{remembered} memories still learned")

    finally:
        server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
