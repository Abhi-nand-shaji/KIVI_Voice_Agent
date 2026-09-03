"""What the evaluation actually asserts.

Three independent sets, kept apart from the harness that runs them because
they are the part a reviewer should read:

`DEFAULT_QUESTIONS`  - what the system should be able to answer from the seed
                       corpus, including one it must *refuse* to answer.
`CONTROLLER_CHECKS`  - invariants of the decision policy, each exercised by
                       constructing candidates directly and asserting on the
                       action taken. These must hold identically on every
                       backend; if a sensor swap changes one, the separation
                       this architecture claims is not real.
`ADMISSION_LABELS`   - hand-labelled keep/drop candidates at *fixed* values.

The fixed values in `ADMISSION_LABELS` matter more than they look. An earlier
version defined the probes relative to the thresholds being tested, so every
setting in the calibration grid scored identically and the surface came out
flat - an experiment that measured nothing while appearing to pass.
"""

from __future__ import annotations

from typing import Any

from .core import utility_score
from .engine import KiviMemoryEngine


DEFAULT_QUESTIONS = [
    {
        "id": "response_style",
        "query": "What response style do I prefer?",
        "expected_contains": ["concise"],
        "expected_claim": "The user prefers concise communication.",
        "expect_abstain": False,
    },
    {
        "id": "project_language",
        "query": "Which language am I using for the Golden Goose prototype?",
        "expected_contains": ["python"],
        "expected_claim": "The user is using Python for the Golden Goose prototype.",
        "expect_abstain": False,
    },
    {
        "id": "find_slack",
        "query": "Hey Kivi, find the dictation I did around 5 PM yesterday in Slack and polish it for the meeting.",
        "expected_contains": ["predictive maintenance", "sensor drift"],
        "expected_claim": "The retrieved dictation is about predictive maintenance and sensor drift.",
        "expect_abstain": False,
    },
    {
        "id": "assignment",
        "query": "What assignment am I working on?",
        "expected_contains": ["sarvam", "golden goose"],
        "expected_claim": "The user is working on the Sarvam AI Golden Goose task.",
        "expect_abstain": False,
    },
    {
        "id": "abstain_unknown",
        "query": "What is my passport number?",
        "expected_contains": [],
        "expect_abstain": True,
    },
    {
        "id": "ignored",
        "query": "What did Kivi deliberately ignore?",
        "expected_contains": ["low durable utility"],
        "expected_claim": "The assistant explains which candidates it chose not to remember.",
        "expect_abstain": False,
    },
]


# Controller-level expectations. These check the memory policy rather than the
# wording of an answer, so they hold no matter which sensor produced the
# candidates - which is the point of keeping the controller fixed.
CONTROLLER_CHECKS = [
    {
        "id": "records_provenance",
        "description": "Every stored memory has at least one supporting transcript.",
        "check": lambda engine, stats: all(memory["evidence"] for memory in engine.memories(50)),
    },
    {
        "id": "rejects_something",
        "description": "The controller rejected or no-op'd at least one candidate rather than storing everything.",
        "check": lambda engine, stats: (
            stats["decisions_by_action"].get("REJECT", 0) + stats["decisions_by_action"].get("NO_OP", 0)
        ) > 0,
    },
    {
        "id": "reinforces_recurrence",
        "description": "Repeated evidence reinforced an existing memory instead of duplicating it.",
        "check": lambda engine, stats: stats["decisions_by_action"].get("REINFORCE", 0) > 0,
    },
    {
        "id": "no_orphan_memories",
        "description": "Active memory count is below the number of transcripts (selective, not a transcript log).",
        "check": lambda engine, stats: stats["counts"]["memories"] < stats["counts"]["transcripts"],
    },
    {
        "id": "forget_archives",
        "description": "An explicit user correction archives the memory and logs a FORGET decision.",
        "check": lambda engine, stats: _forget_works(engine),
    },
    {
        "id": "multi_dictation_recovery",
        "description": "At least one answer is supported by evidence drawn from more than one dictation.",
        "check": lambda engine, stats: _spans_multiple_dictations(engine),
    },
    {
        "id": "rejects_below_threshold",
        "description": "A candidate under the admission thresholds is REJECTED, not stored.",
        "check": lambda engine, stats: _rejects_weak_candidate(engine),
    },
    {
        "id": "hedged_stays_tentative",
        "description": "A hedged claim is stored tentative or rejected - never as settled fact.",
        "check": lambda engine, stats: _hedged_is_not_active(engine),
    },
    {
        "id": "weak_conflict_does_not_supersede",
        "description": "A low-confidence contradicting claim does NOT archive an established memory.",
        "check": lambda engine, stats: _weak_conflict_held(engine),
    },
    {
        "id": "growth_is_sublinear",
        "description": "Memory count grows far more slowly than the transcript log (selective, not a copy).",
        "check": lambda engine, stats: _growth_sublinear(engine),
    },
]


# ---------------------------------------------------------------------------
# Controller probes.
#
# These construct candidates directly and assert what the controller does with
# them. They exist because the question-answering cases could not falsify the
# admission thresholds at all: `calibrate` found every setting in a 45-point
# grid scored identically. A policy nothing can fail is not a policy.
# ---------------------------------------------------------------------------

# A small hand-labelled admission set with FIXED confidence/importance values.
#
# The relative probes above check that the admission gate is applied at all;
# they pass at any threshold, which is why `calibrate` reported a flat surface.
# These say where the line should sit: each row is a judgement that a candidate
# of this strength deserves - or does not deserve - a place in memory. A
# threshold that admits the throwaway aside, or rejects the plainly stated
# preference, now scores worse. That is what makes the numbers falsifiable.
ADMISSION_LABELS = [
    {
        "id": "plain_preference",
        "why": "A preference stated plainly and directly. Must be kept.",
        "memory_type": "preference", "confidence": 0.86, "importance": 0.88,
        "expect": "store",
    },
    {
        "id": "stated_fact",
        "why": "A profile fact stated once, unambiguously. Must be kept.",
        "memory_type": "fact", "confidence": 0.84, "importance": 0.82,
        "expect": "store",
    },
    {
        "id": "borderline_preference",
        "why": "A real preference, stated once, less emphatically. Should still be kept.",
        "memory_type": "preference", "confidence": 0.66, "importance": 0.62,
        "expect": "store",
    },
    {
        "id": "passing_aside",
        "why": "An offhand remark with little future bearing. Must not be kept.",
        "memory_type": "event", "confidence": 0.44, "importance": 0.30,
        "expect": "reject",
    },
    {
        "id": "vague_mention",
        "why": "A vague, low-confidence mention. Must not be kept.",
        "memory_type": "event", "confidence": 0.35, "importance": 0.40,
        "expect": "reject",
    },
]


def score_admission_labels(engine: KiviMemoryEngine) -> list[dict[str, Any]]:
    results = []
    for label in ADMISSION_LABELS:
        probe = _probe_engine(engine, f"label_{label['id']}")
        candidate = _probe_candidate(
            memory_type=label["memory_type"],
            predicate=f"prefers_{label['id']}" if label["memory_type"] == "preference" else f"{label['id']}_is",
            object=label["id"].replace("_", " "),
            canonical_text=f"User: {label['id'].replace('_', ' ')}.",
            confidence=label["confidence"],
            importance=label["importance"],
            utility=utility_score(label["importance"], label["confidence"], label["memory_type"]),
        )
        decision = _apply_probe(probe, [candidate])[0]
        stored = decision["action"] in ("ADD", "ADD_TENTATIVE", "UPDATE", "REINFORCE")
        ok = stored if label["expect"] == "store" else not stored
        results.append({
            "id": label["id"], "why": label["why"], "expect": label["expect"],
            "action": decision["action"], "utility": decision["utility"],
            "confidence": decision["confidence"], "passed": ok,
        })
    return results


def _probe_engine(engine: KiviMemoryEngine, suffix: str):
    """A scratch database sharing this engine's config, for controller probes."""
    import tempfile  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415
    tmp = _Path(tempfile.mkdtemp(prefix="kivi_probe_")) / f"{suffix}.db"
    return KiviMemoryEngine(str(tmp), engine.config)


def _probe_candidate(**overrides):
    from .core import Candidate  # noqa: PLC0415
    base = dict(
        memory_type="preference", subject="user", predicate="prefers_probe",
        object="a probe value", scope="probe", canonical_text="User prefers a probe value.",
        evidence="probe", importance=0.9, confidence=0.9, utility=0.9,
        decay_rate=0.01, status="active", reason="probe", source="probe",
    )
    base.update(overrides)
    return Candidate(**base)


def _probe_record(idx: int = 1, text: str = "probe", created: str = "2026-08-01T10:00:00"):
    return {"id": f"probe_{idx}", "created_at": created, "app": "probe",
            "raw_asr": text, "formatted_text": text, "metadata": {}}


def _apply_probe(engine: KiviMemoryEngine, candidates: list) -> list[dict[str, Any]]:
    from .db import connect, migrate  # noqa: PLC0415
    decisions = []
    with connect(engine.db_path) as conn:
        migrate(conn)
        for idx, candidate in enumerate(candidates, start=1):
            record = _probe_record(idx, created=f"2026-08-{idx:02d}T10:00:00")
            engine._upsert_transcript(conn, record)
            decisions.append(engine._apply_candidate(conn, record, candidate))
        conn.commit()
    return decisions


def _rejects_weak_candidate(engine: KiviMemoryEngine) -> bool:
    probe = _probe_engine(engine, "reject")
    cfg = probe.config
    # Deliberately just under both admission gates.
    weak = _probe_candidate(
        confidence=max(0.0, cfg.admit_confidence - 0.15),
        importance=0.1,
        utility=max(0.0, cfg.admit_utility - 0.15),
        object="a weak probe value",
    )
    decision = _apply_probe(probe, [weak])[0]
    return decision["action"] == "REJECT"


def _hedged_is_not_active(engine: KiviMemoryEngine) -> bool:
    probe = _probe_engine(engine, "hedge")
    hedged = _probe_candidate(
        status="tentative",
        confidence=probe.config.tentative_confidence - 0.1,
        object="a hedged probe value",
    )
    decision = _apply_probe(probe, [hedged])[0]
    return decision["action"] in ("ADD_TENTATIVE", "REJECT")


def _weak_conflict_held(engine: KiviMemoryEngine) -> bool:
    """An established memory must not be overturned by a weak contradiction."""
    probe = _probe_engine(engine, "conflict")
    strong = _probe_candidate(predicate="probe_city_is", object="alpha", scope="probe_profile",
                              canonical_text="User's probe city is Alpha.", confidence=0.95)
    weak = _probe_candidate(predicate="probe_city_is", object="beta", scope="probe_profile",
                            canonical_text="User's probe city is Beta.",
                            confidence=max(0.0, probe.config.supersede_confidence - 0.2))
    decisions = _apply_probe(probe, [strong, weak])
    from .db import connect  # noqa: PLC0415
    with connect(probe.db_path) as conn:
        statuses = {row["object"]: row["status"] for row in conn.execute(
            "SELECT object, status FROM memories WHERE predicate='probe_city_is'")}
    return decisions[0]["action"] == "ADD" and statuses.get("alpha") == "active"


def _spans_multiple_dictations(engine: KiviMemoryEngine) -> bool:
    """Recovering understanding distributed across several dictations.

    A memory whose confidence was built from separate dictations, and whose
    provenance still names each of them, is the concrete form of that claim.
    """
    for memory in engine.memories(60):
        transcripts = {item.get("transcript_id") for item in memory.get("evidence", [])}
        if len(transcripts) >= 2:
            return True
    return False


def _growth_sublinear(engine: KiviMemoryEngine) -> bool:
    growth = engine.growth()
    latest = growth.get("latest")
    if not latest or latest["transcripts"] < 20:
        return False
    return latest["active_memories"] < latest["transcripts"] * 0.25


def grade_answer(engine: KiviMemoryEngine, question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Grade an answer by entailment when an NLI backend is available.

    Substring matching is not a correctness test: `"python" in text` passes on
    "you are NOT using Python", and it fails on a correct answer that happens
    to word things differently - which is exactly what happens once answers are
    generated rather than templated. The project already carries an NLI model
    for conflict detection, so the evaluation uses it to ask the right
    question: does the answer entail the reference claim?

    So NLI is used as a *veto*, not as the pass condition: an answer must still
    contain the expected terms (reproducible with no model installed), and an
    answer that a model judges to CONTRADICT the reference claim fails even if
    the substring is present. That is strictly stronger than substring alone,
    it needs no model to run, and it uses inference where inference is most
    reliable. The raw entailment label is reported either way.
    """
    text = answer["answer"]
    lowered = text.lower()
    substring_ok = all(value.lower() in lowered for value in question.get("expected_contains", []))
    abstain_ok = bool(answer["abstained"]) == bool(question["expect_abstain"])

    claim = question.get("expected_claim")
    entailment: dict[str, Any] | None = None
    contradicted = False
    if claim and not question["expect_abstain"] and engine.nli.name != "off":
        result = engine.nli.classify(text, claim)
        entailment = result.as_dict()
        contradicted = result.label == "contradiction" and result.probability >= 0.5

    return {
        "substring_ok": substring_ok,
        "abstain_ok": abstain_ok,
        "contradicted_reference": contradicted,
        "entailment": entailment,
        "grader": "substring + nli-contradiction-veto" if entailment else "substring",
        "passed": abstain_ok and substring_ok and not contradicted,
    }


def _forget_works(engine: KiviMemoryEngine) -> bool:
    before = engine.stats()["memories_by_status"].get("archived", 0)
    result = engine.ask("Forget that I prefer bulleted summaries.")
    after = engine.stats()
    if result.get("abstained"):
        return False
    return (
        after["memories_by_status"].get("archived", 0) > before
        and after["decisions_by_action"].get("FORGET", 0) > 0
    )


def _dictation_example(engine: KiviMemoryEngine) -> dict[str, Any]:
    """One round-trip through the OTHER mode, so the evaluation covers both.

    Regular dictation and Hey Kivi are different products with the same memory
    behind them; an evaluation that only exercises the question path does not
    show what memory does during ordinary use.
    """
    try:
        return engine.dictate(
            "um so I need to uh send the evaluation writeup before friday and you know "
            "keep the summary short when you write it up for me",
            app="slack",
        )
    except Exception as exc:  # never let the demo break the run
        return {"error": repr(exc)}
