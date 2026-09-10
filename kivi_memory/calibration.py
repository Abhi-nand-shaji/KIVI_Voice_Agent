"""Grid-searching the controller thresholds.

`admit_utility` and `admit_confidence` are the two numbers that decide what
gets remembered, and "we picked 0.55" is not an answer a reviewer should
accept. This sweeps them against the fixed labelled set in `eval_cases.py` and
reports the surface, so the chosen values are a defensible point on a measured
curve rather than a constant somebody typed.

The remaining constants - the utility weights, per-type future value, per-frame
decay and confidence priors - are *not* fitted here. With 500 records and 21
checks, fitting them would be overfitting; they encode an ordering, and their
ordering is defended in comments where they are defined.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import KiviConfig
from .evaluation import run_evaluation


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

# Wide enough to cross the boundary. A grid narrower than the margin in the
# labelled set produces a flat surface and tells you nothing, which is what the
# first version of this did.
CALIBRATION_GRID = {
    "admit_utility": [0.35, 0.45, 0.55, 0.65, 0.75],
    "admit_confidence": [0.30, 0.45, 0.60, 0.75],
    "supersede_confidence": [0.62, 0.72, 0.82],
}


def calibrate(
    db_path: str = "data/kivi.db",
    corpus_path: str = "",   # unused: each trial generates its own, in a temp dir
    out_path: str = "data/calibration.json",
    records: int = 300,
    config: KiviConfig | None = None,
) -> dict[str, Any]:
    """Search the controller's admission thresholds instead of asserting them.

    `admit_utility=0.55`, `admit_confidence=0.50` and
    `supersede_confidence=0.72` were hand-picked numbers. This runs the full
    ingest + evaluation across a grid and reports the surface, so the shipped
    values are a defensible choice with a sensitivity analysis behind them
    rather than three constants someone liked.

    Score prefers, in order: checks passed, then selectivity (fewer memories
    for the same score is a better filter, which is the whole product claim).
    """
    import itertools
    import tempfile

    base = config or KiviConfig.from_env()
    results = []
    keys = list(CALIBRATION_GRID)

    with tempfile.TemporaryDirectory(prefix="kivi_calib_") as tmp:
        for combo in itertools.product(*(CALIBRATION_GRID[key] for key in keys)):
            settings = dict(zip(keys, combo))
            trial = KiviConfig(**{**base.__dict__, **settings})
            run = run_evaluation(
                db_path=f"{tmp}/trial.db",
                # Into the temp dir, not the shared corpus path. Every trial
                # regenerates the corpus at `records` size, so pointing this at
                # data/seed_corpus.jsonl meant `calibrate --records 200`
                # silently replaced the committed 500-record corpus with a
                # 200-record one. Nothing here needs to outlive the sweep.
                corpus_path=f"{tmp}/trial_corpus.jsonl",
                out_path=f"{tmp}/trial.json",
                records=records,
                fresh=True,
                config=trial,
            )
            summary = run["summary"]
            memories = summary["database"]["counts"]["memories"]
            transcripts = max(1, summary["database"]["counts"]["transcripts"])
            results.append({
                "settings": settings,
                "passed": summary["passed"],
                "total": summary["total"],
                "answer_checks": summary["answer_checks"]["passed"],
                "controller_checks": summary["controller_checks"]["passed"],
                "memories": memories,
                "selectivity": round(1 - memories / transcripts, 4),
                "admission_checks": summary["admission_checks"]["passed"],
                # All three families. The admission cases were omitted here,
                # so a trial that lost three checks reported one failing id -
                # and the admission set is precisely what was added to stop the
                # surface coming out flat, i.e. the part worth seeing fail.
                "failing": [case["id"] for case in run["cases"] if not case["passed"]]
                           + [case["id"] for case in run["controller_cases"] if not case["passed"]]
                           + [case["id"] for case in run["admission_cases"] if not case["passed"]],
            })

    results.sort(key=lambda item: (item["passed"], item["selectivity"]), reverse=True)
    best = results[0]

    # Sensitivity: how much does each threshold matter on its own?
    sensitivity = {}
    for index, key in enumerate(keys):
        by_value = {}
        for item in results:
            by_value.setdefault(item["settings"][key], []).append(item["passed"])
        sensitivity[key] = {
            str(value): round(sum(scores) / len(scores), 3) for value, scores in sorted(by_value.items())
        }

    scores = {item["passed"] for item in results}
    memory_counts = [item["memories"] for item in results]
    if len(scores) == 1:
        verdict = (
            "FLAT: every setting in the grid scored identically "
            f"({results[0]['passed']}/{results[0]['total']}), so this evaluation does not "
            "discriminate between them and the shipped thresholds are currently unfalsifiable. "
            f"Admitted-memory count did vary ({min(memory_counts)}-{max(memory_counts)}), so the "
            "thresholds are doing something - the questions just never ask about it. Before "
            "trusting a calibrated value, add cases that SHOULD be rejected and assert they were: "
            "a borderline candidate the controller must not store, a hedged claim that must stay "
            "tentative, a weak conflict that must not supersede."
        )
    else:
        # A dimension whose every value scores identically is unfalsifiable even
        # when the surface as a whole discriminates, and the FLAT verdict only
        # fires when EVERY setting ties. Say which thresholds this evaluation
        # cannot actually defend.
        flat_axes = [key for key, values in sensitivity.items() if len(set(values.values())) == 1]
        untested = [key for key in keys if getattr(KiviConfig(), key) not in CALIBRATION_GRID[key]]
        verdict = (
            f"Scores ranged {min(scores)}-{max(scores)} across {len(results)} settings; the "
            "surface discriminates and `best` is a defensible choice."
        )
        if flat_axes:
            verdict += (
                f" NOTE: {', '.join(flat_axes)} scored identically at every value tested, so this "
                "evaluation does not falsify it - treat the shipped value there as unvalidated."
            )
        if untested:
            verdict += (
                f" NOTE: the shipped value of {', '.join(untested)} is not a point in the grid, so "
                "it was never evaluated; add it to CALIBRATION_GRID before citing this as support."
            )

    payload = {
        "verdict": verdict,
        "best": best,
        "shipped_defaults": {key: getattr(KiviConfig(), key) for key in keys},
        "score_range": sorted(scores),
        "memory_count_range": [min(memory_counts), max(memory_counts)],
        "mean_score_by_threshold_value": sensitivity,
        "grid_size": len(results),
        "records_per_trial": records,
        "all_results": results,
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
