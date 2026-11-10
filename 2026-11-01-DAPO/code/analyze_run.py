"""Summarize DAPO efficiency and validation without loading a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    cycles = read_jsonl(run / "cycles.jsonl")
    groups = read_jsonl(run / "rollouts.jsonl")
    if not cycles or not groups:
        raise ValueError("run has no completed cycles or rollouts")
    buckets = Counter(g["bucket"] for g in groups)
    selected = [g for g in groups if g["accepted"]]
    by_class = Counter(g["gold_conflict"] for g in selected)
    output = {
        "cycles": len(cycles),
        "cycles_with_update": sum(c["updates"] > 0 for c in cycles),
        "optimizer_updates": sum(c["updates"] for c in cycles),
        "attempted_groups": len(groups),
        "accepted_groups": len(selected),
        "active_fraction_of_attempted": sum(g["bucket"] in {"decision_active", "auxiliary_active"}
                                            for g in groups) / len(groups),
        "accepted_fraction_of_attempted": len(selected) / len(groups),
        "decision_active_groups": buckets["decision_active"],
        "all_correct_groups": buckets["all_correct"],
        "bucket_counts": dict(buckets),
        "accepted_by_class": dict(by_class),
        "total_generated_answers": sum(len(g["rewards"]) for g in groups),
        "answers_per_cycle": sum(len(g["rewards"]) for g in groups) / len(cycles),
        "clip_upper": sum(c["clip_upper"] for c in cycles),
        "clip_lower": sum(c["clip_lower"] for c in cycles),
    }
    for name in ("sft_baseline", "step_0050", "step_0100", "step_0150",
                 "step_0200", "step_0250", "step_0300"):
        path = run / f"{name}_metrics.json"
        if path.is_file():
            m = json.loads(path.read_text(encoding="utf-8"))["all"]
            output[name] = {
                "score": m["local_score_present"],
                "emotion_f1": m["emotion_macro_f1_present"],
                "conflict_recall": m["conflict_recall"],
                "aligned_specificity": m["aligned_specificity"],
                "confusion": m["conflict_confusion"],
            }
    best = run / "best.json"
    if best.is_file():
        output["best"] = json.loads(best.read_text(encoding="utf-8"))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
