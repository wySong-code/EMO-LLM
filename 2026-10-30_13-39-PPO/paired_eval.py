"""Paired, conflict-stratified bootstrap of two saved validation prediction files.

This is a diagnostic interval. Selecting the best checkpoint on the same 666
labels still biases its apparent gain; use a separate hidden test for a claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def macro_f1(records: list[dict], indices: list[int], labels: tuple[str, ...]) -> float:
    values = []
    for label in labels:
        tp = fp = fn = 0
        for index in indices:
            row = records[index]
            gold = row["gold"]["pred_emotion"]
            predicted = row["prediction"].get("pred_emotion", "invalid")
            tp += gold == label and predicted == label
            fp += gold != label and predicted == label
            fn += gold == label and predicted != label
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.)
    return sum(values) / len(values)


def score(records: list[dict], indices: list[int], labels: tuple[str, ...]) -> float:
    positive = [index for index in indices if records[index]["gold"]["is_conflict"] == "yes"]
    negative = [index for index in indices if records[index]["gold"]["is_conflict"] == "no"]
    recall = sum(records[index]["prediction"].get("is_conflict") == "yes" for index in positive) / len(positive)
    specificity = sum(records[index]["prediction"].get("is_conflict") == "no" for index in negative) / len(negative)
    return .5 * macro_f1(records, indices, labels) + .3 * recall + .2 * specificity


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def compare(before: list[dict], after: list[dict], *, iterations: int = 2000, seed: int = 42) -> dict:
    if len(before) != len(after) or len(before) != 666:
        raise ValueError("Need the same 666 validation rows in both files")
    if [r["sample_id"] for r in before] != [r["sample_id"] for r in after]:
        raise ValueError("Validation sample order or IDs differ")
    if any(a["gold"] != b["gold"] for a, b in zip(before, after)):
        raise ValueError("Gold labels differ between prediction files")
    labels = tuple(sorted({r["gold"]["pred_emotion"] for r in before}))
    positive = [i for i, r in enumerate(before) if r["gold"]["is_conflict"] == "yes"]
    negative = [i for i, r in enumerate(before) if r["gold"]["is_conflict"] == "no"]
    if not positive or not negative:
        raise ValueError("Both conflict classes are required")
    all_indices = list(range(len(before)))
    baseline = score(before, all_indices, labels)
    candidate = score(after, all_indices, labels)
    rng = random.Random(seed)
    differences = []
    for _ in range(iterations):
        indices = rng.choices(positive, k=len(positive)) + rng.choices(negative, k=len(negative))
        differences.append(score(after, indices, labels) - score(before, indices, labels))
    differences.sort()
    return {"n": len(before), "conflict_n": len(positive), "aligned_n": len(negative),
            "baseline": baseline, "candidate": candidate, "delta": candidate - baseline,
            "bootstrap_iterations": iterations, "delta_ci95": [percentile(differences, .025),
                                                       percentile(differences, .975)],
            "bootstrap_positive_fraction": sum(x > 0 for x in differences) / iterations,
            "selection_bias_note": "Checkpoint chosen on these labels; interval is exploratory"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("Use at least 100 bootstrap iterations")
    print(json.dumps(compare(read(args.baseline), read(args.candidate),
                             iterations=args.iterations, seed=args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
