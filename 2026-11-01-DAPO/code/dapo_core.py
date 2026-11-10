"""CPU-only sampling and bookkeeping for budgeted DAPO adaptation.

No validation/test labels are accepted here. The caller must pass TRAIN rows only.
"""
from __future__ import annotations

from collections import Counter
import random


def group_kind(rewards) -> str:
    """Distinguish decision learning from auxiliary-only reward variation."""
    if len(rewards) < 2:
        raise ValueError("a group needs at least two responses")
    values = [round(r.total, 8) for r in rewards]
    if len(set(values)) == 1:
        return "all_correct" if all(
            r.valid and r.fused == r.vision == r.second == r.decision == 1
            and r.consistent for r in rewards) else "all_same_not_perfect"
    if len({r.decision for r in rewards}) > 1:
        return "decision_active"
    return "auxiliary_active"


def screen_errors(records: list[dict], train_ids: set[str]) -> tuple[set[str], dict]:
    """Find hard TRAIN prompts from a deterministic SFT pass, with ID validation."""
    if {r["sample_id"] for r in records} != train_ids or len(records) != len(train_ids):
        raise ValueError("screen must contain exactly the training sample IDs")
    hard = set()
    errors = Counter()
    for record in records:
        gold, pred = record["gold"], record["prediction"]
        bad = record["parse_error"] is not None or pred != gold
        if bad:
            hard.add(record["sample_id"])
        if pred.get("is_conflict") != gold["is_conflict"]:
            errors["decision"] += 1
        if pred.get("pred_emotion") != gold["pred_emotion"]:
            errors["emotion"] += 1
        if record["parse_error"] is not None:
            errors["format"] += 1
    return hard, {"n": len(records), "hard": len(hard), "errors": dict(errors)}


class DifficultyPool:
    """Select by gold class and source; learn difficulty only from training rollouts."""

    def __init__(self, train: list[dict], hard_ids: set[str], *, seed: int,
                 original_conflict_prob: float = 0.8, hard_prob: float = 0.75):
        self.rng = random.Random(seed)
        self.hard_ids = set(hard_ids)
        self.original_conflict_prob = original_conflict_prob
        self.hard_prob = hard_prob
        self.groups = {
            "original_conflict": [r for r in train if r["sample_type"] == "Conflict" and not r["source_is_generated"]],
            "generated_conflict": [r for r in train if r["sample_type"] == "Conflict" and r["source_is_generated"]],
            "aligned": [r for r in train if r["sample_type"] == "Aligned"],
        }
        if any(not group for group in self.groups.values()):
            raise ValueError("original conflict, generated conflict and aligned pools are required")
        if not 0 <= original_conflict_prob <= 1 or not 0 <= hard_prob <= 1:
            raise ValueError("invalid sampling probability")
        ids = {r["sample_id"] for r in train}
        if not self.hard_ids <= ids:
            raise ValueError("hard pool contains non-training IDs")
        self.seen = Counter()
        self.stats = Counter()

    def choose(self, label: str, *, exclude: set[str] | None = None) -> dict:
        exclude = exclude or set()
        if label == "Conflict":
            kind = "original_conflict" if self.rng.random() < self.original_conflict_prob else "generated_conflict"
        elif label == "Aligned":
            kind = "aligned"
        else:
            raise ValueError(label)
        rows = [r for r in self.groups[kind] if r["sample_id"] not in exclude]
        if not rows:
            raise ValueError(f"no unused training rows in {kind}")
        hard = [r for r in rows if r["sample_id"] in self.hard_ids]
        # Avoid collapsing a bucket onto one or two discovered hard prompts.
        hard_choice_prob = self.hard_prob * min(1.0, len(hard) / 16.0)
        if hard and self.rng.random() < hard_choice_prob:
            rows = hard
            self.stats["hard_draws"] += 1
        # Prefer less-repeated samples within the selected stratum.
        floor = min(self.seen[r["sample_id"]] for r in rows)
        fresh = [r for r in rows if self.seen[r["sample_id"]] <= floor + 1]
        row = self.rng.choice(fresh)
        self.seen[row["sample_id"]] += 1
        self.stats[kind] += 1
        return row

    def observe(self, row: dict, kind: str) -> None:
        self.stats[kind] += 1
        if kind in {"decision_active", "auxiliary_active", "all_same_not_perfect"}:
            self.hard_ids.add(row["sample_id"])

    def summary(self) -> dict:
        return {"hard_pool": len(self.hard_ids), "unique_seen": len(self.seen),
                "draws": dict(self.stats)}
