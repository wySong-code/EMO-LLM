"""Strict, testable reward for the Combined SFT four-field contract.

The scorer uses training labels only. Validation/test labels never enter rollout
sampling. is_uncertain and clarifying_question are deliberately absent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

EMOTIONS = {"positive", "negative", "neutral"}
WEIGHTS = {"fused": 0.35, "vision": 0.10, "second": 0.10, "decision": 0.45}


@dataclass(frozen=True)
class Reward:
    total: float
    fused: float
    vision: float
    second: float
    decision: float
    valid: bool
    consistent: bool


def _unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key: {key}")
        obj[key] = value
    return obj


def parse(text: str, protocol: str) -> dict | None:
    """Require one exact JSON object and the SFT schema; never salvage fragments."""
    second = {"VT": "text_emotion", "VA": "audio_emotion"}.get(protocol)
    if second is None:
        raise ValueError(f"unknown protocol: {protocol}")
    try:
        value = json.loads(text.strip(), object_pairs_hook=_unique_pairs)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"vision_emotion", second, "is_conflict", "pred_emotion"}:
        return None
    if any(not isinstance(value[key], str) or value[key] not in EMOTIONS
           for key in ("vision_emotion", second, "pred_emotion")):
        return None
    if not isinstance(value["is_conflict"], str) or value["is_conflict"] not in {"yes", "no"}:
        return None
    return value


def score(text: str, row: dict) -> Reward:
    pred = parse(text, row["protocol"])
    if pred is None:
        return Reward(-0.20, 0, 0, 0, 0, False, False)
    second = "text_emotion" if row["protocol"] == "VT" else "audio_emotion"
    gold = row["views"]
    fused = float(pred["pred_emotion"] == gold["M12"]["label"])
    vision = float(pred["vision_emotion"] == gold["M1"]["label"])
    other = float(pred[second] == gold["M2"]["label"])
    decision = float((pred["is_conflict"] == "yes") == (row["sample_type"] == "Conflict"))
    consistent = (pred["vision_emotion"] != pred[second]) == (pred["is_conflict"] == "yes")
    # Inconsistency is penalized but cannot dominate correctness against gold.
    value = (WEIGHTS["fused"] * fused + WEIGHTS["vision"] * vision
             + WEIGHTS["second"] * other + WEIGHTS["decision"] * decision
             - (0.05 if not consistent else 0.0))
    return Reward(value, fused, vision, other, decision, True, consistent)


def advantages(rewards: list[Reward], *, gold_conflict: bool, priority: float = 1.5) -> tuple[list[float], str]:
    """Center within one prompt; class priority applies after normalization.

    Identical rewards carry no rank information. Cross-prompt comparison is
    intentionally excluded, including when all candidates are wrong.
    """
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two candidates for the same prompt")
    values = [r.total for r in rewards]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if variance < 1e-12:
        perfect = all(r.valid and r.fused == r.vision == r.second == r.decision == 1
                      and r.consistent for r in rewards)
        return [0.0] * len(rewards), "all_correct" if perfect else "all_same_not_perfect"
    scale = priority if gold_conflict else 1.0
    std = variance ** 0.5
    return [scale * (v - mean) / (std + 1e-6) for v in values], "active"
