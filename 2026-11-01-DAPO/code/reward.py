"""Strict, testable reward for the Combined SFT four-field contract.

The scorer uses training labels only. Validation/test labels never enter rollout
sampling. is_uncertain and clarifying_question are deliberately absent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

EMOTIONS = {"positive", "negative", "neutral"}
REWARD_PROFILES = {
    # Exact reward used by the earlier 7B GRPO run. Keep it for a clean ablation.
    "legacy": {"fused": 0.35, "vision": 0.10, "second": 0.10, "decision": 0.45},
    # The submitted CSV scores fused emotion and conflict, not branch labels.
    # Keep branch supervision, but shift 0.15 of its weight to the decision.
    "score_focused": {"fused": 0.35, "vision": 0.025, "second": 0.025, "decision": 0.60},
}


def reward_weights(profile: str) -> dict[str, float]:
    if profile not in REWARD_PROFILES:
        raise ValueError(f"unknown reward profile: {profile}")
    weights = REWARD_PROFILES[profile]
    if any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"invalid reward weights for {profile}")
    return weights.copy()


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


def score(text: str, row: dict, *, weights: dict[str, float] | None = None) -> Reward:
    if weights is None:
        weights = REWARD_PROFILES["legacy"]
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
    value = (weights["fused"] * fused + weights["vision"] * vision
             + weights["second"] * other + weights["decision"] * decision
             - (0.05 if not consistent else 0.0))
    return Reward(value, fused, vision, other, decision, True, consistent)


def advantages(rewards: list[Reward], *, gold_conflict: bool, priority: float = 1.5,
               decision_bonus: float = 0.0) -> tuple[list[float], str]:
    """Center within one prompt; class priority applies after normalization.

    Identical rewards carry no rank information. Cross-prompt comparison is
    intentionally excluded, including when all candidates are wrong.
    """
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two candidates for the same prompt")
    if decision_bonus < 0:
        raise ValueError("decision_bonus must be nonnegative")
    values = [r.total for r in rewards]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if variance < 1e-12:
        perfect = all(r.valid and r.fused == r.vision == r.second == r.decision == 1
                      and r.consistent for r in rewards)
        return [0.0] * len(rewards), "all_correct" if perfect else "all_same_not_perfect"
    scale = priority if gold_conflict else 1.0
    std = variance ** 0.5
    base = [(v - mean) / (std + 1e-6) for v in values]
    # Group normalization cancels a reward coefficient when only one component
    # varies. Add a separate decision advantage only when answers disagree on
    # the decision; this gives the score-focused profile a real gradient effect.
    decisions = [r.decision for r in rewards]
    decision_mean = sum(decisions) / len(decisions)
    decision_var = sum((v - decision_mean) ** 2 for v in decisions) / len(decisions)
    if decision_bonus and decision_var >= 1e-12:
        decision_std = decision_var ** 0.5
        base = [a + decision_bonus * (d - decision_mean) / (decision_std + 1e-6)
                for a, d in zip(base, decisions)]
    return [scale * a for a in base], "active"
