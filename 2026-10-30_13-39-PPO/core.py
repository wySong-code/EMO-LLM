"""CPU-only data, sampling and reward logic for the second GRPO experiment."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import struct
import math

EMOTIONS = {"positive", "negative", "neutral"}
WEIGHTS = {"fused": .35, "vision": .10, "second": .10, "decision": .45}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def validate_dataset(root: Path) -> tuple[list[dict], list[dict], dict]:
    train, val = read_jsonl(root / "train.jsonl"), read_jsonl(root / "val.jsonl")
    if (len(train), len(val)) != (3352, 666):
        raise ValueError(f"Expected cleaned 3352/666 split, found {len(train)}/{len(val)}")
    train_ids, val_ids = {r["sample_id"] for r in train}, {r["sample_id"] for r in val}
    if len(train_ids) != len(train) or len(val_ids) != len(val) or train_ids & val_ids:
        raise ValueError("Repeated or overlapping sample_id")
    train_videos, val_videos = {r["video"] for r in train}, {r["video"] for r in val}
    if train_videos & val_videos:
        raise ValueError("Train/validation video path overlap")
    for row in train + val:
        if row["protocol"] not in {"VT", "VA"} or row["sample_type"] not in {"Conflict", "Aligned"}:
            raise ValueError(f"Bad protocol/type in {row['sample_id']}")
        m1, m2, m12 = (row["views"][key]["label"] for key in ("M1", "M2", "M12"))
        if not all(x in EMOTIONS for x in (m1, m2, m12)):
            raise ValueError(f"Bad emotion label in {row['sample_id']}")
        if (m1 != m2) != (row["sample_type"] == "Conflict"):
            raise ValueError(f"Modality labels disagree with sample_type: {row['sample_id']}")
        if row["protocol"] == "VT" and not str(row["text_content"]).strip():
            raise ValueError(f"Empty VT text: {row['sample_id']}")
        if not (root / row["video"]).is_file():
            raise FileNotFoundError(root / row["video"])
    counts = Counter((r["sample_type"], r["protocol"], bool(r["source_is_generated"]))
                     for r in train)
    return train, val, {"train": len(train), "val": len(val),
                        "train_buckets": {"|".join(map(str, key)): n for key, n in sorted(counts.items())}}


def validate_assets(base: Path, adapter: Path) -> dict:
    """Check local model/adapter identity and shard presence without torch."""
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if base_config.get("model_type") != "qwen2_5_omni":
        raise ValueError("Base model is not Qwen2.5-Omni")
    if (adapter_config.get("base_model_name_or_path") != "Qwen/Qwen2.5-Omni-3B"
            or adapter_config.get("peft_type") != "LORA" or adapter_config.get("r") != 16):
        raise ValueError("Adapter identity/config differs from published Combined SFT")
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (base / shard).is_file() for shard in shards):
        raise FileNotFoundError("One or more base model shards are missing")
    weights = adapter / "adapter_model.safetensors"
    with weights.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size <= 0 or header_size > 100_000_000:
            raise ValueError("Invalid adapter safetensors header")
        header = json.loads(handle.read(header_size))
    tensors = [part for key, part in header.items() if key != "__metadata__"]
    count = sum(math.prod(part["shape"]) for part in tensors)
    if count != 54_192_128 or {part["dtype"] for part in tensors} != {"F32"}:
        raise ValueError(f"Adapter parameters/dtype mismatch: {count}")
    for filename in ("tokenizer.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(adapter / filename)
    return {"model_type": base_config["model_type"], "base_shards": len(shards),
            "adapter_tensors": len(tensors), "adapter_parameters": count,
            "adapter_dtype": "F32"}


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, part in pairs:
        if key in value:
            raise ValueError(f"Repeated JSON key {key}")
        value[key] = part
    return value


def parse_answer(text: str, protocol: str) -> dict | None:
    other = {"VT": "text_emotion", "VA": "audio_emotion"}.get(protocol)
    if other is None:
        raise ValueError(f"Unknown protocol {protocol}")
    try:
        value = json.loads(text.strip(), object_pairs_hook=_unique_pairs)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"vision_emotion", other, "is_conflict", "pred_emotion"}:
        return None
    if any(not isinstance(value[key], str) or value[key] not in EMOTIONS
           for key in ("vision_emotion", other, "pred_emotion")):
        return None
    if not isinstance(value["is_conflict"], str) or value["is_conflict"] not in {"yes", "no"}:
        return None
    return value


@dataclass(frozen=True)
class Reward:
    total: float
    fused: int
    vision: int
    second: int
    decision: int
    valid: bool
    consistent: bool

    @property
    def perfect(self) -> bool:
        return (self.valid and self.consistent and
                self.fused == self.vision == self.second == self.decision == 1)


def score_answer(text: str, row: dict) -> Reward:
    answer = parse_answer(text, row["protocol"])
    if answer is None:
        return Reward(-.20, 0, 0, 0, 0, False, False)
    other = "text_emotion" if row["protocol"] == "VT" else "audio_emotion"
    fused = int(answer["pred_emotion"] == row["views"]["M12"]["label"])
    vision = int(answer["vision_emotion"] == row["views"]["M1"]["label"])
    second = int(answer[other] == row["views"]["M2"]["label"])
    decision = int((answer["is_conflict"] == "yes") == (row["sample_type"] == "Conflict"))
    consistent = ((answer["vision_emotion"] != answer[other]) == (answer["is_conflict"] == "yes"))
    total = (.35 * fused + .10 * vision + .10 * second + .45 * decision
             - (0 if consistent else .05))
    return Reward(round(total, 6), fused, vision, second, decision, True, consistent)


def token_rewards(total: float, length: int) -> list[float]:
    """Terminal-only reward: every response token scores 0 except the last one.

    This is the standard RLHF/PPO convention -- the answer is scored as a whole,
    and the signal is propagated backwards through GAE rather than repeated on
    every token (which would multiply the effective reward by the sequence length).
    """
    if length < 1:
        raise ValueError("A response needs at least one token")
    values = [0.0] * length
    values[-1] = float(total)
    return values


def gae(rewards: list[float], values: list[float], *, gamma: float = 1.0,
        lam: float = .95) -> tuple[list[float], list[float]]:
    """Generalised advantage estimation for a single terminal-reward response.

    Returns (advantages, returns); ``returns[t] = advantages[t] + values[t]`` is the
    regression target of the value head. The sequence is treated as one episode that
    ends at the final response token, so V(s_T+1) = 0.
    """
    if not values or len(rewards) != len(values):
        raise ValueError("GAE needs one reward per value")
    advantages = [0.0] * len(values)
    returns = [0.0] * len(values)
    running = 0.0
    for index in reversed(range(len(values))):
        following = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * following - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
        returns[index] = running + values[index]
    return advantages, returns


def batch_statistics(advantages: list[float]) -> dict:
    """Descriptive stats used for logging only -- advantages are never rescaled."""
    if not advantages:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(advantages) / len(advantages)
    std = (sum((value - mean) ** 2 for value in advantages) / len(advantages)) ** .5
    return {"n": len(advantages), "mean": round(mean, 6), "std": round(std, 6),
            "min": round(min(advantages), 6), "max": round(max(advantages), 6)}


class StratifiedSampler:
    """Prompt mix of 65/5/30 (original conflict / generated conflict / aligned).

    PPO replaces GRPO's in-group baseline with a learned critic, so a conflict-heavy
    pool is no longer needed to manufacture reward variance. The aligned share is
    raised from 20% to 30% on purpose: both GRPO rounds drifted towards over-reporting
    conflict (specificity 0.9071 -> 0.8986), and aligned prompts are the only ones that
    push ``is_conflict`` back down.
    """

    SHARES = ("original_conflict", "generated_conflict", "aligned")
    PER_HUNDRED = {"original_conflict": 65, "generated_conflict": 5, "aligned": 30}

    def __init__(self, train: list[dict], seed: int = 42):
        self.rng = random.Random(seed)
        self.buckets = {
            "original_conflict": [r for r in train if r["sample_type"] == "Conflict" and not r["source_is_generated"]],
            "generated_conflict": [r for r in train if r["sample_type"] == "Conflict" and r["source_is_generated"]],
            "aligned": [r for r in train if r["sample_type"] == "Aligned"],
        }
        if {key: len(rows) for key, rows in self.buckets.items()} != {
                "original_conflict": 143, "generated_conflict": 268, "aligned": 2941}:
            raise ValueError("Unexpected training buckets")
        self.queues = {key: [] for key in self.buckets}
        self.plan: list[str] = []
        self.hard_aligned: dict[str, dict] = {}
        self.original_seen: set[str] = set()
        self.emitted = 0

    def _cycle(self, key: str) -> dict:
        if not self.queues[key]:
            self.queues[key] = self.buckets[key].copy()
            self.rng.shuffle(self.queues[key])
        return self.queues[key].pop()

    def _draw(self, key: str) -> dict:
        if key == "aligned" and self.hard_aligned and self.rng.random() < .25:
            return self.rng.choice(list(self.hard_aligned.values()))
        return self._cycle(key)

    def next(self) -> tuple[str, dict]:
        if not self.plan:
            self.plan = [key for key in self.SHARES for _ in range(self.PER_HUNDRED[key])]
            self.rng.shuffle(self.plan)
        key = self.plan.pop()
        row = self._draw(key)
        if key == "original_conflict":
            self.original_seen.add(row["sample_id"])
        self.emitted += 1
        return key, row

    def batch(self, size: int) -> list[tuple[str, dict]]:
        """A PPO update batch: ``size`` distinct prompts drawn from the mix plan."""
        if size < 1:
            raise ValueError("Batch size must be positive")
        chosen: list[tuple[str, dict]] = []
        seen: set[str] = set()
        guard = 0
        while len(chosen) < size:
            guard += 1
            if guard > size * 40:
                raise RuntimeError("Sampler cannot find enough distinct prompts")
            key, row = self.next()
            if row["sample_id"] in seen:
                continue
            seen.add(row["sample_id"])
            chosen.append((key, row))
        return chosen

    def observe(self, key: str, row: dict, reward: Reward) -> None:
        if key == "aligned" and reward.valid and reward.decision == 0:
            self.hard_aligned[row["sample_id"]] = row

