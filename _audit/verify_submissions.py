"""Rebuild every submission.csv from its predictions_raw.jsonl and diff byte by byte.

The assembly half of make_submission.py (lines 209-234) has no GPU dependency, so it
can be replayed on CPU. If the rebuilt CSV equals the shipped CSV for all arms, the
submission files are provably the output of the run they claim to come from.

Also copies a clearly named set to _audit/submission_export/ for hand-off.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "submission_export"

QUESTION = ("画面呈现的情绪偏向{vision}，但{modality}传递的情绪却是{second}。"
            "两个模态的信号互相矛盾，我应该以哪一侧作为最终判断？")
MODALITY_NAME = {"VT": "文字", "VA": "语音"}

# (submission dir, display name, kaggle board score or None if never submitted)
ARMS = [
    ("2026-10-29_14-21-GRPO/submission_step50", "3B_GRPO-r1_step50", 0.93638),
    ("2026-10-29_23-22-GRPO/submission_attempt0300", "3B_GRPO-r2_step300", None),
    ("2026-10-30_13-39-PPO/submission_attempt0100", "3B_PPO_step100", None),
    ("2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/sft7b", "7B_SFT", 0.93243),
    ("2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/grpo7b_step50", "7B_GRPO_step50", 0.93444),
    ("2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/grpo7b_step150", "7B_GRPO_step150", 0.93577),
]


def build_question(prediction: dict, protocol: str) -> str:
    second_key = "text_emotion" if protocol == "VT" else "audio_emotion"
    return QUESTION.format(vision=prediction.get("vision_emotion", ""),
                           second=prediction.get(second_key, ""),
                           modality=MODALITY_NAME[protocol])


def rebuild(raw_path: Path, order: list[str]) -> tuple[str, dict]:
    by_id = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_id[r["sample_id"]] = r
    rows, counts = [], Counter()
    for sid in order:
        record = by_id.get(sid, {})
        prediction = record.get("prediction") or {}
        pred = prediction.get("pred_emotion")
        conflict = prediction.get("is_conflict")
        if pred not in ("positive", "negative", "neutral"):
            pred = "negative"
            counts["pred_fallback"] += 1
        if conflict not in ("yes", "no"):
            conflict = "no"
            counts["uncertain_fallback"] += 1
        question = "none" if conflict == "no" else build_question(prediction, record.get("protocol", "VT"))
        counts["emotion_" + pred] += 1
        counts["uncertain_" + conflict] += 1
        rows.append((sid, pred, conflict, question))
    text = "sample_id,pred_emotion,is_uncertain,clarifying_question\n" + "".join(
        "%s,%s,%s,%s\n" % r for r in rows)
    return text, dict(counts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'arm':22s} {'rows':>4s} {'conflict%':>9s} {'rebuild==shipped':>17s}  sha256")
    ok_all = True
    for rel, name, board in ARMS:
        d = BASE / rel
        shipped_path = d / "submission.csv"
        raw_path = d / "predictions_raw.jsonl"
        shipped = shipped_path.read_text(encoding="utf-8")
        order = [r["sample_id"] for r in csv.DictReader(shipped.splitlines())]
        rebuilt, counts = rebuild(raw_path, order)
        same = rebuilt == shipped
        ok_all &= same
        n = len(order)
        conflict_rate = 100.0 * counts.get("uncertain_yes", 0) / n
        digest = hashlib.sha256(shipped.encode()).hexdigest()[:16]
        print(f"{name:22s} {n:>4d} {conflict_rate:>8.1f}% {str(same):>17s}  {digest}")
        label = f"{name}_榜分{board:.5f}" if board else f"{name}_未提交"
        shutil.copy2(shipped_path, OUT / f"{label}.csv")
    print()
    print("全部可复现" if ok_all else "!! 有重建结果与已发货文件不一致")
    print("已复制到:", OUT)
    for p in sorted(OUT.glob("*.csv")):
        print("  ", p.name, f"{p.stat().st_size:,}B")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
