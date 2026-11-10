# -*- coding: utf-8 -*-
"""从各实验的 rollouts.jsonl 逐样本候选记录，离线重算"每一步的奖励侧训练曲线"。

产物：_audit/curves_reward_per_step.csv
  每个训练点一行：候选数、平均奖励、组内奖励标准差、零梯度组占比、decision 正确率等。

说明：
  * 这是 GRPO / DAPO 系目前唯一能画出的"奖励侧"训练曲线——原训练脚本没落盘 loss/KL/entropy。
  * 各实验的奖励函数权重不同（DAPO 两臂改过），所以 mean_reward 只在同一实验内部可比，
    跨实验不做绝对值比较。
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"

# (实验, 臂, 单位, 总数, 相对路径, 列名, 是否有 accepted 标记)
SOURCES = [
    ("3B-GRPO-r1", "grpo3b_r1", "step", 300,
     "2026-10-29_14-21-GRPO/grpo_runs/pilot_01/rollouts.jsonl", "step", False),
    ("3B-GRPO-r2", "grpo3b_r2", "attempt", 600,
     "2026-10-29_23-22-GRPO/runs/pilot_01/rollouts.jsonl", "attempt", False),
    ("7B-GRPO", "grpo7b", "step", 300,
     "2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01/rollouts.jsonl", "step", False),
    ("7B-DAPO", "dapo7b", "step", 300,
     "2026-11-01-DAPO/runs/dapo7b/rollouts.jsonl", "step", True),
    ("7B-DAPO", "dapo7b_legacy", "step", 300,
     "2026-11-01-DAPO/runs/dapo7b_legacy/rollouts.jsonl", "step", True),
]

COLS = ["experiment", "arm", "unit", "unit_value", "progress_pct",
        "groups", "accepted_groups", "candidates",
        "mean_reward", "std_reward", "within_group_std", "zero_spread_group_rate",
        "decision_acc", "conflict_group_share", "aligned_group_share", "top_reward_share",
        "grad_groups", "grad_conflict_share"]


def stat(vals):
    n = len(vals)
    if not n:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5


def parse_rollouts(path, unit_key, has_accepted):
    """返回 {unit_value: 聚合字典}。每行 = 一组 × N 个候选。"""
    buckets = defaultdict(lambda: {"groups": 0, "accepted": 0, "cand": 0,
                                   "rew": [], "gstd": [], "spread0": 0,
                                   "dec": [], "conflict": 0, "aligned": 0,
                                   "top_hits": 0, "top_total": 0,
                                   "grad": 0, "grad_conflict": 0})
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s:
            continue
        o = json.loads(s)
        u = o[unit_key]
        b = buckets[u]
        b["groups"] += 1
        if has_accepted and o.get("accepted"):
            b["accepted"] += 1
        rewards = o.get("rewards") or []
        if not rewards:
            continue
        b["cand"] += len(rewards)
        b["rew"].extend(rewards)
        m, sd = stat(rewards)
        b["gstd"].append(sd)
        has_grad = max(rewards) - min(rewards) > 1e-9
        if not has_grad:
            b["spread0"] += 1
        dc = o.get("decision_correct") or []
        if dc:
            b["dec"].extend(dc)
        # 组标签（gold_conflict 优先，其次 bucket 名）
        gc = str(o.get("gold_conflict", "")).lower()
        bk = str(o.get("bucket", "")).lower()
        is_conf = ("conflict" in gc) or ("conflict" in bk)
        is_alig = ("aligned" in gc) or ("aligned" in bk)
        if is_conf:
            b["conflict"] += 1
        elif is_alig:
            b["aligned"] += 1
        if has_grad:
            b["grad"] += 1
            if is_conf:
                b["grad_conflict"] += 1
        # 组内最高奖励被多少候选命中（奖励饱和的另一个视角）
        top = max(rewards)
        b["top_hits"] += sum(1 for r in rewards if abs(r - top) < 1e-9)
        b["top_total"] += len(rewards)
    return buckets


def parse_ppo(path):
    """PPO 的 rollouts 是每个 attempt 一行，samples 内是 4 个**不同样本**（非同一 prompt 的候选）。

    因此组内极差/零梯度组这类"候选组"指标对 PPO 不适用，一律置空；
    只保留该 attempt 的平均奖励与 decision 正确率。优化侧标量见 build_optimizer_curves.py。
    """
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s:
            continue
        o = json.loads(s)
        samples = o.get("samples") or []
        if not samples:
            continue
        rew = [x.get("reward") for x in samples if x.get("reward") is not None]
        dec = [x.get("decision") for x in samples if x.get("decision") is not None]
        m, sd = stat(rew)
        out[o["attempt"]] = {
            "groups": 1, "accepted": 1, "cand": len(samples),
            "mean_reward": m, "std_reward": sd,
            "within_group_std": None,
            "zero_spread_group_rate": None,
            "decision_acc": (sum(dec) / len(dec)) if dec else None,
            "conflict_group_share": None, "aligned_group_share": None,
            "top_reward_share": None,
        }
    return out


def main():
    rows = []
    for exp, arm, unit, total, rel, key, has_acc in SOURCES:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"缺少 {rel}")
        buckets = parse_rollouts(p, key, has_acc)
        for u in sorted(buckets):
            b = buckets[u]
            mean_r, std_r = stat(b["rew"])
            rows.append({
                "experiment": exp, "arm": arm, "unit": unit, "unit_value": u,
                "progress_pct": round(100.0 * u / total, 2),
                "groups": b["groups"],
                "accepted_groups": b["accepted"],
                "candidates": b["cand"],
                "mean_reward": round(mean_r, 6),
                "std_reward": round(std_r, 6),
                "within_group_std": round(stat(b["gstd"])[0], 6),
                "zero_spread_group_rate": round(b["spread0"] / b["groups"], 6) if b["groups"] else None,
                "decision_acc": round(stat(b["dec"])[0], 6) if b["dec"] else None,
                "conflict_group_share": round(b["conflict"] / b["groups"], 6) if b["groups"] else None,
                "aligned_group_share": round(b["aligned"] / b["groups"], 6) if b["groups"] else None,
                "top_reward_share": round(b["top_hits"] / b["top_total"], 6) if b["top_total"] else None,
                "grad_groups": b["grad"],
                "grad_conflict_share": round(b["grad_conflict"] / b["grad"], 6) if b["grad"] else None,
            })

    ppo = parse_ppo(ROOT / "2026-10-30_13-39-PPO/runs/pilot_01/rollouts.jsonl")
    for u in sorted(ppo):
        d = ppo[u]
        rows.append({
            "experiment": "3B-PPO", "arm": "ppo3b", "unit": "attempt", "unit_value": u,
            "progress_pct": round(100.0 * u / 300, 2),
            "groups": d["groups"], "accepted_groups": d["accepted"], "candidates": d["cand"],
            "mean_reward": round(d["mean_reward"], 6),
            "std_reward": round(d["std_reward"], 6),
            "within_group_std": None,
            "zero_spread_group_rate": None,
            "decision_acc": round(d["decision_acc"], 6) if d["decision_acc"] is not None else None,
            "conflict_group_share": None, "aligned_group_share": None, "top_reward_share": None,
            "grad_groups": None, "grad_conflict_share": None,
        })

    out = AUDIT / "curves_reward_per_step.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[reward] {len(rows)} 行 -> {out.name}")

    by = defaultdict(list)
    for r in rows:
        by[(r["experiment"], r["arm"])].append(r)
    print(f"\n{'实验':<12}{'臂':<16}{'点数':>5}{'均奖励(全程均值)':>18}{'零梯度组(均值)':>16}{'decision(均值)':>16}")
    for (e, a), rs in by.items():
        rs.sort(key=lambda r: r["unit_value"])
        mr = [r["mean_reward"] for r in rs if r["mean_reward"] is not None]
        zs = [r["zero_spread_group_rate"] for r in rs if r["zero_spread_group_rate"] is not None]
        da = [r["decision_acc"] for r in rs if r["decision_acc"] is not None]
        print(f"{e:<12}{a:<16}{len(rs):>5}"
              f"{sum(mr)/len(mr):>18.4f}"
              f"{(str(round(sum(zs)/len(zs),3)) if zs else 'n/a'):>16}"
              f"{(str(round(sum(da)/len(da),3)) if da else 'n/a'):>16}")


if __name__ == "__main__":
    main()
