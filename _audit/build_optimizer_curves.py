# -*- coding: utf-8 -*-
"""汇总"优化器侧"训练曲线（loss / KL / 裁剪比例 / 梯度范数）。

产物：_audit/curves_optimizer_side.csv（长表）
  只有两个实验有真数据：
    7B-SFT  —— train_metrics.jsonl，1638 步，含 loss / gradient_norm / lr / 三模态 task loss
    3B-PPO  —— rollouts.jsonl 每行的 update 字段，300 attempt，含 policy_loss / value_loss /
               kl / clip_fraction / policy_grad_norms / value_grad_norms
  3B GRPO r1/r2、7B GRPO、7B DAPO 两臂在这一侧**没有任何数据**——训练脚本算了但没落盘，
  事后无法补；本表对它们保持零行，不做估算或插值。
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
COLS = ["experiment", "arm", "unit", "unit_value", "progress_pct", "metric", "value"]


def add(rows, exp, arm, unit, u, total, **kw):
    for k, v in kw.items():
        if v is None:
            continue
        try:
            rows.append({"experiment": exp, "arm": arm, "unit": unit, "unit_value": u,
                         "progress_pct": round(100.0 * u / total, 2),
                         "metric": k, "value": round(float(v), 8)})
        except (TypeError, ValueError):
            continue


def sft():
    rows = []
    p = ROOT / "2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/train/train_metrics.jsonl"
    for line in open(p, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        d = json.loads(s)
        add(rows, "7B-SFT", "sft7b", "step", d["step"], 1638,
            loss=d.get("loss"),
            gradient_norm=d.get("gradient_norm"),
            learning_rate=d.get("learning_rate"),
            task_loss_full=d.get("task_loss/full"),
            task_loss_vision=d.get("task_loss/vision_only"),
            task_loss_text=d.get("task_loss/text_only"),
            task_loss_audio=d.get("task_loss/audio_only"),
            grad_norm_language=d.get("gradient_norm/language"),
            grad_norm_vision=d.get("gradient_norm/vision"),
            grad_norm_audio=d.get("gradient_norm/audio"),
            epoch=d.get("epoch"))
    return rows


def ppo():
    rows = []
    p = ROOT / "2026-10-30_13-39-PPO/runs/pilot_01/rollouts.jsonl"
    for line in open(p, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        o = json.loads(s)
        up = o.get("update") or {}
        if not up:
            continue
        gp = up.get("policy_grad_norms") or []
        gv = up.get("value_grad_norms") or []
        add(rows, "3B-PPO", "ppo3b", "attempt", o["attempt"], 300,
            policy_loss=up.get("policy_loss"),
            value_loss=up.get("value_loss"),
            kl=up.get("kl"),
            clip_fraction=up.get("clip_fraction"),
            tokens=up.get("tokens"),
            policy_grad_norm_sum=(sum(gp) if gp else None),
            value_grad_norm_sum=(sum(gv) if gv else None))
    return rows


if __name__ == "__main__":
    rows = sft() + ppo()
    out = AUDIT / "curves_optimizer_side.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[optimizer] {len(rows)} 条 -> {out.name}")

    from collections import defaultdict
    idx = defaultdict(set)
    for r in rows:
        idx[(r["experiment"], r["arm"], r["metric"])].add(r["unit_value"])
    print("\n优化侧序列覆盖：")
    for (e, a, m), v in sorted(idx.items()):
        print(f"  {e:<8}{a:<8}{m:<24}{len(v):>5} 点")

    have = {r["experiment"] for r in rows}
    all_exp = ["3B-GRPO-r1", "3B-GRPO-r2", "3B-PPO", "7B-SFT", "7B-GRPO", "7B-DAPO"]
    print("\n无优化侧数据的实验（脚本未落盘，事后不可补）：")
    for e in all_exp:
        if e not in have:
            print("  -", e)
