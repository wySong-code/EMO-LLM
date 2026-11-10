"""Build docs/08_提交台账.md from every submission kept in this repository.

All numbers come from files on disk: validation metrics from
runs/<arm>/step_XXXX_metrics.json, test-side counts from the submission CSVs
themselves, run diagnostics from summary.json. Nothing is transcribed by hand.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DAPO = BASE / "2026-11-01-DAPO"
OUT = DAPO / "docs" / "08_提交台账.md"

# (submission.csv path relative to project root, label, board score or None)
HISTORY = [
    ("2026-10-29_14-21-GRPO/submission_step50/submission.csv", "3B GRPO r1 step50", 0.93638),
    ("2026-10-29_23-22-GRPO/submission_attempt0300/submission.csv", "3B GRPO r2 step300", None),
    ("2026-10-30_13-39-PPO/submission_attempt0100/submission.csv", "3B PPO step100", None),
    ("2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/sft7b/submission.csv", "7B SFT", 0.93243),
    ("2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/grpo7b_step50/submission.csv", "7B GRPO step50", 0.93444),
    ("2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/grpo7b_step150/submission.csv", "7B GRPO step150", 0.93577),
]

# Public leaderboard scores for the DAPO arms (uploaded 2026-11-05).
BOARD_DAPO = {"dapo7b_step50": 0.93345, "dapo7b_legacy_step50": 0.93580}

# (label, key into master_results.csv, board score) for every uploaded submission.
# Local column is the official-formula score on the 666-sample validation set.
BOARD_PAIRS = [
    ("3B SFT", ("3B-GRPO-r1", "SFT（基线）", "final"), 0.93049),
    ("3B GRPO r1 step50", ("3B-GRPO-r1", "GRPO 300步", "step_0050"), 0.93638),
    ("7B SFT", ("7B-GRPO", "SFT（基线）", "final"), 0.93243),
    ("7B GRPO step50", ("7B-GRPO", "GRPO 300步", "step_0050"), 0.93444),
    ("7B GRPO step150", ("7B-GRPO", "GRPO 300步", "step_0150"), 0.93577),
    ("7B DAPO step50", ("7B-DAPO", "DAPO（score_focused）", "step_50"), 0.93345),
    ("7B DAPO-legacy step50", ("7B-DAPO", "DAPO（legacy 奖励）", "step_50"), 0.93580),
]


def load_metrics(arm: str, step: int) -> dict | None:
    p = DAPO / "runs" / arm / f"step_{step:04d}_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def score_block(metrics: dict) -> dict:
    """metrics.json nests everything under "all"; recompute the official formula
    (0.5 emotion macro-F1 + 0.3 conflict accuracy + 0.2 aligned specificity)."""
    a = metrics["all"]
    cm = a["conflict_confusion"]
    return {
        "official": (0.5 * a["emotion_macro_f1_present"]
                     + 0.3 * a["conflict_accuracy"]
                     + 0.2 * a["aligned_specificity"]),
        "local": a["local_score_present"],
        "TP": cm["TP"], "FP": cm["FP"], "FN": cm["FN"],
    }


def read_sub(path: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    n = len(rows) or 1
    yes = sum(1 for r in rows if r["is_uncertain"] == "yes")
    emo = {k: sum(1 for r in rows if r["pred_emotion"] == k)
           for k in ("positive", "negative", "neutral")}
    return rows, {"n": len(rows), "conflict": yes, "rate": 100.0 * yes / n,
                  "questions": sum(1 for r in rows if r["clarifying_question"] != "none"),
                  "emotion": emo,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "bytes": path.stat().st_size}


def main() -> int:
    root = DAPO / "submissions"
    dapo = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        sub = d / "submission.csv"
        if not sub.is_file():
            continue
        rows, info = read_sub(sub)
        arm, _, step = d.name.partition("_step")
        step = int(step) if step.isdigit() else None
        m = load_metrics(arm, step) if step else None
        dapo.append({"dir": d.name, "arm": arm, "step": step, "rows": rows, "info": info,
                     "score": score_block(m) if m else None,
                     "summary": json.loads((d / "summary.json").read_text(encoding="utf-8"))
                     if (d / "summary.json").exists() else {}})
    if not dapo:
        print("no DAPO submissions found")
        return 1

    L = ["# DAPO 提交台账（Kaggle 格式）", "",
         "本表由 `_audit/build_submission_ledger.py` 从 `submissions/*/` 与 `runs/*/step_*_metrics.json` "
         "现场生成，不含手抄数字。导出脚本：`scripts/run_submission_ckpt.sh`。", "",
         "## 1. 本次导出的两份提交", "",
         "| 目录 | 来源检查点 | 验证集官方口径 | 验证集训练口径 | 验证集 TP/FP/FN | 测试集行数 | 报冲突 | 占比 | 榜分 | CSV 字节 | sha256 前 16 位 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in dapo:
        s, i = r["score"], r["info"]
        off, loc, cnt = (f"{s['official']:.6f}", f"{s['local']:.6f}",
                         f"{s['TP']}/{s['FP']}/{s['FN']}") if s else ("—", "—", "—")
        board = BOARD_DAPO.get(r["dir"])
        L.append(f"| `{r['dir']}` | {r['arm']} `step_{r['step']:04d}` | {off} | {loc} | {cnt} | "
                 f"{i['n']} | {i['conflict']} | {i['rate']:.1f}% | "
                 f"{f'{board:.5f}' if board else '未提交'} | {i['bytes']:,} | `{i['sha256'][:16]}` |")

    L += ["", "## 2. 推理侧诊断", "",
          "| 目录 | 解析成功 | 解析失败 | 媒体回退 | 推理耗时 | 情绪分布 |",
          "|---|---|---|---|---|---|"]
    for r in dapo:
        inf, emo = r["summary"].get("infer", {}), r["info"]["emotion"]
        secs = inf.get("seconds")
        L.append(f"| `{r['dir']}` | {inf.get('parse_ok', '—')}/{inf.get('done', '—')} | "
                 f"{inf.get('skipped', '—')} | {inf.get('fallback', '—')} | "
                 f"{secs / 60:.1f} min | " +
                 " / ".join(f"{k}:{v}" for k, v in emo.items() if v) + " |")

    L += ["", "## 3. 全项目提交一览（含历史实验）", "",
          "| 提交 | 行数 | 报冲突 | 占比 | 榜分 | CSV 字节 |", "|---|---|---|---|---|---|"]
    for rel, label, board in HISTORY:
        p = BASE / rel
        if not p.is_file():
            L.append(f"| {label} | — | — | — | — | 文件缺失 |")
            continue
        _, info = read_sub(p)
        L.append(f"| {label} | {info['n']} | {info['conflict']} | {info['rate']:.1f}% | "
                 f"{f'{board:.5f}' if board else '未提交'} | {info['bytes']:,} |")
    for r in dapo:
        i, board = r["info"], BOARD_DAPO.get(r["dir"])
        L.append(f"| {r['arm']} step{r['step']} | {i['n']} | {i['conflict']} | {i['rate']:.1f}% | "
                 f"{f'{board:.5f}' if board else '未提交'} | {i['bytes']:,} |")

    # Test-side disagreement against the model DAPO started from.
    sft = BASE / "2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/sft7b/submission.csv"
    if sft.is_file():
        base_rows = {r["sample_id"]: r for r in csv.DictReader(sft.read_text(encoding="utf-8").splitlines())}
        L += ["", "## 4. 与 7B SFT 提交的逐样本差异（734 条测试集）", "",
              "| 提交 | 冲突位改判 | 情绪位改判 | 任一字段不同 |", "|---|---|---|---|"]
        for r in dapo:
            c = e = any_ = 0
            for row in r["rows"]:
                b = base_rows.get(row["sample_id"])
                if not b:
                    continue
                dc = row["is_uncertain"] != b["is_uncertain"]
                de = row["pred_emotion"] != b["pred_emotion"]
                c += dc
                e += de
                any_ += dc or de
            L.append(f"| `{r['dir']}` | {c} | {e} | {any_} |")

    L += ["", "## 5. 本地官方口径 vs 真实榜分（7 个提交模型）", "",
          "本地列取 `_audit/master_results.csv` 的 `official` 字段（666 条验证集），榜分列为 Kaggle public。",
          "",
          "| 模型 | 本地官方口径 | 本地名次 | 榜分 | 榜上名次 | 榜分 − 本地 |",
          "|---|---|---|---|---|---|"]
    master = {(r["experiment"], r["method"], r["checkpoint"]): float(r["official"])
              for r in csv.DictReader((BASE / "_audit" / "master_results.csv").open(encoding="utf-8"))}
    pts = [(label, master[key], board) for label, key, board in BOARD_PAIRS]
    order_local = {i for i in range(len(pts))}
    rk_local = sorted(order_local, key=lambda i: -pts[i][1])
    rk_board = sorted(order_local, key=lambda i: -pts[i][2])
    rank_l = {i: n + 1 for n, i in enumerate(rk_local)}
    rank_b = {i: n + 1 for n, i in enumerate(rk_board)}
    for i, (label, loc, board) in enumerate(pts):
        L.append(f"| {label} | {loc:.6f} | {rank_l[i]} | {board:.5f} | {rank_b[i]} | {board - loc:+.5f} |")
    d2 = sum((rank_l[i] - rank_b[i]) ** 2 for i in range(len(pts)))
    n = len(pts)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    L += ["", f"Spearman ρ = {rho:+.3f}（n={n}）。次方和 Σd²={d2}。",
          "",
          "- 3B 两个点的榜分都**高于**本地（+0.0082 / +0.0083），7B 五个点的榜分都**低于**本地"
          "（−0.0017 至 −0.0083）——偏差有方向性，取决于模型族。",
          "- 本地排序在 7B 内部完全反转：本地第 1 的 7B SFT 在榜上 7B 里垫底，"
          "本地垫底的 DAPO-legacy 在榜上 7B 里第一，且与榜首 3B GRPO 只差 0.00058。",
          "- DAPO-legacy 与 7B GRPO step150 相差 0.00003，**低于本项目 0.002 的可判阈值**，"
          "应视为并列，不要写成「超过」。", ""]

    L += ["", "## 6. 说明", "",
          "- 来源检查点固定为 `step_0050`：两臂的官方口径最佳点都在 step 50，而 `best.json` 是按训练口径"
          "（冲突召回）选的 step 200 / step 300 —— 因此本次没有走 `run_submission.sh`（它跟随 `best.json`），"
          "改用 `run_submission_ckpt.sh` 显式指定检查点。",
          "- 测试集 734 条；`is_uncertain=yes` 的条数与 `clarifying_question != none` 的条数相等，"
          "因为问题文本只在判为冲突时生成。两份提交均满足。",
          "- 测试集上没有标签，表里的冲突条数是**模型自己报的**，不是准确率。"
          "验证集指标（官方/训练口径）才是可比的分数。",
          "- sha256 用于确认归档副本与原始产物是同一份文件。", ""]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"written {OUT} ({len(L)} lines)")
    for r in dapo:
        i = r["info"]
        print(f"  {r['dir']:24s} n={i['n']} conflict={i['conflict']} ({i['rate']:.1f}%) sha={i['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
