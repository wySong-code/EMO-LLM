"""汇总 3B 侧（SFT 基线 / GRPO r1 / GRPO r2 / PPO）逐检查点指标，输出 3b_step_curves.csv。

字段与 2026-11-01-DAPO/reports/dapo_step_curves.csv 对齐，便于并表比较。
只读各实验目录下的 *_metrics.json，不重跑评测；official / local 两套口径现场重算，
并与 _audit/master_results.csv 交叉校验，任何不一致直接抛错。
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (实验名, 臂名, step, metrics 相对路径)
SOURCES = [
    ("3B-GRPO-r1", "sft_baseline", 0, "2026-10-29_14-21-GRPO/grpo_runs/pilot_01/sft_baseline_metrics.json"),
] + [
    ("3B-GRPO-r1", "grpo3b_r1", s, f"2026-10-29_14-21-GRPO/grpo_runs/pilot_01/step_{s:04d}_metrics.json")
    for s in (50, 100, 150, 200, 250, 300)
] + [
    ("3B-GRPO-r2", "sft_baseline_r2", 0, "2026-10-29_23-22-GRPO/runs/pilot_01/sft_baseline_metrics.json"),
] + [
    ("3B-GRPO-r2", "grpo3b_r2", s, f"2026-10-29_23-22-GRPO/runs/pilot_01/attempt_{s:04d}_metrics.json")
    for s in (100, 200, 300, 400, 500, 600)
] + [
    ("3B-PPO", "ppo3b", s, f"2026-10-30_13-39-PPO/runs/pilot_01/attempt_{s:04d}_metrics.json")
    for s in (100, 200, 300)
]

COLS = ["experiment", "arm", "step", "local_score", "kaggle_score", "emotion_f1",
        "conflict_f1", "conflict_recall", "aligned_spec", "conflict_acc",
        "TP", "FP", "FN", "TN", "all_fields_correct", "n"]


def scores(m):
    a = m["all"]
    emo, rec = a["emotion_macro_f1_present"], a["conflict_recall"]
    spec, acc = a["aligned_specificity"], a["conflict_accuracy"]
    return (0.5 * emo + 0.3 * rec + 0.2 * spec,
            0.5 * emo + 0.3 * acc + 0.2 * spec)


def main():
    rows = []
    for exp, arm, step, rel in SOURCES:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"缺少指标文件: {rel}")
        m = json.loads(p.read_text(encoding="utf-8"))
        a, c = m["all"], m["all"]["conflict_confusion"]
        local, official = scores(m)
        rows.append({
            "experiment": exp, "arm": arm, "step": step,
            "local_score": round(local, 6), "kaggle_score": round(official, 6),
            "emotion_f1": round(a["emotion_macro_f1_present"], 6),
            "conflict_f1": round(a["conflict_f1"], 6),
            "conflict_recall": round(a["conflict_recall"], 6),
            "aligned_spec": round(a["aligned_specificity"], 6),
            "conflict_acc": round(a["conflict_accuracy"], 6),
            "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
            "all_fields_correct": a["all_fields_correct_n"], "n": a["n"],
        })

    # 与 master_results.csv 交叉校验（只比两边都有的字段）
    master = list(csv.DictReader(open(ROOT / "_audit/master_results.csv", encoding="utf-8")))
    index = {(r["experiment"], r["checkpoint"]): r for r in master}
    checks = 0
    for r in rows:
        if r["step"] == 0:
            continue
        ck = f"step_{r['step']:04d}" if r["experiment"] == "3B-GRPO-r1" else f"attempt_{r['step']:04d}"
        key = (r["experiment"], ck)
        if key not in index:
            continue
        want = float(index[key]["official"])
        if abs(want - r["kaggle_score"]) > 5e-7:
            raise SystemExit(f"{key} 官方口径不一致: 本表 {r['kaggle_score']} vs master {want}")
        checks += 1

    out = ROOT / "_audit/3b_step_curves.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {out}  ({len(rows)} 行, 与 master 校验通过 {checks} 项)")
    for r in rows:
        print(f"{r['experiment']:12s} {r['arm']:16s} step={r['step']:>3} "
              f"local={r['local_score']:.6f} official={r['kaggle_score']:.6f} "
              f"TP/FP/FN={r['TP']}/{r['FP']}/{r['FN']}")


if __name__ == "__main__":
    main()
