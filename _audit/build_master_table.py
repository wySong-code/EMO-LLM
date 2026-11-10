"""把五个实验的全部检查点汇总成一张主表（GitHub 首页唯一数据源）。

输入（全部为本地已有文件）：
    _audit/official_score_bootstrap.csv        3B GRPO r1 / 3B GRPO r2 / 3B PPO / 7B GRPO
    2026-11-01-DAPO/reports/dapo_step_curves.csv  7B DAPO 两臂 12 个检查点
    各实验的 sft_baseline_metrics.json         各实验自己的 SFT 基线
    2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/train/epoch_*_predictions.jsonl  7B SFT 三个 epoch

输出：
    _audit/master_results.csv
    _audit/master_results.md   （可直接贴进 README 的表格）
"""
import csv
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(BASE, "_audit")

BASELINES = [
    ("3B-GRPO-r1", "3B", "SFT", "2026-10-29_14-21-GRPO/grpo_runs/pilot_01"),
    ("3B-GRPO-r2", "3B", "SFT", "2026-10-29_23-22-GRPO/runs/pilot_01"),
    ("3B-PPO", "3B", "SFT", "2026-10-30_13-39-PPO/runs/pilot_01"),
    ("7B-GRPO", "7B", "SFT", "2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01"),
]

COLS = ["experiment", "model", "method", "checkpoint", "official", "official_delta",
        "official_ci_low", "official_ci_high", "official_p_gt0", "local", "local_delta",
        "emotion_f1", "conflict_f1", "conflict_recall", "conflict_acc", "aligned_spec",
        "TP", "FP", "FN", "TN", "significant", "source"]


def metrics_from_json(path):
    d = json.load(open(path, encoding="utf-8"))["all"]
    cc = d["conflict_confusion"]
    f1 = d["emotion_macro_f1_present"]
    rec, spec, acc = d["conflict_recall"], d["aligned_specificity"], d["conflict_accuracy"]
    return dict(official=round(0.5 * f1 + 0.3 * acc + 0.2 * spec, 6),
                local=round(d["local_score_present"], 6),
                emotion_f1=round(f1, 6), conflict_f1=round(d.get("conflict_f1", -1), 6),
                conflict_recall=round(rec, 6), conflict_acc=round(acc, 6),
                aligned_spec=round(spec, 6),
                TP=cc["TP"], FP=cc["FP"], FN=cc["FN"], TN=cc["TN"])


def blank(**kw):
    r = {c: "" for c in COLS}
    r.update(kw)
    return r


def main():
    rows = []

    # 1) 四个早期实验的 SFT 基线 + 各 RL 检查点
    boot = list(csv.DictReader(open(os.path.join(AUDIT, "official_score_bootstrap.csv"), encoding="utf-8")))
    for exp, model, algo, rel in BASELINES:
        p = os.path.join(BASE, rel, "sft_baseline_metrics.json")
        if not os.path.exists(p):
            continue
        m = metrics_from_json(p)
        rows.append(blank(experiment=exp, model=model, method="SFT（基线）", checkpoint="final",
                          official=m["official"], official_delta=0.0, local=m["local"],
                          local_delta=0.0, emotion_f1=m["emotion_f1"],
                          conflict_f1=m["conflict_f1"], conflict_recall=m["conflict_recall"],
                          conflict_acc=m["conflict_acc"], aligned_spec=m["aligned_spec"],
                          TP=m["TP"], FP=m["FP"], FN=m["FN"], TN=m["TN"],
                          significant="—", source="sft_baseline_metrics.json"))

    for r in boot:
        rows.append(blank(
            experiment=r["experiment"], model=r["model"], method=r["method"],
            checkpoint=r["checkpoint"],
            official=float(r["official"]), official_delta=float(r["official_delta"]),
            official_ci_low=float(r["official_ci_low"]), official_ci_high=float(r["official_ci_high"]),
            official_p_gt0=float(r["official_p_gt0"]),
            local=float(r["local"]), local_delta=float(r["local_delta"]),
            emotion_f1=float(r["emotion_f1"]), conflict_f1="",
            conflict_recall=float(r["conflict_recall"]), conflict_acc=float(r["conflict_acc"]),
            aligned_spec=float(r["aligned_spec"]),
            TP=int(r["TP"]), FP=int(r["FP"]), FN=int(r["FN"]), TN=int(r["TN"]),
            significant=r["official_significant"],
            source="official_score_bootstrap.csv"))

    # 2) 7B SFT 三个 epoch
    ep = os.path.join(AUDIT, "sft_epoch_curve.csv")
    if os.path.exists(ep):
        for r in csv.DictReader(open(ep, encoding="utf-8")):
            rows.append(blank(
                experiment="7B-SFT-epochs", model="7B", method="SFT 逐 epoch",
                checkpoint=r["stage"], official=float(r["official"]),
                official_delta="", local=float(r["local"]), local_delta="",
                emotion_f1=float(r["emotion_f1"]), conflict_f1="",
                conflict_recall=float(r["conflict_recall"]), conflict_acc=float(r["conflict_acc"]),
                aligned_spec=float(r["aligned_spec"]),
                TP=int(r["TP"]), FP=int(r["FP"]), FN=int(r["FN"]), TN=int(r["TN"]),
                significant="", source="sft_epoch_curve.csv"))

    # 3) DAPO 两臂
    dapo = os.path.join(BASE, "2026-11-01-DAPO", "reports", "dapo_step_curves.csv")
    if os.path.exists(dapo):
        for r in csv.DictReader(open(dapo, encoding="utf-8")):
            if r["arm"] == "sft_baseline":
                continue
            method = "DAPO（score_focused）" if r["arm"] == "dapo7b" else "DAPO（legacy 奖励）"
            rows.append(blank(
                experiment="7B-DAPO", model="7B", method=method,
                checkpoint="step_" + r["step"],
                official=float(r["kaggle_score"]), official_delta="",
                local=float(r["local_score"]), local_delta="",
                emotion_f1=float(r["emotion_f1"]), conflict_f1=float(r["conflict_f1"]),
                conflict_recall=float(r["conflict_recall"]),
                conflict_acc=float(r["conflict_acc"]), aligned_spec=float(r["aligned_spec"]),
                TP=int(r["TP"]), FP=int(r["FP"]), FN=int(r["FN"]), TN=int(r["TN"]),
                significant="（待补 CI）", source="dapo_step_curves.csv"))

    out = os.path.join(AUDIT, "master_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print("写出", out, len(rows), "行")

    # 每个实验的官方口径最佳点
    print()
    print("%-16s %-18s %-12s %9s %9s %9s %7s %5s %5s" %
          ("实验", "方法", "检查点", "官方分", "Δ官方", "Δ本地", "F1", "TP", "FP"))
    best_rows = []
    groups = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["method"]), []).append(r)
    for (exp, method), sub in groups.items():
        b = max(sub, key=lambda r: float(r["official"]))
        best_rows.append(b)
        print("%-16s %-18s %-12s %9.6f %9s %9s %7.4f %5d %5d" %
              (exp, method, b["checkpoint"], float(b["official"]),
               b["official_delta"] if b["official_delta"] != "" else "—",
               b["local_delta"] if b["local_delta"] != "" else "—",
               float(b["emotion_f1"]), int(b["TP"]), int(b["FP"])))

    # 生成 markdown
    md = ["| 实验 | 模型 | 方法 | 检查点 | 官方口径 | 训练口径 | 情绪 F1 | 冲突 F1 | TP | FP | FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["experiment"], r["model"], r["method"], r["checkpoint"],
            ("%.6f" % float(r["official"])) if r["official"] != "" else "—",
            ("%.6f" % float(r["local"])) if r["local"] != "" else "—",
            ("%.4f" % float(r["emotion_f1"])) if r["emotion_f1"] != "" else "—",
            ("%.4f" % float(r["conflict_f1"])) if r["conflict_f1"] != "" else "—",
            r["TP"], r["FP"], r["FN"]))
    open(os.path.join(AUDIT, "master_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n写出 master_results.md，%d 行数据" % (len(md) - 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
