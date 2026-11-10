"""从已归档的 runs/*/*_metrics.json 重算 DAPO 逐检查点曲线，输出 reports/dapo_step_curves.csv。

数据源是训练时落盘的 *_metrics.json（已归档），
两套口径现场重算，不手抄。生成前先与旧 csv 逐行比对，差异打印出来再覆盖。
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAPO = ROOT / "2026-11-01-DAPO"
OUT = DAPO / "reports" / "dapo_step_curves.csv"
ARMS = ["dapo7b", "dapo7b_legacy"]
STEPS = [50, 100, 150, 200, 250, 300]

COLS = ["arm", "step", "local_score", "kaggle_score", "emotion_f1", "conflict_f1",
        "conflict_recall", "aligned_spec", "conflict_acc",
        "TP", "FP", "FN", "TN", "all_fields_correct", "n"]


def row_of(arm, step, m):
    a, c = m["all"], m["all"]["conflict_confusion"]
    emo = a["emotion_macro_f1_present"]
    rec, spec, acc = a["conflict_recall"], a["aligned_specificity"], a["conflict_accuracy"]
    local = 0.5 * emo + 0.3 * rec + 0.2 * spec
    official = 0.5 * emo + 0.3 * acc + 0.2 * spec
    return {
        "arm": arm, "step": step,
        "local_score": round(local, 6), "kaggle_score": round(official, 6),
        "emotion_f1": round(emo, 6), "conflict_f1": round(a["conflict_f1"], 6),
        "conflict_recall": round(rec, 6), "aligned_spec": round(spec, 6),
        "conflict_acc": round(acc, 6),
        "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
        "all_fields_correct": a["all_fields_correct_n"], "n": a["n"],
    }


def main():
    base_path = DAPO / "runs" / ARMS[0] / "sft_baseline_metrics.json"
    if not base_path.exists():
        raise SystemExit(f"缺少 {base_path}，先跑 _pull_results.py")
    rows = [row_of("sft_baseline", 0, json.loads(base_path.read_text(encoding="utf-8")))]

    for arm in ARMS:
        for s in STEPS:
            p = DAPO / "runs" / arm / f"step_{s:04d}_metrics.json"
            if not p.exists():
                raise SystemExit(f"缺少 {p}")
            rows.append(row_of(arm, s, json.loads(p.read_text(encoding="utf-8"))))

    # 与旧 csv 比对
    diffs = []
    if OUT.exists():
        old = {(r["arm"], int(r["step"])): r for r in csv.DictReader(open(OUT, encoding="utf-8"))}
        for r in rows:
            k = (r["arm"], r["step"])
            if k not in old:
                diffs.append(f"新增 {k}")
                continue
            for col in COLS:
                if str(r[col]) != str(old[k][col]):
                    diffs.append(f"{k}.{col}: 旧 {old[k][col]} -> 新 {r[col]}")
        for k in old:
            if not any((r["arm"], r["step"]) == k for r in rows):
                diffs.append(f"消失 {k}")
        shutil.copy2(OUT, OUT.with_suffix(".csv.prev"))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"写出 {OUT}（{len(rows)} 行）")
    print("与旧 csv 差异:" if diffs else "与旧 csv 完全一致（零差异）")
    for d in diffs:
        print("  ", d)

    print("\narm            step  local     official   TP/FP/FN  all_ok")
    for r in rows:
        print(f"{r['arm']:14s} {r['step']:>4}  {r['local_score']:.6f}  "
              f"{r['kaggle_score']:.6f}  {r['TP']:>2}/{r['FP']:>2}/{r['FN']:>2}   {r['all_fields_correct']}")


if __name__ == "__main__":
    main()
