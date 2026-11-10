"""重算 7B 三方对比（SFT / GRPO / DAPO 两臂），输出 reports/three_way_comparison.csv。

数据源全部是本地的 *_metrics.json：
  7B GRPO  -> 2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01/
  DAPO 两臂 -> 2026-11-01-DAPO/runs/{dapo7b,dapo7b_legacy}/
旧表漏了 GRPO step_0300 与 DAPO_legacy step_0300，本次补全并列出差异。
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRPO = ROOT / "2026-10-30_23-00-Omni7B-SFT-GRPO" / "grpo_run" / "pilot_01"
DAPO = ROOT / "2026-11-01-DAPO" / "runs"
OUT = ROOT / "2026-11-01-DAPO" / "reports" / "three_way_comparison.csv"
STEPS = [50, 100, 150, 200, 250, 300]
COLS = ["experiment", "method", "step", "local_score", "kaggle_score", "emotion_f1",
        "conflict_f1", "conflict_recall", "conflict_acc", "aligned_spec",
        "TP", "FP", "FN", "TN"]


def row_of(method, step, m):
    a, c = m["all"], m["all"]["conflict_confusion"]
    emo = a["emotion_macro_f1_present"]
    rec, spec, acc = a["conflict_recall"], a["aligned_specificity"], a["conflict_accuracy"]
    return {
        "experiment": "7B", "method": method, "step": step,
        "local_score": round(0.5 * emo + 0.3 * rec + 0.2 * spec, 6),
        "kaggle_score": round(0.5 * emo + 0.3 * acc + 0.2 * spec, 6),
        "emotion_f1": round(emo, 6), "conflict_f1": round(a["conflict_f1"], 6),
        "conflict_recall": round(rec, 6), "conflict_acc": round(acc, 6),
        "aligned_spec": round(spec, 6),
        "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
    }


def load(p):
    if not p.exists():
        raise SystemExit(f"缺少 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    rows = [row_of("SFT", 0, load(GRPO / "sft_baseline_metrics.json"))]
    for s in STEPS:
        rows.append(row_of("GRPO", s, load(GRPO / f"step_{s:04d}_metrics.json")))
    for s in STEPS:
        rows.append(row_of("DAPO", s, load(DAPO / "dapo7b" / f"step_{s:04d}_metrics.json")))
    for s in STEPS:
        rows.append(row_of("DAPO_legacy", s, load(DAPO / "dapo7b_legacy" / f"step_{s:04d}_metrics.json")))

    diffs = []
    if OUT.exists():
        old = {(r["method"], int(r["step"])): r for r in csv.DictReader(open(OUT, encoding="utf-8"))}
        for r in rows:
            k = (r["method"], r["step"])
            if k not in old:
                diffs.append(f"新增 {k}  kaggle={r['kaggle_score']}")
                continue
            for col in COLS:
                if str(r[col]) != str(old[k][col]):
                    diffs.append(f"{k}.{col}: 旧 {old[k][col]} -> 新 {r[col]}")
        shutil.copy2(OUT, OUT.with_suffix(".csv.prev"))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {OUT}（{len(rows)} 行）")
    print("与旧表差异:" if diffs else "与旧表完全一致（零差异）")
    for d in diffs:
        print("  ", d)

    print("\nmethod        step  local     kaggle    Δkaggle(vs SFT)  TP/FP/FN")
    base = rows[0]["kaggle_score"]
    for r in rows:
        print(f"{r['method']:12s} {r['step']:>4}  {r['local_score']:.6f}  {r['kaggle_score']:.6f}  "
              f"{r['kaggle_score']-base:+.6f}        {r['TP']:>2}/{r['FP']:>2}/{r['FN']:>2}")


if __name__ == "__main__":
    main()
