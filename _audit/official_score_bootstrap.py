"""用官方 score() 口径重算全部历史实验，并给每个检查点做配对 bootstrap。

为什么需要它：
    项目内所有早期文档（3B GRPO r1/r2、PPO、7B GRPO）的选点与结论都用
    local_score_present = 0.5*F1 + 0.3*冲突召回 + 0.2*特异度，
    而 Kaggle 官方口径是 0.5*F1 + 0.3*冲突准确率 + 0.2*特异度。
    召回不罚误报，准确率罚，两者在多报冲突时方向相反。

本脚本只读已归档的 predictions.jsonl，不修改任何产物。

用法：
    python _audit/official_score_bootstrap.py
输出：
    _audit/official_score_bootstrap.csv
"""
import csv
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "_audit", "official_score_bootstrap.csv")
N_BOOT = 2000
SEED = 20261102

# (实验名, 模型, 算法, run 目录, 检查点 glob)
RUNS = [
    ("3B-GRPO-r1", "3B", "GRPO 300步", "2026-10-29_14-21-GRPO/grpo_runs/pilot_01", "step_*_predictions.jsonl"),
    ("3B-GRPO-r2", "3B", "GRPO 600次", "2026-10-29_23-22-GRPO/runs/pilot_01", "attempt_*_predictions.jsonl"),
    ("3B-PPO", "3B", "PPO 300次", "2026-10-30_13-39-PPO/runs/pilot_01", "attempt_*_predictions.jsonl"),
    ("7B-GRPO", "7B", "GRPO 300步", "2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01", "step_*_predictions.jsonl"),
]


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode(rows):
    """把每行压成整数编码，避免 bootstrap 里反复解析 dict。"""
    emo = sorted({r["gold"]["pred_emotion"] for r in rows})
    emo_id = {e: i for i, e in enumerate(emo)}
    gc = np.array([1 if r["gold"]["is_conflict"] == "yes" else 0 for r in rows], dtype=np.int8)
    pc = np.array([1 if r["prediction"].get("is_conflict") == "yes" else 0 for r in rows], dtype=np.int8)
    ge = np.array([emo_id[r["gold"]["pred_emotion"]] for r in rows], dtype=np.int8)
    pe = np.array([emo_id.get(r["prediction"].get("pred_emotion"), -1) for r in rows], dtype=np.int8)
    return gc, pc, ge, pe, len(emo)


def metrics(gc, pc, ge, pe, n_emo):
    """返回 official（0.3×冲突准确率）与 local（0.3×冲突召回）两套分数。"""
    n = len(gc)
    tp = int(((gc == 1) & (pc == 1)).sum())
    tn = int(((gc == 0) & (pc == 0)).sum())
    fp = int(((gc == 0) & (pc == 1)).sum())
    fn = int(((gc == 1) & (pc == 0)).sum())
    conf_n = int((gc == 1).sum())
    ali_n = int((gc == 0).sum())
    rec = tp / conf_n if conf_n else 0.0
    spec = tn / ali_n if ali_n else 0.0
    acc = (tp + tn) / n
    f1s = []
    for c in range(n_emo):
        a = ge == c
        b = pe == c
        t = int((a & b).sum())
        f = int((~a & b).sum())
        m = int((a & ~b).sum())
        f1s.append(2 * t / (2 * t + f + m) if (2 * t + f + m) else 0.0)
    f1 = sum(f1s) / len(f1s)
    return (0.5 * f1 + 0.3 * acc + 0.2 * spec,
            0.5 * f1 + 0.3 * rec + 0.2 * spec,
            f1, rec, spec, acc, tp, fp, fn, tn)


def bootstrap(base_enc, ckpt_enc, n_boot=N_BOOT, seed=SEED):
    gc, pc, ge, pe, ne = base_enc
    Gc, Pc, Ge, Pe, _ = ckpt_enc
    n = len(gc)
    rng = np.random.default_rng(seed)
    d_off, d_loc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        o1, l1, *_ = metrics(gc[idx], pc[idx], ge[idx], pe[idx], ne)
        o2, l2, *_ = metrics(Gc[idx], Pc[idx], Ge[idx], Pe[idx], ne)
        d_off[i] = o2 - o1
        d_loc[i] = l2 - l1
    return d_off, d_loc


def main():
    results = []
    for exp, model, algo, rel, pat in RUNS:
        run_dir = os.path.join(BASE, rel)
        base_p = os.path.join(run_dir, "sft_baseline_predictions.jsonl")
        if not os.path.exists(base_p):
            print("跳过（无基线预测）:", exp)
            continue
        base_rows = load_rows(base_p)
        base_enc = encode(base_rows)
        b_off, b_loc, *_ = metrics(*base_enc)
        print("=== %s（%s, %s）基线：official %.6f / local %.6f，n=%d"
              % (exp, model, algo, b_off, b_loc, len(base_rows)))

        for f in sorted(glob.glob(os.path.join(run_dir, pat))):
            ckpt_rows = load_rows(f)
            if len(ckpt_rows) != len(base_rows):
                print("  跳过（行数不符）:", os.path.basename(f))
                continue
            # 按 sample_id 对齐，防止顺序差异
            order = {r["sample_id"]: i for i, r in enumerate(base_rows)}
            try:
                ckpt_rows = sorted(ckpt_rows, key=lambda r: order[r["sample_id"]])
            except KeyError:
                print("  跳过（sample_id 不匹配）:", os.path.basename(f))
                continue
            enc = encode(ckpt_rows)
            c_off, c_loc, f1, rec, spec, acc, tp, fp, fn, tn = metrics(*enc)
            d_off, d_loc = bootstrap(base_enc, enc)
            tag = os.path.basename(f).split("_predictions")[0]
            results.append(dict(
                experiment=exp, model=model, method=algo, checkpoint=tag,
                local=round(c_loc, 6), local_delta=round(c_loc - b_loc, 6),
                official=round(c_off, 6), official_delta=round(c_off - b_off, 6),
                official_ci_low=round(float(np.percentile(d_off, 2.5)), 6),
                official_ci_high=round(float(np.percentile(d_off, 97.5)), 6),
                official_p_gt0=round(float((d_off > 0).mean()), 4),
                local_ci_low=round(float(np.percentile(d_loc, 2.5)), 6),
                local_ci_high=round(float(np.percentile(d_loc, 97.5)), 6),
                local_p_gt0=round(float((d_loc > 0).mean()), 4),
                emotion_f1=round(f1, 6), conflict_recall=round(rec, 6),
                conflict_acc=round(acc, 6), aligned_spec=round(spec, 6),
                TP=tp, FP=fp, FN=fn, TN=tn,
                official_significant=("是" if np.percentile(d_off, 2.5) > 0 else "否"),
            ))
            print("  %-10s official %.6f (%+.6f, CI [%+.6f,%+.6f], P>0=%.2f)  local %+.6f"
                  % (tag, c_off, c_off - b_off, results[-1]["official_ci_low"],
                     results[-1]["official_ci_high"], results[-1]["official_p_gt0"], c_loc - b_loc))
        print()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = list(results[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    print("写出", OUT, len(results), "行")

    print()
    print("=== 每个实验按官方口径的最佳点 ===")
    print("%-12s %-10s %8s %8s %8s %s" % ("实验", "检查点", "official", "Δofficial", "Δlocal", "显著"))
    for exp, model, algo, rel, pat in RUNS:
        sub = [r for r in results if r["experiment"] == exp]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["official"])
        print("%-12s %-10s %8.6f %+8.6f %+8.6f %s"
              % (exp, b["checkpoint"], b["official"], b["official_delta"], b["local_delta"], b["official_significant"]))


if __name__ == "__main__":
    sys.exit(main())
