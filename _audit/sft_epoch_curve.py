"""归档分析：SFT epoch 曲线 + 提交集预测差异。

为什么需要它：
    项目内此前只有「最终 adapter」一个 SFT 评测点（local 0.905166 / 官方 0.940752），
    但 sft_run/train/ 下还留着 epoch_1/2/3 三次逐样本预测，从未被分析。
    这三个点能回答「全量微调是否训过头」，也决定 SFT 基线是否已经是最优选点。

    另外 submissions/*/predictions_raw.jsonl 是真正提交给 Kaggle 的 734 条预测，
    之前只用来生成 CSV，没算过模型之间的分歧量 —— 它能解释为什么榜上差距极小。

只读已归档文件。

用法：
    python _audit/sft_epoch_curve.py
输出：
    _audit/sft_epoch_curve.csv
    _audit/submission_disagreement.csv
"""
import csv
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(BASE, "_audit")
N_BOOT = 4000
SEED = 20261102

SFT_DIR = os.path.join(BASE, "2026-10-30_23-00-Omni7B-SFT-GRPO", "sft_run", "train")

SUBMISSIONS = [
    ("7B-SFT", "2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/sft7b/predictions_raw.jsonl"),
    ("7B-GRPO-s50", "2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/grpo7b_step50/predictions_raw.jsonl"),
    ("7B-GRPO-s150", "2026-10-30_23-00-Omni7B-SFT-GRPO/submissions/grpo7b_step150/predictions_raw.jsonl"),
    ("3B-GRPO-r1-s50", "2026-10-29_14-21-GRPO/submission_step50/predictions_raw.jsonl"),
    ("3B-GRPO-r2-s300", "2026-10-29_23-22-GRPO/submission_attempt0300/predictions_raw.jsonl"),
    ("3B-PPO-att100", "2026-10-30_13-39-PPO/submission_attempt0100/predictions_raw.jsonl"),
]


def load_rows(path):
    with open(path, encoding="utf-8") as fp:
        return [json.loads(l) for l in fp if l.strip()]


def encode(rows):
    emo = sorted({r["gold"]["pred_emotion"] for r in rows})
    eid = {e: i for i, e in enumerate(emo)}
    gc = np.array([1 if r["gold"]["is_conflict"] == "yes" else 0 for r in rows], dtype=np.int8)
    pc = np.array([1 if r["prediction"].get("is_conflict") == "yes" else 0 for r in rows], dtype=np.int8)
    ge = np.array([eid[r["gold"]["pred_emotion"]] for r in rows], dtype=np.int8)
    pe = np.array([eid.get(r["prediction"].get("pred_emotion"), -1) for r in rows], dtype=np.int8)
    return gc, pc, ge, pe, len(emo)


def metrics(enc):
    """official = 0.5*F1 + 0.3*冲突准确率 + 0.2*一致特异度（Kaggle 口径）
       local    = 0.5*F1 + 0.3*冲突召回   + 0.2*一致特异度（早期选点口径）"""
    gc, pc, ge, pe, ne = enc
    n = len(gc)
    tp = int(((gc == 1) & (pc == 1)).sum())
    tn = int(((gc == 0) & (pc == 0)).sum())
    fp = int(((gc == 0) & (pc == 1)).sum())
    fn = int(((gc == 1) & (pc == 0)).sum())
    conf_n, ali_n = int((gc == 1).sum()), int((gc == 0).sum())
    rec = tp / conf_n if conf_n else 0.0
    spec = tn / ali_n if ali_n else 0.0
    acc = (tp + tn) / n
    f1s = []
    for c in range(ne):
        a, b = ge == c, pe == c
        t = int((a & b).sum())
        f = int((~a & b).sum())
        m = int((a & ~b).sum())
        f1s.append(2 * t / (2 * t + f + m) if (2 * t + f + m) else 0.0)
    f1 = float(np.mean(f1s))
    return dict(official=0.5 * f1 + 0.3 * acc + 0.2 * spec,
                local=0.5 * f1 + 0.3 * rec + 0.2 * spec,
                f1=f1, rec=rec, acc=acc, spec=spec, TP=tp, FP=fp, FN=fn, TN=tn)


def pair_bootstrap(enc_a, enc_b, n_boot=N_BOOT, seed=SEED):
    """配对 bootstrap：同一样本索引下取两个检查点，返回 (Δofficial, Δlocal) 样本。"""
    gc, pc, ge, pe, ne = enc_a
    Gc, Pc, Ge, Pe, _ = enc_b
    n = len(gc)
    rng = np.random.default_rng(seed)
    d_off, d_loc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = metrics((gc[idx], pc[idx], ge[idx], pe[idx], ne))
        b = metrics((Gc[idx], Pc[idx], Ge[idx], Pe[idx], ne))
        d_off[i] = b["official"] - a["official"]
        d_loc[i] = b["local"] - a["local"]
    return d_off, d_loc


def ci(samples):
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)),
            float((samples > 0).mean()))


def epoch_curve():
    files = sorted(glob.glob(os.path.join(SFT_DIR, "epoch_*_predictions.jsonl")))
    if not files:
        print("未找到 epoch 预测，跳过")
        return []
    rows_out, encs = [], {}
    for f in files:
        tag = os.path.basename(f).split("_predictions")[0]
        rows = load_rows(f)
        enc = encode(rows)
        encs[tag] = enc
        m = metrics(enc)
        rows_out.append(dict(stage=tag, n=len(rows), local=round(m["local"], 6),
                             official=round(m["official"], 6), emotion_f1=round(m["f1"], 6),
                             conflict_recall=round(m["rec"], 6), conflict_acc=round(m["acc"], 6),
                             aligned_spec=round(m["spec"], 6), TP=m["TP"], FP=m["FP"],
                             FN=m["FN"], TN=m["TN"], ci_low="", ci_high="", p_gt0="", note=""))
        print("  %-10s local=%.6f  official=%.6f  F1=%.6f  TP=%d FP=%d FN=%d"
              % (tag, m["local"], m["official"], m["f1"], m["TP"], m["FP"], m["FN"]))

    last = sorted(encs)[-1]
    print("\n  相对 %s 的配对 bootstrap（4000 次）：" % last)
    for tag in sorted(encs):
        if tag == last:
            continue
        d_off, d_loc = pair_bootstrap(encs[last], encs[tag])
        lo, hi, p = ci(d_off)
        lo2, hi2, p2 = ci(d_loc)
        for r in rows_out:
            if r["stage"] == tag:
                r["ci_low"], r["ci_high"], r["p_gt0"] = round(lo, 6), round(hi, 6), round(p, 4)
                r["note"] = "vs %s: Δofficial %+0.6f, Δlocal %+0.6f" % (
                    last, r["official"] - [x for x in rows_out if x["stage"] == last][0]["official"],
                    r["local"] - [x for x in rows_out if x["stage"] == last][0]["local"])
        print("    %-10s Δofficial CI [%+.6f, %+.6f] P(>0)=%.3f | Δlocal CI [%+.6f, %+.6f] P(>0)=%.3f"
              % (tag, lo, hi, p, lo2, hi2, p2))
    return rows_out


def submission_disagreement():
    paths = {}
    for tag, rel in SUBMISSIONS:
        p = os.path.join(BASE, rel)
        if os.path.exists(p):
            paths[tag] = p
        else:
            print("  缺失:", tag)
    data = {t: {r["sample_id"]: r["prediction"] for r in load_rows(p)} for t, p in paths.items()}
    tags = list(data)
    out = []
    print("\n  两两差异（734 条提交集，冲突位+情绪位任一不同即记 1 条）：")
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            common = set(data[a]) & set(data[b])
            diff_c = diff_e = 0
            for sid in common:
                pa, pb = data[a][sid], data[b][sid]
                if pa.get("is_conflict") != pb.get("is_conflict"):
                    diff_c += 1
                if pa.get("pred_emotion") != pb.get("pred_emotion"):
                    diff_e += 1
            out.append(dict(a=a, b=b, n=len(common), conflict_diff=diff_c,
                            emotion_diff=diff_e,
                            any_diff=len([s for s in common
                                          if data[a][s] != data[b][s]])))
    out.sort(key=lambda r: r["any_diff"])
    for r in out:
        print("    %-15s vs %-15s 冲突位差 %3d 条  情绪位差 %3d 条  合计 %3d/734 (%.1f%%)"
              % (r["a"], r["b"], r["conflict_diff"], r["emotion_diff"],
                 r["any_diff"], 100 * r["any_diff"] / r["n"]))
    return out


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("写出", path, len(rows), "行")


def main():
    print("=== 7B SFT epoch 曲线 ===")
    ep = epoch_curve()
    if ep:
        write_csv(os.path.join(AUDIT, "sft_epoch_curve.csv"), ep,
                  ["stage", "n", "local", "official", "emotion_f1", "conflict_recall",
                   "conflict_acc", "aligned_spec", "TP", "FP", "FN", "TN",
                   "ci_low", "ci_high", "p_gt0", "note"])
    print("\n=== 提交集预测差异 ===")
    dis = submission_disagreement()
    if dis:
        write_csv(os.path.join(AUDIT, "submission_disagreement.csv"), dis,
                  ["a", "b", "n", "conflict_diff", "emotion_diff", "any_diff"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
