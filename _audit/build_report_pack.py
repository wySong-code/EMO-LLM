# -*- coding: utf-8 -*-
"""生成汇报用的结构化数据表（只读本地归档）。

产物：
  _audit/report_metrics_by_subset.csv  37 检查点 × 5 子集（all/VT/VA/original/generated）的全指标
  _audit/report_stage_summary.csv      每个实验按 5 个训练阶段聚合的奖励侧与训练信号
  _audit/report_experiment_matrix.csv  实验级元信息矩阵（规模/算法/训练量/最佳点/榜分）

口径与 build_all_curves.py 完全一致，并从它读 CKPT_SOURCES，避免两处路径分叉。
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import build_all_curves as B

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"

SUBSETS = ["all", "VT", "VA", "original", "generated"]

SUB_COLS = ["experiment", "arm", "unit", "unit_value", "progress_pct", "subset", "n",
            "local_score", "official_score",
            "emotion_accuracy", "emotion_macro_f1_present", "emotion_macro_f1_fixed3",
            "conflict_n", "aligned_n", "conflict_recall", "aligned_specificity",
            "conflict_accuracy", "conflict_f1",
            "TP", "TN", "FP", "FN", "invalid_on_conflict", "invalid_on_aligned",
            "all_fields_correct_n", "format_valid_rate", "token_limit_hits", "source"]

# 各实验的 Kaggle public 榜分。**只有真正提交过并拿到分的检查点才有值**，
# 其余一律 None（不要用本地口径分数顶替，那是两套口径）。
KAGGLE = {
    ("3B-SFT", "sft3b", 0): 0.93049,
    ("3B-GRPO-r1", "grpo3b_r1", 50): 0.93638,
    ("7B-SFT", "sft7b", 3): 0.93243,
    ("7B-GRPO", "grpo7b", 50): 0.93444,
    ("7B-GRPO", "grpo7b", 150): 0.93577,
    ("7B-DAPO", "dapo7b", 50): 0.93345,
    ("7B-DAPO", "dapo7b_legacy", 50): 0.93580,
}

META = {
    "3B-SFT": dict(scale="3B", algo="SFT", unit="step", total=1),
    "3B-GRPO-r1": dict(scale="3B", algo="GRPO", unit="step", total=300),
    "3B-GRPO-r2": dict(scale="3B", algo="GRPO(配额改造)", unit="attempt", total=600),
    "3B-PPO": dict(scale="3B", algo="PPO(带 critic)", unit="attempt", total=300),
    "7B-SFT": dict(scale="7B", algo="SFT", unit="epoch", total=3),
    "7B-GRPO": dict(scale="7B", algo="GRPO", unit="step", total=300),
    "7B-DAPO": dict(scale="7B", algo="DAPO(两臂)", unit="step", total=300),
}

# 同名实验里需要按臂区分显示名
ARM_LABEL = {"dapo7b": "7B-DAPO", "dapo7b_legacy": "7B-DAPO-legacy"}
ARM_ALGO = {"dapo7b": "DAPO", "dapo7b_legacy": "DAPO-legacy"}


def build_subset_table():
    rows = []
    for exp, arm, unit, val, total, rel in B.CKPT_SOURCES:
        p = ROOT / rel
        m = json.loads(p.read_text(encoding="utf-8"))
        for sub in SUBSETS:
            if sub not in m:
                continue
            d = m[sub]
            cc = d.get("conflict_confusion", {})
            emo = d.get("emotion_macro_f1_present")
            spec = d.get("aligned_specificity")
            if emo is None or spec is None:
                continue
            rows.append({
                "experiment": exp, "arm": arm, "unit": unit, "unit_value": val,
                "progress_pct": round(100.0 * val / total, 2), "subset": sub,
                "n": d.get("n"),
                "local_score": round(0.5 * emo + 0.3 * d.get("conflict_recall", 0) + 0.2 * spec, 6),
                "official_score": round(0.5 * emo + 0.3 * d.get("conflict_accuracy", 0) + 0.2 * spec, 6),
                "emotion_accuracy": round(d.get("emotion_accuracy", 0), 6),
                "emotion_macro_f1_present": round(emo, 6),
                "emotion_macro_f1_fixed3": round(d.get("emotion_macro_f1_fixed3", 0), 6),
                "conflict_n": d.get("conflict_n"), "aligned_n": d.get("aligned_n"),
                "conflict_recall": round(d.get("conflict_recall", 0), 6),
                "aligned_specificity": round(spec, 6),
                "conflict_accuracy": round(d.get("conflict_accuracy", 0), 6),
                "conflict_f1": round(d.get("conflict_f1", 0), 6),
                "TP": cc.get("TP"), "TN": cc.get("TN"), "FP": cc.get("FP"), "FN": cc.get("FN"),
                "invalid_on_conflict": cc.get("invalid_on_conflict"),
                "invalid_on_aligned": cc.get("invalid_on_aligned"),
                "all_fields_correct_n": d.get("all_fields_correct_n"),
                "format_valid_rate": round(d.get("format_valid_rate", 0), 6),
                "token_limit_hits": d.get("token_limit_hits"),
                "source": rel,
            })
    out = AUDIT / "report_metrics_by_subset.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUB_COLS)
        w.writeheader()
        w.writerows(rows)
    n_ck = len({(r['experiment'], r['arm'], r['unit_value']) for r in rows})
    print(f"[subset] {len(rows)} 行（{n_ck} 检查点 × {len(SUBSETS)} 子集） -> {out.name}")
    return rows


def bucket5(pct):
    if pct <= 20:
        return "1_前20%"
    if pct <= 40:
        return "2_20-40%"
    if pct <= 60:
        return "3_40-60%"
    if pct <= 80:
        return "4_60-80%"
    return "5_末20%"


STAGE_COLS = ["experiment", "arm", "stage", "n_units",
              "mean_reward", "std_reward", "zero_spread_rate", "grad_group_rate",
              "accept_rate", "decision_acc", "all_correct_rate",
              "conflict_grad_share", "top_reward_share", "candidates"]


def build_stage_summary():
    rew = list(csv.DictReader(open(AUDIT / "curves_reward_per_step.csv", encoding="utf-8")))
    sig = list(csv.DictReader(open(AUDIT / "curves_train_signals.csv", encoding="utf-8")))

    # 训练信号按 (实验, 臂, 阶段) 汇总
    sagg = defaultdict(lambda: defaultdict(list))
    for r in sig:
        k = (r["experiment"], r["arm"], bucket5(float(r["progress_pct"])))
        sagg[k][r["metric"]].append(float(r["value"]))

    ragg = defaultdict(list)
    for r in rew:
        k = (r["experiment"], r["arm"], bucket5(float(r["progress_pct"])))
        ragg[k].append(r)

    rows = []
    for k in sorted(ragg):           # 只输出有奖励侧数据的（实验, 臂, 阶段）
        exp, arm, stage = k
        s = sagg.get(k, {})
        rs = ragg[k]
        n_units = len({r["unit_value"] for r in rs})

        def m(name, agg=statistics.fmean):
            v = s.get(name) or []
            return round(agg(v), 6) if v else None

        def rm(name):
            v = [float(r[name]) for r in rs if r.get(name) not in (None, "")]
            return round(statistics.fmean(v), 6) if v else None

        # PPO 的 4 个样本来自不同 prompt，没有"同组候选"概念 → 组级指标一律留空
        is_ppo = arm == "ppo3b"

        gsum = sum(int(r["grad_groups"]) for r in rs if r.get("grad_groups") not in (None, ""))
        gtot = sum(int(r["groups"]) for r in rs if r.get("groups") not in (None, ""))
        wsum, wval = 0.0, 0.0
        for r in rs:
            g = int(r["grad_groups"]) if r.get("grad_groups") not in (None, "") else 0
            sh = r.get("grad_conflict_share")
            if g and sh not in (None, ""):
                wsum += g
                wval += g * float(sh)
        rows.append({
            "experiment": ARM_LABEL.get(arm, exp), "arm": arm, "stage": stage, "n_units": n_units,
            "mean_reward": rm("mean_reward"),
            "std_reward": rm("std_reward"),
            "zero_spread_rate": None if is_ppo else rm("zero_spread_group_rate"),
            "grad_group_rate": None if is_ppo else (round(gsum / gtot, 6) if gtot else None),
            "accept_rate": m("accept_rate"),
            "decision_acc": rm("decision_acc"),
            "all_correct_rate": None,
            "conflict_grad_share": None if is_ppo else (round(wval / wsum, 6) if wsum else None),
            "top_reward_share": None if is_ppo else rm("top_reward_share"),
            "candidates": sum(int(r["candidates"]) for r in rs if r.get("candidates") not in (None, "")),
        })

    # all_correct_rate：组级口径各实验字段不同，按可得字段补
    for r in rows:
        k = (r["experiment"], r["arm"], r["stage"])
        s = sagg.get(k, {})
        ac = s.get("all_correct_groups") or []
        gp = s.get("groups_attempted") or []
        if ac and gp and sum(gp):
            r["all_correct_rate"] = round(sum(ac) / sum(gp), 6)

    out = AUDIT / "report_stage_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STAGE_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[stage] {len(rows)} 行 -> {out.name}")
    return rows


MAT_COLS = ["experiment", "scale", "algo", "unit", "total_units", "n_checkpoints",
            "sft_start_official", "best_official", "best_at", "last_official",
            "best_local", "best_local_at", "kaggle_best", "kaggle_at",
            "mean_grad_group_rate", "mean_zero_spread_rate", "mean_reward"]


_START_CACHE = {}


def start_official(exp):
    """该轮 RL 的 SFT 起点官方口径分数；读各自的 sft_baseline_metrics.json。"""
    rel = B.SFT_BASELINE.get(exp)
    if not rel:
        return None
    if rel not in _START_CACHE:
        p = ROOT / rel
        if not p.exists():
            return None
        _START_CACHE[rel] = round(B.scores(json.loads(p.read_text(encoding="utf-8")))[1], 6)
    return _START_CACHE[rel]


def build_matrix():
    cp = list(csv.DictReader(open(AUDIT / "curves_checkpoints_all.csv", encoding="utf-8")))
    rew = list(csv.DictReader(open(AUDIT / "curves_reward_per_step.csv", encoding="utf-8")))
    by = defaultdict(list)
    for r in cp:
        by[(r["experiment"], r["arm"])].append(r)
    rby = defaultdict(list)
    for r in rew:
        rby[(r["experiment"], r["arm"])].append(r)

    rows = []
    for (exp, arm), rs in by.items():
        rs.sort(key=lambda r: float(r["unit_value"]))
        meta = META[exp]
        name = ARM_LABEL.get(arm, exp)
        best = max(rs, key=lambda r: float(r["official_score"]))
        bl = max(rs, key=lambda r: float(r["local_score"]))

        def at(r):
            u = float(r["unit_value"])
            return "起点(未训练)" if u == 0 else f"{meta['unit']} {int(u)}"

        def at_unit(u):
            return "起点(未训练)" if float(u) == 0 else f"{meta['unit']} {int(u)}"

        kg = [(KAGGLE[k], k[2]) for k in KAGGLE if k[0] == exp and k[1] == arm and KAGGLE[k]]
        kbest = max(kg) if kg else None
        rr = rby.get((exp, arm), [])
        gsum = sum(int(r["grad_groups"]) for r in rr if r.get("grad_groups") not in (None, ""))
        gtot = sum(int(r["groups"]) for r in rr if r.get("groups") not in (None, ""))
        zs = [float(r["zero_spread_group_rate"]) for r in rr if r.get("zero_spread_group_rate") not in (None, "")]
        mr = [float(r["mean_reward"]) for r in rr if r.get("mean_reward") not in (None, "")]
        rows.append({
            "experiment": name, "scale": meta["scale"],
            "algo": ARM_ALGO.get(arm, meta["algo"]),
            "unit": meta["unit"], "total_units": meta["total"], "n_checkpoints": len(rs),
            "sft_start_official": start_official(exp),
            "best_official": best["official_score"], "best_at": at(best),
            "last_official": rs[-1]["official_score"],
            "best_local": bl["local_score"], "best_local_at": at(bl),
            "kaggle_best": kbest[0] if kbest else None,
            "kaggle_at": at_unit(kbest[1]) if kbest else None,
            "mean_grad_group_rate": round(gsum / gtot, 6) if gtot else None,
            "mean_zero_spread_rate": round(statistics.fmean(zs), 6) if zs else None,
            "mean_reward": round(statistics.fmean(mr), 6) if mr else None,
        })
    rows.sort(key=lambda r: (r["scale"], r["experiment"]))
    out = AUDIT / "report_experiment_matrix.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MAT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[matrix] {len(rows)} 行 -> {out.name}")

    print()
    print(f"{'实验':<13}{'规模':<5}{'算法':<16}{'最佳官方':>10}{'@':<12}{'榜分':>9}{'有效组':>9}{'全对组':>9}")
    for r in rows:
        kg = f"{r['kaggle_best']:.5f}" if r["kaggle_best"] else "—"
        gr = f"{r['mean_grad_group_rate']:.3f}" if r["mean_grad_group_rate"] else "—"
        zs = f"{r['mean_zero_spread_rate']:.3f}" if r["mean_zero_spread_rate"] else "—"
        print(f"{r['experiment']:<13}{r['scale']:<5}{r['algo']:<16}{r['best_official']:>10}"
              f"{r['best_at']:<12}{kg:>9}{gr:>9}{zs:>9}")
    return rows


OPT_COLS = ["experiment", "arm", "granularity", "bucket", "n",
            "loss", "task_loss_full", "task_loss_vision", "task_loss_text", "task_loss_audio",
            "gradient_norm", "grad_norm_vision", "grad_norm_language", "grad_norm_audio",
            "learning_rate",
            "policy_loss", "value_loss", "kl", "clip_fraction",
            "policy_grad_norm_sum", "value_grad_norm_sum"]

OPT_METRICS = OPT_COLS[5:]


def build_optimizer_stage():
    """优化侧（loss / KL / 梯度范数）按阶段与按 epoch 的聚合。只有 7B-SFT 与 3B-PPO 有原料。"""
    rows_in = list(csv.DictReader(open(AUDIT / "curves_optimizer_side.csv", encoding="utf-8")))
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows_in:
        m = r["metric"]
        if m not in OPT_METRICS:
            continue
        v = float(r["value"])
        pct = float(r["progress_pct"])
        acc[(r["experiment"], r["arm"], "progress5", bucket5(pct))][m].append(v)

    # SFT 的 epoch 桶按 step 区间硬切（546 步/epoch，与图里的分隔线一致）。
    # 不用日志里的 epoch 字段：它在 1638 步处会多出一个边界值，切出第 4 个空桶。
    for r in rows_in:
        if r["experiment"] != "7B-SFT" or r["metric"] not in OPT_METRICS:
            continue
        st = int(float(r["unit_value"]))
        e = 3 if st > 1092 else (2 if st > 546 else 1)
        acc[("7B-SFT", "sft7b", "epoch", f"epoch {e}")][r["metric"]].append(float(r["value"]))

    rows = []
    for (exp, arm, gran, bucket) in sorted(acc):
        d = acc[(exp, arm, gran, bucket)]
        rec = {"experiment": exp, "arm": arm, "granularity": gran, "bucket": bucket,
               "n": len(next(iter(d.values())))}
        for m in OPT_METRICS:
            v = d.get(m)
            rec[m] = round(statistics.fmean(v), 8) if v else None
        rows.append(rec)
    out = AUDIT / "report_optimizer_stage.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OPT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[optimizer-stage] {len(rows)} 行 -> {out.name}")

    for r in rows:
        if r["granularity"] == "epoch":
            print(f"  7B-SFT {r['bucket']}: loss={r['loss']:.5f} vision={r['task_loss_vision']} "
                  f"text={r['task_loss_text']} audio={r['task_loss_audio']} grad={r['gradient_norm']:.4f}")
    return rows


if __name__ == "__main__":
    build_subset_table()
    print()
    build_stage_summary()
    print()
    build_optimizer_stage()
    print()
    build_matrix()
