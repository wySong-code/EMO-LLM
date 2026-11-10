"""Recompute every comparison in this experiment from the per-sample predictions.

Why this exists: the numbers quoted in the write-up must be reproducible from
artefacts that live next to the code, not from a one-off shell command. Every
metric here goes through `score_omni_modality.score()`, the same function the
training loop uses, so nothing is re-derived with a home-made definition.

Run:  python code/analyze_results.py [--json out.json]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code" / "sft"))
from score_omni_modality import score  # noqa: E402

PROJ3 = ROOT.parent / "2026-10-29_14-21-GRPO"

LABELS = ("positive", "negative", "neutral")


# --------------------------------------------------------------------------- io
def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_id(rows):
    return {r["sample_id"]: r for r in rows}


def safe(path):
    return path if Path(path).exists() else None


# ------------------------------------------------------------------ statistics
def headline(g):
    return {
        "n": g["n"],
        "local_score": g["local_score_present"],
        "macro_f1": g["emotion_macro_f1_present"],
        "acc": g["emotion_accuracy"],
        "vision_acc": g["fields"]["vision_emotion"]["accuracy"],
        "text_acc": g["fields"].get("text_emotion", {}).get("accuracy"),
        "audio_acc": g["fields"].get("audio_emotion", {}).get("accuracy"),
        "recall": g["conflict_recall"],
        "spec": g["aligned_specificity"],
        "cf1": g["conflict_f1"],
        "conf": g["conflict_confusion"],
        "fmt": g["format_valid_rate"],
        "tokens_hit": g["token_limit_hits"],
    }


def per_class_prf(rows, field="pred_emotion"):
    """Precision/recall/F1 per label, computed over rows whose gold has `field`."""
    sub = [r for r in rows if field in r["gold"]]
    gold = [r["gold"][field] for r in sub]
    pred = [r["prediction"].get(field, "invalid") for r in sub]
    out = {}
    for lab in LABELS:
        tp = sum(a == lab and b == lab for a, b in zip(gold, pred))
        fp = sum(a != lab and b == lab for a, b in zip(gold, pred))
        fn = sum(a == lab and b != lab for a, b in zip(gold, pred))
        out[lab] = {
            "support": sum(a == lab for a in gold),
            "pred": sum(b == lab for b in pred),
            "p": tp / (tp + fp) if tp + fp else None,
            "r": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    return out


def conflict_by_source(rows):
    out = {}
    for src in ("original", "generated"):
        sub = [r for r in rows if r["source"] == src]
        if not sub:
            continue
        c = score(sub)["all"]["conflict_confusion"]
        n_conf = sum(r["gold"]["is_conflict"] == "yes" for r in sub)
        n_al = len(sub) - n_conf
        out[src] = {
            "n": len(sub),
            "n_conflict": n_conf,
            "conf": c,
            "recall": c["TP"] / n_conf if n_conf else None,
            "spec": c["TN"] / n_al if n_al else None,
        }
    return out


def transition(a, b, field="pred_emotion"):
    """2x2 of correct/wrong for model a vs model b on the shared sample ids."""
    ids = [k for k in a if k in b]
    cell = Counter()
    for k in ids:
        ca = a[k]["gold"][field] == a[k]["prediction"].get(field)
        cb = b[k]["gold"][field] == b[k]["prediction"].get(field)
        cell[(ca, cb)] += 1
    return {
        "n": len(ids),
        "both_right": cell[(True, True)],
        "a_only": cell[(True, False)],
        "b_only": cell[(False, True)],
        "both_wrong": cell[(False, False)],
    }


def bootstrap(a, b, reps=1000, seed=42):
    """Paired bootstrap over sample indices; both models see the same resample."""
    try:
        import numpy as np
    except ImportError:
        return {}
    ids = sorted(k for k in a if k in b)
    rows_a = [a[k] for k in ids]
    rows_b = [b[k] for k in ids]
    keys = ("local_score_present", "emotion_macro_f1_present", "emotion_accuracy",
            "conflict_recall", "aligned_specificity", "conflict_f1")
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    n = len(ids)

    def flat(g):
        # 把嵌套字段摊平，vision 准确率不在 group() 的一层键里
        d = {k: g.get(k) for k in keys}
        d["vision_accuracy"] = g["fields"]["vision_emotion"]["accuracy"]
        return d

    for _ in range(reps):
        idx = rng.integers(0, n, n)
        sa = flat(score([rows_a[i] for i in idx])["all"])
        sb = flat(score([rows_b[i] for i in idx])["all"])
        if sa["local_score_present"] is None or sb["local_score_present"] is None:
            continue
        for k in list(keys) + ["vision_accuracy"]:
            draws[k].append(sb[k] - sa[k])
    out = {}
    for k, vals in draws.items():
        if not vals:
            continue
        v = np.array(vals)
        out[k] = {
            "diff": float(np.mean(v)),
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
            "p_gt0": float((v > 0).mean()),
        }
    return out


def group_reward_stats(path):
    """How often a sampled group carries any learning signal at all."""
    rows = load_rows(path)
    zero_var = 0
    buckets = Counter()
    stds = []
    for r in rows:
        rew = r["rewards"]
        buckets[r["bucket"]] += 1
        spread = max(rew) - min(rew)
        stds.append(spread)
        if spread < 1e-9:
            zero_var += 1
    return {
        "groups": len(rows),
        "zero_variance_groups": zero_var,
        "active_groups": len(rows) - zero_var,
        "active_rate": (len(rows) - zero_var) / len(rows) if rows else None,
        "buckets": dict(buckets),
        "mean_spread": sum(stds) / len(stds) if stds else None,
    }


def step_updates(path):
    """Count gradient updates per step from the trainer's own STEP lines."""
    if not path or not Path(path).exists():
        return {}
    pat = re.compile(r"STEP (\d+) updates (\d+) buckets (.+)$")
    steps = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            steps.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    if not steps:
        return {}
    buckets = Counter()
    for _, _, desc in steps:
        for name, cnt in re.findall(r"'(\w+)': (\d+)", desc):
            buckets[name] += int(cnt)
    return {
        "steps": len(steps),
        "steps_with_update": sum(1 for _, k, _ in steps if k > 0),
        "total_updates": sum(k for _, k, _ in steps),
        "buckets": dict(buckets),
    }


def token_stats(rows):
    tok = [r.get("output_tokens") or 0 for r in rows]
    tok.sort()
    n = len(tok)
    return {
        "mean": sum(tok) / n if n else None,
        "p50": tok[n // 2] if n else None,
        "p99": tok[int(n * 0.99)] if n else None,
        "max": tok[-1] if n else None,
        "limit_hits": sum(bool(r.get("token_limit_hit")) for r in rows),
        "parse_errors": sum(r.get("parse_error") is not None for r in rows),
    }


def consistency(rows):
    """Does the model keep pred_emotion consistent with its own is_conflict?"""
    valid = [r for r in rows if r.get("parse_error") is None]
    same = 0
    declared_conflict = 0
    follow_vision_when_conflict = 0
    follow_other_when_conflict = 0
    for r in valid:
        p = r["prediction"]
        other = p.get("text_emotion" if r["protocol"] == "VT" else "audio_emotion")
        if other is None:
            continue
        is_same = p["vision_emotion"] == other
        if is_same == (p["is_conflict"] == "yes"):
            same += 1
        if p["is_conflict"] == "yes":
            declared_conflict += 1
            if p["pred_emotion"] == p["vision_emotion"]:
                follow_vision_when_conflict += 1
            elif p["pred_emotion"] == other:
                follow_other_when_conflict += 1
    return {
        "self_consistent_rate": same / len(valid) if valid else None,
        "declared_conflict": declared_conflict,
        "when_conflict_follow_vision": follow_vision_when_conflict,
        "when_conflict_follow_other": follow_other_when_conflict,
        "when_conflict_third_choice": declared_conflict - follow_vision_when_conflict
        - follow_other_when_conflict,
    }


# ---------------------------------------------------------------------- report
def fmt(x, nd=4):
    return "—" if x is None else ("%.*f" % (nd, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "reports" / "analysis_metrics.json"))
    args = ap.parse_args()

    sft_dir = ROOT / "sft_run" / "train"
    grpo_dir = ROOT / "grpo_run" / "pilot_01"
    p3_dir = PROJ3 / "grpo_runs" / "pilot_01"

    epochs = {}
    for e in (1, 2, 3):
        p = safe(sft_dir / ("epoch_%d_predictions.jsonl" % e))
        if p:
            epochs[e] = load_rows(p)

    grpo_states = {}
    base7 = safe(grpo_dir / "sft_baseline_predictions.jsonl")
    if base7:
        grpo_states["7B SFT(baseline)"] = load_rows(base7)
    for step in (50, 100, 150, 200, 250, 300):
        p = safe(grpo_dir / ("step_%04d_predictions.jsonl" % step))
        if p:
            grpo_states["7B GRPO step%d" % step] = load_rows(p)

    base3 = safe(p3_dir / "sft_baseline_predictions.jsonl")
    step3 = safe(p3_dir / "step_0050_predictions.jsonl")
    rows3 = load_rows(base3) if base3 else None
    rows3_50 = load_rows(step3) if step3 else None

    out = {"generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z"}

    # ---- 1. SFT per-epoch ----
    print("=" * 78)
    print("1. 7B SFT 逐轮（666 条验证集，贪心解码，同一套 scoring）")
    print("=" * 78)
    print("%-8s %8s %8s %8s %8s %8s %8s" % ("epoch", "score", "macroF1", "acc",
                                            "vision", "recall", "spec"))
    out["sft_epochs"] = {}
    for e, rows in epochs.items():
        h = headline(score(rows)["all"])
        out["sft_epochs"][e] = h
        print("%-8d %8s %8s %8s %8s %8s %8s" % (
            e, fmt(h["local_score"]), fmt(h["macro_f1"]), fmt(h["acc"]),
            fmt(h["vision_acc"]), fmt(h["recall"]), fmt(h["spec"])))

    # ---- 2. headline table over all states ----
    states = {}
    for e, rows in epochs.items():
        states["7B SFT ep%d" % e] = rows
    states.update(grpo_states)
    if rows3:
        states["3B SFT"] = rows3
    if rows3_50:
        states["3B GRPO step50"] = rows3_50

    print()
    print("=" * 78)
    print("2. 全状态主表")
    print("=" * 78)
    hdr = "%-18s %7s %7s %7s %7s %7s %7s %6s"
    print(hdr % ("state", "score", "macroF1", "vision", "recall", "spec", "cF1", "fmt"))
    out["states"] = {}
    for name, rows in states.items():
        h = headline(score(rows)["all"])
        out["states"][name] = h
        print(hdr % (name, fmt(h["local_score"], 3), fmt(h["macro_f1"], 3),
                     fmt(h["vision_acc"], 3), fmt(h["recall"], 3),
                     fmt(h["spec"], 3), fmt(h["cf1"], 3), fmt(h["fmt"], 2)))

    # ---- 3. protocol split for the two headline models ----
    print()
    print("=" * 78)
    print("3. 分协议（VT 画面+文字 / VA 画面+音频）")
    print("=" * 78)
    out["protocol"] = {}
    for name in ("3B SFT", "7B SFT ep3", "7B GRPO step50"):
        if name not in states:
            continue
        s = score(states[name])
        out["protocol"][name] = {p: headline(s[p]) for p in ("VT", "VA") if s[p]["n"]}
        for p in ("VT", "VA"):
            if not s[p]["n"]:
                continue
            h = out["protocol"][name][p]
            print("%-18s %-3s n=%3d score=%s macroF1=%s recall=%s spec=%s" % (
                name, p, h["n"], fmt(h["local_score"], 3), fmt(h["macro_f1"], 3),
                fmt(h["recall"], 3), fmt(h["spec"], 3)))

    # ---- 4. conflict by source ----
    print()
    print("=" * 78)
    print("4. 冲突判定，按数据来源拆（original=真实视频，generated=合成冲突）")
    print("=" * 78)
    out["conflict_by_source"] = {}
    for name in ("3B SFT", "7B SFT ep3", "3B GRPO step50", "7B GRPO step50"):
        if name not in states:
            continue
        d = conflict_by_source(states[name])
        out["conflict_by_source"][name] = d
        for src, v in d.items():
            print("%-18s %-10s n=%3d 冲突%3d  TP=%3d FP=%3d FN=%3d TN=%3d  recall=%s spec=%s" % (
                name, src, v["n"], v["n_conflict"], v["conf"]["TP"], v["conf"]["FP"],
                v["conf"]["FN"], v["conf"]["TN"], fmt(v["recall"], 3), fmt(v["spec"], 3)))

    # ---- 5. per-class P/R/F1 ----
    print()
    print("=" * 78)
    print("5. 综合情感 pred_emotion 逐类别 P/R/F1")
    print("=" * 78)
    out["per_class"] = {}
    for name in ("3B SFT", "7B SFT ep3", "7B GRPO step50"):
        if name not in states:
            continue
        d = per_class_prf(states[name])
        out["per_class"][name] = d
        for lab in LABELS:
            v = d[lab]
            print("%-18s %-9s sup=%3d pred=%3d  P=%s R=%s F1=%s" % (
                name, lab, v["support"], v["pred"], fmt(v["p"], 3),
                fmt(v["r"], 3), fmt(v["f1"], 3)))

    # ---- 6. transitions ----
    print()
    print("=" * 78)
    print("6. 逐样本迁移（共享 sample_id 上 pred_emotion 对/错）")
    print("=" * 78)
    out["transition"] = {}
    pairs = [("3B SFT", "7B SFT ep3"), ("3B SFT", "7B GRPO step50"),
             ("7B SFT ep3", "7B GRPO step50"), ("7B GRPO step50", "7B GRPO step150")]
    for a, b in pairs:
        if a in states and b in states:
            t = transition(by_id(states[a]), by_id(states[b]))
            t["from"], t["to"] = a, b
            out["transition"]["%s -> %s" % (a, b)] = t
            print("%-34s n=%d 都对=%3d 仅前对=%2d 仅后对=%2d 都错=%3d" % (
                "%s -> %s" % (a, b), t["n"], t["both_right"], t["a_only"],
                t["b_only"], t["both_wrong"]))

    # ---- 7. bootstrap ----
    print()
    print("=" * 78)
    print("7. 配对 bootstrap（1000 次，seed 42，官方 score()）")
    print("=" * 78)
    out["bootstrap"] = {}
    bpairs = [("3B SFT", "7B SFT ep3"), ("7B SFT ep3", "7B GRPO step50"),
              ("3B SFT", "3B GRPO step50")]
    for a, b in bpairs:
        if a in states and b in states:
            r = bootstrap(by_id(states[a]), by_id(states[b]))
            if not r:
                continue
            out["bootstrap"]["%s -> %s" % (a, b)] = r
            print("[%s -> %s]" % (a, b))
            for k, v in r.items():
                print("  %-26s diff=%+.4f 95%%CI[%+.4f, %+.4f] P(>0)=%.3f" % (
                    k, v["diff"], v["lo"], v["hi"], v["p_gt0"]))

    # ---- 8. GRPO learning signal ----
    print()
    print("=" * 78)
    print("8. GRPO 的学习信号：奖励饱和了吗")
    print("=" * 78)
    out["grpo_signal"] = {
        "7B": group_reward_stats(grpo_dir / "rollouts.jsonl")
        if (grpo_dir / "rollouts.jsonl").exists() else {},
        "7B_steps": step_updates(ROOT / "logs" / "grpo_train.log"),
        "3B": group_reward_stats(p3_dir / "rollouts.jsonl")
        if (p3_dir / "rollouts.jsonl").exists() else {},
        "3B_steps": step_updates(PROJ3 / "grpo_runs" / "pilot_01.log"),
    }
    for tag in ("3B", "7B"):
        d = out["grpo_signal"][tag]
        s = out["grpo_signal"][tag + "_steps"]
        if d:
            print("%s  组数=%d  零方差组=%d  有效组=%d (%.1f%%)  平均奖励极差=%.3f  分桶=%s" % (
                tag, d["groups"], d["zero_variance_groups"], d["active_groups"],
                100 * d["active_rate"], d["mean_spread"], d["buckets"]))
        if s:
            print("     梯度更新：%d 步中 %d 步有更新，累计 %d 次，分桶=%s" % (
                s["steps"], s["steps_with_update"], s["total_updates"], s["buckets"]))

    # ---- 9. decode / engineering stats ----
    print()
    print("=" * 78)
    print("9. 解码与自洽性")
    print("=" * 78)
    out["decode"] = {}
    for name in ("3B SFT", "7B SFT ep3", "7B GRPO step50"):
        if name not in states:
            continue
        t = token_stats(states[name])
        c = consistency(states[name])
        out["decode"][name] = {"tokens": t, "consistency": c}
        print("%-18s out_tok mean=%.1f p50=%d p99=%d max=%d 触顶=%d 解析失败=%d 自洽=%.4f" % (
            name, t["mean"], t["p50"], t["p99"], t["max"], t["limit_hits"],
            t["parse_errors"], c["self_consistent_rate"]))
        print("%-18s 判冲突 %d 条：跟随画面 %d，跟随另一模态 %d，第三种 %d" % (
            "", c["declared_conflict"], c["when_conflict_follow_vision"],
            c["when_conflict_follow_other"], c["when_conflict_third_choice"]))

    # ---- 10. GRPO trajectory table ----
    print()
    print("=" * 78)
    print("10. GRPO 轨迹（7B 与 3B 同口径）")
    print("=" * 78)
    out["grpo_trajectory"] = {}
    for tag, pdir, base in (("7B", grpo_dir, base7), ("3B", p3_dir, base3)):
        if not pdir.exists():
            continue
        seq = [("base", load_rows(base))] if base else []
        for step in (50, 100, 150, 200, 250, 300):
            p = pdir / ("step_%04d_predictions.jsonl" % step)
            if p.exists():
                seq.append((str(step), load_rows(p)))
        out["grpo_trajectory"][tag] = []
        print("[%s]" % tag)
        print("  %-6s %7s %7s %7s %7s %7s" % ("step", "score", "macroF1", "recall",
                                              "spec", "cF1"))
        for step, rows in seq:
            h = headline(score(rows)["all"])
            out["grpo_trajectory"][tag].append({"step": step, **h})
            print("  %-6s %7s %7s %7s %7s %7s" % (
                step, fmt(h["local_score"], 4), fmt(h["macro_f1"], 4),
                fmt(h["recall"], 4), fmt(h["spec"], 4), fmt(h["cf1"], 4)))

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print()
    print("written", args.json)


if __name__ == "__main__":
    main()
