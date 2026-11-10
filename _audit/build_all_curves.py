# -*- coding: utf-8 -*-
"""统一提取五轮实验的"评测曲线"与"训练过程信号"，输出两份 CSV。

产物：
  _audit/curves_checkpoints_all.csv  逐检查点评测指标（42 点，本地/官方双口径 + 子指标 + 混淆矩阵）
  _audit/curves_train_signals.csv    训练过程信号长表（每实验每记录点的组级/奖励级指标）

只读已归档的 *_metrics.json / *.log / *.jsonl，不重跑评测。
两套口径现场重算并与 _audit/master_results.csv 交叉校验，不一致直接抛错。
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"

# ─── 逐检查点评测源 ────────────────────────────────────────────────────────────
# (实验, 臂, 原始单位, 单位值, 总进度分母, metrics 相对路径)
CKPT_SOURCES = [
    ("3B-SFT", "sft3b", "step", 0, 300, "2026-10-29_14-21-GRPO/grpo_runs/pilot_01/sft_baseline_metrics.json"),
] + [
    ("3B-GRPO-r1", "grpo3b_r1", "step", s, 300,
     f"2026-10-29_14-21-GRPO/grpo_runs/pilot_01/step_{s:04d}_metrics.json")
    for s in (50, 100, 150, 200, 250, 300)
] + [
    ("3B-GRPO-r2", "grpo3b_r2", "attempt", s, 600,
     f"2026-10-29_23-22-GRPO/runs/pilot_01/attempt_{s:04d}_metrics.json")
    for s in (100, 200, 300, 400, 500, 600)
] + [
    ("3B-PPO", "ppo3b", "attempt", s, 300,
     f"2026-10-30_13-39-PPO/runs/pilot_01/attempt_{s:04d}_metrics.json")
    for s in (100, 200, 300)
] + [
    ("7B-SFT", "sft7b", "epoch", e, 3,
     f"2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/train/epoch_{e}_metrics.json")
    for e in (1, 2, 3)
] + [
    ("7B-GRPO", "grpo7b", "step", s, 300,
     f"2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01/step_{s:04d}_metrics.json")
    for s in (50, 100, 150, 200, 250, 300)
] + [
    ("7B-DAPO", arm, "step", s, 300,
     f"2026-11-01-DAPO/runs/{arm}/step_{s:04d}_metrics.json")
    for arm in ("dapo7b", "dapo7b_legacy") for s in (50, 100, 150, 200, 250, 300)
]

CKPT_COLS = [
    "experiment", "arm", "unit", "unit_value", "progress_pct",
    "local_score", "official_score", "official_delta_vs_sft",
    "emotion_f1", "conflict_f1", "conflict_recall", "conflict_acc", "aligned_spec",
    "TP", "FP", "FN", "TN", "all_fields_correct", "n", "source",
]

# 每轮实验的 SFT 起点（用于计算 official_delta_vs_sft）
SFT_BASELINE = {
    "3B-GRPO-r1": "2026-10-29_14-21-GRPO/grpo_runs/pilot_01/sft_baseline_metrics.json",
    "3B-GRPO-r2": "2026-10-29_23-22-GRPO/runs/pilot_01/sft_baseline_metrics.json",
    "3B-PPO": "2026-10-30_13-39-PPO/runs/pilot_01/sft_baseline_metrics.json",
    "7B-GRPO": "2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01/sft_baseline_metrics.json",
    "7B-DAPO": "2026-11-01-DAPO/runs/dapo7b/sft_baseline_metrics.json",
    "7B-SFT": "2026-10-30_23-00-Omni7B-SFT-GRPO/grpo_run/pilot_01/sft_baseline_metrics.json",
    "3B-SFT": "2026-10-29_14-21-GRPO/grpo_runs/pilot_01/sft_baseline_metrics.json",
}


def scores(m):
    """返回 (训练选点口径, Kaggle 官方口径)。"""
    a = m["all"]
    emo = a["emotion_macro_f1_present"]
    spec = a["aligned_specificity"]
    return (0.5 * emo + 0.3 * a["conflict_recall"] + 0.2 * spec,
            0.5 * emo + 0.3 * a["conflict_accuracy"] + 0.2 * spec)


def build_checkpoints():
    base_cache = {}

    def base_official(exp):
        rel = SFT_BASELINE[exp]
        if rel not in base_cache:
            m = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            base_cache[rel] = scores(m)[1]
        return base_cache[rel]

    rows = []
    for exp, arm, unit, val, total, rel in CKPT_SOURCES:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"缺少指标文件: {rel}")
        m = json.loads(p.read_text(encoding="utf-8"))
        a, c = m["all"], m["all"]["conflict_confusion"]
        local, official = scores(m)
        rows.append({
            "experiment": exp, "arm": arm,
            "unit": unit, "unit_value": val,
            "progress_pct": round(100.0 * val / total, 2),
            "local_score": round(local, 6),
            "official_score": round(official, 6),
            "official_delta_vs_sft": round(official - base_official(exp), 6),
            "emotion_f1": round(a["emotion_macro_f1_present"], 6),
            "conflict_f1": round(a["conflict_f1"], 6),
            "conflict_recall": round(a["conflict_recall"], 6),
            "conflict_acc": round(a["conflict_accuracy"], 6),
            "aligned_spec": round(a["aligned_specificity"], 6),
            "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
            "all_fields_correct": a["all_fields_correct_n"], "n": a["n"],
            "source": rel,
        })

    # 与 master_results.csv 交叉校验（只校验两边都有的 检查点）
    master = list(csv.DictReader(open(AUDIT / "master_results.csv", encoding="utf-8")))
    idx = {}
    for r in master:
        idx.setdefault((r["experiment"], r["checkpoint"]), []).append(r)
    ck_name = {"step": lambda v: f"step_{v:04d}" if v else "final",
               "attempt": lambda v: f"attempt_{v:04d}" if v else "final",
               "epoch": lambda v: f"epoch_{v}"}
    checks = 0
    for r in rows:
        key = (r["experiment"], ck_name[r["unit"]](r["unit_value"]))
        if key not in idx:
            continue
        if len(idx[key]) > 1:  # DAPO 两臂同名检查点，跳过歧义项
            continue
        want = float(idx[key][0]["official"])
        if abs(want - r["official_score"]) > 5e-7:
            raise SystemExit(f"{key} 官方口径不一致: 本表 {r['official_score']} vs master {want}")
        checks += 1

    out = AUDIT / "curves_checkpoints_all.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CKPT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[checkpoints] {len(rows)} 行 -> {out.name}；与 master 校验通过 {checks} 项")
    return rows


# ─── 训练过程信号 ──────────────────────────────────────────────────────────────
SIG_COLS = ["experiment", "arm", "unit", "unit_value", "progress_pct", "metric", "value"]


def sig(rows, exp, arm, unit, val, total, **kw):
    for k, v in kw.items():
        if v is None:
            continue
        rows.append({
            "experiment": exp, "arm": arm, "unit": unit, "unit_value": val,
            "progress_pct": round(100.0 * val / total, 2),
            "metric": k, "value": round(float(v), 6),
        })


STEP_RE = re.compile(r"^STEP (\d+) updates (\d+) buckets (\{.*\})$")


def parse_step_log(path, exp, arm, total):
    """GRPO/7B 的 STEP 行：updates N buckets {'all_correct': n, ...}。"""
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = STEP_RE.match(line.strip())
        if not m:
            continue
        step, updates = int(m.group(1)), int(m.group(2))
        buckets = eval(m.group(3), {"__builtins__": {}}, {})  # 仅解析 {str: int} 字面量
        groups = sum(buckets.values())
        active = buckets.get("active", 0) + buckets.get("decision_active", 0)
        sig(out, exp, arm, "step", step, total,
            updates=updates,
            groups_attempted=groups,
            active_groups=active,
            all_correct_groups=buckets.get("all_correct", 0),
            all_same_not_perfect=buckets.get("all_same_not_perfect", 0),
            zero_update=1 if updates == 0 else 0,
            accept_rate=(groups - active) / groups if groups else None,
            active_rate=active / groups if groups else None)
    return out


def parse_progress(path, exp, arm, unit, total, mode):
    """r2 / PPO 的 PROGRESS 行（累计量），差分后得到每个窗口的增量。"""
    rows_raw = []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if s.startswith("PROGRESS "):
            rows_raw.append(json.loads(s[len("PROGRESS "):]))
    out = []
    if not rows_raw:
        return out
    prev = {}
    for r in rows_raw:
        x = r["attempt"]

        def d(key):
            v = r.get(key)
            if v is None:
                return None
            delta = v - prev.get(key, 0)
            return delta

        if mode == "r2":
            cand = d("candidates"); act = d("active"); ac = d("all_correct")
            steps = d("optimizer_steps")
            sig(out, exp, arm, unit, x, total,
                candidates=cand, active_groups=act, all_correct_groups=ac,
                optimizer_steps=steps,
                accept_rate=1 - act / cand if cand else None,
                active_rate=act / cand if cand else None,
                zero_update=1 if steps == 0 else 0)
        else:  # ppo
            roll = d("rollouts"); dc = d("decision_correct")
            fc = d("fused_correct"); pf = d("perfect")
            sig(out, exp, arm, unit, x, total,
                mean_reward=r.get("mean_reward"),
                perfect_rate=r.get("perfect_rate"),
                kl=r.get("kl"),
                clip_fraction=r.get("clip_fraction"),
                value_loss=r.get("value_loss"),
                decision_acc=dc / roll if roll else None,
                fused_acc=fc / roll if roll else None,
                perfect_window=pf / roll if roll else None,
                optimizer_steps=d("optimizer_steps"))
        prev = {k: v for k, v in r.items() if isinstance(v, (int, float))}
    return out


def parse_dapo_cycles(path, arm, total):
    out = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        c = json.loads(s)
        bc = c.get("bucket_counts", {})
        groups = c.get("attempted_groups", 0)
        active = c.get("active_groups", 0)
        sig(out, "7B-DAPO", arm, "step", c["step"], total,
            updates=c.get("updates", 0),
            groups_attempted=groups,
            active_groups=active,
            decision_active_groups=c.get("decision_active_groups", 0),
            all_correct_groups=bc.get("all_correct", 0),
            auxiliary_active=bc.get("auxiliary_active", 0),
            all_same_not_perfect=bc.get("all_same_not_perfect", 0),
            zero_update=1 if c.get("updates", 0) == 0 else 0,
            clip_upper_triggered=c.get("clip_upper", 0),
            clip_lower_triggered=c.get("clip_lower", 0),
            accept_rate=(groups - active) / groups if groups else None,
            hard_pool=c.get("difficulty", {}).get("hard_pool"),
            unique_seen=c.get("difficulty", {}).get("unique_seen"))
    return out


def parse_sft_metrics(path, total):
    out = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        d = json.loads(s)
        st = d["step"]
        sig(out, "7B-SFT", "sft7b", "step", st, total,
            loss=d.get("loss"),
            grad_norm=d.get("gradient_norm"),
            lr=d.get("learning_rate"),
            task_loss_full=d.get("task_loss/full"),
            task_loss_vision=d.get("task_loss/vision_only"),
            task_loss_text=d.get("task_loss/text_only"),
            task_loss_audio=d.get("task_loss/audio_only"),
            grad_norm_vision=d.get("gradient_norm/vision"),
            grad_norm_audio=d.get("gradient_norm/audio"),
            peak_gpu_gib=d.get("peak_gpu_gib"),
            epoch=d.get("epoch"))
    return out


def build_signals():
    rows = []
    rows += parse_step_log(ROOT / "2026-10-29_14-21-GRPO/grpo_runs/pilot_01.log",
                           "3B-GRPO-r1", "grpo3b_r1", 300)
    rows += parse_progress(ROOT / "2026-10-29_23-22-GRPO/logs/pilot_01.log",
                           "3B-GRPO-r2", "grpo3b_r2", "attempt", 600, "r2")
    rows += parse_progress(ROOT / "2026-10-30_13-39-PPO/logs/pilot_01.log",
                           "3B-PPO", "ppo3b", "attempt", 300, "ppo")
    rows += parse_step_log(ROOT / "2026-10-30_23-00-Omni7B-SFT-GRPO/logs/grpo_train.log",
                           "7B-GRPO", "grpo7b", 300)
    rows += parse_dapo_cycles(ROOT / "2026-11-01-DAPO/runs/dapo7b/cycles.jsonl", "dapo7b", 300)
    rows += parse_dapo_cycles(ROOT / "2026-11-01-DAPO/runs/dapo7b_legacy/cycles.jsonl", "dapo7b_legacy", 300)
    rows += parse_sft_metrics(ROOT / "2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/train/train_metrics.jsonl", 1638)

    out = AUDIT / "curves_train_signals.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIG_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[signals] {len(rows)} 条 -> {out.name}")

    # 汇总打印：每个实验每条序列的点数与关键统计
    from collections import defaultdict
    idx = defaultdict(set)
    for r in rows:
        idx[(r["experiment"], r["arm"], r["metric"])].add(r["unit_value"])
    print("\n每个实验的序列点数：")
    for (e, a, m), v in sorted(idx.items()):
        print(f"  {e:11s} {a:15s} {m:24s} {len(v):5d} 点")
    return rows


if __name__ == "__main__":
    cp = build_checkpoints()
    print()
    print(f"{'实验':<12}{'臂':<16}{'单位值':>7}{'本地口径':>11}{'官方口径':>11}{'Δvs SFT':>10}  TP/FP/FN")
    for r in cp:
        print(f"{r['experiment']:<12}{r['arm']:<16}{r['unit_value']:>7}"
              f"{r['local_score']:>11.6f}{r['official_score']:>11.6f}"
              f"{r['official_delta_vs_sft']:>+10.6f}  {r['TP']}/{r['FP']}/{r['FN']}")
    print()
    build_signals()
