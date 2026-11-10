# -*- coding: utf-8 -*-
"""生成「分实验数据附录」：把每个实验/臂的阶段数据、评测数据、来源文件写成可直接引用的表格。

产物：
  _audit/report_stage_fine.csv    6 个实验/臂 × 10 段（十分位）的奖励侧+优化侧+组级指标
  _audit/report_traceability.csv  每个数据块 → 源文件 → 行数 → 生成脚本 → 复算命令
  docs/07_分实验数据附录.md        按实验分节，表格由本脚本渲染（不手抄数字）

只读已归档文件，不重跑模型。
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
DOCS = ROOT / "docs"

ARM_LABEL = {"sft3b": "3B-SFT", "grpo3b_r1": "3B-GRPO-r1", "grpo3b_r2": "3B-GRPO-r2",
             "ppo3b": "3B-PPO", "sft7b": "7B-SFT", "grpo7b": "7B-GRPO",
             "dapo7b": "7B-DAPO", "dapo7b_legacy": "7B-DAPO-legacy"}

# 实验/臂 → 归档目录、基座、算法、训练单元总数、原始日志与 rollout 文件
EXP_META = {
    "3B-SFT": dict(dir="2026-10-29_14-21-GRPO", base="Qwen2.5-Omni-3B", algo="SFT（前序阶段产物）",
                   unit="点", total="1 次评测", note="作为三轮 3B RL 的共同起点，只评过一次，没有训练过程记录"),
    "3B-GRPO-r1": dict(dir="2026-10-29_14-21-GRPO", base="Qwen2.5-Omni-3B + SFT adapter",
                       algo="GRPO（第一轮，基线配方）", unit="step", total=300,
                       log="grpo_runs/pilot_01.log",
                       roll="grpo_runs/pilot_01/rollouts.jsonl",
                       cyc="grpo_runs/pilot_01/cycles.jsonl"),
    "3B-GRPO-r2": dict(dir="2026-10-29_23-22-GRPO", base="Qwen2.5-Omni-3B + SFT adapter",
                       algo="GRPO（第二轮，attempt 级采样 + 难度配额）", unit="attempt", total=600,
                       log="logs/pilot_01.log", roll="runs/pilot_01/rollouts.jsonl",
                       cyc="runs/pilot_01/cycles.jsonl"),
    "3B-PPO": dict(dir="2026-10-30_13-39-PPO", base="Qwen2.5-Omni-3B + SFT adapter",
                   algo="PPO（actor-critic + value head）", unit="attempt", total=300,
                   log="logs/pilot_01.log", roll="runs/pilot_01/rollouts.jsonl",
                   cyc="runs/pilot_01/cycles.jsonl"),
    "7B-SFT": dict(dir="2026-10-30_23-00-Omni7B-SFT-GRPO", base="Qwen2.5-Omni-7B", algo="SFT（3 epoch）",
                   unit="epoch", total=3, log="logs/sft_train.log",
                   opt="sft_run/train/train_metrics.jsonl"),
    "7B-GRPO": dict(dir="2026-10-30_23-00-Omni7B-SFT-GRPO", base="Qwen2.5-Omni-7B + SFT adapter",
                    algo="GRPO（与 3B 同配方迁到 7B）", unit="step", total=300,
                    log="logs/grpo_train.log", roll="grpo_run/pilot_01/rollouts.jsonl",
                    cyc="grpo_run/pilot_01/cycles.jsonl"),
    "7B-DAPO": dict(dir="2026-11-01-DAPO", base="Qwen2.5-Omni-7B + SFT adapter",
                    algo="DAPO（四项补丁）", unit="step", total=300,
                    log="logs/dapo_train_dapo7b.log", roll="runs/dapo7b/rollouts.jsonl",
                    cyc="runs/dapo7b/cycles.jsonl"),
    "7B-DAPO-legacy": dict(dir="2026-11-01-DAPO", base="Qwen2.5-Omni-7B + SFT adapter",
                           algo="DAPO（仅 Dynamic Sampling 的对照臂）", unit="step", total=300,
                           log="logs/dapo_train_dapo7b_legacy.log",
                           roll="runs/dapo7b_legacy/rollouts.jsonl",
                           cyc="runs/dapo7b_legacy/cycles.jsonl"),
}

SECTION_ORDER = ["3B-SFT", "3B-GRPO-r1", "3B-GRPO-r2", "3B-PPO",
                 "7B-SFT", "7B-GRPO", "7B-DAPO", "7B-DAPO-legacy"]

FINE_COLS = ["experiment", "arm", "unit", "stage", "progress_from", "progress_to",
             "n_units", "n_groups", "n_candidates",
             "mean_reward", "std_reward", "within_group_std", "zero_spread_rate",
             "grad_groups_n", "grad_group_rate", "grad_conflict_share", "conflict_group_share",
             "aligned_group_share", "top_reward_share", "decision_acc",
             "accept_rate", "all_correct_rate", "optimizer_steps",
             "loss", "policy_loss", "value_loss", "kl", "grad_norm", "clip_fraction",
             "task_loss_vision", "task_loss_text", "task_loss_audio", "lr",
             "source"]


def decile(pct):
    """按训练进度划十分位：1..10。pct 恰好 0 归入第 1 段。"""
    i = int(pct // 10)
    return min(i + 1, 10)


def num(v):
    return float(v) if v not in (None, "") else None


def mean(vals):
    if not vals:
        return None
    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 6) if vals else None


def load(name):
    with open(AUDIT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_fine():
    rew = load("curves_reward_per_step.csv")
    sig = load("curves_train_signals.csv")
    opt = load("curves_optimizer_side.csv")

    rby = defaultdict(list)
    for r in rew:
        rby[(r["experiment"], r["arm"], decile(float(r["progress_pct"])))].append(r)
    sby = defaultdict(lambda: defaultdict(list))
    for r in sig:
        sby[(r["experiment"], r["arm"], decile(float(r["progress_pct"])))][r["metric"]].append(float(r["value"]))
    oby = defaultdict(lambda: defaultdict(list))
    for r in opt:
        oby[(r["experiment"], r["arm"], decile(float(r["progress_pct"])))][r["metric"]].append(float(r["value"]))

    rows = []
    for (exp, arm, d) in sorted(rby, key=lambda k: (SECTION_ORDER.index(ARM_LABEL[k[1]]), k[2])):
        rs = rby[(exp, arm, d)]
        meta = EXP_META[ARM_LABEL[arm]]
        is_ppo = arm == "ppo3b"
        o = oby.get((exp, arm, d), {})
        s = sby.get((exp, arm, d), {})

        gsum = sum(int(r["grad_groups"]) for r in rs if r["grad_groups"] not in (None, ""))
        gtot = sum(int(r["groups"]) for r in rs if r["groups"] not in (None, ""))
        wsum = wval = 0.0
        for r in rs:
            g = int(r["grad_groups"]) if r["grad_groups"] not in (None, "") else 0
            sh = num(r["grad_conflict_share"])
            if g and sh is not None:
                wsum += g
                wval += g * sh

        # 该段覆盖的原始单元范围，便于回溯到具体行
        units = sorted(int(float(r["unit_value"])) for r in rs)
        rows.append({
            "experiment": ARM_LABEL[arm], "arm": arm, "unit": meta["unit"],
            "stage": f"{d}/10",
            "progress_from": round(100.0 * (d - 1) / 10, 1),
            "progress_to": round(100.0 * d / 10, 1),
            "n_units": len({r["unit_value"] for r in rs}),
            "n_groups": gtot or None,
            "n_candidates": sum(int(r["candidates"]) for r in rs if r["candidates"] not in (None, "")),
            "mean_reward": mean([num(r["mean_reward"]) for r in rs]),
            "std_reward": mean([num(r["std_reward"]) for r in rs]),
            "within_group_std": mean([num(r["within_group_std"]) for r in rs]),
            "zero_spread_rate": None if is_ppo else mean([num(r["zero_spread_group_rate"]) for r in rs]),
            "grad_groups_n": None if is_ppo else gsum,
            "grad_group_rate": None if is_ppo else (round(gsum / gtot, 6) if gtot else None),
            "grad_conflict_share": None if is_ppo else (round(wval / wsum, 6) if wsum else None),
            "conflict_group_share": mean([num(r["conflict_group_share"]) for r in rs]),
            "aligned_group_share": mean([num(r["aligned_group_share"]) for r in rs]),
            "top_reward_share": None if is_ppo else mean([num(r["top_reward_share"]) for r in rs]),
            "decision_acc": mean([num(r["decision_acc"]) for r in rs]),
            "accept_rate": mean(s.get("accept_rate")),
            "all_correct_rate": (round(sum(s["all_correct_groups"]) / sum(s["groups_attempted"]), 6)
                                 if s.get("all_correct_groups") and s.get("groups_attempted")
                                 and sum(s["groups_attempted"]) else None),
            "optimizer_steps": mean(s.get("optimizer_steps")),
            "loss": mean(o.get("loss")),
            "policy_loss": mean(o.get("policy_loss")),
            "value_loss": mean(o.get("value_loss")),
            "kl": mean(o.get("kl")),
            "grad_norm": mean(o.get("grad_norm") or o.get("policy_grad_norm_sum")),
            "clip_fraction": mean(o.get("clip_fraction")),
            "source": (f"{meta['dir']}/  |  {meta['unit']} {units[0]}–{units[-1]}"),
        })

    rows = rows + build_sft_fine()   # 7B-SFT 只有优化侧，单独补 10 段
    rows.sort(key=lambda r: (SECTION_ORDER.index(r["experiment"]), int(r["stage"].split("/")[0])))

    out = AUDIT / "report_stage_fine.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FINE_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[fine] {len(rows)} 行（{len({r['arm'] for r in rows})} 臂，最多 10 段/臂） -> {out.name}")
    return rows


def build_sft_fine():
    """7B-SFT 没有 rollouts，奖励侧为空；但它是唯一有逐步 loss 的实验，按十分位单独聚一段。"""
    opt = load("curves_optimizer_side.csv")
    by = defaultdict(lambda: defaultdict(list))
    for r in opt:
        if r["experiment"] != "7B-SFT":
            continue
        by[decile(float(r["progress_pct"]))][r["metric"]].append(float(r["value"]))
    rows = []
    for d in sorted(by):
        o = by[d]
        steps = len(o.get("loss", []))
        rows.append({
            "experiment": "7B-SFT", "arm": "sft7b", "unit": "step", "stage": f"{d}/10",
            "progress_from": round(100.0 * (d - 1) / 10, 1),
            "progress_to": round(100.0 * d / 10, 1),
            "n_units": steps, "n_groups": None, "n_candidates": None,
            "mean_reward": None, "std_reward": None, "within_group_std": None,
            "zero_spread_rate": None, "grad_group_rate": None, "grad_conflict_share": None,
            "conflict_group_share": None, "aligned_group_share": None, "top_reward_share": None,
            "decision_acc": None, "accept_rate": None, "all_correct_rate": None,
            "optimizer_steps": None,
            "loss": mean(o.get("loss")), "policy_loss": None, "value_loss": None,
            "kl": mean(o.get("kl")), "grad_norm": mean(o.get("gradient_norm")),
            "clip_fraction": None,
            "task_loss_vision": mean(o.get("task_loss_vision")),
            "task_loss_text": mean(o.get("task_loss_text")),
            "task_loss_audio": mean(o.get("task_loss_audio")),
            "lr": mean(o.get("learning_rate")),
            "source": f"2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/train/train_metrics.jsonl  |  step 区间见 progress",
        })
    return rows


def build_traceability():
    """每个数据块 -> 源文件 -> 规模 -> 生成脚本 -> 复算命令。"""
    blocks = [
        ("评测：37 检查点全指标", "_audit/curves_checkpoints_all.csv", "37 行",
         "build_all_curves.py", "python _audit/build_all_curves.py",
         "37 个 *_metrics.json（各实验 runs/ 下的评测输出）"),
        ("评测：子集拆解", "_audit/report_metrics_by_subset.csv", "185 行",
         "build_report_pack.py", "python _audit/build_report_pack.py",
         "同上，取 metrics.json 里的 all/VT/VA/original/generated 五个子字典"),
        ("训练信号：逐步组级", "_audit/curves_train_signals.csv", "31,175 行",
         "build_all_curves.py", "python _audit/build_all_curves.py",
         "各实验 cycles.jsonl + 日志 STEP/PROGRESS 行"),
        ("奖励侧：逐步", "_audit/curves_reward_per_step.csv", "2,100 行",
         "build_reward_curves.py", "python _audit/build_reward_curves.py",
         "各实验 rollouts.jsonl（每组 8 候选的 rewards 数组离线重算）"),
        ("优化侧：逐步", "_audit/curves_optimizer_side.csv", "19,985 行",
         "build_optimizer_curves.py", "python _audit/build_optimizer_curves.py",
         "仅 7B SFT train_metrics.jsonl（1638 步）+ 3B PPO rollouts.jsonl 的 update 字典（300 条）"),
        ("阶段聚合：5 段", "_audit/report_stage_summary.csv", "30 行",
         "build_report_pack.py", "python _audit/build_report_pack.py", "由上面两份逐步表聚合"),
        ("阶段聚合：10 段", "_audit/report_stage_fine.csv", "60 行",
         "build_appendix.py", "python _audit/build_appendix.py", "同上，按十分位切"),
        ("优化侧：按阶段", "_audit/report_optimizer_stage.csv", "行",
         "build_report_pack.py", "python _audit/build_report_pack.py", "仅 7B SFT 3 epoch + PPO 5 段"),
        ("实验矩阵", "_audit/report_experiment_matrix.csv", "8 行",
         "build_report_pack.py", "python _audit/build_report_pack.py", "汇总上面各表 + 手工录入的榜分"),
        ("显著性：配对 bootstrap", "_audit/official_score_bootstrap.csv", "行",
         "_audit/official_score_bootstrap.py", "python _audit/official_score_bootstrap.py",
         "37 检查点两两配对（2000 次重采样）"),
    ]
    for i, b in enumerate(blocks):
        p = ROOT / b[1]
        if p.exists():
            n = sum(1 for _ in open(p, encoding="utf-8")) - 1
            blocks[i] = (b[0], b[1], f"{n:,} 行", b[3], b[4], b[5])

    out = AUDIT / "report_traceability.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["数据块", "产物文件", "规模", "生成脚本", "复算命令", "原始来源"])
        w.writerows(blocks)
    print(f"[trace] {len(blocks)} 行 -> {out.name}")
    return blocks


def fmt(v, nd=4, pct=False):
    if v in (None, ""):
        return "—"
    if pct:
        return f"{float(v) * 100:.1f}%"
    return f"{float(v):.{nd}f}"


def build_doc(fine, trace):
    ck = load("curves_checkpoints_all.csv")
    ckby = defaultdict(list)
    for r in ck:
        ckby[r["arm"]].append(r)

    L = []
    A = L.append
    A("# 分实验数据附录")
    A("")
    A("> 本文件由 `_audit/build_appendix.py` 自动渲染，**所有数字均由脚本从本地归档重算**，不手抄。")
    A("> 与 `docs/06_项目汇报数据手册.md` 的分工：06 号是「按主题」（口径、结论、组合建议），")
    A("> 本文是「按实验」（每个实验自己有哪些数据、每个数字来自哪个文件）。")
    A("")
    A("## 0. 先看这张：数据块与来源")
    A("")
    A("| 数据块 | 产物 | 规模 | 复算命令 | 原始来源 |")
    A("|---|---|---|---|---|")
    for name, path, size, script, cmd, src in trace:
        A(f"| {name} | `{path}` | {size} | `{cmd.split('/')[-1]}` | {src} |")
    A("")
    A("三层数据的完整度不同，汇报时必须区分：")
    A("")
    A("| 层 | 覆盖 | 能回答的问题 | 不能回答的问题 |")
    A("|---|---|---|---|")
    A("| 评测层（37 点 × 5 子集） | **全部 8 个实验/臂** | 分数怎么变、冲突抓得准不准、哪个子集难 | 为什么变 |")
    A("| 奖励层（2,100 个训练单元） | **全部 6 个 RL 臂**（PPO 为 attempt 级） | 有多少组能产生梯度、梯度偏向哪类样本 | 策略偏移多快、是否发散 |")
    A("| 优化层（loss/KL/grad_norm） | **只有 7B SFT 与 3B PPO（2/8）** | 收敛速度、KL 漂移、clip 触发率 | 其余 6 个实验/臂的同类问题 |")
    A("")
    A("---")

    for name in SECTION_ORDER:
        meta = EXP_META[name]
        A("")
        A(f"## {name}")
        A("")
        arms = [k for k, v in ARM_LABEL.items() if v == name]
        arm = arms[0]
        A("| 项 | 值 |")
        A("|---|---|")
        A(f"| 归档目录 | `{meta['dir']}/` |")
        A(f"| 基座 | {meta['base']} |")
        A(f"| 算法 | {meta['algo']} |")
        if name == "3B-SFT":
            A("| 训练单元 | —（本项目内只有一次评测，无训练过程记录） |")
        else:
            A(f"| 训练单元 | {meta['total']} {meta['unit']} |")
        A(f"| 评测点数 | {len(ckby.get(arm, []))} |")
        if meta.get("note"):
            A(f"| 备注 | {meta['note']} |")
        A("")

        # —— 评测点表
        rows = sorted(ckby.get(arm, []), key=lambda r: float(r["unit_value"]))
        if rows:
            A(f"### {name} · 评测点（{meta['unit']}）")
            A("")
            A(f"| {meta['unit']} | 官方口径 | Δ vs 起点 | 本地口径 | 情绪F1 | 冲突F1 | 召回 | 特异度 | TP/FP/FN |")
            A("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                u = int(float(r["unit_value"]))
                tag = "起点" if u == 0 else str(u)
                A(f"| {tag} | {fmt(r['official_score'], 6)} | {float(r['official_delta_vs_sft']):+.6f} | "
                  f"{fmt(r['local_score'], 6)} | {fmt(r['emotion_f1'])} | {fmt(r['conflict_f1'])} | "
                  f"{fmt(r['conflict_recall'])} | {fmt(r['aligned_spec'])} | "
                  f"{r['TP']}/{r['FP']}/{r['FN']} |")
            A("")
            A(f"源文件：`{meta['dir']}/` 下各 `*_metrics.json`（路径逐点记录在 "
              f"`_audit/curves_checkpoints_all.csv` 的 `source` 列）。")
            A("")

        # —— 十分位阶段表
        fr = [r for r in fine if r["arm"] == arm]
        fr.sort(key=lambda r: int(r["stage"].split("/")[0]))

        if arm == "sft7b" and fr:
            A(f"### {name} · 十分位训练进度（1638 步 / 3 epoch）")
            A("")
            A("SFT 没有 `rollouts.jsonl`（不是 RL），所以没有奖励侧；但它是全项目唯一记录了逐步 loss 的实验。")
            A("")
            A("| 段 | step 范围 | 步数 | 总 loss | 视觉 loss | 文本 loss | 音频 loss | 梯度范数 | 学习率 |")
            A("|---|---|---|---|---|---|---|---|---|")
            for r in fr:
                lo = round(1638 * (int(r["stage"].split("/")[0]) - 1) / 10)
                hi = round(1638 * int(r["stage"].split("/")[0]) / 10)
                A(f"| {r['stage']} | {lo}–{hi} | {r['n_units']} | {fmt(r['loss'], 5)} | "
                  f"{fmt(r['task_loss_vision'], 5)} | {fmt(r['task_loss_text'], 5)} | "
                  f"{fmt(r['task_loss_audio'], 5)} | {fmt(r['grad_norm'], 3)} | "
                  f"{r['lr'] if r['lr'] else '—'} |")
            A("")
        elif arm == "ppo3b" and fr:
            A(f"### {name} · 十分位阶段（{meta['unit']}）")
            A("")
            A("PPO 每次 attempt 用 4 个**不同 prompt** 的样本，不存在「同组候选」，因此零方差组、"
              "有梯度组等组级指标不适用，只给奖励与优化侧。")
            A("")
            A("| 段 | 单元范围 | 单元数 | 平均奖励 | policy_loss | value_loss | KL | clip 触发率 | 梯度范数 |")
            A("|---|---|---|---|---|---|---|---|---|")
            for r in fr:
                A(f"| {r['stage']} | {r['progress_from']:.0f}–{r['progress_to']:.0f}% | {r['n_units']} | "
                  f"{fmt(r['mean_reward'], 4)} | {fmt(r['policy_loss'], 5)} | "
                  f"{fmt(r['value_loss'], 5)} | {fmt(r['kl'], 6)} | {fmt(r['clip_fraction'], 4)} | "
                  f"{fmt(r['grad_norm'], 3)} |")
            A("")
            kl = [r["kl"] for r in fr if r["kl"] is not None]
            cf = [r["clip_fraction"] for r in fr if r["clip_fraction"] is not None]
            raw = load("curves_optimizer_side.csv")
            def pmax(metric):
                v = [float(x["value"]) for x in raw
                     if x["experiment"] == "3B-PPO" and x["metric"] == metric]
                return max(v) if v else None
            kmax, cmax = pmax("kl"), pmax("clip_fraction")
            A(f"`policy_loss` 是裁剪后的代理目标，**可以为负、绝对值也不代表好坏**（代理目标无下界，"
              f"段均值区间 {min(r['policy_loss'] for r in fr if r['policy_loss'] is not None):.4f} ~ "
              f"{max(r['policy_loss'] for r in fr if r['policy_loss'] is not None):.4f}）。"
              f"判断策略更新幅度应看 `KL`（段均值 {min(kl):.1e} ~ {max(kl):.1e}）与 `clip 触发率`"
              f"（段均值 {min(cf) * 100:.3f}% ~ {max(cf) * 100:.3f}%）；"
              f"单次最大值 KL {kmax:.4g}、clip {cmax * 100:.2f}%，仍是量级极小 → **裁剪机制在本任务基本不触发**。")
            A("")
        elif fr:
            A(f"### {name} · 十分位阶段（每 10% 训练进度一段）")
            A("")
            A("| 段 | 单元范围 | 单元数 | 尝试组 | 平均奖励 | 零方差组 | 有梯度组 | 有梯度组数 | 冲突占比 | 决策正确率 | 接受率 |")
            A("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in fr:
                A(f"| {r['stage']} | {r['progress_from']:.0f}–{r['progress_to']:.0f}% | {r['n_units']} | "
                  f"{r['n_groups'] or '—'} | {fmt(r['mean_reward'], 4)} | {fmt(r['zero_spread_rate'], 3)} | "
                  f"{fmt(r['grad_group_rate'], 3)} | {r['grad_groups_n'] if r['grad_groups_n'] is not None else '—'} | "
                  f"{fmt(r['grad_conflict_share'], 3)} | "
                  f"{fmt(r['decision_acc'], 3)} | {fmt(r['accept_rate'], 3)} |")
            A("")
            A("`零方差组` = 组内 8 个候选奖励完全相同的组占比（该组无梯度）；`有梯度组` = 1 − 零方差组；"
              "`有梯度组数` 给的是绝对条数——**冲突占比在组数少于 5 时只由个位数样本决定，不要单独引用**"
              "（DAPO 末段就是这种情况）；`接受率` = 产生实际参数更新的组比例。")
            A("")

        if fr and meta.get("roll"):
            what = ("逐 attempt 一条，4 个样本来自不同 prompt" if arm == "ppo3b"
                    else "逐单元一条，含每组 8 个候选的 rewards")
            A(f"源文件：`{meta['dir']}/{meta['roll']}`（{what}）；"
              f"分段区间见 `_audit/report_stage_fine.csv` 的 `source` 列。")
            A("")

        # —— 7B-SFT 的三 epoch 汇总
        if arm == "sft7b":
            A("### 7B-SFT · 按 epoch 汇总")
            A("")
            A("| epoch | 平均 loss | 视觉 loss | 文本 loss | 音频 loss | 梯度范数 |")
            A("|---|---|---|---|---|---|")
            for r in load("report_optimizer_stage.csv"):
                if r["experiment"] != name or r["granularity"] != "epoch":
                    continue
                A(f"| {r['bucket']} | {fmt(r['loss'], 5)} | {fmt(r['task_loss_vision'], 5)} | "
                  f"{fmt(r['task_loss_text'], 5)} | {fmt(r['task_loss_audio'], 5)} | "
                  f"{fmt(r['gradient_norm'], 3)} |")
            A("")
            A(f"源文件：`{meta['dir']}/{meta['opt']}`（逐步，1638 行）；"
              f"epoch 按 546 步硬切（日志里的 epoch 字段在末步多一个边界值，不能直接分组）。")
            A("")

        # —— 缺口
        gaps = []
        if name == "3B-SFT":
            gaps.append("3B SFT 是前序阶段的产物，本项目内**没有它的训练记录**（无日志、无 `cycles.jsonl`、无 loss），"
                        "只有一次评测结果；它在本项目里的角色是三轮 3B RL 的共同起点。")
        elif not meta.get("opt") and arm != "ppo3b":
            gaps.append("**没有 loss / KL / 梯度范数**：训练脚本算了这三个标量但没有写入 `cycles.jsonl`，"
                        "日志也只打组级计数 → 事后无法补，除非重训。")
        if arm == "ppo3b":
            gaps.append("组级指标（零方差组、有梯度组）不适用：PPO 每次 attempt 的 4 个样本来自不同 prompt。")
        if name == "3B-SFT":
            gaps.append("只有一个评测点，没有训练曲线（SFT 在本项目里作为 RL 起点使用，未单独训 3B SFT）。")
        if gaps:
            A("### 缺口")
            A("")
            for g in gaps:
                A(f"- {g}")
            A("")
        A("---")

    # —— 十分位趋势汇总（自动算，不手写）
    A("")
    A("## 附一：十分位趋势横向对照")
    A("")
    A("同一张表把 6 个 RL 臂放在一起，看「训练越久，还能学到东西的组有多少」。`有梯度组数` 后两段之和"
      "是关键：越接近 0，说明模型在该阶段已经空转。")
    A("")
    TAILS = {}
    A("| 实验/臂 | 单元 | 有梯度组数（1→10 段） | 末两段合计 | 平均奖励（首段→末段） | 末段零方差组 |")
    A("|---|---|---|---|---|---|")
    for name in SECTION_ORDER:
        if name in ("3B-SFT", "7B-SFT"):
            continue
        arm = [k for k, v in ARM_LABEL.items() if v == name][0]
        rs = sorted([r for r in fine if r["arm"] == arm], key=lambda r: int(r["stage"].split("/")[0]))
        if not rs:
            continue
        seq = [r["grad_groups_n"] for r in rs]
        r0 = rs[0]["mean_reward"]
        r1 = rs[-1]["mean_reward"]
        if arm == "ppo3b" or all(v is None for v in seq):
            A(f"| {name} | {rs[0]['unit']} | 不适用（每次 attempt 的 4 个样本来自不同 prompt） | — | "
              f"{fmt(r0, 4)} → {fmt(r1, 4)} | — |")
            TAILS[name] = None
            continue
        seq_s = " → ".join(str(v) for v in seq)
        tail = sum(v for v in seq[-2:])
        head = sum(v for v in seq[:2])
        TAILS[name] = (head, tail)
        A(f"| {name} | {rs[0]['unit']} | {seq_s} | **{tail}** | "
          f"{fmt(r0, 4)} → {fmt(r1, 4)} | {fmt(rs[-1]['zero_spread_rate'], 3)} |")
    A("")
    # 结论句按实测值生成，避免手写数字漂移
    if TAILS.get("7B-DAPO") and TAILS.get("3B-GRPO-r1"):
        d_head, d_tail = TAILS["7B-DAPO"]
        r_head, r_tail = TAILS["3B-GRPO-r1"]
        r2_tail = TAILS["3B-GRPO-r2"][1] if TAILS.get("3B-GRPO-r2") else None
        A(f"读法：DAPO 前两段共 {d_head} 个有梯度组，末两段只剩 **{d_tail}** 个；"
          f"3B GRPO r1 是 {r_head} → {r_tail}"
          + (f"，r2 靠难度配额把末两段撑到 {r2_tail} 组。" if r2_tail else "。")
          + "**这正是「换算法/加补丁」都救不回来的地方**——不是梯度算错了，是可学的组本身没有了。")
    A("")

    A("## 附二：一句话回答「有没有完整训练曲线数据」")
    A("")
    A("**评测曲线完整**（37 个检查点，含 TP/FP/FN 与 5 个子集）；**奖励侧曲线完整**（2,100 个训练单元，"
      "由 `rollouts.jsonl` 离线重算）；**优化侧曲线在 8 个实验/臂里只有 2 个有**（7B SFT、3B PPO）。"
      "GRPO 系四个实验的 loss/KL 从未落盘，全库没有 wandb / tensorboard，"
      "因此不能声称它们「loss 收敛」或「训练发散」，只能从奖励与组级信号推断。")
    A("")

    out = DOCS / "07_分实验数据附录.md"
    out.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"[doc] {len(L)} 行 -> docs/{out.name}")
    return out


if __name__ == "__main__":
    fine = build_fine()
    print()
    trace = build_traceability()
    print()
    build_doc(fine, trace)
