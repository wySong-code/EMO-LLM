# -*- coding: utf-8 -*-
"""根据 _audit/curves_*.csv 绘制全部训练曲线图，输出到 docs/figures/。

只读 CSV，不重算指标。中文字体用 Windows 自带 SimHei。
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CJK = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
CJK_B = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")
plt.rcParams["axes.unicode_minus"] = False

# 统一配色（中国习惯：涨=红、跌=绿，但此处用于区分实验，用中性色系避免误导）
COLORS = {
    "sft": "#6b7280",
    "r1": "#1f77b4",
    "r2": "#0ea5a4",
    "ppo": "#a855f7",
    "g7b": "#ea580c",
    "dapo": "#dc2626",
    "legacy": "#92400e",
    "sft7b": "#4b5563",
}
GRID = dict(color="#e5e7eb", linewidth=0.8, alpha=0.9)


def load_ckpt():
    rows = list(csv.DictReader(open(AUDIT / "curves_checkpoints_all.csv", encoding="utf-8")))
    for r in rows:
        for k in ("unit_value", "progress_pct", "local_score", "official_score",
                  "official_delta_vs_sft", "emotion_f1", "conflict_f1", "conflict_recall",
                  "conflict_acc", "aligned_spec", "TP", "FP", "FN", "TN"):
            r[k] = float(r[k])
    return rows


def load_sig():
    rows = list(csv.DictReader(open(AUDIT / "curves_train_signals.csv", encoding="utf-8")))
    d = defaultdict(list)
    for r in rows:
        d[(r["experiment"], r["arm"], r["metric"])].append((float(r["unit_value"]), float(r["value"])))
    for k in d:
        d[k].sort()
    return d


def roll(y, w):
    """滑动均值，边界用可用点补齐。"""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = max(1, min(w, len(y)))
    k = np.ones(w) / w
    pad = w // 2
    ypad = np.pad(y, (pad, w - 1 - pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontproperties=CJK_B, fontsize=11.5, pad=8)
    ax.set_xlabel(xlabel, fontproperties=CJK, fontsize=9.5)
    ax.set_ylabel(ylabel, fontproperties=CJK, fontsize=9.5)
    ax.grid(True, **GRID)
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def series(rows, exp, arm=None):
    out = [r for r in rows if r["experiment"] == exp and (arm is None or r["arm"] == arm)]
    out.sort(key=lambda r: r["unit_value"])
    return out


# ─── 图 1：3B 四组官方口径曲线 ────────────────────────────────────────────────
def fig_3b(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    base = series(rows, "3B-SFT")[0]["official_score"]
    # 三轮总步数不同（r1 300 step、r2 600 attempt、PPO 300 attempt），故统一按进度百分比横轴
    specs = [("3B-GRPO-r1", "grpo3b_r1", "GRPO 第一轮（300 step）", COLORS["r1"], "o", "step"),
             ("3B-GRPO-r2", "grpo3b_r2", "GRPO 第二轮（600 attempt）", COLORS["r2"], "s", "attempt"),
             ("3B-PPO", "ppo3b", "PPO（300 attempt）", COLORS["ppo"], "^", "attempt")]
    for exp, arm, label, c, mk, unit in specs:
        s = series(rows, exp, arm)
        ax.plot([r["progress_pct"] for r in s], [r["official_score"] for r in s],
                marker=mk, color=c, lw=1.6, ms=5, label=label)
        best = max(s, key=lambda r: r["official_score"])
        ax.annotate(f"{best['official_score']:.4f}",
                    (best["progress_pct"], best["official_score"]),
                    textcoords="offset points", xytext=(4, 8), fontsize=7.5,
                    color=c, fontproperties=CJK)
    ax.axhline(base, color=COLORS["sft"], ls="--", lw=1.2, label=f"3B SFT 基线 {base:.4f}")
    style(ax, "3B：官方口径分数（横轴按各自训练进度归一）", "训练进度（%）", "官方口径分数")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="lower left")

    ax = axes[1]
    for exp, arm, label, c, mk, unit in specs:
        s = series(rows, exp, arm)
        ax.plot([r["progress_pct"] for r in s], [r["official_delta_vs_sft"] for r in s],
                marker=mk, color=c, lw=1.6, ms=5, label=label)
    ax.axhline(0, color=COLORS["sft"], ls="--", lw=1.2, label="各自 SFT 起点")
    style(ax, "3B：相对各自 SFT 起点的增益", "训练进度（%）", "Δ 官方口径")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="lower left")

    fig.tight_layout()
    p = FIG / "fig01_3b_score_curves.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 2：7B 四条线 ─────────────────────────────────────────────────────────
def fig_7b(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1),
                             gridspec_kw={"width_ratios": [0.85, 1.25, 1.25]})

    # (a) SFT 三个 epoch：单独一张，横轴用 epoch，绝不和 step 混轴
    ax = axes[0]
    sft = series(rows, "7B-SFT")
    ax.plot([r["unit_value"] for r in sft], [r["official_score"] for r in sft],
            marker="D", color=COLORS["sft7b"], lw=1.8, ms=7, label="官方口径")
    for r in sft:
        ax.annotate(f"{r['official_score']:.4f}", (r["unit_value"], r["official_score"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=7.6, color=COLORS["sft7b"], fontproperties=CJK)
    best = max(sft, key=lambda r: r["official_score"])
    ax.scatter([best["unit_value"]], [best["official_score"]], s=170, facecolors="none",
               edgecolors="#111827", linewidths=1.4, zorder=5)
    ax.set_xticks([1, 2, 3])
    style(ax, "7B SFT：3 个 epoch", "epoch", "官方口径分数")
    ax.set_ylim(min(r["official_score"] for r in sft) - 0.004,
                max(r["official_score"] for r in sft) + 0.005)
    ax.annotate("交付用的是 ep3\n但 ep2 更高", (3, sft[-1]["official_score"]),
                textcoords="offset points", xytext=(-58, -30), fontsize=7.4,
                fontproperties=CJK, color="#374151")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="lower left")

    # (b) GRPO / DAPO 两臂：统一 step 轴，SFT 起点作为 step 0
    ax = axes[1]
    g = series(rows, "7B-GRPO", "grpo7b")
    base = g[0]["official_score"]
    ax.plot([0] + [r["unit_value"] for r in g], [base] + [r["official_score"] for r in g],
            marker="o", color=COLORS["g7b"], lw=1.6, ms=5, label="GRPO")
    for arm, label, c, mk in [("dapo7b", "DAPO", COLORS["dapo"], "^"),
                              ("dapo7b_legacy", "DAPO-legacy", COLORS["legacy"], "v")]:
        s = series(rows, "7B-DAPO", arm)
        ax.plot([0] + [r["unit_value"] for r in s], [base] + [r["official_score"] for r in s],
                marker=mk, color=c, lw=1.6, ms=5, label=label)
    ax.axhline(base, color=COLORS["sft"], ls="--", lw=1.2, label=f"SFT 起点 {base:.4f}")
    style(ax, "7B：GRPO / DAPO 两臂（step 轴，0 = SFT）", "step", "官方口径分数")
    ax.set_xlim(-10, 315)
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="lower left")

    # (c) FP 上升、TP 不动
    ax = axes[2]
    for arm, label, c, mk in [("dapo7b", "DAPO", COLORS["dapo"], "^"),
                              ("dapo7b_legacy", "DAPO-legacy", COLORS["legacy"], "v")]:
        s = series(rows, "7B-DAPO", arm)
        ax.plot([r["progress_pct"] for r in s], [r["FP"] for r in s],
                marker=mk, color=c, lw=1.6, ms=5, label=f"{label} FP")
    ax.plot([r["progress_pct"] for r in g], [r["FP"] for r in g],
            marker="o", color=COLORS["g7b"], lw=1.6, ms=5, label="GRPO FP")
    ax.axhline(sft[-1]["FP"], color=COLORS["sft7b"], ls="--", lw=1.2,
               label=f"SFT FP={int(sft[-1]['FP'])}")
    ax2 = ax.twinx()
    ax2.plot([r["progress_pct"] for r in g], [r["TP"] for r in g], color=COLORS["g7b"],
             ls=":", lw=1.2, label="GRPO TP（右轴）")
    ax2.set_ylabel("TP（命中）", fontproperties=CJK, fontsize=9.5)
    ax2.tick_params(labelsize=8.5)
    ax2.set_ylim(50, 70)
    style(ax, "7B：误警 FP 上升而 TP 几乎不动", "训练进度（%）", "FP（误警数）")
    ax.set_ylim(20, 60)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, prop=CJK, fontsize=7.4, frameon=False, loc="lower right")

    fig.tight_layout()
    p = FIG / "fig02_7b_score_curves.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 3：本地口径 vs 官方口径散点 ───────────────────────────────────────────
def load_ledger():
    """从 docs/08_提交台账.md 第 5 节解析"本地名次 vs 榜上名次"，避免手抄。"""
    p = ROOT / "2026-11-01-DAPO" / "docs" / "08_提交台账.md"
    rows, in_sec = [], False
    for line in open(p, encoding="utf-8"):
        s = line.strip()
        if s.startswith("## 5."):
            in_sec = True
            continue
        if in_sec and s.startswith("## "):
            break
        if not in_sec or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 6 or cells[0] in ("模型",) or set(cells[0]) <= set("-: "):
            continue
        rows.append({"model": cells[0],
                     "local": float(cells[1]), "local_rank": int(cells[2]),
                     "kaggle": float(cells[3]), "kaggle_rank": int(cells[4])})
    return rows


def fig_local_vs_official(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    ax = axes[0]
    groups = [("3B-GRPO-r1", "grpo3b_r1", "3B GRPO r1", COLORS["r1"], "o"),
              ("3B-GRPO-r2", "grpo3b_r2", "3B GRPO r2", COLORS["r2"], "s"),
              ("3B-PPO", "ppo3b", "3B PPO", COLORS["ppo"], "^"),
              ("7B-SFT", "sft7b", "7B SFT", COLORS["sft7b"], "D"),
              ("7B-GRPO", "grpo7b", "7B GRPO", COLORS["g7b"], "o"),
              ("7B-DAPO", "dapo7b", "7B DAPO", COLORS["dapo"], "^"),
              ("7B-DAPO", "dapo7b_legacy", "7B DAPO-legacy", COLORS["legacy"], "v")]
    for exp, arm, label, c, mk in groups:
        s = series(rows, exp, arm)
        ax.scatter([r["local_score"] for r in s], [r["official_score"] for r in s],
                   s=60, color=c, marker=mk, label=label, alpha=0.9,
                   edgecolors="white", linewidths=0.7)
    ax.annotate("同一实验的点沿竖直方向拉开：\n本地分挤在窄区间，官方分摆动更大",
                (0.8665, 0.9445), fontsize=7.8, fontproperties=CJK, color="#374151")
    style(ax, "本地验证集口径 vs 官方口径（37 个检查点，纵轴散布更大）",
          "本地口径（0.3×冲突召回）", "官方口径（0.3×冲突准确率）")
    ax.legend(prop=CJK, fontsize=7.4, frameon=False, ncol=2, loc="lower right")

    # (b) 七个真实提交模型的名次逆转
    ax = axes[1]
    led = load_ledger()
    left_x, right_x = 0.0, 1.0
    for r in led:
        is_3b = r["model"].startswith("3B")
        c = COLORS["r1"] if is_3b else COLORS["g7b"]
        moved = r["local_rank"] - r["kaggle_rank"]
        lw = 2.2 if abs(moved) >= 3 else 1.2
        ax.plot([left_x, right_x], [r["local_rank"], r["kaggle_rank"]],
                color=c, lw=lw, alpha=0.9, marker="o", ms=5,
                ls="-" if abs(moved) < 3 else "--")
        ax.annotate(f"{r['model']}", (left_x, r["local_rank"]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.6, fontproperties=CJK, color=c)
        ax.annotate(f"{r['model']}  {r['kaggle']:.5f}", (right_x, r["kaggle_rank"]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=7.6, fontproperties=CJK, color=c)
    ax.set_xlim(-0.55, 1.68)
    ax.set_ylim(7.6, 0.4)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(["本地验证集名次", "Kaggle 榜分名次"], fontproperties=CJK, fontsize=9.5)
    ax.set_yticks(range(1, 8))
    ax.set_yticklabels([f"第 {i}" for i in range(1, 8)], fontproperties=CJK, fontsize=8.5)
    ax.grid(True, axis="y", **GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("7 个真实提交模型：本地名次与榜分名次（实线=稳定，虚线=大幅逆转）",
                 fontproperties=CJK_B, fontsize=11, pad=8)

    fig.tight_layout()
    p = FIG / "fig03_local_vs_official.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 4：3B GRPO r1 训练信号 ───────────────────────────────────────────────
def fig_r1_signal(sig, rows):
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    x, up = zip(*sig[("3B-GRPO-r1", "grpo3b_r1", "updates")])
    x = np.array(x, dtype=float); up = np.array(up, dtype=float)
    _, zv = zip(*sig[("3B-GRPO-r1", "grpo3b_r1", "zero_update")])
    zv = np.array(zv, dtype=float)

    # 每 20 步一箱，避免 300 步逐点柱状变成"竖线森林"
    BIN = 20
    edges = np.arange(1, 301 + BIN, BIN)
    cx, csum, czero = [], [], []
    for lo in edges[:-1]:
        mask = (x >= lo) & (x < lo + BIN)
        if not mask.any():
            continue
        cx.append(lo + BIN / 2 - 0.5)
        csum.append(up[mask].sum())
        czero.append(zv[mask].mean() * 100)
    ax.bar(cx, csum, width=BIN * 0.8, color=COLORS["r1"], alpha=0.55,
           label="每 20 步窗口内的有效更新次数（左轴）")
    style(ax, "3B GRPO 第一轮：有效更新总量与零更新占比", "", "有效更新次数 / 20 步")
    ax2 = ax.twinx()
    ax2.plot(cx, czero, color="#111827", lw=1.8, marker="o", ms=3.4,
             label="零更新步占比（右轴，%）")
    ax2.set_ylabel("零更新步占比（%）", fontproperties=CJK, fontsize=9.5)
    ax2.tick_params(labelsize=8.5)
    ax2.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, prop=CJK, fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, max(csum) * 1.35)

    ax = axes[1]
    s = series(rows, "3B-GRPO-r1", "grpo3b_r1")
    sx = [0] + [r["unit_value"] for r in s]
    sy = [series(rows, "3B-SFT")[0]["official_score"]] + [r["official_score"] for r in s]
    ax.plot(sx, sy, marker="o", color=COLORS["r1"], lw=1.8, ms=6, label="官方口径分数")
    ax.axhline(sy[0], color=COLORS["sft"], ls="--", lw=1.1, label=f"SFT 基线 {sy[0]:.4f}")
    for xx, yy in zip(sx, sy):
        ax.annotate(f"{yy:.4f}", (xx, yy), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.4, color=COLORS["r1"], fontproperties=CJK)
    best_i = int(np.argmax(sy))
    ax.scatter([sx[best_i]], [sy[best_i]], s=150, facecolors="none",
               edgecolors="#111827", linewidths=1.4, zorder=5)
    style(ax, "3B GRPO 第一轮：6 个评测点的官方口径（step50 为唯一高于基线的点）", "step", "官方口径分数")
    ax.set_ylim(min(sy) - 0.004, max(sy) + 0.006)
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    p = FIG / "fig04_3b_grpo_r1_signal.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 5：r2 与 PPO 的信号 ─────────────────────────────────────────────────
def fig_r2_ppo(sig, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    x, a = zip(*sig[("3B-GRPO-r2", "grpo3b_r2", "accept_rate")])
    ax.plot(x, a, color=COLORS["r2"], lw=1.4, marker="o", ms=3, label="接受组比例（每 10 attempt 窗口）")
    ax.plot(x, roll(a, 5), color="#0f766e", lw=2.2, label="5 窗口滑动均值")
    ax.set_ylim(0.80, 1.0)
    style(ax, "3B GRPO 第二轮：接受组比例（越高=越多组有梯度）", "attempt", "接受组 / 候选组")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="lower left")

    ax = axes[1]
    x, r = zip(*sig[("3B-PPO", "ppo3b", "mean_reward")])
    ax.plot(x, r, color=COLORS["ppo"], lw=1.5, marker="o", ms=3.4, label="平均奖励")
    ax.set_ylim(0.93, 0.96)
    ax2 = ax.twinx()
    xk, k = zip(*sig[("3B-PPO", "ppo3b", "kl")])
    ax2.plot(xk, k, color="#7c3aed", ls="--", lw=1.2, alpha=0.85, label="KL（右轴）")
    ax2.set_ylabel("KL 散度", fontproperties=CJK, fontsize=9.5)
    ax2.tick_params(labelsize=8.5)
    style(ax, "3B PPO：平均奖励与 KL", "attempt", "平均奖励")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, prop=CJK, fontsize=8, frameon=False, loc="lower left")

    fig.tight_layout()
    p = FIG / "fig05_3b_r2_ppo_signal.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 6：7B GRPO vs DAPO 接受率 ───────────────────────────────────────────
def fig_7b_signal(sig):
    """注意：GRPO 每步固定 2 组、DAPO 每步 3–6 组动态采样，绝对组数不可比，故一律用比例。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    series_def = [("7B-GRPO", "grpo7b", "7B GRPO", COLORS["g7b"]),
                  ("7B-DAPO", "dapo7b", "DAPO", COLORS["dapo"]),
                  ("7B-DAPO", "dapo7b_legacy", "DAPO-legacy", COLORS["legacy"])]

    ax = axes[0]
    for exp, arm, label, c in series_def:
        x, v = zip(*sig[(exp, arm, "zero_update")])
        ax.plot(x, roll(v, 15) * 100, lw=1.7, color=c, label=f"{label}")
    ax.set_ylim(0, 100)
    style(ax, "7B：零更新步占比（15 步滑窗）—— 越高说明越多步拿不到梯度", "step", "零更新步占比（%）")
    ax.legend(prop=CJK, fontsize=7.6, frameon=False, loc="lower center", ncol=3)

    ax = axes[1]
    for exp, arm, label, c in series_def:
        xa, ga = zip(*sig[(exp, arm, "all_correct_groups")])
        xb, gb = zip(*sig[(exp, arm, "groups_attempted")])
        d = dict(zip(xb, gb))
        xs = [x for x in xa if d.get(x)]
        ys = [a / d[x] for x, a in zip(xa, ga) if d.get(x)]
        ax.plot(xs, roll(ys, 15) * 100, lw=1.7, color=c, label=f"{label}")
    ax.set_ylim(0, 100)
    style(ax, "7B：组内 8 个候选全对的比例（奖励饱和的直接证据）", "step", "全对组占比（%）")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    p = FIG / "fig06_7b_grpo_dapo_signal.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 7：7B SFT loss ─────────────────────────────────────────────────────
def fig_sft_loss(sig, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    x, loss = zip(*sig[("7B-SFT", "sft7b", "loss")])
    x = np.array(x); loss = np.array(loss)
    ax.semilogy(x, loss, color="#9ca3af", lw=0.55, alpha=0.35, label="逐步 loss（原始）")
    ax.semilogy(x, roll(loss, 41), color="#111827", lw=1.8, label="41 步滑动均值")
    for e, xx in [(1, 546), (2, 1092), (3, 1638)]:
        ax.axvline(xx, color="#d1d5db", ls=":", lw=1)
        ax.annotate(f"epoch {e}", xy=(xx, 0), xycoords=("data", "axes fraction"),
                    xytext=(-30, 6), textcoords="offset points",
                    fontsize=7.6, fontproperties=CJK, color="#6b7280")
    style(ax, "7B SFT：训练 loss（对数轴，1638 步 / 3 epoch）", "step", "loss（log）")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(5e-6, 3.0)

    ax = axes[1]
    for key, label, c in [("task_loss_vision", "仅视觉任务 loss", "#2563eb"),
                          ("task_loss_text", "仅文本任务 loss", "#059669"),
                          ("task_loss_audio", "仅音频任务 loss", "#d97706")]:
        x, v = zip(*sig[("7B-SFT", "sft7b", key)])
        ax.semilogy(x, roll(v, 41), lw=1.6, color=c, label=label)
    style(ax, "7B SFT：三模态任务 loss（41 步滑窗）", "step", "task loss（log）")
    ax.legend(prop=CJK, fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    p = FIG / "fig07_7b_sft_loss.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ─── 图 8：TP vs FP ────────────────────────────────────────────────────────
def fig_tp_fp(rows):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    groups = [("3B-GRPO-r1", "grpo3b_r1", "3B GRPO r1", COLORS["r1"], "o"),
              ("3B-GRPO-r2", "grpo3b_r2", "3B GRPO r2", COLORS["r2"], "s"),
              ("3B-PPO", "ppo3b", "3B PPO", COLORS["ppo"], "^"),
              ("7B-SFT", "sft7b", "7B SFT", COLORS["sft7b"], "D"),
              ("7B-GRPO", "grpo7b", "7B GRPO", COLORS["g7b"], "o"),
              ("7B-DAPO", "dapo7b", "7B DAPO", COLORS["dapo"], "^"),
              ("7B-DAPO", "dapo7b_legacy", "7B DAPO-legacy", COLORS["legacy"], "v")]
    for exp, arm, label, c, mk in groups:
        s = series(rows, exp, arm)
        ax.plot([r["FP"] for r in s], [r["TP"] for r in s], color=c, lw=1.0, alpha=0.45)
        ax.scatter([r["FP"] for r in s], [r["TP"] for r in s], s=44, color=c,
                   marker=mk, label=label, edgecolors="white", linewidths=0.6, zorder=3)
    ax.annotate("验证集真冲突 74 条\n(P+R 的天花板)", (48, 63), fontsize=8,
                fontproperties=CJK, color="#374151")
    style(ax, "命中 TP 与误警 FP 的权衡：所有 37 个检查点", "FP（把一致样本判成冲突）", "TP（正确识别冲突）")
    ax.legend(prop=CJK, fontsize=7.8, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    p = FIG / "fig08_tp_fp_tradeoff.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


if __name__ == "__main__":
    rows = load_ckpt()
    sig = load_sig()
    made = [fig_3b(rows), fig_7b(rows), fig_local_vs_official(rows),
            fig_r1_signal(sig, rows), fig_r2_ppo(sig, rows),
            fig_7b_signal(sig), fig_sft_loss(sig, rows), fig_tp_fp(rows)]
    for p in made:
        print("写出", p.name, "%.0f KB" % (p.stat().st_size / 1024))
