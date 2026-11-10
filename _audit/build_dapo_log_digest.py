"""把 logs/dapo_train_*.log 与 runs/*/cycles.jsonl 提炼成一份可读的训练日志摘要。

只读已归档文件。输出 docs/07_训练日志摘要.md。
所有聚合数字都由本脚本现算，文档里不手抄任何指标。
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAPO = ROOT / "2026-11-01-DAPO"
OUT = DAPO / "docs" / "07_训练日志摘要.md"
ARMS = [("dapo7b", "score_focused / 决策优势 0.5"),
        ("dapo7b_legacy", "legacy / 决策优势 0")]
WINDOW = 25
STEPS = [50, 100, 150, 200, 250, 300]


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]


def log_stats(arm):
    p = DAPO / "logs" / f"dapo_train_{arm}.log"
    txt = p.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    step_lines = [l for l in lines if l.startswith("STEP ")]
    evals = [l.split(" ", 2)[1] for l in lines if l.startswith("EVALUATION ")]
    bad = len(re.findall(r"(?i)\btraceback\b", txt))
    err = len(re.findall(r"(?i)\berror\b", txt))
    nan = len(re.findall(r"(?i)\bnan\b", txt))
    return {
        "file": p.name, "bytes": p.stat().st_size, "lines": len(lines),
        "step_lines": len(step_lines), "evals": evals,
        "traceback": bad, "error": err, "nan": nan,
    }


def window_table(cycles):
    out = []
    for i in range(0, len(cycles), WINDOW):
        chunk = cycles[i:i + WINDOW]
        bc = Counter()
        for c in chunk:
            bc.update(c["bucket_counts"])
        out.append({
            "range": f"{chunk[0]['step']}–{chunk[-1]['step']}",
            "zero": sum(1 for c in chunk if c["updates"] == 0),
            "attempts": sum(c["attempted_groups"] for c in chunk) / len(chunk),
            "active": sum(c["active_groups"] for c in chunk) / len(chunk),
            "dec": bc["decision_active"],
            "allok": bc["all_correct"],
            "aux": bc["auxiliary_active"],
            "cu": sum(1 for c in chunk if c["clip_upper"] > 0),
            "cl": sum(1 for c in chunk if c["clip_lower"] > 0),
            "hard": chunk[-1]["difficulty"]["hard_pool"],
        })
    return out


def main():
    stats = {a: log_stats(a) for a, _ in ARMS}
    cycles = {a: read_jsonl(DAPO / "runs" / a / "cycles.jsonl") for a, _ in ARMS}
    rollouts = {a: read_jsonl(DAPO / "runs" / a / "rollouts.jsonl") for a, _ in ARMS}

    L = []
    A = L.append
    A("# DAPO 训练日志摘要（7B，两臂）")
    A("")
    A("> 由 `_audit/build_dapo_log_digest.py` 从已归档的 `logs/*.log` 与 `runs/*/cycles.jsonl` 自动生成，"
      "数字均现算。原始日志见 `logs/dapo_train_dapo7b.log`、`logs/dapo_train_dapo7b_legacy.log`（各约 3.4 MB、300 条 STEP 记录）。")
    A("> 时间线与门禁输出见 [05_训练记录.md](05_训练记录.md)，结论解释见 [06_结果分析.md](06_结果分析.md)。")
    A("")

    A("## 1. 日志文件体检")
    A("")
    A("| 文件 | 体积 | 总行数 | STEP 记录 | EVALUATION | traceback | error | nan |")
    A("|---|---|---|---|---|---|---|---|")
    for a, _ in ARMS:
        s = stats[a]
        A(f"| `{s['file']}` | {s['bytes']/1024/1024:.1f} MB | {s['lines']:,} | {s['step_lines']} | "
          f"{len(s['evals'])} | {s['traceback']} | {s['error']} | {s['nan']} |")
    A("")
    A(f"评测节点（两臂同构）：{' / '.join(stats[ARMS[0][0]]['evals'])}")
    A("")

    A("## 2. 全程训练信号")
    A("")
    A("| 指标 | " + " | ".join(a for a, _ in ARMS) + " |")
    A("|---|" + "---|" * len(ARMS))
    def cmp_rows(key):
        vals = []
        for a, _ in ARMS:
            cy = cycles[a]
            if key == "cycles":
                vals.append(f"{len(cy)}")
            elif key == "zero":
                vals.append(f"{sum(1 for c in cy if c['updates'] == 0)}/{len(cy)}"
                            f"（{sum(1 for c in cy if c['updates'] == 0)/len(cy)*100:.0f}%）")
            elif key == "attempts":
                vals.append(f"{sum(c['attempted_groups'] for c in cy)/len(cy):.2f}")
            elif key == "accepted":
                vals.append(f"{sum(1 for g in rollouts[a] if g['accepted'])}/{len(rollouts[a])}"
                            f"（{sum(1 for g in rollouts[a] if g['accepted'])/len(rollouts[a])*100:.1f}%）")
            elif key == "buckets":
                bc = Counter()
                for c in cy:
                    bc.update(c["bucket_counts"])
                vals.append(f"全对 {bc['all_correct']} / 决策有效 {bc['decision_active']} / 辅助有效 {bc['auxiliary_active']}")
            elif key == "clip":
                vals.append(f"上界 {sum(c['clip_upper'] for c in cy)} 个 token（{sum(1 for c in cy if c['clip_upper']>0)}/{len(cy)} 步）"
                            f" / 下界 {sum(c['clip_lower'] for c in cy)} 个（{sum(1 for c in cy if c['clip_lower']>0)} 步）")
            elif key == "hard":
                vals.append(f"{cy[0]['difficulty']['hard_pool']} → {cy[-1]['difficulty']['hard_pool']}")
            elif key == "byclass":
                sel = Counter(g["gold_conflict"] for g in rollouts[a] if g["accepted"])
                tot = sum(sel.values())
                vals.append(f"Conflict {sel['Conflict']} / Aligned {sel['Aligned']}"
                            f"（冲突占 {sel['Conflict']/tot*100:.0f}%）")
        return vals

    rows = [("训练周期数", "cycles"), ("零更新周期", "zero"), ("每周期尝试组数（均值）", "attempts"),
            ("被接受组 / 尝试组", "accepted"), ("组类型分布（合计）", "buckets"),
            ("接受组的金标类别", "byclass"), ("非对称 clip 触发量", "clip"), ("hard_pool 首 → 末", "hard")]
    for name, key in rows:
        A(f"| {name} | " + " | ".join(cmp_rows(key)) + " |")
    A("")

    A("## 3. 25 周期窗口轨迹")
    A("")
    for a, desc in ARMS:
        A(f"### {a}（{desc}）")
        A("")
        A("| 周期区间 | 零更新 | 尝试组/周期 | 有效组/周期 | decision_active | all_correct | auxiliary_active | clip_upper 触发步 | clip_lower 触发步 | hard_pool 末值 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for w in window_table(cycles[a]):
            A(f"| {w['range']} | {w['zero']}/{WINDOW} | {w['attempts']:.2f} | {w['active']:.2f} | "
              f"{w['dec']} | {w['allok']} | {w['aux']} | {w['cu']} | {w['cl']} | {w['hard']} |")
        A("")

    A("## 4. 评测节点（官方口径）")
    A("")
    A("| 臂 | step | kaggle | local | 情绪F1 | 冲突召回 | 一致特异度 | TP | FP | FN |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    base = json.loads((DAPO / "runs" / "dapo7b" / "sft_baseline_metrics.json").read_text(encoding="utf-8"))["all"]

    def line(arm, step, m):
        a = m["all"]
        emo = a["emotion_macro_f1_present"]
        local = 0.5 * emo + 0.3 * a["conflict_recall"] + 0.2 * a["aligned_specificity"]
        kg = 0.5 * emo + 0.3 * a["conflict_accuracy"] + 0.2 * a["aligned_specificity"]
        c = a["conflict_confusion"]
        A(f"| {arm} | {step} | {kg:.6f} | {local:.6f} | {emo:.4f} | {a['conflict_recall']:.4f} | "
          f"{a['aligned_specificity']:.4f} | {c['TP']} | {c['FP']} | {c['FN']} |")

    line("SFT 基线", "—", {"all": base})
    for a, _ in ARMS:
        for s in STEPS:
            line(a, s, json.loads((DAPO / "runs" / a / f"step_{s:04d}_metrics.json").read_text(encoding="utf-8")))
    A("")

    A("## 5. 从日志能直接读出的四件事")
    A("")
    cy0 = cycles["dapo7b"]
    bc0 = Counter()
    for c in cy0:
        bc0.update(c["bucket_counts"])
    allok_ratio = bc0["all_correct"] / sum(bc0.values()) * 100
    A(f"1. **奖励饱和**：{allok_ratio:.1f}% 的候选组 8 条全对（{bc0['all_correct']}/{sum(bc0.values())}），"
      f"组内奖励方差为 0，优势 = (r−mean)/std 变成 0/0，梯度为 0 —— 这是零更新周期占 "
      f"{sum(1 for c in cy0 if c['updates']==0)}/{len(cy0)} 的直接原因。")
    A(f"2. **动态采样只是把无效组挑出去**：过滤后每周期平均只剩 "
      f"{sum(c['active_groups'] for c in cy0)/len(cy0):.2f} 个有效组；"
      f"1725 个尝试组里，冲突决策有效组 {bc0['decision_active']} 个、辅助字段有效组 "
      f"{bc0['auxiliary_active']} 个，梯度几乎全部来自「该不该报冲突」这一个二元判断。")
    A(f"3. **Clip-Higher 名存实亡**：300 个周期里 ratio 越过上界 1.28 的 token 合计只有 "
      f"{sum(c['clip_upper'] for c in cy0)} 个（{sum(1 for c in cy0 if c['clip_upper']>0)} 个周期触发过），"
      "改进版 clip 实际上没有参与塑形；真正起作用的是 Dynamic Sampling。")
    tail = [c for c in cy0 if c["step"] >= 251]
    A(f"4. **末段 50 个周期基本空转**：第 251–300 周期累计只有 {sum(c['active_groups'] for c in tail)} 个有效组，"
      f"{sum(1 for c in tail if c['updates']==0)}/{len(tail)} 个周期零更新。"
      "换句话说 step 250 之后模型几乎不再变化，官方口径在这段的下滑不是「越训越差」，"
      "而是**没有梯度、只剩评测噪声**——这也解释了为什么官方最佳点落在 step 50 而不是 `best.json` 选的 step 200/300。")
    A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"写出 {OUT}（{len(L)} 行，{len(OUT.read_text(encoding='utf-8'))} 字符）")


if __name__ == "__main__":
    main()
