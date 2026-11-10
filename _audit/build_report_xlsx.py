# -*- coding: utf-8 -*-
"""把 _audit 下的汇报数据表打包成一个多 sheet 的 Excel 工作簿。

只读取已生成的 CSV，不重算。
产物：docs/汇报数据包.xlsx
"""
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
OUT = ROOT / "docs" / "汇报数据包.xlsx"

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10, name="微软雅黑")
TITLE_FONT = Font(bold=True, size=14, name="微软雅黑", color="1F3864")
NOTE_FONT = Font(size=10, name="微软雅黑", color="374151")
THIN = Side(style="thin", color="D1D5DB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# (sheet 名, CSV 文件, 中文标题, 一句话说明)
SHEETS = [
    ("01_实验矩阵", "report_experiment_matrix.csv",
     "实验级矩阵：8 个实验/臂的最佳点、榜分与训练信号",
     "「最佳官方」按 Kaggle 官方口径（0.3×冲突准确率）取最大；「榜分」只对真正提交过的检查点有值，其余为 —。"),
    ("02_评测明细", "curves_checkpoints_all.csv",
     "逐检查点评测指标（37 个检查点，all 子集）",
     "local_score 是训练选点用的口径（0.3×冲突召回），official_score 是 Kaggle 口径（0.3×冲突准确率），两者在多报冲突时反向。"),
    ("03_子集明细", "report_metrics_by_subset.csv",
     "五子集拆解（37 检查点 × all/VT/VA/original/generated = 185 行）",
     "VT=视觉+文本，VA=视觉+音频，original=原始样本，generated=生成样本。同一检查点在不同子集上的表现差异很大，汇报时按需引用。"),
    ("04_阶段聚合", "report_stage_summary.csv",
     "奖励侧按训练阶段聚合（6 个实验 × 5 段）",
     "grad_group_rate = 有梯度组占尝试组的比例；conflict_grad_share = 有梯度组里冲突样本的占比。PPO 无「同组候选」概念，组级列留空。"),
    ("05_优化侧阶段", "report_optimizer_stage.csv",
     "优化侧按阶段/epoch 聚合（只有 7B SFT 与 3B PPO 有原料）",
     "其余四个实验的训练脚本算了 loss/KL/grad_norm 但没有落盘，无法事后补算。"),
    ("06_奖励逐步", "curves_reward_per_step.csv",
     "逐训练步的奖励侧明细（2100 行）",
     "全部由 rollouts.jsonl 离线重算，含每组 8 候选的奖励均值/标准差/零方差比例。"),
    ("07_优化逐步", "curves_optimizer_side.csv",
     "逐步 loss / KL / 梯度范数（19985 行，仅 SFT 与 PPO）",
     "7B SFT 1638 步 × 11 项；3B PPO 300 attempt × 7 项。"),
    ("08_训练信号", "curves_train_signals.csv",
     "训练过程信号长表（31175 条）",
     "从日志与 cycles.jsonl 提取：updates、accept_rate、all_correct_groups、zero_update、clip 触发数等。"),
    ("09_阶段明细10段", "report_stage_fine.csv",
     "按十分位切分的阶段明细（7 个实验/臂 × 最多 10 段 = 70 行）",
     "grad_groups_n 是有梯度组的绝对条数——冲突占比在该数小于 5 时只由个位数样本决定，不要单独引用。"),
    ("10_数据来源", "report_traceability.csv",
     "每个数据块对应的产物文件、规模、生成脚本与原始来源",
     "汇报时被问「这个数怎么来的」，直接指这一张表；所有脚本只读已归档文件，可离线复现。"),
    ("11_完整性核查", "report_completeness.csv",
     "本地数据完整性逐项核查（2026-11-10 逐文件实测，37 行）",
     "「完整」= 本地已有且条数整齐；A 类可补（权重未归档）；B 类永久不可补（从未落盘）；C/D 类重复或过场、价值低。详见 docs/08。"),
]

NUM_FMT = {"local_score": "0.000000", "official_score": "0.000000",
           "official_delta_vs_sft": "+0.000000;-0.000000",
           "best_official": "0.000000", "last_official": "0.000000",
           "best_local": "0.000000", "sft_start_official": "0.000000",
           "kaggle_best": "0.000000", "value": "0.00000000"}


def style_header(ws, ncol, row=1):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def autofit(ws, ncol, max_width=34):
    for c in range(1, ncol + 1):
        width = 9
        for r in range(1, min(ws.max_row, 300) + 1):
            v = ws.cell(row=r, column=c).value
            if v is None:
                continue
            s = str(v)
            # 中文按 2 个字符宽估算
            w = sum(2 if ord(ch) > 127 else 1 for ch in s)
            width = max(width, min(w + 2, max_width))
        ws.column_dimensions[get_column_letter(c)].width = width


def add_csv_sheet(wb, name, csv_name, title, note):
    p = AUDIT / csv_name
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    ws = wb.create_sheet(name)

    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws["A2"] = note
    ws["A2"].font = NOTE_FONT
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=min(8, max(len(rows[0]) if rows else 6, 6)))
    ws.row_dimensions[2].height = 30

    hdr_row = 4
    if not rows:
        return ws
    cols = list(rows[0].keys())
    for i, c in enumerate(cols, start=1):
        ws.cell(row=hdr_row, column=i, value=c)
    style_header(ws, len(cols), hdr_row)

    for r_i, r in enumerate(rows, start=hdr_row + 1):
        for c_i, c in enumerate(cols, start=1):
            v = r[c]
            if v == "":
                v = None
            elif c in NUM_FMT or c in ("mean_reward", "std_reward", "within_group_std"):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    pass
            elif c in ("unit_value", "n", "TP", "TN", "FP", "FN", "n_units", "n_checkpoints",
                       "total_units", "groups", "accepted_groups", "candidates", "conflict_n",
                       "aligned_n", "all_fields_correct_n", "all_correct_groups",
                       "grad_groups", "grad_groups_n", "n_groups", "n_candidates",
                       "optimizer_steps", "token_limit_hits", "invalid_on_conflict",
                       "invalid_on_aligned", "progress_pct"):
                try:
                    v = int(float(v)) if float(v).is_integer() else float(v)
                except (TypeError, ValueError):
                    pass
            cell = ws.cell(row=r_i, column=c_i, value=v)
            cell.border = BORDER
            cell.font = Font(size=9, name="微软雅黑")
            if c in NUM_FMT and isinstance(v, float):
                cell.number_format = NUM_FMT[c]
            elif isinstance(v, float):
                cell.number_format = "0.0000"

    ws.freeze_panes = ws.cell(row=hdr_row + 1, column=1)
    ws.auto_filter.ref = f"A{hdr_row}:{get_column_letter(len(cols))}{hdr_row + len(rows)}"
    autofit(ws, len(cols))
    return ws


def build_readme(wb):
    ws = wb.active
    ws.title = "00_说明"
    ws["A1"] = "MM-Emotion 项目汇报数据包"
    ws["A1"].font = Font(bold=True, size=18, name="微软雅黑", color="1F3864")
    ws["A2"] = "五轮实验（3B GRPO 两轮 + 3B PPO + 7B SFT/GRPO + 7B DAPO 两臂）的结构化数据汇总"
    ws["A2"].font = Font(size=11, name="微软雅黑", color="374151")

    lines = [
        ("", ""),
        ("【口径】", ""),
        ("官方口径 official_score", "0.5×情绪宏F1 + 0.3×冲突准确率 + 0.2×一致特异度（与 Kaggle 评测一致）"),
        ("训练口径 local_score", "0.5×情绪宏F1 + 0.3×冲突召回 + 0.2×一致特异度（训练选点用这个）"),
        ("两者关系", "在多报冲突时必然反向；本项目 7 个真提交模型的本地排序与榜分排序 Spearman ρ = −0.107"),
        ("", ""),
        ("【各 sheet】", ""),
        ("01_实验矩阵", "8 个实验/臂一行：最佳官方分、最佳点、榜分、有效组比例"),
        ("02_评测明细", "37 个检查点的全部评测指标 + 混淆矩阵"),
        ("03_子集明细", "同一批检查点按 VT / VA / original / generated 四类样本拆开（185 行）"),
        ("04_阶段聚合", "训练前 20% / 20-40% / 40-60% / 60-80% / 末 20% 五段的奖励侧指标"),
        ("05_优化侧阶段", "loss / 各模态 loss / 梯度范数，按阶段与按 epoch（仅 SFT 与 PPO）"),
        ("06_奖励逐步", "2100 个训练步的组内奖励均值、方差、零方差比例、有效组数"),
        ("07_优化逐步", "19985 行逐步 loss/KL/clip/梯度范数"),
        ("08_训练信号", "31175 条日志信号（updates、接受率、全对组数、零更新、clip 触发）"),
        ("09_阶段明细10段", "每 10% 训练进度一段，含「有梯度组数」绝对条数（70 行）"),
        ("10_数据来源", "每个数据块 → 产物文件 → 规模 → 复算命令 → 原始来源（10 行）"),
        ("11_完整性核查", "本地有什么、缺什么、能不能补（37 行，含 24 个未归档的检查点权重）"),
        ("", ""),
        ("【数据来源】", "全部从本地归档的 *_metrics.json / rollouts.jsonl / cycles.jsonl / 训练日志离线重算，"),
        ("", "不重跑评测。生成脚本见 _audit/build_report_pack.py 与 build_all_curves.py。"),
        ("", ""),
        ("【已知缺口】", "3B GRPO r1/r2、7B GRPO、DAPO 两臂共 4 个实验的 loss/KL/grad_norm 从未落盘（脚本算了但没存），"),
        ("", "且全库无 wandb / tensorboard 记录 → 这四轮不存在 loss 曲线，只能引用奖励侧与组级信号。"),
        ("", "检查点权重另有 24 个（≈5.3 GB）未归档，只影响重新推理/接着训练，不影响任何指标复算。"),
        ("", ""),
        ("【数字可判性】", "734 条测试集上，模型间 0.001–0.006 的榜分差相当于个位数样本；小于 0.002 的差不宜当结论。"),
    ]
    r = 4
    for a, b in lines:
        ws.cell(row=r, column=1, value=a).font = Font(bold=a.startswith("【"), size=10, name="微软雅黑",
                                                      color="1F3864" if a.startswith("【") else "111827")
        ws.cell(row=r, column=2, value=b).font = Font(size=10, name="微软雅黑", color="374151")
        r += 1
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 96
    for row in range(4, r):
        ws.cell(row=row, column=2).alignment = Alignment(wrap_text=True, vertical="center")
    return ws


def main():
    wb = Workbook()
    build_readme(wb)
    for name, csv_name, title, note in SHEETS:
        ws = add_csv_sheet(wb, name, csv_name, title, note)
        print(f"  {'OK ' if ws else 'MISS'} {name:<14} <- {csv_name}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"\n-> {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(wb.sheetnames)} sheets)")


if __name__ == "__main__":
    main()
