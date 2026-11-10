# 跨模态情感冲突判别（Affective Conflict Detection）

在 Qwen2.5-Omni（3B / 7B）上做**单模型四字段结构化输出**的多模态情感理解，后训练方法覆盖 SFT 与 GRPO / PPO / DAPO。

---

## 一、任务描述

单个模型一次推理输出严格 JSON 四字段，避免多模型级联的误差累积：

```json
{"vision_emotion": "negative", "text_emotion": "neutral",
 "is_conflict": "yes", "pred_emotion": "negative"}
```

`is_conflict` 表示画面与另一模态（文本或音频）情绪是否矛盾；`pred_emotion` 是融合判定。

| 项 | 值 | 说明 |
|---|---|---|
| 训练集 | 3,352 条 / 冲突 411 条（12.3%） | 由成对样本派生为四池后共 8,730 条，按 2/3 : 1/9×3 混批 |
| 验证集 | 666 条 / 冲突 74 条（11.1%） | 本地唯一有标签的评测集 |
| 测试集 | 734 条 | Kaggle 提交，public 约 70% |
| 监督序列 | 恒 26 token | 只有 4 个标签位参与损失 |

数据分布要点：

1. 冲突类占 12.3%，原始来源内部低至 4.8%；
2. 78% 样本落在重复文本上（同一句最多 9 次），语言先验本身即是一条捷径；
3. 融合判定「中性」类仅 7 条，标签空间实际接近二分类。

评测口径：0.5×情绪宏 F1 + 0.3×冲突识别准确率 + 0.2×一致特异度；官方打分与本地打分函数的差异见 `docs/02_关键发现.md` 与 `docs/01_实验结果总表.md`。

---

## 二、项目架构

```
.
├── README.md                       ← 任务描述与项目架构
│
├── docs/                           ← 跨实验文档（唯一文档入口，编号 01–08 连续）
│   ├── 00_文档索引.md              ← 从哪篇读起：按用途 / 按问题两条查法
│   ├── 01_实验结果总表.md          ← 37 个检查点的唯一数字源
│   ├── 02_关键发现.md              ← F1–F10 十条结论
│   ├── 03_审计与勘误记录.md        ← 11 处问题与结论变更
│   ├── 04_实验与文档审计.md        ← 五轮实验的文档覆盖矩阵 + 归档完整性
│   ├── 05_训练曲线数据清单.md      ← 三层曲线（评测/奖励侧/优化侧）谁有谁没有
│   ├── 06_项目汇报数据手册.md      ← 汇报用：按主题组织（口径、结论、组合建议）
│   ├── 07_分实验数据附录.md        ← 汇报用：按实验组织（十分位阶段表 + 溯源）
│   ├── 08_本地数据完整性核查.md    ← 逐文件实测：本地有什么、缺什么、能不能补
│   ├── 汇报数据包.xlsx             ← 12 个 sheet 的汇总数据包（汇报直接打开）
│   └── figures/                    ← 8 张曲线图（_audit/plot_curves.py 生成）
│
├── _audit/                         ← 可复算的审计脚本与产物（只读本地归档）
│   ├── official_score_bootstrap.py ← 历史检查点官方口径重算 + 配对 bootstrap
│   ├── sft_epoch_curve.py          ← 7B SFT 三 epoch 曲线 + 提交集差异
│   ├── build_master_table.py       ← 汇总 35 个检查点成一张表
│   ├── build_all_curves.py         ← 37 个检查点评测曲线 + 组级训练信号
│   ├── build_reward_curves.py      ← 从 rollouts 重算逐步奖励曲线（2100 点）
│   ├── build_optimizer_curves.py   ← loss/KL/梯度范数（仅 SFT 与 PPO 有）
│   ├── build_report_pack.py        ← 汇报表：子集明细 / 阶段聚合 / 实验矩阵
│   ├── build_appendix.py           ← 分实验附录：十分位阶段明细 + 数据溯源表
│   ├── build_completeness.py       ← 数据完整性核查表
│   ├── build_report_xlsx.py        ← 打包成 docs/汇报数据包.xlsx
│   ├── plot_curves.py              ← 由上述 CSV 出图
│   ├── check_privacy.py            ← 隐私自检：个人路径/网络地址/机构名称/密钥
│   ├── check_links.py              ← 文档引用体检（markdown 链接与带目录的路径）
│   └── *.csv                       ← 上述脚本的输出
│
├── 2026-10-29_14-21-GRPO/          ← 3B GRPO 第一轮（含 结果分析.md）
├── 2026-10-29_23-22-GRPO/          ← 3B GRPO 第二轮（配额改造）
├── 2026-10-30_13-39-PPO/           ← 3B 真 PPO
├── 2026-10-30_23-00-Omni7B-SFT-GRPO/ ← 7B 迁移 + GRPO（docs/ 套件式，01–03）
├── 2026-11-01-DAPO/                ← 7B DAPO 两臂（docs/ 套件式，00–08 + 可行性与改造方案.md）
│
├── base_model/                     ← 基座（3 分片，11.4 GB，可直接加载）
├── model_repo/                     ← 3B SFT 完整仓库（final + checkpoint-1638）
├── training_dataset/               ← 训练与测试数据（2.5 GB，含 media.tar；核验结果在同目录）
│
└── README_下载说明.md              ← 数据获取方式与目录约定（数据集本身不入库）
```

每个实验目录内自成体系：`策略、实现与运行说明.md`（总览）+ `docs/`（策略 / 训练记录 / 结果分析）+ `code/`（SFT 与 RL 实现）+ `reports/`（指标 JSON / CSV）+ `submissions/`（Kaggle 提交文件）。

文档入口是 `docs/00_文档索引.md`，可按用途或按问题两条路径检索。
