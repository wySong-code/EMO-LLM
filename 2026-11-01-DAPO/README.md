# 7B SFT → 预算受控 DAPO 实验

这是从原 7B SFT 最优 adapter 接续的**新实验**，不会覆盖原 7B SFT / GRPO 结果。本地归档路径是 `sft_run/adapter_best`；若只有 `sft_run/train/checkpoints/checkpoint-1638`，入口也会自动回退到它。**先看[一页纸总览](docs/00_总览.md)**（改了什么 / 跟谁比 / 结论），细节再按需展开：[发现过程与奖励重设计](docs/04_得分差异复盘与奖励重设计.md)、[策略文档](docs/STRATEGY.md)、[验证文档](docs/VALIDATION.md)、[训练记录](docs/05_训练记录.md)、[训练日志摘要](docs/07_训练日志摘要.md)、[提交台账](docs/08_提交台账.md)、[结果分析](docs/06_结果分析.md)。

## 文件

- `code/train_dapo.py`：多模态训练器，继承已跑通的 7B GRPO 载入、预处理、分设备 logprob 与验证链路。
- `code/dapo_core.py`：训练集难度池、分层限额补采、候选组类型判定。
- `code/reward.py`：提供原 7B 奖励与面向提交分数的决策加强奖励，并在有决策分歧的组增加决策优势。
- `code/sft/`：原 7B SFT 辅助代码的副本，提供同一套视频/音频预处理与本地评分。
- `code/test_dapo.py`：不需要 torch 的单元测试。
- `scripts/run_dapo.sh`：入口脚本；先做数据、环境与前后向门禁，再正式训练。
- `scripts/run_submission.sh`：训练结束后按 `best.json` 导出 734 行提交 CSV；若 RL 没超过基线，自动使用 SFT adapter。
- `scripts/run_submission_ckpt.sh`：按**显式指定的检查点**导出提交（`<arm> <step> <name> [gpu]`）。`best.json` 用的是训练口径，官方口径最佳点通常不在一处，需要这样点名导出。

## 运行方式

`2026-11-01-DAPO` 与 7B 实验目录、`training_dataset/` **同级**，运行前需存在以下路径：

```text
~/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B/
~/grpo_mm/MM-Emotion-Combined-SFT-20261026/
  training_dataset/Dataset/{train.jsonl,val.jsonl,videos/}
  2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/{adapter_best/ 或 train/checkpoints/checkpoint-1638/}
  2026-11-01-DAPO/
```

进入新实验目录：

```bash
cd ~/grpo_mm/MM-Emotion-Combined-SFT-20261026/2026-11-01-DAPO
bash scripts/run_dapo.sh
```

可用后台方式运行：

```bash
mkdir -p logs
nohup bash scripts/run_dapo.sh > logs/launcher.log 2>&1 &
tail -f logs/launcher.log
```

入口会自动跑 `code/test_dapo.py`、`--check-data`、`--check-env`、真实前后向 smoke test，再开始 300 个**采样周期**。每周期最多试 4 个冲突组与 2 个一致组，每组 8 个回答；不保证每周期一定更新。默认奖励是 `score_focused`（融合情绪 0.35、分支各 0.025、冲突决策 0.60），有决策分歧时额外加入 0.5 倍标准化决策优势。已经存在 `runs/dapo7b` 或 `preflight/gpu_smoke_dapo7b` 时会拒绝覆盖。

产物在 `runs/dapo7b/`：`config.json`、`train_screen_predictions.jsonl`、`train_screen_summary.json`、`sft_baseline_metrics.json`、`rollouts.jsonl`、`cycles.jsonl`、每 50 周期的验证预测/指标、`checkpoints/`、`best.json`。日志在 `logs/`。

训练完成后可运行 `bash scripts/run_submission.sh`。脚本使用 `~/grpo_mm/test_release.jsonl` 和 `~/MM-LLMs-Emotion-main/Dataset`；路径不同时用 `DAPO_TEST`、`DAPO_TEST_MEDIA` 覆盖。

**奖励对照（即“第二臂”）**：原 7B 奖励可用 `DAPO_RUN_TAG=dapo7b_legacy DAPO_REWARD_PROFILE=legacy DAPO_DECISION_BONUS=0 bash scripts/run_dapo.sh` 运行，产物写入 `runs/dapo7b_legacy/`，与默认实验隔离。`DAPO_RUN_TAG` 也决定 smoke 目录和日志名；导出对应提交时沿用该变量。

**该对照已跑完**：两臂官方口径分差仅 **0.0004**（0.932950 vs 0.932570），说明奖励系数与决策优势这组改动无效，**原计划的单因素消融不必再做**。详见 [06_结果分析 §4.1](docs/06_结果分析.md)。

**可调路径**：`DAPO_PYTHON`、`DAPO_BASE`、`DAPO_ADAPTER`、`DAPO_DATA` 环境变量覆盖入口默认路径。若目录与上图不同，设置对应变量即可；无需修改训练代码。

## 结果判定

基线为同评测链路的 7B SFT，本地分 **0.905166**、官方口径 **0.940752**。三代方法（同一 SFT 起点、同一 666 条验证集、同一 `score()` 链路）对照如下，逐点数据见 `reports/three_way_comparison.csv`：

| 方法（按各口径最佳点） | local（0.3×冲突**召回**） | kaggle（0.3×冲突**准确率**） | TP/FP/FN |
|---|---|---|---|
| 7B SFT（基线） | 0.905166 | **0.940752** | 60/33/14 |
| 7B GRPO step150 | 0.909925（+0.0048） | 0.939655（−0.0011） | 61/38/13 |
| 7B DAPO step200（local 最佳） | **0.919887（+0.0147）** | 0.932950（−0.0078） | 64/51/10 |
| 7B DAPO step50（官方口径最佳） | 0.905124 | 0.937556（−0.0032） | 60/40/14 |
| 7B DAPO-legacy step300 | 0.918155（+0.0130） | 0.932570（−0.0082） | 64/48/9 |

**DAPO 与 GRPO 谁更好，取决于用哪个口径，而且结论相反**：训练口径下 DAPO 比 GRPO 高 0.0100，官方口径下低 0.0021。两个都低于 SFT。

**两个口径方向相反**：recall 涨（TP 60→64）、accuracy 跌（FP 33→51/48）。12 个评测点的官方口径分**全部低于 SFT 基线**。另有两点需要注意：①按官方口径，DAPO 两臂的最佳点都是 **step50**，训练越久官方分越低；②`best.json` 按 local 口径选出的 step200/300 恰是官方口径最差区间，选点逻辑必须改。

> ⚠️ **2026-11-05 勘误（第二轮，已上榜）**：两臂 step_0050 已提交 Kaggle，**榜分 0.93580（legacy）/ 0.93345（score_focused）**，**都高于 7B SFT 的 0.93243**——上文「12 个评测点全部低于 SFT」只在本地验证集成立，在榜单上被推翻。legacy 排全项目第二，与榜首 3B GRPO（0.93638）差 0.00058、与 7B GRPO step150（0.93577）差 0.00003，**低于 0.002 的可判阈值，属并列而非超越**。同时本地名次与榜上名次的 Spearman ρ 由 +0.10（n=5）变为 **−0.107（n=7）**。
>
> ⚠️ **2026-11-02 勘误**：本文档早期版本据此建议「提交用 SFT adapter」。该建议已被真实榜分否证——本地第一的 7B SFT 在 Kaggle public 只有 0.93243，而本地垫底的 3B GRPO step50 是 0.93638（榜首）。本地验证集与榜分排序 Spearman ρ 当时算得 0.10（n=5，补入 DAPO 两臂后为 −0.107），**不能用于选提交物**。详见 [06_结果分析 §7](docs/06_结果分析.md) 与仓库根 `docs/03_审计与勘误记录.md`。台账见 [08_提交台账](docs/08_提交台账.md)。

机制、归因与下一步见 [06_结果分析](docs/06_结果分析.md)，时间线与产物见 [05_训练记录](docs/05_训练记录.md)。

