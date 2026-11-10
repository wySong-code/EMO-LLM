# submission_attempt0300 — Kaggle 提交文件（GRPO 第二轮最优检查点）

| 项目 | 值 |
|---|---|
| 生成时间 | 2026-10-30 05:28 (UTC+8) |
| 检查点 | `runs/pilot_01/checkpoints/attempt_0300`（本轮最佳，综合评分 0.8959） |
| 基座 | Qwen2.5-Omni-3B + SFT adapter（Kaggle 0.93049）+ GRPO 第二轮 LoRA |
| 推理脚本 | `_deploy/make_submission.py`（未修改，与训练/评测同一套 prompt、8 帧 448px、严格四字段解析） |
| 测试集 | `~/grpo_mm/test_release.jsonl`，734 条（VT 387 / VA 347） |
| 耗时 | 2776.8 s（46.3 min） |
| CSV md5 | `2a60553665dcfa4dc6b01901aaa502f3` |

## 解析质量

- `parse_ok` 734 / 734，`parse_errors` 空，`fallback` 0 —— 模型输出全部为合法四字段 JSON。
- 无一条需要模板兜底。

## 分布

| 字段 | 计数 |
|---|---|
| pred_emotion = negative | 446 |
| pred_emotion = positive | 288 |
| is_uncertain = no | 600 |
| is_uncertain = yes | 134 |

## 与第一轮（step_0050）对比

734 条里只有 16 条发生变化：

- 情绪极性翻转 8 条（positive→negative 5、negative→positive 3）
- `is_uncertain` 变化 10 条
- `yes` 从 126 增至 134（+8），与验证集上「冲突召回上升、误报同步上升」的趋势一致

## 校验（9 项全 PASS）

行数 734、列名与第一轮完全一致、`sample_id` 唯一、id 集合与顺序均与第一轮一致、
`pred_emotion` ∈ {positive, negative}、`is_uncertain` ∈ {yes, no}、`clarifying_question` 非空、
`is_uncertain` 与 `clarifying_question`（非 none）严格对应。

## 说明

`clarifying_question` 由 `make_submission.py` 根据模型自己输出的 `vision/second modality` 标签
按模板拼接，模型不直接生成该字段（训练与评测均不含它），官方评分也只读
`pred_emotion` / `is_uncertain`。

## 文件

- `submission.csv` — 提交文件，734 行 + 表头
- `predictions_raw.jsonl` — 每条的原始模型输出（含四字段 JSON），便于复核
- `summary.json` — 解析统计与耗时
