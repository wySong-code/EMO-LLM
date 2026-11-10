# submission_attempt0100 — PPO 最优检查点生成的提交文件

## 这是什么

用 PPO 训练（`2026-10-30_13-39-PPO`）选出的最优检查点 **attempt_0100**，在完整测试集上重新推理得到的 Kaggle 提交文件。

| 项 | 值 |
|---|---|
| 起点适配器 | 课程基线 Combined-SFT（Qwen2.5-Omni-3B Thinker + LoRA） |
| 训练算法 | PPO（学习式批评家 + GAE + 逐 token 裁剪） |
| 选用检查点 | `runs/pilot_01/checkpoints/attempt_0100` |
| 验证集综合评分 | 0.8908（SFT 基线 0.8840，+0.0068） |
| 推理耗时 | 1127.4 秒（734 条，1.54 秒/条） |
| 解码方式 | 贪心（温度 0），单条，可复现 |

## 文件

| 文件 | 内容 |
|---|---|
| `submission.csv` | **提交文件**，734 行 + 表头，列序 `sample_id, pred_emotion, is_uncertain, clarifying_question` |
| `summary.json` | 推理侧统计：解析成功率、耗时、标签分布 |
| `predictions_raw.jsonl` | 每条样本的原始模型输出（未后处理），用于复查 |
| `validate.json` | 11 项提交校验结果（全部 PASS） |

## 校验结果

11 项全部通过：

- 表头与第一轮提交完全一致
- 行数 734，`sample_id` 唯一（734/734）
- `sample_id` 集合与顺序均与第一轮一致（可作为直接替换）
- `pred_emotion` ∈ {positive, negative}
- `is_uncertain` ∈ {yes, no}
- `clarifying_question` 无空值；`yes` 必带问题、`no` 恰为 `none`，零违规

模型侧：734 条全部解析成功（`parse_ok=734`），**解析失败 0 条、回退默认值 0 条、跳过 0 条**。

## 标签分布

| | GRPO 第一轮 | GRPO 第二轮 | PPO（本文件） |
|---|---|---|---|
| negative | 444 | 446 | **443** |
| positive | 290 | 288 | **291** |
| is_uncertain = yes | 126 (17.2%) | 134 (18.3%) | **128 (17.4%)** |
| is_uncertain = no | 608 | 600 | **606** |

与 GRPO 第二轮提交相比，极性标签只变了 **7 条（0.95%）**，不确定性只变了 **8 条（1.09%）**。
即本轮 PPO 在测试集上是一个**保守的微调**，不是行为翻转。

值得记一笔：验证集上特异度从 0.9139 降到 0.9088（误报 51→54），但测试集的"多问一句"比例反而是 17.4%，
低于第二轮的 18.3%、接近第一轮的 17.2%。说明验证集上的特异度波动有一部分是采样噪声，
不宜据此断定模型在真实分布上变得更爱追问。

## 复现

```bash
# 从最佳检查点重新生成（约 19 分钟）
~/grpo_mm/MM-Emotion-Combined-SFT-20261026/_deploy/run_submit3.sh attempt_0100 submission_attempt0100
```

训练全流程见上级目录 `策略、实现与运行说明.md` 与 `PPO训练策略与结果记录_20261030.docx`。
