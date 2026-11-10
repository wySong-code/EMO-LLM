# step_0050 GRPO 模型 → 测试集提交文件

生成时间 2026-10-29 22:49（北京）｜总耗时 19 分 13 秒
模型：原始 Qwen2.5-Omni-3B + `checkpoints/step_0050` 适配器（GRPO 最优点，验证集综合 0.8934）

## 文件

| 文件 | 说明 |
| --- | --- |
| `submission.csv` | **提交文件**，734 行 + 表头，UTF-8 无 BOM、LF 行尾，md5 `69db9d69…` |
| `predictions_raw.jsonl` | 734 条原始输出，含模型原始 JSON、解析状态、token 数，便于审计 |
| `summary.json` | 运行汇总（条数、解析成功数、情绪与不确定性分布、耗时） |

## 提交前校验（课程基线自带的 `validate_submission.py`，未归档）

```
行数(数据): 734 / 模板 734
情绪分布: {'negative': 444, 'positive': 290}
不确定率: 126/734 = 17.17%
结果: PASS —— 可直接提交
```

七项硬校验全部通过：UTF-8 无 BOM、无 CR、表头与模板一致、行数一致、sample_id 集合与顺序一致、
每行 4 列、取值合法。

## 推理链路（与训练/评测严格同源）

| 环节 | 做法 |
| --- | --- |
| 预处理 | 直接 import 项目的 `TenInputs`（8 帧、长边 448、crf 18；VA 抽 16k 单声道） |
| System prompt | 直接 import `omni_modality_task.system_prompt`，非另写 |
| 生成 | `do_sample=False` 贪心，`max_new_tokens=192`，bf16 + SDPA |
| 解析 | 直接 import `omni_modality_task.parse_answer`（严格四字段，不容错） |
| transformers | 4.57.6，与训练期完全一致 |

因为 prompt、抽帧、解析三处都复用同一份代码，测试集输出与验证集评测处于同一分布。

## 四字段 → 提交列映射

| 提交列 | 来源 |
| --- | --- |
| `pred_emotion` | 四字段 JSON 的 `pred_emotion` |
| `is_uncertain` | 四字段 JSON 的 `is_conflict`（yes/no） |
| `clarifying_question` | 由 `vision_emotion` 与第二模态情绪按模板生成；无冲突时填 `none`。官方评分不评此列 |

## 结果分布

| 切片 | 条数 | 判 yes | 占比 |
| --- | ---: | ---: | ---: |
| ch_sims_v2 / VT | 307 | 38 | 12.4% |
| ch_sims_v2 / VA | 336 | 18 | 5.4% |
| gen / VT | 80 | 59 | 73.8% |
| gen / VA | 11 | 11 | 100.0% |
| 合计 | 734 | 126 | 17.2% |

## 与验证集切片的对照（证明测试集行为未漂移）

验证集 666 条上，同一适配器的表现：

| 切片 | 条数 | 真冲突 | 判 yes | 命中 | 误报 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ch_sims_v2 / VT | 284 | 20 | 39 | 9 | 30 |
| ch_sims_v2 / VA | 316 | 5 | 22 | 2 | 20 |
| gen / VT | 61 | 44 | 43 | 42 | 1 |
| gen / VA | 5 | 5 | 5 | 5 | 0 |

测试集上各切片的判 yes 率与验证集同一切片基本同量级，说明没有「换数据就崩」。

## 没有做后处理

在验证集上试过三档后处理，全部不划算，故提交文件是模型原始输出：

| 后处理 | 验证集综合分 | 相对原始 |
| --- | ---: | ---: |
| 原始（本次提交） | 0.8934 | — |
| 砍 ch_sims VA 的 yes | 0.8921 | −0.0014 |
| 砍 gen accept_c 的 yes | 0.8937 | +0.0003（噪声级） |
| 砍全部 VA 的 yes | 0.8718 | −0.0216 |

注意这与 r1 那版不同：r1 在 ch_sims VA 段的 yes 全落空，所以砍掉能赚 +0.0054；而 step_0050 在该段
22 条 yes 里命中 2 条，砍掉反而亏。

## 复现

```bash
python _deploy/make_submission.py \
  --base ../base_model \
  --adapter grpo_runs/pilot_01/checkpoints/step_0050 \
  --code ../../model_repo/code --processor-from ../../model_repo \
  --media-root ../../training_dataset/Dataset \
  --test ../../test_release.jsonl \
  --out submission_step50
```

734 条约 19 分钟（约 1.5 秒/条）；脚本 `_deploy/make_submission.py` 未改动项目任何源文件。
