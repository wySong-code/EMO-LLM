# 检查记录与启动门禁

## 已完成的离线检查（2026-11-01）

1. `python -m py_compile code/train_dapo.py code/dapo_core.py code/reward.py code/test_dapo.py code/analyze_run.py`：通过。
2. `python code/test_dapo.py`：5 个单元测试通过，覆盖零方差/决策分歧组、训练 ID 隔离、冲突与一致分桶、严格奖励与优势，以及新旧奖励权重和决策优势的方向。
3. `python code/train_dapo.py --check-data`：通过；训练 3352、验证 666，训练冲突 411（真实 143、合成 268）、验证冲突 74；验证无同视频或同 ID 交叉，媒体路径均存在。
4. 核实本地 `2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/adapter_best` 含 `adapter_config.json`、`adapter_model.safetensors`、tokenizer/processor 配置。它是 `sft_run/train/checkpoints/checkpoint-1638` 的改名副本；因此入口先找 `adapter_best`，找不到再回退到原 checkpoint。
5. 脚本、代码及文档均放在与 7B 实验及 `training_dataset/` 同级的新目录。没有改动旧实验文件。
6. 对比 SHA-256：`overfit_omni_modality.py`、`evaluate_omni_batched.py`、`omni_modality_task.py` 与原 7B 实验逐字相同。**2026-11-01 奖励修订后，`reward.py` 已不再与原 7B GRPO 相同**；`legacy` 配置保留原四项系数与默认零决策优势，可作对照。

## 奖励修订后的离线复核

- `python -m compileall -q code`：通过。
- `python code/test_dapo.py`：5/5 通过。
- `python code/train_dapo.py --check-data --reward-profile score_focused --decision-advantage-bonus 0.5`：通过，训练 3352 / 验证 666，抽样 100 周期无泄漏。
- `python code/train_dapo.py --check-data --reward-profile legacy --decision-advantage-bonus 0`：通过，划分和抽样与上组相同。

上述只验证 Python 逻辑和数据路径；**未验证任何新奖励配置的真实前后向或实际提分**。〔2026-11-01 14:05 后已补上，见下节〕

上述检查只覆盖逻辑与数据路径，不能证明 7B 的完整前后向、峰值内存或最终分数。**不能把“逻辑检查通过”写成“训练成功”。**

**2026-11-01 14:05 更新**：四道门禁**全部通过**后才开始训练（详见 [05_训练记录](05_训练记录.md) 时间线）：

- 单元测试 + `DATA_CHECK`：训练 3352 / 验证 666，抽样隔离通过；
- 环境：torch 2.8.0、CUDA、依赖齐备；
- 真实前后向 smoke（`preflight/gpu_smoke_<tag>/smoke_test.json`）：VT / VA 各 greedy 生成、金标与反例打分、非对称裁剪与 token 级 loss 前后向、LoRA 梯度分支。

两臂各跑满 300 周期，`grep -ciE 'error|traceback|nan'` = 0。**原“smoke 待执行”的记录已作废，四道门禁均已实机通过。**

## 启动时自动执行的阻断门禁

`scripts/run_dapo.sh` 在正式训练之前按顺序执行：

1. 检查 7B 基座、SFT adapter、train/val 路径与 Python 可执行文件。
2. 执行纯 Python 单元测试及全数据/媒体核验。
3. 验证环境依赖和 CUDA。
4. 单独加载 7B policy / reference，分别对 VT、VA 样本做 greedy 生成、两个金标/反例候选打分、DAPO 非对称裁剪与 token 级 loss 前后向、LoRA 梯度分支核验；**不执行优化器步**。结果落在 `preflight/gpu_smoke_<DAPO_RUN_TAG>/smoke_test.json`。
5. 只有前四项全部成功，才创建 `runs/dapo7b/` 并开始训练。

启动失败时检查 `logs/gpu_smoke_<DAPO_RUN_TAG>.log`、`logs/dapo_train_<DAPO_RUN_TAG>.log`。`runs/<DAPO_RUN_TAG>/` 或 `preflight/gpu_smoke_<DAPO_RUN_TAG>/` 已存在时拒绝覆盖；重跑请设置新 `DAPO_RUN_TAG`，避免把不同实验混写。

## 训练后应核验

```bash
python code/analyze_run.py --run runs/dapo7b
```

对比基线时，不能只看 `cycles_with_update`。至少核对：

- 每个周期的 `attempted_groups`、`active_groups`、`decision_active_groups`，以及总生成回答数和墙钟；
- 74 条验证冲突中的真实 25 条与合成 49 条分别的命中；
- 592 条一致样本的误报与情绪 Macro-F1；
- `clip_upper/clip_lower` 是否真的触发：若始终为零，不把 Clip-Higher 写成收益来源；
- 最佳检查点是否真的超过同评测链路的 SFT 基线及旧 GRPO 峰值，并做配对重采样检验。

结果若不显著或恶化，应如实归档负结果。论文 DAPO 的数学题增益不能直接外推到本任务的短 JSON 多模态情感识别。
