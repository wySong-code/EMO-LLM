# Combined SFT → GRPO：策略、实现和验收

建立时间：2026-10-29 14:21（香港时间）。本目录是**新实验**，没有覆盖课程基线 SFT adapter，也没有宣称 GRPO 已训练或提分。代码针对 `MM-Emotion-Combined-SFT-20261026` 的目录结构编写。

## 1. 起点与目标

起点为 `model_repo/` 的 Combined SFT 第 3 轮 adapter。它从 Qwen2.5-Omni-3B 原始基座训练，完整任务输出 `vision_emotion`、`text_emotion` 或 `audio_emotion`、`is_conflict`、`pred_emotion`，并以单模态辅助任务训练 M1/M2。它**不直接生成** `is_uncertain` 或追问；提交阶段由 `is_conflict` 映射不确定性，追问另行生成。因此 GRPO 沿用同一四字段契约，不引入旧项目的四字段 `is_uncertain/clarifying_question` 奖励。

已存的 666 条验证指标：综合情绪 Macro-F1 0.94645、冲突命中 58/74、一致误报 50/592、总分 0.89147。冲突命中分解为原始来源 11/25、生成来源 47/49。理论剩余分数空间：情绪约 0.0268、冲突召回约 0.0649、一致校准约 0.0169。GRPO 首先尝试修复原始冲突漏报，并将情绪 F1 下降限制在 0.01 以内。

## 2. 数据边界

训练只读 `training_dataset/Dataset/train.jsonl` 的 3352 条（已由 SFT 版本排除 2 条与验证视频重复的记录）；验证只读固定 666 条。训练冲突 411 条，其中原始来源 143、生成来源 268。代码检查 ID 不重复、train/val ID 与视频路径不交叉、M1/M2 是否与 `sample_type` 一致，以及媒体是否存在。`training_dataset/media.tar` 已解压到 `training_dataset/Dataset/videos/`。

每个 rollout step 从训练集抽**一条冲突、一条一致**样本。冲突样本以 0.67 概率从原始来源抽取，其余从生成来源抽；一致样本目前均匀抽取。这是起始实验采样，不是根据 val 错例直接训练。采样用固定 seed 42，可重现。与旧项目中对两桶奖励做跨桶比较不同，优势只在同一输入的 8 个候选内部计算。

## 3. 奖励定义

只使用当前任务的可核对标注：

| 项 | 条件 | 权重 |
|---|---|---:|
| 综合情绪 | `pred_emotion == M12.label` | **0.35** |
| 画面情绪 | `vision_emotion == M1.label` | **0.10** |
| 第二模态情绪 | VT 的 `text_emotion` / VA 的 `audio_emotion` 等于 `M2.label` | **0.10** |
| 冲突决策 | `is_conflict` 等于 `sample_type` | **0.45** |

严格 JSON 解析要求字段集合与协议完全匹配；缺失、额外字段、重复键或非法取值给 `-0.20`。输出的两路情绪是否相异，应与 `is_conflict` 一致；不一致扣 `0.05`，但“自洽”本身不额外加分，以免奖励两个都错却彼此一致的答案。**没有独立的格式正奖励、追问奖励或 `is_uncertain` 奖励。**实现见 `reward.py`。

这个逐样本奖励是**代理目标**。官方 Macro-F1 在全数据集上计算，不能分解为逐条 0/1 的完全同构奖励；因此只用固定验证集的官方口径选 checkpoint。提高 `0.45` 不保证冲突训练强度同步提高，因为 GRPO 在每题内部标准化总奖励。

## 4. GRPO 更新细节

一个 step 有两个不同 prompt，各独立采样 8 个回答，共 16 个候选。默认温度 1.1、top-p 0.95、最多 128 个新 token。每题计算 `A_i=(R_i-均值)/(标准差+1e-6)`；冲突题在**标准化之后**乘 1.5，一致题乘 1.0。这样类别权重不会在同题标准化时被整体抵消。两个题的候选绝不混在一起算均值或方差。

同题 8 个答案奖励全同：

- 若四个字段全对、格式正确且自洽，记录 `all_correct`，不更新。
- 若未达上述条件，记录 `all_same_not_perfect`，本轮不制造跨题优势。后续通过提高候选多样性、改采样或追加监督数据处理；当前代码不伪造梯度。

优化器是 AdamW、学习率 `2e-6`。同一批采样答案做 2 次 PPO 风格更新，token 概率比裁剪区间 `[0.8,1.2]`；每次更新后对可训练参数裁剪梯度范数至 1。参考策略是**冻结的原 SFT adapter**，KL 系数 0.02；训练代码采用非负的采样 KL 估计。策略模型与参考模型分开加载，需先在真实设备上做显存预检。代码对整个生成答案计算策略梯度，格式差异由严格解析控制；如将来需要只更新标签 token，必须另外实现并验证 token 位置映射，不能假定当前版本已经做到。

这里没有沿用旧项目的 `PERFECT_R=1.05` 或跨桶兜底。一个全同但低于满分的桶，其组内优势本来就是 0；改阈值也无法创造训练信号。本方案四个正项合计 1.00，满分就是 1.00，因此 `1.19` 在这里尤其不适用。只有出现组内奖励差异的候选，才计算原策略与参考策略概率并参与反传；优化损失按实际参与的候选数归一。

## 5. 验证、选模与可归因实验

启动前先对 SFT 原 adapter 跑同一套 666 条验证，记录基线。默认每 50 个 rollout step 做一次完整 greedy 验证；总预算 300 step。为给同时驻留的冻结参考模型留出余量，默认验证 batch 为 4，可用 `--eval-batch-size` 调整。记录总分、情绪 Macro-F1、冲突召回、一致校准和协议/来源切片。`best.json` 只在**情绪 F1 不低于 SFT 基线 0.01**且总分严格超过当前最佳时更新；否则 SFT 保持最优。每个评测点的 adapter 仍独立保存，便于事后复核。`rollouts.jsonl` 记录每题 8 个奖励、决策正确性和零方差原因，不保存媒体内容或验证标签到训练输入。

建议按同样预算比较三组，而不是一次把全部策略叠上去：

1. **起始对照**：SFT 原模型与本目录默认 GRPO。
2. **奖励消融**：综合情绪/两路证据/冲突从 `0.35/0.10/0.10/0.45` 改为此前保守档 `0.50/0.10/0.10/0.30`；其余不动。
3. **采样消融**：保持奖励不变，比较原始冲突抽取概率 0.67 与按训练集自然来源比例 `143/411≈0.348`。不得用验证集错例加入训练。

判断是否采用新 checkpoint：验证总分提高，原始冲突 11/25 有实质改善，一致误报未明显扩大，情绪 F1 守住保护线。验证集只有 74 条冲突，单条命中就带来约 `0.3/74=0.00405` 的分数变化；小幅涨分应结合错误清单和多 seed 复验。Public 榜成绩应与本地验证分开记，不以一次提交反推策略优劣。

## 6. 运行方法

课程基线**有环境记录，但没有可直接复现的 `requirements.txt`、`environment.yml` 或容器镜像**。已记录的训练环境是 Linux + CUDA、torch `2.4.0+cu124`、transformers `4.57.6`、peft `0.17.1`、wandb `0.22.3`；原配置见 `model_repo/experiment/training_config.json`。GRPO 代码还依赖 accelerate、safetensors、av、numpy、soundfile、imageio-ffmpeg、Pillow 等，项目未锁定这些依赖的版本。建议在同类 Linux + CUDA 环境中运行，避免 Python 版本与 CUDA 轮子不匹配。

`python train_grpo.py --steps 1` 在缺依赖的环境里会在 `import torch` 处停止。这属于**环境阻断**，不能据此声称加载、生成、反传路径已跑通。因此设两级验收：`--check-env` 打印缺失依赖，`ready=false` 时返回非零退出码；`--smoke-test` 在目标设备上加载两份模型，对 VT 和 VA 各做一次真实生成、策略/参考 logprob 与 LoRA 反传，核对输出 JSON、视觉/语言/音频梯度，但不执行优化器更新，也不跑 666 条验证。只有烟测通过后再启动完整训练。

```powershell
cd 'MM-Emotion-Combined-SFT-20261026/2026-10-29_14-21-GRPO'
python -m unittest -v test_reward.py
python train_grpo.py --check-data
python train_grpo.py --check-env
python train_grpo.py --smoke-test --out 'MM-Emotion-Combined-SFT-20261026/grpo_runs/smoke_01'
python train_grpo.py --out 'MM-Emotion-Combined-SFT-20261026/grpo_runs/pilot_01'
```

脚本的模型、adapter 和数据默认路径均相对于脚本所在目录自动解析，例如：

```bash
cd /path/to/MM-Emotion-Combined-SFT-20261026/2026-10-29_14-21-GRPO
python train_grpo.py --check-data
python train_grpo.py --check-env
python train_grpo.py --smoke-test --out ../grpo_runs/smoke_01
python train_grpo.py --out ../grpo_runs/pilot_01
```

完整训练需留出两份模型同时加载的显存。如果显存不足，可先降低 `--eval-batch-size`；候选答案已经逐条生成，并不存在可继续降低的候选并行数。若仍不足，需实现“同一基座双 adapter 切换”或参考模型 CPU 卸载并重新验算 KL，不能直接删除参考约束后仍称其为相同实验。`--check-data` 和 `--check-env` 不接触设备，也不会创建训练输出。`--smoke-test` 会创建独立输出目录；若失败，请保留其中的 `config.json` 和终端错误，修正后使用新的 `--out` 路径重试。

## 7. 代码文件

- `reward.py`：严格解析、四分项奖励、同题优势和零方差分类。
- `train_grpo.py`：数据预检、两题各 8 候选、冻结参考策略、裁剪更新、验证与 checkpoint 选择。
- `test_reward.py`：格式、奖励值、组内优势和冲突后置权重的 CPU 测试。

已执行的离线检查：三个 Python 文件通过语法检查；3 个单元测试通过；`--check-data` 显示 train=3352、val=666、train_conflict=411、原始冲突=143、生成冲突=268。用全量 train+val 的标准答案核对，VT 2124 条、VA 1894 条共 4018 条的奖励均为 1.00。**训练代码尚未在任何真实设备上跑首步，因此加载、显存、生成与梯度路径仍需烟测。**
