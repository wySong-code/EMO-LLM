# Combined SFT 项目数据下载

本目录包含以下三个独立 Hugging Face 仓库的本地快照：

| 本地目录 | 来源 | 内容 |
|---|---|---|
| `model_repo/` | `xiaomoqingfeng/MM-Emotion-Qwen2.5-Omni-3B-SFT-Combined-20261026` | 第 3 轮 LoRA adapter、processor/tokenizer、训练与评测代码、配置和结果 |
| `training_dataset/` | `xiaomoqingfeng/MM-Emotion-Combined-SFT-20261026` | train/val、派生单模态样本、prompts、清单和 `media.tar` |
| `base_model/` | `Qwen/Qwen2.5-Omni-3B` | 运行 LoRA 所需的原始基座权重及配置 |

建议阅读顺序：`model_repo/README.md` → `model_repo/experiment/training_config.json` → `training_dataset/data/prompts.json` → `model_repo/code/train_omni_combined_sft.py`。模型卡中的示例需将基座路径设为本目录的 `base_model/`，adapter 设为 `model_repo/`。

这份下载覆盖**该 Combined SFT 版本**，不代表同组所有 SFT、GRPO、DPO 实验；它没有第 1、2 轮 checkpoint，也没有课程测试集标签。桌面已有的 `Dataset/` 是另外一份原始课程数据，并未在此覆盖。

`.hf-cache/` 与各仓库内 `.cache/` 为下载元数据；确认不需断点续传后可清理。核验结果见 `training_dataset/下载核验.json`。
