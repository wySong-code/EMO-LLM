# EMO-LLM 本地 Gradio Demo

这是一个纯 Gradio 的单进程 Demo：页面打开时自动加载模型，模型和 LoRA 可以在页面顶部选择，点击「加载模型」可以重新加载。

## 文件

- `app.py`：Gradio 界面与本地模型加载/推理
- `samples.json`：13 个验证集样本、VT/VA prediction 与 gold
- `model_options.json`：模型与 LoRA 配置
- `requirements-gradio.txt`：Gradio 依赖

## 模型与数据

`model_options.json` 同时保留两种用法：

- 本地已有权重时，设置 `EMO_ASSETS_ROOT` 指向包含 `base_3b/`、`model_3b/`、`code_sft/`、`data/Dataset/` 的目录
- 没有本地权重时，`base_model` / `repo_id` 使用 Hugging Face 仓库（需要下载约 12 GB 基座）

如果没有设置 `EMO_ASSETS_ROOT`，默认查找仓库根目录下的 `demo_assets/`。样本视频从 `EMO_DATA_ROOT` 或 `demo_assets/data/Dataset/` 读取，不随仓库提交。

公开仓库不包含模型权重、课程数据或视频。

## 启动

```bash
# 1. 先准备 Python 环境（需 torch、transformers、peft、av、soundfile）
# 2. 安装 Gradio
pip install -r demo/requirements-gradio.txt

# 3. 指向本地资产（有本地权重时）
EMO_ASSETS_ROOT=/path/to/demo_assets EMO_DATA_ROOT=/path/to/demo_assets/data/Dataset \
  python demo/app.py
```

Windows PowerShell：

```powershell
$env:EMO_ASSETS_ROOT="F:\path\to\demo_assets"
$env:EMO_DATA_ROOT="F:\path\to\demo_assets\data\Dataset"
python demo\app.py
```

没有本地权重时，`model_options.json` 里的 Hugging Face 仓库会被 `from_pretrained` 自动下载；不设置 `EMO_DATA_ROOT` 时，样本视频需要放在 `demo_assets/data/Dataset/`。

打开 `http://127.0.0.1:7860/`。

验证环境与耗时（2026-11-23）：

| 项 | 值 |
|---|---|
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB 显存 |
| 运行时 | torch 2.8.0+cu126，CUDA 12.6 |
| 模型 | Qwen2.5-Omni-3B + LoRA checkpoint-1638，基座权重约 12 GB |
| 页面自动加载模型 | 约 30–40 秒 |
| 单样本 VT 推理 | 约 2–2.5 分钟 |

8 GB 显存放不下完整模型时，accelerate 会把部分层放到 CPU 和磁盘（offload），所以推理时间明显长于纯 GPU 推理。

## 页面功能

- 13 个验证集样本，支持 sample ID 或文本搜索
- 模型 / LoRA 下拉框，旁边有「加载模型」按钮
- 页面首次打开自动加载默认模型，状态显示「模型已加载」
- VT / VA 两种协议使用完整中文名称
- Gold 表为 `字段 / Gold / 模型 / 结果`
- 绿色 ✓：一致；黄色 △：前两个模态错误导致的连带偏差；红色 ×：错误
- 推理时显示 Gradio 默认进度条和耗时，按钮不可重复点击
- 所有防御性/兜底说明集中在 HTML 末尾的不可见容器
