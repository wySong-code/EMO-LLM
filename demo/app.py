"""EMO-LLM 本地 Gradio Demo（单进程，不使用旧 HTML）

启动：
    demo_assets/venv/Scripts/python.exe demo/redesign/app.py
    # 或先激活 demo_assets/venv 后
    python demo/redesign/app.py

页面打开后会自然加载本地模型；模型和 LoRA 选择框旁边有“加载模型”按钮
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
_DEFAULT_ASSETS = WORKSPACE / "demo_assets"
ASSETS_ROOT = Path(
    os.environ.get("EMO_ASSETS_ROOT", str(_DEFAULT_ASSETS if _DEFAULT_ASSETS.exists() else WORKSPACE))
).resolve()
DATA_ROOT = Path(os.environ.get("EMO_DATA_ROOT", str(ASSETS_ROOT / "data" / "Dataset"))).resolve()
SAMPLES = json.loads((HERE / "samples.json").read_text(encoding="utf-8"))
SAMPLE_BY_LABEL = {item["label"]: item for item in SAMPLES}
MODELS = json.loads((HERE / "model_options.json").read_text(encoding="utf-8"))["models"]
MODEL_BY_LABEL = {item["label"]: item for item in MODELS}
DEFAULT_LABEL = SAMPLES[0]["label"]

GREEN = "background:#d1fadf;color:#05603a;padding:2px 7px;border-radius:4px;font-weight:600"
YELLOW = "background:#fef0c7;color:#b54708;padding:2px 7px;border-radius:4px;font-weight:600"
RED = "background:#fee4e2;color:#b42318;padding:2px 7px;border-radius:4px;font-weight:600"

DEFENSE = """
<div id="defensive-copy" hidden aria-hidden="true" style="display:none" data-purpose="defensive-copy">
  <p>本页面为课程作业 Demo，示例来自验证集，单样本输出不代表整体准确率</p>
  <p>“真实推理”表示本机模型直接返回的输出；“真实推理结果回放”表示使用已保存的真实模型结果，不表示浏览器中正在运行模型</p>
  <p>模型加载失败、接口超时或视频格式不支持时，保留本地结果与封面，不伪造实时推理</p>
  <p>页面不收集、不上传任何用户数据；样例视频仅用于课程演示</p>
  <p>课程数据仅限本课程作业使用，不要外传或上传公开平台</p>
</div>
"""

_MODEL = {
    "key": None,
    "model": None,
    "processor": None,
    "inputs": None,
    "task": None,
    "by_id": None,
    "device": None,
}


def get_sample(label: str) -> dict:
    return SAMPLE_BY_LABEL.get(label, SAMPLES[0])


def model_labels() -> tuple[list[str], list[str], str, str]:
    model_names = [item["label"] for item in MODELS]
    default_model = MODELS[0]
    lora_names = [adapter["label"] for adapter in default_model["adapters"]]
    return model_names, lora_names, default_model["label"], lora_names[0]


def selected_assets(model_label: str, lora_label: str) -> tuple[str, str]:
    model = MODEL_BY_LABEL.get(model_label, MODELS[0])
    adapter = next(
        (item for item in model["adapters"] if item["label"] == lora_label),
        model["adapters"][0],
    )
    base_path = (ASSETS_ROOT / model["base_path"]).resolve()
    adapter_path = (ASSETS_ROOT / adapter["path"]).resolve()
    return (
        str(base_path) if base_path.exists() else model["base_model"],
        str(adapter_path) if adapter_path.exists() else adapter["repo_id"],
    )


def video_path(item: dict) -> str:
    return str((DATA_ROOT / item["video"]).resolve())


def choice_label(item: dict) -> str:
    kind = "冲突" if item.get("sample_type") == "Conflict" else "一致"
    return f"{kind} · {item['label']} · {item['text'].replace('。', '')[:28]}"


def sample_id(item: dict, protocol: str) -> str:
    return item["sample_vt"] if protocol == "VT" else item["sample_va"]


def field_keys(protocol: str) -> list[str]:
    second = "text_emotion" if protocol == "VT" else "audio_emotion"
    return ["vision_emotion", second, "is_conflict", "pred_emotion"]


def field_names(protocol: str) -> list[str]:
    second = "文本情绪" if protocol == "VT" else "音频情绪"
    return ["视觉情绪", second, "模态冲突", "最终情绪"]


def match_cell(key: str, prediction: dict | None, gold: dict, protocol: str) -> str:
    if prediction is None:
        return "—"
    if str(prediction.get(key)) == str(gold.get(key)):
        return f'<span style="{GREEN}" title="一致">✓</span>'
    bases = field_keys(protocol)[:2]
    base_wrong = any(str(prediction.get(k)) != str(gold.get(k)) for k in bases)
    if key in field_keys(protocol)[2:] and base_wrong:
        return f'<span style="{YELLOW}" title="受前两个模态错误连带影响">△</span>'
    return f'<span style="{RED}" title="错误">×</span>'


def gold_table(item: dict, protocol: str, prediction: dict | None = None) -> list[list[str]]:
    gold = item[protocol]["gold"]
    rows = []
    for key, name in zip(field_keys(protocol), field_names(protocol)):
        rows.append([
            name,
            str(gold.get(key, "")),
            "—" if prediction is None else str(prediction.get(key, "")),
            match_cell(key, prediction, gold, protocol),
        ])
    return rows


def load_local_model(model_label: str, lora_label: str) -> str:
    base_path, adapter_path = selected_assets(model_label, lora_label)
    key = (str(base_path), str(adapter_path))
    if _MODEL["key"] == key and _MODEL["model"] is not None:
        return f"模型已加载 · {model_label} / {lora_label}"

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    code_root = (ASSETS_ROOT / "code_sft").resolve()
    if not code_root.exists():
        code_root = (WORKSPACE / "2026-10-30_23-00-Omni7B-SFT-GRPO" / "code" / "sft").resolve()
    sys.path.insert(0, str(code_root))
    from overfit_omni_modality import TenInputs
    import omni_modality_task as task

    processor = AutoProcessor.from_pretrained(str(base_path))
    tokenizer = processor.tokenizer
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    offload_dir = Path(tempfile.mkdtemp(prefix="emo-llm-offload-")).resolve()
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=False,
        device_map="auto",
        offload_dir=str(offload_dir),
    )
    model.eval()
    val_path = (DATA_ROOT / "val.jsonl").resolve()
    val_rows = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    media_root = DATA_ROOT
    cache_root = (ASSETS_ROOT / "media_cache").resolve()
    inputs = TenInputs(processor, media_root, cache_root, val_rows)
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    _MODEL.update({
        "key": key,
        "model": model,
        "processor": processor,
        "inputs": inputs,
        "task": task,
        "by_id": {row["sample_id"]: row for row in val_rows},
        "device": device,
        "engine": f"{model_label} / {lora_label}",
    })
    return f"模型已加载 · {model_label} / {lora_label}"


def local_infer(sample: str, protocol: str) -> dict:
    if _MODEL["model"] is None:
        raise RuntimeError("模型尚未加载")
    import torch

    row = _MODEL["by_id"].get(sample)
    if row is None:
        raise KeyError(f"sample_id not in val.jsonl: {sample}")
    started = time.monotonic()
    batch = _MODEL["inputs"].build(row, audio_first=False, training=False)
    device = _MODEL["device"]
    batch = {
        key: (value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
    with torch.inference_mode():
        generated = _MODEL["model"].generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=192,
            use_cache=True,
            eos_token_id=_MODEL["processor"].tokenizer.eos_token_id,
            pad_token_id=_MODEL["processor"].tokenizer.pad_token_id,
        )
    offset = batch["input_ids"].shape[1]
    tokens = generated[0, offset:].tolist()
    eos = _MODEL["processor"].tokenizer.eos_token_id
    if eos in tokens:
        tokens = tokens[:tokens.index(eos) + 1]
    raw = _MODEL["processor"].decode(tokens, skip_special_tokens=True)
    prediction, error = _MODEL["task"].parse_answer(raw, row["protocol"])
    return {
        "prediction": prediction,
        "parse_error": error,
        "raw_output": raw,
        "elapsed_s": round(time.monotonic() - started, 2),
        "engine": _MODEL.get("engine", "Qwen2.5-Omni-3B + LoRA"),
        "device": str(device),
    }


def initial_state(label: str, protocol: str):
    item = get_sample(label)
    sid = sample_id(item, protocol)
    return (
        gr.update(value=video_path(item)),
        gr.update(value=item["text"].replace("。", "") if protocol == "VT" else "", visible=protocol == "VT"),
        "### 等待开始",
        gold_table(item, protocol),
        "### 冲突解释\n等待推理后显示\n\n### 澄清追问\n等待推理后，在这里显示当前实现会追问的问题",
        "",
        "推理耗时：—",
        f"样本 `{sid}` 已就绪",
        gr.update(interactive=True, value="开始推理"),
    )


def load_with_progress(model_label: str, lora_label: str):
    yield "模型加载中…", gr.update(interactive=False, value="加载中…")
    try:
        message = load_local_model(model_label, lora_label)
        yield message, gr.update(interactive=True, value="加载模型")
    except Exception as exc:
        yield f"模型加载失败：{type(exc).__name__}: {str(exc)[:180]}", gr.update(interactive=True, value="加载模型")


def run(label: str, protocol: str, model_label: str, lora_label: str):
    item = get_sample(label)
    sid = sample_id(item, protocol)
    started = time.monotonic()
    key = tuple(str(path) for path in selected_assets(model_label, lora_label))
    yield (
        "### 正在推理…",
        gold_table(item, protocol),
        "### 冲突解释\n正在等待模型输出\n\n### 澄清追问\n推理结束后会在这里显示",
        "",
        "⏱ 推理中… 正在计时",
        "**处理中** · 正在加载/调用本地模型",
        gr.update(interactive=False, value="推理中…"),
    )

    live_error = ""
    live_data = None
    try:
        if _MODEL["key"] != key or _MODEL["model"] is None:
            load_local_model(model_label, lora_label)
        live_data = local_infer(sid, protocol)
    except Exception as exc:  # explicit fallback, never pretend live
        live_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    if live_data:
        prediction = live_data["prediction"]
        raw = live_data.get("raw_output") or json.dumps(prediction, ensure_ascii=False)
        status = (
            f"**真实推理** · {live_data.get('engine', 'Qwen2.5-Omni-3B + LoRA')} · "
            f"{live_data.get('device', '')} · {live_data.get('elapsed_s', '—')} 秒"
        )
    else:
        prediction = item[protocol]["prediction"]
        raw = json.dumps(prediction, ensure_ascii=False)
        status = "**真实推理结果回放** · 本地模型不可用，显示已保存的真实模型结果"
        if live_error:
            status += f" · {live_error}"

    if prediction.get("is_conflict") == "yes":
        verdict = "### 判断：存在冲突"
    elif prediction.get("is_conflict") == "no":
        verdict = "### 判断：不存在冲突"
    else:
        verdict = "### 判断：格式异常"

    explanation_text = item[protocol].get("explanation") or (
        "视觉与%s两个模态的情绪%s，%s"
        % (
            "文本" if protocol == "VT" else "音频",
            "相反" if prediction.get("is_conflict") == "yes" else "一致",
            "判定存在冲突" if prediction.get("is_conflict") == "yes" else "不判定为冲突",
        )
    )
    question_text = item[protocol].get("question") or "你在当时更想表达哪一种感受？是画面里的情绪，还是文字/语气里的情绪？"
    explanation = "### 冲突解释\n" + explanation_text + "\n\n### 澄清追问\n" + question_text
    elapsed = (
        live_data.get("elapsed_s", round(time.monotonic() - started, 2))
        if live_data
        else round(time.monotonic() - started, 2)
    )
    yield (
        verdict,
        gold_table(item, protocol, prediction),
        explanation,
        raw,
        f"推理耗时：{elapsed} 秒" + ("（真实推理）" if live_data else "（结果回放）"),
        status,
        gr.update(interactive=True, value="开始推理"),
    )


def build_ui():
    choices = [(choice_label(item), item["label"]) for item in SAMPLES]
    model_names, lora_names, default_model, default_lora = model_labels()
    with gr.Blocks(title="EMO-LLM Demo") as demo:
        gr.Markdown(
            "# EMO-LLM\n"
            "跨模态情感冲突判别 · Qwen2.5-Omni-3B + LoRA（checkpoint-1638）"
        )
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        model_picker = gr.Dropdown(
                            choices=model_names,
                            value=default_model,
                            label="模型",
                        )
                        gr.Markdown("基于 demo/redesign/model_options.json")
                    with gr.Column(scale=1, min_width=220):
                        lora_picker = gr.Dropdown(
                            choices=lora_names,
                            value=default_lora,
                            label="LoRA",
                        )
                        load_button = gr.Button("加载模型", variant="secondary")
                model_status = gr.Markdown("模型未加载")
                sample_picker = gr.Dropdown(
                    choices=choices,
                    value=DEFAULT_LABEL,
                    label="样本选择（支持搜索）",
                    filterable=True,
                )
                protocol_picker = gr.Radio(
                    choices=[("视频 + 文本", "VT"), ("视频 + 音频", "VA")],
                    value="VT",
                    label="输入协议",
                )
                video = gr.Video(
                    value=video_path(SAMPLES[0]),
                    label="输入视频",
                )
                text = gr.Textbox(
                    value=SAMPLES[0]["text"].replace("。", ""),
                    label="待分析文本（VA 模式隐藏）",
                    interactive=False,
                )
            with gr.Column(scale=5):
                verdict = gr.Markdown("### 等待开始")
                gold = gr.Dataframe(
                    value=gold_table(SAMPLES[0], "VT"),
                    headers=["字段", "Gold", "模型", "结果"],
                    datatype=["str", "str", "str", "html"],
                    label="Gold 对照（模型 vs Gold）",
                    interactive=False,
                    wrap=True,
                )
                gr.HTML(
                    '<div style="font-size:14px;line-height:1.9">'
                    '<div><span style="color:#067647;font-weight:700">绿色 ✓</span>：一致</div>'
                    '<div><span style="color:#b54708;font-weight:700">黄色 △</span>：前两个模态判断错误，'
                    '导致“模态冲突/最终情绪”连带偏差</div>'
                    '<div><span style="color:#b42318;font-weight:700">红色 ×</span>：错误，'
                    '前两个模态都正确时“模态冲突/最终情绪”也按红色</div>'
                    '</div>'
                )
                explanation = gr.Markdown(
                    "### 冲突解释\n等待推理后显示\n\n### 澄清追问\n等待推理后，在这里显示当前实现会追问的问题"
                )
                raw = gr.Code(label="raw_output（底部）", language=None, lines=3)
                timing = gr.Markdown("推理耗时：—")
        with gr.Row():
            start = gr.Button("开始推理", variant="primary")
            reset = gr.Button("重置")
        status = gr.Markdown("样本已就绪")
        gr.HTML(value=DEFENSE, elem_id="defense-wrap")

        common_inputs = [sample_picker, protocol_picker]
        common_outputs = [video, text, verdict, gold, explanation, raw, timing, status, start]
        sample_picker.change(initial_state, common_inputs, common_outputs)
        protocol_picker.change(initial_state, common_inputs, common_outputs)
        reset.click(initial_state, common_inputs, common_outputs)

        load_button.click(load_with_progress, [model_picker, lora_picker], [model_status, load_button])
        demo.load(load_with_progress, [model_picker, lora_picker], [model_status, load_button])
        start.click(
            run,
            [sample_picker, protocol_picker, model_picker, lora_picker],
            [verdict, gold, explanation, raw, timing, status, start],
            show_progress="full",
            show_progress_on=[timing],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_api=False)
