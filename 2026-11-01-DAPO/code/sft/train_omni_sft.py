"""First SFT experiment: supervised emotion/conflict, without invented QA labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil

import av
import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from baseline import prepare_media, ensure_audio, extract_json

SYSTEM = """你是一个情感识别助手。根据提供的画面和文字或音频，判断说话者的综合情感，以及两种模态的情感是否冲突。
情感类别只能是 positive（正向）、negative（负向）、neutral（中性）。
只输出一个 JSON 对象，不要解释或 Markdown。字段必须为：
pred_emotion: positive/negative/neutral；is_conflict: yes/no。"""
CLASSES = ["negative", "positive", "neutral"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_rows(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len({r["sample_id"] for r in rows}) == len(rows), "Duplicate sample IDs"
    for r in rows:
        assert r["protocol"] in ("VT", "VA")
        assert r["views"]["M12"]["label"] in CLASSES
        assert r["sample_type"] in ("Aligned", "Conflict")
    return rows


def clean_split(root, out):
    train, val = read_rows(root / "train.jsonl"), read_rows(root / "val.jsonl")
    fingerprints = {}
    for video in sorted({r["video"] for r in train + val}):
        fingerprints[video] = hashlib.sha256((root / video).read_bytes()).digest()
    val_media = {fingerprints[r["video"]] for r in val}
    excluded = [r for r in train if fingerprints[r["video"]] in val_media]
    train = [r for r in train if fingerprints[r["video"]] not in val_media]
    report = {
        "train_count": len(train), "val_count": len(val),
        "train_protocols": dict(Counter(r["protocol"] for r in train)),
        "train_labels": dict(Counter(r["views"]["M12"]["label"] for r in train)),
        "train_sample_types": dict(Counter(r["sample_type"] for r in train)),
        "excluded_validation_duplicates": [{"sample_id": r["sample_id"], "video": r["video"]} for r in excluded],
        "use_in_main": "All records retained; the course has not specified this flag's filtering semantics.",
        "supervision": ["views.M12.label", "sample_type"],
        "uncertainty_supervision": "Not available; this run does not train uncertainty or clarifying questions.",
    }
    write_json(out / "data_summary.json", report)
    for name, rows in (("train_manifest", train), ("val_manifest", val)):
        (out / (name + ".jsonl")).write_text("".join(json.dumps({"sample_id": r["sample_id"], "video": r["video"]}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return train, val


class Inputs:
    def __init__(self, processor, root, cache, frames, max_side, max_length):
        self.processor, self.root, self.cache = processor, root, cache
        self.frames, self.max_side, self.max_length = frames, max_side, max_length

    def build(self, row, training=False, *, system_prompt=None, followup=None):
        original = self.root / row["video"]
        video, audio, meta = prepare_media(original, self.cache, self.frames, self.max_side)
        duration = json.loads(meta.read_text())["duration"]
        with av.open(str(video)) as decoded:
            frames = np.stack([f.to_ndarray(format="rgb24") for f in decoded.decode(video=0)])
        content = [{"type": "video"}]
        wave = None
        if row["protocol"] == "VT":
            content.append({"type": "text", "text": "任务：结合视频画面和以下文字判断情感。\n文字：" + row["text_content"]})
        else:
            ensure_audio(original, audio)
            samples, rate = sf.read(audio, dtype="float32")
            assert rate == 16000 and samples.ndim == 1
            wave = [samples]
            content.extend([{"type": "text", "text": "任务：结合视频画面和音频判断情感。"}, {"type": "audio"}])
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt or SYSTEM}]},
                    {"role": "user", "content": content}]
        if followup:
            assert not training, "Follow-up turns are inference only"
            messages.extend(followup)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], videos=[frames], audio=wave, return_tensors="pt",
            videos_kwargs={"do_sample_frames": False, "fps": len(frames) / duration, "use_audio_in_video": False},
            audio_kwargs={"sampling_rate": 16000},
        )
        assert "pixel_values_videos" in inputs
        assert ("input_features" in inputs) == (row["protocol"] == "VA"), "Wrong modality wiring"
        if training:
            answer = json.dumps({"pred_emotion": row["views"]["M12"]["label"],
                                 "is_conflict": "yes" if row["sample_type"] == "Conflict" else "no"}, separators=(",", ":"))
            tokens = self.processor.tokenizer.encode(answer + self.processor.tokenizer.eos_token, add_special_tokens=False)
            target = torch.tensor([tokens], dtype=torch.long)
            prompt_ids = inputs["input_ids"]
            inputs["input_ids"] = torch.cat([prompt_ids, target], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(target)], dim=1)
            inputs["labels"] = torch.cat([torch.full_like(prompt_ids, -100), target], dim=1)
            assert (inputs["labels"] != -100).sum().item() == len(tokens)
        if inputs["input_ids"].shape[-1] > self.max_length:
            raise ValueError(f"Sequence too long, no silent truncation: {row['sample_id']}")
        return dict(inputs)

    def collate(self, rows):
        assert len(rows) == 1, "Use gradient accumulation; mixed media micro-batch is one."
        return self.build(rows[0], training=True)


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def on_device(inputs):
    return {k: v.to("cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            if torch.is_tensor(v) else v for k, v in inputs.items()}


class CheckedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Frozen encoders stay deterministic when Trainer sets model.train().
        base = model.get_base_model()
        base.visual.eval()
        base.audio_tower.eval()
        result = model(**inputs)
        if not torch.isfinite(result.loss):
            raise FloatingPointError("Non-finite SFT loss")
        return (result.loss, result) if return_outputs else result.loss


class Evidence(TrainerCallback):
    def __init__(self, out):
        self.out = out

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError("Missing or non-finite LoRA gradients")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise RuntimeError("All LoRA gradients are zero")

    def on_log(self, args, state, control, logs=None, **kwargs):
        value = {"step": state.global_step, "epoch": state.epoch, **(logs or {}),
                 "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "updated_utc": datetime.now(timezone.utc).isoformat()}
        with (self.out / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(value) + "\n")
        write_json(self.out / "status.json", {"stage": "training", **value})
        print(json.dumps(value), flush=True)


def load_base(path, out):
    model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa", output_loading_info=True,
    )
    write_json(out / "model_loading.json", info)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError("Incomplete base checkpoint load; inspect model_loading.json")
    print("Loaded base parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    return model


def score(rows):
    from sklearn.metrics import accuracy_score, f1_score
    def group(data):
        if not data:
            return {"n": 0}
        y = [r["gold_emotion"] for r in data]
        p = [r["prediction"].get("pred_emotion") if r["prediction"].get("pred_emotion") in CLASSES else "invalid" for r in data]
        cy = [r["gold_conflict"] for r in data]
        cp = [r["prediction"].get("is_conflict") if r["prediction"].get("is_conflict") in ("yes", "no") else "invalid" for r in data]
        return {"n": len(data), "emotion_accuracy": accuracy_score(y, p),
                "emotion_macro_f1_present": f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0),
                "emotion_macro_f1_fixed3": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0),
                "conflict_accuracy": accuracy_score(cy, cp),
                "conflict_f1": f1_score(cy, cp, labels=["yes"], average="macro", zero_division=0),
                "json_valid_rate": sum(r["format_valid"] for r in data) / len(data)}
    return {"all": group(rows), **{p: group([r for r in rows if r["protocol"] == p]) for p in ("VT", "VA")}}


def evaluate(model, inputs, rows, out, name, run=None):
    model.eval()
    model.config.use_cache = True
    records = []
    with (out / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            batch = on_device(inputs.build(row))
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=96, do_sample=False, use_cache=True,
                                           eos_token_id=inputs.processor.tokenizer.eos_token_id,
                                           pad_token_id=inputs.processor.tokenizer.pad_token_id)
            text = inputs.processor.decode(generated[0, batch["input_ids"].shape[-1]:], skip_special_tokens=True)
            prediction = extract_json(text)
            try:
                valid = json.loads(text) == prediction and prediction.get("pred_emotion") in CLASSES and prediction.get("is_conflict") in ("yes", "no")
            except json.JSONDecodeError:
                valid = False
            record = {"sample_id": row["sample_id"], "protocol": row["protocol"],
                      "gold_emotion": row["views"]["M12"]["label"],
                      "gold_conflict": "yes" if row["sample_type"] == "Conflict" else "no",
                      "prediction": prediction, "raw_output": text, "format_valid": valid}
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                write_json(out / "status.json", {"stage": name, "completed": i + 1, "total": len(rows)})
                print(f"{name} {i + 1}/{len(rows)}", flush=True)
            del generated, batch
    metrics = score(records)
    write_json(out / f"{name}_metrics.json", metrics)
    if run:
        run.log({f"{name}/{group}/{key}": value for group, values in metrics.items() for key, value in values.items()})
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--resume")
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--wandb-project", default="MM-LLMs-Emotion")
    args = p.parse_args()
    out, root = Path(args.output).resolve(), Path(args.dataset).resolve()
    if out.exists() and (out / "adapter").exists() and not args.resume:
        raise FileExistsError("Use a new experiment directory; an adapter already exists.")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    torch.set_num_threads(4)
    # This run uses Thinker text output; the Talker-only prompt warning is irrelevant.
    logging.getLogger().addFilter(
        lambda record: not record.getMessage().startswith("System prompt modified, audio output"))
    train, val = clean_split(root, out)
    if args.smoke:
        groups = {}
        for row in train:
            groups.setdefault((row["protocol"], row["sample_type"], row.get("source_is_generated", False)), row)
        train = list(groups.values())
    config = {**vars(args), "model_id": "Qwen/Qwen2.5-Omni-3B", "seed": 42,
              "frames": 8, "max_side": 448, "lora_rank": 16, "lora_alpha": 32,
              "learning_rate": 1e-4, "gradient_accumulation": 1 if args.smoke else 16,
              "system_prompt": SYSTEM, "train_records": len(train), "val_records": len(val),
              "gpu": torch.cuda.get_device_name(0), "versions": {name: importlib.metadata.version(name)
              for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "wandb", "av", "numpy")}}
    write_json(out / "config.json", config)
    shutil.copyfile(__file__, out / "train_source.py")
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = Inputs(processor, root, out.parent / "media_cache", 8, 448, 4096)
    if args.smoke:
        checks = []
        for row in train:
            batch = inputs.build(row, training=True)
            checks.append({"sample_id": row["sample_id"], "protocol": row["protocol"],
                           "input_keys": sorted(batch), "tokens": batch["input_ids"].shape[-1],
                           "supervised_tokens": int((batch["labels"] != -100).sum())})
        write_json(out / "input_checks.json", checks)
        del batch
    model = load_base(args.model, out)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    ))
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and ".model.layers." in n for n, _ in trainable)
    write_json(out / "trainable_parameters.json", {"count": sum(v for _, v in trainable), "modules": [n for n, _ in trainable]})
    tracked_parameter = next(p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    before_update = tracked_parameter.detach().float().cpu().clone()
    run = None
    if not args.smoke:
        import wandb
        run = wandb.init(project=args.wandb_project, name=out.name, config=config, dir=str(out),
                         settings=wandb.Settings(save_code=False, disable_git=True, console="off"))
        write_json(out / "wandb_run.json", {"id": run.id, "url": run.url, "project": run.project, "entity": run.entity})
        print("WANDB_RUN_URL=" + run.url, flush=True)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"], num_train_epochs=args.epochs,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch", max_grad_norm=1.0,
        logging_steps=1, save_steps=25, save_total_limit=3, eval_strategy="no",
        dataloader_num_workers=0, remove_unused_columns=False, label_names=["labels"],
        report_to=[] if args.smoke else ["wandb"], seed=42, data_seed=42, disable_tqdm=True,
    )
    trainer = CheckedTrainer(model=model, args=training_args, train_dataset=Rows(train),
                             data_collator=inputs.collate, callbacks=[Evidence(out)])
    trainer.model_accepts_loss_kwargs = False
    trained = trainer.train(resume_from_checkpoint=args.resume)
    update_l1 = (tracked_parameter.detach().float().cpu() - before_update).abs().sum().item()
    if update_l1 <= 0:
        raise RuntimeError("The checked LoRA parameter did not change")
    write_json(out / "training_summary.json", {**trained.metrics, "global_step": trainer.state.global_step,
                                              "checked_adapter_update_l1": update_l1,
                                              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30})
    trainer.save_model(str(out / "adapter"))
    processor.save_pretrained(out / "adapter")
    trainer.save_state()
    if args.smoke:
        # Reload actual saved weights into this model and exercise both input protocols.
        model.load_adapter(str(out / "adapter"), adapter_name="reload_check", is_trainable=False)
        model.set_adapter("reload_check")
        samples = [next(r for r in val if r["protocol"] == protocol) for protocol in ("VT", "VA")]
        evaluate(model, inputs, samples, out, "reload_smoke")
    elif not args.skip_evaluation:
        model.gradient_checkpointing_disable()
        evaluate(model, inputs, val, out, "sft_val", run)
        # Matched prompt/media/settings, so the first SFT comparison is reproducible.
        with model.disable_adapter():
            evaluate(model, inputs, val, out, "base_matched_val", run)
    write_json(out / "status.json", {"stage": "complete", "global_step": trainer.state.global_step})
    if run:
        run.finish()


if __name__ == "__main__":
    main()
