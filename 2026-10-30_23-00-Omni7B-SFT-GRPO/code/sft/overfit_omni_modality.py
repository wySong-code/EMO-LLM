"""Memorization diagnostic on ten TRAIN rows; no validation or test training."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import time

import av
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, TrainerCallback, TrainingArguments, set_seed

import train_omni_sft as original
from baseline import prepare_media, ensure_audio
import omni_modality_task as task


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class TenInputs:
    def __init__(self, processor, root, cache, rows):
        self.processor, self.root, self.cache = processor, root, cache
        self.rows = rows
        self.epoch = 0
        self.index = {r["sample_id"]: i for i, r in enumerate(rows)}
        self.cached = {}
        self.audit = []

    def build(self, row, audio_first=False, training=False):
        key = (row["sample_id"], audio_first, training)
        if key in self.cached:
            return {k: v.clone() if torch.is_tensor(v) else v for k, v in self.cached[key].items()}
        source = self.root / row["video"]
        video, audio, meta = prepare_media(source, self.cache, 8, 448)
        duration = json.loads(meta.read_text())["duration"]
        with av.open(str(video)) as decoded:
            frames = np.stack([f.to_ndarray(format="rgb24") for f in decoded.decode(video=0)])
        content = [{"type": "video"}, {"type": "text", "text": "请判断所提供内容的情感。"}]
        wave = None
        if row["protocol"] == "VT":
            content.append({"type": "text", "text": "待分析文字：\n" + row["text_content"]})
        else:
            ensure_audio(source, audio)
            samples, rate = sf.read(audio, dtype="float32")
            assert rate == 16000 and samples.ndim == 1 and np.isfinite(samples).all()
            assert len(samples) and np.any(samples != 0)
            wave = [samples]
            content.append({"type": "audio"})
        messages = [{"role": "system", "content": [{"type": "text", "text": task.system_prompt(audio_first)}]},
                    {"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert row["sample_id"] not in prompt and row["video"] not in prompt
        batch = dict(self.processor(text=[prompt], videos=[frames], audio=wave, return_tensors="pt",
                     videos_kwargs={"do_sample_frames": False, "fps": len(frames) / duration,
                                    "use_audio_in_video": False}, audio_kwargs={"sampling_rate": 16000}))
        assert "pixel_values_videos" in batch
        assert ("input_features" in batch) == (row["protocol"] == "VA")
        prompt_length = batch["input_ids"].shape[-1]
        if training:
            answer = json.dumps(task.target(row), separators=(",", ":"))
            encoded = self.processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
            tokens = encoded["input_ids"] + [self.processor.tokenizer.eos_token_id]
            target_ids = torch.tensor([tokens], dtype=torch.long)
            batch["input_ids"] = torch.cat([batch["input_ids"], target_ids], dim=1)
            batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(target_ids)], dim=1)
            batch["labels"] = torch.cat([torch.full((1, prompt_length), -100, dtype=torch.long), target_ids], dim=1)
            positions = {}
            for field in task.target(row):
                match = re.search('"' + field + '":"([^"]+)"', answer)
                start, end = match.span(1)
                selected = [i + prompt_length for i, (a, b) in enumerate(encoded["offset_mapping"])
                            if a < end and b > start]
                assert selected and all(batch["labels"][0, p] != -100 for p in selected)
                positions[field] = selected
            assert (batch["labels"] != -100).sum().item() == len(tokens)
            assert torch.all(batch["labels"][0, :prompt_length] == -100)
            batch["_field_positions"] = positions
            self.audit.append({"sample_id": row["sample_id"], "protocol": row["protocol"],
                               "prompt_order": "VA_first" if audio_first else "VT_first",
                               "prompt_tokens": prompt_length, "supervised_tokens": len(tokens),
                               "answer": answer, "field_supervised_tokens": {k: len(v) for k, v in positions.items()},
                               "has_audio_features": "input_features" in batch,
                               "input_keys": sorted(k for k in batch if not k.startswith("_"))})
        if batch["input_ids"].shape[-1] > 4096:
            raise ValueError(f"Sequence exceeds 4096 tokens: {row['sample_id']}")
        self.cached[key] = batch
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    def collate(self, rows):
        assert len(rows) == 1
        row = rows[0]
        # Each sample sees both orders on alternating passes; no order/protocol coupling.
        audio_first = bool((self.index[row["sample_id"]] + self.epoch) % 2)
        return self.build(row, audio_first=audio_first, training=True)


class FieldTrainer(original.CheckedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_losses = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        positions = inputs.pop("_field_positions")
        loss, result = super().compute_loss(model, inputs, return_outputs=True)
        # Report actual next-token losses on label values, separately from JSON syntax.
        with torch.no_grad():
            for field, indexes in positions.items():
                index = torch.tensor(indexes, device=result.logits.device)
                logits = result.logits[0, index - 1].float()
                gold = inputs["labels"][0, index]
                field_loss = F.cross_entropy(logits, gold).item()
                if not math.isfinite(field_loss):
                    raise FloatingPointError(f"Non-finite {field} loss")
                self.field_losses[field].append(field_loss)
        return (loss, result) if return_outputs else loss


def evaluate(model, inputs, rows, out, name, run=None, step=0):
    previous_cache = getattr(model.config, "use_cache", True)
    model.eval()
    model.config.use_cache = True
    records = []
    start = time.monotonic()
    with (out / f"{name}_predictions.jsonl").open("x", encoding="utf-8") as handle:
        for audio_first in (False, True):
            for row in rows:
                # Gold labels are not used to construct the generation prompt.
                media_row = {k: row[k] for k in ("sample_id", "protocol", "video", "text_content")}
                batch = original.on_device(inputs.build(media_row, audio_first, training=False))
                with torch.inference_mode():
                    generated = model.generate(**batch, max_new_tokens=192, do_sample=False, use_cache=True,
                                               eos_token_id=inputs.processor.tokenizer.eos_token_id,
                                               pad_token_id=inputs.processor.tokenizer.pad_token_id)
                tokens = generated[0, batch["input_ids"].shape[-1]:]
                raw = inputs.processor.decode(tokens, skip_special_tokens=True)
                prediction, error = task.parse_answer(raw, row["protocol"])
                record = {"sample_id": row["sample_id"], "protocol": row["protocol"],
                          "prompt_order": "VA_first" if audio_first else "VT_first",
                          "gold": task.target(row), "prediction": prediction, "parse_error": error,
                          "raw_output": raw, "output_tokens": len(tokens),
                          "token_limit_hit": len(tokens) >= 192 and tokens[-1].item() != inputs.processor.tokenizer.eos_token_id}
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    metrics = task.score(records)
    metrics["seconds"] = time.monotonic() - start
    original.write_json(out / f"{name}_metrics.json", metrics)
    print("EVALUATION", name, json.dumps(metrics["all"]), flush=True)
    if run:
        logs = {"diagnostic/global_step": step, "memorization/all_fields_accuracy": metrics["all"]["all_fields_accuracy"],
                "memorization/format_valid_rate": metrics["all"]["format_valid_n"] / len(records)}
        for field, stat in metrics["all"]["fields"].items():
            logs[f"memorization/{field}_accuracy"] = stat["accuracy"]
        run.log(logs)
    model.config.use_cache = previous_cache
    return metrics, records


class Progress(original.Evidence):
    def __init__(self, out, inputs, rows, run):
        super().__init__(out)
        self.inputs, self.rows, self.run = inputs, rows, run
        self.trainer = None
        self.checkpoints, self.evaluations = [], []
        self.passed = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.inputs.epoch = int(round(state.epoch or 0))

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = dict(logs or {})
        for field, losses in self.trainer.field_losses.items():
            if losses:
                values[f"label_loss/{field}"] = sum(losses) / len(losses)
        self.trainer.field_losses.clear()
        super().on_log(args, state, control, values, **kwargs)
        if self.run:
            self.run.log({"diagnostic/global_step": state.global_step, "epoch": state.epoch, **values})

    def on_save(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        assert (checkpoint / "adapter_model.safetensors").exists()
        assert (checkpoint / "optimizer.pt").exists()
        entry = {"epoch": epoch, "step": state.global_step, "path": str(checkpoint)}
        self.checkpoints.append(entry)
        original.write_json(self.out / "checkpoints.json", self.checkpoints)
        if epoch <= 3 or epoch % 5 == 0 or epoch >= args.num_train_epochs:
            original.write_json(self.out / "status.json", {"stage": "evaluating", **entry, "updated_utc": timestamp()})
            metrics, _ = evaluate(model, self.inputs, self.rows, self.out, f"epoch_{epoch:03d}", self.run, state.global_step)
            self.evaluations.append({**entry, "metrics": metrics})
            original.write_json(self.out / "evaluation_history.json", self.evaluations)
            if metrics["all"]["all_fields_correct_n"] == 20:
                self.passed = True
                control.should_training_stop = True
            model.train()
            model.config.use_cache = False
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    torch.set_num_threads(4)
    root = Path(args.dataset)
    selection = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    chosen = selection["sample_ids"]
    assert len(chosen) == len(set(chosen)) == 10
    train = {r["sample_id"]: r for r in original.read_rows(root / "train.jsonl")}
    val_ids = {r["sample_id"] for r in original.read_rows(root / "val.jsonl")}
    assert not val_ids.intersection(chosen) and set(chosen).issubset(train)
    rows = [train[k] for k in chosen]
    assert sum(r["protocol"] == "VT" for r in rows) == 5
    for row in rows:
        assert task.target(row) == selection["targets"][row["sample_id"]]
    shutil.copyfile(args.samples, out / "samples.json")
    for source in (Path(__file__), Path(original.__file__), Path(task.__file__), Path(original.__file__).with_name("baseline.py")):
        shutil.copyfile(source, out / source.name)
    config = {**vars(args), "model_id": "Qwen/Qwen2.5-Omni-3B", "seed": 42,
              "train_unique_samples": 10, "protocols": {"VT": 5, "VA": 5},
              "lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05,
              "learning_rate": 1e-4, "scheduler": "cosine", "warmup_ratio": .03,
              "micro_batch": 1, "gradient_accumulation_steps": 16,
              "effective_samples_per_update": 10,
              "note": "Ten rows per epoch: Trainer flushes the partial accumulation at each epoch. This diagnostic repeats training rows until both prompt orders are 10/10 or reaches the epoch budget. Not validation performance.",
              "frames": 8, "max_side": 448, "max_length": 4096, "audio_rate": 16000,
              "max_new_tokens": 192, "dtype": "bfloat16", "attention": "sdpa",
              "prompts": {"VT_first": task.system_prompt(False), "VA_first": task.system_prompt(True)},
              "gpu": torch.cuda.get_device_name(0), "started_utc": timestamp(),
              "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate", "wandb")}}
    original.write_json(out / "config.json", config)
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = TenInputs(processor, root, out.parent / "media_cache", rows)
    for row in rows:
        for order in (False, True):
            inputs.build(row, order, training=True)
            inputs.build(row, order, training=False)
    original.write_json(out / "input_checks.json", inputs.audit)
    model = original.load_base(args.model, out)
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project="MM-LLMs-Emotion", name=out.name, config=config, dir=str(out), mode=args.wandb_mode,
                         settings=wandb.Settings(save_code=False, disable_git=True, console="off"))
        run.define_metric("diagnostic/global_step")
        run.define_metric("memorization/*", step_metric="diagnostic/global_step")
        run.define_metric("label_loss/*", step_metric="diagnostic/global_step")
        original.write_json(out / "wandb_run.json", {"id": run.id, "url": run.url, "project": run.project, "entity": run.entity})
        print("WANDB_RUN_URL=" + str(run.url), flush=True)
    original.write_json(out / "status.json", {"stage": "baseline", "updated_utc": timestamp()})
    baseline, _ = evaluate(model, inputs, rows, out, "base", run)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"))
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and ".model.layers." in n for n, _ in trainable)
    original.write_json(out / "trainable_parameters.json", {"count": sum(v for _, v in trainable), "modules": [n for n, _ in trainable]})
    tracked = next(p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    before = tracked.detach().float().cpu().clone()
    callback = Progress(out, inputs, rows, run)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=args.max_epochs, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=.03,
        weight_decay=.01, bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch", max_grad_norm=1,
        logging_steps=1, save_strategy="epoch", save_total_limit=None, eval_strategy="no",
        dataloader_num_workers=0, remove_unused_columns=False, label_names=["labels"], report_to=[],
        seed=42, data_seed=42, disable_tqdm=True)
    trainer = FieldTrainer(model=model, args=training_args, train_dataset=original.Rows(rows),
                           data_collator=inputs.collate, callbacks=[callback])
    callback.trainer = trainer
    trainer.model_accepts_loss_kwargs = False
    trained = trainer.train()
    update_l1 = (tracked.detach().float().cpu() - before).abs().sum().item()
    assert update_l1 > 0
    trainer.save_model(str(out / "adapter"))
    processor.save_pretrained(out / "adapter")
    trainer.save_state()
    model.gradient_checkpointing_disable()
    model.load_adapter(str(out / "adapter"), adapter_name="reload_check", is_trainable=False)
    model.set_adapter("reload_check")
    reloaded, _ = evaluate(model, inputs, rows, out, "reloaded", run, trainer.state.global_step)
    passed = callback.passed and reloaded["all"]["all_fields_correct_n"] == 20
    summary = {"passed": passed, "criterion": "10/10 exact four-field outputs under BOTH shared-prompt orders, reproduced after adapter reload",
               "unique_training_samples": 10, "evaluated_generations_per_check": 20,
               "baseline": baseline, "reloaded": reloaded, "completed_epochs": trainer.state.epoch,
               "global_step": trainer.state.global_step, "training": trained.metrics,
               "checked_adapter_update_l1": update_l1, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
               "finished_utc": timestamp(), "scope": "Training-set memorization only; no generalization claim"}
    original.write_json(out / "summary.json", summary)
    original.write_json(out / "status.json", {"stage": "complete", "passed": passed,
                        "epoch": trainer.state.epoch, "global_step": trainer.state.global_step, "updated_utc": timestamp()})
    if run:
        run.summary["memorization_passed"] = passed
        run.finish()
    print("COMPLETE", json.dumps({"passed": passed, "epoch": trainer.state.epoch,
                                  "step": trainer.state.global_step, "exact": reloaded["all"]["all_fields_correct_n"]}), flush=True)


if __name__ == "__main__":
    main()
