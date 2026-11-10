"""Full three-epoch SFT with the same four-field contract as the overfit test."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, TrainingArguments, set_seed

import train_omni_sft as original
import omni_modality_task as task
import overfit_omni_modality as pilot
import score_omni_modality as scoring


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class FullInputs(pilot.TenInputs):
    """Reuse the audited builder, keeping CPU tensor memory bounded for full data."""
    def __init__(self, *args, audit_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_path = audit_path
        self.audit_path.touch(exist_ok=False)

    def build(self, *args, **kwargs):
        batch = super().build(*args, **kwargs)
        self.cached.clear()
        if self.audit:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                for record in self.audit:
                    handle.write(json.dumps({"training_epoch_index": self.epoch, **record}, ensure_ascii=False) + "\n")
            self.audit.clear()
        return batch


def evaluate(model, inputs, rows, out, name, run, step, audio_first=False):
    previous_cache = getattr(model.config, "use_cache", True)
    model.eval()
    model.config.use_cache = True
    records = []
    started = time.monotonic()
    with (out / f"{name}_predictions.jsonl").open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            media_row = {key: row[key] for key in ("sample_id", "protocol", "video", "text_content")}
            batch = original.on_device(inputs.build(media_row, audio_first=audio_first, training=False))
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=192, do_sample=False, use_cache=True,
                    eos_token_id=inputs.processor.tokenizer.eos_token_id,
                    pad_token_id=inputs.processor.tokenizer.pad_token_id)
            tokens = generated[0, batch["input_ids"].shape[-1]:]
            raw = inputs.processor.decode(tokens, skip_special_tokens=True)
            prediction, error = task.parse_answer(raw, row["protocol"])
            record = {"sample_id": row["sample_id"], "protocol": row["protocol"],
                      "source": "generated" if row.get("source_is_generated") else "original",
                      "prompt_order": "VA_first" if audio_first else "VT_first", "gold": task.target(row),
                      "prediction": prediction, "parse_error": error, "raw_output": raw,
                      "output_tokens": len(tokens),
                      "token_limit_hit": len(tokens) >= 192 and tokens[-1].item() != inputs.processor.tokenizer.eos_token_id}
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 1 or index % 25 == 0 or index == len(rows):
                status = {"stage": "evaluating", "evaluation": name, "completed": index, "n": len(rows),
                          "global_step": step, "updated_utc": timestamp()}
                original.write_json(out / "status.json", status)
                print(json.dumps(status), flush=True)
    metrics = scoring.score(records)
    metrics["seconds"] = time.monotonic() - started
    original.write_json(out / f"{name}_metrics.json", metrics)
    if run:
        logs = {"evaluation/global_step": step}
        for group, stats in metrics.items():
            if not isinstance(stats, dict) or not stats.get("n"):
                continue
            for key in ("emotion_accuracy", "emotion_macro_f1_present", "conflict_recall", "aligned_specificity", "format_valid_rate", "local_score_present"):
                if stats[key] is not None:
                    logs[f"{name}/{group}/{key}"] = stats[key]
            for field, values in stats["fields"].items():
                logs[f"{name}/{group}/{field}_accuracy"] = values["accuracy"]
                logs[f"{name}/{group}/{field}_macro_f1"] = values["macro_f1_present"]
        run.log(logs)
    print("EVALUATION", name, json.dumps(metrics["all"]), flush=True)
    model.config.use_cache = previous_cache
    return metrics, records


class FullProgress(original.Evidence):
    def __init__(self, out, inputs, val, run):
        super().__init__(out)
        self.inputs, self.val, self.run = inputs, val, run
        self.trainer = None
        self.checkpoints = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.inputs.epoch = int(round(state.epoch or 0))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 3:
            delta = (self.tracked_parameter.detach().float().cpu() - self.before_update).abs().sum().item()
            if delta <= 0:
                raise RuntimeError("Full training started but the checked LoRA parameter did not update")
            original.write_json(self.out / "startup_check.json", {
                "global_step": state.global_step, "epoch": state.epoch,
                "checked_adapter_update_l1": delta, "gradient_checks": "finite and nonzero before every optimizer step",
                "updated_utc": timestamp()})

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = dict(logs or {})
        for field, losses in self.trainer.field_losses.items():
            if losses:
                values[f"label_loss/{field}"] = sum(losses) / len(losses)
        self.trainer.field_losses.clear()
        super().on_log(args, state, control, values, **kwargs)
        if self.run:
            self.run.log({"training/global_step": state.global_step, "epoch": state.epoch, **values})

    def on_save(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"):
            if not (checkpoint / name).is_file():
                raise RuntimeError(f"Incomplete epoch checkpoint: {checkpoint / name}")
        self.inputs.processor.save_pretrained(checkpoint)
        entry = {"epoch": epoch, "global_step": state.global_step, "path": str(checkpoint)}
        self.checkpoints.append(entry)
        original.write_json(self.out / "checkpoints.json", self.checkpoints)
        metrics, _ = evaluate(model, self.inputs, self.val, self.out, f"epoch_{epoch:02d}_val", self.run, state.global_step)
        entry["validation_metrics"] = metrics
        original.write_json(self.out / "checkpoints.json", self.checkpoints)
        model.train()
        model.config.use_cache = False
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    pilot_summary = json.loads((Path(args.pilot_output) / "summary.json").read_text())
    if not pilot_summary["passed"] or pilot_summary["reloaded"]["all"]["all_fields_correct_n"] != 20:
        raise RuntimeError("The ten-example overfit test must pass including adapter reload")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    original.write_json(out / "status.json", {"stage": "preparing", "updated_utc": timestamp()})
    set_seed(42)
    torch.set_num_threads(4)
    root = Path(args.dataset)
    train, val = original.clean_split(root, out)
    assert len(train) == 3352 and len(val) == 666
    for row in train + val:
        task.target(row)
    data_summary = json.loads((out / "data_summary.json").read_text())
    data_summary["supervision"] = ["views.M1.label", "views.M2.label", "sample_type", "views.M12.label"]
    data_summary["conflict_equals_modality_label_inequality"] = True
    original.write_json(out / "data_summary.json", data_summary)
    for source in (Path(__file__), Path(original.__file__), Path(task.__file__), Path(pilot.__file__),
                   Path(scoring.__file__), Path(original.__file__).with_name("baseline.py")):
        shutil.copyfile(source, out / source.name)
    shutil.copyfile(Path(args.pilot_output) / "summary.json", out / "pilot_summary.json")
    config = {**vars(args), "model_id": "Qwen/Qwen2.5-Omni-3B", "initial_adapter": None,
              "train_records": len(train), "val_records": len(val), "seed": 42,
              "lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05,
              "learning_rate": 1e-4, "scheduler": "cosine", "warmup_ratio": .03,
              "micro_batch": 1, "gradient_accumulation_steps": 16,
              "frames": 8, "max_side": 448, "max_length": 4096, "audio_rate": 16000,
              "max_new_tokens": 192, "dtype": "bfloat16", "attention": "sdpa",
              "prompt_orders": "Alternated by training epoch and row index, independent of protocol",
              "prompts": {"VT_first": task.system_prompt(False), "VA_first": task.system_prompt(True)},
              "checkpoint_selection": "Highest full-validation local score; exact ties choose earliest epoch",
              "baseline_evaluation": "After training, disable LoRA and evaluate original base with identical inputs and decoding",
              "gpu": torch.cuda.get_device_name(0), "started_utc": timestamp(),
              "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate", "wandb")}}
    original.write_json(out / "config.json", config)
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = FullInputs(processor, root, out.parent / "media_cache", train, audit_path=out / "input_checks.jsonl")
    model = original.load_base(args.model, out)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"))
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and ".model.layers." in n for n, _ in trainable)
    original.write_json(out / "trainable_parameters.json", {"count": sum(v for _, v in trainable), "modules": [n for n, _ in trainable]})
    tracked = next(p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    before = tracked.detach().float().cpu().clone()
    import wandb
    run = wandb.init(project="MM-LLMs-Emotion", name=out.name, config=config, dir=str(out),
                     settings=wandb.Settings(save_code=False, disable_git=True, console="off"))
    run.define_metric("training/global_step")
    run.define_metric("evaluation/global_step")
    run.define_metric("label_loss/*", step_metric="training/global_step")
    original.write_json(out / "wandb_run.json", {"id": run.id, "url": run.url, "project": run.project, "entity": run.entity})
    print("WANDB_RUN_URL=" + str(run.url), flush=True)
    callback = FullProgress(out, inputs, val, run)
    callback.tracked_parameter, callback.before_update = tracked, before
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=args.epochs, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=.03,
        weight_decay=.01, bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch", max_grad_norm=1.,
        logging_steps=1, save_strategy="epoch", save_total_limit=None, eval_strategy="no",
        dataloader_num_workers=0, remove_unused_columns=False, label_names=["labels"], report_to=[],
        seed=42, data_seed=42, disable_tqdm=True)
    trainer = pilot.FieldTrainer(model=model, args=training_args, train_dataset=original.Rows(train),
                                 data_collator=inputs.collate, callbacks=[callback])
    callback.trainer = trainer
    trainer.model_accepts_loss_kwargs = False
    trained = trainer.train()
    update_l1 = (tracked.detach().float().cpu() - before).abs().sum().item()
    assert update_l1 > 0
    original.write_json(out / "training_summary.json", {**trained.metrics, "global_step": trainer.state.global_step,
                        "checked_adapter_update_l1": update_l1, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30})
    trainer.save_model(str(out / "final_adapter"))
    processor.save_pretrained(out / "final_adapter")
    trainer.save_state()
    assert len(callback.checkpoints) == args.epochs
    best = max(callback.checkpoints, key=lambda c: (c["validation_metrics"]["all"]["local_score_present"], -c["epoch"]))
    original.write_json(out / "checkpoint_selection.json", best)
    model.gradient_checkpointing_disable()
    best_adapter = out / "best_adapter"
    best_adapter.mkdir(exist_ok=False)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(Path(best["path"]) / name, best_adapter / name)
    processor.save_pretrained(best_adapter)
    model.load_adapter(str(best_adapter), adapter_name="best_epoch", is_trainable=False)
    model.set_adapter("best_epoch")
    # A deterministic protocol/source/type subset checks order sensitivity and saved weights.
    buckets = {}
    for row in val:
        buckets.setdefault((row["protocol"], bool(row.get("source_is_generated")), row["sample_type"]), []).append(row)
    diagnostic = [row for group in buckets.values() for row in group[:3]]
    original.write_json(out / "routing_diagnostic_samples.json", [r["sample_id"] for r in diagnostic])
    evaluate(model, inputs, diagnostic, out, "best_reload_VT_first", run, trainer.state.global_step)
    evaluate(model, inputs, diagnostic, out, "best_reload_VA_first", run, trainer.state.global_step, audio_first=True)
    with model.disable_adapter():
        evaluate(model, inputs, val, out, "base_matched_val", run, trainer.state.global_step)
    original.write_json(out / "status.json", {"stage": "complete", "global_step": trainer.state.global_step,
                        "best_epoch": best["epoch"], "updated_utc": timestamp()})
    run.summary["best_epoch"] = best["epoch"]
    run.summary["best_validation_local_score"] = best["validation_metrics"]["all"]["local_score_present"]
    run.finish()


if __name__ == "__main__":
    main()
