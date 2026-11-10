"""Measure the real memory ceiling of Qwen2.5-Omni-7B on one device.

The 3B recipe used microbatch 4 and eval batch 16, so neither number
can be copied blindly. Two questions have to be answered before a multi-hour run:

  1. does one training micro-batch fit next to the ~16.6 GiB of bf16 weights,
  2. which greedy generation batch size do the grader and the submission afford.

The two stages run in separate processes on purpose: a failed backward leaves partly
allocated gradient buffers behind, and every later measurement would be poisoned.

Only ~2% of the parameters are trainable in the real run (LoRA), so the probe attaches
the same LoRA configuration instead of training the frozen base — otherwise backward
allocates 16.6 GiB of gradient buffers that the real run never has.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch


def gib(x):
    return round(x / 2 ** 30, 3)


def build_batch(inputs, processor, rows):
    pad = processor.tokenizer.pad_token_id
    items = [inputs.build(r, training=True) for r in rows]
    width = max(x["input_ids"].shape[1] for x in items)
    batch = {}
    for key in items[0]:
        if key.startswith("_"):
            continue
        values = [x[key] for x in items]
        if key in ("input_ids", "attention_mask", "labels"):
            fill = {"input_ids": pad, "attention_mask": 0, "labels": -100}[key]
            batch[key] = torch.cat([torch.nn.functional.pad(
                v, (0, width - v.shape[1]), value=fill) for v in values])
        elif all(torch.is_tensor(v) for v in values):
            if key in ("input_features", "feature_attention_mask"):
                maximum = max(v.shape[-1] for v in values)
                values = [torch.nn.functional.pad(v, (0, maximum - v.shape[-1])) for v in values]
            batch[key] = torch.cat(values, dim=0)
    return {k: v.to("cuda:0", dtype=torch.bfloat16 if torch.is_floating_point(v) else v.dtype)
            for k, v in batch.items()}, width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--stage", default="train", choices=["train", "generate"])
    ap.add_argument("--rows", default="4,2,1")
    ap.add_argument("--batches", default="16,8,4,2,1")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(args.code))
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from overfit_omni_modality import TenInputs

    torch.backends.cuda.matmul.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    report = {"stage": args.stage, "gpu": torch.cuda.get_device_name(0),
              "total_gib": gib(total), "free_before_gib": gib(free)}

    processor = AutoProcessor.from_pretrained(args.base)
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "cuda:0"}, low_cpu_mem_usage=True)
    report["base_weights_gib"] = gib(torch.cuda.memory_allocated())

    sizes = [int(x) for x in (args.rows if args.stage == "train" else args.batches).split(",") if x.strip()]
    train_rows = [json.loads(line) for line in
                  (args.dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()[:max(sizes)]]
    val_rows = [json.loads(line) for line in
                (args.dataset / "val.jsonl").read_text(encoding="utf-8").splitlines()[:max(sizes)]]

    if args.stage == "train":
        from train_omni_multimodal_sft import attach_lora
        from train_omni_unimodal_sft import answer_ce
        scratch = (Path(args.out).parent / "probe_lora") if args.out else Path("/tmp/probe_lora")
        scratch.mkdir(parents=True, exist_ok=True)
        model = attach_lora(base, scratch)
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["weights_plus_lora_gib"] = gib(torch.cuda.memory_allocated())
        inputs = TenInputs(processor, args.dataset, args.cache, train_rows)
        results = []
        for size in sizes:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            record = {"rows": size}
            started = time.monotonic()
            try:
                batch, width = build_batch(inputs, processor, train_rows[:size])
                record["tokens"] = width
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
                    means, _ = answer_ce(logits, batch["labels"])
                    loss = means.mean()
                record["logits_gib"] = gib(logits.numel() * logits.element_size())
                del logits
                loss.backward()
                torch.cuda.synchronize()
                record.update(ok=True, seconds=round(time.monotonic() - started, 2),
                              peak_gib=gib(torch.cuda.max_memory_allocated()))
                del batch, loss, means
            except torch.OutOfMemoryError as error:
                record.update(ok=False, error=str(error).splitlines()[0][:160],
                              peak_gib=gib(torch.cuda.max_memory_allocated()))
            finally:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            results.append(record)
            print("PROBE_TRAIN", json.dumps(record), flush=True)
        report["train"] = results
    else:
        from sample_omni_dpo_data import build as build_eval, generate
        model = base
        model.eval()
        model.config.use_cache = True
        inputs = TenInputs(processor, args.dataset, args.cache, val_rows)
        results = []
        for size in sizes:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            record = {"batch_size": size}
            started = time.monotonic()
            try:
                items = [build_eval(inputs, r) for r in val_rows[:size]]
                answers = generate(model, processor, items, val_rows[:size], False, 20261026)
                record.update(ok=True, parsed=sum(a["parse_error"] is None for a in answers),
                              tokens=max(len(i["input_ids"][0]) for i in items),
                              seconds=round(time.monotonic() - started, 2),
                              peak_gib=gib(torch.cuda.max_memory_allocated()))
                del items, answers
            except torch.OutOfMemoryError as error:
                record.update(ok=False, error=str(error).splitlines()[0][:160],
                              peak_gib=gib(torch.cuda.max_memory_allocated()))
            torch.cuda.empty_cache()
            results.append(record)
            print("PROBE_GENERATE", json.dumps(record), flush=True)
        report["generation"] = results

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        target = args.out.with_name(args.out.stem + "_" + args.stage + ".json")
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PROBE_DONE", json.dumps({"stage": args.stage}), flush=True)


if __name__ == "__main__":
    main()
