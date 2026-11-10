"""Generate the Kaggle submission CSV from the 7B base + a selected adapter.

Nothing in the project is modified: preprocessing (TenInputs), the system prompt and
the four-field parser are imported from the copied code/sft tree, so the prompt, the 8-frame
448px sampling and the strict JSON parsing are byte-identical to training/evaluation.

Mapping to the submission schema (Dataset/sample_submission.csv):
    pred_emotion        <- four-field JSON "pred_emotion"
    is_uncertain        <- four-field JSON "is_conflict"  (yes/no)
    clarifying_question <- templated from vision/second modality, "none" when no conflict

The official score only reads pred_emotion / is_uncertain (+ conflict truth), so the
question text is a template and is never scored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_protocol(sample_id: str) -> str:
    """Same rule as the published gen_submission.py: ch_sims carries the protocol in
    field 2, while gen:* carries source_dataset there and needs the _va suffix test."""
    parts = sample_id.split(":")
    tag = parts[1] if len(parts) > 1 else "VT"
    if parts[0] == "ch_sims_v2":
        return tag if tag in ("VT", "VA") else "VT"
    return "VA" if tag.endswith("_va") else "VT"


def groups(rows: list[dict], size: int):
    pending: list[dict] = []
    for row in rows:
        if pending and (len(pending) == size or pending[-1]["protocol"] != row["protocol"]):
            yield pending
            pending = []
        pending.append(row)
    if pending:
        yield pending


def collate(items, pad_id):
    """Verbatim copy of sample_omni_dpo_data.collate (left-pad text only)."""
    import torch
    assert items and all(set(x) == set(items[0]) for x in items)
    width = max(x["input_ids"].shape[1] for x in items)
    result = {}
    for key in items[0]:
        values = [x[key] for x in items]
        if key in ("input_ids", "attention_mask"):
            fill = pad_id if key == "input_ids" else 0
            result[key] = torch.cat([torch.nn.functional.pad(v, (width - v.shape[1], 0), value=fill)
                                     for v in values], dim=0)
        elif all(torch.is_tensor(v) for v in values):
            if key in ("input_features", "feature_attention_mask"):
                maximum = max(v.shape[-1] for v in values)
                values = [torch.nn.functional.pad(v, (0, maximum - v.shape[-1])) for v in values]
            assert all(v.shape[1:] == values[0].shape[1:] for v in values), (key, [v.shape for v in values])
            result[key] = torch.cat(values, dim=0)
        elif all(isinstance(v, list) for v in values):
            result[key] = [item for value in values for item in value]
        else:
            assert all(v == values[0] for v in values), key
            result[key] = values[0]
    return result


QUESTION = ("画面呈现的情绪偏向{vision}，但{modality}传递的情绪却是{second}。"
            "两个模态的信号互相矛盾，我应该以哪一侧作为最终判断？")
MODALITY_NAME = {"VT": "文字", "VA": "语音"}


def build_question(prediction: dict, protocol: str) -> str:
    second_key = "text_emotion" if protocol == "VT" else "audio_emotion"
    return QUESTION.format(vision=prediction["vision_emotion"],
                           second=prediction[second_key],
                           modality=MODALITY_NAME[protocol])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="raw Qwen2.5-Omni-7B directory")
    ap.add_argument("--adapter", type=Path, required=True, help="selected SFT or DAPO adapter directory")
    ap.add_argument("--code", type=Path, required=True, help="model_repo/code for the published preprocessing")
    ap.add_argument("--processor-from", type=Path, default=None, help="defaults to --adapter")
    ap.add_argument("--media-root", type=Path, required=True, help="dataset root holding videos/...")
    ap.add_argument("--test", type=Path, required=True, help="test_release.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.code))
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from overfit_omni_modality import TenInputs
    import omni_modality_task as task

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frames").mkdir(exist_ok=True)

    rows = read_jsonl(args.test)
    if args.limit:
        rows = rows[: args.limit]
    prepared = []
    for row in rows:
        prepared.append({"sample_id": row["sample_id"],
                         "protocol": infer_protocol(row["sample_id"]),
                         "video": row["video"],
                         "text_content": row.get("text", row.get("text_content", ""))})
    print("TEST", json.dumps({"n": len(prepared),
                              "protocol": dict(Counter(r["protocol"] for r in prepared))}, ensure_ascii=False), flush=True)

    processor_source = args.processor_from or args.adapter
    try:
        processor = AutoProcessor.from_pretrained(processor_source)
    except Exception as error:
        print("processor fallback to", args.processor_from, "->", error, flush=True)
        processor = AutoProcessor.from_pretrained(args.processor_from)
    tok = processor.tokenizer

    started = time.monotonic()
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("LOADED", json.dumps({"trainable": len(trainable),
                                "load_seconds": round(time.monotonic() - started, 1)}), flush=True)

    inputs = TenInputs(processor, args.media_root, args.out / "frames", prepared)

    def on_device(batch):
        return {k: (v.to(args.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                    if torch.is_tensor(v) else v) for k, v in batch.items()}

    records = []
    done = 0
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    with (args.out / "predictions_raw.jsonl").open("w", encoding="utf-8") as handle:
        for batch_rows in groups(prepared, args.batch_size):
            items, rows_for_gen = [], []
            for row in batch_rows:
                try:
                    items.append(inputs.build(row, audio_first=False, training=False))
                    rows_for_gen.append(row)
                except Exception as error:
                    if row["protocol"] == "VA":
                        print("VA media failed, falling back to VT prompt:", row["sample_id"], error, flush=True)
                        fallback = dict(row, protocol="VT")
                        items.append(inputs.build(fallback, audio_first=False, training=False))
                        rows_for_gen.append(dict(row, protocol="VT", _fallback=True))
                    else:
                        print("media failed, skipped:", row["sample_id"], error, flush=True)
                        records.append({"sample_id": row["sample_id"], "protocol": row["protocol"],
                                        "parse_error": "media_failed:" + str(error)[:120],
                                        "prediction": {}, "raw_output": ""})
                inputs.cached.clear()
            if not items:
                continue
            cpu = collate(items, pad)
            gpu = on_device(cpu)
            with torch.inference_mode():
                generated = model.generate(**gpu, do_sample=False, num_beams=1,
                                           max_new_tokens=args.max_new_tokens, use_cache=True,
                                           eos_token_id=eos, pad_token_id=pad)
            offset = cpu["input_ids"].shape[1]
            for index, row in enumerate(rows_for_gen):
                tokens = generated[index, offset:].tolist()
                if eos in tokens:
                    tokens = tokens[: tokens.index(eos) + 1]
                raw = processor.decode(tokens, skip_special_tokens=True)
                prediction, error = task.parse_answer(raw, row["protocol"])
                records.append({"sample_id": row["sample_id"], "protocol": row["protocol"],
                                "prompt_protocol": row["protocol"],
                                "fallback": bool(row.get("_fallback")),
                                "prediction": prediction, "parse_error": error, "raw_output": raw,
                                "output_tokens": len(tokens), "generation": "greedy"})
                handle.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
            done += len(rows_for_gen)
            handle.flush()
            elapsed = time.monotonic() - started
            print(json.dumps({"stage": "infer", "done": done, "n": len(prepared),
                              "elapsed_s": round(elapsed, 1)}), flush=True)
            del cpu, gpu, generated, items

    order = {r["sample_id"]: i for i, r in enumerate(prepared)}
    metrics = {"done": done, "skipped": len(prepared) - done,
               "parse_ok": sum(r["parse_error"] is None for r in records),
               "parse_errors": dict(Counter(r["parse_error"] for r in records if r["parse_error"])),
               "fallback": sum(bool(r.get("fallback")) for r in records),
               "seconds": round(time.monotonic() - started, 1)}
    print("INFER_DONE", json.dumps(metrics, ensure_ascii=False), flush=True)

    # Assemble the CSV in the exact order of the official template.
    by_id = {r["sample_id"]: r for r in records}
    submission = args.out / "submission.csv"
    counts = Counter()
    with submission.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sample_id,pred_emotion,is_uncertain,clarifying_question\n")
        for row in prepared:
            record = by_id.get(row["sample_id"], {})
            prediction = record.get("prediction") or {}
            pred = prediction.get("pred_emotion")
            conflict = prediction.get("is_conflict")
            if pred not in ("positive", "negative", "neutral"):
                pred = "negative"
                counts["pred_fallback"] += 1
            if conflict not in ("yes", "no"):
                conflict = "no"
                counts["uncertain_fallback"] += 1
            question = "none" if conflict == "no" else build_question(prediction, row["protocol"])
            counts["emotion_" + pred] += 1
            counts["uncertain_" + conflict] += 1
            handle.write("%s,%s,%s,%s\n" % (row["sample_id"], pred, conflict, question))
    summary = {"rows": len(prepared), "submission": str(submission),
               "counts": dict(counts), "infer": metrics}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print("SUBMISSION", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
