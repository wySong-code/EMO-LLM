"""Scan the clean TRAIN split with frozen SFT and build four-label preference pairs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import omni_modality_task as task


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def error_types(gold, prediction, protocol):
    second = task.second_key(protocol)
    wrong = [k for k in gold if prediction.get(k) != gold[k]]
    if not wrong:
        return [], []
    kinds = []
    if gold["is_conflict"] == "yes" and prediction["is_conflict"] == "no":
        kinds.append("missed_conflict")
    if gold["is_conflict"] == "no" and prediction["is_conflict"] == "yes":
        kinds.append("false_conflict")
    if (gold["is_conflict"] == prediction["is_conflict"] == "no"
            and prediction["vision_emotion"] == prediction[second]
            and "vision_emotion" in wrong and second in wrong):
        kinds.append("aligned_both_modalities_wrong")
    if "vision_emotion" in wrong or second in wrong:
        kinds.append("modality_error")
        if "pred_emotion" not in wrong:
            kinds.append("modality_wrong_final_correct")
    if "pred_emotion" in wrong:
        kinds.append("final_emotion_error")
    if wrong == ["pred_emotion"]:
        kinds.append("final_emotion_only")
    if prediction["is_conflict"] != ("yes" if prediction["vision_emotion"] != prediction[second] else "no"):
        kinds.append("predicted_label_inconsistency")
    return wrong, kinds


def inspect_record(record, row):
    """Use course targets, not model-provided correctness or reward fields."""
    gold = task.target(row)
    prediction, error = task.parse_answer(record["raw_output"], row["protocol"])
    if record.get("gold") != gold:
        raise ValueError("Stored target differs from course labels: " + row["sample_id"])
    if error:
        return None, "format:" + error
    if record.get("token_limit_hit"):
        return None, "truncated"
    wrong, kinds = error_types(gold, prediction, row["protocol"])
    if not wrong:
        return None, "all_four_correct"
    pair = {"sample_id": row["sample_id"], "protocol": row["protocol"],
            "source": "generated" if row.get("source_is_generated") else "original",
            "sample_type": row["sample_type"], "video": row["video"],
            "chosen": json.dumps(gold, ensure_ascii=False, separators=(",", ":")),
            "rejected": json.dumps({k: prediction[k] for k in gold}, ensure_ascii=False, separators=(",", ":")),
            "wrong_fields": wrong, "error_types": kinds,
            "generation": record["generation"], "seed": record["seed"]}
    return pair, "selected"


def clean_rows(dataset, manifest):
    entries = read_jsonl(manifest)
    ids = [r["sample_id"] for r in entries]
    if len(ids) != len(set(ids)) or len(ids) != 3352:
        raise ValueError("Expected exactly 3352 unique clean training IDs")
    original = read_jsonl(dataset / "train.jsonl")
    lookup = {r["sample_id"]: r for r in original}
    if len(lookup) != len(original):
        raise ValueError("Duplicate IDs in course train.jsonl")
    rows = [lookup[sid] for sid in ids]
    videos = {r["video"] for r in rows}
    for entry, row in zip(entries, rows):
        if entry["video"] != row["video"]:
            raise ValueError("Training manifest video mismatch")
        task.target(row)
    split_check = {"clean_train": len(rows), "course_train": len(original)}
    # Read held-out identities only; their labels never enter the selection process.
    for name, filename, expected in (("validation", "val.jsonl", 666), ("test", "test_release.jsonl", 734)):
        held = read_jsonl(dataset / filename)
        id_overlap = set(ids).intersection(r["sample_id"] for r in held)
        video_overlap = videos.intersection(r["video"] for r in held)
        split_check[name] = {"n": len(held), "id_overlap": len(id_overlap), "video_overlap": len(video_overlap)}
        if len(held) != expected or id_overlap or video_overlap:
            raise ValueError("Unexpected split or overlap: " + json.dumps(split_check[name]))
    return rows, split_check


def summarize(rows, records):
    lookup = {r["sample_id"]: r for r in rows}
    pairs, correct, excluded = [], [], []
    seen, selected_per_id = set(), Counter()
    for record in records:
        row = lookup[record["sample_id"]]
        pair, reason = inspect_record(record, row)
        if pair is None:
            destination = correct if reason == "all_four_correct" else excluded
            destination.append({"sample_id": row["sample_id"], "reason": reason})
            continue
        identity = (pair["sample_id"], pair["rejected"])
        if identity in seen or selected_per_id[pair["sample_id"]] >= 2:
            continue
        seen.add(identity)
        selected_per_id[pair["sample_id"]] += 1
        pairs.append(pair)
    fields = Counter(k for p in pairs for k in p["wrong_fields"])
    kinds = Counter(k for p in pairs for k in p["error_types"])
    summary = {"scanned_inputs": len({r["sample_id"] for r in records}), "total_clean_train": len(rows),
               "generated_answers": len(records), "selected_pairs": len(pairs),
               "selected_inputs": len(selected_per_id), "all_four_correct_answers": len(correct),
               "excluded_answers": len(excluded), "error_type_counts_nonexclusive": dict(kinds),
               "wrong_field_counts": dict(fields),
               "selected_by_source": dict(Counter(p["source"] for p in pairs)),
               "selected_by_protocol": dict(Counter(p["protocol"] for p in pairs)),
               "selected_by_gold_type": dict(Counter(p["sample_type"] for p in pairs)),
               "selected_strata": dict(Counter("/".join((p["source"], p["sample_type"], p["protocol"])) for p in pairs)),
               "sampling_policy": "one greedy answer per clean training input; no class resampling",
               "selection_rule": "any of the four course label values differs",
               "training_started": False, "updated_utc": stamp()}
    return pairs, correct, excluded, summary


def export_selection(rows, records, dataset, cache, out):
    from prepare_omni_grpo import make_record
    lookup = {r["sample_id"]: r for r in rows}
    pairs, correct, excluded, summary = summarize(rows, records)
    dpo, sft = [], []
    for pair in pairs:
        row = lookup[pair["sample_id"]]
        record = make_record(row, dataset, cache)
        messages = record["messages"] + [{"role": "assistant", "content": pair["chosen"]}]
        common = {"messages": messages, "videos": record["videos"], "audios": record["audios"],
                  "sample_id": row["sample_id"]}
        dpo.append({**common, "rejected_response": pair["rejected"]})
        sft.append(common)
    write_jsonl(out / "preference_pairs.jsonl", pairs)
    write_jsonl(out / "dpo_train.jsonl", dpo)
    write_jsonl(out / "sft_error_control.jsonl", sft)
    write_jsonl(out / "correct_pool_manifest.jsonl", correct)
    write_jsonl(out / "excluded_answers.jsonl", excluded)
    write_json(out / "selection_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    for key in ("model", "adapter", "dataset", "manifest", "cache", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and not args.resume:
        raise FileExistsError("Output already exists; inspect it before explicitly resuming")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "process.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoProcessor, set_seed
    import train_omni_sft as original
    from overfit_omni_modality import TenInputs

    started = time.monotonic()
    try:
        rows, split_check = clean_rows(args.dataset, args.manifest)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
        config.update(training=False, question_generation=False, sample_split="clean_train_only",
                      frames=8, max_side=448, audio_rate=16000, do_sample=False, max_new_tokens=192,
                      dtype="bfloat16", attention="sdpa", system_prompt=task.system_prompt(False),
                      versions={n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft")})
        if args.resume:
            if json.loads((out / "config.json").read_text()) != config:
                raise ValueError("Resume configuration differs from saved run")
        else:
            write_json(out / "config.json", config)
            shutil.copyfile(__file__, out / "selection_source.py")
        write_json(out / "split_check.json", split_check)
        raw_path = out / "train_predictions.jsonl"
        records = read_jsonl(raw_path) if raw_path.exists() else []
        if len(records) != len({r["sample_id"] for r in records}):
            raise ValueError("Duplicate scan IDs")
        expected_ids = [r["sample_id"] for r in rows]
        if [r["sample_id"] for r in records] != expected_ids[:len(records)]:
            raise ValueError("Existing records do not match clean manifest prefix")
        previous = len(records)
        write_json(out / "status.json", {"stage": "loading_model", "completed": previous,
                   "total": len(rows), "pid": os.getpid(), "updated_utc": stamp()})
        torch.set_num_threads(4)
        set_seed(args.seed)
        processor = AutoProcessor.from_pretrained(str(args.model))
        model = PeftModel.from_pretrained(original.load_base(str(args.model), out), str(args.adapter)).eval()
        assert not any(p.requires_grad for p in model.parameters())
        wanted = load_file(str(args.adapter / "adapter_model.safetensors"))
        loaded = get_peft_model_state_dict(model)
        assert set(wanted) == set(loaded)
        assert all(torch.equal(v.float(), loaded[k].detach().cpu().float()) for k, v in wanted.items())
        write_json(out / "adapter_check.json", {"tensors": len(wanted), "weights_equal": True,
                   "all_parameters_frozen": True, "adapter": str(args.adapter)})
        del wanted, loaded
        model.config.use_cache = True
        inputs = TenInputs(processor, args.dataset, args.cache, [])
        inference_started = time.monotonic()
        with raw_path.open("a", encoding="utf-8") as f:
            for index, row in enumerate(rows[previous:], previous + 1):
                # Only media and the course input text are supplied to generation.
                media = {k: row[k] for k in ("sample_id", "protocol", "video", "text_content")}
                batch = original.on_device(inputs.build(media, audio_first=False, training=False))
                inputs.cached.clear()
                assert ("input_features" in batch) == (row["protocol"] == "VA")
                with torch.inference_mode():
                    generated = model.generate(**batch, do_sample=False, max_new_tokens=192, use_cache=True,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id)
                tokens = generated[0, batch["input_ids"].shape[-1]:]
                raw = processor.decode(tokens, skip_special_tokens=True)
                prediction, error = task.parse_answer(raw, row["protocol"])
                record = {"sample_id": row["sample_id"], "protocol": row["protocol"],
                          "source": "generated" if row.get("source_is_generated") else "original",
                          "sample_type": row["sample_type"], "gold": task.target(row),
                          "prediction": prediction, "parse_error": error, "raw_output": raw,
                          "generation": "greedy", "seed": args.seed, "output_tokens": len(tokens),
                          "prompt_tokens": batch["input_ids"].shape[-1],
                          "has_audio_features": "input_features" in batch,
                          "token_limit_hit": bool(len(tokens) >= 192 and tokens[-1].item() != processor.tokenizer.eos_token_id)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                records.append(record)
                del batch, generated, tokens
                if index == previous + 1 or index % 25 == 0 or index == len(rows):
                    _, _, _, summary = summarize(rows, records)
                    elapsed = time.monotonic() - inference_started
                    status = {"stage": "scanning", "completed": index, "total": len(rows),
                              "selected_pairs": summary["selected_pairs"],
                              "error_types": summary["error_type_counts_nonexclusive"],
                              "elapsed_seconds": round(time.monotonic() - started, 1),
                              "seconds_per_input": round(elapsed / (index - previous), 3),
                              "remaining_seconds_estimate": round(elapsed / (index - previous) * (len(rows) - index)),
                              "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                              "pid": os.getpid(), "updated_utc": stamp()}
                    write_json(out / "status.json", status)
                    write_json(out / "selection_summary.partial.json", summary)
                    print(json.dumps(status), flush=True)
        summary = export_selection(rows, records, args.dataset, args.cache, out)
        write_json(out / "status.json", {**status, "stage": "complete", "training_started": False,
                   "finished_utc": stamp(), "elapsed_seconds": round(time.monotonic() - started, 1)})
        print("SELECTION_COMPLETE", json.dumps(summary), flush=True)
    except Exception:
        write_json(out / "failure.json", {"stage": "failed", "updated_utc": stamp(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
