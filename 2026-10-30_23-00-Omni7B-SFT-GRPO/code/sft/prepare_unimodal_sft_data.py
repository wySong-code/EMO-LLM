"""Derive single-modality supervision from the fixed, cleaned course training split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import omni_modality_task as task


SINGLE_PROMPT = (
    "你是一个情感识别助手。只分析实际提供的一种模态，任务说明和格式规则不属于待分析文字。\n"
    "vision_emotion 依据画面判断；text_emotion 依据待分析文字判断；"
    "audio_emotion 依据实际音频判断。\n"
    "情感类别只能是 positive（正向）、negative（负向）、neutral（中性）。\n"
    "只输出一个平铺的 JSON 对象，不要解释或 Markdown。根据实际输入选择一种格式：\n"
    "仅画面：vision_emotion。\n仅待分析文字：text_emotion。\n仅音频：audio_emotion。\n"
    "只输出实际提供模态对应的一个字段，不输出未提供模态的字段、is_conflict、"
    "pred_emotion 或 clarifying_question。"
)
GROUP_MASS = {"full": 2 / 3, "vision_only": 1 / 9, "text_only": 1 / 9, "audio_only": 1 / 9}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def derive(rows):
    pools = {name: [] for name in GROUP_MASS}
    groups = {name: defaultdict(list) for name in ("vision_only", "text_only", "audio_only")}
    for row in rows:
        gold = task.target(row)
        inputs = {"video": row["video"]}
        if row["protocol"] == "VT":
            assert isinstance(row["text_content"], str) and row["text_content"].strip()
            inputs["text"] = row["text_content"]
            groups["text_only"][row["text_content"]].append(row)
        else:
            inputs["audio_from_video"] = row["video"]
            groups["audio_only"][row["video"]].append(row)
        pools["full"].append({
            "example_id": "full:" + row["sample_id"], "task": "full", "protocol": row["protocol"],
            "prompt_family": "full", "inputs": inputs, "target": gold,
            "source_sample_ids": [row["sample_id"]], "split": "train",
        })
        groups["vision_only"][row["video"]].append(row)
    for name, grouped in groups.items():
        source_field = "M1" if name == "vision_only" else "M2"
        output_field = name.replace("_only", "_emotion")
        for number, (content, sources) in enumerate(grouped.items(), 1):
            labels = {r["views"][source_field]["label"] for r in sources}
            if len(labels) != 1:
                raise ValueError(f"Same {name} input has different course labels; no relabeling is allowed")
            if name == "vision_only":
                inputs = {"video": content}
            elif name == "text_only":
                inputs = {"text": content}
            else:
                inputs = {"audio_from_video": content}
            pools[name].append({
                "example_id": f"{name}:{number:06d}", "task": name, "prompt_family": "single",
                "inputs": inputs, "target": {output_field: next(iter(labels))},
                "source_sample_ids": [r["sample_id"] for r in sources], "split": "train",
            })
    total = sum(len(pool) for pool in pools.values())
    for name, pool in pools.items():
        assert pool
        weight = total * GROUP_MASS[name] / len(pool)
        for example in pool:
            example["loss_weight"] = weight
    return pools


def validate(pools, rows, val, root):
    by_id = {r["sample_id"]: r for r in rows}
    val_ids = {r["sample_id"] for r in val}
    val_videos = {r["video"] for r in val}
    seen = set()
    for name, pool in pools.items():
        for example in pool:
            assert example["example_id"] not in seen
            seen.add(example["example_id"])
            assert example["split"] == "train"
            assert set(example["source_sample_ids"]) <= set(by_id)
            assert not set(example["source_sample_ids"]) & val_ids
            inputs = example["inputs"]
            assert set(inputs) <= {"video", "text", "audio_from_video"}
            for key in ("video", "audio_from_video"):
                if key in inputs:
                    source = (root / inputs[key]).resolve()
                    assert source.is_relative_to(root.resolve()) and source.is_file()
                    assert inputs[key] not in val_videos
            for source_id in example["source_sample_ids"]:
                original = by_id[source_id]
                if name == "full":
                    assert example["target"] == task.target(original)
                    assert inputs["video"] == original["video"]
                    assert ("audio_from_video" in inputs) == (original["protocol"] == "VA")
                    assert ("text" in inputs) == (original["protocol"] == "VT")
                    if original["protocol"] == "VT":
                        assert inputs["text"] == original["text_content"]
                    else:
                        assert inputs["audio_from_video"] == original["video"]
                else:
                    key = {"vision_only": "video", "text_only": "text", "audio_only": "audio_from_video"}[name]
                    field = name.replace("_only", "_emotion")
                    source_field = "M1" if name == "vision_only" else "M2"
                    assert set(inputs) == {key}
                    assert example["target"] == {field: original["views"][source_field]["label"]}
                    assert inputs[key] == (original["text_content"] if key == "text" else original["video"])
                    assert name != "text_only" or original["protocol"] == "VT"
                    assert name != "audio_only" or original["protocol"] == "VA"
    all_examples = [x for pool in pools.values() for x in pool]
    total = len(all_examples)
    assert len(pools["full"]) == len(rows)
    assert abs(sum(e["loss_weight"] for e in all_examples) / total - 1) < 1e-10
    for name, pool in pools.items():
        assert abs(sum(e["loss_weight"] for e in pool) / total - GROUP_MASS[name]) < 1e-10
    return {
        "unique_example_ids": True, "all_sources_in_fixed_clean_train": True,
        "all_full_examples_preserved": True, "no_validation_ids_or_video_paths": True,
        "course_labels_unchanged": True, "no_missing_modality_targets": True,
        "all_media_paths_exist": True, "group_loss_weights_verified": True,
        "visual_deduplication": "Exact video path; not pixel-level matching across different files",
        "text_deduplication": "Exact original text, without normalization",
        "audio_deduplication": "Exact source video path among VA records",
        "media_tensor_and_token_alignment": "Not run; requires the single-modality data loader",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("Dataset"))
    parser.add_argument("--train-manifest", type=Path, default=Path("results/modality-sft-20261025/startup/train_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/omni-unimodal-sft-data-20261026"))
    parser.add_argument("--evidence", type=Path, default=Path("results/omni-unimodal-sft-data-20261026"))
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()
    assert args.micro_batch > 0 and args.gradient_accumulation > 0
    raw = read_jsonl(args.dataset / "train.jsonl")
    val = read_jsonl(args.dataset / "val.jsonl")
    manifest = read_jsonl(args.train_manifest)
    by_id = {r["sample_id"]: r for r in raw}
    assert len(by_id) == len(raw) == 3354
    assert len(manifest) == len({m["sample_id"] for m in manifest}) == 3352
    rows = [by_id[m["sample_id"]] for m in manifest]
    assert all(row["video"] == item["video"] for row, item in zip(rows, manifest))
    assert len(val) == 666
    pools = derive(rows)
    checks = validate(pools, rows, val, args.dataset)
    total = sum(len(pool) for pool in pools.values())
    summary = {
        "status": "data_prepared_training_not_started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_train_records": 3354, "clean_train_records": len(rows), "val_records": len(val),
        "train_manifest": args.train_manifest.as_posix(), "output": args.output.as_posix(),
        "training_inputs": total, "auxiliary_inputs": total - len(rows),
        "new_independent_source_records": 0,
        "counts": {name: len(pool) for name, pool in pools.items()},
        "full_protocols": dict(Counter(r["protocol"] for r in rows)),
        "auxiliary_label_counts": {name: dict(Counter(next(iter(e["target"].values())) for e in pool))
                                   for name, pool in pools.items() if name != "full"},
        "duplicate_views_removed": {"vision_only": len(rows) - len(pools["vision_only"]),
                                    "text_only": sum(r["protocol"] == "VT" for r in rows) - len(pools["text_only"]),
                                    "audio_only": sum(r["protocol"] == "VA" for r in rows) - len(pools["audio_only"])},
        "checks": checks,
    }
    plan = {
        "status": "proposal_not_launched", "base_model": "Qwen/Qwen2.5-Omni-3B",
        "batch": {"micro_batch_per_gpu": args.micro_batch,
                  "gradient_accumulation_steps": args.gradient_accumulation, "world_size": 1,
                  "effective_batch": args.micro_batch * args.gradient_accumulation,
                  "optimizer_steps_per_mixed_epoch": math.ceil(total / (args.micro_batch * args.gradient_accumulation)),
                  "last_optimizer_group_examples": total % (args.micro_batch * args.gradient_accumulation),
                  "partial_batch": "Keep the tail; normalize by the actual example count of the optimizer group",
                  "preflight": "Test micro-batch 4, including full VT/VA mixed with vision-only, text-only and audio-only; no batch-1 fallback"},
        "coverage": "One shuffled pass over every full and auxiliary example per mixed-data epoch",
        "groups": {name: {"count": len(pool), "loss_mass": GROUP_MASS[name], "per_example_weight": pool[0]["loss_weight"]}
                   for name, pool in pools.items()},
        "loss": "Mean(loss_weight * per-example mean assistant-token CE). Mask input and padding.",
        "weight_note": "Use the fixed weights. Do not divide by each micro-batch's sum of weights; at micro-batch 1 that cancels weighting.",
        "prompt_policy": "Full VT/VA instructions are unchanged. Auxiliary examples share a separate V/T/A prompt with a single available-modality output.",
        "vision_only": "Feed decoded video frames only; use_audio_in_video=false; no input audio features or analyzed transcript.",
        "audio_only": "Decode the source video's audio to mono 16kHz; feed audio only, with no video frames or transcript.",
        "text_only": "Feed only the original VT text, with no media features.",
        "validation": "Unchanged 666 full VT/VA inputs; also report original conflicts, aligned false positives and all four label accuracies.",
        "label_policy": "Only course M1/M2 labels for auxiliaries. No M12, conflict or question supervision from a single modality.",
        "implementation_status": "Manifest preparation complete. Training loader, supervised token checks and GPU preflight still required.",
        "active_encoder_training": "Independent experiment; no changes to the existing visual/audio LoRA run.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.evidence.mkdir(parents=True, exist_ok=False)
    for name, pool in pools.items():
        write_jsonl(args.output / f"{name}_train.jsonl", pool)
    write_jsonl(args.output / "train_pool.jsonl", [e for pool in pools.values() for e in pool])
    write_json(args.output / "prompts.json", {"full": {"VT_first": task.system_prompt(False), "VA_first": task.system_prompt(True)},
                                             "single": SINGLE_PROMPT})
    for dest in (args.output, args.evidence):
        write_json(dest / "preparation_summary.json", summary)
        write_json(dest / "mixture_plan.json", plan)
    write_json(args.evidence / "example_schemas.json", {
        name: {"task": name, "input_fields": list(pool[0]["inputs"]), "target": pool[0]["target"],
               "prompt_family": pool[0]["prompt_family"], "loss_weight": pool[0]["loss_weight"]}
        for name, pool in pools.items()
    })
    print(json.dumps({"counts": summary["counts"], "training_inputs": total,
                      "output": str(args.output), "status": summary["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
