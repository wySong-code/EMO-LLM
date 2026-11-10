"""Benchmark padded Omni inference, then sample clean TRAIN inputs for DPO."""
import argparse
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import omni_modality_task as task
import select_omni_dpo_data as selection


def groups(rows, size):
    pending = []
    for index, row in enumerate(rows):
        if pending and (len(pending) == size or pending[-1][1]["protocol"] != row["protocol"]):
            yield pending
            pending = []
        pending.append((index, row))
    if pending:
        yield pending


def collate(items, pad_id):
    """Keep each example's encoded media unchanged; left-pad only text tokens."""
    import torch
    assert items and all(set(x) == set(items[0]) for x in items)
    width = max(x["input_ids"].shape[1] for x in items)
    result = {}
    for key in items[0]:
        values = [x[key] for x in items]
        if key in ("input_ids", "attention_mask"):
            fill = pad_id if key == "input_ids" else 0
            result[key] = torch.cat([torch.nn.functional.pad(v, (width-v.shape[1], 0), value=fill)
                                     for v in values], dim=0)
        elif all(torch.is_tensor(v) for v in values):
            if key in ("input_features", "feature_attention_mask"):
                maximum = max(v.shape[-1] for v in values)
                values = [torch.nn.functional.pad(v, (0, maximum-v.shape[-1])) for v in values]
            assert all(v.shape[1:] == values[0].shape[1:] for v in values), (key, [v.shape for v in values])
            result[key] = torch.cat(values, dim=0)
        elif all(isinstance(v, list) for v in values):
            result[key] = [item for value in values for item in value]
        else:
            assert all(v == values[0] for v in values), key
            result[key] = values[0]
    for index, item in enumerate(items):
        real = result["input_ids"][index][result["attention_mask"][index].bool()]
        assert torch.equal(real, item["input_ids"][0][item["attention_mask"][0].bool()])
    assert result["video_grid_thw"].shape[0] == len(items)
    if "input_features" in result:
        assert result["input_features"].shape[0] == len(items)
    return result


def build(inputs, row):
    media = {k: row[k] for k in ("sample_id", "protocol", "video", "text_content")}
    item = inputs.build(media, audio_first=False, training=False)
    inputs.cached.clear()
    return item


def generate(model, processor, items, rows, sample, seed):
    import torch
    from transformers import set_seed
    import train_omni_sft as original
    set_seed(seed)
    cpu = collate(items, processor.tokenizer.pad_token_id)
    batch = original.on_device(cpu)
    kwargs = {"do_sample": sample, "num_beams": 1, "max_new_tokens": 192, "use_cache": True,
              "eos_token_id": processor.tokenizer.eos_token_id,
              "pad_token_id": processor.tokenizer.pad_token_id}
    if sample:
        kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
    with torch.inference_mode():
        outputs = model.generate(**batch, **kwargs)
    records = []
    for i, row in enumerate(rows):
        tokens = outputs[i, batch["input_ids"].shape[1]:].tolist()
        eos = processor.tokenizer.eos_token_id
        if eos in tokens:
            tokens = tokens[:tokens.index(eos)+1]
        raw = processor.decode(tokens, skip_special_tokens=True)
        prediction, error = task.parse_answer(raw, row["protocol"])
        records.append({"sample_id": row["sample_id"], "protocol": row["protocol"],
                        "source": "generated" if row.get("source_is_generated") else "original",
                        "sample_type": row["sample_type"], "gold": task.target(row),
                        "prediction": prediction, "parse_error": error, "raw_output": raw,
                        "generation": "temperature_1" if sample else "greedy", "seed": seed,
                        "temperature": 1.0 if sample else None, "batch_size": len(rows),
                        "output_tokens": len(tokens), "prompt_tokens": int(items[i]["attention_mask"].sum()),
                        "padded_prompt_tokens": batch["input_ids"].shape[1],
                        "has_audio_features": "input_features" in items[i],
                        "token_limit_hit": bool(len(tokens) >= 192 and tokens[-1] != eos)})
    del batch, outputs
    return records


def choose_benchmark(rows, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["protocol"], bool(row.get("source_is_generated")), row["sample_type"])].append(row)
    selected = []
    for protocol in ("VT", "VA"):
        chosen = []
        for key in sorted(k for k in buckets if k[0] == protocol):
            chosen.extend(rng.sample(buckets[key], min(2, len(buckets[key]))))
        used = {r["sample_id"] for r in chosen}
        candidates = [r for r in rows if r["protocol"] == protocol and r["sample_id"] not in used]
        chosen.extend(rng.sample(candidates, 4-len(chosen)))
        selected.extend(chosen)
    assert len(selected) == len({r["sample_id"] for r in selected}) == 8
    return selected


def benchmark(args, model, processor, inputs, rows):
    import torch
    prior = selection.read_jsonl(args.greedy_source/"train_predictions.jsonl")
    reference = {r["sample_id"]: r for r in prior}
    selected = choose_benchmark([r for r in rows if r["sample_id"] in reference], args.seed)
    begin = time.monotonic()
    items = {r["sample_id"]: build(inputs, r) for r in selected}
    prepare_seconds = time.monotonic()-begin
    selection.write_json(args.output/"samples.json", {
        "sample_ids": [r["sample_id"] for r in selected], "n": len(selected),
        "cpu_preprocessing_seconds": prepare_seconds,
        "shapes": {r["sample_id"]: {k: list(v.shape) if torch.is_tensor(v) else v
                                   for k,v in items[r["sample_id"]].items()} for r in selected}})
    for warm_rows in (selected[:2], selected[4:6]):
        generate(model, processor, [items[r["sample_id"]] for r in warm_rows], warm_rows, False, args.seed)
    results = []
    for size in (2, 4):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        answers = []
        try:
            for group in groups(selected, size):
                batch_rows = [r for _,r in group]
                answers.extend(generate(model, processor, [items[r["sample_id"]] for r in batch_rows],
                                        batch_rows, False, args.seed))
            torch.cuda.synchronize()
            seconds = time.monotonic()-begin
            agreement = sum(r["parse_error"] is None and r["prediction"] == reference[r["sample_id"]]["prediction"]
                            for r in answers)
            invalid = sum(r["parse_error"] is not None or r["token_limit_hit"] for r in answers)
            result = {"batch_size": size, "n": len(answers), "generation_seconds": round(seconds,3),
                      "generation_samples_per_second": round(len(answers)/seconds,3),
                      "estimated_serial_end_to_end_samples_per_second": round(len(answers)/(seconds+prepare_seconds),3),
                      "peak_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30,3),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30,3),
                      "label_agreement_with_batch1": agreement, "invalid_answers": invalid,
                      "passed": agreement == len(selected) and invalid == 0}
            selection.write_jsonl(args.output/f"batch_{size}_predictions.jsonl", answers)
        except torch.cuda.OutOfMemoryError:
            result = {"batch_size": size, "passed": False, "error": "cuda_out_of_memory"}
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        selection.write_json(args.output/"benchmark.partial.json", results)
        print("BATCH_RESULT", json.dumps(result), flush=True)
    eligible = [r for r in results if r["passed"] and r["peak_reserved_gib"] < 55]
    assert eligible, "No batch configuration passed"
    best = max(eligible, key=lambda r:r["estimated_serial_end_to_end_samples_per_second"])
    stochastic = []
    for group in groups(selected, best["batch_size"]):
        batch_rows = [r for _,r in group]
        stochastic.extend(generate(model, processor, [items[r["sample_id"]] for r in batch_rows],
                                   batch_rows, True, args.seed+group[0][0]))
    selection.write_jsonl(args.output/"temperature1_smoke_predictions.jsonl", stochastic)
    summary = {"stage": "complete", "recommended_batch_size": best["batch_size"], "results": results,
               "cpu_preprocessing_seconds": round(prepare_seconds,3),
               "temperature1_smoke": {"n": len(stochastic),
                   "valid": sum(r["parse_error"] is None and not r["token_limit_hit"] for r in stochastic),
                   "wrong": sum(r["parse_error"] is None and r["prediction"] != r["gold"] for r in stochastic)},
               "training_started": False, "updated_utc": selection.stamp()}
    selection.write_json(args.output/"benchmark.json", summary)
    selection.write_json(args.output/"status.json", summary)
    print("BENCHMARK_COMPLETE", json.dumps(summary), flush=True)


def sample(args, model, processor, inputs, rows):
    import torch
    prior = selection.read_jsonl(args.greedy_source/"train_predictions.jsonl")
    previous_config = json.loads((args.greedy_source/"config.json").read_text())
    for key in ("model", "adapter", "dataset", "manifest", "cache"):
        assert str(getattr(args, key)) == previous_config[key], key
    lookup = {r["sample_id"]: r for r in rows}
    for record in prior:
        selection.inspect_record(record, lookup[record["sample_id"]])
    selection.write_jsonl(args.output/"greedy_predictions_preserved.jsonl", prior)
    raw_path = args.output/"sampled_predictions.jsonl"
    answers = selection.read_jsonl(raw_path) if raw_path.exists() else []
    expected = [(repeat, i, r["sample_id"]) for repeat in range(1,args.repeats+1) for i,r in enumerate(rows)]
    assert [(r["sampling_pass"],r["manifest_index"],r["sample_id"]) for r in answers] == expected[:len(answers)]
    already = len(answers)
    start = time.monotonic()
    total = len(rows)*args.repeats
    with raw_path.open("a",encoding="utf-8") as handle:
        for repeat in range(1,args.repeats+1):
            offset = max(0, already-(repeat-1)*len(rows))
            if offset >= len(rows):
                continue
            for group in groups(rows[offset:], args.batch_size):
                batch_rows = [r for _,r in group]
                cpu = [build(inputs,r) for r in batch_rows]
                batch_seed = args.seed+repeat*1000000+offset+group[0][0]
                records = generate(model,processor,cpu,batch_rows,True,batch_seed)
                for (i,row), record in zip(group,records):
                    record.update(sampling_pass=repeat, manifest_index=offset+i)
                    handle.write(json.dumps(record,ensure_ascii=False)+"\n")
                    answers.append(record)
                handle.flush()
                del cpu
                elapsed = time.monotonic()-start
                completed = len(answers)
                status = {"stage":"sampling","completed_answers":completed,"total_answers":total,
                          "unique_inputs_sampled":len({r["sample_id"] for r in answers}),
                          "total_inputs":len(rows),"sampling_pass":repeat,"batch_size":args.batch_size,
                          "temperature":1.0,"elapsed_seconds":round(elapsed,1),
                          "answers_per_second":round((completed-already)/elapsed,3),
                          "remaining_seconds_estimate":round(elapsed/(completed-already)*(total-completed)),
                          "peak_gpu_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
                          "pid":os.getpid(),"training_started":False,"updated_utc":selection.stamp()}
                selection.write_json(args.output/"status.json",status)
                if completed % (args.batch_size*8) < args.batch_size or completed == total:
                    _,_,_,summary = selection.summarize(rows,prior+answers)
                    summary["sampling_policy"] = "two temperature-1 samples for every clean training input, plus preserved partial greedy scan"
                    selection.write_json(args.output/"selection_summary.partial.json",summary)
                    print(json.dumps({**status,"selected_pairs":summary["selected_pairs"],
                                      "selected_inputs":summary["selected_inputs"]}),flush=True)
    summary = selection.export_selection(rows,prior+answers,args.dataset,args.cache,args.output)
    summary.update(sampling_policy="two temperature-1 samples for every clean training input, plus preserved partial greedy scan",
                   sampled_answers=len(answers),greedy_answers_preserved=len(prior),temperature=1.0,
                   sampling_passes=args.repeats,batch_size=args.batch_size,
                   actual_batch_size_answer_counts=dict(Counter(r["batch_size"] for r in answers)),
                   distinct_inputs_with_pairs=summary["selected_inputs"])
    selection.write_json(args.output/"selection_summary.json",summary)
    selection.write_json(args.output/"status.json",{"stage":"complete","completed_answers":len(answers),
        "total_answers":total,"total_inputs":len(rows),"selected_pairs":summary["selected_pairs"],
        "selected_inputs":summary["selected_inputs"],"finished_utc":selection.stamp(),"training_started":False})
    print("SAMPLING_COMPLETE",json.dumps(summary),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("benchmark","sample"),required=True)
    for key in ("model","adapter","dataset","manifest","cache","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--greedy-source",type=Path)
    parser.add_argument("--benchmark",type=Path)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--seed",type=int,default=20261026)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    assert args.batch_size in (1,2,4,8,16) and args.repeats == 2
    if args.mode == "sample":
        assert args.greedy_source and args.benchmark
        bench = json.loads(args.benchmark.read_text())
        assert bench["stage"] == "complete" and bench["recommended_batch_size"] == args.batch_size
        assert bench["temperature1_smoke"]["valid"] == bench["temperature1_smoke"]["n"]
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (args.output/"process.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    import train_omni_sft as original
    from overfit_omni_modality import TenInputs
    try:
        rows, split_check = selection.clean_rows(args.dataset,args.manifest)
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k != "resume"}
        config.update(temperature=1.0,top_p=1.0,top_k=0,max_new_tokens=192,frames=8,max_side=448,
                      audio_rate=16000,attention="sdpa",dtype="bfloat16",training=False,
                      system_prompt=task.system_prompt(False))
        if args.resume:
            previous_config = json.loads((args.output/"config.json").read_text())
            stable = lambda c: {k:v for k,v in c.items() if k not in ("batch_size","benchmark")}
            assert stable(previous_config) == stable(config), "Resume changed more than batch settings"
            if previous_config != config:
                history_path = args.output/"batch_changes.json"
                history = json.loads(history_path.read_text()) if history_path.exists() else []
                done = len(selection.read_jsonl(args.output/"sampled_predictions.jsonl"))
                history.append({"previous_batch_size":previous_config["batch_size"],"batch_size":args.batch_size,
                                "completed_answers_before_change":done,"benchmark":str(args.benchmark),
                                "updated_utc":selection.stamp()})
                selection.write_json(history_path,history)
                selection.write_json(args.output/f"config_before_batch_change_{len(history)}.json",previous_config)
                shutil.copyfile(__file__,args.output/f"sampling_source_batch_change_{len(history)}.py")
                selection.write_json(args.output/"config.json",config)
        else:
            selection.write_json(args.output/"config.json",config)
            shutil.copyfile(__file__,args.output/"sampling_source.py")
        selection.write_json(args.output/"split_check.json",split_check)
        selection.write_json(args.output/"status.json",{"stage":"loading_model","pid":os.getpid(),"updated_utc":selection.stamp()})
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(str(args.model))
        model = PeftModel.from_pretrained(original.load_base(str(args.model),args.output),str(args.adapter)).eval()
        assert not any(p.requires_grad for p in model.parameters())
        wanted = load_file(str(args.adapter/"adapter_model.safetensors"))
        loaded = get_peft_model_state_dict(model)
        assert set(wanted) == set(loaded)
        assert all(torch.equal(v.float(),loaded[k].detach().cpu().float()) for k,v in wanted.items())
        selection.write_json(args.output/"adapter_check.json",{"tensors":len(wanted),"weights_equal":True,
            "all_parameters_frozen":True,"adapter":str(args.adapter)})
        del wanted,loaded
        model.config.use_cache = True
        inputs = TenInputs(processor,args.dataset,args.cache,[])
        if args.mode == "benchmark":
            benchmark(args,model,processor,inputs,rows)
        else:
            sample(args,model,processor,inputs,rows)
    except Exception:
        selection.write_json(args.output/"failure.json",{"updated_utc":selection.stamp(),"traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
