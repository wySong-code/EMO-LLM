"""Second GRPO experiment: train-only hard coverage, adaptive groups, two-GPU native loading.

Run --check-data and --check-env before --smoke-test. The latter performs real
VT/VA generation and backward passes without updating the adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

from core import StratifiedSampler, advantages, group_status, score_answer, validate_assets, validate_dataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REQUIRED = {"transformers": "4.57.6", "peft": "0.17.1"}


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def environment_status() -> dict:
    modules = ("torch", "transformers", "peft", "accelerate", "av", "numpy",
               "soundfile", "imageio_ffmpeg", "PIL", "safetensors")
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": sys.version.split()[0], "modules": available, "versions": versions,
              "cuda": False, "gpu_count": 0, "gpu_names": []}
    if available["torch"]:
        try:
            import torch
            result["cuda"] = torch.cuda.is_available()
            if result["cuda"]:
                result["gpu_count"] = torch.cuda.device_count()
                result["gpu_names"] = [torch.cuda.get_device_name(index)
                                       for index in range(result["gpu_count"])]
                result["bf16"] = torch.cuda.is_bf16_supported()
        except Exception as error:
            result["torch_error"] = repr(error)
    result["versions_match"] = all(versions[key] == version for key, version in REQUIRED.items())
    result["ready"] = all(available.values()) and result["cuda"] and result.get("bf16", False) \
                      and result["versions_match"]
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "training_dataset/Dataset")
    parser.add_argument("--base", type=Path, default=PROJECT / "base_model")
    parser.add_argument("--adapter", type=Path, default=PROJECT / "model_repo")
    parser.add_argument("--out", type=Path, default=HERE / "runs/pilot_01")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--reference-device", choices=("auto", "cuda:0", "cuda:1"), default="auto")
    parser.add_argument("--attempts", type=int, default=600)
    parser.add_argument("--initial-candidates", type=int, default=4)
    parser.add_argument("--rescue-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--rescue-temperature", type=float, default=1.45)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--kl-beta", type=float, default=.02)
    parser.add_argument("--clip-eps", type=float, default=.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--decision-advantage-scale", type=float, default=1.5)
    parser.add_argument("--other-advantage-scale", type=float, default=.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--f1-drop-guard", type=float, default=.01)
    parser.add_argument("--specificity-drop-guard", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.attempts < 1 or args.initial_candidates < 2 or args.rescue_candidates < 0
            or args.temperature <= 0 or args.rescue_temperature <= 0
            or not 0 < args.top_p <= 1 or args.max_new_tokens < 1 or args.lr <= 0
            or args.kl_beta < 0 or not 0 < args.clip_eps < 1 or args.ppo_epochs < 1
            or args.decision_advantage_scale <= 0 or args.other_advantage_scale <= 0
            or args.eval_every < 1 or args.eval_batch_size < 1 or args.patience < 1
            or args.f1_drop_guard < 0 or args.specificity_drop_guard < 0):
        parser.error("Invalid training or evaluation parameter")
    return args


def main() -> None:
    args = arguments()
    train, val, data_summary = validate_dataset(args.dataset_root)
    print("DATA_CHECK", json.dumps(data_summary, ensure_ascii=False), flush=True)
    asset_summary = validate_assets(args.base, args.adapter)
    print("ASSET_CHECK", json.dumps(asset_summary, ensure_ascii=False), flush=True)
    if args.check_data:
        return
    status = environment_status()
    print("ENV_CHECK", json.dumps(status, ensure_ascii=False), flush=True)
    if args.check_env:
        if not status["ready"]:
            raise SystemExit(2)
        return
    if not status["ready"] and not (args.allow_version_mismatch and status["cuda"]
                                      and status.get("bf16", False) and all(status["modules"].values())):
        raise RuntimeError("GPU environment is not ready; run --check-env and use recorded versions")
    if args.reference_device == "cuda:1" and status["gpu_count"] < 2:
        raise RuntimeError("--reference-device cuda:1 needs two visible CUDA devices")
    ref_device = ("cuda:1" if status["gpu_count"] >= 2 else "cuda:0") \
                 if args.reference_device == "auto" else args.reference_device
    # Original peer code is the source of truth for video/audio preprocessing
    # and the exact fixed validation metric.
    sys.path.insert(0, str(args.adapter / "code"))
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration, set_seed
    from overfit_omni_modality import TenInputs
    from evaluate_omni_batched import evaluate

    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "config.json", {**vars(args), "reference_device_resolved": ref_device,
                                           "environment": status, "data": data_summary,
                                           "assets": asset_summary})
    torch.set_num_threads(4)
    set_seed(args.seed)
    sampler = StratifiedSampler(train, args.seed)
    processor = AutoProcessor.from_pretrained(args.adapter, local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer needs EOS and PAD token IDs")

    def load(device: str, trainable: bool):
        base, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": device}, low_cpu_mem_usage=True, output_loading_info=True,
            local_files_only=True)
        write_json(args.out / ("policy_loading.json" if trainable else "reference_loading.json"), info)
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError(f"Incomplete base model load on {device}; inspect loading JSON")
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=trainable,
                                         local_files_only=True, torch_device=device)
        model.eval()  # Dropout off: sampled old logprobs match update-time logprobs.
        model.config.use_cache = False
        adapter_devices = {str(parameter.device) for name, parameter in model.named_parameters()
                           if "lora_" in name}
        if adapter_devices != {device}:
            raise RuntimeError(f"Adapter placement mismatch: expected {device}, got {adapter_devices}")
        if not trainable:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        return model

    policy = load("cuda:0", True)
    reference = load(ref_device, False)
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if sum(p.numel() for _, p in trainable) != 54_192_128 or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Loaded adapter does not expose the expected 54,192,128 LoRA parameters")
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=args.lr, weight_decay=0.0)
    inputs = TenInputs(processor, args.dataset_root, args.out / "media_cache", train + val)

    def to_device(item: dict, device: str) -> dict:
        return {key: value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                if torch.is_tensor(value) else value for key, value in item.items()
                if not key.startswith("_")}

    def logprobs(model, prompt: dict, answer: torch.Tensor, *, grad: bool) -> torch.Tensor:
        device = next(model.parameters()).device
        batch = to_device(prompt, str(device))
        answer_ids = answer.to(device).reshape(1, -1)
        prefix = batch["input_ids"].shape[1]
        batch["input_ids"] = torch.cat([batch["input_ids"], answer_ids], dim=1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(answer_ids)], dim=1)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch, use_cache=False).logits[:, prefix - 1:prefix - 1 + answer_ids.shape[1]].float()
            selected = torch.log_softmax(logits, -1).gather(-1, answer_ids.unsqueeze(-1))
            selected = selected.squeeze(0).squeeze(-1)
        return selected if grad else selected.detach().cpu()

    def generate(prompt: dict, temperature: float) -> torch.Tensor:
        gpu_prompt = to_device(prompt, "cuda:0")
        with torch.inference_mode():
            output = policy.generate(**gpu_prompt, do_sample=True, temperature=temperature,
                top_p=args.top_p, max_new_tokens=args.max_new_tokens, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        answer = output[0, gpu_prompt["input_ids"].shape[1]:].detach().cpu().clone()
        if answer.numel() == 0:
            raise RuntimeError("Model returned zero answer tokens")
        return answer

    def update_policy(prompt: dict, answers: list[torch.Tensor], adv: list[float]) -> list[float]:
        """PPO-clipped GRPO update, shared by the GPU smoke test and full run."""
        if len(answers) != len(adv) or not answers or all(abs(value) < 1e-8 for value in adv):
            raise ValueError("Update needs matching answers and nonzero advantages")
        old_lp = [logprobs(policy, prompt, answer, grad=False) for answer in answers]
        ref_lp = [logprobs(reference, prompt, answer, grad=False) for answer in answers]
        gradient_norms = []
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            for answer, old, ref, advantage in zip(answers, old_lp, ref_lp, adv):
                now = logprobs(policy, prompt, answer, grad=True)
                old, ref = old.to(now.device), ref.to(now.device)
                ratio = (now - old).clamp(-20, 20).exp()
                surrogate = -torch.minimum(ratio * advantage,
                    ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantage).mean()
                delta = now - ref
                kl = (torch.exp((-delta).clamp(-20, 20)) + delta - 1).mean()
                loss = (surrogate + args.kl_beta * kl) / len(answers)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GRPO loss")
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_((p for _, p in trainable), 1.0,
                                                  error_if_nonfinite=True)
            if not torch.isfinite(norm) or norm.item() <= 0:
                raise RuntimeError("No finite positive GRPO gradient")
            gradient_norms.append(float(norm.item()))
            optimizer.step()
        return gradient_norms

    def evaluate_now(name: str, step: int) -> dict:
        result, _ = evaluate(policy, inputs, val, args.out, name, None, step,
                             batch_size=args.eval_batch_size)
        inputs.cached.clear()
        return result["all"]

    def smoke() -> None:
        checks = []
        vt_row = None
        vt_prompt = None
        for protocol in ("VT", "VA"):
            row = next(r for r in train if r["protocol"] == protocol)
            prompt = inputs.build(row, audio_first=False, training=False)
            if protocol == "VT":
                vt_row, vt_prompt = row, prompt
            inputs.cached.clear()
            gpu_prompt = to_device(prompt, "cuda:0")
            with torch.inference_mode():
                output = policy.generate(**gpu_prompt, do_sample=False,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            answer = output[0, gpu_prompt["input_ids"].shape[1]:].detach().cpu().clone()
            if answer.numel() == 0:
                raise RuntimeError(f"Empty {protocol} answer")
            reward = score_answer(tokenizer.decode(answer, skip_special_tokens=True), row)
            if not reward.valid:
                raise RuntimeError(f"Invalid {protocol} JSON during smoke test")
            old = logprobs(policy, prompt, answer, grad=False)
            ref = logprobs(reference, prompt, answer, grad=False)
            optimizer.zero_grad(set_to_none=True)
            now = logprobs(policy, prompt, answer, grad=True)
            if not torch.isfinite(now).all() or not torch.isfinite(old).all() or not torch.isfinite(ref).all():
                raise FloatingPointError(f"Nonfinite {protocol} logprobs")
            (-now.mean()).backward()
            counts = Counter()
            for name, parameter in trainable:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Nonfinite gradient: {name}")
                if not torch.count_nonzero(gradient).item():
                    continue
                group = ("vision" if ".visual.blocks." in name else
                         "audio" if ".audio_tower.layers." in name else
                         "language" if ".model.layers." in name else "unexpected")
                counts[group] += 1
            needed = {"vision", "language"} | ({"audio"} if protocol == "VA" else set())
            if any(counts[group] == 0 for group in needed):
                raise RuntimeError(f"Missing {protocol} modality gradients: {dict(counts)}")
            checks.append({"protocol": protocol, "answer_tokens": int(answer.numel()),
                           "reward": reward.total, "nonzero_gradient_tensors": dict(counts)})
            optimizer.zero_grad(set_to_none=True)
            del prompt, gpu_prompt, output, answer, now, old, ref
        import omni_modality_task as task
        correct = task.target(vt_row)
        incorrect = dict(correct)
        incorrect["is_conflict"] = "no" if correct["is_conflict"] == "yes" else "yes"
        answers = []
        rewards = []
        for value in (correct, incorrect):
            text = json.dumps(value, separators=(",", ":"))
            ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            answers.append(torch.tensor(ids, dtype=torch.long))
            rewards.append(score_answer(text, vt_row))
        adv, _ = advantages(rewards)
        update_norms = update_policy(vt_prompt, answers, adv)
        result = {"passed": True, "reference_device": ref_device, "checks": checks,
                  "grpo_synthetic_update_norms": update_norms,
                  "peak_policy_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                  "peak_reference_gib": round(torch.cuda.max_memory_allocated(int(ref_device[-1])) / 2**30, 3)}
        write_json(args.out / "smoke_test.json", result)
        print("SMOKE_TEST", json.dumps(result, ensure_ascii=False), flush=True)

    if args.smoke_test:
        smoke()
        return

    baseline = evaluate_now("sft_baseline", 0)
    best = {"source": str(args.adapter), "attempt": 0, "score": baseline["local_score_present"],
            "emotion_f1": baseline["emotion_macro_f1_present"],
            "aligned_specificity": baseline["aligned_specificity"]}
    write_json(args.out / "best.json", best)
    failed_evals = 0
    totals = Counter()
    started = time.monotonic()
    for attempt in range(1, args.attempts + 1):
        bucket, row = sampler.next()
        prompt = inputs.build(row, audio_first=bool(sampler.rng.randrange(2)), training=False)
        inputs.cached.clear()
        answers = [generate(prompt, args.temperature) for _ in range(args.initial_candidates)]
        rewards = [score_answer(tokenizer.decode(answer, skip_special_tokens=True), row)
                   for answer in answers]
        initial_status = group_status(rewards)
        if initial_status == "all_same_not_perfect":
            answers.extend(generate(prompt, args.rescue_temperature)
                           for _ in range(args.rescue_candidates))
            rewards = [score_answer(tokenizer.decode(answer, skip_special_tokens=True), row)
                       for answer in answers]
        final_status = group_status(rewards)
        adv, decision_varies = advantages(rewards,
            decision_scale=args.decision_advantage_scale,
            other_scale=args.other_advantage_scale)
        sampler.observe(bucket, row, rewards)
        totals["attempts"] += 1
        totals["candidates"] += len(answers)
        totals[final_status] += 1
        totals["decision_variant"] += int(decision_varies)
        totals["invalid_candidates"] += sum(not reward.valid for reward in rewards)
        append_json(args.out / "rollouts.jsonl", {
            "attempt": attempt, "sample_id": row["sample_id"], "bucket": bucket,
            "protocol": row["protocol"], "initial_status": initial_status,
            "final_status": final_status, "decision_varies": decision_varies,
            "rewards": [r.total for r in rewards],
            "fused_correct": [r.fused for r in rewards],
            "vision_correct": [r.vision for r in rewards],
            "second_correct": [r.second for r in rewards],
            "decision_correct": [r.decision for r in rewards],
            "valid": [r.valid for r in rewards], "consistent": [r.consistent for r in rewards],
            "answer_tokens": [int(answer.numel()) for answer in answers]})
        if final_status == "active":
            update_policy(prompt, answers, adv)
            totals["optimizer_steps"] += args.ppo_epochs
        if attempt % 10 == 0:
            print("PROGRESS", json.dumps({"attempt": attempt, **totals,
                "original_covered": len(sampler.original_seen),
                "elapsed_seconds": round(time.monotonic() - started)}, ensure_ascii=False), flush=True)
        if attempt % args.eval_every == 0 or attempt == args.attempts:
            metrics = evaluate_now(f"attempt_{attempt:04d}", attempt)
            eligible = (metrics["emotion_macro_f1_present"] >= baseline["emotion_macro_f1_present"] - args.f1_drop_guard
                        and metrics["aligned_specificity"] >= baseline["aligned_specificity"] - args.specificity_drop_guard)
            improved = eligible and metrics["local_score_present"] > best["score"]
            selected = {"attempt": attempt, "score": metrics["local_score_present"],
                        "emotion_f1": metrics["emotion_macro_f1_present"],
                        "conflict_recall": metrics["conflict_recall"],
                        "aligned_specificity": metrics["aligned_specificity"],
                        "eligible": eligible, "improved": improved,
                        "original_covered": len(sampler.original_seen), "rollouts": dict(totals)}
            write_json(args.out / f"attempt_{attempt:04d}_selection.json", selected)
            checkpoint = args.out / "checkpoints" / f"attempt_{attempt:04d}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            policy.save_pretrained(checkpoint, safe_serialization=True)
            processor.save_pretrained(checkpoint)
            if improved:
                best = {"source": str(checkpoint), **selected}
                write_json(args.out / "best.json", best)
                failed_evals = 0
            else:
                failed_evals += 1
            if failed_evals >= args.patience:
                print("EARLY_STOP", json.dumps({"attempt": attempt, "failed_evals": failed_evals}), flush=True)
                break
    write_json(args.out / "summary.json", {"baseline": baseline, "best": best,
              "final_attempt": attempt, "totals": dict(totals),
              "original_covered": len(sampler.original_seen),
              "elapsed_seconds": round(time.monotonic() - started)})


if __name__ == "__main__":
    main()
