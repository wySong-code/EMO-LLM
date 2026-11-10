"""GRPO continuation of a Combined-SFT adapter (Qwen2.5-Omni-7B variant).

Identical recipe to the 3B `2026-10-29_14-21-GRPO` step-50 run; only the model, the
cards and two memory adaptations change (see 策略、实现与运行说明.md).
Use --check-data for a CPU-only preflight. No validation labels enter training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

from reward import advantages, score

PROJECT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preflight(dataset_root: Path) -> tuple[list[dict], list[dict], dict]:
    train, val = (read_jsonl(dataset_root / name) for name in ("train.jsonl", "val.jsonl"))
    ids, val_ids = {r["sample_id"] for r in train}, {r["sample_id"] for r in val}
    if len(ids) != len(train) or len(val_ids) != len(val) or ids & val_ids:
        raise ValueError("duplicate or overlapping sample_id")
    if len(train) != 3352 or len(val) != 666:
        raise ValueError(f"unexpected split size: {len(train)} / {len(val)}")
    val_videos = {r["video"] for r in val}
    if any(r["video"] in val_videos for r in train):
        raise ValueError("train/validation video path overlap")
    for row in train + val:
        if row["protocol"] not in {"VT", "VA"} or row["sample_type"] not in {"Conflict", "Aligned"}:
            raise ValueError(f"invalid protocol or sample_type: {row['sample_id']}")
        m1, m2 = (row["views"][m]["label"] for m in ("M1", "M2"))
        if (m1 != m2) != (row["sample_type"] == "Conflict"):
            raise ValueError(f"label disagreement: {row['sample_id']}")
        if not (dataset_root / row["video"]).is_file():
            raise FileNotFoundError(dataset_root / row["video"])
    stats = {"train": len(train), "val": len(val),
             "train_conflict": sum(r["sample_type"] == "Conflict" for r in train),
             "val_conflict": sum(r["sample_type"] == "Conflict" for r in val),
             "train_original_conflict": sum(r["sample_type"] == "Conflict" and not r["source_is_generated"] for r in train),
             "train_generated_conflict": sum(r["sample_type"] == "Conflict" and r["source_is_generated"] for r in train)}
    return train, val, stats


def choose_pair(train: list[dict], rng: random.Random, original_probability: float) -> list[dict]:
    originals = [r for r in train if r["sample_type"] == "Conflict" and not r["source_is_generated"]]
    generated = [r for r in train if r["sample_type"] == "Conflict" and r["source_is_generated"]]
    aligned = [r for r in train if r["sample_type"] == "Aligned"]
    if not originals or not generated or not aligned:
        raise ValueError("all three training buckets must be nonempty")
    conflict = rng.choice(originals if rng.random() < original_probability else generated)
    return [conflict, rng.choice(aligned)]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_status() -> dict:
    modules = ("torch", "transformers", "peft", "accelerate", "av", "numpy",
               "soundfile", "imageio_ffmpeg", "PIL", "safetensors")
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version.split()[0], "modules": available,
              "versions": versions, "cuda": False, "gpu": None}
    if available["torch"]:
        try:
            import torch
            result["cuda"] = torch.cuda.is_available()
            if result["cuda"]:
                result["gpu"] = torch.cuda.get_device_name(0)
        except Exception as error:
            result["torch_import_error"] = str(error)
    result["ready"] = all(available.values()) and result["cuda"]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=PROJECT / "training_dataset/Dataset")
    ap.add_argument("--base", type=Path, default=Path(os.path.expanduser("~/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B")))
    ap.add_argument("--adapter", type=Path, default=PROJECT / "sft_run/train/adapter")
    ap.add_argument("--code", type=Path, default=PROJECT / "code/sft",
                    help="SFT code tree supplying preprocessing/evaluation helpers")
    ap.add_argument("--out", type=Path, default=PROJECT / "grpo_run/pilot_01")
    ap.add_argument("--media-cache", type=Path, default=None,
                    help="reuse decoded clips; defaults to <out>/media_cache")
    ap.add_argument("--policy-device", type=int, default=0)
    ap.add_argument("--reference-device", type=int, default=1,
                    help="policy and reference are too large to share one device")
    ap.add_argument("--check-data", action="store_true", help="CPU-only split and media preflight")
    ap.add_argument("--check-env", action="store_true", help="report imports, versions and CUDA without loading models")
    ap.add_argument("--smoke-test", action="store_true", help="GPU VT/VA generation and backward check; no optimizer step")
    ap.add_argument("--steps", type=int, default=300, help="rollout batches; each has two prompts")
    ap.add_argument("--candidates", type=int, default=8, help="samples per prompt")
    ap.add_argument("--ppo-epochs", type=int, default=2, help="updates per fixed rollout batch")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--original-conflict-prob", type=float, default=0.67)
    ap.add_argument("--conflict-advantage-scale", type=float, default=1.5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--f1-drop-guard", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if (not 0 < args.original_conflict_prob < 1 or args.candidates < 2 or args.steps < 1
            or args.ppo_epochs < 1 or args.eval_every < 1 or args.eval_batch_size < 1 or args.lr <= 0
            or args.kl_beta < 0 or not 0 < args.clip_eps < 1
            or args.temperature <= 0 or not 0 < args.top_p <= 1
            or args.max_new_tokens < 1 or args.conflict_advantage_scale <= 0
            or args.f1_drop_guard < 0):
        ap.error("invalid training or sampling parameter")
    train, val, stats = preflight(args.dataset_root)
    print("DATA_CHECK", json.dumps(stats, ensure_ascii=False), flush=True)
    if args.check_data:
        return
    if args.check_env:
        status = environment_status()
        print("ENV_CHECK", json.dumps(status, ensure_ascii=False), flush=True)
        if not status["ready"]:
            raise SystemExit(2)
        return

    # Heavy imports occur only after the data check; use the published SFT code
    # for exactly the same video/audio preprocessing and validation metric.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.code))
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration, set_seed
    from overfit_omni_modality import TenInputs
    from evaluate_omni_batched import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("This training command requires a CUDA GPU; --check-data works on CPU")
    args.out.mkdir(parents=True, exist_ok=False)
    recorded = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_json(args.out / "config.json", {**recorded, "dataset_root": str(args.dataset_root),
                                           "base": str(args.base), "adapter": str(args.adapter),
                                           "code": str(args.code), "out": str(args.out),
                                           "preflight": stats})
    rng = random.Random(args.seed)
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.adapter)
    tok = processor.tokenizer
    if tok.eos_token_id is None:
        raise RuntimeError("EOS token is required")
    policy_where, reference_where = f"cuda:{args.policy_device}", f"cuda:{args.reference_device}"

    def load(device: str, trainable: bool):
        base, loading = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": device}, low_cpu_mem_usage=True, output_loading_info=True)
        write_json(args.out / ("policy_loading.json" if trainable else "reference_loading.json"), loading)
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise RuntimeError("Incomplete base checkpoint load; inspect loading JSON")
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=trainable)
        model.eval()  # Deterministic dropout-off logprobs, including during gradient forward.
        model.config.use_cache = False
        # 7B activations do not fit next to 17 GiB of weights, so the gradient forward
        # recomputes layers instead of storing them (see grad_pass below).
        if trainable:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not trainable:
            for p in model.parameters():
                p.requires_grad_(False)
        return model

    policy = load(policy_where, True)
    reference = load(reference_where, False)
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable)
    write_json(args.out / "trainable_parameters.json",
               {"count": trainable_count, "parameters": len(trainable),
                "all_lora": all("lora_" in name for name, _ in trainable)})
    if trainable_count <= 0 or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Loaded policy does not expose LoRA parameters")
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=args.lr)
    inputs = TenInputs(processor, args.dataset_root,
                       args.media_cache or (args.out / "media_cache"), train + val)

    def on_gpu(item: dict, device: str = policy_where) -> dict:
        return {k: v.to(device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                if torch.is_tensor(v) else v for k, v in item.items() if not k.startswith("_")}

    def logprobs(model, prompt: dict, answer: torch.Tensor, grad: bool,
                 device: str = policy_where) -> torch.Tensor:
        """Token log-probabilities of `answer` given `prompt`.

        The 3B run converted the full [len, vocab] logits block to float32 and ran one
        log_softmax over it. 7B cannot afford the extra ~4 GiB, so the same arithmetic is
        done in row chunks: bf16 -> float32 is exact, so the result is bit-identical.
        """
        batch = on_gpu(prompt, device)
        ids = answer.to(device).reshape(1, -1)
        n = batch["input_ids"].shape[1]
        batch["input_ids"] = torch.cat([batch["input_ids"], ids], dim=1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(ids)], dim=1)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch, use_cache=False).logits[:, n - 1:n - 1 + ids.shape[1], :]
            pieces = []
            for start in range(0, logits.shape[1], 512):
                chunk = logits[:, start:start + 512, :].float()
                pieces.append(torch.log_softmax(chunk, dim=-1).gather(
                    -1, ids[:, start:start + 512].unsqueeze(-1)).squeeze(0).squeeze(-1))
            lp = torch.cat(pieces, dim=0)
        return lp if grad else lp.detach().cpu()

    @contextmanager
    def grad_pass(model):
        """Checkpointing only fires in train mode, but LoRA dropout must stay off."""
        saved = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        previous_cache = model.config.use_cache
        for module, _ in saved:
            module.p = 0.0
        model.train()
        model.config.use_cache = False
        try:
            yield
        finally:
            for module, probability in saved:
                module.p = probability
            model.eval()
            model.config.use_cache = previous_cache

    def evaluate_now(name: str, step: int):
        metrics, _ = evaluate(policy, inputs, val, args.out, name, None, step,
                              batch_size=args.eval_batch_size)
        inputs.cached.clear()
        return metrics

    if args.smoke_test:
        checks = []
        for protocol in ("VT", "VA"):
            row = next(r for r in train if r["protocol"] == protocol)
            prompt = inputs.build(row, training=False)
            inputs.cached.clear()
            gpu_prompt = on_gpu(prompt)
            with torch.inference_mode():
                generated = policy.generate(**gpu_prompt, do_sample=False,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            answer = generated[0, gpu_prompt["input_ids"].shape[1]:].detach().cpu().clone()
            if not answer.numel():
                raise RuntimeError(f"Empty generation for {protocol}")
            reference_lp = logprobs(reference, prompt, answer, False, reference_where)
            optimizer.zero_grad(set_to_none=True)
            with grad_pass(policy):
                policy_lp = logprobs(policy, prompt, answer, True)
                if not torch.isfinite(policy_lp).all() or not torch.isfinite(reference_lp).all():
                    raise FloatingPointError(f"Non-finite log probability for {protocol}")
                (-policy_lp.mean()).backward()
            gradients = [p.grad for _, p in trainable if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise FloatingPointError(f"Missing or non-finite LoRA gradient for {protocol}")
            if not any(torch.count_nonzero(g).item() for g in gradients):
                raise RuntimeError(f"Zero LoRA gradient for {protocol}")
            group_grads = {"language": 0, "vision": 0, "audio": 0}
            for name, parameter in trainable:
                if parameter.grad is None or not torch.count_nonzero(parameter.grad).item():
                    continue
                if ".visual.blocks." in name:
                    group_grads["vision"] += 1
                elif ".audio_tower.layers." in name:
                    group_grads["audio"] += 1
                elif ".model.layers." in name:
                    group_grads["language"] += 1
            expected = {"language", "vision"} | ({"audio"} if protocol == "VA" else set())
            if any(group_grads[group] == 0 for group in expected):
                raise RuntimeError(f"Missing expected modality gradient for {protocol}: {group_grads}")
            earned = score(tok.decode(answer, skip_special_tokens=True), row)
            if not earned.valid:
                raise RuntimeError(f"Invalid four-field JSON in {protocol} smoke test")
            checks.append({"protocol": protocol, "sample_id": row["sample_id"],
                           "answer_tokens": int(answer.numel()),
                           "reward": earned.total, "gradient_tensors": len(gradients),
                           "gradient_groups": group_grads})
            optimizer.zero_grad(set_to_none=True)
            del gpu_prompt, generated, answer, reference_lp, policy_lp
        result = {"passed": True, "checks": checks,
                  "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        write_json(args.out / "smoke_test.json", result)
        print("SMOKE_TEST", json.dumps(result, ensure_ascii=False), flush=True)
        return

    baseline = evaluate_now("sft_baseline", 0)["all"]
    floor = baseline["emotion_macro_f1_present"] - args.f1_drop_guard
    best = {"source": "sft_baseline", "score": baseline["local_score_present"],
            "emotion_f1": baseline["emotion_macro_f1_present"], "step": 0}
    write_json(args.out / "best.json", best)
    rewards_file = args.out / "rollouts.jsonl"
    for step in range(1, args.steps + 1):
        pair = choose_pair(train, rng, args.original_conflict_prob)
        rollouts = []
        reasons = Counter()
        for row in pair:
            prompt = inputs.build(row, audio_first=bool(rng.randrange(2)), training=False)
            inputs.cached.clear()
            gpu_prompt = on_gpu(prompt)
            candidates = []
            for _ in range(args.candidates):
                with torch.inference_mode():
                    generated = policy.generate(**gpu_prompt, do_sample=True,
                        temperature=args.temperature, top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
                answer = generated[0, gpu_prompt["input_ids"].shape[1]:].detach().cpu().clone()
                text = tok.decode(answer, skip_special_tokens=True)
                reward = score(text, row)
                candidates.append({"answer": answer, "reward": reward})
                del generated
            adv, reason = advantages([c["reward"] for c in candidates],
                gold_conflict=row["sample_type"] == "Conflict", priority=args.conflict_advantage_scale)
            reasons[reason] += 1
            for cand, value in zip(candidates, adv):
                cand["advantage"] = value
                cand["prompt"] = prompt
                if abs(value) > 1e-8:
                    with torch.no_grad():
                        cand["old"] = logprobs(policy, prompt, cand["answer"], False)
                        cand["ref"] = logprobs(reference, prompt, cand["answer"], False, reference_where)
                rollouts.append(cand)
            with rewards_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "sample_id": row["sample_id"],
                    "source": "generated" if row["source_is_generated"] else "original",
                    "gold_conflict": row["sample_type"], "bucket": reason,
                    "rewards": [round(c["reward"].total, 5) for c in candidates],
                    "decision_correct": [c["reward"].decision for c in candidates]}, ensure_ascii=False) + "\n")
        updates = 0
        if any(abs(c["advantage"]) > 1e-8 for c in rollouts):
            active_count = sum(abs(c["advantage"]) > 1e-8 for c in rollouts)
            with grad_pass(policy):
                for _ in range(args.ppo_epochs):
                    optimizer.zero_grad(set_to_none=True)
                    for cand in rollouts:
                        if abs(cand["advantage"]) < 1e-8:
                            continue
                        now = logprobs(policy, cand["prompt"], cand["answer"], True)
                        old, ref = cand["old"].to(now.device), cand["ref"].to(now.device)
                        ratio = (now - old).clamp(-20, 20).exp()
                        a = cand["advantage"]
                        unclipped = ratio * a
                        clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * a
                        policy_loss = -torch.minimum(unclipped, clipped).mean()
                        d = now - ref
                        kl = (torch.exp((-d).clamp(-20, 20)) + d - 1).mean()
                        loss = (policy_loss + args.kl_beta * kl) / active_count
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_((p for _, p in trainable), 1.0,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
        print("STEP", step, "updates", updates, "buckets", dict(reasons), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_now(f"step_{step:04d}", step)["all"]
            current = {"step": step, "score": metrics["local_score_present"],
                       "emotion_f1": metrics["emotion_macro_f1_present"],
                       "conflict_recall": metrics["conflict_recall"],
                       "aligned_specificity": metrics["aligned_specificity"],
                       "floor": floor, "eligible": metrics["emotion_macro_f1_present"] >= floor}
            write_json(args.out / f"step_{step:04d}_selection.json", current)
            checkpoint = args.out / "checkpoints" / f"step_{step:04d}"
            checkpoint.mkdir(parents=True)
            policy.save_pretrained(checkpoint, safe_serialization=True)
            processor.save_pretrained(checkpoint)
            if current["eligible"] and current["score"] > best["score"]:
                best = {"source": str(checkpoint), **current}
                write_json(args.out / "best.json", best)


if __name__ == "__main__":
    main()
