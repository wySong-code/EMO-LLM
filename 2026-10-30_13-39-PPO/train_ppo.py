"""PPO on top of the published Combined-SFT adapter (Qwen2.5-Omni-3B Thinker).

Differences from the two GRPO rounds in this project, and why they exist:

* baseline     GRPO subtracts the mean reward of 4 candidates sampled from the SAME
               prompt (a Monte-Carlo group baseline). PPO instead learns V(s_t) with a
               scalar head on the policy backbone and subtracts it token by token.
* credit       GRPO assigns one scalar advantage to the whole sequence. PPO spreads the
               terminal reward over the response with GAE(gamma=1, lambda=0.95).
* batching     one PPO update mixes 4 DIFFERENT prompts with 1 rollout each, so the
               gradient is not tied to the difficulty of a single prompt.
* filtering    GRPO skips groups whose 4 candidates all score the same (79% of the first
               round, 68% of the second). PPO never discards rollouts -- every rollout
               becomes a value-regression target.

The critic is deliberately tiny: a LayerNorm + Linear(2048 -> 1) over the last hidden
state, zero-initialised, with the hidden state detached so the value loss can never
touch the LoRA weights. It adds 4,097 parameters and ~1 MiB of activations, which keeps
the two-GPU memory budget of the GRPO rounds intact (policy 18.2 GiB / reference 9.3 GiB).

Run --check-data and --check-env before --smoke-test. The smoke test performs real
VT/VA generation, a real PPO update and reports peak memory without touching the adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

from core import (StratifiedSampler, batch_statistics, gae, score_answer, token_rewards,
                  validate_assets, validate_dataset)

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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "training_dataset/Dataset")
    parser.add_argument("--base", type=Path, default=PROJECT / "base_model")
    parser.add_argument("--adapter", type=Path, default=PROJECT / "model_repo")
    parser.add_argument("--out", type=Path, default=HERE / "runs/pilot_01")
    parser.add_argument("--media-cache", type=Path, default=None,
                        help="content-addressed media cache; defaults to <out>/media_cache. "
                             "Point this outside <out> so the launch guard can rename <out> "
                             "without throwing away already decoded videos.")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--reference-device", choices=("auto", "cuda:0", "cuda:1"), default="auto")
    parser.add_argument("--attempts", type=int, default=600)
    parser.add_argument("--batch-prompts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--kl-beta", type=float, default=.02)
    parser.add_argument("--clip-eps", type=float, default=.2)
    parser.add_argument("--value-clip", type=float, default=.2)
    parser.add_argument("--value-coef", type=float, default=.5)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--f1-drop-guard", type=float, default=.01)
    parser.add_argument("--specificity-drop-guard", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.attempts < 1 or args.batch_prompts < 1 or args.temperature <= 0
            or not 0 < args.top_p <= 1 or args.max_new_tokens < 1 or args.lr <= 0
            or args.value_lr <= 0 or args.kl_beta < 0 or not 0 < args.clip_eps < 1
            or not 0 < args.value_clip < 1 or args.value_coef < 0 or args.ppo_epochs < 1
            or not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1
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
    # The peer code is the source of truth for video/audio preprocessing and metrics.
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
        model.eval()  # Dropout off: sampled old logprobs and values match update-time ones.
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

    def resolve_hidden_size(model) -> int:
        """The Thinker config nests the language width under text_config."""
        candidates = [getattr(model.config, "hidden_size", None)]
        for name in ("text_config", "thinker_config"):
            sub = getattr(model.config, name, None)
            if sub is None:
                continue
            candidates.append(getattr(sub, "hidden_size", None))
            inner = getattr(sub, "text_config", None)
            if inner is not None:
                candidates.append(getattr(inner, "hidden_size", None))
        for value in candidates:
            if value:
                return int(value)
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            raise RuntimeError("Cannot determine the language hidden size")
        return int(embeddings.weight.shape[1])

    hidden_size = resolve_hidden_size(policy)

    class ValueHead(torch.nn.Module):
        """Scalar critic on the frozen backbone output; the input is detached on purpose."""

        def __init__(self, width: int):
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.linear = torch.nn.Linear(width, 1)
            torch.nn.init.zeros_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.linear(self.norm(hidden.detach().float())).squeeze(-1)

    value_head = ValueHead(hidden_size).to("cuda:0", dtype=torch.float32)
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=args.lr, weight_decay=0.0)
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.value_lr, weight_decay=0.0)
    critic_parameters = int(sum(p.numel() for p in value_head.parameters()))
    cache_dir = args.media_cache if args.media_cache is not None else (args.out / "media_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    inputs = TenInputs(processor, args.dataset_root, cache_dir, train + val)

    def to_device(item: dict, device: str) -> dict:
        return {key: value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                if torch.is_tensor(value) else value for key, value in item.items()
                if not key.startswith("_")}

    def forward_terms(model, prompt: dict, answer: torch.Tensor, *, grad: bool,
                      need_values: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-token logprobs of ``answer`` (and critic values when requested).

        The logits slice starts at ``prefix - 1`` so position i predicts answer token i,
        which is exactly the alignment the value head must use for V(s_t).
        """
        device = str(next(model.parameters()).device)
        batch = to_device(prompt, device)
        answer_ids = answer.to(device).reshape(1, -1)
        prefix = batch["input_ids"].shape[1]
        batch["input_ids"] = torch.cat([batch["input_ids"], answer_ids], dim=1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(answer_ids)], dim=1)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch, use_cache=False, output_hidden_states=need_values)
            logits = output.logits[:, prefix - 1:prefix - 1 + answer_ids.shape[1]].float()
            selected = torch.log_softmax(logits, -1).gather(-1, answer_ids.unsqueeze(-1))
            logprob = selected.squeeze(0).squeeze(-1)
            values = None
            if need_values:
                if output.hidden_states is None:
                    raise RuntimeError("Backbone returned no hidden states; cannot run the critic")
                hidden = output.hidden_states[-1][:, prefix - 1:prefix - 1 + answer_ids.shape[1]]
                values = value_head(hidden).squeeze(0)
        if not grad:
            logprob = logprob.detach().cpu()
            values = None if values is None else values.detach().cpu()
        return logprob, values

    def rollout(prompt: dict, temperature: float) -> torch.Tensor:
        gpu_prompt = to_device(prompt, "cuda:0")
        with torch.inference_mode():
            output = policy.generate(**gpu_prompt, do_sample=True, temperature=temperature,
                top_p=args.top_p, max_new_tokens=args.max_new_tokens, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        answer = output[0, gpu_prompt["input_ids"].shape[1]:].detach().cpu().clone()
        if answer.numel() == 0:
            raise RuntimeError("Model returned zero answer tokens")
        return answer

    def ppo_update(samples: list[dict]) -> dict:
        """Clipped surrogate + k3 KL + clipped value regression, two optimisers."""
        stats = Counter()
        stats["policy_loss"] = 0.0
        stats["value_loss"] = 0.0
        stats["kl"] = 0.0
        stats["clip_fraction"] = 0.0
        stats["tokens"] = 0
        norms_policy: list[float] = []
        norms_value: list[float] = []
        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            epoch_policy = epoch_value = epoch_kl = 0.0
            epoch_clipped = epoch_tokens = 0
            for sample in samples:
                now, value = forward_terms(policy, sample["prompt"], sample["answer"],
                                           grad=True, need_values=True)
                old = sample["old_logprob"].to(now.device)
                ref = sample["ref_logprob"].to(now.device)
                adv = sample["advantages"].to(now.device)
                ret = sample["returns"].to(now.device)
                old_value = sample["old_values"].to(now.device)
                log_ratio = (now - old).clamp(-20, 20)
                ratio = log_ratio.exp()
                unclipped = ratio * adv
                clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * adv
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                delta = now - ref
                kl = (torch.exp((-delta).clamp(-20, 20)) + delta - 1).mean()
                value_clipped = old_value + (value - old_value).clamp(-args.value_clip, args.value_clip)
                value_loss = .5 * torch.maximum((value - ret) ** 2, (value_clipped - ret) ** 2).mean()
                loss = ((policy_loss + args.kl_beta * kl) / len(samples)
                        + args.value_coef * value_loss / len(samples))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                loss.backward()
                epoch_policy += float(policy_loss.detach())
                epoch_value += float(value_loss.detach())
                epoch_kl += float(kl.detach())
                epoch_clipped += int((torch.abs(log_ratio) > torch.log(torch.tensor(1 + args.clip_eps))).sum())
                epoch_tokens += int(now.numel())
            policy_norm = torch.nn.utils.clip_grad_norm_((p for _, p in trainable), 1.0,
                                                         error_if_nonfinite=True).item()
            value_norm = torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0,
                                                        error_if_nonfinite=True).item()
            if not value_norm > 0:
                raise RuntimeError("Critic received no gradient")
            optimizer.step()
            value_optimizer.step()
            norms_policy.append(round(policy_norm, 6))
            norms_value.append(round(value_norm, 6))
            stats["policy_loss"] += epoch_policy / len(samples)
            stats["value_loss"] += epoch_value / len(samples)
            stats["kl"] += epoch_kl / len(samples)
            stats["clip_fraction"] += epoch_clipped / max(epoch_tokens, 1)
            stats["tokens"] += epoch_tokens
        for key in ("policy_loss", "value_loss", "kl", "clip_fraction"):
            stats[key] = round(stats[key] / args.ppo_epochs, 6)
        stats["policy_grad_norms"] = norms_policy
        stats["value_grad_norms"] = norms_value
        return dict(stats)

    def evaluate_now(name: str, step: int) -> dict:
        result, _ = evaluate(policy, inputs, val, args.out, name, None, step,
                             batch_size=args.eval_batch_size)
        inputs.cached.clear()
        return result["all"]

    def smoke() -> None:
        checks = []
        vt_row = vt_prompt = None
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
            old, old_value = forward_terms(policy, prompt, answer, grad=False, need_values=True)
            ref, _ = forward_terms(reference, prompt, answer, grad=False, need_values=False)
            if not torch.isfinite(old).all() or not torch.isfinite(ref).all():
                raise FloatingPointError(f"Nonfinite {protocol} logprobs")
            if old_value is None or not torch.isfinite(old_value).all():
                raise FloatingPointError(f"Nonfinite {protocol} critic values")
            advantage, ret = gae(token_rewards(reward.total, int(answer.numel())),
                                 old_value.tolist(), gamma=args.gamma, lam=args.gae_lambda)
            checks.append({"protocol": protocol, "answer_tokens": int(answer.numel()),
                           "reward": reward.total,
                           "critic_min": round(float(old_value.min()), 5),
                           "critic_max": round(float(old_value.max()), 5),
                           "adv_first": round(advantage[0], 5), "adv_last": round(advantage[-1], 5),
                           "returns_last": round(ret[-1], 5)})
            del prompt, gpu_prompt, output, answer, old, ref, old_value
            inputs.cached.clear()
        # Real PPO update on a two-sample synthetic batch (gold answer vs flipped decision).
        import omni_modality_task as task
        correct = task.target(vt_row)
        wrong = dict(correct)
        wrong["is_conflict"] = "no" if correct["is_conflict"] == "yes" else "yes"
        samples = []
        for value in (correct, wrong):
            text = json.dumps(value, separators=(",", ":"))
            ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            answer = torch.tensor(ids, dtype=torch.long)
            reward = score_answer(text, vt_row)
            old, old_value = forward_terms(policy, vt_prompt, answer, grad=False, need_values=True)
            ref, _ = forward_terms(reference, vt_prompt, answer, grad=False, need_values=False)
            advantage, ret = gae(token_rewards(reward.total, int(answer.numel())),
                                 old_value.tolist(), gamma=args.gamma, lam=args.gae_lambda)
            samples.append({"prompt": vt_prompt, "answer": answer, "old_logprob": old,
                            "ref_logprob": ref, "old_values": old_value,
                            "advantages": torch.tensor(advantage, dtype=torch.float32),
                            "returns": torch.tensor(ret, dtype=torch.float32),
                            "reward": reward.total})
        stats = ppo_update(samples)
        result = {"passed": True, "reference_device": ref_device,
                  "critic_parameters": critic_parameters, "hidden_size": hidden_size,
                  "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                  "checks": checks, "ppo_update": stats,
                  "critic_after": [round(float(v), 5) for v in
                                   value_head(torch.zeros(1, 2, hidden_size, device="cuda:0")).tolist()[0]],
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
    totals["reward_sum"] = 0.0
    totals["advantage_abs_sum"] = 0.0
    totals["value_target_sum"] = 0.0
    started = time.monotonic()
    for attempt in range(1, args.attempts + 1):
        batch = sampler.batch(args.batch_prompts)
        samples: list[dict] = []
        record = {"attempt": attempt, "samples": []}
        for bucket, row in batch:
            prompt = inputs.build(row, audio_first=bool(sampler.rng.randrange(2)), training=False)
            inputs.cached.clear()
            answer = rollout(prompt, args.temperature)
            text = tokenizer.decode(answer, skip_special_tokens=True)
            reward = score_answer(text, row)
            old, old_value = forward_terms(policy, prompt, answer, grad=False, need_values=True)
            ref, _ = forward_terms(reference, prompt, answer, grad=False, need_values=False)
            advantage, ret = gae(token_rewards(reward.total, int(answer.numel())),
                                 old_value.tolist(), gamma=args.gamma, lam=args.gae_lambda)
            samples.append({"prompt": prompt, "answer": answer, "old_logprob": old,
                            "ref_logprob": ref, "old_values": old_value,
                            "advantages": torch.tensor(advantage, dtype=torch.float32),
                            "returns": torch.tensor(ret, dtype=torch.float32)})
            sampler.observe(bucket, row, reward)
            totals["reward_sum"] += reward.total
            totals["advantage_abs_sum"] += sum(abs(x) for x in advantage) / len(advantage)
            totals["value_target_sum"] += ret[-1] / len(advantage)
            totals["invalid_candidates"] += int(not reward.valid)
            totals["perfect"] += int(reward.perfect)
            totals["decision_correct"] += reward.decision
            totals["fused_correct"] += reward.fused
            record["samples"].append({
                "sample_id": row["sample_id"], "bucket": bucket, "protocol": row["protocol"],
                "reward": reward.total, "fused": reward.fused, "vision": reward.vision,
                "second": reward.second, "decision": reward.decision,
                "valid": reward.valid, "consistent": reward.consistent,
                "answer_tokens": int(answer.numel()),
                "old_value_head": round(float(old_value[-1]), 5),
                "advantage": batch_statistics(advantage), "answer": text[:400]})
        update = ppo_update(samples)
        totals["attempts"] += 1
        totals["rollouts"] += len(samples)
        totals["optimizer_steps"] += args.ppo_epochs
        record["update"] = update
        record["bucket_counts"] = dict(Counter(bucket for bucket, _ in batch))
        append_json(args.out / "rollouts.jsonl", record)
        if attempt % 10 == 0:
            n = totals["rollouts"]
            print("PROGRESS", json.dumps({
                "attempt": attempt, **{k: v for k, v in totals.items() if not isinstance(v, float)},
                "mean_reward": round(totals["reward_sum"] / n, 4),
                "perfect_rate": round(totals["perfect"] / n, 4),
                "kl": update["kl"], "clip_fraction": update["clip_fraction"],
                "value_loss": update["value_loss"],
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
                        "original_covered": len(sampler.original_seen),
                        "update": update, "rollouts": {k: v for k, v in totals.items()
                                                       if not isinstance(v, float)},
                        "mean_reward": round(totals["reward_sum"] / totals["rollouts"], 4)}
            write_json(args.out / f"attempt_{attempt:04d}_selection.json", selected)
            checkpoint = args.out / "checkpoints" / f"attempt_{attempt:04d}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            policy.save_pretrained(checkpoint, safe_serialization=True)
            processor.save_pretrained(checkpoint)
            torch.save({"hidden_size": hidden_size, "state_dict": value_head.state_dict()},
                       checkpoint / "value_head.pt")
            if improved:
                best = {"source": str(checkpoint), **selected}
                write_json(args.out / "best.json", best)
                failed_evals = 0
            else:
                failed_evals += 1
            if failed_evals >= args.patience:
                print("EARLY_STOP", json.dumps({"attempt": attempt, "failed_evals": failed_evals}), flush=True)
                break
    write_json(args.out / "summary.json", {
        "baseline": baseline, "best": best, "final_attempt": attempt,
        "totals": dict(totals), "critic_parameters": critic_parameters,
        "hidden_size": hidden_size, "gamma": args.gamma, "gae_lambda": args.gae_lambda,
        "mean_reward": round(totals["reward_sum"] / max(totals["rollouts"], 1), 4),
        "original_covered": len(sampler.original_seen),
        "elapsed_seconds": round(time.monotonic() - started)})


if __name__ == "__main__":
    main()
