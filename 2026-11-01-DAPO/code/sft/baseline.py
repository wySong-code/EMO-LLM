"""Zero-shot validation with identical sampled media and prompts across models.

Run from the repository root. Labels are used ONLY after generation for scoring.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image
import soundfile as sf
import torch
from transformers import AutoProcessor

SYSTEM = """你是一个情感识别助手。根据提供的画面和文字或音频，判断说话者的情感。
情感类别只能是 positive（正向）、negative（负向）、neutral（中性）。
分别考虑画面和另一种信息的情感是否冲突。冲突不一定意味着无法判断；
只有证据不足、无法可靠判断时才表达不确定，并提出一个具体的澄清问题。
只输出一个 JSON 对象，不要解释或 Markdown。字段必须为：
pred_emotion: positive/negative/neutral；
is_conflict: yes/no；
is_uncertain: yes/no；
clarifying_question: 需要澄清时写一个简短中文问题，否则写字符串 none。"""
CLASSES = ["negative", "positive", "neutral"]
MODELS = {
    "gemma-e2b": "google/gemma-4-E2B-it",
    "gemma-e4b": "google/gemma-4-E4B-it",
    "qwen-omni-3b": "Qwen/Qwen2.5-Omni-3B",
    "qwen-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen35-08b": "Qwen/Qwen3.5-0.8B",
    "qwen35-2b": "Qwen/Qwen3.5-2B",
}


def prepare_media(video, cache, frames=8, max_side=448):
    """Uniform frame sampling over the ENTIRE clip; audio kept independently."""
    key = hashlib.sha256(video.read_bytes()).hexdigest()
    target = cache / f"{key}-f{frames}-s{max_side}"
    target.mkdir(parents=True, exist_ok=True)
    out_video, out_audio = target / "video.mp4", target / "audio.wav"
    meta_path = target / "meta.json"
    if not meta_path.exists():
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 25)
            # Dataset clips are short. Resize immediately to avoid keeping full HD frames.
            decoded = []
            for frame in container.decode(video=0):
                im = frame.to_image()
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                w, h = (max(2, x // 2 * 2) for x in im.size)
                decoded.append(np.array(im.resize((w, h))))
            if not decoded:
                raise ValueError(f"No frames decoded: {video}")
            duration = len(decoded) / fps
        indices = np.linspace(0, len(decoded) - 1, frames).round().astype(int)
        selected = [decoded[i] for i in indices]
        from fractions import Fraction
        with av.open(str(out_video), "w") as output:
            stream = output.add_stream("libx264", rate=Fraction(frames / duration).limit_denominator(10000))
            stream.width, stream.height = selected[0].shape[1], selected[0].shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18"}
            for arr in selected:
                for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        meta_path.write_text(json.dumps({
            "sha256": key, "original_frames": len(decoded), "duration": duration,
            "selected_indices": indices.tolist(), "max_side": max_side,
        }), encoding="utf-8")
    return out_video, out_audio, meta_path


def ensure_audio(original, audio):
    if not audio.exists():
        temp = audio.with_suffix(".tmp.wav")
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
            "-i", str(original), "-vn", "-ac", "1", "-ar", "16000", str(temp),
        ], check=True, capture_output=True)
        temp.replace(audio)


def extract_json(text):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and "pred_emotion" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def is_json_object(text):
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def summarize(records):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    def metric(rows):
        y = [r["gold_emotion"] for r in rows]
        def valid_value(row, key, allowed):
            value = row.get("prediction", {}).get(key)
            return value if isinstance(value, str) and value in allowed else "__invalid__"
        pred = [valid_value(r, "pred_emotion", CLASSES) for r in rows]
        cp = [valid_value(r, "is_conflict", ("yes", "no")) for r in rows]
        cy = [r["gold_conflict"] for r in rows]
        if not rows:
            return {"n": 0}
        pr, re, f, support = precision_recall_fscore_support(
            y, pred, labels=CLASSES, zero_division=0)
        cpr, cre, _, _ = precision_recall_fscore_support(
            cy, cp, labels=["yes"], zero_division=0)
        return {
            "n": len(rows),
            "emotion_accuracy": accuracy_score(y, pred),
            "emotion_macro_f1_fixed3": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
            "emotion_macro_f1_present": f1_score(y, pred, labels=sorted(set(y)), average="macro", zero_division=0),
            "emotion_per_class": {k: {"precision": float(p), "recall": float(r), "f1": float(v), "support": int(s)}
                                  for k, p, r, v, s in zip(CLASSES, pr, re, f, support)},
            "conflict_accuracy": accuracy_score(cy, cp),
            "conflict_f1": f1_score(cy, cp, labels=["yes"], average="macro", zero_division=0),
            "conflict_precision": float(cpr[0]),
            "conflict_recall": float(cre[0]),
            "json_valid_rate": sum(r.get("format_valid", False) for r in rows) / len(rows),
            "uncertain_rate": sum(r.get("prediction", {}).get("is_uncertain") == "yes" for r in rows) / len(rows),
            "error_count": sum("error" in r for r in rows),
        }
    return {
        "all": metric(records),
        "VT": metric([r for r in records if r["protocol"] == "VT"]),
        "VA": metric([r for r in records if r["protocol"] == "VA"]),
        "real": metric([r for r in records if not r["generated"]]),
        "generated": metric([r for r in records if r["generated"]]),
        "use_in_main": metric([r for r in records if r["use_in_main"]]),
        "median_seconds": statistics.median([r["seconds"] for r in records]) if records else None,
        "max_peak_allocated_gib": max((r.get("peak_allocated_gib", 0) for r in records), default=0),
        "note": "No uncertainty ground truth exists; uncertain_rate is descriptive, not accuracy. Invalid outputs count as errors. Fixed3 macro-F1 includes neutral with zero support.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", default="Dataset/val.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", help="Local snapshot of the same official checkpoint")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", choices=["all", "VT", "VA"], default="all")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--media-cache", default="outputs/media_cache")
    args = parser.parse_args()
    visual_only = args.model.startswith("qwen35")
    if visual_only and args.protocol != "VT":
        parser.error("Qwen3.5 has no audio encoder; explicitly select --protocol VT")
    torch.manual_seed(42)
    torch.set_num_threads(8)
    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.protocol != "all":
        rows = [row for row in rows if row["protocol"] == args.protocol]
    if args.smoke:
        # One real/generated aligned/conflict example for each protocol when present.
        groups = {}
        for row in rows:
            key = (row["protocol"], row["sample_type"], row.get("source_is_generated", False))
            groups.setdefault(key, row)
        rows = list(groups.values())
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    previous = []
    pred_path = out / "predictions.jsonl"
    if pred_path.exists():
        previous = [json.loads(s) for s in pred_path.read_text(encoding="utf-8").splitlines()]
    completed = {r["sample_id"] for r in previous}
    model_id = MODELS[args.model]
    attention = (
        {"text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"}
        if args.model.startswith("gemma") else "sdpa"
    )
    config = {
        **vars(args), "model_id": model_id, "system_prompt": SYSTEM,
        "dtype": "bfloat16", "do_sample": False, "seed": 42,
        "attention": attention,
        "chat_template_kwargs": {"enable_thinking": False} if visual_only else {},
        "expected_records": len(rows),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "versions": {x: importlib.metadata.version(x) for x in ["torch", "transformers", "accelerate", "av"]},
        "gpu": torch.cuda.get_device_name(0),
    }
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Resume configuration differs. Use a new output directory.")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "baseline_source.py").write_bytes(Path(__file__).read_bytes())
    print("Loading", model_id, flush=True)
    model_path = args.model_path or model_id
    processor = AutoProcessor.from_pretrained(model_path)
    if args.model.startswith("gemma"):
        from transformers import Gemma4ForConditionalGeneration
        cls = Gemma4ForConditionalGeneration
    elif visual_only:
        from transformers import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    else:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        cls = Qwen2_5OmniThinkerForConditionalGeneration
    model = cls.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=attention)
    model.eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = list(eos_ids) if isinstance(eos_ids, (list, tuple)) else ([] if eos_ids is None else [eos_ids])
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.append(processor.tokenizer.eos_token_id)
    eos_ids = sorted(set(eos_ids))
    config["model_commit"] = getattr(model.config, "_commit_hash", None)
    # Commit is a runtime observation, separate from the stable resume configuration.
    (out / "model_revision.json").write_text(json.dumps({
        "commit": config["model_commit"], "eos_token_ids": eos_ids,
        "loaded_parameters": sum(p.numel() for p in model.parameters()),
        "model_class": type(model).__name__, "processor_class": type(processor).__name__,
    }), encoding="utf-8")
    cache = Path(args.media_cache)
    cache.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset).parent
    results = previous
    with pred_path.open("a", encoding="utf-8") as fp:
        for i, row in enumerate(rows):
            if row["sample_id"] in completed:
                continue
            start = time.monotonic()
            result = {
                "sample_id": row["sample_id"], "protocol": row["protocol"],
                "gold_emotion": row["views"]["M12"]["label"],
                "gold_conflict": "yes" if row["sample_type"] == "Conflict" else "no",
                "generated": row.get("source_is_generated", False),
                "use_in_main": row.get("use_in_main", True),
            }
            try:
                original = dataset_root / row["video"]
                video, audio, meta = prepare_media(original, cache, args.frames, args.max_side)
                content = [{"type": "video", "path": str(video.resolve())}]
                if row["protocol"] == "VT":
                    prompt = "任务：结合视频画面和以下文字判断情感。\n文字：" + row["text_content"]
                elif row["protocol"] == "VA":
                    prompt = "任务：结合视频画面和音频判断情感。"
                else:
                    raise ValueError("Unknown protocol")
                content.append({"type": "text", "text": prompt})
                if row["protocol"] == "VA":
                    ensure_audio(original, audio)
                    content.append({"type": "audio", "path": str(audio.resolve())})
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
                video_kwargs = {"do_sample_frames": False}
                if args.model.startswith("qwen-omni"):
                    # Qwen constructs time positions from this FPS, independently of
                    # the decoder's metadata. Our cached clip retains original duration.
                    video_kwargs.update({
                        "fps": args.frames / json.loads(meta.read_text())["duration"],
                        "use_audio_in_video": False,
                    })
                # Decode explicitly: no dependency on changing torchcodec defaults,
                # and both models receive exactly the same frames and audio.
                with av.open(str(video)) as decoded_video:
                    frame_array = np.stack([f.to_ndarray(format="rgb24") for f in decoded_video.decode(video=0)])
                duration = json.loads(meta.read_text())["duration"]
                from transformers.video_utils import VideoMetadata
                video_kwargs["video_metadata"] = [VideoMetadata(
                    total_num_frames=len(frame_array), fps=len(frame_array)/duration,
                    duration=duration, frames_indices=list(range(len(frame_array))),
                )]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **config["chat_template_kwargs"],
                )
                result["rendered_prompt_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                audio_values = None
                if row["protocol"] == "VA":
                    wave, rate = sf.read(audio, dtype="float32")
                    assert rate == 16000 and wave.ndim == 1
                    audio_values = [wave]
                processor_kwargs = dict(
                    text=[text], videos=[frame_array],
                    return_tensors="pt", videos_kwargs=video_kwargs,
                )
                if not visual_only:
                    processor_kwargs.update(audio=audio_values, audio_kwargs={"sampling_rate": 16000})
                inputs = processor(**processor_kwargs).to(model.device, dtype=torch.bfloat16)
                result["input_tokens"] = inputs["input_ids"].shape[-1]
                result["input_keys"] = sorted(inputs.keys())
                if row["protocol"] == "VA" and not any("audio" in k or k == "input_features" for k in inputs):
                    raise RuntimeError("Audio missing from VA model inputs")
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                ids = generated[0, inputs["input_ids"].shape[-1]:]
                text = processor.decode(ids, skip_special_tokens=True)
                prediction = extract_json(text)
                result.update({
                    "raw_output": text, "prediction": prediction,
                    "output_tokens": len(ids),
                    "format_valid": is_json_object(text) and (
                        prediction.get("pred_emotion") in CLASSES and
                        prediction.get("is_conflict") in ("yes", "no") and
                        prediction.get("is_uncertain") in ("yes", "no") and
                        isinstance(prediction.get("clarifying_question"), str)
                    ),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "media": json.loads(meta.read_text()),
                })
                del inputs, generated
            except Exception:
                result["error"] = traceback.format_exc()
                print(result["error"], flush=True)
                torch.cuda.empty_cache()
            result["seconds"] = round(time.monotonic() - start, 3)
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            fp.flush()
            results.append(result)
            (out / "metrics.json").write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{i+1}/{len(rows)} {row['sample_id']} {result.get('prediction',{})} {result['seconds']}s", flush=True)
            if len(results) >= 3 and all("error" in r for r in results[-3:]):
                raise RuntimeError("Three consecutive errors; stopping to avoid wasting GPU time")
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
