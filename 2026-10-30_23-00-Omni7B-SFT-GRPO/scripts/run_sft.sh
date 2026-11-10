#!/bin/bash
# Qwen2.5-Omni-7B combined-SFT, same data/mixture/hyper-parameters as the published
# 3B run (3 epochs, effective batch 16, LoRA r16 a32 d0.05, lr 1e-4 cosine, warmup 3%).
# Only the model and the per-card micro-batch change; the effective batch stays 16.
#
# usage: run_sft.sh [micro_batch] [eval_batch]
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/miniconda3/envs/Emo-LLM/bin/python
MB=${1:-4}
EVALB=${2:-16}
BASE=$HOME/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B
DATASETSRC=$HOME/grpo_mm/MM-Emotion-Combined-SFT-20261026/training_dataset/Dataset
VIDEOSRC=$HOME/MM-LLMs-Emotion-main/Dataset/videos
POOLSRC=$HOME/grpo_mm/MM-Emotion-Combined-SFT-20261026/training_dataset
SHARED_CACHE=$HOME/MM-LLMs-Emotion-main/outputs/media_cache
CACHE=$ROOT/sft_run/media_cache

# Two path views the training script insists on, both content-identical to upstream:
#  * --data must hold train_pool.jsonl (mixture) together with the two manifests, but
#    upstream keeps them in data/ and manifests/ — join them by symlink;
#  * validate() asserts every video resolves *inside* --dataset, and the shipped
#    Dataset/videos is a symlink pointing outside — so give the run a real directory
#    whose videos/ is a hard-link tree (same inodes, no extra bytes) and whose jsonl
#    files are the same filtered 3352/666 rows.
POOL=$ROOT/sft_run/data_view
mkdir -p "$POOL"
ln -sfn "$POOLSRC/data/train_pool.jsonl" "$POOL/train_pool.jsonl"
ln -sfn "$POOLSRC/manifests/train_manifest.jsonl" "$POOL/train_manifest.jsonl"
ln -sfn "$POOLSRC/manifests/val_manifest.jsonl" "$POOL/val_manifest.jsonl"

DATASET=$ROOT/sft_run/dataset_view
mkdir -p "$DATASET"
ln -sfn "$DATASETSRC/train.jsonl" "$DATASET/train.jsonl"
ln -sfn "$DATASETSRC/val.jsonl" "$DATASET/val.jsonl"
[ -d "$DATASET/videos" ] || cp -al "$VIDEOSRC" "$DATASET/videos"

export PYTHONPATH=$HOME/grpo_mm/envs/tf457:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled

mkdir -p "$ROOT/logs" "$CACHE"
# Reuse every already-decoded clip; prepare_media is content addressed, so hard
# links are safe and new clips land only in this run's cache.
cp -al "$SHARED_CACHE"/. "$CACHE"/ 2>/dev/null || true
echo "[$(date '+%F %T')] cache entries: $(ls "$CACHE" | wc -l)"

CARD=$("$PY" "$ROOT/scripts/pick_cards.py" 1 800) || { echo "no free card"; exit 3; }
export CUDA_VISIBLE_DEVICES=$CARD
echo "[$(date '+%F %T')] card=$CARD micro_batch=$MB eval_batch=$EVALB"

if [ ! -f "$ROOT/sft_run/preflight/summary.json" ]; then
  "$PY" -u "$ROOT/code/sft/train_omni_combined_sft.py" \
    --mode pilot --model "$BASE" --model-id Qwen/Qwen2.5-Omni-7B \
    --dataset "$DATASET" --data "$POOL" --cache "$CACHE" \
    --output "$ROOT/sft_run/preflight" --micro-batch "$MB" \
    >> "$ROOT/logs/sft_preflight.log" 2>&1
fi
grep -q '"passed": true' "$ROOT/sft_run/preflight/summary.json" || {
  echo "preflight failed, see logs/sft_preflight.log"; exit 4; }

if [ ! -d "$ROOT/sft_run/train" ]; then
  "$PY" -u "$ROOT/code/sft/train_omni_combined_sft.py" \
    --mode train --model "$BASE" --model-id Qwen/Qwen2.5-Omni-7B \
    --dataset "$DATASET" --data "$POOL" --cache "$CACHE" \
    --output "$ROOT/sft_run/train" --pilot "$ROOT/sft_run/preflight" \
    --epochs 3 --eval-batch "$EVALB" --micro-batch "$MB" \
    >> "$ROOT/logs/sft_train.log" 2>&1
fi
echo "[$(date '+%F %T')] sft finished"
tail -n 3 "$ROOT/sft_run/train/status.json"
