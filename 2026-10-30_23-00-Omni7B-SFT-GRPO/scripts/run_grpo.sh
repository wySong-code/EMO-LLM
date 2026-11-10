#!/bin/bash
# Qwen2.5-Omni-7B GRPO, recipe byte-for-byte the one used for the 3B step-50 run
# (2026-10-29_14-21-GRPO): 300 steps, 8 candidates, 2 PPO epochs, lr 2e-6,
# KL 0.02, clip 0.2, temperature 1.1, top_p 0.95, max_new_tokens 128,
# original_conflict_prob 0.67, advantage scale 1.5, eval every 50, guard 0.01.
#
# usage: run_grpo.sh [sft_checkpoint]
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/miniconda3/envs/Emo-LLM/bin/python
BASE=$HOME/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B
SHARED_CACHE=$HOME/MM-LLMs-Emotion-main/outputs/media_cache
CACHE=$ROOT/grpo_run/media_cache
OUT=$ROOT/grpo_run/pilot_01

export PYTHONPATH=$HOME/grpo_mm/envs/tf457:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$ROOT/logs" "$CACHE"
cp -al "$SHARED_CACHE"/. "$CACHE"/ 2>/dev/null || true

ADAPTER=${1:-}
if [ -z "$ADAPTER" ]; then
  ADAPTER=$("$PY" -c "import json,sys;print(json.load(open('$ROOT/sft_run/train/best_checkpoint.json'))['path'])")
fi
echo "[$(date '+%F %T')] adapter=$ADAPTER"

CARDS=$("$PY" "$ROOT/scripts/pick_cards.py" 2 800) || { echo "no free card pair"; exit 3; }
export CUDA_VISIBLE_DEVICES=$CARDS
echo "[$(date '+%F %T')] cards=$CARDS (policy=0 reference=1)"

if [ "$(echo "$CARDS" | cut -d, -f1)" = "$(echo "$CARDS" | cut -d, -f2)" ]; then
  echo "policy and reference must be on different cards"; exit 3
fi

"$PY" -u "$ROOT/code/grpo/train_grpo.py" \
  --base "$BASE" --adapter "$ADAPTER" --code "$ROOT/code/sft" \
  --dataset-root "$HOME/grpo_mm/MM-Emotion-Combined-SFT-20261026/training_dataset/Dataset" \
  --out "$OUT" --media-cache "$CACHE" \
  --policy-device 0 --reference-device 1 \
  --steps 300 --candidates 8 --ppo-epochs 2 --lr 2e-6 --kl-beta 0.02 --clip-eps 0.2 \
  --temperature 1.1 --top-p 0.95 --max-new-tokens 128 \
  --original-conflict-prob 0.67 --conflict-advantage-scale 1.5 \
  --eval-every 50 --eval-batch-size 4 --f1-drop-guard 0.01 --seed 42 \
  > "$ROOT/logs/grpo_train.log" 2>&1
echo "[$(date '+%F %T')] grpo finished"
cat "$OUT/best.json"
