#!/bin/bash
# Kaggle-format CSV from one checkpoint. Preprocessing, system prompt and the strict
# four-field parser come from code/sft, so the prompt and JSON handling are identical
# to training; only the weights differ between the SFT and GRPO rounds.
#
# usage: run_submit.sh <adapter_dir> <out_name>
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/miniconda3/envs/Emo-LLM/bin/python
ADAPTER=${1:?adapter dir required}
NAME=${2:?output name required}

export PYTHONPATH=$HOME/grpo_mm/envs/tf457:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CARD=$("$PY" "$ROOT/scripts/pick_cards.py" 1 2000) || { echo "no free card"; exit 3; }
export CUDA_VISIBLE_DEVICES=$CARD
mkdir -p "$ROOT/logs"
echo "[$(date '+%F %T')] card=$CARD adapter=$ADAPTER -> $NAME"

"$PY" -u "$ROOT/code/make_submission.py" \
  --base "$HOME/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B" \
  --adapter "$ADAPTER" \
  --code "$ROOT/code/sft" \
  --media-root "$HOME/MM-LLMs-Emotion-main/Dataset" \
  --test "$HOME/grpo_mm/test_release.jsonl" \
  --out "$ROOT/submissions/$NAME" \
  --batch-size 4 --max-new-tokens 192 \
  > "$ROOT/logs/submit_$NAME.log" 2>&1
echo "[$(date '+%F %T')] done"; tail -n 5 "$ROOT/logs/submit_$NAME.log"
