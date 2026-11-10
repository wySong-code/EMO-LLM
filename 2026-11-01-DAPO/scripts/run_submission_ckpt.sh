#!/usr/bin/env bash
# Export a Kaggle submission from an explicit checkpoint.
#
# run_submission.sh follows runs/<arm>/best.json, which is selected by the training
# metric (conflict recall). The official leaderboard metric uses conflict accuracy,
# so the official optimum sits at a different step (step_0050 for both DAPO arms).
# This script lets the caller name the checkpoint directly instead.
#
#   ./run_submission_ckpt.sh <arm> <step> <name> [gpu]
#   ./run_submission_ckpt.sh dapo7b 50 dapo7b_step50 1
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PARENT=$(cd "$ROOT/.." && pwd)
PY=${DAPO_PYTHON:-$HOME/miniconda3/envs/Emo-LLM/bin/python}
BASE=${DAPO_BASE:-$HOME/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B}
TEST=${DAPO_TEST:-$HOME/grpo_mm/test_release.jsonl}
MEDIA=${DAPO_TEST_MEDIA:-$HOME/MM-LLMs-Emotion-main/Dataset}

ARM=${1:?arm}
STEP=${2:?step}
NAME=${3:-${ARM}_step${STEP}}
CARD=${4:-}

ADAPTER=$ROOT/runs/$ARM/checkpoints/step_$(printf '%04d' "$STEP")
DEST=$ROOT/submissions/$NAME

for required in "$PY" "$BASE/config.json" "$TEST"; do
    if [[ ! -e "$required" ]]; then echo "Missing $required" >&2; exit 2; fi
done
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then echo "Adapter missing: $ADAPTER" >&2; exit 2; fi
if [[ -e "$DEST" ]]; then echo "Submission directory exists: $DEST" >&2; exit 2; fi

export PYTHONPATH="$HOME/grpo_mm/envs/tf457:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z "$CARD" ]]; then
    CARD=$("$PY" "$ROOT/scripts/pick_cards.py" 1 800)
fi
export CUDA_VISIBLE_DEVICES="$CARD"
mkdir -p "$ROOT/logs" "$ROOT/submissions"
echo "START $(date -Is) arm=$ARM step=$STEP card=$CARD dest=$DEST"
"$PY" -u "$ROOT/code/make_submission.py" \
    --base "$BASE" --adapter "$ADAPTER" --code "$ROOT/code/sft" \
    --media-root "$MEDIA" --test "$TEST" --out "$DEST" \
    --batch-size 4 --max-new-tokens 192 \
    > "$ROOT/logs/submission_$NAME.log" 2>&1
echo "DONE $(date -Is) $DEST/submission.csv"
