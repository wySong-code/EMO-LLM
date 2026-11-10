#!/usr/bin/env bash
# Export a Kaggle-format CSV from the selected DAPO checkpoint (or SFT if no gain).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PARENT=$(cd "$ROOT/.." && pwd)
PY=${DAPO_PYTHON:-$HOME/miniconda3/envs/Emo-LLM/bin/python}
BASE=${DAPO_BASE:-$HOME/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B}
SFT=$PARENT/2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/adapter_best
if [[ ! -f "$SFT/adapter_config.json" ]]; then
    SFT=$PARENT/2026-10-30_23-00-Omni7B-SFT-GRPO/sft_run/train/checkpoints/checkpoint-1638
fi
RUN_TAG=${DAPO_RUN_TAG:-dapo7b}
BEST=$ROOT/runs/$RUN_TAG/best.json
TEST=${DAPO_TEST:-$HOME/grpo_mm/test_release.jsonl}
MEDIA=${DAPO_TEST_MEDIA:-$HOME/MM-LLMs-Emotion-main/Dataset}
NAME=${1:-${RUN_TAG}_best}
DEST=$ROOT/submissions/$NAME

for required in "$PY" "$BASE/config.json" "$BEST" "$TEST"; do
    if [[ ! -e "$required" ]]; then echo "Missing $required" >&2; exit 2; fi
done
if [[ -e "$DEST" ]]; then echo "Submission directory exists: $DEST" >&2; exit 2; fi
SOURCE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$BEST")
ADAPTER=$SOURCE
if [[ "$SOURCE" == "sft_baseline" ]]; then ADAPTER=$SFT; fi
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then echo "Adapter missing: $ADAPTER" >&2; exit 2; fi

export PYTHONPATH="$HOME/grpo_mm/envs/tf457:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CARD=$("$PY" "$ROOT/scripts/pick_cards.py" 1 800)
export CUDA_VISIBLE_DEVICES="$CARD"
mkdir -p "$ROOT/logs" "$ROOT/submissions"
"$PY" -u "$ROOT/code/make_submission.py" \
    --base "$BASE" --adapter "$ADAPTER" --code "$ROOT/code/sft" \
    --media-root "$MEDIA" --test "$TEST" --out "$DEST" \
    --batch-size 4 --max-new-tokens 192 \
    > "$ROOT/logs/submission_$NAME.log" 2>&1
echo "$DEST/submission.csv"
