#!/bin/bash
# Run one Phase 7 question at one (mode, kr, position, run_idx) configuration.
# Usage:
#   bash run_q.sh A1 windowed 0.65 mid 0
#   bash run_q.sh B3 bsa     0.65 late 2
#
# Arguments: <question_id> <mode:windowed|bsa|baseline> <kr> <position:early|mid|late> <run_idx>
# Output:    tools/niah/results/<id>_<mode>_<kr>_pos-<pos>_r<run>.json
# Resume:    if result file exists, skip immediately (exit 0)

set -euo pipefail
QID="${1:?question id required}"
MODE="${2:?mode required}"          # windowed | bsa | baseline
KR="${3:-1.00}"
POS="${4:-early}"
RUN="${5:-0}"
TIMEOUT="${PHASE7_TIMEOUT:-600}"    # override: PHASE7_TIMEOUT=300 bash run_q.sh ...
MAX_GEN="${PHASE7_MAXGEN:-512}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${SCRIPT_DIR}/../.."
cd "$REPO"
export LD_LIBRARY_PATH=./build/bin

MODEL="/root/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT="/root/Qwen3.5-0.8B-Q8_0.gguf"

FIXTURE="tools/niah/fixtures/${QID}_pos-${POS}.jsonl"
RESULTS_DIR="tools/niah/results"
RESULT_FILE="${RESULTS_DIR}/${QID}_${MODE}_${KR}_pos-${POS}_r${RUN}.json"

mkdir -p "$RESULTS_DIR"

# Resume: skip if already done
if [ -f "$RESULT_FILE" ]; then
    pass=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('pass',False))")
    echo "  SKIP $QID $MODE kr=$KR pos=$POS r=$RUN [cached: pass=$pass]"
    exit 0
fi

if [ ! -f "$FIXTURE" ]; then
    echo "  ERROR: fixture not found: $FIXTURE" >&2
    exit 1
fi

# Build draft args for non-baseline modes
DRAFT_ARGS=""
PFLASH_ARGS=""
if [ "$MODE" = "windowed" ]; then
    DRAFT_ARGS="--draft $DRAFT --draft-cache-k f32"
    PFLASH_ARGS="--pflash-threshold 0 --pflash-keep-ratio $KR \
                 --pflash-window 4096 --pflash-sink 2048 --pflash-recent 4096 \
                 --pflash-block-size 128 --pflash-layer -1"
elif [ "$MODE" = "bsa" ]; then
    DRAFT_ARGS="--draft $DRAFT --draft-cache-k f32"
    PFLASH_ARGS="--pflash-threshold 0 --pflash-keep-ratio $KR \
                 --pflash-window 0 --pflash-bsa \
                 --pflash-sink 2048 --pflash-recent 4096 \
                 --pflash-block-size 128 --pflash-layer -1"
fi
# baseline: no draft, no PFlash args

CTX=$(python3 -c "
import json
line = open('$FIXTURE').readline()
d = json.loads(line)
print(d.get('context_tokens', 32768) + 2048)
")

printf "  RUN  %-4s %-8s kr=%-4s pos=%-5s r=%s ctx=%s\n" "$QID" "$MODE" "$KR" "$POS" "$RUN" "$CTX"

RAW=$(timeout "$TIMEOUT" ./build/bin/llama-niah \
    --model "$MODEL" \
    --fixture "$FIXTURE" \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --gpu-layers -1 --no-warmup \
    --ctx-size "$CTX" \
    --max-gen "$MAX_GEN" \
    $DRAFT_ARGS $PFLASH_ARGS \
    --output json 2>/dev/null) || true

if [ -z "$RAW" ]; then
    echo "  TIMEOUT/FAIL $QID $MODE kr=$KR pos=$POS r=$RUN" >&2
    # Write a failure record so resume doesn't re-attempt indefinitely
    echo "{\"id\":\"$QID\",\"mode\":\"$MODE\",\"kr\":\"$KR\",\"position\":\"$POS\",\"run\":$RUN,\"pass\":false,\"answer\":\"TIMEOUT\",\"status\":\"TIMEOUT\"}" > "$RESULT_FILE"
    exit 0
fi

RESULT_LINE=$(echo "$RAW" | grep '^{' | tail -1)
if [ -z "$RESULT_LINE" ]; then
    echo "  NO_JSON $QID $MODE kr=$KR pos=$POS r=$RUN" >&2
    echo "{\"id\":\"$QID\",\"mode\":\"$MODE\",\"kr\":\"$KR\",\"position\":\"$POS\",\"run\":$RUN,\"pass\":false,\"answer\":\"NO_OUTPUT\",\"status\":\"ERROR\"}" > "$RESULT_FILE"
    exit 0
fi

# Augment result with run metadata
python3 -c "
import json, sys
d = json.loads(sys.argv[1])
d['mode']     = '$MODE'
d['kr']       = '$KR'
d['position'] = '$POS'
d['run']      = $RUN
print(json.dumps(d))
" "$RESULT_LINE" > "$RESULT_FILE"

PASS=$(python3 -c "import json; print(json.load(open('$RESULT_FILE'))['pass'])")
printf "  %-4s %-4s %-8s kr=%-4s pos=%-5s → %s\n" "$QID" "$PASS" "$MODE" "$KR" "$POS" "$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d['answer'][:60])" 2>/dev/null)"
