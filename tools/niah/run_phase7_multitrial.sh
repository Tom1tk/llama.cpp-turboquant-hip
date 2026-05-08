#!/bin/bash
# Phase 7 multi-trial baseline (10 context shuffles per size)
# Usage: bash run_phase7_multitrial.sh [pack|baseline|sweep]
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=./build/bin

MODEL="/root/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT="/root/Qwen3.5-0.8B-Q8_0.gguf"
N_TRIALS=10
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="/tmp/phase7_${TIMESTAMP}"

PFLASH_FILES=(
    include/pflash.h
    src/llama-cparams.h
    src/llama-context.h
    src/llama-context.cpp
    src/llama-graph.h
    src/llama-graph.cpp
    tools/niah/niah.cpp
    tools/niah/pflash.cpp
    tools/niah/test_bsa.cpp
    ggml/include/ggml.h
    ggml/src/ggml.c
    ggml/src/ggml-cuda/pflash-bsa.cu
    ggml/src/ggml-cuda/pflash-bsa.cuh
    ggml/src/ggml-cuda/pflash-score.cu
    ggml/src/ggml-cuda/ggml-cuda.cu
    README.md
)

pack() {
    mkdir -p "$OUTDIR"
    echo "=== PACKING (${N_TRIALS} shuffles per size) ==="

    for SIZE in 32k 64k 96k; do
        case "$SIZE" in
            32k) BUDGET=30000 ;;
            64k) BUDGET=60000 ;;
            96k) BUDGET=92000 ;;
        esac

        for TRIAL in $(seq 0 $((N_TRIALS - 1))); do
            SEED=$((10 + TRIAL))
            PREFIX="${OUTDIR}/ctx_${SIZE}_t${TRIAL}"

            python3 tools/niah/pack_code_context.py \
                --files "${PFLASH_FILES[@]}" --budget "$BUDGET" \
                --order shuffle --seed "$SEED" \
                --out "${PREFIX}.txt" --manifest "${PREFIX}_manifest.json"

            python3 tools/niah/gen_code_fixtures.py \
                --context "${PREFIX}.txt" \
                --manifest "${PREFIX}_manifest.json" \
                --questions tools/niah/questions.yaml \
                --out "${PREFIX}_fixtures.jsonl"
        done
        echo "  $SIZE: ${N_TRIALS} packs with $(wc -l < "${OUTDIR}/ctx_${SIZE}_t0_fixtures.jsonl") questions each"
    done
}

baseline() {
    echo "=== BASELINE (100% keep, no draft, ${N_TRIALS} packs per size) ==="
    local BASELINE_CSV="${OUTDIR}/baseline.csv"
    echo "size,trial,type,id,n_tokens,pass,answer_trunc" > "$BASELINE_CSV"

    for SIZE in 32k 64k 96k; do
        case "$SIZE" in
            32k) CTX=32768 ;;
            64k) CTX=65536 ;;
            96k) CTX=98304 ;;
        esac

        for TRIAL in $(seq 0 $((N_TRIALS - 1))); do
            FIX="${OUTDIR}/ctx_${SIZE}_t${TRIAL}_fixtures.jsonl"
            [ -f "$FIX" ] || { echo "  MISSING $FIX — run pack first"; continue; }

            printf "  %s trial=%s " "$SIZE" "$TRIAL"
            timeout 300 ./build/bin/llama-niah \
                --model "$MODEL" \
                --fixture "$FIX" \
                --cache-type-k turbo3 --cache-type-v turbo3 \
                --gpu-layers -1 --ctx-size "$CTX" \
                --no-warmup --output json 2>/dev/null \
                | python3 -c "
import json, sys
rs = [json.loads(l) for l in sys.stdin if l.startswith('{') and 'pass' in l]
n_pass = sum(1 for r in rs if r['pass'])
print(f'pass={n_pass}/{len(rs)} ({100*n_pass//max(len(rs),1)}%)', end='')
with open('${BASELINE_CSV}', 'a') as f:
    for r in rs:
        at = r.get('answer', '')[:80].replace(',', ';').replace('\"', '')
        f.write(f'${SIZE},${TRIAL},{r.get(\"type\",\"?\")},{r.get(\"id\",\"?\")},{r.get(\"prompt_tokens\",0)},{r[\"pass\"]},\"{at}\"\n')
" 2>/dev/null
            echo
        done
    done

    echo
    echo "=== BASELINE SUMMARY ==="
    python3 -c "
import csv, collections, sys
rows = list(csv.DictReader(open('${BASELINE_CSV}')))
by_sz = collections.defaultdict(list)
for r in rows: by_sz[r['size']].append(r)
for sz in ['32k', '64k', '96k']:
    rs = by_sz.get(sz, [])
    n = len(rs)
    n_pass = sum(1 for r in rs if r['pass'] == 'True')
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        by_type[r['type']][1] += 1
        if r['pass'] == 'True': by_type[r['type']][0] += 1
    print(f'{sz}: {n_pass}/{n} ({100*n_pass//max(n,1)}%)', end='')
    for t in 'ABCDE':
        tp, tt = by_type.get(t, [0, 0])
        if tt: print(f'  {t}:{tp}/{tt}', end='')
    print()
"
}

sweep() {
    echo "=== PFLASH SWEEP @ 64k (7 ratios × 2 modes × ${N_TRIALS} trials) ==="
    local SWEEP_CSV="${OUTDIR}/sweep.csv"
    echo "mode,kr,trial,type,id,pass" > "$SWEEP_CSV"
    CTX=65536

    for KR in 0.50 0.55 0.60 0.65 0.70 0.75 0.80; do
        for MODE in windowed bsa; do
            if [ "$MODE" = "windowed" ]; then
                EXTRA="--pflash-window 4096"
            else
                EXTRA="--pflash-window 0 --pflash-bsa --pflash-sink 2048 --pflash-recent 4096"
            fi

            for TRIAL in $(seq 0 $((N_TRIALS - 1))); do
                FIX="${OUTDIR}/ctx_64k_t${TRIAL}_fixtures.jsonl"
                [ -f "$FIX" ] || continue

                printf "  %s kr=%.2f trial=%s " "$MODE" "$KR" "$TRIAL"
                timeout 300 ./build/bin/llama-niah \
                    --model "$MODEL" --draft "$DRAFT" \
                    --fixture "$FIX" \
                    --cache-type-k turbo3 --cache-type-v turbo3 \
                    --draft-cache-k f32 --gpu-layers -1 --no-warmup \
                    --ctx-size "$CTX" \
                    --pflash-keep-ratio "$KR" --pflash-threshold 0 --pflash-block-size 128 \
                    --pflash-sink 2048 --pflash-recent 4096 --pflash-layer -1 \
                    $EXTRA \
                    --output json 2>/dev/null \
                    | python3 -c "
import json, sys
rs = [json.loads(l) for l in sys.stdin if l.startswith('{') and 'pass' in l]
n_pass = sum(1 for r in rs if r['pass'])
print(f'pass={n_pass}/{len(rs)} ({100*n_pass//max(len(rs),1)}%)', end='')
with open('${SWEEP_CSV}', 'a') as f:
    for r in rs:
        f.write(f'${MODE},${KR},${TRIAL},{r.get(\"type\",\"?\")},{r.get(\"id\",\"?\")},{r[\"pass\"]}\n')
" 2>/dev/null
                echo
            done
        done
    done

    echo
    echo "=== SWEEP SUMMARY ==="
    python3 -c "
import csv, collections
rows = list(csv.DictReader(open('${SWEEP_CSV}')))
by_kr_mode = collections.defaultdict(lambda: [0, 0])
for r in rows:
    key = (r['mode'], r['kr'])
    by_kr_mode[key][1] += 1
    if r['pass'] == 'True': by_kr_mode[key][0] += 1
for (mode, kr), (p, t) in sorted(by_kr_mode.items(), key=lambda x: (x[0][0], float(x[0][1]))):
    pct = 100 * p // t if t else 0
    print(f'{mode:>7} kr={kr}  {p}/{t} ({pct}%)')
"
}

case "${1:-}" in
    pack)     pack ;;
    baseline) baseline ;;
    sweep)    sweep ;;
    all)
        pack
        baseline
        sweep
        ;;
    *)
        echo "Usage: $0 [pack|baseline|sweep|all]"
        echo "  pack     - generate ${N_TRIALS} context shuffles per size"
        echo "  baseline - run all questions at 100% keep (${N_TRIALS} trials)"
        echo "  sweep    - PFlash sweep at 64k (7 ratios × 2 modes × ${N_TRIALS} trials)"
        ;;
esac
