#!/bin/bash
# Phase 6C batch sweep — run tomorrow back-to-back
# Usage: bash run_phase6c.sh [tier1|tier2|mask|dispatch|all]
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=./build/bin

MODEL="/root/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT="/root/Qwen3.5-0.8B-Q8_0.gguf"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_niah() {
    local ctx=$1 fix=$2
    shift 2
    timeout 600 ./build/bin/llama-niah \
        --model "$MODEL" --draft "$DRAFT" --fixture "$fix" \
        --cache-type-k turbo3 --cache-type-v turbo3 \
        --draft-cache-k f32 --gpu-layers -1 --no-warmup --output json \
        --ctx-size "$ctx" --batch-size 2048 --ubatch-size 512 \
        "$@" 2>/dev/null
}

aggregate() {
    local ctx=$1 mode=$2 kr=$3
    local json_lines=$(cat)
    if [ -z "$json_lines" ]; then
        echo "$ctx,$mode,$kr,0,0,0,0,0,0"
        echo "  [$mode ctx=$ctx kr=$kr] NO OUTPUT"
        return
    fi
    python3 -c "
import json, sys
results = [json.loads(l) for l in sys.stdin if l.strip().startswith('{')]
if not results: sys.exit(0)
n = len(results)
n_pass = sum(1 for r in results if r.get('pass'))
mean_draft = sum(r.get('pflash_draft_ms', 0) for r in results) / n
mean_ttft = sum(r.get('ttft_ms', 0) for r in results) / n
kept_pct = sum(r.get('pflash_kept_tokens',0)/max(r.get('prompt_tokens',1),1)*100 for r in results) / n
print(f'$ctx,$mode,$kr,{n},{n_pass},{100*n_pass/n:.0f},{mean_draft:.0f},{mean_ttft:.0f},{kept_pct:.1f}')
print(f'  [$mode ctx=$ctx kr=$kr] n={n} pass={n_pass}/{n} ({100*n_pass/n:.0f}%) draft={mean_draft:.0f}ms ttft={mean_ttft:.0f}ms', file=__import__('sys').stderr)
" <<< "$json_lines"
}

# --- Tier 1: Quality floor discovery ---
tier1() {
    local OUT="/tmp/phase6c_tier1_${TIMESTAMP}.csv"
    echo "ctx,mode,keep,n_trials,n_pass,pass_pct,mean_draft_ms,mean_ttft_ms,mean_kept_pct" > "$OUT"
    echo "=== TIER 1: Quality floor ==="
    for CTX in 32768 131072; do
        for KR in 0.50 0.55 0.60 0.65 0.70 0.75 0.80; do
            FIX="tools/niah/niah_${CTX}_multi.jsonl"
            [ -f "$FIX" ] || { echo "  MISSING $FIX"; continue; }

            echo -n "[WIN ctx=$CTX kr=$KR] "
            run_niah "$CTX" "$FIX" \
                --pflash-keep-ratio "$KR" --pflash-block-size 128 \
                --pflash-sink 2048 --pflash-recent 4096 \
                --pflash-threshold 0 --pflash-window 4096 --pflash-layer -1 \
                | aggregate "$CTX" "win" "$KR" >> "$OUT"

            echo -n "[BSA ctx=$CTX kr=$KR] "
            run_niah "$CTX" "$FIX" \
                --pflash-keep-ratio "$KR" --pflash-block-size 128 \
                --pflash-sink 2048 --pflash-recent 4096 \
                --pflash-threshold 0 --pflash-window 0 --pflash-bsa --pflash-layer -1 \
                | aggregate "$CTX" "bsa" "$KR" >> "$OUT"
        done
    done
    echo "Done: $OUT"
}

# --- Tier 2: Full context sweep ---
tier2() {
    local OUT="/tmp/phase6c_tier2_${TIMESTAMP}.csv"
    echo "ctx,mode,keep,n_trials,n_pass,pass_pct,mean_draft_ms,mean_ttft_ms,mean_kept_pct" > "$OUT"
    echo "=== TIER 2: Full context sweep ==="
    RATIOS=(0.60 0.65 0.70)  # Update after Tier 1 analysis
    for CTX in 16384 32768 51200 65536 102400 131072; do
        for KR in "${RATIOS[@]}"; do
            FIX="tools/niah/niah_${CTX}_multi.jsonl"
            [ -f "$FIX" ] || { echo "  MISSING $FIX"; continue; }

            echo -n "[WIN ctx=$CTX kr=$KR] "
            run_niah "$CTX" "$FIX" \
                --pflash-keep-ratio "$KR" --pflash-block-size 128 \
                --pflash-sink 2048 --pflash-recent 4096 \
                --pflash-threshold 0 --pflash-window 4096 --pflash-layer -1 \
                | aggregate "$CTX" "win" "$KR" >> "$OUT"

            echo -n "[BSA ctx=$CTX kr=$KR] "
            run_niah "$CTX" "$FIX" \
                --pflash-keep-ratio "$KR" --pflash-block-size 128 \
                --pflash-sink 2048 --pflash-recent 4096 \
                --pflash-threshold 0 --pflash-window 0 --pflash-bsa --pflash-layer -1 \
                | aggregate "$CTX" "bsa" "$KR" >> "$OUT"
        done
    done
    echo "Done: $OUT"
}

# --- BSA mask size sensitivity sweep ---
mask_sweep() {
    local OUT="/tmp/phase6c_mask_${TIMESTAMP}.csv"
    echo "ctx,mask,sink,recent,keep,n_trials,n_pass,pass_pct,mean_draft_ms,mean_ttft_ms,mean_kept_pct" > "$OUT"
    echo "=== MASK SWEEP ==="
    for CTX in 32768 131072; do
        for CONFIG in "S 512 512" "M 1024 1024" "L 2048 2048" "XL 2048 4096"; do
            read -r name sink recent <<< "$CONFIG"
            FIX="tools/niah/niah_${CTX}_multi.jsonl"
            [ -f "$FIX" ] || { echo "  MISSING $FIX"; continue; }

            echo -n "[BSA-$name ctx=$CTX] "
            run_niah "$CTX" "$FIX" \
                --pflash-keep-ratio 0.65 --pflash-block-size 128 \
                --pflash-sink "$sink" --pflash-recent "$recent" \
                --pflash-threshold 0 --pflash-window 0 --pflash-bsa --pflash-layer -1 \
                | aggregate "$CTX" "bsa-$name" "0.65" >> "$OUT"
        done
    done
    echo "Done: $OUT"
}

# --- Dispatch overhead diagnostic ---
dispatch() {
    echo "=== DISPATCH OVERHEAD ==="
    for UBATCH in 256 512 1024 2048 4096; do
        FIX="tools/niah/niah_131072_multi.jsonl"
        [ -f "$FIX" ] || { echo "  MISSING $FIX"; continue; }

        echo -n "[ubatch=$UBATCH] "
        run_niah 131072 "$FIX" \
            --pflash-keep-ratio 0.65 --pflash-block-size 128 \
            --pflash-sink 2048 --pflash-recent 4096 \
            --pflash-threshold 0 --pflash-window 0 --pflash-bsa --pflash-layer -1 \
            --ubatch-size "$UBATCH" \
            | python3 -c "
import json, sys, statistics
lines = [json.loads(l) for l in sys.stdin if l.startswith('{')]
if not lines: print('NO OUTPUT'); sys.exit(0)
drafts = [r['pflash_draft_ms'] for r in lines]
print(f'n={len(drafts)} mean_draft={statistics.mean(drafts):.0f}ms min={min(drafts):.0f}ms max={max(drafts):.0f}ms')"
    done
}

case "${1:-all}" in
    tier1)   tier1 ;;
    tier2)   tier2 ;;
    mask)    mask_sweep ;;
    dispatch) dispatch ;;
    all)     tier1; tier2; mask_sweep; dispatch ;;
    *)       echo "Usage: $0 [tier1|tier2|mask|dispatch|all]" ;;
esac
