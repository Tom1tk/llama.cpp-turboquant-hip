#!/bin/bash
# PFlash keep-ratio × context-size sweep — v2 (post RoPE position fix)
# Output: sweep_v2_TIMESTAMP.csv with columns:
#   ctx,keep_ratio,total_tok,kept_tok,draft_ms,prefill_ms,prefill_tps,ttft_ms,pass,recovered
#
# Usage:  bash sweep_pflash.sh [--quick] [--baseline-only]
#   --quick          Test only 16k + 20k at 0.65–0.85 (verifies RoPE fix)
#   --baseline-only   Only run baseline (no PFlash) for all sizes

set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=./build/bin

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="/tmp/sweep_v2_${TIMESTAMP}.csv"
HEADER="ctx,keep,total,kept,draft_ms,pref_ms,tps,ttft_ms,pass,recv"
echo "$HEADER" > "$RESULTS"

MODEL="/root/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT="/root/Qwen3.5-0.8B-Q8_0.gguf"

declare -A FIXTURES
FIXTURES[16384]="tools/niah/niah_16k.jsonl"
FIXTURES[20480]="tools/niah/niah_20k.jsonl"
FIXTURES[25600]="tools/niah/niah_25k.jsonl"
FIXTURES[32768]="tools/niah/niah_32k.jsonl"
FIXTURES[51200]="tools/niah/niah_50k.jsonl"
FIXTURES[65536]="tools/niah/niah_64k.jsonl"
FIXTURES[102400]="tools/niah/niah_100k.jsonl"

QUICK="${1:-}"
BASELINE_ONLY="${2:-}"

run_niah() {
    local ctx=$1 fix=$2 draft_cfg=""
    shift 2
    if [ -n "$DRAFT" ]; then
        draft_cfg="--draft $DRAFT --draft-cache-k f32"
    fi
    timeout 300 ./build/bin/llama-niah \
        --model "$MODEL" \
        --fixture "$fix" \
        --cache-type-k turbo3 --cache-type-v turbo3 \
        --gpu-layers -1 --no-warmup --output json \
        $draft_cfg \
        "$@" \
        --ctx-size "$ctx" \
        --batch-size 2048 --ubatch-size 512 \
        2>/dev/null
}

parse_and_log() {
    local ctx=$1 kr=$2 json_line=$3
    if [ -z "$json_line" ]; then
        echo "$ctx,$kr,0,0,0,0,0,0,0,0" >> "$RESULTS"
        echo "  ❌ NO OUTPUT"
        return
    fi
    local total=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt_tokens'])")
    local kept=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['pflash_kept_tokens'])")
    local draft=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['pflash_draft_ms'])")
    local pref=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['prefill_ms'])")
    local prefus=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['prefill_us'])")
    local ttft=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['ttft_ms'])")
    local pass=$(echo "$json_line" | python3 -c "import json,sys; print(1 if json.load(sys.stdin)['pass'] else 0)")
    local recv=$(echo "$json_line" | python3 -c "import json,sys; print(json.load(sys.stdin)['recovered'])")
    local tps=$(python3 -c "print(f'{1e6*$total/$prefus:.0f}')")
    local pas="PASS"; [ "$pass" = "0" ] && pas="FAIL"
    printf "  %-5s %-4s tot=%-6d kept=%-6d (%-3.0f%%) draft=%-6.0fms pref=%-6.0fms t/s=%-5s ttft=%-6.0fms %s\n" \
        "ctx=$ctx" "kr=$kr" "$total" "$kept" "$(python3 -c "print(f'{100*$kept/$total:.0f}')")" "$draft" "$pref" "$tps" "$ttft" "$pas"
    echo "$ctx,$kr,$total,$kept,$draft,$pref,$tps,$ttft,$pass,$recv" >> "$RESULTS"
}

run_pflash_test() {
    local ctx=$1 kr=$2 fix=$3
    local json
    json=$(run_niah "$ctx" "$fix" \
        --pflash-keep-ratio "$kr" \
        --pflash-block-size 128 \
        --pflash-sink 2048 --pflash-recent 4096 \
        --pflash-threshold 0 --pflash-window 4096 \
        --pflash-layer -1)
    local json_line=$(echo "$json" | grep '^{"answer"')
    parse_and_log "$ctx" "$kr" "$json_line"
}

run_baseline_test() {
    local ctx=$1 fix=$2
    local json=$(run_niah "$ctx" "$fix")
    local json_line=$(echo "$json" | grep '^{"answer"')
    parse_and_log "$ctx" "1.0" "$json_line"
}

# ============= BASELINES =============
echo "=== BASELINES (no PFlash) ==="
if [ "$BASELINE_ONLY" = "--baseline-only" ] || [ "$QUICK" = "--quick" ]; then
    CTXS=(16384 20480)
    [ "$BASELINE_ONLY" = "--baseline-only" ] && CTXS=(16384 20480 25600 32768 51200 65536 102400)
else
    CTXS=(16384 20480 25600 32768 51200 65536 102400)
fi
for CTX in "${CTXS[@]}"; do
    FIX="${FIXTURES[$CTX]}"
    echo -n "  ${CTX}... "
    run_baseline_test "$CTX" "$FIX"
done

[ "$BASELINE_ONLY" = "--baseline-only" ] && exit 0

# ============= SWEEP =============
echo ""
echo "=== PFLASH SWEEP ==="

if [ "$QUICK" = "--quick" ]; then
    SIZES=(16384 20480)
else
    SIZES=(16384 20480 25600 32768 51200 65536 102400)
fi
RATIOS=(0.65 0.70 0.75 0.80 0.85)
TOTAL=$(( ${#SIZES[@]} * ${#RATIOS[@]} ))
COUNT=0

for CTX in "${SIZES[@]}"; do
    FIX="${FIXTURES[$CTX]}"
    for KR in "${RATIOS[@]}"; do
        COUNT=$((COUNT+1))
        echo -n "[$COUNT/$TOTAL] "
        run_pflash_test "$CTX" "$KR" "$FIX"
    done
done

# ============= SUMMARY =============
echo ""
echo "=== RESULTS: $RESULTS ==="
python3 << PYEOF
import csv, os

sweep = list(csv.DictReader(open('$RESULTS')))
baselines = {int(r['ctx']): r for r in sweep if r['keep'] == '1.0'}
pflash = [r for r in sweep if r['keep'] != '1.0']

print(f"{'Context':>8} {'R':>4} {'Kept':>7} {'Kept%':>5} {'Draft':>8} {'Prefill':>10} {'TPS':>6} {'EffTTFT':>10} {'BL_TTFT':>10} {'Speedup':>8} {'NIAH':>6}")
print("-" * 100)

for r in pflash:
    ctx = int(r['ctx'])
    kr = float(r['keep'])
    total = int(r['total'])
    kept = int(r['kept'])
    draft = float(r['draft_ms'])
    pref = float(r['pref_ms'])
    ttft = float(r['ttft_ms'])
    tps = int(r['tps'])
    pass_v = int(r['pass'])
    bl = baselines.get(ctx)
    if bl:
        bl_ttft = float(bl['ttft_ms'])
        eff = draft + pref
        sup = f"{100*(bl_ttft - eff)/bl_ttft:+.1f}%" if pass_v else "N/A(Q)"
        print(f"{ctx:>8} {kr:>4.2f} {kept:>7} {100*kept/total:>4.1f}% {draft:>8.0f} {pref:>10.0f} {tps:>6} {eff:>10.0f} {bl_ttft:>10.0f} {sup:>8} {'PASS' if pass_v else 'FAIL':>6}")
    else:
        print(f"{ctx:>8} {kr:>4.2f} {kept:>7} {100*kept/total:>4.1f}% {draft:>8.0f} {pref:>10.0f} {tps:>6} {ttft:>10.0f} {'—':>10} {'—':>8} {'PASS' if pass_v else 'FAIL':>6}")

print()
print("Baselines:")
for ctx in sorted(baselines.keys(), key=int):
    r = baselines[ctx]
    print(f"  {ctx:>6}   total={r['total']}  pref={float(r['pref_ms']):.0f}ms  tps={r['tps']}  ttft={float(r['ttft_ms']):.0f}ms")
PYEOF
