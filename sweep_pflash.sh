#!/bin/bash
# PFlash keep-ratio × context-size sweep
# Output: sweep_results.csv with columns:
#   ctx,keep_ratio,total_tok,kept_tok,draft_ms,prefill_ms,prefill_tps,ttft_ms,pass,recovered

set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=./build/bin

RESULTS="/tmp/sweep_results.csv"
echo "ctx,keep_ratio,total_tok,kept_tok,draft_ms,prefill_ms,prefill_tps,ttft_ms,pass,recovered" > "$RESULTS"

CONTEXT_SIZES=(16384 20480 25600 32768 51200 65536 102400)
KEEP_RATIOS=(0.65 0.70 0.75 0.80 0.85)
WINDOW=${PFLASH_WINDOW:-4096}

# Function to get fixture file for a context size
fixture_for() {
    local ctx=$1
    if [ "$ctx" -eq 16384 ]; then echo "tools/niah/niah_16k.jsonl"
    elif [ "$ctx" -eq 20480 ]; then echo "tools/niah/niah_20k.jsonl"
    elif [ "$ctx" -eq 25600 ]; then echo "tools/niah/niah_25k.jsonl"
    elif [ "$ctx" -eq 32768 ]; then echo "tools/niah/niah_32k.jsonl"
    elif [ "$ctx" -eq 51200 ]; then echo "tools/niah/niah_50k.jsonl"
    elif [ "$ctx" -eq 65536 ]; then echo "tools/niah/niah_64k.jsonl"
    elif [ "$ctx" -eq 102400 ]; then echo "tools/niah/niah_100k.jsonl"
    else echo "UNKNOWN"
    fi
}

TOTAL=$(( ${#CONTEXT_SIZES[@]} * ${#KEEP_RATIOS[@]} ))
COUNT=0

for CTX in "${CONTEXT_SIZES[@]}"; do
    FIX=$(fixture_for "$CTX")
    for KR in "${KEEP_RATIOS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "=== [$COUNT/$TOTAL] ctx=${CTX} keep=${KR} ==="

        json=$(timeout 300 ./build/bin/llama-niah \
            --model /root/Qwen3.6-27B-UD-Q4_K_XL.gguf \
            --fixture "$FIX" \
            --cache-type-k turbo3 --cache-type-v turbo3 \
            --gpu-layers -1 \
            --no-warmup --output json \
            --draft /root/Qwen3.5-0.8B-Q8_0.gguf \
            --draft-cache-k f32 \
            --pflash-keep-ratio "$KR" \
            --pflash-block-size 128 \
            --pflash-sink 2048 --pflash-recent 4096 \
            --pflash-threshold 0 \
            --pflash-window "$WINDOW" \
            --ctx-size "$CTX" \
            --batch-size 2048 --ubatch-size 512 \
            2>/dev/null)

        total=$(echo "$json" | grep -o '"n_prompt":[0-9]*' | cut -d: -f2)
        kept=$(echo "$json" | grep -o '"pflash_kept_tokens":[0-9]*' | cut -d: -f2)
        draft=$(echo "$json" | grep -o '"pflash_draft_ms":[0-9.]*' | cut -d: -f2)
        pref=$(echo "$json" | grep -o '"prefill_ms":[0-9.]*' | cut -d: -f2)
        prefs=$(echo "$json" | grep -o '"prefill_us":[0-9]*' | cut -d: -f2)
        ttft=$(echo "$json" | grep -o '"ttft_ms":[0-9.]*' | cut -d: -f2)
        pass=$(echo "$json" | grep -o '"pass":true\|"pass":false' | head -1 | cut -d: -f2)
        recv=$(echo "$json" | grep -o '"recovered":[0-9]*' | cut -d: -f2)

        # Calculate prefill t/s from us
        if [ -n "$prefs" ] && [ "$prefs" -gt 0 ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
            tps=$(python3 -c "print(f'{1e6*$total/$prefs:.0f}')" 2>/dev/null || echo "0")
        else
            tps="0"
        fi

        [ -z "$total" ] && total=0
        [ -z "$kept" ] && kept=0
        [ -z "$draft" ] && draft=0
        [ -z "$pref" ] && pref=0
        [ -z "$ttft" ] && ttft=0
        [ "$pass" = "true" ] && pass="1" || pass="0"

        echo "$CTX,$KR,$total,$kept,$draft,$pref,$tps,$ttft,$pass,$recv" >> "$RESULTS"
        echo "  total=$total kept=$kept draft=${draft}ms pref=${pref}ms tps=${tps} ttft=${ttft}ms pass=$pass recv=$recv"
    done
done

echo ""
echo "=== SWEEP COMPLETE ==="
echo "Results: $RESULTS"
awk -F, '
  NR==1 {print; next}
  {
    eff=$5+$8
    imp=($8==0)?0:(eff/$8-1.0)
    printf "ctx=%-6d kr=%-5s tot=%-5d kept=%-5d (%.0f%%) draft=%-6.0f pref=%-6.0f tps=%-6s eff=%-6.0f pass=%s recv=%s\n",
      $1,$2,$3,$4,100*$4/$3,$5,$6,$7,eff,$9,$10
  }
' "$RESULTS"