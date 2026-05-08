#!/bin/bash
# Phase 7 batch — code context packing, baseline, PFlash sweeps
# Usage: bash run_phase7.sh [pack|baseline|sweep|type-e|all]
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=./build/bin

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL="/root/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT="/root/Qwen3.5-0.8B-Q8_0.gguf"

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

# --- PACK: assemble context files ---
pack() {
    echo "=== PACKING CODE CONTEXTS ==="
    python3 tools/niah/pack_code_context.py \
        --files "${PFLASH_FILES[@]}" --budget 30000 --order shuffle --seed 10 \
        --out /tmp/ctx_32k.txt --manifest /tmp/ctx_32k_manifest.json

    python3 tools/niah/pack_code_context.py \
        --files "${PFLASH_FILES[@]}" --budget 60000 --order shuffle --seed 20 \
        --out /tmp/ctx_64k.txt --manifest /tmp/ctx_64k_manifest.json

    python3 tools/niah/pack_code_context.py \
        --files "${PFLASH_FILES[@]}" --budget 92000 --order shuffle --seed 30 \
        --out /tmp/ctx_96k.txt --manifest /tmp/ctx_96k_manifest.json

    # Generate fixtures
    for SIZE in 32k 64k 96k; do
        python3 tools/niah/gen_code_fixtures.py \
            --context /tmp/ctx_${SIZE}.txt \
            --manifest /tmp/ctx_${SIZE}_manifest.json \
            --questions tools/niah/questions.yaml \
            --out tools/niah/code_fixtures_${SIZE}.jsonl
        echo "  code_fixtures_${SIZE}.jsonl: $(wc -l < tools/niah/code_fixtures_${SIZE}.jsonl) fixtures"
    done

    # Mid-only variants
    for SIZE in 64k 96k; do
        python3 tools/niah/gen_code_fixtures.py \
            --context /tmp/ctx_${SIZE}.txt \
            --manifest /tmp/ctx_${SIZE}_manifest.json \
            --questions tools/niah/questions.yaml \
            --filter-bucket mid \
            --out tools/niah/code_fixtures_${SIZE}_mid.jsonl
        echo "  code_fixtures_${SIZE}_mid.jsonl: $(wc -l < tools/niah/code_fixtures_${SIZE}_mid.jsonl) mid-answer fixtures"
    done

    # Type E only
    python3 tools/niah/gen_code_fixtures.py \
        --context /tmp/ctx_64k.txt \
        --manifest /tmp/ctx_64k_manifest.json \
        --questions tools/niah/questions.yaml \
        --filter-type E \
        --out /tmp/code_fixtures_64k_typeE.jsonl
}

# --- BASELINE: 100% keep, no draft ---
baseline() {
    echo "=== BASELINE (keep_ratio=1.0, no draft) ==="
    for SIZE in 32k 64k 96k; do
        CTX=$(python3 -c "import json; print(json.load(open('/tmp/ctx_${SIZE}_manifest.json'))['total_tokens_approx'] + 500)")
        FIX="tools/niah/code_fixtures_${SIZE}.jsonl"
        [ -f "$FIX" ] || { echo "  MISSING $FIX — run pack first"; continue; }

        echo -n "  $SIZE (ctx=$CTX): "
        timeout 900 ./build/bin/llama-niah \
            --model "$MODEL" \
            --fixture "$FIX" \
            --cache-type-k turbo3 --cache-type-v turbo3 \
            --gpu-layers -1 --ctx-size "$CTX" \
            --max-gen 512 --output json 2>/dev/null \
            | python3 -c "
import json,sys,collections
rs=[json.loads(l) for l in sys.stdin if l.startswith('{')]
by_type=collections.defaultdict(list)
for r in rs: by_type[r.get('type','?')].append(r)
total=len(rs); n_pass=sum(1 for r in rs if r['pass'])
print(f'pass={n_pass}/{total} ({100*n_pass//total if total else 0}%)')
for t in 'ABCDE':
    rts=by_type.get(t,[])
    if rts:
        p=sum(1 for r in rts if r['pass'])
        print(f'    Type {t}: {p}/{len(rts)}')
for r in rs:
    s='PASS' if r['pass'] else 'FAIL'
    print(f'    [{s}] {r.get(\"id\",\"?\")} ({r.get(\"type\",\"?\")}): {r.get(\"answer\",\"\")[:120]}')
"
    done
}

# --- PFLASH SWEEP: 64k, both modes ---
sweep() {
    local OUT="/tmp/phase7_sweep_${TIMESTAMP}.csv"
    echo "kr,mode,n,n_pass,pass_pct,type_a,type_b,type_c,type_d,type_e" > "$OUT"
    echo "=== PFLASH SWEEP @ 64k ==="
    CTX=65536
    FIX="tools/niah/code_fixtures_64k.jsonl"
    [ -f "$FIX" ] || { echo "  MISSING $FIX — run pack first"; return; }

    for KR in 0.50 0.55 0.60 0.65 0.70 0.75 0.80; do
        for MODE in windowed bsa; do
            if [ "$MODE" = "windowed" ]; then
                EXTRA="--pflash-window 4096"
            else
                EXTRA="--pflash-window 0 --pflash-bsa --pflash-sink 2048 --pflash-recent 4096"
            fi

            echo -n "  $MODE kr=$KR: "
            timeout 600 ./build/bin/llama-niah \
                --model "$MODEL" --draft "$DRAFT" \
                --fixture "$FIX" \
                --cache-type-k turbo3 --cache-type-v turbo3 \
                --draft-cache-k f32 --gpu-layers -1 \
                --ctx-size "$CTX" \
                --pflash-keep-ratio "$KR" --pflash-threshold 0 --pflash-block-size 128 \
                --pflash-sink 2048 --pflash-recent 4096 --pflash-layer -1 \
                $EXTRA \
                --output json 2>/dev/null \
                | python3 -c "
import json,sys,collections
rs=[json.loads(l) for l in sys.stdin if l.startswith('{')]
by_type=collections.defaultdict(list)
for r in rs: by_type[r.get('type','?')].append(r)
total=len(rs); n_pass=sum(1 for r in rs if r['pass'])
a_p=sum(1 for r in by_type.get('A',[]) if r['pass']); a_t=len(by_type.get('A',[]))
b_p=sum(1 for r in by_type.get('B',[]) if r['pass']); b_t=len(by_type.get('B',[]))
c_p=sum(1 for r in by_type.get('C',[]) if r['pass']); c_t=len(by_type.get('C',[]))
d_p=sum(1 for r in by_type.get('D',[]) if r['pass']); d_t=len(by_type.get('D',[]))
e_p=sum(1 for r in by_type.get('E',[]) if r['pass']); e_t=len(by_type.get('E',[]))
pct=100*n_pass//total if total else 0
print(f'pass={n_pass}/{total} ({pct}%) A:{a_p}/{a_t} B:{b_p}/{b_t} C:{c_p}/{c_t} D:{d_p}/{d_t} E:{e_p}/{e_t}')
" 2>/dev/null
        done
    done
}

# --- MID-ONLY SWEEP ---
mid_sweep() {
    echo "=== MID-ONLY SWEEP @ 64k ==="
    for KR in 0.60 0.65 0.70; do
        for MODE in windowed bsa; do
            if [ "$MODE" = "windowed" ]; then
                EXTRA="--pflash-window 4096"
                FIX="tools/niah/code_fixtures_64k_mid.jsonl"
            else
                EXTRA="--pflash-window 0 --pflash-bsa --pflash-sink 2048 --pflash-recent 4096"
                FIX="tools/niah/code_fixtures_64k_mid.jsonl"
            fi
            [ -f "$FIX" ] || { echo "  MISSING $FIX — run pack first"; return; }

            echo -n "  $MODE kr=$KR (mid-only): "
            timeout 600 ./build/bin/llama-niah \
                --model "$MODEL" --draft "$DRAFT" \
                --fixture "$FIX" \
                --cache-type-k turbo3 --cache-type-v turbo3 \
                --draft-cache-k f32 --gpu-layers -1 \
                --ctx-size 65536 \
                --pflash-keep-ratio "$KR" --pflash-threshold 0 --pflash-block-size 128 \
                --pflash-sink 2048 --pflash-recent 4096 --pflash-layer -1 \
                $EXTRA \
                --output json 2>/dev/null \
                | python3 -c "
import json,sys
rs=[json.loads(l) for l in sys.stdin if l.startswith('{')]
p=sum(1 for r in rs if r['pass'])
print(f'mid-answer pass={p}/{len(rs)}')
"
        done
    done
}

# --- TYPE E ANOMALY DETECTION SWEEP ---
type_e() {
    echo "=== TYPE E SWEEP ==="
    local FIX="/tmp/code_fixtures_64k_typeE.jsonl"
    [ -f "$FIX" ] || { echo "  MISSING $FIX — run pack first"; return; }

    for KR in 1.00 0.75 0.65 0.55; do
        echo "--- Type E at kr=$KR ---"
        timeout 600 ./build/bin/llama-niah \
            --model "$MODEL" --draft "$DRAFT" \
            --fixture "$FIX" \
            --cache-type-k turbo3 --cache-type-v turbo3 \
            --draft-cache-k f32 --gpu-layers -1 \
            --ctx-size 65536 \
            --pflash-keep-ratio "$KR" --pflash-threshold 0 \
            --pflash-window 4096 --pflash-block-size 128 \
            --pflash-sink 2048 --pflash-recent 4096 --pflash-layer -1 \
            --output json 2>/dev/null \
            | python3 -c "
import json,sys
for l in sys.stdin:
    if not l.startswith('{'): continue
    r=json.loads(l)
    s='PASS' if r['pass'] else 'FAIL'
    print(f'[{s}] {r.get(\"id\",\"?\")}')
    ans=r.get('answer','')
    if ans:
        print(f'  Answer: {ans[:200]}')
    print()
"
    done
}

case "${1:-none}" in
    pack)     pack ;;
    baseline) baseline ;;
    sweep)    sweep ;;
    mid)      mid_sweep ;;
    type-e)   type_e ;;
    all)
        pack
        baseline
        sweep
        mid_sweep
        type_e
        ;;
    *)
        echo "Usage: $0 [pack|baseline|sweep|mid|type-e|all]"
        echo ""
        echo "  pack     - assemble 32k/64k/96k context packs + generate fixtures"
        echo "  baseline - run all questions at 100% keep (no draft)"
        echo "  sweep    - PFlash sweep at 64k (7 ratios × 2 modes)"
        echo "  mid      - mid-answer-only sweep at 64k (3 ratios × 2 modes)"
        echo "  type-e   - Type E anomaly sweep with full answer inspection"
        echo "  all      - run everything"
        ;;
esac
