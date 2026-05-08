#!/bin/bash
# Run a batch of Phase 7 questions.
# Usage:
#   bash run_tier.sh --tier fast --mode windowed --kr 0.65 --positions early mid late
#   bash run_tier.sh --questions A1,B1,E1 --mode bsa --kr 0.50,0.65,0.80
#   bash run_tier.sh --tier all --mode baseline
#
# Tiers:
#   fast     : A1 A2 A3 A5 E3                     (~45s each, all positions ~5m)
#   medium   : A4 B2 C3 D1a D1b E1 E2 E4          (~60s each, all positions ~12m)
#   slow     : B1 B3 B4 B5 B6 C1 C2 D2a D2b       (~75s each, all positions ~20m)
#   core     : A1-A5 B1 B2 B4 C3 D1a D1b E1-E4    (all except giant files)
#   all      : all 22 questions

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Defaults
TIER="core"
QUESTIONS=""
MODES="windowed"
KRS="0.65"
POSITIONS="early"
RUNS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)         TIER="$2";        shift 2 ;;
        --questions)    QUESTIONS="$2";   shift 2 ;;
        --mode|--modes) MODES="$2";       shift 2 ;;
        --kr|--krs)     KRS="$2";         shift 2 ;;
        --positions)    POSITIONS="$2";   shift 2 ;;
        --runs)         RUNS="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Tier definitions
case "$TIER" in
    fast)   IDS="A1 A2 A3 A5 E3" ;;
    medium) IDS="A4 B2 C3 D1a D1b E1 E2 E4" ;;
    slow)   IDS="B1 B3 B4 B5 B6 C1 C2 D2a D2b" ;;
    core)   IDS="A1 A2 A3 A4 A5 B1 B2 B4 C3 D1a D1b E1 E2 E3 E4" ;;
    all)    IDS="A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 B6 C1 C2 C3 D1a D1b D2a D2b E1 E2 E3 E4" ;;
    custom) IDS="${QUESTIONS//,/ }" ;;
    *)      IDS="${QUESTIONS//,/ }" ;;
esac

# Parse comma-separated lists
IFS=',' read -ra MODE_LIST  <<< "$MODES"
IFS=',' read -ra KR_LIST    <<< "$KRS"
IFS=',' read -ra POS_LIST   <<< "$POSITIONS"

TOTAL=$(($(echo $IDS | wc -w) * ${#MODE_LIST[@]} * ${#KR_LIST[@]} * ${#POS_LIST[@]} * RUNS))
echo "=== Phase 7 Tier: $TIER | $TOTAL runs ==="
COUNT=0
for QID in $IDS; do
    for MODE in "${MODE_LIST[@]}"; do
        for KR in "${KR_LIST[@]}"; do
            for POS in "${POS_LIST[@]}"; do
                for ((R=0; R<RUNS; R++)); do
                    COUNT=$((COUNT + 1))
                    printf "[%d/%d] " "$COUNT" "$TOTAL"
                    bash "$SCRIPT_DIR/run_q.sh" "$QID" "$MODE" "$KR" "$POS" "$R"
                done
            done
        done
    done
done
echo "=== Done ==="
