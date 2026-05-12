#!/bin/bash
# Phase 11 chunked validation sweep.
# Each chunk answers one validation-gate question from the Plan.
# Run individually:   bash run_phase11.sh 1
#                      bash run_phase11.sh 2
#                      ...
#                      bash run_phase11.sh 5
# Resume-safe: re-run any chunk any time; completed configs are cached.

set -euo pipefail
CHUNK="${1:?usage: run_phase11.sh <chunk 1-5>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
NC='\033[0m'

log()  { printf "${GRN}[phase11-%s]${NC} %s\n" "$CHUNK" "$*"; }
warn() { printf "${YEL}[phase11-%s] WARN${NC} %s\n" "$CHUNK" "$*"; }

run_tier() {
    # Pass through extra pflash flags via existing env-var mechanism
    export PHASE7_PFLASH_EXTRA="${PFLASH_EXTRA:-}"
    # Encode scoring method in result filename to avoid cross-method cache collision
    # (different methods produce different results for same (qid, mode, kr, pos))
    export PHASE7_METHOD_SUFFIX="${METHOD_SUFFIX:-}"
    # Reduce max-gen to speed up sweep (most answers are <100 tokens)
    export PHASE7_MAXGEN="${PHASE7_MAXGEN:-512}"
    # Longer timeout for slow questions
    export PHASE7_TIMEOUT="${PHASE7_TIMEOUT:-600}"
    bash "$SCRIPT_DIR/run_tier.sh" "$@"
}

# ── Chunk definitions ───────────────────────────────────────────────────
case "$CHUNK" in
  1)
# ┌─ Chunk 1: P2 validation smoke (~6-10 min) ──────────────────────────┐
# │ §5.3 gate: small-ctx mid tier. Verify adaptive anchors + both        │
# │ scoring methods produce valid output. Score debug dumps.             │
# │ Questions: A2, A4, A5 (≤13k tokens, small contexts).                 │
# │ Configs: 3 QIDs × 2 methods (centrality + obs-attn) = 6 + 3 bsl = 9 │
# └──────────────────────────────────────────────────────────────────────┘
    log "=== Chunk 1: P2 smoke — small-ctx mid, BSA, both methods, adaptive anchors ==="

    # BSA mid, centraliy, adaptive anchors on, debug scores
    log "--- BSA mid, centrality, adaptive anchors ---"
    PFLASH_EXTRA="--pflash-score-method centrality --pflash-adaptive-anchors --pflash-debug-scores"
    METHOD_SUFFIX="_centrality"
    run_tier --questions A2,A4,A5 --mode bsa --kr 0.65 --positions mid

    # BSA mid, obs-attn, adaptive anchors on, debug scores
    log "--- BSA mid, obs-attn, adaptive anchors ---"
    PFLASH_EXTRA="--pflash-score-method obs-attn --pflash-adaptive-anchors --pflash-debug-scores"
    METHOD_SUFFIX="_obs-attn"
    run_tier --questions A2,A4,A5 --mode bsa --kr 0.65 --positions mid

    # Baseline (no compression)
    log "--- Baseline (no PFlash) ---"
    METHOD_SUFFIX=""
    run_tier --questions A2,A4,A5 --mode baseline --kr 1.00 --positions mid
    ;;

  2)
# ┌─ Chunk 2: P1 core — BSA mid obs-attn, fast+medium tier (~8-15 min) ─┐
# │ §4.7 condition 1: BSA mid kr=0.65 with obs-attn.                     │
# │ First half (smaller contexts, faster): A1-A5, B2-B6, C3, D1a-D1b,   │
# │ E1-E4. Quick pulse check — if obs-attn is working.                   │
# │ Configs: 17 QIDs = 17                                                │
# └──────────────────────────────────────────────────────────────────────┘
    log "=== Chunk 2: P1 core — BSA mid obs-attn, fast+medium tier (17 questions) ==="
    PFLASH_EXTRA="--pflash-score-method obs-attn --pflash-coverage-zones 4"
    METHOD_SUFFIX="_obs-attn"
    run_tier --questions A1,A2,A3,A4,A5,B2,B3,B4,B5,B6,C3,D1a,D1b,E1,E2,E3,E4 \
             --mode bsa --kr 0.65 --positions mid
    ;;

  3)
# ┌─ Chunk 3: P1 core — BSA mid obs-attn, slow tier (~10-20 min) ───────┐
# │ §4.7 condition 1 continued: remaining 5 slow questions + centrality  │
# │ baseline on ALL 22 for per-question A/B diff.                        │
# │ Slow: B1, C1, C2, D2a, D2b (≥16k tokens)                            │
# │ Configs: 5 obs-attn + 22 centrality = 27                            │
# └──────────────────────────────────────────────────────────────────────┘
    log "=== Chunk 3: BSA mid obs-attn (slow tier) + BSA mid centrality (all, A/B baseline) ==="

    # Slow tier obs-attn
    log "--- Slow tier obs-attn ---"
    PFLASH_EXTRA="--pflash-score-method obs-attn --pflash-coverage-zones 4"
    METHOD_SUFFIX="_obs-attn"
    run_tier --questions B1,C1,C2,D2a,D2b \
             --mode bsa --kr 0.65 --positions mid

    # Full mid centrality baseline (Phase 10 comparison)
    log "--- All mid BSA centrality (Phase 10 A/B baseline) ---"
    PFLASH_EXTRA="--pflash-score-method centrality --pflash-coverage-zones 4"
    METHOD_SUFFIX="_centrality"
    run_tier --tier all --mode bsa --kr 0.65 --positions mid
    ;;

  4)
# ┌─ Chunk 4: Early regression — BSA obs-attn (~12-20 min) ─────────────┐
# │ §4.7 condition 2: early positions must not regress (≥33/34).         │
# │ Configs: 22 QIDs = 22                                               │
# └──────────────────────────────────────────────────────────────────────┘
    log "=== Chunk 4: BSA early obs-attn — regression check ==="
    PFLASH_EXTRA="--pflash-score-method obs-attn --pflash-coverage-zones 4"
    METHOD_SUFFIX="_obs-attn_early"
    run_tier --tier all --mode bsa --kr 0.65 --positions early
    ;;

  5)
# ┌─ Chunk 5: Late regression + windowed sanity (~20-30 min) ────────────┐
# │ §4.7 condition 2: late positions. §4.7 condition 3: windowed mid    │
# │ sanity (obs-attn should not regress windowed mode).                  │
# │ Configs: 22 late + 22 windowed mid = 44                             │
# └──────────────────────────────────────────────────────────────────────┘
    log "=== Chunk 5: BSA late obs-attn + windowed mid obs-attn ==="

    # Late regression
    log "--- BSA late obs-attn ---"
    PFLASH_EXTRA="--pflash-score-method obs-attn --pflash-coverage-zones 4"
    METHOD_SUFFIX="_obs-attn_late"
    run_tier --tier all --mode bsa --kr 0.65 --positions late

    # Windowed mid sanity
    log "--- Windowed mid obs-attn ---"
    PHASE7_MAXGEN=512
    PFLASH_EXTRA="--pflash-score-method obs-attn --pflash-coverage-zones 4"
    METHOD_SUFFIX="_obs-attn_windowed"
    run_tier --tier all --mode windowed --kr 0.65 --positions mid
    ;;

  *)
    echo "Unknown chunk: $CHUNK (expected 1-5)" >&2
    echo ""
    echo "Phase 11 Chunked Validation Schedule:"
    echo "  Chunk 1 (6-10 min):  P2 smoke — small-ctx mid, both methods, adaptive anchors"
    echo "  Chunk 2 (8-15 min):  P1 core — BSA mid obs-attn, fast+medium tier"
    echo "  Chunk 3 (10-20 min): P1 core — BSA mid obs-attn slow tier + centrality A/B"
    echo "  Chunk 4 (12-20 min): Early regression — BSA early obs-attn"
    echo "  Chunk 5 (20-30 min): Late regression + windowed sanity"
    echo ""
    echo "  Total: ~55-95 min across 5 independently-runnable chunks"
    exit 1
    ;;
esac

log "=== Chunk $CHUNK complete ==="
# Quick summary for this chunk
python3 "$SCRIPT_DIR/aggregate_results.py" 2>/dev/null || true
