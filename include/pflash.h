#pragma once

#include "llama.h"

#include <vector>
#include <string>
#include <utility>

struct pflash_params {
    // Path to the draft model (small GGUF)
    std::string draft_model_path;

    // Fraction of tokens to keep
    float keep_ratio = 0.75f;

    // Minimum number of tokens to always keep
    int32_t min_keep_tokens = 2048;

    // Number of prefix tokens to always keep (sink tokens)
    int32_t sink_tokens = 2048;

    // Number of suffix tokens to always keep (recent tokens)
    int32_t recent_tokens = 4096;

    // Scoring block size
    int32_t block_size = 128;

    // Scoring layer index (negative = auto-select)
    int32_t score_layer = -1;

    // Minimum source tokens to apply PFlash
    int32_t threshold_tokens = 8192;

    // KV cache types for the drafter
    std::string draft_cache_type_k = "f16";
    std::string draft_cache_type_v = "f16";

    // Chunked window size for Phases 2-3 (0 = disabled, process full prompt)
    int32_t window_size = 4096;

    // GPU layers for draft model (-1 = all, 0 = CPU only, N = first N layers)
    int32_t draft_gpu_layers = -1;

    // Use Block-Sparse Attention for drafter (Phase 5C+)
    bool use_bsa = false;

    // Auto-mode: BSA single-pass if n_tokens <= threshold, windowed otherwise
    // 0 = manual (user controls window_size directly)
    int32_t bsa_auto_threshold = 50000;

    // Adaptive keep ratio: auto-adjust based on context size
    // When true, keep_ratio is derived from context size bands:
    //   n < 25k → 0.80   (anchors dominate; scoring barely activates)
    //   n < 64k → 0.70   (moderate scoring utility)
    //   n ≥ 64k → 0.65   (full scoring benefit)
    bool keep_ratio_auto = false;

    // Minimum scoring budget: skip draft pass when scoring_budget < this value
    // Prevents wasting a GPU draft pass when most kept tokens are anchors
    int32_t min_scoring_budget = 2048;
};

struct pflash_span {
    int32_t start;
    int32_t end; // exclusive
};

struct pflash_result {
    std::vector<llama_token> tokens;
    std::vector<pflash_span> spans;
    int32_t source_count = 0;
    int32_t kept_count = 0;
    int64_t score_us = 0;
    int64_t select_us = 0;
    int64_t gather_us = 0;
    int64_t draft_us = 0;
    bool bypassed = false;
};

// Initialize a draft model context from a GGUF file
// n_gpu_layers: number of layers to offload to GPU (-1 = all, 0 = CPU only)
// Returns a new llama_context configured for the drafter
struct llama_context * pflash_init_draft(
    const std::string & model_path,
    int32_t n_ctx,
    const std::string & cache_type_k,
    const std::string & cache_type_v,
    int32_t n_gpu_layers = -1,
    bool use_bsa = false);

// Select which spans to keep based on scores and configuration
std::vector<pflash_span> pflash_select(
    const std::vector<float> & scores,
    int32_t block_size,
    int32_t n_tokens,
    int32_t sink_tokens,
    int32_t recent_tokens,
    float keep_ratio,
    int32_t min_keep_tokens);

// Gather the selected tokens from the source tokens
std::vector<llama_token> pflash_gather(
    const std::vector<llama_token> & source,
    const std::vector<pflash_span> & spans);

// Full PFlash compression pipeline
// Returns compressed token sequence (or original if bypassed)
pflash_result pflash_compress(
    struct llama_context * draft_ctx,
    const std::vector<llama_token> & tokens,
    const pflash_params & params);
