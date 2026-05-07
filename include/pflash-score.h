#pragma once

#include <stdint.h>

// GPU-accelerated PFlash block scoring
// K tensor must be a GPU-resident ggml_tensor from llama_kv_cache_get_k_tensor
// Returns number of scores written, or 0 on error
int32_t pflash_score_gpu(
    const float * d_k_data,    // device pointer to K tensor data [n_tokens, kv_dim]
    int32_t n_tokens,
    int32_t kv_dim,
    int32_t block_size,
    float * scores_out);       // host buffer for n_blocks scores
