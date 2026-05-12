#pragma once

#include <stdint.h>

// GPU-accelerated PFlash block scoring (centrality method)
// K tensor must be a GPU-resident ggml_tensor from llama_kv_cache_get_k_tensor
// Returns number of scores written, or 0 on error
int32_t pflash_score_gpu(
    const float * d_k_data,    // device pointer to K tensor data [n_tokens, kv_dim]
    int32_t n_tokens,
    int32_t kv_dim,
    int32_t block_size,
    float * scores_out);       // host buffer for n_blocks scores

// GPU-accelerated observation-window attention scoring (SnapKV-style)
// Compute proxy_q = mean(K[tokens-W:tokens]), then per-token attention scores,
// then per-block max after SnapKV pooling.
// Returns number of block scores written, or 0 on error.
int32_t pflash_score_gpu_obs_attn(
    const float * d_k_data,
    int32_t n_tokens,
    int32_t kv_dim,
    int32_t block_size,
    int32_t obs_window,
    int32_t pool_kernel,
    float * scores_out);
