#include "ggml-cuda.h"
#include "ggml-backend.h"

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

// Compute mean K vector for each block
// Grid: n_blocks, each block computes mean_K for all kv_dim dimensions
// Block: up to 256 threads, each thread handles 1 dimension
static __global__ void pflash_mean_k_kernel(
    const float * __restrict__ K,
    float * __restrict__ mean_K,
    int n_tokens,
    int kv_dim,
    int block_size)
{
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= kv_dim) return;

    int start = b * block_size;
    int end = min(start + block_size, n_tokens);
    int n = end - start;

    float sum = 0.0f;
    for (int p = start; p < end; p++) {
        sum += K[(size_t)p * kv_dim + d];
    }
    mean_K[(size_t)b * kv_dim + d] = sum / (float)n;
}

// Compute cosine similarity scores from mean_K and last_K
// Grid: n_blocks, each block computes one score
// Uses shared memory for reduction across kv_dim dimensions
static __global__ void pflash_score_kernel(
    const float * __restrict__ mean_K,
    const float * __restrict__ last_K,
    float * __restrict__ scores,
    int n_blocks,
    int kv_dim)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[256];

    // Load last_K into shared (each thread loads 1 element)
    if (tid < kv_dim) sdata[tid] = last_K[tid];
    __syncthreads();

    float dot = 0.0f, nrm = 0.0f;
    const float * mk = &mean_K[(size_t)b * kv_dim];

    for (int d = tid; d < kv_dim; d += blockDim.x) {
        dot += mk[d] * sdata[d];
        nrm += mk[d] * mk[d];
    }

    // Parallel reduction: dot
    sdata[tid] = dot;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    float block_dot = sdata[0];

    // Parallel reduction: nrm
    sdata[tid] = nrm;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    float block_nrm = sdata[0];

    // Parallel reduction: last_K norm (re-read from global, sdata was overwritten above)
    sdata[tid] = 0.0f;
    for (int d = tid; d < kv_dim; d += blockDim.x) {
        float val = last_K[d];
        sdata[tid] += val * val;
    }
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    float last_nrm = sdata[0];

    if (tid == 0) {
        float denom = sqrtf(fmaxf(block_nrm, 1e-12f)) * sqrtf(fmaxf(last_nrm, 1e-12f));
        scores[b] = block_dot / denom;
    }
}

int32_t pflash_score_gpu(
    const float * d_k_data,
    int32_t n_tokens,
    int32_t kv_dim,
    int32_t block_size,
    float * scores_out)
{
    if (!d_k_data || n_tokens <= 0 || kv_dim <= 0 || block_size <= 0 || !scores_out) {
        return 0;
    }

    int32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    int threads = kv_dim < 256 ? (int)kv_dim : 256;

    // Round up to power of 2 for parallel reduction correctness
    int threads_p2 = 1;
    while (threads_p2 < threads) threads_p2 *= 2;
    if (threads_p2 > 256) threads_p2 = 256;

    float * d_mean_K = nullptr;
    float * d_scores = nullptr;
    hipError_t err;

    err = hipMalloc(&d_mean_K, (size_t)n_blocks * kv_dim * sizeof(float));
    if (err != hipSuccess) return 0;
    err = hipMalloc(&d_scores, (size_t)n_blocks * sizeof(float));
    if (err != hipSuccess) { hipFree(d_mean_K); return 0; }

    const float * d_last_K = d_k_data + (size_t)(n_tokens - 1) * kv_dim;

    dim3 grid_dim(n_blocks);
    dim3 block_dim_mean(threads < 64 ? 64 : threads);
    hipLaunchKernelGGL(pflash_mean_k_kernel, grid_dim, block_dim_mean, 0, 0,
        d_k_data, d_mean_K, n_tokens, kv_dim, block_size);

    dim3 block_dim_score(threads_p2);
    hipLaunchKernelGGL(pflash_score_kernel, grid_dim, block_dim_score, 0, 0,
        d_mean_K, d_last_K, d_scores, n_blocks, kv_dim);

    err = hipMemcpy(scores_out, d_scores, (size_t)n_blocks * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        hipFree(d_mean_K);
        hipFree(d_scores);
        return 0;
    }

    hipFree(d_mean_K);
    hipFree(d_scores);

    return n_blocks;
}
