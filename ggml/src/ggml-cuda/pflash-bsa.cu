#include "common.cuh"
#include <hip/hip_runtime.h>
#include <math.h>

#define BSA_BLOCK_SIZE 128

// Single-query BSA attention: O[h][:] = softmax(Q[h] @ K[h]^T) @ V[h]
// across selected KV blocks. One block per head, iterates KV blocks.
// Uses online softmax with cooperative dot-product reduction.
//
// Each thread handles one output dimension (for D <= blockDim.x, the common case).
// All threads cooperate on Q·K dot product via shared-memory reduction.
// m_i and l_i are consistent across threads since all read the same reduced s.
//
// Q: [n_heads, D]     — single query vector per head
// K: [n_heads, n_kv, D] — key cache, interleaved layout via head_dim and kv_stride
// V: [n_heads, n_kv, D] — value cache, same layout as K
// O: [n_heads, D]     — attention output per head
//
// Memory layout: K[head h, token ki, dim d] at offset:
//   h * head_dim + ki * kv_stride + d
// where head_dim < kv_stride (kv_stride = n_embd_k_gqa, includes all heads).

template <int D>
static __global__ void pflash_bsa_single_query(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    const int * __restrict__ block_mask,
    int n_selected,
    float * __restrict__ O,
    float scale,
    int n_kv,
    int kv_stride,
    int head_dim)
{
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;

    const size_t h_off = (size_t)h * head_dim;

    __shared__ float sdata[BSA_BLOCK_SIZE];

    const float * Q_head = Q + (size_t)h * D;

    // Output accumulators — each thread handles dimensions [tid, tid+blockDim.x, ...)
    // For D <= blockDim.x, each thread handles exactly 0 or 1 dimensions
    float O_reg[D / 128 + 1] = {0.0f};  // upper bound for any blockDim.x >= 32

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int bi = 0; bi < n_selected; bi++) {
        int kv_block = block_mask[bi];
        int k_start  = kv_block * BSA_BLOCK_SIZE;
        int k_end    = min(k_start + BSA_BLOCK_SIZE, n_kv);

        for (int ki = k_start; ki < k_end; ki++) {
            // Cooperative dot product Q·K[ki]
            const size_t row_off = (size_t)ki * kv_stride;
            const float * k_row = K + h_off + row_off;

            float partial = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) {
                partial += Q_head[d] * k_row[d];
            }

            sdata[tid] = partial;
            __syncthreads();

            // Reduction: sum partial dot across threads
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset) sdata[tid] += sdata[tid + offset];
                __syncthreads();
            }
            float s = sdata[0] * scale;

            // Online softmax — all threads compute same m_i, l_i from the reduced s
            float m_old = m_i;
            float m_new = fmaxf(m_old, s);
            float exp_scale = expf(m_old - m_new);

            // De-scale running output
            for (int d = tid; d < D; d += blockDim.x) {
                O_reg[d / blockDim.x] *= exp_scale;
            }

            float p = expf(s - m_new);
            l_i = l_i * exp_scale + p;
            m_i = m_new;

            // Accumulate: O += p * V[ki]
            const float * v_row = V + h_off + row_off;
            for (int d = tid; d < D; d += blockDim.x) {
                O_reg[d / blockDim.x] += p * v_row[d];
            }
        }
    }

    // Normalize and write output
    float inv_l = 1.0f / fmaxf(l_i, 1e-12f);
    for (int d = tid; d < D; d += blockDim.x) {
        O[(size_t)h * D + d] = O_reg[d / blockDim.x] * inv_l;
    }
}

// Public entry point — dispatches to templated kernel based on head_dim
int32_t pflash_bsa_forward(
    const float * d_Q,
    const float * d_K,
    const float * d_V,
    const int * d_block_mask,
    int n_selected,
    float * d_O,
    float scale,
    int n_heads,
    int n_kv,
    int head_dim,
    int kv_stride)
{
    if (!d_Q || !d_K || !d_V || !d_block_mask || !d_O) return -1;
    if (n_selected <= 0 || n_heads <= 0 || n_kv <= 0 || head_dim <= 0) return -1;

    int threads = min(256, head_dim);
    if (threads < 32) threads = 32;

    // Round up to power of 2 for reduction safety
    int threads_p2 = 1;
    while (threads_p2 < threads) threads_p2 *= 2;
    if (threads_p2 > 256) threads_p2 = 256;

    // Dispatch based on head_dim
    switch (head_dim) {
        case 64:
            hipLaunchKernelGGL((pflash_bsa_single_query<64>),
                dim3(n_heads), dim3(threads_p2), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale, n_kv, kv_stride, head_dim);
            break;
        case 128:
            hipLaunchKernelGGL((pflash_bsa_single_query<128>),
                dim3(n_heads), dim3(threads_p2), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale, n_kv, kv_stride, head_dim);
            break;
        case 256:
            hipLaunchKernelGGL((pflash_bsa_single_query<256>),
                dim3(n_heads), dim3(threads_p2), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale, n_kv, kv_stride, head_dim);
            break;
        default:
            return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return -1;

    return 0;
}
