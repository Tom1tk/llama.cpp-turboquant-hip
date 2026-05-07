#include "common.cuh"
#include <hip/hip_runtime.h>
#include <math.h>

#define BSA_BLOCK_SIZE 128

// Single-query BSA attention: O[h][:] = softmax(Q[h] @ K[h]^T) @ V[h]
// across selected KV blocks. One block per head, iterates KV blocks.
// Uses online softmax for numerical stability.
//
// Q: [n_heads, D]     — single query vector per head
// K: [n_heads, n_kv, D] — key cache
// V: [n_heads, n_kv, D] — value cache
// O: [n_heads, D]     — attention output per head

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
    int stride_kv)
{
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float sdata[BSA_BLOCK_SIZE * 2];

    // Pointers for this head
    const float * Q_head = Q + (size_t)h * D;
    const float * K_head = K + (size_t)h * stride_kv;
    const float * V_head = V + (size_t)h * stride_kv;

    // Online softmax accumulators (register-based for efficiency)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_reg[D];

    #pragma unroll
    for (int d = 0; d < D; d++) {
        O_reg[d] = 0.0f;
    }

    for (int bi = 0; bi < n_selected; bi++) {
        int kv_block = block_mask[bi];
        int k_start  = kv_block * BSA_BLOCK_SIZE;
        int k_end    = min(k_start + BSA_BLOCK_SIZE, n_kv);
        int k_len    = k_end - k_start;

        // Process tokens in this block in chunks
        for (int ki_base = 0; ki_base < k_len; ki_base += blockDim.x) {
            int ki = ki_base + tid;

            // Compute S = Q @ K[ki]^T (scalar per KV token)
            float s_i = -INFINITY;
            if (ki < k_len) {
                float dot = 0.0f;
                const float * k_row = K_head + (size_t)(k_start + ki) * stride_kv;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    dot += Q_head[d] * k_row[d];
                }
                s_i = dot * scale;
            }

            // Online softmax update per token
            float new_m = m_i;
            float exp_prev = 1.0f;
            if (ki < k_len && !isinf(s_i)) {
                if (s_i > new_m) new_m = s_i;
                exp_prev = expf(m_i - new_m);
            }

            // Scale running sum and output
            if (tid == 0) {
                l_i *= exp_prev;
                m_i = new_m;
            }
            __syncthreads();

            // Apply scaling to O_reg
            if (tid == 0 && exp_prev != 1.0f) {
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    O_reg[d] *= exp_prev;
                }
            }
            __syncthreads();

            // Compute P_i = exp(s_i - m_i) and accumulate O
            float p_i = 0.0f;
            if (ki < k_len && !isinf(s_i)) {
                p_i = expf(s_i - m_i);
                l_i += p_i;

                const float * v_row = V_head + (size_t)(k_start + ki) * stride_kv;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    O_reg[d] += p_i * v_row[d];
                }
            }

            __syncthreads();
        }
    }

    // Normalize: O = O / l_i
    float l_normalized = (tid == 0) ? 1.0f / fmaxf(l_i, 1e-12f) : 0.0f;
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        O[(size_t)h * D + d] = O_reg[d] * l_normalized;
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
    int stride_kv)
{
    if (!d_Q || !d_K || !d_V || !d_block_mask || !d_O) return -1;
    if (n_selected <= 0 || n_heads <= 0 || n_kv <= 0 || head_dim <= 0) return -1;

    int threads = min(256, head_dim);
    if (threads < 32) threads = 32;

    // Dispatch based on head_dim
    switch (head_dim) {
        case 64:
            hipLaunchKernelGGL((pflash_bsa_single_query<64>),
                dim3(n_heads), dim3(threads), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale, n_kv, stride_kv);
            break;
        case 128:
            hipLaunchKernelGGL((pflash_bsa_single_query<128>),
                dim3(n_heads), dim3(threads), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale, n_kv, stride_kv);
            break;
        case 256:
            hipLaunchKernelGGL((pflash_bsa_single_query<256>),
                dim3(n_heads), dim3(threads), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale, n_kv, stride_kv);
            break;
        default:
            return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return -1;

    return 0;
}
