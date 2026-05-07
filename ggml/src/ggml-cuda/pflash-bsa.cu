#include "common.cuh"
#include <hip/hip_runtime.h>
#include <math.h>

#define BSA_BLOCK_SIZE 128
#define BSA_TILED_THREADS 256

// ============================================================================
// Single-query BSA (decode: 1 token per sequence)
// ============================================================================

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

    float O_reg[D / 128 + 1] = {0.0f};

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int bi = 0; bi < n_selected; bi++) {
        int kv_block = block_mask[bi];
        int k_start  = kv_block * BSA_BLOCK_SIZE;
        int k_end    = min(k_start + BSA_BLOCK_SIZE, n_kv);

        for (int ki = k_start; ki < k_end; ki++) {
            const size_t row_off = (size_t)ki * kv_stride;
            const float * k_row = K + h_off + row_off;

            float partial = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) {
                partial += Q_head[d] * k_row[d];
            }

            sdata[tid] = partial;
            __syncthreads();

            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset) sdata[tid] += sdata[tid + offset];
                __syncthreads();
            }
            float s = sdata[0] * scale;

            float m_old = m_i;
            float m_new = fmaxf(m_old, s);
            float exp_scale = expf(m_old - m_new);

            for (int d = tid; d < D; d += blockDim.x) {
                O_reg[d / blockDim.x] *= exp_scale;
            }

            float p = expf(s - m_new);
            l_i = l_i * exp_scale + p;
            m_i = m_new;

            const float * v_row = V + h_off + row_off;
            for (int d = tid; d < D; d += blockDim.x) {
                O_reg[d / blockDim.x] += p * v_row[d];
            }
        }
    }

    float inv_l = 1.0f / fmaxf(l_i, 1e-12f);
    for (int d = tid; d < D; d += blockDim.x) {
        O[(size_t)h * D + d] = O_reg[d / blockDim.x] * inv_l;
    }
}

// ============================================================================
// Tiled multi-query BSA (prefill: N tokens at once)
//
// Design: one thread per Q row (threads 0..BQ-1). Each thread:
//   Pass 1: Load K, compute S_ij, find max, rescale O, store S in registers
//   Pass 2: Load V, compute P_ij from S_ij, accumulate O
//   After all blocks: normalize and write out
//
// Register usage per thread (for D=64, BKV=64):
//   S_reg[64] = 64 floats (dot products)
//   O_reg[64] = 64 floats (output accumulator)
//   m_reg, l_reg, scale, etc. = ~10 floats
//   Total: ~138 registers — fine for RDNA3 (256 VGPR limit)
//
// Shared memory:
//   Q_tile[BQ][D+1]   (16640 bytes)
//   KV_tile[BKV][D+1] (16640 bytes, reused K→V)
//   O_tile[BQ][D+1]   (16640 bytes)
//   m_tile[BQ]         (256 bytes)
//   l_tile[BQ]         (256 bytes)
//   Total: 50,432 bytes
// ============================================================================

template <int D, int BQ, int BKV>
static __global__ void pflash_bsa_tiled(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    const int * __restrict__ block_mask,
    int n_selected,
    float * __restrict__ O,
    float scale,
    int n_q,
    int n_kv,
    int n_heads,
    int n_heads_kv,
    int q_stride,
    int q_head_stride,
    int kv_stride,
    int kv_head_stride)
{
    const int q_tile_idx = blockIdx.x;
    const int h          = blockIdx.y;
    const int tid        = threadIdx.x;

    const int q_tile_start = q_tile_idx * BQ;
    const int q_tile_end   = min(q_tile_start + BQ, n_q);
    const int n_q_local    = q_tile_end - q_tile_start;
    const int qi = tid;  // thread tid handles Q row qi (only for tid < n_q_local)

    const int h_kv = h * n_heads_kv / n_heads;

    __shared__ float Q_tile  [BQ][D + 1];
    __shared__ float KV_tile [BKV][D + 1];
    __shared__ float O_tile  [BQ][D + 1];
    __shared__ float m_tile  [BQ];
    __shared__ float l_tile  [BQ];

    int global_q_pos = -1;
    float S_reg[BKV];
    float O_reg[D];
    float m_reg, l_reg;

    if (qi < n_q_local) {
        global_q_pos = q_tile_start + qi;

        // Load Q row into shared (cooperative)
        const float * q_row = Q + (size_t)global_q_pos * q_stride + (size_t)h * q_head_stride;
        for (int d = 0; d < D; d++) {
            Q_tile[qi][d] = q_row[d];
        }

        // Init accumulators
        m_reg = -INFINITY;
        l_reg = 0.0f;
        for (int d = 0; d < D; d++) {
            O_reg[d] = 0.0f;
        }
    }
    // Extra threads help load Q_tile for rows beyond BQ via coop loading
    for (int r = tid; r < n_q_local; r += BSA_TILED_THREADS) {
        if (r >= BQ) {
            const float * q_row2 = Q + (size_t)(q_tile_start + r) * q_stride + (size_t)h * q_head_stride;
            for (int d = 0; d < D; d++) {
                Q_tile[r][d] = q_row2[d];
            }
        }
    }
    __syncthreads();

    // Step 2: Iterate over selected KV blocks
    for (int bi = 0; bi < n_selected; bi++) {
        int kv_block = block_mask[bi];
        int k_start  = kv_block * BKV;
        int k_end    = min(k_start + BKV, n_kv);
        int n_kv_local = k_end - k_start;

        if (n_kv_local <= 0) continue;

        // Step 2a: Load K block into KV_tile
        for (int ki = tid; ki < n_kv_local; ki += BSA_TILED_THREADS) {
            const int global_ki = k_start + ki;
            const float * k_row = K + (size_t)global_ki * kv_stride + (size_t)h_kv * kv_head_stride;
            for (int d = 0; d < D; d++) {
                KV_tile[ki][d] = k_row[d];
            }
        }
        __syncthreads();

        // Step 2b: Compute S_ij, find max, rescale O (PASS 1)
        if (qi < n_q_local) {
            float m_old = m_reg;

            for (int ki = 0; ki < n_kv_local; ki++) {
                int global_kv_pos = k_start + ki;

                if (global_kv_pos > global_q_pos) {
                    S_reg[ki] = -INFINITY;
                } else {
                    float s = 0.0f;
                    for (int d = 0; d < D; d++) {
                        s += Q_tile[qi][d] * KV_tile[ki][d];
                    }
                    S_reg[ki] = s * scale;
                }

                m_reg = fmaxf(m_reg, S_reg[ki]);
            }

            // Rescale running accumulators
            float exp_scale = expf(m_old - m_reg);
            if (exp_scale < 1.0f) {
                for (int d = 0; d < D; d++) {
                    O_reg[d] *= exp_scale;
                }
            }
            l_reg *= exp_scale;
        }
        __syncthreads();

        // Store m_tile for normalization step
        if (qi < n_q_local) {
            m_tile[qi] = m_reg;
        }
        __syncthreads();

        // Step 2c: Load V block into KV_tile (reuse buffer)
        for (int ki = tid; ki < n_kv_local; ki += BSA_TILED_THREADS) {
            const int global_ki = k_start + ki;
            const float * v_row = V + (size_t)global_ki * kv_stride + (size_t)h_kv * kv_head_stride;
            for (int d = 0; d < D; d++) {
                KV_tile[ki][d] = v_row[d];
            }
        }
        __syncthreads();

        // Step 2d: Compute P_ij and accumulate O (PASS 2)
        if (qi < n_q_local) {
            for (int ki = 0; ki < n_kv_local; ki++) {
                float s = S_reg[ki];
                if (s == -INFINITY) continue;

                float p = expf(s - m_reg);
                l_reg += p;

                for (int d = 0; d < D; d++) {
                    O_reg[d] += p * KV_tile[ki][d];
                }
            }
        }
        __syncthreads();
    }

    // Step 4: Write results to shared memory for normalization
    if (qi < n_q_local) {
        l_tile[qi] = l_reg;
        for (int d = 0; d < D; d++) {
            O_tile[qi][d] = O_reg[d];
        }
    }
    __syncthreads();

    // Step 5: Normalize in shared memory
    if (qi < n_q_local) {
        float inv_l = 1.0f / fmaxf(l_tile[qi], 1e-12f);
        for (int d = 0; d < D; d++) {
            O_tile[qi][d] *= inv_l;
        }
    }
    __syncthreads();

    // Step 6: Write output to global memory
    if (qi < n_q_local) {
        float * o_row = O + (size_t)global_q_pos * q_stride + (size_t)h * q_head_stride;
        for (int d = 0; d < D; d++) {
            o_row[d] = O_tile[qi][d];
        }
    }
}

// ============================================================================
// Host-side launchers
// ============================================================================

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

    int threads_p2 = 1;
    while (threads_p2 < threads) threads_p2 *= 2;
    if (threads_p2 > 256) threads_p2 = 256;

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

int32_t pflash_bsa_tiled_forward(
    const float * d_Q,
    const float * d_K,
    const float * d_V,
    const int * d_block_mask,
    int n_selected,
    float * d_O,
    float scale,
    int n_heads,
    int n_heads_kv,
    int n_q,
    int n_kv,
    int head_dim,
    int q_stride,
    int q_head_stride,
    int kv_stride,
    int kv_head_stride)
{
    if (!d_Q || !d_K || !d_V || !d_block_mask || !d_O) return -1;
    if (n_selected <= 0 || n_heads <= 0 || n_kv <= 0 || head_dim <= 0 || n_q <= 0) return -1;

    const int BQ_VAL = 64;
    const int BKV_VAL = 64;

    int n_q_tiles = (n_q + BQ_VAL - 1) / BQ_VAL;

    switch (head_dim) {
        case 64:
            hipLaunchKernelGGL((pflash_bsa_tiled<64, BQ_VAL, BKV_VAL>),
                dim3(n_q_tiles, n_heads), dim3(BSA_TILED_THREADS), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale,
                n_q, n_kv, n_heads, n_heads_kv,
                q_stride, q_head_stride, kv_stride, kv_head_stride);
            break;
        case 128:
            hipLaunchKernelGGL((pflash_bsa_tiled<128, BQ_VAL, BKV_VAL>),
                dim3(n_q_tiles, n_heads), dim3(BSA_TILED_THREADS), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale,
                n_q, n_kv, n_heads, n_heads_kv,
                q_stride, q_head_stride, kv_stride, kv_head_stride);
            break;
        case 256:
            hipLaunchKernelGGL((pflash_bsa_tiled<256, BQ_VAL, BKV_VAL>),
                dim3(n_q_tiles, n_heads), dim3(BSA_TILED_THREADS), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale,
                n_q, n_kv, n_heads, n_heads_kv,
                q_stride, q_head_stride, kv_stride, kv_head_stride);
            break;
        default:
            return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return -1;

    return 0;
}

// ============================================================================
// ggml_cuda wrapper — dispatches to tiled (n_q > 1) or single-query (n_q == 1)
// ============================================================================

void ggml_cuda_pflash_bsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q          = dst->src[0];
    const ggml_tensor * k          = dst->src[1];
    const ggml_tensor * v          = dst->src[2];
    const ggml_tensor * block_mask = dst->src[3];

    GGML_UNUSED(ctx);

    const int n_heads    = q->ne[2];
    const int n_heads_kv = k->ne[2];
    const int n_q        = q->ne[1];
    const int n_kv       = k->ne[1];
    const int head_dim   = q->ne[0];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int n_selected = block_mask->ne[0];

    const int q_stride       = q->nb[1] / sizeof(float);
    const int q_head_stride  = q->nb[2] / sizeof(float);
    const int kv_stride      = k->nb[1] / sizeof(float);
    const int kv_head_stride = k->nb[2] / sizeof(float);

    if (n_q == 1) {
        GGML_ASSERT(q->nb[1] / sizeof(float) == head_dim ||
                    q_stride == head_dim);
        pflash_bsa_forward(
            (const float*)q->data, (const float*)k->data, (const float*)v->data,
            (const int*)block_mask->data, n_selected,
            (float*)dst->data, scale, n_heads, n_kv, head_dim, kv_stride);
    } else {
        pflash_bsa_tiled_forward(
            (const float*)q->data, (const float*)k->data, (const float*)v->data,
            (const int*)block_mask->data, n_selected,
            (float*)dst->data, scale, n_heads, n_heads_kv, n_q, n_kv, head_dim,
            q_stride, q_head_stride, kv_stride, kv_head_stride);
    }

    GGML_ASSERT(hipGetLastError() == hipSuccess);
}

bool ggml_cuda_pflash_bsa_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    return dst->src[0] != nullptr &&
           dst->src[0]->type == GGML_TYPE_F32 &&
           dst->src[1]->type == GGML_TYPE_F32 &&
           dst->src[2]->type == GGML_TYPE_F32 &&
           dst->src[3]->type == GGML_TYPE_I32;
}
