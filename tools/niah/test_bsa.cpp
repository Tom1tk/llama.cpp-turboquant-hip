#include "llama.h"
#include "log.h"

#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#ifdef GGML_USE_HIP
extern "C" {
int32_t pflash_bsa_forward(
    const float * d_Q, const float * d_K, const float * d_V,
    const int * d_block_mask, int n_selected,
    float * d_O, float scale,
    int n_heads, int n_heads_kv, int n_q, int n_kv, int head_dim,
    int q_stride, int q_head_stride, int o_stride, int o_head_stride,
    int kv_stride, int kv_head_stride);
}
#endif

static void fill_random(float * data, size_t n, float scale = 1.0f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

static float max_rel_error(const std::vector<float> & a, const std::vector<float> & b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float denom = std::max(std::abs(a[i]), std::abs(b[i]));
        if (denom > 1e-6f) {
            float rel = std::abs(a[i] - b[i]) / denom;
            if (rel > max_err) max_err = rel;
        }
    }
    return max_err;
}

// CPU reference: full attention with causal mask + softmax
static std::vector<float> cpu_attention(
    const std::vector<float> & Q, // [n_q, n_heads, D] row-major
    const std::vector<float> & K, // [n_kv, n_heads, D] row-major
    const std::vector<float> & V, // [n_kv, n_heads, D] row-major
    int n_q, int n_heads, int n_kv, int D,
    float scale)
{
    std::vector<float> O((size_t)n_q * n_heads * D, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        for (int qi = 0; qi < n_q; qi++) {
            // Compute all attention scores S[ki]
            std::vector<float> S(n_kv, -INFINITY);
            float m_max = -INFINITY;

            for (int ki = 0; ki <= qi; ki++) {  // causal mask
                float s = 0.0f;
                for (int d = 0; d < D; d++) {
                    s += Q[((size_t)qi * n_heads + h) * D + d] *
                         K[((size_t)ki * n_heads + h) * D + d];
                }
                S[ki] = s * scale;
                if (S[ki] > m_max) m_max = S[ki];
            }

            // Online softmax
            float l = 0.0f;
            float * out = &O[((size_t)qi * n_heads + h) * D];
            for (int d = 0; d < D; d++) out[d] = 0.0f;

            for (int ki = 0; ki <= qi; ki++) {
                if (S[ki] != -INFINITY) {
                    float p = expf(S[ki] - m_max);
                    l += p;
                    for (int d = 0; d < D; d++) {
                        out[d] += p * V[((size_t)ki * n_heads + h) * D + d];
                    }
                }
            }

            l = std::max(l, 1e-12f);
            for (int d = 0; d < D; d++) out[d] /= l;
        }
    }
    return O;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    (void)argc; (void)argv;

    printf("=== BSA Kernel Unit Test ===\n");

    const int D = 256;        // head_dim (matching Qwen3.5-0.8B)
    const int n_q = 16;       // query tokens
    const int n_kv = 256;     // KV tokens
    const int n_heads = 8;    // attention heads
    const int n_heads_kv = 8; // no GQA for unit test (CPU reference doesn't handle GQA)
    const float attn_scale = 1.0f / sqrtf((float)D);

    const int BSA_BLOCK = 128;
    const int n_blocks = (n_kv + BSA_BLOCK - 1) / BSA_BLOCK;

    // Allocate host memory
    const size_t q_size = (size_t)n_q * n_heads * D;
    const size_t kv_size = (size_t)n_kv * n_heads_kv * D;
    std::vector<float> h_Q(q_size);
    std::vector<float> h_K(kv_size);
    std::vector<float> h_V(kv_size);
    std::vector<float> h_O(q_size);

    // Contiguous strides: Q[qi][h][d], K[ki][h][d]
    int q_stride = n_heads * D;
    int q_head_stride = D;
    int kv_stride = n_heads_kv * D;
    int kv_head_stride = D;

    fill_random(h_Q.data(), q_size, 1.0f);
    fill_random(h_K.data(), kv_size, 1.0f);
    fill_random(h_V.data(), kv_size, 1.0f);

    // CPU reference
    auto cpu_O = cpu_attention(h_Q, h_K, h_V, n_q, n_heads, n_kv, D, attn_scale);

    // Block mask: attend to ALL KV blocks (full attention)
    std::vector<int32_t> block_mask(n_blocks);
    for (int b = 0; b < n_blocks; b++) block_mask[b] = b;

#ifdef GGML_USE_HIP
    // GPU: allocate and copy
    float *d_Q, *d_K, *d_V, *d_O;
    int *d_mask;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_O, q_size * sizeof(float));
    hipMalloc(&d_mask, n_blocks * sizeof(int32_t));

    hipMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_mask, block_mask.data(), n_blocks * sizeof(int32_t), hipMemcpyHostToDevice);

    int ret = pflash_bsa_forward(
        d_Q, d_K, d_V, d_mask, n_blocks, d_O, attn_scale,
        n_heads, n_heads_kv, n_q, n_kv, D,
        q_stride, q_head_stride, q_stride, q_head_stride,
        kv_stride, kv_head_stride);

    if (ret != 0) {
        fprintf(stderr, "FAIL: pflash_bsa_forward returned %d\n", ret);
        return 1;
    }

    hipDeviceSynchronize();
    hipMemcpy(h_O.data(), d_O, q_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O); hipFree(d_mask);
#else
    fprintf(stderr, "GPU not available, skipping GPU test\n");
    return 0;
#endif

    // Compare
    float max_err = max_rel_error(cpu_O, h_O);
    printf("  Max relative error: %.6f\n", max_err);

    const float tol = 1e-3f;
    if (max_err <= tol) {
        printf("  PASS: BSA matches CPU reference within tolerance\n");
        return 0;
    } else {
        printf("  FAIL: BSA deviates from CPU reference (max_err=%.6f > tol=%.6f)\n", max_err, tol);

        // Print first few elements for debugging
        printf("  First 8 outputs:\n");
        printf("    CPU: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", cpu_O[i]);
        printf("\n    GPU: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", h_O[i]);
        printf("\n");
        return 1;
    }
}
