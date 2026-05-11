#include "llama.h"
#include "log.h"
#include "pflash-score.h"

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

static std::vector<float> cpu_attention(
    const std::vector<float> & Q,
    const std::vector<float> & K,
    const std::vector<float> & V,
    int n_q, int n_heads, int n_kv, int D,
    float scale)
{
    std::vector<float> O((size_t)n_q * n_heads * D, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        for (int qi = 0; qi < n_q; qi++) {
            std::vector<float> S(n_kv, -INFINITY);
            float m_max = -INFINITY;
            for (int ki = 0; ki <= qi; ki++) {
                float s = 0.0f;
                for (int d = 0; d < D; d++) {
                    s += Q[((size_t)qi * n_heads + h) * D + d] *
                         K[((size_t)ki * n_heads + h) * D + d];
                }
                S[ki] = s * scale;
                if (S[ki] > m_max) m_max = S[ki];
            }
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

static void cpu_score(
    const std::vector<float> & K,
    int n_tokens, int kv_dim, int block_size,
    std::vector<float> & scores_out)
{
    int n_blocks = (n_tokens + block_size - 1) / block_size;
    scores_out.assign((size_t)n_blocks, 0.0f);
    const float * last_K = &K[(size_t)(n_tokens - 1) * kv_dim];
    float last_len = 0.0f;
    for (int i = 0; i < kv_dim; i++) last_len += last_K[i] * last_K[i];
    last_len = sqrtf(fmaxf(last_len, 1e-12f));
    for (int b = 0; b < n_blocks; b++) {
        int start = b * block_size;
        int end = std::min(start + block_size, n_tokens);
        std::vector<double> mean_buf((size_t)kv_dim, 0.0);
        for (int p = start; p < end; p++) {
            const float * kp = &K[(size_t)p * kv_dim];
            for (int i = 0; i < kv_dim; i++) mean_buf[i] += kp[i];
        }
        float inv = 1.0f / (float)(end - start);
        for (int i = 0; i < kv_dim; i++) mean_buf[i] *= inv;
        float dot = 0.0f, ml = 0.0f;
        for (int i = 0; i < kv_dim; i++) {
            dot += (float)mean_buf[i] * last_K[i];
            ml += (float)(mean_buf[i] * mean_buf[i]);
        }
        scores_out[b] = dot / (sqrtf(fmaxf(ml, 1e-12f)) * last_len);
    }
}

static int test_bsa_kernel() {
    printf("\n--- BSA Kernel Unit Test ---\n");

    const int D = 256;
    const int n_q = 16;
    const int n_kv = 256;
    const int n_heads = 8;
    const int n_heads_kv = 8;
    const float attn_scale = 1.0f / sqrtf((float)D);

    const int BSA_BLOCK = 128;
    const int n_blocks = (n_kv + BSA_BLOCK - 1) / BSA_BLOCK;

    const size_t q_size = (size_t)n_q * n_heads * D;
    const size_t kv_size = (size_t)n_kv * n_heads_kv * D;
    std::vector<float> h_Q(q_size);
    std::vector<float> h_K(kv_size);
    std::vector<float> h_V(kv_size);
    std::vector<float> h_O(q_size);

    int q_stride = n_heads * D;
    int q_head_stride = D;
    int kv_stride = n_heads_kv * D;
    int kv_head_stride = D;

    fill_random(h_Q.data(), q_size, 1.0f);
    fill_random(h_K.data(), kv_size, 1.0f);
    fill_random(h_V.data(), kv_size, 1.0f);

    auto cpu_O = cpu_attention(h_Q, h_K, h_V, n_q, n_heads, n_kv, D, attn_scale);

    std::vector<int32_t> block_mask(n_blocks);
    for (int b = 0; b < n_blocks; b++) block_mask[b] = b;

#ifdef GGML_USE_HIP
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

    float max_err = max_rel_error(cpu_O, h_O);
    printf("  Max relative error: %.6f\n", max_err);

    const float tol = 1e-3f;
    if (max_err <= tol) {
        printf("  PASS: BSA matches CPU reference within tolerance\n");
        return 0;
    } else {
        printf("  FAIL: BSA deviates from CPU reference (max_err=%.6f > tol=%.6f)\n", max_err, tol);
        printf("  First 8 outputs:\n");
        printf("    CPU: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", cpu_O[i]);
        printf("\n    GPU: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", h_O[i]);
        printf("\n");
        return 1;
    }
}

static int test_score_kernel() {
    printf("\n--- Scoring Kernel Unit Test ---\n");

    const int D = 256;
    const int n_tokens = 512;
    const int block_size = 128;
    const int n_blocks = (n_tokens + block_size - 1) / block_size;

    std::vector<float> h_K((size_t)n_tokens * D);
    fill_random(h_K.data(), (size_t)n_tokens * D, 1.0f);

    std::vector<float> cpu_scores;
    cpu_score(h_K, n_tokens, D, block_size, cpu_scores);

    std::vector<float> gpu_scores((size_t)n_blocks, 0.0f);

#ifdef GGML_USE_HIP
    float * d_K = nullptr;
    hipMalloc(&d_K, (size_t)n_tokens * D * sizeof(float));
    hipMemcpy(d_K, h_K.data(), (size_t)n_tokens * D * sizeof(float), hipMemcpyHostToDevice);

    int32_t n_scored = pflash_score_gpu(d_K, n_tokens, D, block_size, gpu_scores.data());
    hipDeviceSynchronize();
    hipFree(d_K);

    if (n_scored != n_blocks) {
        fprintf(stderr, "FAIL: pflash_score_gpu returned %d (expected %d)\n", n_scored, n_blocks);
        return 1;
    }
#else
    fprintf(stderr, "GPU not available, skipping GPU test\n");
    return 0;
#endif

    float max_err = max_rel_error(cpu_scores, gpu_scores);
    printf("  Max relative error: %.6f\n", max_err);

    const float tol = 1e-3f;
    if (max_err <= tol) {
        printf("  PASS: pflash_score matches CPU reference within tolerance\n");
        return 0;
    } else {
        printf("  FAIL: pflash_score deviates from CPU reference (max_err=%.6f > tol=%.6f)\n", max_err, tol);
        printf("  First 8 scores:\n");
        printf("    CPU: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", cpu_scores[i]);
        printf("\n    GPU: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", gpu_scores[i]);
        printf("\n");
        return 1;
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();
    (void)argc; (void)argv;

    printf("=== PFlash Kernel Unit Tests ===\n");

    int bsa_result = test_bsa_kernel();
    int score_result = test_score_kernel();

    if (bsa_result == 0 && score_result == 0) {
        printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    }
    printf("\n=== SOME TESTS FAILED ===\n");
    return bsa_result ? bsa_result : score_result;
}
