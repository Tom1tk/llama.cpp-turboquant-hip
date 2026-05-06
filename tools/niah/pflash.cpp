#include "pflash.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

struct llama_context * pflash_init_draft(
    const std::string & model_path,
    int32_t n_ctx,
    const std::string & cache_type_k,
    const std::string & cache_type_v)
{
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("pflash: failed to load draft model: %s", model_path.c_str());
        return nullptr;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;

    if (cache_type_k == "q8_0") {
        cparams.type_k = GGML_TYPE_Q8_0;
    } else if (cache_type_k == "f16") {
        cparams.type_k = GGML_TYPE_F16;
    } else {
        cparams.type_k = GGML_TYPE_F32;
    }
    if (cache_type_v == "q8_0") {
        cparams.type_v = GGML_TYPE_Q8_0;
    } else if (cache_type_v == "f16") {
        cparams.type_v = GGML_TYPE_F16;
    } else {
        cparams.type_v = GGML_TYPE_F32;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("pflash: failed to create draft context");
        llama_model_free(model);
        return nullptr;
    }

    return ctx;
}

std::vector<float> pflash_score(
    struct llama_context * draft_ctx,
    const std::vector<llama_token> & tokens,
    int32_t score_layer,
    int32_t block_size)
{
    const int32_t n_tokens = (int32_t)tokens.size();
    if (n_tokens == 0) return {};

    // Auto-select scoring layer: use layer 0 (shallowest)
    if (score_layer < 0) {
        score_layer = 0;
    }

    const int32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    std::vector<float> scores(n_blocks, 0.0f);

    // Read K cache for a specific position
    // Returns n_kv_heads * head_dim floats, or 0 on failure
    // Read K values for each position in the prompt
    // We need: n_tokens positions, each with n_kv_heads * head_dim values
    const int32_t n_layer = (int32_t)llama_model_n_layer(llama_get_model(draft_ctx));
    if (score_layer >= n_layer) {
        LOG_WRN("pflash: score_layer %d >= n_layer %d, using 0", score_layer, n_layer);
        score_layer = 0;
    }

    // Determine the head dimension and kv head count from the model
    const int32_t n_head = (int32_t)llama_model_n_head(llama_get_model(draft_ctx));
    const int32_t n_embd = (int32_t)llama_model_n_embd(llama_get_model(draft_ctx));
    const int32_t n_kv_heads = (int32_t)llama_model_n_head_kv(llama_get_model(draft_ctx));
    const int32_t head_dim = n_embd / n_head;

    const int32_t kv_dim = n_kv_heads * head_dim;

    // Allocate buffer for one position's K values
    std::vector<float> k_buf(kv_dim);

    // For each position, read K from cache
    // Cache KV data per position so we don't repeatedly read the same positions
    std::vector<std::vector<float>> all_k(n_tokens, std::vector<float>(kv_dim));

    for (int32_t pos = 0; pos < n_tokens; pos++) {
        int32_t n_read = llama_kv_cache_read_k_for_pos(draft_ctx, score_layer, pos, 0, k_buf.data());
        if (n_read <= 0) {
            LOG_ERR("pflash: failed to read K cache at pos %d layer %d", pos, score_layer);
            return {};
        }
        memcpy(all_k[pos].data(), k_buf.data(), kv_dim * sizeof(float));
    }

    // Get the last position's K vector for reference
    const std::vector<float> & last_k = all_k[n_tokens - 1];
    float last_len = 0.0f;
    for (int32_t i = 0; i < kv_dim; i++) {
        last_len += last_k[i] * last_k[i];
    }
    last_len = std::sqrt(std::max(last_len, 1e-12f));

    // For each block, compute mean K and cosine similarity to last K
    for (int32_t b = 0; b < n_blocks; b++) {
        int32_t start = b * block_size;
        int32_t end = std::min(start + block_size, n_tokens);

        // Compute mean K for this block
        std::vector<float> mean_k(kv_dim, 0.0f);
        int32_t block_len = end - start;
        for (int32_t p = start; p < end; p++) {
            const auto & k = all_k[p];
            for (int32_t i = 0; i < kv_dim; i++) {
                mean_k[i] += k[i];
            }
        }
        float inv_len = 1.0f / (float)block_len;
        for (int32_t i = 0; i < kv_dim; i++) {
            mean_k[i] *= inv_len;
        }

        // Cosine similarity: dot(mean_k, last_k) / (||mean_k|| * ||last_k||)
        float dot = 0.0f, mean_len = 0.0f;
        for (int32_t i = 0; i < kv_dim; i++) {
            dot += mean_k[i] * last_k[i];
            mean_len += mean_k[i] * mean_k[i];
        }
        mean_len = std::sqrt(std::max(mean_len, 1e-12f));

        scores[b] = dot / (mean_len * last_len);
    }

    return scores;
}

std::vector<pflash_span> pflash_select(
    const std::vector<float> & scores,
    int32_t block_size,
    int32_t n_tokens,
    int32_t sink_tokens,
    int32_t recent_tokens,
    float keep_ratio,
    int32_t min_keep_tokens)
{
    std::vector<pflash_span> result;

    if (n_tokens <= 0) return result;

    // Compute target budget
    int32_t target_kept = std::max(min_keep_tokens, (int32_t)std::ceil(keep_ratio * n_tokens));
    target_kept = std::min(target_kept, n_tokens);

    // Early exit: keep everything
    if (target_kept >= n_tokens) {
        result.push_back({0, n_tokens});
        return result;
    }

    // Build mandatory anchors
    std::vector<pflash_span> anchors;

    // Sink tokens (prefix)
    int32_t sink_end = std::min(sink_tokens, n_tokens);
    if (sink_end > 0) {
        anchors.push_back({0, sink_end});
    }

    // Recent tokens (suffix)
    int32_t recent_start = std::max(0, n_tokens - recent_tokens);
    if (recent_start < n_tokens) {
        anchors.push_back({recent_start, n_tokens});
    }

    // Coalesce anchors (merge overlapping)
    if (anchors.size() >= 2) {
        std::sort(anchors.begin(), anchors.end(),
            [](const pflash_span & a, const pflash_span & b) { return a.start < b.start; });
        std::vector<pflash_span> merged;
        merged.push_back(anchors[0]);
        for (size_t i = 1; i < anchors.size(); i++) {
            if (anchors[i].start <= merged.back().end) {
                merged.back().end = std::max(merged.back().end, anchors[i].end);
            } else {
                merged.push_back(anchors[i]);
            }
        }
        anchors = merged;
    }

    // Count anchored tokens
    int32_t anchored = 0;
    for (const auto & a : anchors) {
        anchored += (a.end - a.start);
    }

    // If anchors already meet budget, return them
    if (anchored >= target_kept) {
        return anchors;
    }

    // Middle budget: tokens to select from unanchored regions
    int32_t middle_budget = target_kept - anchored;

    // Build sorted list of blocks by score (descending)
    int32_t n_blocks = (int32_t)scores.size();
    std::vector<std::pair<int32_t, float>> ranked;
    for (int32_t b = 0; b < n_blocks; b++) {
        // Skip blocks that are fully covered by anchors
        int32_t b_start = b * block_size;
        int32_t b_end = std::min(b_start + block_size, n_tokens);

        bool fully_covered = false;
        for (const auto & a : anchors) {
            if (b_start >= a.start && b_end <= a.end) {
                fully_covered = true;
                break;
            }
        }
        if (!fully_covered) {
            ranked.push_back({b, scores[b]});
        }
    }

    // Sort by score descending, then by block index ascending (tiebreaker)
    std::sort(ranked.begin(), ranked.end(),
        [](const auto & a, const auto & b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });

    // Select blocks greedily
    int32_t middle_kept = 0;
    std::vector<pflash_span> selected;
    for (const auto & r : ranked) {
        if (middle_kept >= middle_budget) break;

        int32_t b = r.first;
        int32_t b_start = b * block_size;
        int32_t b_end = std::min(b_start + block_size, n_tokens);

        // Compute incremental tokens (not covered by existing spans)
        int32_t incr = b_end - b_start;
        for (const auto & a : anchors) {
            int32_t overlap_start = std::max(b_start, a.start);
            int32_t overlap_end = std::min(b_end, a.end);
            if (overlap_start < overlap_end) {
                incr -= (overlap_end - overlap_start);
            }
        }
        for (const auto & s : selected) {
            int32_t overlap_start = std::max(b_start, s.start);
            int32_t overlap_end = std::min(b_end, s.end);
            if (overlap_start < overlap_end) {
                incr -= (overlap_end - overlap_start);
            }
        }

        if (incr > 0) {
            selected.push_back({b_start, b_end});
            middle_kept += incr;
        }
    }

    // Combine anchors and selected spans
    result = anchors;
    result.insert(result.end(), selected.begin(), selected.end());

    // Coalesce all spans
    std::sort(result.begin(), result.end(),
        [](const pflash_span & a, const pflash_span & b) { return a.start < b.start; });
    std::vector<pflash_span> coalesced;
    coalesced.push_back(result[0]);
    for (size_t i = 1; i < result.size(); i++) {
        if (result[i].start <= coalesced.back().end) {
            coalesced.back().end = std::max(coalesced.back().end, result[i].end);
        } else {
            coalesced.push_back(result[i]);
        }
    }

    return coalesced;
}

std::vector<llama_token> pflash_gather(
    const std::vector<llama_token> & source,
    const std::vector<pflash_span> & spans)
{
    std::vector<llama_token> result;
    result.reserve(source.size());

    for (const auto & span : spans) {
        for (int32_t i = span.start; i < span.end && i < (int32_t)source.size(); i++) {
            result.push_back(source[i]);
        }
    }

    return result;
}

pflash_result pflash_compress(
    struct llama_context * draft_ctx,
    const std::vector<llama_token> & tokens,
    const pflash_params & params)
{
    pflash_result res;
    res.source_count = (int32_t)tokens.size();

    // Check threshold
    if ((int32_t)tokens.size() < params.threshold_tokens) {
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }

    if (!draft_ctx) {
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }

    // Step 1: Run draft model on the full prompt
    int64_t t0 = ggml_time_us();
    llama_memory_clear(llama_get_memory(draft_ctx), true);
    llama_synchronize(draft_ctx);

    int32_t n_batch = 2048;
    for (int32_t i = 0; i < (int32_t)tokens.size(); i += n_batch) {
        int32_t n = std::min((int32_t)tokens.size() - i, n_batch);
        auto batch = llama_batch_get_one(const_cast<llama_token *>(&tokens[i]), n);
        if (llama_decode(draft_ctx, batch) != 0) {
            LOG_ERR("pflash: draft decode failed at position %d", i);
            res.bypassed = true;
            res.tokens = tokens;
            return res;
        }
    }
    llama_synchronize(draft_ctx);
    res.draft_us = ggml_time_us() - t0;

    // Step 2: Score blocks
    t0 = ggml_time_us();
    auto scores = pflash_score(draft_ctx, tokens, params.score_layer, params.block_size);
    if (scores.empty()) {
        LOG_ERR("pflash: scoring failed, bypassing");
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }
    res.score_us = ggml_time_us() - t0;

    // Step 3: Select spans
    t0 = ggml_time_us();
    auto spans = pflash_select(
        scores, params.block_size, (int32_t)tokens.size(),
        params.sink_tokens, params.recent_tokens,
        params.keep_ratio, params.min_keep_tokens);
    res.select_us = ggml_time_us() - t0;

    // Step 4: Gather tokens
    t0 = ggml_time_us();
    res.tokens = pflash_gather(tokens, spans);
    res.gather_us = ggml_time_us() - t0;

    res.spans = spans;
    res.kept_count = (int32_t)res.tokens.size();

    // If compression didn't reduce, bypass
    if (res.kept_count >= res.source_count) {
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }

    return res;
}
