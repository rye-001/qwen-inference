#pragma once

#include "forward-pass-base.h"
#include "../kv-cache/simple-kv-cache.h"

/**
 * Forward pass for Qwen2 and Qwen3 architectures.
 * 
 * All layers are uniform transformer blocks with standard softmax attention.
 * KV cache spans all layers with identity index mapping.
 * 
 * This class preserves the exact same public API as the original
 * Qwen3ForwardPass — existing call sites (CLI, server, tests) need
 * no changes.
 */
class Qwen3ForwardPass : public ForwardPassBase {
    // Grant the unit test access to private members for validation.
    friend class RopeCorrectnessTest_ApplyRopeMatchesGoldenValues_Test;
    friend class GQAAttentionCorrectnessTest_GQAAttentionMatchesGoldenValues_Test;
public:
    explicit Qwen3ForwardPass(const Qwen3Model& model, const Qwen3Metadata* metadata, uint32_t context_len, uint32_t max_batch_size = 1);
    ~Qwen3ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0) override;

    struct ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions
    ) override;

    // --- Input setting (after graph allocation) ---
    void set_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    int pos) override;

    void set_batched_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    const std::vector<uint32_t>& slots,
                    const std::vector<int32_t>& positions) override;

    // --- Cache management ---
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
    }

    void clear_slot(uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
    }

    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }

private:
    std::unique_ptr<simple_kv_cache> kv_cache_;

    // Pre-computed RoPE tables
    std::vector<float> rope_cos_cached_;
    std::vector<float> rope_sin_cached_;

    // --- Qwen2/3-specific attention builders ---
    ggml_tensor* _build_attention_layer(
        ggml_cgraph* gf,
        simple_kv_cache* kv_cache,
        ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
        int layer_idx, float kq_scale,
        uint32_t n_tokens, uint32_t slot_idx, int il);

    ggml_tensor* _build_batched_attention_layer(
        ggml_cgraph* gf,
        simple_kv_cache* kv_cache,
        ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
        int layer_idx, float kq_scale,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions,
        ggml_tensor* kq_mask,
        ggml_tensor* gather_indices, int il);
};