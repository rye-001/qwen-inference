#pragma once

#include <vector>
#include <cstdint>

#include "qwen3-model.h"
#include "../kv-cache/simple-kv-cache.h"


struct ggml_context;
struct ggml_tensor;
struct ggml_graph;
class Qwen3Model;

class Qwen3ForwardPass {
    // Grant the unit test access to private members for validation.
    friend class RopeCorrectnessTest_ApplyRopeMatchesGoldenValues_Test;
    friend class GQAAttentionCorrectnessTest_GQAAttentionMatchesGoldenValues_Test;
public:
    explicit Qwen3ForwardPass(const Qwen3Model& model, const Qwen3Metadata* metadata, uint32_t context_len, uint32_t max_batch_size = 1);
    ~Qwen3ForwardPass();

    // Builds the computation graph for a given set of input tokens.
    // Does NOT allocate or compute the graph.
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0);

    // Builds a batched computation graph for multiple slots
    // tokens: Concatenated input tokens (one per slot for decoding)
    // slots: Slot ID for each token
    // positions: Sequence position for each token
    struct ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens, 
        const std::vector<uint32_t>& slots, 
        const std::vector<int32_t>& positions
    );

    // Set input tensors AFTER allocation
    void set_inputs(ggml_cgraph* gf, 
                    const std::vector<int32_t>& tokens, 
                    int pos);

    void set_batched_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens, 
                    const std::vector<uint32_t>& slots,
                    const std::vector<int32_t>& positions);

    // Get output logits from GPU to CPU
    std::vector<float> get_output_logits(ggml_cgraph* gf);

    // Get output logits for a specific batch slot
    std::vector<float> get_output_logits_for_slot(ggml_cgraph* gf, uint32_t slot_index);



    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) {
      if (kv_cache_) {
        kv_cache_->advance(n_tokens, slot_idx);
      }
    }

    void clear_slot(uint32_t slot_idx) {
      if (kv_cache_) {
        kv_cache_->clear_slot(slot_idx);
      }
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) {
      if (kv_cache_) {
        kv_cache_->set_pos(pos, slot_idx);
      }
    }
    
    uint32_t get_cache_pos(uint32_t slot_idx) const { 
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0; 
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) {
        if (kv_cache_) {
            kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
        }
    }

    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }

private:
    const Qwen3Metadata& meta_;
    const Qwen3Model& model_;
    struct ggml_context* ctx_;
    std::vector<uint8_t> ctx_buffer_;
    std::unique_ptr<simple_kv_cache> kv_cache_;
    
    // Pre-computed RoPE tables for freq_base=1000000.0
    std::vector<float> rope_cos_cached_;
    std::vector<float> rope_sin_cached_;
    
    ggml_tensor * build_norm(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * mw,
        int   il) const;

    ggml_tensor* _ffn_swiglu(
    ggml_cgraph * gf,
    ggml_tensor* cur,
    ggml_tensor* gate,
    ggml_tensor* up,
    ggml_tensor* down,
    int il
    );


    ggml_tensor * _build_attention_layer(
    ggml_cgraph * gf,
    simple_kv_cache * kv_cache,
    ggml_tensor * q, 
    ggml_tensor * k, 
    ggml_tensor * v,
    int layer_idx, 
    float kq_scale,
    uint32_t n_tokens,
    uint32_t slot_idx,
    int il);

    ggml_tensor * _build_batched_attention_layer(
    ggml_cgraph * gf,
    simple_kv_cache * kv_cache,
    ggml_tensor * q, 
    ggml_tensor * k, 
    ggml_tensor * v,
    int layer_idx, 
    float kq_scale,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    ggml_tensor * kq_mask,
    ggml_tensor * gather_indices,
    int il);

    ggml_tensor * _build_attn_mha(        ggml_cgraph * gf,
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         float kq_scale,
         uint32_t pos,
        int il) const;

    void set_tensor_name(ggml_cgraph* gf, ggml_tensor* tensor, const char* name, int il = -1) const;

    struct ggml_tensor* embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens); 

        };

