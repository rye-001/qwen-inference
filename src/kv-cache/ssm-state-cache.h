#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <memory>
#include <cstdint>

/**
 * Fixed-size state storage for GatedDeltaNet (SSM) layers in Qwen 3.5.
 * 
 * Stores two state types per SSM layer per slot:
 *   - Recurrent state S: [d_state, d_state, n_heads] — the learned state matrix
 *   - Conv state:        [conv_dim, conv_kernel]     — causal conv1d sliding window
 *
 * Unlike the KV cache, these do NOT grow with sequence length.
 * For Qwen3.5-0.8B: S is 128×128×16 per layer, conv is 6144×4 per layer.
 *
 * Layout in backend buffer:
 *   recurrent: n_ssm_layers tensors of [d_state * d_state * n_heads, n_slots]
 *   conv:      n_ssm_layers tensors of [conv_dim * conv_kernel, n_slots]
 *
 * This class does NOT know about physical layer indices — it uses SSM-layer
 * indices (0..n_ssm_layers-1). The forward pass maps physical → SSM index.
 */
class ssm_state_cache {
public:
    ssm_state_cache(
        uint32_t n_ssm_layers,
        uint32_t n_slots,
        uint32_t d_state,       // ssm.state_size (128)
        uint32_t n_heads,       // ssm.group_count (16)
        uint32_t conv_dim,      // QKV projection width (6144 = 3 * ssm.inner_size)
        uint32_t conv_kernel,   // ssm.conv_kernel (4)
        ggml_backend_t backend
    );

    ~ssm_state_cache() = default;

    // Read recurrent state for [layer, slot] into dst (must be d_state*d_state*n_heads floats)
    void get_recurrent_state(uint32_t layer, uint32_t slot, float* dst) const;

    // Write recurrent state for [layer, slot] from src
    void set_recurrent_state(uint32_t layer, uint32_t slot, const float* src);

    // Read conv state for [layer, slot] into dst (must be conv_dim*conv_kernel floats)
    void get_conv_state(uint32_t layer, uint32_t slot, float* dst) const;

    // Write conv state for [layer, slot] from src
    void set_conv_state(uint32_t layer, uint32_t slot, const float* src);

    // Zero all state (recurrent + conv) for a slot across all layers
    void clear_slot(uint32_t slot);

    // Copy all state from src_slot to dst_slot
    void clone_slot(uint32_t src_slot, uint32_t dst_slot);

    // Total bytes allocated
    size_t total_bytes() const;

    // Accessors for forward pass graph building
    uint32_t get_n_ssm_layers() const { return n_ssm_layers_; }
    uint32_t get_d_state() const { return d_state_; }
    uint32_t get_n_heads() const { return n_heads_; }
    uint32_t get_conv_dim() const { return conv_dim_; }
    uint32_t get_conv_kernel() const { return conv_kernel_; }

    // Get raw tensor for graph view creation (forward pass will need this)
    ggml_tensor* get_recurrent_tensor(uint32_t layer) { return recurrent_state_[layer]; }
    ggml_tensor* get_conv_tensor(uint32_t layer) { return conv_state_[layer]; }

private:
    const uint32_t n_ssm_layers_;
    const uint32_t n_slots_;
    const uint32_t d_state_;
    const uint32_t n_heads_;
    const uint32_t conv_dim_;
    const uint32_t conv_kernel_;

    // Per-element sizes (in floats, for offset calculations)
    const size_t recurrent_slot_floats_;  // d_state * d_state * n_heads
    const size_t conv_slot_floats_;       // conv_dim * conv_kernel

    ggml_backend_t backend_;

    // ggml context for tensor metadata
    std::unique_ptr<ggml_context, void(*)(ggml_context*)> ctx_;

    // Backend buffer holding all state data
    std::unique_ptr<ggml_backend_buffer, void(*)(ggml_backend_buffer*)> buf_;

    // Tensors: one per SSM layer
    // recurrent: [recurrent_slot_floats, n_slots] per layer
    // conv:      [conv_slot_floats, n_slots] per layer
    std::vector<ggml_tensor*> recurrent_state_;
    std::vector<ggml_tensor*> conv_state_;

    void init_cache();
};
