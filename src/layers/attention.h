#pragma once
// Attention layer graph-building functions.
//
// Responsibility: construct the KV-cache-backed attention subgraph for one
//   transformer layer. Three entry points:
//     build_attn_mha         — core Q@K^T→softmax→@V kernel (GQA-aware)
//     build_attention         — prefill path: cache write + full-history MHA
//     build_batched_attention — decode path: per-slot scatter + gather MHA
//
// Public surface: the three free functions declared below.
// State owned: none (KV cache is passed in by the caller).
// Invariants: all tensors are appended to the caller's ggml_cgraph;
//   no ggml_context is created here.
// Reference: Qwen3ForwardPass::_build_attention_layer and
//   ForwardPassBase::build_attn_mha in src/models/forward_pass_base.cpp.
// Unit test: tests/unit/test_attention.cpp (Phase 2).

#include "ggml.h"
#include <cstdint>
#include <vector>

class simple_kv_cache;
struct ggml_context;
struct ggml_cgraph;

// Core MHA kernel: Q@K^T → softmax(scale, mask) → @V.
// Handles GQA, head permutation, stream splitting, and contiguous recombination.
// Extracted from ForwardPassBase::build_attn_mha — identical logic.
ggml_tensor* build_attn_mha(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* kq_mask,
    ggml_tensor* sinks,
    float        kq_scale,
    uint32_t     pos,
    int          il);

// Prefill / single-slot attention.
// Writes K/V to the cache at the current slot position, reads the full
// cached sequence, builds a per-layer causal mask, then runs MHA.
// n_embd_head and n_head_kv are the per-architecture dimension values
// computed by the caller from Qwen3Metadata.
// Extracted from Qwen3ForwardPass::_build_attention_layer — identical logic.
ggml_tensor* build_attention(
    ggml_context*     ctx,
    ggml_cgraph*      gf,
    simple_kv_cache*  kv_cache,
    ggml_tensor*      q,
    ggml_tensor*      k,
    ggml_tensor*      v,
    int               layer_idx,
    float             kq_scale,
    uint32_t          n_tokens,
    uint32_t          slot_idx,
    int               il,
    int               n_embd_head,
    int               n_head_kv);

// Decode / batched multi-slot attention.
// Scatters K/V into each slot, gathers the full KV history via indices,
// applies a shared causal mask, then runs MHA.
// Reads n_embd_head and n_head_kv from k->ne[0]/ne[1].
// Extracted from Qwen3ForwardPass::_build_batched_attention_layer — identical logic.
ggml_tensor* build_batched_attention(
    ggml_context*                   ctx,
    ggml_cgraph*                    gf,
    simple_kv_cache*                kv_cache,
    ggml_tensor*                    q,
    ggml_tensor*                    k,
    ggml_tensor*                    v,
    int                             layer_idx,
    float                           kq_scale,
    const std::vector<uint32_t>&    slots,
    const std::vector<int32_t>&     positions,
    ggml_tensor*                    kq_mask,
    ggml_tensor*                    gather_indices,
    int                             il);

// ── Qwen3.5 attention variants ───────────────────────────────────────────────
// Qwen3.5 uses a joint Q+Gate projection, Q/K RMS norms, partial RoPE, and
// sigmoid gating after the attention output. These differ structurally from the
// Qwen2/3 attention above, so they live as separate free functions.

// Prefill / single-slot attention for Qwen3.5.
// Takes raw normed input cur; performs Q/K/V projections, Q/K norms, partial
// RoPE, KV cache write + full-history MHA, then sigmoid gating + output proj.
// Extracted from Qwen35ForwardPass::build_attention_layer — identical logic.
ggml_tensor* build_qwen35_attention(
    ggml_context*    ctx,
    ggml_cgraph*     gf,
    simple_kv_cache* kv_cache,
    ggml_tensor*     cur,          // normed input [n_embd, n_tokens]
    ggml_tensor*     inp_pos,      // [n_tokens]
    int              kv_cache_layer,
    uint32_t         n_tokens,
    uint32_t         slot_idx,
    int              il,
    ggml_tensor*     w_q,          // attn_q_weight (joint Q+Gate)
    ggml_tensor*     w_q_norm,     // attn_q_norm_weight
    ggml_tensor*     w_k,          // attn_k_weight
    ggml_tensor*     w_k_norm,     // attn_k_norm_weight
    ggml_tensor*     w_v,          // attn_v_weight
    ggml_tensor*     w_out,        // attn_output_weight
    int              n_embd_head,
    int              n_head,
    int              n_head_kv,
    int              n_rot,
    float            freq_base,
    int              context_length,
    float            rms_norm_eps);

// Decode / batched multi-slot attention for Qwen3.5.
// Same Qwen3.5-specific projections/norms/gating as above, but operates on
// a batch of slots with pre-built kq_mask and gather_indices.
// Extracted from Qwen35ForwardPass::build_batched_attention_layer — identical logic.
ggml_tensor* build_qwen35_batched_attention(
    ggml_context*                   ctx,
    ggml_cgraph*                    gf,
    simple_kv_cache*                kv_cache,
    ggml_tensor*                    cur,           // normed input [n_embd, n_batch]
    ggml_tensor*                    inp_pos,       // [n_batch]
    ggml_tensor*                    kq_mask,       // [n_kv_len, 1, 1, n_batch]
    ggml_tensor*                    gather_indices,// [n_batch * n_kv_len]
    int                             kv_cache_layer,
    const std::vector<uint32_t>&    slots,
    const std::vector<int32_t>&     positions,
    int                             il,
    ggml_tensor*                    w_q,
    ggml_tensor*                    w_q_norm,
    ggml_tensor*                    w_k,
    ggml_tensor*                    w_k_norm,
    ggml_tensor*                    w_v,
    ggml_tensor*                    w_out,
    int                             n_embd_head,
    int                             n_head,
    int                             n_head_kv,
    int                             n_rot,
    float                           freq_base,
    int                             context_length,
    float                           rms_norm_eps);
