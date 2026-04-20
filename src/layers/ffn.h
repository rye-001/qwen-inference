#pragma once
// SwiGLU FFN graph-building function.
//
// Responsibility: fused up/gate/down FFN block with SwiGLU activation.
// Public surface: build_ffn_swiglu free function below.
// State owned: none (ggml_context is passed in by the caller).
// Invariants: appends to the caller's ggml_cgraph; creates no ggml_context.
// Reference: ForwardPassBase::ffn_swiglu in src/models/forward_pass_base.cpp.
// Unit test: tests/unit/test_ffn.cpp (Phase 2).

#include "ggml.h"

struct ggml_context;
struct ggml_cgraph;

// SwiGLU FFN: down(swiglu(gate(cur), up(cur))).
// Sets tensor names ffn_up.N, ffn_gate.N, ffn_swiglu.N for debuggability.
// Extracted from ForwardPassBase::ffn_swiglu — identical logic.
ggml_tensor* build_ffn_swiglu(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il);
