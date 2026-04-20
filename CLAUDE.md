## Project Overview

A high-performance inference engine for Qwen-family models, running on Apple
Silicon via Metal. Written in C++ using ggml as the tensor computation backend.

## Architecture Direction

The engine is being refactored from a monolithic, model-specific design toward
a composable, layer-type-driven architecture. See `docs/modular-layer-architecture.md`
for the full blueprint.

Key principles:
- **Layer-type modules** (Attention, SSM, DeltaNet, MoE, Dense FFN) are independent,
  testable units that build into a shared `ggml_cgraph`. They are graph-building
  functions, not standalone executables.
- **Models are recipes** — a model file composes layer modules and owns the residual
  stream. Adding a new model should require zero changes to existing modules.
- **Architecture-agnostic systems** (sampling/grammar, speculative decoding, weight
  quantization, Metal dispatch) sit above or below the layer modules.
- The refactor follows strict phasing: extract first (bit-for-bit identical),
  stabilize interfaces second, add new layer types third, optimize kernels fourth.
  Never combine extraction with optimization in the same step.

## Current Capabilities (Decode Phase)

1. Weights: K-Quants
2. KV Precision: TurboQuant
3. KV Volume: SnapKV
4. Output Leash: Grammar / TokenTrie
5. Decode Speed: Speculative Decoding (in progress)

## Target Models

- Qwen2/2.5
- Qwen3-32B (pure transformer — attention + dense FFN)
- Qwen3.5 (attention + SSM)
- Qwen 3.6-35B-A3B (hybrid — DeltaNet + attention + MoE, arch: `qwen35moe` in GGUF)

## ggml Constraints

- All layer modules must share one `ggml_context` and build into one `ggml_cgraph`.
- Avoid premature `ggml_cont()` / `ggml_cpy()` — they pin scratch buffer memory.
- MoE dispatch must use a fused `ggml_custom_op`, not chained `ggml_mul_mat` per
  expert (kernel launch overhead on Metal kills performance otherwise).
- KV cache (append semantics) and recurrent state (overwrite semantics) are
  fundamentally different — never unify their implementations behind a shared base
  that assumes one update pattern.

## Collaboration Invariants

Load-bearing rules that protect the codebase from drifting into shapes that
are hostile to both human engineers under time pressure and LLM agents under
context limits. If any rule ever conflicts with good engineering, good
engineering wins and the rule is revised. Full list in
`docs/modular-layer-architecture.md` → Collaboration Invariants.

- **Locality of reasoning.** A change to one layer type should not require
  reading more than two other files. Reviewers reject violations regardless
  of whether the change compiles.
- **Test co-location.** `src/<path>/<module>.cpp` has its unit test at
  `tests/unit/test_<module>.cpp`. Always. No nesting variants.
- **No dumping grounds.** `src/` contains only concept-named directories
  (`layers/`, `state/`, `models/`, `metal/`, `sampling/`, `quant/`,
  `loader/`, `cli/`, `server/`). No `util/`, `common/`, `misc/`, `helpers/`.
- **Fail-loud error contract.** Errors at module boundaries name the slot
  or parameter, the expected value, and the actual value, in that order.
  Silent fallbacks and best-effort recovery are forbidden at module
  boundaries.
- **No cleverness that hides the public surface.** Prefer verbose and
  explicit over clever and terse. No template metaprogramming, runtime
  reflection, or multi-level indirection. Generated code lives under
  `src/generated/` and is never hand-edited.

## Collaboration Style

Proactively flag architectural smells, redundant abstractions, wrong abstraction levels, or design improvements — don't wait to be asked. If a completed task reveals something worth questioning about the broader design, say so immediately.
