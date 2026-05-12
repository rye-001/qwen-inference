# Future Work

Capabilities the modular layer architecture could enable, but that are not
commitments of the refactor in [`modular-layer-architecture.md`](modular-layer-architecture.md).

Each item is tagged with a confidence level so we don't confuse "the seam
is there" with "we know this works":

- **Design-ready** — clear implementation path, known prior art, risk is
  engineering effort not research.
- **Needs validation** — mechanism is plausible but has an open correctness
  or quality question that must be answered before committing.
- **Unproven research** — speculative. The modular architecture makes the
  experiment cheap, but the feature itself is not guaranteed to work.

None of these should be used to scope-creep any phase of the refactor.

---

## Design-ready

### Conversation branching via cheap state snapshots

Hybrid models have mostly fixed-size recurrent state (~60 MB DeltaNet) plus
a small attention KV cache (~5 MB for 10 attention layers at modest context).
Snapshotting is ~65 MB per branch vs gigabytes for a pure transformer. Forking
10 ways costs ~650 MB — within reach for agent backends doing tree-of-thoughts
search or multi-sample scoring.

**Caveat.** Only the DeltaNet state is truly constant-size; the attention KV
cache still grows with position. At long contexts the KV fork cost dominates
again. Branching is cheap at the moment of fork; retained branches accumulate.

### Per-layer-type precision budgeting

GGUF supports per-tensor quant types. The modular framework gives the
architectural rationale for which tensors deserve which precision:

- Attention weights — recall-critical, only 10 layers in Qwen 3.6 → high
  precision (e.g. Q6_K) is cheap to afford.
- DeltaNet weights — 30 layers, bulk of the non-MoE params, empirically
  more noise-tolerant → low precision (e.g. Q2_K / Q3_K) candidates.
- MoE expert weights — biggest chunk of the model by mass, only 8 of 256
  active per token → mid precision (e.g. Q3_K).

Implemented by configuring `llama-quantize` with a per-layer-type
importance map.

**Caveat.** The specific bit-widths above are hunches, not measurements.
Before shipping a role-quantized model, run perplexity and task evals on a
held-out set per Phase 5's "measured, not assumed" gate. Don't reference
these numbers as committed targets.

### Short-term / long-term memory split

Sliding-window attention on the 10 attention layers plus full-state DeltaNet
gives precise recall inside the window and fuzzy recall outside. The window
size becomes a hardware knob rather than a model hyperparameter.

**Caveat.** Sliding-window attention changes the model's effective context
and is not a trained-for regime for Qwen 3.6. Quality impact needs measurement.

---

## Needs validation

### Self-speculative decoding via layer-type skipping

Idea: skip the 10 attention layers during draft generation, verify with the
full (40-layer) forward pass. No separate draft model needed.

**Open correctness problem.** DeltaNet state must advance for every accepted
token. A draft pass that omits attention still runs DeltaNet — so its DeltaNet
state diverges from what the full-model verifier would have computed (because
in the full model, attention contributes to the residual stream that feeds the
next layer's DeltaNet input). Reconciling the two states during verification
is the whole design problem, and it is the same reason recurrent architectures
are traditionally hostile to speculation. This needs a real design doc with a
correctness proof sketch before we commit engineering time.

### Streaming prefill ("progressive JPEG for LLMs")

DeltaNet is O(n); attention is O(n²). Start generating from DeltaNet + MoE
while attention KV caches build in parallel.

**Open quality problem.** "Start generating with degraded layers, then upgrade"
means the first tokens are produced by a model that is meaningfully different
from the trained target. Users may see early tokens retracted or contradicted
mid-stream. This might be a UX win or a UX disaster; it needs A/B testing on
real prompts before we build it.

### Persistent DeltaNet memory

DeltaNet state is position-independent and fixed-size. Save to disk, reload
next session. In principle a "long-term memory" alternative to RAG.

**Open questions.**
1. Does resuming from a saved state produce coherent continuations, or does
   the absence of the prior token stream's attention contributions cause
   silent drift?
2. **Linear combinability of states is conjectural.** The delta rule produces
   additive low-rank updates mathematically, but that does not imply two
   states learned from different conversations compose meaningfully at the
   semantic level. This has not been demonstrated in published literature.
   Do not promise this externally until it's been tested.

### Constant-size prompt caching

Cache a shared system prompt as DeltaNet state + small attention KV. ~65 MB
vs gigabytes for full-KV prompt caching.

**Open question.** The "fuzzy tolerance to prompt edits" claim needs
measurement. Small edits may cause larger-than-expected output divergence,
especially for structured prompts (JSON schemas, tool definitions) where
the model is sensitive to exact tokens.

### Grammar-aware recurrent state correction

When grammar/trie forces a different token than the model sampled, the
DeltaNet state was updated based on the expected token. Rolling back and
re-running DeltaNet with the emitted token would prevent error compounding
in long structured outputs.

**Open question.** How often does grammar force a different token in
practice on our workloads, and does the un-corrected state actually cause
quality issues on our current outputs? Measure before optimizing.

---

## Unproven research

### Hallucination detection via attention replay

Replay the 10 attention layers on a generated token and inspect where
attention concentrates. Diffuse → "generated from DeltaNet memory"; peaked →
"grounded in input."

**Research status.** Using attention weights as evidence of causal grounding
is the Jain & Wallace (2019) "Attention is not Explanation" debate, with
substantial follow-up arguing both sides. The heuristic is sometimes
informative but is not a reliable hallucination detector. Shipping this as
"auditability for enterprise" would overclaim.

### Free semantic search from MoE routing

Each token's active experts (8-of-256) form a 256-bit fingerprint; use
Hamming distance as a similarity metric.

**Research status.** Published analyses of MoE routing (including Mixtral's
original paper) find expert selection is surprisingly noisy and not cleanly
semantic at the token level. Routing clusters by syntactic role more than
by topic. This *might* work as a coarse filter on top of a real retrieval
system; it is not a free embedding replacement.

### Layer-type activity dashboard (residual deltas)

Measure ||x_out − x_in||₂ per layer per token, aggregate by layer type.
Potentially useful as an inference-telemetry signal.

**Research status.** Useful as a debugging tool internally. As an externally
advertised interpretability feature, it is subject to the same "attention
weight ≠ causal contribution" caveat as replay-based detection. The signal
is real; the interpretation is contested.

### Speculative preview UI (faded draft tokens)

Render DeltaNet-only draft tokens in grey while the full model verifies;
confirmed tokens turn solid, rejected ones self-correct visibly.

**Depends on self-speculative decoding** working correctly first. The UI
idea is downstream of the correctness problem above.
