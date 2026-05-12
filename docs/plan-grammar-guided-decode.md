# Grammar-Guided Decode: Sparse LM Head + Forced-Token Elision

**Goal:** Exploit grammar-determined token sets to (a) shrink the LM-head matmul when only k tokens are valid, and (b) skip the forward pass entirely when only one token is valid.
**Impact (rough, on Qwen 3.6-A3B with a constrained DSL):**
- Sparse LM head alone: ~5–10% more tok/s.
- Combined with forced-token elision: ~1.5–2× tok/s on grammar-heavy workloads (e.g. [tests/system_prompt_order_mngmt.txt](../tests/system_prompt_order_mngmt.txt)).

This plan is the consequence of a "do we need the whole LLM for this DSL?" stress test. The answer: no — and the existing `sampling/` ↔ `models/` seam can host the optimization without bending any layer module. That makes it a healthy architectural pressure test, not just a feature.

---

## Core idea

Today: forward pass produces full `[vocab]` logits → grammar masks invalid tokens → sample.
Proposed: grammar tells us the valid set *before* the forward pass → we use that knowledge to shrink (or skip) the forward pass.

Two distinct surgeries, with very different leverage:

| Trie state at step | Today | Proposed |
|---|---|---|
| 1 valid token (forced) | full forward + mask + sample | **emit token, batched-append to KV, no forward** |
| k valid tokens, k small | full forward + mask + sample | **forward with sparse LM head (k rows only)** |
| unrestricted | full forward + sample | unchanged |

The forced-token case is where the big wins live — JS DSLs like the order-mgmt prompt are 40–60% boilerplate (`await `, `const x = `, `({ `, `});`). The sparse-head case is smaller but is the right thing to land first because the wiring it requires (`grammar → recipe valid_indices`) is also what forced-token elision needs.

---

## Architectural framing

**This is not "model surgery."** The model is unchanged. Both tricks live at the **sampling ↔ recipe** seam:

- `sampling/` already owns the TokenTrie, which already maintains the valid set per slot.
- `models/` (the recipe) already builds the LM head; it just builds it dense today.

What changes is the **direction of the dependency at decode time**: today it's forward → grammar; we add a backward edge grammar → forward(valid_indices). No layer module touched. No new ggml op. Generalizes to any grammar-constrained workload (not just this DSL).

**Constraints we will respect:**

- No `if (domain == "order_mgmt")` anywhere. The grammar is data; the engine handles "constrained decoding" as a general capability.
- KV append vs recurrent overwrite stays distinct (per CLAUDE.md). Forced-token elision touches KV append only; recurrent state still runs its update on every step regardless.
- No layer module gains a sampling-aware code path. The optimization is contained to recipe + sampling + decode loop.

---

## Phase A — Sparse LM head (the warm-up)

**What changes mathematically:**

```cpp
// dense (today, forward_pass_base.cpp:104-108)
cur = ggml_mul_mat(ctx_, output_weight, cur);            // [vocab]

// sparse (new path)
W_k     = ggml_get_rows(ctx_, output_weight, valid_idx); // [k, hidden]
logits_k = ggml_mul_mat(ctx_, W_k, cur);                 // [k]
```

`valid_idx` is a `[k]` int32 input tensor populated host-side before `ggml_graph_compute`.

**Why two graphs, not one.** ggml dislikes dynamic shapes inside a graph. Cleanest: build a dense graph and a sparse graph at init; the decode loop picks per step based on `|valid_set|` (e.g. `k < vocab/8`). Above the threshold, dense wins; below, sparse wins. Pattern already exists — see [src/models/qwen3.cpp:595](../src/models/qwen3.cpp:595) (`_build_output_head_graph`).

**Why this matters more on MoE.** On dense Qwen3-32B the LM head is ~1% of forward — barely worth touching. On Qwen3.6-A3B only ~3B of 35B params activate per token, so the body shrinks ~10× and the LM head climbs to ~10–15% of forward. The headline target model is exactly where this earns its keep.

### Files touched (Phase A)

Core math:
- [src/models/forward_pass_base.cpp:100](../src/models/forward_pass_base.cpp:100) — add `build_output_head_sparse(gf, cur, valid_idx_input)`.
- [src/models/forward_pass_base.h:161](../src/models/forward_pass_base.h:161) — declare it + hold the `valid_indices` input tensor handle.

Recipe call sites (per opt-in model):
- [src/models/qwen36.cpp:311](../src/models/qwen36.cpp:311), [:436](../src/models/qwen36.cpp:436) — primary target (MoE, biggest payoff).
- [src/models/qwen3.cpp:182](../src/models/qwen3.cpp:182), [src/models/qwen35.cpp:265](../src/models/qwen35.cpp:265), [:779](../src/models/qwen35.cpp:779) — secondary.
- Gemma recipes only if needed.

Each opt-in recipe builds two output-head graphs and exposes a selector.

Sampling (produce `valid_indices` *before* the forward pass):
- [src/sampling/token-trie.h](../src/sampling/token-trie.h) / [.cpp](../src/sampling/token-trie.cpp) — add `current_valid_token_ids(slot) -> std::vector<int32_t>`. The trie's bitset already has this; just an enumeration helper.
- [src/sampling/sampling.cpp](../src/sampling/sampling.cpp) — split today's `sample(dense_logits, …)` into:
  - `peek_valid_set(context) -> vector<int32_t>` — called before forward.
  - `sample_sparse(sparse_logits, valid_idx, context)` — maps k-logits back to token IDs.

Decode loop (the selector):
- [src/cli/chat.cpp:142](../src/cli/chat.cpp:142), [:190](../src/cli/chat.cpp:190)
- [src/cli/complete.cpp:89](../src/cli/complete.cpp:89), [:136](../src/cli/complete.cpp:136), [:185](../src/cli/complete.cpp:185)
- [src/server/http_server.cpp:267](../src/server/http_server.cpp:267), [:301](../src/server/http_server.cpp:301)

All three currently do the same three-step dance (run_prefill → tail logits → sample). The selector lifts cleanly into a helper in `src/core/` so the three callers each become a one-liner.

**Note.** The three-way duplication in the decode loop is pre-existing — this plan surfaces it, doesn't cause it. Worth fixing while wiring this through.

### Done criteria (Phase A)

1. Recipe builds two graphs at init; selector picks based on `|valid_set|` and threshold.
2. Bit-for-bit identical sampling for a grammar-free workload (sparse path never taken).
3. End-to-end tok/s on Qwen 3.6-A3B with the order-mgmt prompt: target +5–10%.
4. No new graph node on the dense path. No `valid_indices` input plumbed through layers.

---

## Phase B — Forced-token elision (the main event)

**The mechanism.** When the TokenTrie state has exactly one valid token, the next token is determined. We:
1. Emit that token to the output stream.
2. Append it to the KV cache via a batched mini-prefill (potentially across multiple consecutive forced tokens).
3. Update the trie state.
4. Resume normal decode at the next *branching* state.

No forward pass for forced positions. The recurrent state (DeltaNet) still needs to advance for those positions because its state mutates — so a forced run becomes a "prefill of length r" on the recurrent layers, with the LM head skipped entirely (we're not predicting; we already know the tokens).

**Why this is harder than Phase A.** Phase A is one matmul. Phase B is a control-flow change in the decode loop and a new interaction with `state/`:
- The recurrent state must support "advance by r tokens without producing logits."
- The KV cache must support "append r tokens in one batched dispatch."
- Both probably exist for prefill already — Phase B's job is to reach into prefill-mode for runs of length r mid-decode.

**Why it composes with Phase A.** At a forced position, Phase B skips the work. At a branching position with small k, Phase A makes the work cheap. Different stretches, different optimizations, same seam.

### Files likely touched (Phase B, sketch only)

- `src/sampling/token-trie.{h,cpp}` — add `forced_run() -> vector<int32_t>` returning all consecutive forced tokens until the next branching state.
- `src/state/` — verify the KV cache and DeltaNet state both expose a "prefill r tokens" entry point usable mid-decode; if not, that's the load-bearing refactor.
- `src/core/` decode helper — orchestrate: while trie says forced, batch-advance state, emit tokens, no LM head.
- `src/cli/*`, `src/server/*` — call the new helper; no change to their structure beyond Phase A.

This phase is genuinely architectural — it's the stress test for whether the sampling↔state seam is as clean as it looks today. If `state/` cannot host "prefill r tokens during decode" without leaking into layer modules, that's an interface defect worth fixing before adding the feature.

### Done criteria (Phase B)

1. On the order-mgmt prompt: total tok/s on Qwen 3.6-A3B reaches ~1.5–2× baseline (combined with Phase A).
2. Forced-token elision verified bit-for-bit against a non-elided run (same output sequence, just produced faster).
3. No `if (grammar)` branches inside any layer module.
4. `state/` exposes its prefill entry points cleanly; KV append vs recurrent overwrite stay distinct.

---

## What we are explicitly *not* doing

- **Not distilling a specialist model.** Out of scope for an inference engine.
- **Not baking any specific grammar into the engine.** The order-mgmt DSL is a benchmark, not a hard-coded path.
- **Not unifying KV cache and recurrent state.** CLAUDE.md is explicit; this plan respects it.
- **Not optimizing layer modules for constrained decoding.** Layer code stays unaware.

---

## Order of operations

1. **Phase A first.** Lower risk, smaller surface, proves the grammar→recipe wiring.
2. **Lift the decode-loop duplication** into a `src/core/` helper while doing Phase A — three callers, three identical edits, obvious refactor.
3. **Phase B second**, only after Phase A is landed and the wiring is verified.
4. Re-measure on Qwen 3.6-A3B with the order-mgmt prompt at each phase boundary.

If Phase A's wins land below 3% or the wiring proves ugly, that's a strong signal to pause and rethink the seam before committing to Phase B.

---

## Related

- [docs/plan-grammar-trie.md](plan-grammar-trie.md) — the TokenTrie this plan builds on.
- [docs/plan-resolve-once.md](plan-resolve-once.md) — orthogonal grammar-side optimization; both target the same workload but optimize different layers of the stack.
- [docs/phase4-investigation.md](phase4-investigation.md) — measured cost breakdown that contextualizes "what fraction of forward is the LM head."
- [docs/future-work.md](future-work.md) — conversation branching entry; unrelated but adjacent (also a sampling/state-seam feature).
