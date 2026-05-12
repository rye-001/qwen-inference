# KV Cache Unification — Research Spike

> **Recommendation: Path A — extract a shared per-layer builder, one mechanical PR.**
> The two prefill loops differ only in which `simple_kv_cache*` instance they pass;
> no new cache interface is required. `CompressedKVStore` is not a candidate for
> polymorphic unification — it is a CPU-only backing store, not an alternative graph cache.

---

## 1. Line-level diff of the two prefill loops

| Aspect | Non-TQ path (`build_prefill_graph`) | TQ path (`_build_layer_batch_graph`) |
|---|---|---|
| **Location** | `src/models/qwen3.cpp:138–245` | `src/models/qwen3.cpp:785–831` |
| **Graph scope** | Monolithic: embedding → all layers → output head | Partial: hidden-state input → N-layer batch → `"layer_out"` |
| **Input tensor** | Token embedding via `embedding(gf, tokens)` (line 123) | `ggml_set_input(inpL)` — 2D F32 tensor, set externally (line 772) |
| **Loop range** | `il = 0 … n_layers` (line 138) | `il = il_start … il_end` (line 785) |
| **Attention call** | `build_attention(ctx_, gf, kv_cache_.get(), …)` (line 214) | `build_attention(ctx_, gf, tq_scratch_cache_.get(), …)` (line 821) |
| **Cache type** | `simple_kv_cache*` | `simple_kv_cache*` — **same type** |
| **Per-layer nodes** | norm → QKV proj → qwen3-norm → RoPE → attention → out-proj → residual → FFN → residual | identical |
| **Output** | `build_output_head(gf, inpL)` appended | `ggml_set_name(cur, "layer_out")` only |
| **Tensor naming** | Verbose `set_tensor_name()` per node (lines 144–241) | Minimal: only `"inpL"`, `"inp_pos"`, `"layer_out"` |

**Classification of differences:**

- **(a) Different cache object, same function** — `kv_cache_.get()` vs `tq_scratch_cache_.get()`. **Both are `simple_kv_cache*`.** The cache type is already shared; no new interface is needed.
- **(b) Genuinely different graph nodes** — **None** in the per-layer body. The graph-scope difference (embedding+head vs hidden-state I/O) is in the outer wrapper, not inside the loop.
- **(c) Incidental** — Tensor naming verbosity; variable declaration position of `kq_scale`; bias check ordering for qwen2.

**Decode and batched-decode:**
The decode path (`build_decoding_graph`, lines 253–387) has **no TQ equivalent**. It is a single path that always uses `kv_cache_.get()` at line 360. There is no decode-path duplication. The TQ path defers decode to watermark-based incremental decompression: before each decode graph compute, only tokens absent from scratch are decompressed from `tq_store_` (CPU-side, outside the ggml graph).

---

## 2. Public-surface comparison

| Method | `simple_kv_cache` | `CompressedKVStore` | Notes |
|---|---|---|---|
| `get_k(ctx, il, n_kv, slot)` → `ggml_tensor*` | ✓ | ✗ | ggml tensor views; used by `build_attention` |
| `get_v(ctx, il, n_kv, slot)` → `ggml_tensor*` | ✓ | ✗ | |
| `cpy_k(ctx, k_cur, il, slot)` → `ggml_tensor*` | ✓ | ✗ | Schedules in-graph write |
| `cpy_v(ctx, v_cur, il, slot)` → `ggml_tensor*` | ✓ | ✗ | |
| `gather_k/v(ctx, gf, il, indices, …)` → `ggml_tensor*` | ✓ | ✗ | Used by `build_batched_attention` |
| `get_k_cache_tensor(layer)` / `get_v_cache_tensor(layer)` | ✓ | ✗ | Raw tensor access |
| `truncate_to_position(pos, slot)` | ✓ | ✗ | O(1) head-pointer rewind |
| `compress_token_k/v(layer, slot, pos, float*)` | ✗ | ✓ | CPU WHT + Lloyd-Max quant |
| `decompress_token_k/v(layer, slot, pos, float*)` | ✗ | ✓ | CPU dequant |
| `advance(slot, n)` | `advance(n_tokens, slot_idx)` | `advance(slot, n_tokens)` | Arg order differs |
| `set_pos(slot, pos)` / `get_pos(slot)` | ✓ (same semantics) | ✓ (same semantics) | |
| `compact(…)` | `compact(slot_idx, retained)` | `compact(layer, slot, retained)` | Different scope and arg order |
| `clone_slot(src, dst, n_tokens)` | tensor memcpy via backend | compressed byte copy | Same signature, different semantics |
| `clear_slot(slot)` | ✓ | ✓ | |
| `reset_sequence(seq_id)` / `memory_bytes()` | ✓ (via `LayerState`) | ✓ (via `LayerState`) | Shared base |

**Summary:** The two classes share only the `LayerState` interface (`reset_sequence`, `memory_bytes`). `CompressedKVStore` has zero ggml methods — it is a pure-CPU byte-buffer store. `simple_kv_cache` has zero compression methods. A `KVCacheBase` that both implement would require stub-throwing every ggml method in `CompressedKVStore` and every compress/decompress method in `simple_kv_cache` — a leaky abstraction with no call site that would use both variants polymorphically.

---

## 3. Caller survey

All three attention builders hard-code `simple_kv_cache*`:

```
src/layers/attention.h:53   build_attention(…, simple_kv_cache* kv_cache, …)
src/layers/attention.h:73   build_batched_attention(…, simple_kv_cache* kv_cache, …)
src/layers/attention.h:96   build_gated_attention(…, simple_kv_cache* kv_cache, …)
src/layers/attention.h:227  AttentionLayer::kv_cache_  (member field)
```

`CompressedKVStore` is never passed to any attention builder. It is accessed only in the CPU-side calls `_tq_decompress_layer()` / `_tq_compress_new()` that bracket each ggml graph compute call.

**Cache selection:** Runtime dispatch via `kv_quant_bits` constructor parameter (`src/models/qwen3.cpp:22–95`):

- `kv_quant_bits < 2` → allocates `kv_cache_` (`simple_kv_cache`), leaves `tq_store_` null
- `kv_quant_bits ≥ 2` → allocates `tq_store_` (`CompressedKVStore`) + `tq_scratch_cache_` (`simple_kv_cache`), leaves `kv_cache_` null

Dispatch in `run_prefill` (`src/models/qwen3.cpp:847`): `if (!tq_store_)` selects the monolithic path; `else` selects the per-batch TQ path.

---

## 4. Recommendation

**Path A — extract a shared per-layer builder, single mechanical PR.**

The per-layer loop body is identical in both prefill paths (Step 1 found zero class-(b) differences). Both already accept `simple_kv_cache*` — no new polymorphic interface is needed. The fix is to extract a shared free function:

```cpp
// proposed
static ggml_tensor* build_transformer_layer(
    ggml_context*, ggml_cgraph*, simple_kv_cache*,
    ggml_tensor* cur, ggml_tensor* inp_pos,
    const TransformerBlock&, const ModelMetadata&,
    uint32_t il, uint32_t slot_idx, uint32_t n_tokens);
```

`build_prefill_graph` and `_build_layer_batch_graph` call this helper and retain their own outer structure (the monolithic graph scope vs partial-graph scope is load-bearing and must stay separate). The tensor-naming verbosity difference is incidental and should be harmonised during the extraction.

**Why not Path B:** No design question remains open. The graph-shape difference (class b) is zero inside the loop; the outer graph scope difference is intentional and well-understood. A design doc would not resolve anything not already visible from the diff.

**Why not Path C:** The duplication is not load-bearing. Both loops compute the same nodes; keeping them separate serves no purpose and makes future attention or FFN changes require two edits.

**Why `CompressedKVStore` stays separate from `simple_kv_cache`:** They are complementary, not interchangeable. `CompressedKVStore` is a CPU long-term backing store; `simple_kv_cache` is ggml working memory. No call site needs to treat them polymorphically. This is the correct resolution of the Phase 2 interface question deferred in `docs/plan-modular-layer-arch-impl.md` PR 1.2 — *"kept as two concrete implementations of a common interface"* means `LayerState` only, not a shared `KVCacheBase`.

---

## 5. Open questions

1. ~~**Tensor naming strategy**~~ — **Closed.** Verbose naming chosen: `build_transformer_layer` adopts the `build_prefill_graph` convention (`set_name("Qcur.N")`, `"ffn_inp.N"`, etc.) for both prefill paths. Graph naming is build-time only; the per-`ggml_set_name` call overhead in the TQ hot path is negligible. This also improves TQ debuggability, which previously had near-zero intermediate naming.

2. ~~**`_build_single_layer_graph` vs `_build_layer_batch_graph`**~~ — **Closed. Deleted.** `Qwen3ForwardPass::_build_single_layer_graph` (previously at `src/models/qwen3.cpp:660`, declared in `src/models/qwen3.h`) had zero callers outside its own definition. Both the definition and declaration were removed as part of the `build_transformer_layer` extraction PR.

3. ~~**Test coverage of the TQ path**~~ — **Closed.** `tests/unit/test_qwen3_tq_prefill.cpp` adds a canary baseline regression test for `_build_layer_batch_graph` / `run_prefill` with `kv_quant_bits=2`. The baseline (`tests/fixtures/qwen3_tq_prefill_baseline.json`) is committed; the test passed 3× locally confirming determinism. The extraction PR for `build_transformer_layer` can now proceed with a regression gate in place.

4. **SnapKV interaction** — `run_prefill` applies SnapKV after graph compute (`src/models/qwen3.cpp:844`). Does SnapKV compact `tq_scratch_cache_`, `tq_store_`, or both? The compaction signatures differ (see Step 2). Confirm the compact call site before the extraction PR touches the outer graph structure.

5. **Batched-decode TQ path** — Decode always uses `kv_cache_`. If a TQ-enabled session needs batched decode, does it decompress all context into scratch first, or is batched decode unsupported under TQ? This is not documented and affects whether the decode path unification question will resurface later.
