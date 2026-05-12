# PR 4.2 — Fused DeltaNet Decode-Step Kernel(s)

**Goal:** Reduce per-layer DeltaNet decode cost from ~0.75 ms to ~0.25 ms by fusing the small "epoxy" ops (l2_norm, repeat, rms_norm, broadcast mul, silu, state cpy) around the two existing heavy kernels (`ggml_ssm_conv`, `ggml_gated_delta_net`).
**Impact:** End-to-end ceiling ≈ **1.46×** decode tok/s on Qwen 3.6 (50.5 → ~35 ms/token). See [docs/phase4-investigation.md → DeltaNet sub-stage findings](phase4-investigation.md#deltanet-sub-stage-findings--q1-resolved-launch-bound) for the measurement that motivates this PR.

---

## What we're fusing

The 6-stage breakdown (per token, per DeltaNet layer):

| Stage        | Δ (µs) | Heavy compute                  | Small "epoxy" ops we want to absorb |
|---|---:|---|---|
| qkv_proj     | ~250 (true cost)¹ | `mul_mat(w_qkv)`         | reshape (free) |
| gate_proj    |  ~75 (estimated)¹ | `mul_mat(w_gate, w_beta, w_a)` | sigmoid, softplus, mul, add |
| conv_update  | 137 | `ggml_ssm_conv`                | `concat`, `cpy` (state writeback), `silu` |
| state_update | 199 | `ggml_gated_delta_net`         | `l2_norm` × 2, `repeat_4d` × 2, `cpy` (state writeback) |
| out_norm     |  21 | (none — all small)             | `rms_norm`, `mul(w_norm)`, `silu(z)`, `mul` |
| out_proj     |  63 | `mul_mat(w_out)`               | reshape (free) |

¹ qkv_proj/gate_proj true costs are inferred — the cumulative breakdown's stage 0 absorbs the JIT-compile burst, and stage 1 went negative as a result.

**The matmuls (`mul_mat`, `ssm_conv`, `gated_delta_net`) stay as separate kernels.** Each is already one well-tuned Metal dispatch; lumping them into one mega-kernel would just lower occupancy. **What we fuse is the ~10 small ops between them**, where dispatch overhead dominates compute.

---

## Architectural decision: split, not parameterize

Per CLAUDE.md's *parameterize-vs-split* test, we explicitly choose **split** here:

- The fused ops are pure pre/post-processing of two existing kernels (`ssm_conv`, `gated_delta_net`). They have no other natural callers in the codebase.
- Folding them into the existing kernels (parameterize) would bloat the signatures of those kernels with optional flags (`do_l2_norm_inputs`, `do_rms_norm_output`, etc.), which is exactly the failure mode CLAUDE.md warns against ("a fat function whose params are orthogonal silos that no single call site uses together").
- Two new bespoke fused ops, each one tightly scoped to "the small ops adjacent to one heavy kernel," is the cleaner seam.

**The candidate ops:**

1. **`ggml_deltanet_pre_state`** — fuses pre-`gated_delta_net` epoxy.
   Inputs: `Q_conv`, `K_conv` (views into ssm_conv output)
   Output: `Q_l2_repeated`, `K_l2_repeated` (one tensor per output, or one packed tensor)
   Internal: l2_norm × 2 → repeat_4d × 2.
   Replaces 4 dispatches with 1.

2. **`ggml_deltanet_post_state`** — fuses post-`gated_delta_net` epoxy.
   Inputs: gated_delta_net output (per-token), `z` (output gate from `mul_mat(w_gate)`), `w_norm`
   Output: `gated` (ready for `mul_mat(w_out)`)
   Internal: rms_norm → broadcast mul(w_norm) → silu(z) → mul.
   Replaces 4 dispatches with 1.

**Conv-side fusion is deferred** as a stretch goal (PR 4.2.C). The conv stage is 137 µs; absorbing concat + cpy + silu around `ssm_conv` is harder (needs new state-writeback semantics in the kernel) and saves at most ~50 µs. Land 4.2.A and 4.2.B first; re-measure; decide whether 4.2.C is worth the surgery.

---

## Sub-PR breakdown

**Each sub-PR is independently mergeable**. After each one, re-run `deltanet-substage-breakdown` and confirm the predicted delta drop. Stop early if cumulative speedup hits diminishing returns.

### PR 4.2.A — `ggml_deltanet_post_state` (out_norm + gate)
Smallest surgery, exercises the integration pattern end-to-end.
- Patch vendored ggml: header decl, CPU reference impl, op enum, builder.
- Add Metal kernel `kernel_deltanet_post_state` in `ggml-metal.metal`.
- Add dispatch case in `ggml-metal-ops.cpp`, pipeline lookup in `device.cpp`, supports_op gate in `ggml-metal-device.m`.
- Update `src/layers/deltanet.cpp` to call new op instead of the 4-op chain.
- Bit-for-bit equivalence test: `tests/unit/test_deltanet_post_state.cpp` against the reference 4-op graph at decode shape.
- Re-measure. Expected: out_norm stage delta drops from 21 µs → ~3 µs.

### PR 4.2.B — `ggml_deltanet_pre_state` (l2_norm + repeat × 2)
- Same five-file patch pattern as 4.2.A.
- Trickier because it produces two outputs (Q and K). Two options:
  - (a) Pack Q and K into one output tensor, view it on the consumer side.
  - (b) Two sibling ops `_pre_state_q` and `_pre_state_k`. Simpler kernel each, more dispatch overhead.
  - **Recommend (a)** — packing is cheap, halves dispatch count.
- Equivalence test in `tests/unit/test_deltanet_pre_state.cpp`.
- Re-measure. Expected: state_update stage delta drops from 199 µs → ~120 µs (the `gated_delta_net` cost itself, which is the irreducible part).

### PR 4.2.C (stretch) — `ggml_deltanet_conv_step`
- Absorbs the `concat` + `cpy` (conv state writeback) + `silu` around `ssm_conv`.
- Significantly more surgery: the kernel must do an in-place state shift on its weight buffer, which `ssm_conv` currently doesn't.
- **Decision gate:** only attempt if 4.2.A + 4.2.B leave >250 µs/layer on the table. Otherwise close as "no further action — diminishing returns."

---

## Risks and unknowns

1. **Quantized loads.** `w_norm` is typically F32 in our checkpoints (norm weights are tiny, not worth quantizing), but verify in the q2_K_XL inventory. If quantized, the kernel must handle the unpack — adds ~50 lines of shader code per quant scheme. Check before starting 4.2.A.

2. **Numerical drift.** Chained ops in one kernel rounded once differ from the same chain rounded between dispatches. The equivalence tests should use a relative-error tolerance (~1e-5 for F32), not bit-exactness. We have precedent — `gated_delta_net`'s test is approximate, not exact.

3. **State writeback semantics.** Both `pre_state` and `post_state` are pure (no side effects), so this is only a 4.2.C concern. Note for 4.2.C: the conv state writeback is a `ggml_cpy` of a *view* of the conv kernel's input back into a *separate* persistent tensor — fusing it requires the kernel to receive two output buffers (one transient, one persistent state). Non-trivial.

4. **"Will this generalize beyond Qwen 3.6?"** GatedDeltaNet is currently a Qwen 3.6 feature. If Qwen 3.7 changes the gating arithmetic, our fused kernels become dead weight. Mitigation: keep the reference 4-op chain as a `QINF_DN_FALLBACK` CMake flag (paralleling the existing `QINF_MOE_FALLBACK` pattern noted in memory). If the fused kernels diverge from a future model, flip the flag and ship the fallback — the original ops are still there in vendored ggml.

5. **Vendored-ggml diff drift.** Each new op puts us further from upstream ggml. This is already true (`gated_delta_net` is non-upstream). Mitigation: keep all our diffs under one well-named patch range so a future ggml bump can be replayed.

---

## Falsification plan

A speedup is only real if it shows up in **two** independent measurements:

1. **Per-layer breakdown** — `deltanet-substage-breakdown` re-run after each sub-PR. Stage delta(s) for the targeted stage must drop within 20% of prediction.
2. **End-to-end decode tok/s** — `perf-baseline` (PR 4.0 binary) before/after. Improvement must be at least 50% of the per-layer delta × 30 layers, or we have a methodology problem (e.g. real forward pass overlaps work that the breakdown serializes).

If 4.2.A lands and tok/s does not move within (1) and (2), **stop** and investigate. Do not pile on 4.2.B and 4.2.C hoping speedups compose — they only compose if each one is real in isolation.

---

## Vendored-ggml patch delivery — **decision required before code lands**

Initial draft of this plan assumed `ggml_gated_delta_net` was a project-local patch we could mimic. **That assumption was wrong.** Verification (2026-05-08): vendored ggml is at clean upstream tag `b8390` (`HEAD detached at b8390`, `nothing to commit`). `gated_delta_net` is upstream ggml, not a project patch. There is no five-file patch precedent in this repo.

To add `ggml_deltanet_post_state`, we must pick a delivery mechanism:

| Option | Mechanism | Maintenance cost | Verdict |
|---|---|---|---|
| **A. Fork ggml** | Replace `FetchContent` URL with our fork; rebase per upstream bump | Heavy — every bump = a real merge | Heavy but well-understood |
| **B. CMake `PATCH_COMMAND`** | Keep upstream URL+tag pinned, ship `patches/*.patch` files, apply at fetch | One-time infra (~30 lines CMake), then per-feature one patch file | **Recommended** |
| C. Hand-rolled MTLCommandBuffer | Bypass ggml scheduler just for DeltaNet decode; dispatch our own Metal kernels | Significant new dispatch surface in `src/metal/` | Avoids ggml diff at the cost of duplicating scheduler/allocator concerns |
| D. Upstream to ggml | PR `ggml_deltanet_post_state` to llama.cpp | Long timeline, no guarantee | Dead end for our timeline |
| E. Graph-level only | Drop kernel-level fusion; pursue `ggml_set_no_alloc`, command-buffer batching, etc. | None | Likely cannot hit 1.46× ceiling, but stays inside existing seam |

**Recommendation: B.** Keeps our "downstream of llama.cpp" story intact, makes our diffs auditable as standalone files in `patches/`, lets us bump the ggml tag without losing our ops (just resolve patch conflicts once). The infra cost is ~30 lines of CMake. Sub-PR 4.2.A would land:

- `patches/0001-ggml-add-deltanet-post-state-op.patch` — single patch covering ggml.h, ggml.c, ops.cpp, ggml-metal.metal, ggml-metal-ops.cpp, ggml-metal-device.m
- `CMakeLists.txt` — add `PATCH_COMMAND` to the existing `FetchContent_Declare`
- `src/layers/deltanet.cpp` — call new op
- `tests/unit/test_deltanet_post_state.cpp` — equivalence test

Pre-flight check before B: confirm `FETCHCONTENT_UPDATES_DISCONNECTED ON` (already set) plays correctly with `PATCH_COMMAND` — patches must apply cleanly on first fetch and be idempotent on subsequent reconfigures. CMake's standard pattern uses `git apply --check || git apply` for this.

## Files touched per sub-PR (under Option B)

```
patches/<NNNN>-ggml-add-<op_name>.patch             (the upstream-style diff)
CMakeLists.txt                                      (PATCH_COMMAND wiring; one-time only)
src/layers/deltanet.cpp                              (call new op instead of N-op chain)
tests/unit/test_deltanet_<post|pre>_state.cpp        (bit-equivalence-style test)
tests/CMakeLists.txt                                 (wire the new test target)
```

---

## Out of scope

- **`ggml_flash_attn_ext` switch for the 10 attention layers** — separate concern (PR 4.4 in `src/metal/CMakeLists.txt`'s comment), unblocked by Q2 in the investigation doc, not by this PR.
- **MoE-side optimization** — already ruled out at 1.13× ceiling. CLAUDE.md notes the only remaining MoE candidate is patching ggml-metal's `MUL_MAT_ID` for sparse routing, which is a Phase 5+ topic.
- **CPU backend kernels.** Decode runs on Metal in production. CPU reference impls only need to be correct, not fast — they exist for the equivalence tests and CPU-only CI.
