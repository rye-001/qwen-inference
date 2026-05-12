# ANE-Offloaded LM Head

**Goal:** Run the LM-head matmul on the Apple Neural Engine (ANE) instead of the GPU, freeing GPU cycles for the next token's body and pipelining the two compute units.
**Status:** **Speculative / research-flavored.** Lower priority than the grammar-guided decode plan ([docs/plan-grammar-guided-decode.md](plan-grammar-guided-decode.md)). Do that first; revisit this when bored.
**Impact (estimated, Qwen 3.6-A3B):** ~10–15% more tok/s if pipelining materializes. Pure latency win is smaller; the real value is freed GPU bandwidth in multi-tenant serving.

---

## Why this exists at all

Every M-series Mac has three compute units sharing unified memory: CPU, GPU, ANE. Today ggml-metal uses only the GPU. The ANE is idle during inference. That's free silicon nobody in the open-source LLM space (llama.cpp, MLX, vLLM-equivalents) is using.

**The LM head is the right candidate** because:
- It's one big matmul (`[vocab × hidden]` ≈ 150K × 4K on Qwen 3.6) — exactly ANE's shape sweet spot.
- It's the only forward-pass node with no dependency on layer-module internals.
- On MoE models (A3B body activates ~3B of 35B), the LM head is ~10–15% of forward cost, so the win is real.
- It's at the end of the graph, so it pipelines cleanly: ANE runs LM head for token N while GPU starts the body for token N+1.

**Why this is "weird, riskier, more interesting":**
- ANE isn't programmable like Metal. You feed it precompiled CoreML mlpackages, not kernels.
- No public API mixes ggml + CoreML in one graph; the offload is at the C++ level, not the compute-graph level.
- Apple's CoreML scheduler decides per-op which compute unit runs — `CPU_AND_NE` is a *request*, not a guarantee. Verifying ANE actually engages requires Instruments profiling.

---

## Architectural framing

This is a **new backend** parallel to `src/metal/`, not a modification to any existing layer or recipe. Same discipline applies:

- The ANE backend's only consumer is the LM-head node in the recipe. No layer module is aware.
- The dispatch decision (`use_ane_lm_head = true|false`) lives at recipe build time. Decode loop is unchanged in shape.
- Backend stays optional — Intel Macs and pre-macOS14 builds use GPU as today. Build flag: `QINF_ANE`.

Composition with [plan-grammar-guided-decode.md](plan-grammar-guided-decode.md) Phase A (sparse LM head):
- Static-shape mlpackages cannot naturally express dynamic-`k` `get_rows`. Two workable patterns:
  - **Mask after full ANE matmul.** Always do dense ANE matmul; mask the result on CPU. Works if ANE is fast enough that wasted work doesn't matter.
  - **Bucketed mlpackages.** Precompile mlpackages for `k≤32`, `k≤256`, dense. Dispatch to the right one. Ugly but mechanical.

Decision deferred — measure dense-ANE first.

---

## Phase 0 — Conversion script and correctness gate

Before any C++ work, prove the math survives the conversion.

**Offline Python script** (one-time per model, lives in `tools/convert_lm_head_ane.py`):

```python
import torch, coremltools as ct, numpy as np

weight = load_output_weight_from_gguf("qwen36.gguf")   # [vocab, hidden], fp16
vocab, hidden = weight.shape

class LMHead(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab, bias=False)
        self.linear.weight.data = torch.from_numpy(w)
    def forward(self, x):
        return self.linear(x)

model = LMHead(weight).eval()
example = torch.zeros(1, hidden, dtype=torch.float16)
traced = torch.jit.trace(model, example)

ml = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, hidden), dtype=np.float16)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.macOS14,
    convert_to="mlprogram",
)
ml = ct.optimize.coreml.linear_quantize_weights(ml, mode="linear_symmetric", dtype=np.int8)
ml.save("qwen36_lm_head.mlpackage")
```

**Correctness gate:** before trusting the mlpackage, run a small bench script that compares ggml's `output_weight @ hidden` output to CoreML's output across ~100 random `hidden` vectors. They must agree within fp16 tolerance (~1e-3 relative). Axis order is the usual culprit — fix it here, not later.

**Time:** ~1 day if `coremltools` cooperates.
**Output of this phase:** a `qwen36_lm_head.mlpackage` shipping next to `qwen36.gguf`, plus a verification report.

---

## Phase 1 — Single-token through ANE, end-to-end

**Goal:** prove the ANE path works at all. Don't worry about speed yet.

New files:
- `src/coreml/CMakeLists.txt`
- `src/coreml/lm_head_ane.h` — clean C++ API: `class AneLmHead { std::vector<float> forward(const float* hidden, size_t hidden_size); };`
- `src/coreml/lm_head_ane.mm` — Objective-C++ implementation calling CoreML's `MLModel` API.
- `src/coreml/README.md` — what this backend does and doesn't do.

Existing files touched:
- `CMakeLists.txt` — add `QINF_ANE` build flag, link CoreML framework conditionally.
- `src/models/forward_pass_base.{h,cpp}` — `build_output_head()` gains a branch: if `ane_lm_head_ != nullptr`, skip the ggml matmul and tag the graph output as "needs ANE post-processing"; the decode loop runs the ANE step after `ggml_graph_compute`.
- `src/models/qwen36.cpp` — first opt-in recipe; instantiate `AneLmHead` at model load if enabled.

Build-flag default: **off**. ANE is opt-in until proven.

**Done criteria:**
- End-to-end generation works with `QINF_ANE=ON` on macOS 14+.
- Output tokens identical to GPU-only path within fp16 tolerance.
- `Instruments → Core ML` profile confirms the LM head ops are actually running on ANE, not falling back to GPU/CPU. (This is non-negotiable. If the scheduler ignores our request and runs it on GPU, the whole exercise is moot.)

**Time:** ~1 week.

---

## Phase 2 — Pipelining (where the win actually comes from)

Single-token ANE will probably show **no speedup** — possibly a slight regression — because the tensor handoff (read back ~600KB of logits per token at fp16) is non-trivial. That's expected. The win is **pipelining**: ANE runs LM head for token N while GPU runs body for token N+1.

This requires:
- The decode loop to dispatch the ANE call *asynchronously* — start the ANE forward, immediately begin building the next token's GPU graph, then sync on ANE before sampling.
- One token of extra latency on the first decode step (the pipeline has to fill). Negligible.
- Two-deep KV cache write ordering — token N's KV is written by token N's GPU body, not by the ANE LM head. So KV ordering is unchanged. Good.

Files touched:
- `src/cli/chat.cpp`, `src/cli/complete.cpp`, `src/server/http_server.cpp` — the same three decode loops as the grammar-guided plan. If that plan landed first and lifted the duplication into `src/core/`, this is a one-file change instead of three.
- `src/coreml/lm_head_ane.{h,mm}` — add async API: `submit(hidden) -> handle`, `wait(handle) -> logits`.

**Done criteria:**
- End-to-end tok/s on Qwen 3.6-A3B improves by ≥10% vs GPU-only baseline.
- Multi-tenant throughput (N concurrent sessions) improves by ≥15% — the GPU cycles freed are exactly the kind of slack multi-tenant serving consumes.
- Profile shows GPU and ANE genuinely overlapping in time (not sequential).

**Time:** ~1–2 weeks.

---

## Kill criterion

If after Phase 2 end-to-end tok/s improves by **<10%**, kill the backend. Reasons it might fail:
- Tensor handoff overhead (logits read-back) eats the gain.
- Apple's scheduler keeps deciding to run the op on GPU anyway (this happens — Instruments tells you).
- Pipelining doesn't materialize because the GPU body and ANE LM head aren't independent enough at the framework level.

ANE work is the kind of project that becomes a tarpit without a pre-committed kill criterion. **Pre-commit to <10% = abandon.**

The architectural cost of carrying a third backend is real: an extra ~600 lines of `.mm` code, an extra mlpackage per supported model, a Python conversion script in CI, and a build-flag matrix that doubles. None of that is justified by a 5% win.

---

## Wrinkles to be honest about

1. **ANE scheduling is opaque.** `CPU_AND_NE` is a request. Apple decides per-op. You don't get a knob to *force* ANE — you get heuristics (use fp16, avoid unsupported ops, keep shapes stable) and Instruments to verify.

2. **K-Quants don't survive conversion.** The ANE copy of the LM head is fp16 or int8, separate from the GGUF's K-Quant copy. Carry ~300–600 MB extra weight memory per model. Acceptable for the target use case but not free.

3. **CoreML versioning.** mlpackages have a `minimum_deployment_target`. macOS 14 is a reasonable floor (current is 15). Pre-14 macs fall back to GPU. Document this clearly.

4. **Intel Macs are out.** ANE is M-series only. Build flag must compile out cleanly on Intel.

5. **Sparse LM head composition is unresolved.** See "Architectural framing" above. Don't try to solve this in Phase 0–2; revisit after Phase 2 lands if the grammar plan's Phase A is already in place.

6. **No public mixed ggml/CoreML graph.** The offload is at the C++ call-graph level, not the compute-graph level. ggml runs the body, ANE runs the head, the decode loop stitches them. This is the only honest way; don't try to be clever.

---

## Why this stays in `docs/` and not in `src/` for now

Because the higher-leverage work is the grammar-guided decode plan, and because the kill-criterion for this one is real. We document so the idea isn't lost; we don't build until the cheaper experiments are done and we have appetite for a research-flavored side quest.

If/when this lands and works, it's a **defensible differentiation** — "Qwenium uses the whole chip; everyone else uses 1/3 of it." That's worth a blog post and probably a workshop paper. But not worth derailing the architecture for.

---

## Related

- [docs/plan-grammar-guided-decode.md](plan-grammar-guided-decode.md) — do this first. Lifts decode-loop duplication that this plan benefits from.
- [docs/future-work.md](future-work.md) — companion entry to be added linking to this plan.
- Apple docs: [Optimizing Models for the Neural Engine](https://developer.apple.com/documentation/coreml/optimizing-models-for-the-neural-engine), [coremltools](https://apple.github.io/coremltools/).
