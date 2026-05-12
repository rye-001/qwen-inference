#pragma once
// breakdown_harness.h — shared timing harness for Phase 4 per-stage breakdowns.
//
// Methodology mirrored from the original moe_breakdown.cpp: build cumulative
// sub-graphs that terminate one stage later than the previous, time each
// (median over MOE_ITERS / *_ITERS samples after warmup), report stage cost as
// median(g_N) − median(g_{N-1}). Caveats from the moe report still apply:
//
//   - The first stage's cumulative absorbs per-iter scheduler overhead
//     (sched_reset + alloc_graph + tensor_set + sched_graph_compute + sync).
//     Stage deltas after the first cancel that overhead out.
//   - Stage attribution can be fuzzy when the Metal command scheduler runs
//     adjacent ops in parallel within a single command buffer (gate/up in MoE
//     was the canonical example). The total cumulative is robust; sub-stage
//     deltas are indicative.
//
// Public surface:
//   BuiltGraph       — owns ctx + gf + meta_buf + tagged input/output
//   median()         — sorted-middle helper
//   time_subgraph()  — sched_reserve + warmup + iters of run_once()
//
// Each breakdown binary defines its own Stage enum and build_subgraph(stop)
// closure; this header is the cross-cutting glue.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace breakdown {

using Clock = std::chrono::steady_clock;

struct BuiltGraph {
    ggml_context*        ctx     = nullptr;
    ggml_cgraph*         gf      = nullptr;
    ggml_tensor*         input   = nullptr;
    ggml_tensor*         output  = nullptr;
    std::vector<uint8_t> meta_buf;
};

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Time a single sub-graph: reserve scheduler buffers for its shape, warm up,
// then collect `iters` median samples in milliseconds.
//
// upload_input: optional callback invoked after sched_alloc_graph and before
//   sched_graph_compute. Use it to ggml_backend_tensor_set the input(s) for
//   the current iteration. Pass an empty std::function if your sub-graph has
//   no inputs to upload (e.g. fully-synthetic graphs reading from constants).
inline double time_subgraph(
    BuiltGraph&                         bg,
    ggml_backend_sched_t                sched,
    const std::function<void()>&        upload_input,
    int                                 warmup,
    int                                 iters,
    const char*                         label = nullptr)
{
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_reserve(sched, bg.gf)) {
        std::cerr << "breakdown_harness: ggml_backend_sched_reserve failed"
                  << (label ? std::string(" for ") + label : std::string())
                  << "\n";
        std::exit(1);
    }

    auto run_once = [&]() {
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, bg.gf);
        if (upload_input) upload_input();
        ggml_backend_sched_graph_compute(sched, bg.gf);
        ggml_backend_sched_synchronize(sched);
    };

    for (int i = 0; i < warmup; ++i) run_once();

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        run_once();
        auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median(std::move(samples));
}

// Allocate a fresh BuiltGraph with an empty cgraph and a generous meta buffer.
// Caller fills bg.input / bg.output and calls ggml_build_forward_expand on the
// terminating tensor before passing to time_subgraph.
inline BuiltGraph make_graph(size_t meta_bytes = 64 * 1024 * 1024,
                             int    max_nodes  = 16384)
{
    BuiltGraph bg;
    bg.meta_buf.resize(meta_bytes);

    ggml_init_params ip{};
    ip.mem_size   = bg.meta_buf.size();
    ip.mem_buffer = bg.meta_buf.data();
    ip.no_alloc   = true;
    bg.ctx = ggml_init(ip);
    bg.gf  = ggml_new_graph_custom(bg.ctx, max_nodes, false);
    return bg;
}

}  // namespace breakdown
