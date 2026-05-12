// test_deltanet_pre_state.cpp
//
// Equivalence test for ggml_deltanet_pre_state (qinf downstream-patched op).
//
// Reference: the chained 4-op graph this op fuses replaces:
//     q_n = l2_norm(q, eps)
//     k_n = l2_norm(k, eps)
//     q_r = repeat_4d(q_n, head_k_dim, num_v_heads, n_seq_tokens, n_seqs)  // tile semantics
//     k_r = repeat_4d(k_n, head_k_dim, num_v_heads, n_seq_tokens, n_seqs)
// The fused op packs (q_r, k_r) into one tensor of shape
// [head_k_dim, 2 * num_v_heads, n_seq_tokens, n_seqs], with rows
// [0, num_v_heads) = q_r and [num_v_heads, 2 * num_v_heads) = k_r.
//
// This test builds both graphs on the parameterized backend with the same
// inputs and asserts elementwise agreement within a small tolerance.
// Tolerance band matches the post-state test (consistent with the
// gated_delta_net upstream test's approximate-equality contract).
//
// Two configurations are exercised:
//   1. GQA decode shape (num_k_heads=16, num_v_heads=32 — Qwen 3.6).
//   2. Symmetric shape  (num_k_heads=num_v_heads=32 — fast path, no repeat).
//
// No model file required.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

namespace {

constexpr int64_t HEAD_K_DIM    = 128;
constexpr int64_t N_SEQ_TOKENS  = 1;
constexpr int64_t N_SEQS        = 1;
constexpr float   EPS           = 1e-5f;

constexpr float ABS_TOL = 1e-5f;
constexpr float REL_TOL = 1e-4f;

struct Inputs {
    std::vector<float> q;       // [head_k_dim, num_k_heads, n_seq_tokens, n_seqs]
    std::vector<float> k;       // same shape as q
    int64_t            num_k_heads;
    int64_t            num_v_heads;
};

Inputs make_random_inputs(uint32_t seed, int64_t num_k_heads, int64_t num_v_heads) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);

    const size_t n = static_cast<size_t>(HEAD_K_DIM) * num_k_heads * N_SEQ_TOKENS * N_SEQS;

    Inputs in{
        .q = std::vector<float>(n),
        .k = std::vector<float>(n),
        .num_k_heads = num_k_heads,
        .num_v_heads = num_v_heads,
    };
    for (auto& v : in.q) v = d(rng);
    for (auto& v : in.k) v = d(rng);
    return in;
}

ggml_tensor* new_qk_input(ggml_context* ctx, int64_t num_k_heads, const char* name) {
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        HEAD_K_DIM, num_k_heads, N_SEQ_TOKENS, N_SEQS);
    ggml_set_input(t);
    ggml_set_name(t, name);
    return t;
}

// Reference: l2_norm × 2 + repeat_4d × 2 (when num_k_heads != num_v_heads),
// then concat along dim 1 to match the fused op's packed output layout.
ggml_tensor* build_reference_chain(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  q,
    ggml_tensor*  k,
    int64_t       num_v_heads,
    float         eps) {

    ggml_tensor* q_n = ggml_l2_norm(ctx, q, eps);
    ggml_tensor* k_n = ggml_l2_norm(ctx, k, eps);

    if (q->ne[1] != num_v_heads) {
        q_n = ggml_repeat_4d(ctx, q_n, HEAD_K_DIM, num_v_heads, N_SEQ_TOKENS, N_SEQS);
        k_n = ggml_repeat_4d(ctx, k_n, HEAD_K_DIM, num_v_heads, N_SEQ_TOKENS, N_SEQS);
    }

    // Make both contiguous before concat — ggml_concat requires it for matching strides.
    q_n = ggml_cont(ctx, q_n);
    k_n = ggml_cont(ctx, k_n);

    ggml_tensor* packed = ggml_concat(ctx, q_n, k_n, 1);
    ggml_set_name(packed, "qk_packed_ref");
    ggml_build_forward_expand(gf, packed);
    return packed;
}

ggml_tensor* build_fused(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  q,
    ggml_tensor*  k,
    int64_t       num_v_heads,
    float         eps) {

    ggml_tensor* packed = ggml_deltanet_pre_state(ctx, q, k, static_cast<int32_t>(num_v_heads), eps);
    ggml_set_name(packed, "qk_packed_fused");
    ggml_build_forward_expand(gf, packed);
    return packed;
}

std::vector<float> run_graph(
    ggml_backend_t      backend,
    ggml_cgraph*        gf,
    ggml_context*       ctx,
    ggml_tensor*        out,
    const Inputs&       in) {

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    EXPECT_NE(buf, nullptr);

    ggml_tensor* q = ggml_graph_get_tensor(gf, "q");
    ggml_tensor* k = ggml_graph_get_tensor(gf, "k");
    EXPECT_NE(q, nullptr);
    EXPECT_NE(k, nullptr);

    ggml_backend_tensor_set(q, in.q.data(), 0, in.q.size() * sizeof(float));
    ggml_backend_tensor_set(k, in.k.data(), 0, in.k.size() * sizeof(float));

    EXPECT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    return result;
}

enum class BackendKind { CPU, Metal };

const char* backend_name(BackendKind k) {
    switch (k) {
        case BackendKind::CPU:   return "CPU";
        case BackendKind::Metal: return "Metal";
    }
    return "?";
}

ggml_backend_t make_backend(BackendKind k) {
    switch (k) {
        case BackendKind::CPU:
            return ggml_backend_cpu_init();
        case BackendKind::Metal:
#ifdef GGML_USE_METAL
            return ggml_backend_metal_init();
#else
            return nullptr;
#endif
    }
    return nullptr;
}

class DeltaNetPreStateTest : public ::testing::TestWithParam<BackendKind> {
protected:
    void SetUp() override {
        backend_ = make_backend(GetParam());
        if (!backend_) {
            GTEST_SKIP() << "Backend " << backend_name(GetParam()) << " not available";
        }
    }
    void TearDown() override {
        if (backend_) ggml_backend_free(backend_);
    }
    ggml_backend_t backend_ = nullptr;

    void run_case(int64_t num_k_heads, int64_t num_v_heads, uint32_t seed, const char* label) {
        Inputs in = make_random_inputs(seed, num_k_heads, num_v_heads);

        // Reference graph (CPU oracle).
        std::vector<float> ref_out;
        {
            ggml_backend_t cpu = ggml_backend_cpu_init();
            ASSERT_NE(cpu, nullptr);

            ggml_init_params p{1024 * 1024, nullptr, true};
            ggml_context* ctx = ggml_init(p);
            ASSERT_NE(ctx, nullptr);

            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_tensor* q = new_qk_input(ctx, num_k_heads, "q");
            ggml_tensor* k = new_qk_input(ctx, num_k_heads, "k");

            ggml_tensor* out = build_reference_chain(ctx, gf, q, k, num_v_heads, EPS);
            ref_out = run_graph(cpu, gf, ctx, out, in);
            ggml_free(ctx);
            ggml_backend_free(cpu);
        }

        // Fused graph on backend_.
        std::vector<float> fused_out;
        {
            ggml_init_params p{1024 * 1024, nullptr, true};
            ggml_context* ctx = ggml_init(p);
            ASSERT_NE(ctx, nullptr);

            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_tensor* q = new_qk_input(ctx, num_k_heads, "q");
            ggml_tensor* k = new_qk_input(ctx, num_k_heads, "k");

            ggml_tensor* out = build_fused(ctx, gf, q, k, num_v_heads, EPS);
            fused_out = run_graph(backend_, gf, ctx, out, in);
            ggml_free(ctx);
        }

        ASSERT_EQ(ref_out.size(), fused_out.size());

        double max_abs_err = 0.0;
        double max_rel_err = 0.0;
        for (size_t i = 0; i < ref_out.size(); ++i) {
            const double a = ref_out[i];
            const double b = fused_out[i];
            const double abs_err = std::abs(a - b);
            const double rel_err = abs_err / std::max(std::abs(a), 1e-30);
            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, rel_err);
            ASSERT_TRUE(abs_err <= ABS_TOL || rel_err <= REL_TOL)
                << "backend=" << backend_name(GetParam())
                << " case=" << label
                << " i=" << i << " ref=" << a << " fused=" << b
                << " abs=" << abs_err << " rel=" << rel_err;
        }
        std::cerr << "  [" << backend_name(GetParam()) << "] " << label
                  << " " << HEAD_K_DIM << "×" << num_k_heads << "→" << num_v_heads
                  << " — max_abs=" << max_abs_err
                  << " max_rel=" << max_rel_err << "\n";
    }
};

TEST_P(DeltaNetPreStateTest, MatchesChainedReference_GQADecode) {
    run_case(/*num_k_heads=*/16, /*num_v_heads=*/32, 0xCAFE, "GQA");
}

TEST_P(DeltaNetPreStateTest, MatchesChainedReference_NoRepeat) {
    run_case(/*num_k_heads=*/32, /*num_v_heads=*/32, 0xC0DE, "no-repeat");
}

TEST_P(DeltaNetPreStateTest, FiniteOutput_GQA) {
    Inputs in = make_random_inputs(0xDEADBEEFu, 16, 32);

    ggml_init_params p{1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* q = new_qk_input(ctx, 16, "q");
    ggml_tensor* k = new_qk_input(ctx, 16, "k");

    ggml_tensor* out = build_fused(ctx, gf, q, k, 32, EPS);
    auto vals = run_graph(backend_, gf, ctx, out, in);

    for (size_t i = 0; i < vals.size(); ++i) {
        ASSERT_TRUE(std::isfinite(vals[i])) << "non-finite at i=" << i << " value=" << vals[i];
    }
    ggml_free(ctx);
}

INSTANTIATE_TEST_SUITE_P(
    Backends,
    DeltaNetPreStateTest,
    ::testing::Values(BackendKind::CPU, BackendKind::Metal),
    [](const ::testing::TestParamInfo<BackendKind>& info) {
        return std::string(backend_name(info.param));
    });

}  // namespace
