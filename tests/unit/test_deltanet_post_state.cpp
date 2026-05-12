// test_deltanet_post_state.cpp
//
// Equivalence test for ggml_deltanet_post_state (qinf downstream-patched op).
//
// Reference: the chained 4-op graph this op fuses replaces:
//     normed = rms_norm(x, eps)
//     normed = normed * w_norm        (broadcast over leading dims)
//     z_silu = silu(z)
//     gated  = normed * z_silu
//
// This test builds both graphs on the CPU backend with the same inputs,
// runs them, and asserts elementwise agreement within a small tolerance.
// Tolerance is loose by design — the fused op accumulates rounding once
// per row rather than once per intermediate tensor, so bit-exactness is
// not expected. 1e-5 absolute / 1e-4 relative is consistent with the
// gated_delta_net upstream test's tolerance band.
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

// Decode-shape parameters mirrored from the Qwen 3.6 GatedDeltaNet path.
constexpr int64_t HEAD_V_DIM    = 128;
constexpr int64_t NUM_V_HEADS   = 32;
constexpr int64_t N_SEQ_TOKENS  = 1;
constexpr int64_t N_SEQS        = 1;
constexpr float   EPS           = 1e-5f;

constexpr float ABS_TOL = 1e-5f;
constexpr float REL_TOL = 1e-4f;

struct Inputs {
    std::vector<float> x;       // [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    std::vector<float> z;       // same shape as x
    std::vector<float> w_norm;  // [head_v_dim]
};

Inputs make_random_inputs(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dx(-0.5f, 0.5f);
    std::uniform_real_distribution<float> dz(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dw(0.5f, 1.5f);

    const size_t n_xz = static_cast<size_t>(HEAD_V_DIM) * NUM_V_HEADS * N_SEQ_TOKENS * N_SEQS;

    Inputs in;
    in.x.resize(n_xz);
    in.z.resize(n_xz);
    in.w_norm.resize(static_cast<size_t>(HEAD_V_DIM));

    for (auto& v : in.x)      v = dx(rng);
    for (auto& v : in.z)      v = dz(rng);
    for (auto& v : in.w_norm) v = dw(rng);
    return in;
}

ggml_tensor* new_xshape(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        HEAD_V_DIM, NUM_V_HEADS, N_SEQ_TOKENS, N_SEQS);
    ggml_set_input(t);
    ggml_set_name(t, name);
    return t;
}

// Build the reference 4-op chain as a sub-graph and return its output tensor.
ggml_tensor* build_reference_chain(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  x,
    ggml_tensor*  z,
    ggml_tensor*  w_norm,
    float         eps) {

    ggml_tensor* normed = ggml_rms_norm(ctx, x, eps);
    normed              = ggml_mul(ctx, normed, w_norm);
    ggml_tensor* z_silu = ggml_silu(ctx, z);
    ggml_tensor* gated  = ggml_mul(ctx, normed, z_silu);
    ggml_set_name(gated, "gated_ref");
    ggml_build_forward_expand(gf, gated);
    return gated;
}

// Build a graph using the fused op.
ggml_tensor* build_fused(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  x,
    ggml_tensor*  z,
    ggml_tensor*  w_norm,
    float         eps) {

    ggml_tensor* gated = ggml_deltanet_post_state(ctx, x, z, w_norm, eps);
    ggml_set_name(gated, "gated_fused");
    ggml_build_forward_expand(gf, gated);
    return gated;
}

// Allocate + run a single-op graph on the CPU backend; return the output as a flat vector.
std::vector<float> run_graph(
    ggml_backend_t      backend,
    ggml_cgraph*        gf,
    ggml_context*       ctx,
    ggml_tensor*        out,
    const Inputs&       in) {

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    EXPECT_NE(buf, nullptr);

    ggml_tensor* x      = ggml_graph_get_tensor(gf, "x");
    ggml_tensor* z      = ggml_graph_get_tensor(gf, "z");
    ggml_tensor* w_norm = ggml_graph_get_tensor(gf, "w_norm");
    EXPECT_NE(x, nullptr);
    EXPECT_NE(z, nullptr);
    EXPECT_NE(w_norm, nullptr);

    ggml_backend_tensor_set(x,      in.x.data(),      0, in.x.size()      * sizeof(float));
    ggml_backend_tensor_set(z,      in.z.data(),      0, in.z.size()      * sizeof(float));
    ggml_backend_tensor_set(w_norm, in.w_norm.data(), 0, in.w_norm.size() * sizeof(float));

    EXPECT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    return result;
}

// Backend factory. Returns nullptr if the requested backend is not available
// (e.g. Metal on a non-Apple build), in which case the parameterized fixture
// self-skips for that backend.
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

class DeltaNetPostStateTest : public ::testing::TestWithParam<BackendKind> {
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
};

TEST_P(DeltaNetPostStateTest, MatchesChainedReference_Decode) {
    Inputs in = make_random_inputs(0xCAFE);

    // ── Reference graph (always CPU — bit-stable oracle) ─────────────────────
    std::vector<float> ref_out;
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ASSERT_NE(cpu, nullptr);

        ggml_init_params p{1024 * 1024, nullptr, true};
        ggml_context* ctx = ggml_init(p);
        ASSERT_NE(ctx, nullptr);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* x      = new_xshape(ctx, "x");
        ggml_tensor* z      = new_xshape(ctx, "z");
        ggml_tensor* w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_V_DIM);
        ggml_set_input(w_norm);
        ggml_set_name(w_norm, "w_norm");

        ggml_tensor* out = build_reference_chain(ctx, gf, x, z, w_norm, EPS);
        ref_out = run_graph(cpu, gf, ctx, out, in);
        ggml_free(ctx);
        ggml_backend_free(cpu);
    }

    // ── Fused graph (under test on backend_) ─────────────────────────────────
    std::vector<float> fused_out;
    {
        ggml_init_params p{1024 * 1024, nullptr, true};
        ggml_context* ctx = ggml_init(p);
        ASSERT_NE(ctx, nullptr);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* x      = new_xshape(ctx, "x");
        ggml_tensor* z      = new_xshape(ctx, "z");
        ggml_tensor* w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_V_DIM);
        ggml_set_input(w_norm);
        ggml_set_name(w_norm, "w_norm");

        ggml_tensor* out = build_fused(ctx, gf, x, z, w_norm, EPS);
        fused_out = run_graph(backend_, gf, ctx, out, in);
        ggml_free(ctx);
    }

    // ── Compare ──────────────────────────────────────────────────────────────
    ASSERT_EQ(ref_out.size(), fused_out.size());

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        const double a   = ref_out[i];
        const double b   = fused_out[i];
        const double abs_err = std::abs(a - b);
        const double rel_err = abs_err / std::max(std::abs(a), 1e-30);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        ASSERT_TRUE(abs_err <= ABS_TOL || rel_err <= REL_TOL)
            << "backend=" << backend_name(GetParam())
            << " i=" << i << " ref=" << a << " fused=" << b
            << " abs=" << abs_err << " rel=" << rel_err;
    }
    std::cerr << "  [" << backend_name(GetParam()) << "] decode-shape "
              << HEAD_V_DIM << "×" << NUM_V_HEADS
              << "×" << N_SEQ_TOKENS << "×" << N_SEQS
              << " — max_abs=" << max_abs_err
              << " max_rel=" << max_rel_err << "\n";
}

TEST_P(DeltaNetPostStateTest, FiniteOutput) {
    // Sanity: output contains no NaN / inf for ordinary inputs.
    Inputs in = make_random_inputs(0xDEADBEEFu);

    ggml_init_params p{1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* x      = new_xshape(ctx, "x");
    ggml_tensor* z      = new_xshape(ctx, "z");
    ggml_tensor* w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_V_DIM);
    ggml_set_input(w_norm);
    ggml_set_name(w_norm, "w_norm");

    ggml_tensor* out = build_fused(ctx, gf, x, z, w_norm, EPS);
    auto vals = run_graph(backend_, gf, ctx, out, in);

    for (size_t i = 0; i < vals.size(); ++i) {
        ASSERT_TRUE(std::isfinite(vals[i])) << "non-finite at i=" << i << " value=" << vals[i];
    }
    ggml_free(ctx);
}

INSTANTIATE_TEST_SUITE_P(
    Backends,
    DeltaNetPostStateTest,
    ::testing::Values(BackendKind::CPU, BackendKind::Metal),
    [](const ::testing::TestParamInfo<BackendKind>& info) {
        return std::string(backend_name(info.param));
    });

}  // namespace
