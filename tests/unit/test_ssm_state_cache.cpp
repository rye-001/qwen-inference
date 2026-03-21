/**
 * test_ssm_state_cache.cpp — Step 3 TDD for Qwen 3.5 support
 * 
 * Tests the SSM state cache: fixed-size recurrent state S and
 * conv sliding window storage. No model file needed.
 *
 * Run: ./qwen3-unit-tests --gtest_filter="SSMStateCache*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "../../src/kv-cache/ssm-state-cache.h"
#include "ggml.h"
#include "ggml-cpu.h"

// Qwen3.5-0.8B dimensions
static constexpr uint32_t D_STATE    = 128;   // ssm.state_size
static constexpr uint32_t N_HEADS    = 16;    // ssm.group_count
static constexpr uint32_t CONV_DIM   = 6144;  // 3 * ssm.inner_size (QKV projection width)
static constexpr uint32_t CONV_K     = 4;     // ssm.conv_kernel
static constexpr uint32_t N_SSM_LAYERS = 18;  // 24 total - 6 attention = 18 SSM
static constexpr uint32_t N_SLOTS    = 4;     // batch slots for testing

class SSMStateCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        cache_ = std::make_unique<ssm_state_cache>(
            N_SSM_LAYERS, N_SLOTS,
            D_STATE, N_HEADS,
            CONV_DIM, CONV_K,
            backend_
        );
    }

    void TearDown() override {
        cache_.reset();
        if (backend_) ggml_backend_free(backend_);
    }

    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ssm_state_cache> cache_;
};

// ============================================================
// Test: Construction succeeds
// ============================================================
TEST_F(SSMStateCacheTest, ConstructionSucceeds) {
    EXPECT_NE(cache_, nullptr);
}

// ============================================================
// Test: Memory footprint is reasonable
// ============================================================
TEST_F(SSMStateCacheTest, MemoryFootprint) {
    size_t bytes = cache_->total_bytes();

    // Recurrent state: d_state * d_state * n_heads * n_slots * n_layers * sizeof(float)
    // = 128 * 128 * 16 * 4 * 18 * 4 = ~75 MB
    // Conv state: conv_dim * conv_kernel * n_slots * n_layers * sizeof(float)
    // = 6144 * 4 * 4 * 18 * 4 = ~7 MB
    // Total ~82 MB — sanity check it's in the right ballpark
    size_t expected_min = 70 * 1024 * 1024;   // 70 MB
    size_t expected_max = 100 * 1024 * 1024;  // 100 MB

    EXPECT_GT(bytes, expected_min) << "Cache too small: " << bytes / (1024*1024) << " MB";
    EXPECT_LT(bytes, expected_max) << "Cache too large: " << bytes / (1024*1024) << " MB";
}

// ============================================================
// Test: Recurrent state starts zeroed
// ============================================================
TEST_F(SSMStateCacheTest, RecurrentStateStartsZeroed) {
    // Read recurrent state for layer 0, slot 0
    std::vector<float> state(D_STATE * D_STATE * N_HEADS);
    cache_->get_recurrent_state(0, 0, state.data());

    for (size_t i = 0; i < state.size(); ++i) {
        EXPECT_FLOAT_EQ(state[i], 0.0f) << "Non-zero at index " << i;
    }
}

// ============================================================
// Test: Conv state starts zeroed
// ============================================================
TEST_F(SSMStateCacheTest, ConvStateStartsZeroed) {
    std::vector<float> state(CONV_DIM * CONV_K);
    cache_->get_conv_state(0, 0, state.data());

    for (size_t i = 0; i < state.size(); ++i) {
        EXPECT_FLOAT_EQ(state[i], 0.0f) << "Non-zero at index " << i;
    }
}

// ============================================================
// Test: Write and read back recurrent state
// ============================================================
TEST_F(SSMStateCacheTest, RecurrentStateWriteRead) {
    const size_t state_size = D_STATE * D_STATE * N_HEADS;
    std::vector<float> write_data(state_size);
    // Fill with deterministic pattern
    for (size_t i = 0; i < state_size; ++i) {
        write_data[i] = sinf(static_cast<float>(i) * 0.01f);
    }

    // Write to layer 5, slot 2
    cache_->set_recurrent_state(5, 2, write_data.data());

    // Read back
    std::vector<float> read_data(state_size);
    cache_->get_recurrent_state(5, 2, read_data.data());

    for (size_t i = 0; i < state_size; ++i) {
        EXPECT_FLOAT_EQ(read_data[i], write_data[i]) << "Mismatch at index " << i;
    }
}

// ============================================================
// Test: Write and read back conv state
// ============================================================
TEST_F(SSMStateCacheTest, ConvStateWriteRead) {
    const size_t state_size = CONV_DIM * CONV_K;
    std::vector<float> write_data(state_size);
    for (size_t i = 0; i < state_size; ++i) {
        write_data[i] = cosf(static_cast<float>(i) * 0.02f);
    }

    cache_->set_conv_state(3, 1, write_data.data());

    std::vector<float> read_data(state_size);
    cache_->get_conv_state(3, 1, read_data.data());

    for (size_t i = 0; i < state_size; ++i) {
        EXPECT_FLOAT_EQ(read_data[i], write_data[i]) << "Mismatch at index " << i;
    }
}

// ============================================================
// Test: Slot isolation — writing to one slot doesn't affect others
// ============================================================
TEST_F(SSMStateCacheTest, SlotIsolation) {
    const size_t rec_size = D_STATE * D_STATE * N_HEADS;

    // Write to slot 0
    std::vector<float> data_slot0(rec_size, 1.0f);
    cache_->set_recurrent_state(0, 0, data_slot0.data());

    // Write to slot 1
    std::vector<float> data_slot1(rec_size, 2.0f);
    cache_->set_recurrent_state(0, 1, data_slot1.data());

    // Verify slot 0 still has 1.0
    std::vector<float> readback(rec_size);
    cache_->get_recurrent_state(0, 0, readback.data());
    for (size_t i = 0; i < rec_size; ++i) {
        EXPECT_FLOAT_EQ(readback[i], 1.0f) << "Slot 0 corrupted at index " << i;
    }

    // Verify slot 1 has 2.0
    cache_->get_recurrent_state(0, 1, readback.data());
    for (size_t i = 0; i < rec_size; ++i) {
        EXPECT_FLOAT_EQ(readback[i], 2.0f) << "Slot 1 wrong at index " << i;
    }
}

// ============================================================
// Test: Layer isolation — writing to one layer doesn't affect others
// ============================================================
TEST_F(SSMStateCacheTest, LayerIsolation) {
    const size_t rec_size = D_STATE * D_STATE * N_HEADS;

    std::vector<float> data_layer0(rec_size, 3.0f);
    cache_->set_recurrent_state(0, 0, data_layer0.data());

    std::vector<float> data_layer1(rec_size, 7.0f);
    cache_->set_recurrent_state(1, 0, data_layer1.data());

    // Verify layer 0 still has 3.0
    std::vector<float> readback(rec_size);
    cache_->get_recurrent_state(0, 0, readback.data());
    for (size_t i = 0; i < rec_size; ++i) {
        EXPECT_FLOAT_EQ(readback[i], 3.0f) << "Layer 0 corrupted at index " << i;
    }
}

// ============================================================
// Test: clear_slot zeros both recurrent and conv state
// ============================================================
TEST_F(SSMStateCacheTest, ClearSlot) {
    const size_t rec_size = D_STATE * D_STATE * N_HEADS;
    const size_t conv_size = CONV_DIM * CONV_K;

    // Write non-zero data to slot 1 across multiple layers
    std::vector<float> rec_data(rec_size, 42.0f);
    std::vector<float> conv_data(conv_size, 99.0f);
    for (uint32_t il = 0; il < N_SSM_LAYERS; ++il) {
        cache_->set_recurrent_state(il, 1, rec_data.data());
        cache_->set_conv_state(il, 1, conv_data.data());
    }

    // Clear slot 1
    cache_->clear_slot(1);

    // Verify all layers of slot 1 are zeroed
    std::vector<float> readback_rec(rec_size);
    std::vector<float> readback_conv(conv_size);
    for (uint32_t il = 0; il < N_SSM_LAYERS; ++il) {
        cache_->get_recurrent_state(il, 1, readback_rec.data());
        for (size_t i = 0; i < rec_size; ++i) {
            EXPECT_FLOAT_EQ(readback_rec[i], 0.0f) 
                << "Recurrent state not cleared: layer " << il << " index " << i;
        }
        cache_->get_conv_state(il, 1, readback_conv.data());
        for (size_t i = 0; i < conv_size; ++i) {
            EXPECT_FLOAT_EQ(readback_conv[i], 0.0f)
                << "Conv state not cleared: layer " << il << " index " << i;
        }
    }

    // Verify slot 0 is unaffected (should still be zero from init)
    cache_->get_recurrent_state(0, 0, readback_rec.data());
    // Slot 0 was never written, so still zero — just a sanity check
}

// ============================================================
// Test: clone_slot copies both recurrent and conv state
// ============================================================
TEST_F(SSMStateCacheTest, CloneSlot) {
    const size_t rec_size = D_STATE * D_STATE * N_HEADS;
    const size_t conv_size = CONV_DIM * CONV_K;

    // Write distinctive data to slot 0
    std::vector<float> rec_data(rec_size);
    std::vector<float> conv_data(conv_size);
    for (size_t i = 0; i < rec_size; ++i)  rec_data[i] = sinf(i * 0.005f);
    for (size_t i = 0; i < conv_size; ++i) conv_data[i] = cosf(i * 0.003f);

    for (uint32_t il = 0; il < N_SSM_LAYERS; ++il) {
        cache_->set_recurrent_state(il, 0, rec_data.data());
        cache_->set_conv_state(il, 0, conv_data.data());
    }

    // Clone slot 0 → slot 3
    cache_->clone_slot(0, 3);

    // Verify slot 3 matches slot 0
    std::vector<float> readback_rec(rec_size);
    std::vector<float> readback_conv(conv_size);
    for (uint32_t il = 0; il < N_SSM_LAYERS; ++il) {
        cache_->get_recurrent_state(il, 3, readback_rec.data());
        for (size_t i = 0; i < rec_size; ++i) {
            EXPECT_FLOAT_EQ(readback_rec[i], rec_data[i])
                << "Clone mismatch (recurrent): layer " << il << " index " << i;
        }
        cache_->get_conv_state(il, 3, readback_conv.data());
        for (size_t i = 0; i < conv_size; ++i) {
            EXPECT_FLOAT_EQ(readback_conv[i], conv_data[i])
                << "Clone mismatch (conv): layer " << il << " index " << i;
        }
    }
}

// ============================================================
// Test: clone doesn't affect source slot
// ============================================================
TEST_F(SSMStateCacheTest, CloneDoesNotAffectSource) {
    const size_t rec_size = D_STATE * D_STATE * N_HEADS;

    std::vector<float> src_data(rec_size, 5.0f);
    cache_->set_recurrent_state(0, 0, src_data.data());

    cache_->clone_slot(0, 2);

    // Modify the clone
    std::vector<float> new_data(rec_size, 99.0f);
    cache_->set_recurrent_state(0, 2, new_data.data());

    // Source should be unchanged
    std::vector<float> readback(rec_size);
    cache_->get_recurrent_state(0, 0, readback.data());
    for (size_t i = 0; i < rec_size; ++i) {
        EXPECT_FLOAT_EQ(readback[i], 5.0f) << "Source corrupted at index " << i;
    }
}
