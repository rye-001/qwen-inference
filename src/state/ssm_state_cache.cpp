#include "ssm_state_cache.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

ssm_state_cache::ssm_state_cache(
    uint32_t n_ssm_layers,
    uint32_t n_slots,
    uint32_t d_state,
    uint32_t n_heads,
    uint32_t conv_dim,
    uint32_t conv_kernel,
    ggml_backend_t backend
) :
    n_ssm_layers_(n_ssm_layers),
    n_slots_(n_slots),
    d_state_(d_state),
    n_heads_(n_heads),
    conv_dim_(conv_dim),
    conv_kernel_(conv_kernel),
    recurrent_slot_floats_(d_state * d_state * n_heads),
    conv_slot_floats_(conv_dim * conv_kernel),
    backend_(backend),
    ctx_(nullptr, ggml_free),
    buf_(nullptr, ggml_backend_buffer_free)
{
    init_cache();
}

void ssm_state_cache::init_cache() {
    // Context for tensor metadata (2 tensors per layer + headroom)
    const size_t ctx_size = (n_ssm_layers_ * 2 + 64) * ggml_tensor_overhead();

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ctx_.reset(ggml_init(params));

    recurrent_state_.resize(n_ssm_layers_);
    conv_state_.resize(n_ssm_layers_);

    for (uint32_t il = 0; il < n_ssm_layers_; ++il) {
        // Recurrent state: [recurrent_slot_floats, n_slots]
        // Flat per-slot dimension for easy offset arithmetic
        recurrent_state_[il] = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32,
            recurrent_slot_floats_, n_slots_
        );

        // Conv state: [conv_slot_floats, n_slots]
        conv_state_[il] = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32,
            conv_slot_floats_, n_slots_
        );
    }

    // Allocate backend buffer
    ggml_backend_buffer_type_t buffer_type;
    if (backend_) {
        buffer_type = ggml_backend_get_default_buffer_type(backend_);
    } else {
        buffer_type = ggml_backend_cpu_buffer_type();
    }

    buf_.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), buffer_type));
    if (!buf_) {
        throw std::runtime_error("ssm_state_cache: failed to allocate backend buffer");
    }

    // Zero-initialize all state (using tensor_set for ggml compatibility)
    {
        std::vector<uint8_t> zeros_rec(recurrent_slot_floats_ * n_slots_ * sizeof(float), 0);
        std::vector<uint8_t> zeros_conv(conv_slot_floats_ * n_slots_ * sizeof(float), 0);
        for (uint32_t il = 0; il < n_ssm_layers_; ++il) {
            ggml_backend_tensor_set(recurrent_state_[il], zeros_rec.data(), 0, zeros_rec.size());
            ggml_backend_tensor_set(conv_state_[il], zeros_conv.data(), 0, zeros_conv.size());
        }
    }

    size_t total = ggml_backend_buffer_get_size(buf_.get());
    std::cout << "SSM state cache allocated: " << total / (1024 * 1024) << " MB "
              << "(" << n_ssm_layers_ << " layers, " << n_slots_ << " slots)" << std::endl;
}

void ssm_state_cache::get_recurrent_state(uint32_t layer, uint32_t slot, float* dst) const {
    GGML_ASSERT(layer < n_ssm_layers_);
    GGML_ASSERT(slot < n_slots_);

    size_t offset_bytes = slot * recurrent_slot_floats_ * sizeof(float);
    size_t nbytes = recurrent_slot_floats_ * sizeof(float);
    ggml_backend_tensor_get(recurrent_state_[layer], dst, offset_bytes, nbytes);
}

void ssm_state_cache::set_recurrent_state(uint32_t layer, uint32_t slot, const float* src) {
    GGML_ASSERT(layer < n_ssm_layers_);
    GGML_ASSERT(slot < n_slots_);

    size_t offset_bytes = slot * recurrent_slot_floats_ * sizeof(float);
    size_t nbytes = recurrent_slot_floats_ * sizeof(float);
    ggml_backend_tensor_set(recurrent_state_[layer], src, offset_bytes, nbytes);
}

void ssm_state_cache::get_conv_state(uint32_t layer, uint32_t slot, float* dst) const {
    GGML_ASSERT(layer < n_ssm_layers_);
    GGML_ASSERT(slot < n_slots_);

    size_t offset_bytes = slot * conv_slot_floats_ * sizeof(float);
    size_t nbytes = conv_slot_floats_ * sizeof(float);
    ggml_backend_tensor_get(conv_state_[layer], dst, offset_bytes, nbytes);
}

void ssm_state_cache::set_conv_state(uint32_t layer, uint32_t slot, const float* src) {
    GGML_ASSERT(layer < n_ssm_layers_);
    GGML_ASSERT(slot < n_slots_);

    size_t offset_bytes = slot * conv_slot_floats_ * sizeof(float);
    size_t nbytes = conv_slot_floats_ * sizeof(float);
    ggml_backend_tensor_set(conv_state_[layer], src, offset_bytes, nbytes);
}

void ssm_state_cache::clear_slot(uint32_t slot) {
    GGML_ASSERT(slot < n_slots_);

    std::vector<uint8_t> zeros_rec(recurrent_slot_floats_ * sizeof(float), 0);
    std::vector<uint8_t> zeros_conv(conv_slot_floats_ * sizeof(float), 0);

    for (uint32_t il = 0; il < n_ssm_layers_; ++il) {
        size_t rec_offset = slot * recurrent_slot_floats_ * sizeof(float);
        ggml_backend_tensor_set(recurrent_state_[il], zeros_rec.data(), rec_offset, zeros_rec.size());

        size_t conv_offset = slot * conv_slot_floats_ * sizeof(float);
        ggml_backend_tensor_set(conv_state_[il], zeros_conv.data(), conv_offset, zeros_conv.size());
    }
}

void ssm_state_cache::clone_slot(uint32_t src_slot, uint32_t dst_slot) {
    GGML_ASSERT(src_slot < n_slots_);
    GGML_ASSERT(dst_slot < n_slots_);

    // Scratch context for temporary views
    std::vector<uint8_t> scratch_buf(512 * 1024); // 512KB
    struct ggml_init_params params = {
        .mem_size   = scratch_buf.size(),
        .mem_buffer = scratch_buf.data(),
        .no_alloc   = true,
    };
    ggml_context* scratch_ctx = ggml_init(params);

    for (uint32_t il = 0; il < n_ssm_layers_; ++il) {
        // Clone recurrent state
        {
            ggml_tensor* src = ggml_view_1d(scratch_ctx, recurrent_state_[il],
                recurrent_slot_floats_,
                src_slot * recurrent_slot_floats_ * sizeof(float));
            ggml_tensor* dst = ggml_view_1d(scratch_ctx, recurrent_state_[il],
                recurrent_slot_floats_,
                dst_slot * recurrent_slot_floats_ * sizeof(float));

            src->buffer = recurrent_state_[il]->buffer;
            dst->buffer = recurrent_state_[il]->buffer;
            ggml_backend_tensor_copy(src, dst);
        }

        // Clone conv state
        {
            ggml_tensor* src = ggml_view_1d(scratch_ctx, conv_state_[il],
                conv_slot_floats_,
                src_slot * conv_slot_floats_ * sizeof(float));
            ggml_tensor* dst = ggml_view_1d(scratch_ctx, conv_state_[il],
                conv_slot_floats_,
                dst_slot * conv_slot_floats_ * sizeof(float));

            src->buffer = conv_state_[il]->buffer;
            dst->buffer = conv_state_[il]->buffer;
            ggml_backend_tensor_copy(src, dst);
        }
    }

    ggml_free(scratch_ctx);
}

size_t ssm_state_cache::total_bytes() const {
    if (!buf_) return 0;
    return ggml_backend_buffer_get_size(buf_.get());
}
