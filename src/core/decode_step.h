#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "../models/forward_pass_base.h"
#include "../sampling/sampling.h"

#include "ggml-backend.h"

// Perform one decode step for a single slot.
//
// Calls peek_valid_set before the forward pass; selects the sparse LM head
// when the grammar narrows the candidate set below vocab_size / 8, and falls
// back to the dense path otherwise.  Grammar state is advanced inside via
// sampler->accept_token so the caller does not need to touch the grammar.
//
// Returns the sampled token ID.
int32_t decode_step(
    ForwardPassBase*       fp,
    ggml_backend_sched_t   scheduler,
    qwenium::Sampler*      sampler,
    int32_t                token,
    uint32_t               slot,
    const std::vector<int32_t>&    history,
    const std::vector<std::string>& vocab,
    uint32_t               vocab_size,
    bool                   force_dense = false  // diagnostic: skip sparse head selection
);
