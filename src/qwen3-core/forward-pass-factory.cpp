// forward-pass-factory.cpp
#include "forward-pass-factory.h"
#include "forward-pass.h"
#include "forward-pass-qwen35.h"

std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Qwen3Model& model,
    const Qwen3Metadata* metadata,
    uint32_t context_len,
    uint32_t max_batch_size)
{
    if (metadata->architecture == "qwen35") {
        return std::make_unique<Qwen35ForwardPass>(model, metadata, context_len, max_batch_size);
    }
    return std::make_unique<Qwen3ForwardPass>(model, metadata, context_len, max_batch_size);
}