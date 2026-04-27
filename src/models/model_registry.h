#pragma once
// model_registry.h — architecture → ForwardPass factory dispatch.
//
// Replaces the if/else chains that used to live in chat.cpp / complete.cpp /
// main.cpp. Each model recipe (qwen3.cpp, qwen35.cpp, qwen36.cpp, gemma1.cpp,
// ...) registers itself once via REGISTER_MODEL or by calling
// register_model() at startup. CLI dispatch then becomes a single
// create_forward_pass() lookup.
//
// Fail-loud contract: looking up an unregistered architecture throws
// std::runtime_error naming the function, the expected list, and the actual
// architecture string.

#include "forward_pass_base.h"
#include "qwen3-model.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

using ForwardPassFactory = std::function<std::unique_ptr<ForwardPassBase>(
    const Qwen3Model&        model,
    const Qwen3Metadata*     metadata,
    uint32_t                 context_len,
    uint32_t                 max_batch_size,
    int                      kv_quant_bits)>;

// Register a factory for an architecture string. Overwrites if already present.
void register_model(const std::string& architecture, ForwardPassFactory factory);

// Unregister a factory (used in tests to clean up after registering fakes).
void unregister_model(const std::string& architecture);

// Returns true iff `architecture` is registered.
bool is_architecture_registered(const std::string& architecture);

// List currently registered architecture strings (for diagnostics).
std::vector<std::string> registered_architectures();

// Create a forward pass for the architecture in `metadata`. Throws
// std::runtime_error if the architecture is not registered.
std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Qwen3Model&    model,
    const Qwen3Metadata* metadata,
    uint32_t             context_len,
    uint32_t             max_batch_size,
    int                  kv_quant_bits);

// Register all built-in model recipes (qwen2/3, qwen35, qwen35moe, gemma).
// Idempotent — safe to call multiple times. Call once from CLI startup.
void register_builtin_models();
