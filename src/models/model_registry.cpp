#include "model_registry.h"

#include "qwen3.h"
#include "qwen35.h"
#include "qwen36.h"
#include "gemma1.h"

#include <map>
#include <mutex>
#include <stdexcept>

namespace {
std::map<std::string, ForwardPassFactory>& registry()
{
    static std::map<std::string, ForwardPassFactory> r;
    return r;
}

std::mutex& registry_mu()
{
    static std::mutex m;
    return m;
}
} // namespace

void register_model(const std::string& architecture, ForwardPassFactory factory)
{
    std::lock_guard<std::mutex> g(registry_mu());
    registry()[architecture] = std::move(factory);
}

void unregister_model(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    registry().erase(architecture);
}

bool is_architecture_registered(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    return registry().count(architecture) > 0;
}

std::vector<std::string> registered_architectures()
{
    std::lock_guard<std::mutex> g(registry_mu());
    std::vector<std::string> out;
    out.reserve(registry().size());
    for (const auto& [k, _] : registry()) out.push_back(k);
    return out;
}

std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Qwen3Model&    model,
    const Qwen3Metadata* metadata,
    uint32_t             context_len,
    uint32_t             max_batch_size,
    int                  kv_quant_bits)
{
    ForwardPassFactory factory;
    {
        std::lock_guard<std::mutex> g(registry_mu());
        auto it = registry().find(metadata->architecture);
        if (it == registry().end()) {
            std::string registered;
            for (const auto& [k, _] : registry()) {
                if (!registered.empty()) registered += ", ";
                registered += "'" + k + "'";
            }
            throw std::runtime_error(
                "create_forward_pass: expected one of: " + registered +
                ", got '" + metadata->architecture + "'");
        }
        factory = it->second;
    }
    return factory(model, metadata, context_len, max_batch_size, kv_quant_bits);
}

void register_builtin_models()
{
    // Qwen2/3 share Qwen3ForwardPass.
    auto qwen3_factory = [](const Qwen3Model& m, const Qwen3Metadata* meta,
                            uint32_t ctx, uint32_t bs, int kvb) {
        return std::make_unique<Qwen3ForwardPass>(m, meta, ctx, bs, kvb);
    };
    register_model("qwen2", qwen3_factory);
    register_model("qwen3", qwen3_factory);

    register_model("qwen35", [](const Qwen3Model& m, const Qwen3Metadata* meta,
                                uint32_t ctx, uint32_t bs, int kvb) {
        return std::make_unique<Qwen35ForwardPass>(m, meta, ctx, bs, kvb);
    });

    register_model("qwen35moe", [](const Qwen3Model& m, const Qwen3Metadata* meta,
                                   uint32_t ctx, uint32_t bs, int kvb) {
        return std::make_unique<Qwen36ForwardPass>(m, meta, ctx, bs, kvb);
    });

    register_model("gemma", [](const Qwen3Model& m, const Qwen3Metadata* meta,
                                uint32_t ctx, uint32_t bs, int kvb) {
        return std::make_unique<Gemma1ForwardPass>(m, meta, ctx, bs, kvb);
    });
}
