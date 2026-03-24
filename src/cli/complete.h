#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ggml.h"
#include "ggml-backend.h"

#include "cli-args.h"
#include "speculative-bridge.h"

#include "../qwen3-core/qwen3-model.h"
#include "../qwen3-core/forward-pass-factory.h"
#include "../qwen3-core/tokenizer.h"
#include "../qwen3-core/sampling.h"
#include "../qwen3-core/grammar.h"
#include "../qwen3-core/speculative.h"
#include "../qwen3-core/vocab_utils.h"

class Qwen3Model;

namespace qwen3 {
    class Grammar;
    class SpeculativeDecoder;
}

/// Run single-prompt (non-interactive) generation.
/// Returns process exit code.
int run_complete(
    Qwen3Model& model,
    const CliArgs& args,
    std::unique_ptr<qwen3::Grammar>& grammar,        // nullable
    qwen3::SpeculativeDecoder* spec,                  // nullable
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
);