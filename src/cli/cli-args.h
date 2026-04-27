#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <sstream>

struct CliArgs {
    std::string model_path;
    std::string prompt;
    int max_tokens = 100;
    float temperature = 0.7f;
    float repetition_penalty = 1.1f;
    int top_k = 40;
    float top_p = 0.95f;
    bool verbose = false;
    bool help = false;
    bool run_test = false;
    bool chat_mode = false;
    std::string vocab_prune_list_path;
    uint32_t context_length = 4096;
    std::string token_log_path;
    std::string grammar_file;
    std::string system_prompt;
    // TurboQuant KV cache compression (0 = disabled, 2/3/4 = bit width)
    int kv_quant_bits = 0;
    // SnapKV post-prefill KV eviction (0 = disabled)
    uint32_t snapkv_budget = 0;
    uint32_t snapkv_window = 32;
    // PLD options
    bool speculative = false;
    int pld_ngram_size = 3;
    int pld_max_draft = 5;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

// Chat-template kinds. Distinct GGUF architectures may share a kind; the
// mapping is centralized in detect_chat_template().
enum class ChatTemplateKind {
    ChatML,   // Qwen2/3 family: <|im_start|>role\n...<|im_end|>\n
    Gemma,    // Gemma family:  <start_of_turn>role\n...<end_of_turn>\n
};

inline ChatTemplateKind detect_chat_template(const std::string& architecture) {
    if (architecture == "gemma" || architecture == "gemma2" ||
        architecture == "gemma3" || architecture == "gemma4") {
        return ChatTemplateKind::Gemma;
    }
    return ChatTemplateKind::ChatML;
}

// Gemma role mapping: HF templates use "model" for the assistant turn.
// "system" has no first-class slot in Gemma 1's chat template; we emit it
// as a user turn whose content is the system prompt — matching what the
// reference HF tokenizer does when add_generation_prompt is true and a
// system message is present.
inline std::string gemma_role_for(const std::string& role) {
    if (role == "assistant") return "model";
    if (role == "system")    return "user";
    return role;
}

// Format chat history into the appropriate template for `kind`.
// add_assistant_prompt: if true, append the open turn for the assistant
// (so the model can sample directly).
inline std::string apply_chat_template(const std::vector<ChatMessage>& history,
                                       ChatTemplateKind kind,
                                       bool add_assistant_prompt = true) {
    std::ostringstream out;
    if (kind == ChatTemplateKind::Gemma) {
        for (const auto& m : history) {
            out << "<start_of_turn>" << gemma_role_for(m.role) << "\n"
                << m.content << "<end_of_turn>\n";
        }
        if (add_assistant_prompt) {
            out << "<start_of_turn>model\n";
        }
        return out.str();
    }

    // ChatML (Qwen)
    for (const auto& m : history) {
        out << "<|im_start|>" << m.role << "\n"
            << m.content << "<|im_end|>\n";
    }
    if (add_assistant_prompt) {
        out << "<|im_start|>assistant\n";
    }
    return out.str();
}

// Backward-compatible overload — defaults to ChatML so existing call sites
// that haven't been migrated keep their current behavior.
inline std::string apply_chat_template(const std::vector<ChatMessage>& history,
                                       bool add_assistant_prompt = true) {
    return apply_chat_template(history, ChatTemplateKind::ChatML, add_assistant_prompt);
}

inline std::string make_readable(std::string str) {
        size_t pos = 0;
    while ((pos = str.find("\xC4\xA0", pos)) != std::string::npos) {
        str.replace(pos, 2, " ");
        pos += 1;
    }
    return str;
}

inline void print_token(std::string str) {
    std::cout << make_readable(str) << std::flush;
}
