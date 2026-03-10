#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "ggml.h"
#include "ggml-backend.h"

class QwenGGUFLoader;
class Tokenizer;

struct TensorMetadata {
    std::string name;
    ggml_type type;
    std::vector<uint64_t> shape;
    uint64_t offset;
};

// Tokenizer-specific structures
enum class TokenType : uint8_t {
    NORMAL = 1,
    UNKNOWN = 2,
    CONTROL = 3,
    USER_DEFINED = 4,
    UNUSED = 5,
    BYTE = 6
};

struct Qwen3Metadata {
    // Model metadata
    std::string model_name;
    std::string architecture;
    uint32_t block_count;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t attention_key_length;
    uint32_t attention_value_length;
    float rope_freq_base;
    float rms_norm_eps;
    
    // Tokenizer metadata (from GGUF)
    std::string tokenizer_type;              // "gpt2" tokenizer.ggml.model
    std::string tokenizer_pre;               // "qwen2" tokenizer.ggml.pre
    std::vector<std::string> id_to_token;    // tokenizer.ggml.tokens [151936 elements]
    std::vector<TokenType> token_types;      // tokenizer.ggml.token_type [151936 elements]
    std::vector<std::string> merges;         // tokenizer.ggml.merges [151387 elements]
    
    // Special token IDs (from GGUF)
    int32_t eos_token_id;       // 151645
    int32_t bos_token_id;       // 151643  
    int32_t padding_token_id;   // 151643
    bool add_bos_token;
    
    std::unordered_map<std::string, TensorMetadata> tensor_inventory;
};

enum class Qwen3ModelSize {
    QWEN3_0_6B,
    QWEN3_8B,
    QWEN3_32B
};

// Structs to hold the model's tensors
struct TransformerBlock {
    // Attention
    struct ggml_tensor* attn_norm_weight;
    struct ggml_tensor* attn_q_weight;
    struct ggml_tensor* attn_k_weight;
    struct ggml_tensor* attn_v_weight;
    struct ggml_tensor* attn_output_weight;

    // Qwen3-specific
    struct ggml_tensor* attn_q_norm_weight;
    struct ggml_tensor* attn_k_norm_weight;

    // Qwen2-specific
    struct ggml_tensor* attn_q_bias;
    struct ggml_tensor* attn_k_bias;
    struct ggml_tensor* attn_v_bias;

    // Feed-forward
    struct ggml_tensor* ffn_norm_weight;
    struct ggml_tensor* ffn_gate_weight;
    struct ggml_tensor* ffn_up_weight;
    struct ggml_tensor* ffn_down_weight;
};

class Qwen3Model {
public:
    Qwen3Model();
    ~Qwen3Model();

    void load_metadata(const std::string& model_path);
    void load_tensors();

    bool validate_architecture() const;
    Qwen3ModelSize detect_model_size() const;
    std::string get_model_size_string() const;
    void print_metadata() const;
    
    const Qwen3Metadata& get_metadata() const { return metadata_; }
    struct ggml_tensor* get_token_embedding_weight() const { return token_embd_weight_; }
    struct ggml_tensor* get_output_norm_weight() const { return output_norm_weight_; }
    struct ggml_tensor* get_output_weight() const { return output_weight_; }
    struct ggml_context* get_context() const { return model_context_; }
    const TransformerBlock& get_block(int i) const { return blocks_[i]; }
    const ggml_backend_t& get_backend_metal() const { return backend_metal_;}
    const ggml_backend_t& get_backend_cpu() const { return backend_cpu_;}
    
    // Phase 1: Backend getters (for future use in Phase 2/3)
    ggml_backend_sched_t get_scheduler() const { return sched_; }
    bool has_metal_backend() const { return backend_metal_ != nullptr; }

private:
    Qwen3Metadata metadata_;
    bool is_loaded_;
    std::unique_ptr<QwenGGUFLoader> loader_;
    ggml_context* model_context_;

    // Phase 1: Backend infrastructure
    ggml_backend_t backend_cpu_;
    ggml_backend_t backend_metal_;
    ggml_backend_sched_t sched_;
    ggml_backend_buffer_t weights_buffer_;

    // Tensors
    struct ggml_tensor* token_embd_weight_;
    struct ggml_tensor* output_norm_weight_;
    struct ggml_tensor* output_weight_;

    std::vector<TransformerBlock> blocks_;
    
    // Helper method for tensor assignment (reduces code duplication)
    void assign_tensor_pointers(const std::unordered_map<std::string, ggml_tensor*>& tensors);

    // Tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;
public:
    Tokenizer* get_tokenizer() const { return tokenizer_.get(); }
};