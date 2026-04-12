#include "tokenizer.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <map>
#include <set>
#include <iostream>
#include <queue>       // std::priority_queue
#include <functional>  // std::greater
#include <vector>      // std::vector (likely already included)

Tokenizer::Tokenizer(const Qwen3Metadata* metadata) : metadata_(metadata) {
    if (metadata_->id_to_token.empty()) {
        throw std::runtime_error("Invalid vocabulary: token list is empty.");
    }
    
    // Build token_to_id map
    token_to_id_.reserve(metadata_->id_to_token.size());
    for (size_t i = 0; i < metadata_->id_to_token.size(); ++i) {
        token_to_id_[metadata_->id_to_token[i]] = static_cast<int32_t>(i);
    }
    
    // Parse merge rules and populate the optimized map
    merges_.reserve(metadata_->merges.size());
    for (int i = 0; i < metadata_->merges.size(); ++i) {
        const std::string& merge_str = metadata_->merges[i];
        size_t space_pos = merge_str.find(' ');
        if (space_pos == std::string::npos) {
            throw std::runtime_error("Invalid merge format: " + merge_str);
        }
        
        std::string first_token_str = merge_str.substr(0, space_pos);
        std::string second_token_str = merge_str.substr(space_pos + 1);
        
        auto it1 = token_to_id_.find(first_token_str);
        auto it2 = token_to_id_.find(second_token_str);
        if (it1 == token_to_id_.end() || it2 == token_to_id_.end()) {
            // Skip merges with tokens not in the vocabulary
            continue;
        }
        
        std::string merged_token_str = first_token_str + second_token_str;
        auto merged_it = token_to_id_.find(merged_token_str);
        if (merged_it == token_to_id_.end()) {
            throw std::runtime_error("Merged token not found in vocab: " + merged_token_str);
        }
        
        merges_[{it1->second, it2->second}] = {i, merged_it->second};
    }
    
    // Initialize special tokens
    initialize_special_tokens();

    // Find and set the UNK token ID
    unk_token_id_ = -1;
    for (size_t i = 0; i < metadata_->token_types.size(); ++i) {
        if (metadata_->token_types[i] == TokenType::UNKNOWN) {
            unk_token_id_ = static_cast<int32_t>(i);
            break;
        }
    }
    
    // Initialize byte mapping for proper UTF-8 handling
    initialize_byte_mapping();
    
    // Build a comprehensive regex for pre-tokenization
    std::string special_token_pattern;
    for (const auto& [token_str, token_id] : special_tokens_) {
        if (!special_token_pattern.empty()) {
            special_token_pattern += "|";
        }
        // Escape special regex characters in the token string
        std::string escaped_token;
        for (char c : token_str) {
            if (std::string("()[]{}|?*+.").find(c) != std::string::npos) {
                escaped_token += '\\';
            }
            escaped_token += c;
        }
        special_token_pattern += escaped_token;
    }

    std::string base_pattern = R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+)";

    pretokenization_regex_ = std::regex(
        special_token_pattern + "|" + base_pattern,
        std::regex_constants::ECMAScript
    );
}


void Tokenizer::initialize_special_tokens() {
    // Add EOS from metadata
    special_tokens_["<|endoftext|>"] = metadata_->eos_token_id;
    special_token_ids_.insert(metadata_->eos_token_id);

    // Insert bos/padding only if they were actually set in metadata
    if (metadata_->bos_token_id >= 0) {
        special_token_ids_.insert(metadata_->bos_token_id);
    }
    if (metadata_->padding_token_id >= 0) {
        special_token_ids_.insert(metadata_->padding_token_id);
    }

    // Look up ChatML tokens from vocabulary instead of hardcoding IDs.
    // This works for both qwen2/3 (where they're at 151644/151645)
    // and qwen35 (where they're at different positions in the 248K vocab).
    const std::vector<std::string> chatml_tokens = {
        "<|im_start|>", "<|im_end|>"
    };
    for (const auto& tok_str : chatml_tokens) {
        auto it = token_to_id_.find(tok_str);
        if (it != token_to_id_.end()) {
            special_tokens_[tok_str] = it->second;
            special_token_ids_.insert(it->second);
        }
    }

    // Find all special tokens in vocabulary (CONTROL or USER_DEFINED type)
    for (size_t i = 0; i < metadata_->token_types.size(); ++i) {
        if (metadata_->token_types[i] == TokenType::CONTROL ||
            metadata_->token_types[i] == TokenType::USER_DEFINED) {
            special_token_ids_.insert(static_cast<int32_t>(i));
            special_tokens_[metadata_->id_to_token[i]] = static_cast<int32_t>(i);
        }
    }
}


void Tokenizer::initialize_byte_mapping() {
    // Standard GPT-2 byte-level mapping to unique Unicode characters.
    std::map<int, int> byte_to_unicode_map;
    std::set<int> printable_bytes;

    // Populate with printable ASCII and some extended chars that are mapped 1-to-1
    for (int i = 33; i <= 126; ++i) { printable_bytes.insert(i); byte_to_unicode_map[i] = i; }
    for (int i = 161; i <= 172; ++i) { printable_bytes.insert(i); byte_to_unicode_map[i] = i; }
    for (int i = 174; i <= 255; ++i) { printable_bytes.insert(i); byte_to_unicode_map[i] = i; }

    // Assign remaining byte values to higher Unicode code points
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (printable_bytes.find(b) == printable_bytes.end()) {
            byte_to_unicode_map[b] = 256 + n;
            n++;
        }
    }

    byte_encoder_.resize(256);
    for (int b = 0; b < 256; ++b) {
        int unicode_val = byte_to_unicode_map[b];
        std::string& mapped_str = byte_encoder_[b];
        
        // Simple UTF-8 encoding for the Unicode code point
        if (unicode_val < 128) {
            mapped_str += static_cast<char>(unicode_val);
        } else {
            mapped_str += static_cast<char>(0xC0 | (unicode_val >> 6));
            mapped_str += static_cast<char>(0x80 | (unicode_val & 0x3F));
        }
        
        byte_decoder_[mapped_str] = b;
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::sregex_iterator iter(text.begin(), text.end(), pretokenization_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        std::string match = iter->str();
        if (!match.empty()) {
            tokens.push_back(match);
        }
    }
    return tokens;
}

// Gets all pairs of adjacent tokens.
std::vector<std::pair<int32_t, int32_t>> get_pairs(const std::vector<int32_t>& tokens) {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    if (tokens.size() < 2) return pairs;
    pairs.reserve(tokens.size() - 1);
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        pairs.emplace_back(tokens[i], tokens[i+1]);
    }
    return pairs;
}

std::vector<int32_t> Tokenizer::apply_bpe(const std::vector<int32_t>& byte_tokens) const {
    if (byte_tokens.size() <= 1) return byte_tokens;

    const size_t n = byte_tokens.size();

    // Doubly-linked list for O(1) merge operations
    struct Node {
        int32_t token;
        int prev = -1;
        int next = -1;
        bool deleted = false;
    };
    
    std::vector<Node> nodes(n);
    for (size_t i = 0; i < n; ++i) {
        nodes[i].token = byte_tokens[i];
        nodes[i].prev = (i > 0) ? static_cast<int>(i - 1) : -1;
        nodes[i].next = (i < n - 1) ? static_cast<int>(i + 1) : -1;
    }
    
    // Min-heap: (rank, position) — lower rank = higher priority
    using Entry = std::pair<int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    
    // Helper: get merge info for pair starting at position i
    auto get_merge = [&](int i) -> std::pair<int, int32_t> {
        if (i < 0 || nodes[i].next < 0) return {-1, -1};
        auto it = merges_.find({nodes[i].token, nodes[nodes[i].next].token});
        if (it == merges_.end()) return {-1, -1};
        return it->second;  // {rank, merged_token}
    };
    
    // Initialize queue with all mergeable pairs
    for (size_t i = 0; i + 1 < n; ++i) {
        auto [rank, merged] = get_merge(i);
        if (rank >= 0) {
            pq.push({rank, static_cast<int>(i)});
        }
    }
    
    // Process merges in rank order
    while (!pq.empty()) {
        auto [rank, pos] = pq.top();
        pq.pop();
        
        // Skip stale entries (node deleted or pair changed)
        if (nodes[pos].deleted) continue;
        int next_pos = nodes[pos].next;
        if (next_pos < 0 || nodes[next_pos].deleted) continue;
        
        // Verify merge is still valid (tokens may have changed)
        auto [current_rank, merged_token] = get_merge(pos);
        if (current_rank != rank) continue;
        
        // Perform merge: pos absorbs next_pos
        nodes[pos].token = merged_token;
        nodes[next_pos].deleted = true;
        
        // Update linked list
        nodes[pos].next = nodes[next_pos].next;
        if (nodes[next_pos].next >= 0) {
            nodes[nodes[next_pos].next].prev = pos;
        }
        
        // Enqueue new pairs formed by merge
        if (nodes[pos].prev >= 0) {
            auto [r, m] = get_merge(nodes[pos].prev);
            if (r >= 0) pq.push({r, nodes[pos].prev});
        }
        {
            auto [r, m] = get_merge(pos);
            if (r >= 0) pq.push({r, pos});
        }
    }
    
    // Collect result by traversing from head
    std::vector<int32_t> result;
    result.reserve(n);
    
    // Find first non-deleted node
    int head = 0;
    while (head < static_cast<int>(n) && nodes[head].deleted) ++head;
    
    // Traverse linked list
    for (int pos = head; pos >= 0; pos = nodes[pos].next) {
        result.push_back(nodes[pos].token);
    }
    
    return result;
}

std::vector<int32_t> Tokenizer::encode_with_special_tokens(const std::string& text) const {
    // Check if entire text is a special token
    auto special_it = special_tokens_.find(text);
    if (special_it != special_tokens_.end()) {
        return {special_it->second};
    }

    // Check cache (thread-safe read)
    {
        std::lock_guard<std::mutex> lock(bpe_cache_mutex_);
        auto cache_it = bpe_cache_.find(text);
        if (cache_it != bpe_cache_.end()) {
            return cache_it->second;
        }
    }
    
    // Cache miss - compute BPE
    if (text.empty()) return {};
    
    // Convert text to byte-level tokens
    std::vector<int32_t> byte_tokens = encode_single_token(text);
    
    // Apply BPE merges
    std::vector<int32_t> result = apply_bpe(byte_tokens);
    
    // Store in cache (thread-safe write)
    {
        std::lock_guard<std::mutex> lock(bpe_cache_mutex_);
        bpe_cache_[text] = result;
    }
    
    return result;
}

std::vector<int32_t> Tokenizer::encode_single_token(const std::string& text) const {
    std::vector<int32_t> byte_tokens;
    byte_tokens.reserve(text.length());
    for (char c : text) {
        const std::string& byte_str = byte_encoder_[static_cast<unsigned char>(c)];
        auto token_it = token_to_id_.find(byte_str);
        if (token_it != token_to_id_.end()) {
            byte_tokens.push_back(token_it->second);
        } else {
            byte_tokens.push_back(unk_token_id_);
        }
    }
    return byte_tokens;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};
    
    // Step 1: Pre-tokenization (handles special tokens and regex splitting)
    std::vector<std::string> pretokens = pretokenize(text);
    
    // Step 2: Apply BPE to each pre-token
    std::vector<int32_t> result;
    for (const std::string& pretoken : pretokens) {
        std::vector<int32_t> encoded = encode_with_special_tokens(pretoken);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }
    
    return result;
}

std::string Tokenizer::_decode_original_token(int32_t original_id) const {
    if (original_id < 0 || original_id >= static_cast<int32_t>(metadata_->id_to_token.size())) {
        return "<ID_OOB>";
    }

    const std::string& token = metadata_->id_to_token[original_id];

    // Special tokens are returned as-is
    if (is_special_token(original_id)) {
        return token;
    }

    // Regular tokens need byte decoding.
    // GPT-2 byte encoding maps byte values to Unicode characters, some of which
    // are multi-byte in UTF-8 (e.g., space 0x20 → Ġ U+0120 → \xC4\xA0).
    // We must try multi-byte lookups before single-byte.
    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        // Try 2-byte UTF-8 sequence first (covers all GPT-2 byte mappings > 127)
        if (i + 1 < token.size()) {
            std::string two_byte = token.substr(i, 2);
            auto it = byte_decoder_.find(two_byte);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
                i += 2;
                continue;
            }
        }
        // Fall back to single-byte lookup
        std::string one_byte(1, token[i]);
        auto it = byte_decoder_.find(one_byte);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += token[i];
        }
        i++;
    }

    return result;
}

std::string Tokenizer::decode(int32_t token_id) const {
    return _decode_original_token(token_id);
}

bool Tokenizer::is_special_token(int32_t token_id) const {
    return special_token_ids_.count(token_id) > 0;
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    for (int32_t token_id : token_ids) {
        result += decode(token_id);
    }
    return result;
}

int32_t Tokenizer::get_special_token_id(const std::string& token_str) const {
    auto it = special_tokens_.find(token_str);
    if (it != special_tokens_.end()) {
        return it->second;
    }
    return unk_token_id_;
}

int32_t Tokenizer::get_eos_token_id() const {
    return metadata_->eos_token_id;
}

const std::vector<std::string>& Tokenizer::get_vocabulary() const {
    return metadata_->id_to_token;
}