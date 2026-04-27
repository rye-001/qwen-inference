// test_loader.cpp — PR 2.8
//
// Unit tests for src/loader/: GGUFLoader construction, Tokenizer build,
// and known-answer ASCII round-trip.  No model file required.
//
// Memory baseline: calculate_tensors_memory_size() for Qwen3-0.6B-Q8_0 ≈ 620 MB.
// This is recorded for Phase 4 regression detection, not enforced here.
//
// Run: ./qwen3-loader-tests

#include <gtest/gtest.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/loader/gguf_loader.h"
#include "../../src/loader/tokenizer.h"

// ── Fixture: minimal byte-level vocabulary ────────────────────────────────────
//
// Mirrors Tokenizer::initialize_byte_mapping() exactly so that
// encode(x) → decode() round-trips any ASCII text without a real model file.
// 256 byte tokens (indices 0–255) + one EOS control token at index 256.
static Qwen3Metadata make_byte_level_metadata() {
    std::map<int, int> b2u;
    std::set<int> printable;
    for (int i = 33;  i <= 126; ++i) { printable.insert(i); b2u[i] = i; }
    for (int i = 161; i <= 172; ++i) { printable.insert(i); b2u[i] = i; }
    for (int i = 174; i <= 255; ++i) { printable.insert(i); b2u[i] = i; }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!printable.count(b)) b2u[b] = 256 + n++;
    }

    auto to_utf8 = [](int u) -> std::string {
        std::string s;
        if (u < 128) {
            s += static_cast<char>(u);
        } else {
            s += static_cast<char>(0xC0 | (u >> 6));
            s += static_cast<char>(0x80 | (u & 0x3F));
        }
        return s;
    };

    Qwen3Metadata m;
    m.id_to_token.resize(257);
    m.token_types.resize(257, TokenType::NORMAL);
    for (int b = 0; b < 256; ++b) {
        m.id_to_token[b] = to_utf8(b2u[b]);
    }
    m.id_to_token[256]  = "<|endoftext|>";
    m.token_types[256]  = TokenType::CONTROL;
    m.eos_token_id      = 256;
    m.bos_token_id      = -1;
    m.padding_token_id  = -1;
    return m;
}

// ── Inventory schema tests (PR G1.2) ──────────────────────────────────────────

// Build a minimal Qwen3Metadata with a synthetic tensor inventory for tests.
// Tensor types and shapes are not exercised here — only required-key presence.
static TensorMetadata fake_tensor(const std::string& name) {
    TensorMetadata t;
    t.name = name;
    t.type = GGML_TYPE_F32;
    t.shape = {1};
    t.offset = 0;
    return t;
}

static Qwen3Metadata make_complete_gemma_inventory(uint32_t blocks = 2) {
    Qwen3Metadata m;
    m.architecture = "gemma";
    m.block_count = blocks;
    m.tensor_inventory["token_embd.weight"] = fake_tensor("token_embd.weight");
    m.tensor_inventory["output_norm.weight"] = fake_tensor("output_norm.weight");
    const std::vector<std::string> per_block = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "attn_norm.weight", "ffn_norm.weight"
    };
    for (uint32_t i = 0; i < blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) {
            m.tensor_inventory[p + t] = fake_tensor(p + t);
        }
    }
    return m;
}

static Qwen3Metadata make_complete_qwen3_inventory(uint32_t blocks = 2) {
    Qwen3Metadata m;
    m.architecture = "qwen3";
    m.block_count = blocks;
    m.tensor_inventory["token_embd.weight"] = fake_tensor("token_embd.weight");
    m.tensor_inventory["output_norm.weight"] = fake_tensor("output_norm.weight");
    const std::vector<std::string> per_block = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight"
    };
    for (uint32_t i = 0; i < blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) {
            m.tensor_inventory[p + t] = fake_tensor(p + t);
        }
    }
    return m;
}

TEST(InventorySchema, GemmaCompleteInventoryPasses) {
    auto m = make_complete_gemma_inventory();
    EXPECT_NO_THROW(validate_gemma_inventory(m));
    EXPECT_NO_THROW(validate_inventory_for_architecture(m));
}

TEST(InventorySchema, GemmaMissingGlobalTensorFailsLoudly) {
    auto m = make_complete_gemma_inventory();
    m.tensor_inventory.erase("output_norm.weight");
    try {
        validate_gemma_inventory(m);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("gemma"), std::string::npos) << msg;
        EXPECT_NE(msg.find("output_norm.weight"), std::string::npos) << msg;
    }
}

TEST(InventorySchema, GemmaMissingPerBlockTensorFailsLoudly) {
    auto m = make_complete_gemma_inventory(3);
    m.tensor_inventory.erase("blk.1.ffn_down.weight");
    try {
        validate_gemma_inventory(m);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("blk.1.ffn_down.weight"), std::string::npos) << msg;
    }
}

TEST(InventorySchema, GemmaTiedEmbeddingsAcceptedNoSeparateOutput) {
    // Gemma reuses token_embd.weight for the LM head — no `output.weight`.
    // Validator must NOT require output.weight.
    auto m = make_complete_gemma_inventory();
    EXPECT_TRUE(m.tensor_inventory.find("output.weight") == m.tensor_inventory.end());
    EXPECT_NO_THROW(validate_gemma_inventory(m));
}

TEST(InventorySchema, Qwen3InventoryUnaffectedByDispatch) {
    auto m = make_complete_qwen3_inventory();
    EXPECT_NO_THROW(validate_inventory_for_architecture(m));
}

TEST(InventorySchema, DispatchUnknownArchitectureFailsLoudly) {
    Qwen3Metadata m;
    m.architecture = "totally_unknown";
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("validate_inventory_for_architecture"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got 'totally_unknown'"), std::string::npos) << msg;
    }
}

// ── Architecture allow-list tests (PR G1.1) ───────────────────────────────────

TEST(Loader, ArchitectureAllowListAcceptsQwen3) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "qwen3";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsQwen2) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "qwen2";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsQwen35) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "qwen35";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsQwen35Moe) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "qwen35moe";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsGemma) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "gemma";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListRejectsUnknownWithFailLoudFormat) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "unknown_arch_xyz";
    try {
        loader.validate_architecture(m);
        FAIL() << "expected GGUFLoadError for unknown architecture";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        // Fail-loud contract: name the parameter, expected value, then actual.
        EXPECT_NE(msg.find("validate_architecture"), std::string::npos) << msg;
        EXPECT_NE(msg.find("expected one of"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got 'unknown_arch_xyz'"), std::string::npos) << msg;
        // Allow-list members must appear in the message so users can self-correct.
        EXPECT_NE(msg.find("'gemma'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'qwen3'"), std::string::npos) << msg;
    }
}

TEST(Loader, ArchitectureAllowListRejectsEmpty) {
    QwenGGUFLoader loader;
    Qwen3Metadata m; m.architecture = "";
    EXPECT_THROW(loader.validate_architecture(m), GGUFLoadError);
}

// ── GGUFLoader build tests ────────────────────────────────────────────────────

TEST(Loader, GGUFLoaderDefaultConstructs) {
    QwenGGUFLoader loader;
    EXPECT_FALSE(loader.is_loaded());
}

TEST(Loader, CreateLoaderFactoryReturnsNonNull) {
    auto loader = create_gguf_loader();
    ASSERT_NE(loader, nullptr);
    EXPECT_FALSE(loader->is_loaded());
}

TEST(Loader, LoadNonexistentPathThrows) {
    QwenGGUFLoader loader;
    EXPECT_THROW(
        loader.load_model("/tmp/qwen_inference_test_nonexistent_loader.gguf"),
        std::exception
    );
}

// ── Tokenizer build tests ─────────────────────────────────────────────────────

TEST(Loader, TokenizerVocabSizeMatchesInput) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    EXPECT_EQ(tok.get_vocabulary().size(), 257u);
}

TEST(Loader, TokenizerReportsEOSTokenId) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    EXPECT_EQ(tok.get_eos_token_id(), 256);
}

TEST(Loader, TokenizerDecodeEOSReturnsControlString) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    // EOS is CONTROL type; decode returns the token string verbatim.
    EXPECT_EQ(tok.decode(256), "<|endoftext|>");
}

// ── Known-answer round-trip tests ─────────────────────────────────────────────
//
// No BPE merges in this vocabulary.  Every ASCII char encodes to exactly one
// byte token and decodes back via the GPT-2 byte mapping.

TEST(Loader, TokenizerRoundTripSingleWord) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "Hello";
    auto ids = tok.encode(text);
    ASSERT_EQ(ids.size(), 5u);  // H e l l o → 5 byte tokens (no merges)
    EXPECT_EQ(tok.decode(ids), text);
}

TEST(Loader, TokenizerRoundTripPhraseWithSpace) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "hello world";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST(Loader, TokenizerRoundTripDigitsAndPunctuation) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "count 42!";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}
