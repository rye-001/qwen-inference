// test_chat_template.cpp — PR G1.7
//
// Verifies the chat-template renderer produces:
//   - Qwen ChatML format for "qwen3"/"qwen2"
//   - Gemma format for "gemma"
// and that role-mapping (assistant → model) is applied consistently.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../../src/cli/cli-args.h"

TEST(ChatTemplate, DetectArchitectureMaps) {
    EXPECT_EQ(detect_chat_template("qwen3"), ChatTemplateKind::ChatML);
    EXPECT_EQ(detect_chat_template("qwen2"), ChatTemplateKind::ChatML);
    EXPECT_EQ(detect_chat_template("qwen35"), ChatTemplateKind::ChatML);
    EXPECT_EQ(detect_chat_template("qwen35moe"), ChatTemplateKind::ChatML);
    EXPECT_EQ(detect_chat_template("gemma"), ChatTemplateKind::Gemma);
    // Future Gemma generations resolve to the same template.
    EXPECT_EQ(detect_chat_template("gemma2"), ChatTemplateKind::Gemma);
    EXPECT_EQ(detect_chat_template("gemma3"), ChatTemplateKind::Gemma);
    EXPECT_EQ(detect_chat_template("gemma4"), ChatTemplateKind::Gemma);
}

TEST(ChatTemplate, ChatMLRendersIMTagsAndAssistantPrompt) {
    std::vector<ChatMessage> hist = {{"user", "hi"}};
    std::string out = apply_chat_template(hist, ChatTemplateKind::ChatML, true);
    EXPECT_EQ(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
}

TEST(ChatTemplate, GemmaSingleUserTurnWithAssistantPrompt) {
    std::vector<ChatMessage> hist = {{"user", "hi"}};
    std::string out = apply_chat_template(hist, ChatTemplateKind::Gemma, true);
    EXPECT_EQ(out, "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n");
}

TEST(ChatTemplate, GemmaAssistantRoleMapsToModel) {
    std::vector<ChatMessage> hist = {{"assistant", "yo"}};
    std::string out = apply_chat_template(hist, ChatTemplateKind::Gemma, false);
    EXPECT_EQ(out, "<start_of_turn>model\nyo<end_of_turn>\n");
}

TEST(ChatTemplate, GemmaSystemRoleEmittedAsUser) {
    // Gemma has no system slot — content lands in a user turn.
    std::vector<ChatMessage> hist = {{"system", "you are helpful"}};
    std::string out = apply_chat_template(hist, ChatTemplateKind::Gemma, false);
    EXPECT_EQ(out, "<start_of_turn>user\nyou are helpful<end_of_turn>\n");
}

TEST(ChatTemplate, GemmaMultiTurnDialog) {
    std::vector<ChatMessage> hist = {
        {"user", "hi"},
        {"assistant", "hello"},
        {"user", "and now?"},
    };
    std::string out = apply_chat_template(hist, ChatTemplateKind::Gemma, true);
    const std::string expected =
        "<start_of_turn>user\nhi<end_of_turn>\n"
        "<start_of_turn>model\nhello<end_of_turn>\n"
        "<start_of_turn>user\nand now?<end_of_turn>\n"
        "<start_of_turn>model\n";
    EXPECT_EQ(out, expected);
}

TEST(ChatTemplate, BackwardCompatibleSingleArgUsesChatML) {
    std::vector<ChatMessage> hist = {{"user", "x"}};
    std::string out = apply_chat_template(hist, true);
    EXPECT_EQ(out, "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n");
}

TEST(ChatTemplate, GemmaWithoutAssistantPromptOmitsOpenTurn) {
    std::vector<ChatMessage> hist = {{"user", "hi"}};
    std::string out = apply_chat_template(hist, ChatTemplateKind::Gemma, false);
    EXPECT_EQ(out, "<start_of_turn>user\nhi<end_of_turn>\n");
}
