#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <utility>

namespace inference {

class GGUFParser;

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

enum class ChatFormat {
    Generic,
    Zephyr,
    Llama3,
    ChatML,
    Mistral,
    Gemma,
};

// BPE tokenizer compatible with Llama/Mistral style tokenizer.json files.
class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const std::string& model_path);

    // Load from tokenizer.json (HuggingFace format)
    void load(const std::string& path);

    // Load vocab/merges/special ids embedded in a GGUF file
    void load_from_gguf(const GGUFParser& parser);

    // Chat template support
    void set_chat_template(std::string tmpl) { chat_template_ = std::move(tmpl); }
    bool has_chat_template() const { return !chat_template_.empty(); }
    const std::string& chat_template() const { return chat_template_; }
    ChatFormat chat_format() const;
    bool chat_needs_bos() const;
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt = true) const;
    std::string eos_piece() const;

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = true, bool add_eos = false) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens) const;

    // Special tokens
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }
    int unk_id() const { return unk_id_; }

    int vocab_size() const { return vocab_size_; }

private:
    void load_huggingface(const std::string& path);
    std::vector<int> encode_plain(const std::string& text) const;
    void collect_special_pieces();

    // BPE core
    std::vector<std::string> utf8_symbols(const std::string& text) const;
    std::vector<int> bpe_encode_symbols(const std::vector<std::string>& symbols) const;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_piece_;
    // Merge priority: key = "left\x01right", value = rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;
    // Exact-match special markers such as <|user|>, <s>, </s>, <start_of_turn>
    std::vector<std::pair<std::string, int>> special_pieces_;
    std::string chat_template_;

    int vocab_size_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int pad_id_ = 0;
    int unk_id_ = 0;
};

}  // namespace inference
