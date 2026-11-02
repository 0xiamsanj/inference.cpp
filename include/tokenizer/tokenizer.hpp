#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdint>

namespace inference {

class GGUFParser;

// BPE tokenizer compatible with Llama/Mistral style tokenizer.json files.
class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const std::string& model_path);

    // Load from tokenizer.json (HuggingFace format)
    void load(const std::string& path);

    // Load vocab/merges/special ids embedded in a GGUF file
    void load_from_gguf(const GGUFParser& parser);

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

    // BPE core
    std::vector<std::string> utf8_symbols(const std::string& text) const;
    std::vector<int> bpe_encode_symbols(const std::vector<std::string>& symbols) const;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_piece_;
    // Merge priority: key = "left\x01right", value = rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;

    int vocab_size_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int pad_id_ = 0;
    int unk_id_ = 0;
};

}  // namespace inference