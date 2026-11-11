#include "tokenizer/tokenizer.hpp"
#include "gguf/gguf_parser.hpp"
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <cstdio>
#include <cctype>
#include <climits>

namespace inference {

namespace {

// Decode a JSON string literal body (the part between the quotes) into UTF-8.
std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c != '\\') {
            out += c;
            continue;
        }
        if (i + 1 >= s.size()) break;
        char e = s[++i];
        switch (e) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'u': {
                if (i + 4 >= s.size()) break;
                unsigned cp = 0;
                for (int k = 1; k <= 4; ++k) {
                    char h = s[i + k];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                }
                i += 4;
                // Handle surrogate pairs
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < s.size() &&
                    s[i + 1] == '\\' && s[i + 2] == 'u') {
                    unsigned lo = 0;
                    for (int k = 3; k <= 6; ++k) {
                        char h = s[i + k];
                        lo <<= 4;
                        if (h >= '0' && h <= '9') lo |= (h - '0');
                        else if (h >= 'a' && h <= 'f') lo |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') lo |= (h - 'A' + 10);
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
                // Encode codepoint as UTF-8
                if (cp < 0x80) {
                    out += static_cast<char>(cp);
                } else if (cp < 0x800) {
                    out += static_cast<char>(0xC0 | (cp >> 6));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    out += static_cast<char>(0xE0 | (cp >> 12));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                } else {
                    out += static_cast<char>(0xF0 | (cp >> 18));
                    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                break;
            }
            default: out += e; break;
        }
    }
    return out;
}

// Read a quoted JSON string starting at index `pos` (which must point at '"').
// Returns the unescaped content and advances `pos` past the closing quote.
std::string read_json_string(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') throw std::runtime_error("JSON: expected string");
    ++pos;
    std::string raw;
    while (pos < s.size()) {
        char c = s[pos];
        if (c == '\\') {
            if (pos + 1 >= s.size()) break;
            raw += c;
            raw += s[pos + 1];
            pos += 2;
            continue;
        }
        if (c == '"') { ++pos; break; }
        raw += c;
        ++pos;
    }
    return json_unescape(raw);
}

// Find the matching closing brace/bracket for the one at `open`.
size_t match_bracket(const std::string& s, size_t open) {
    char o = s[open];
    char c = (o == '{') ? '}' : ']';
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = open; i < s.size(); ++i) {
        char ch = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') { in_str = true; continue; }
        if (ch == o) ++depth;
        else if (ch == c) {
            if (--depth == 0) return i;
        }
    }
    throw std::runtime_error("JSON: unbalanced brackets");
}

}  // namespace

Tokenizer::Tokenizer(const std::string& model_path) {
    load(model_path);
}

void Tokenizer::load(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    if (ext != ".json") {
        throw std::runtime_error(
            "Only HuggingFace tokenizer.json is supported (got '" + ext + "'). "
            "Download tokenizer.json from the model repository.");
    }
    load_huggingface(path);
}

void Tokenizer::load_from_gguf(const GGUFParser& parser) {
    std::vector<std::string> tokens;
    try {
        tokens = parser.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
    } catch (...) {
        throw std::runtime_error("GGUF has no tokenizer.ggml.tokens metadata");
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        int id = static_cast<int>(i);
        vocab_[tokens[i]] = id;
        id_to_piece_[id] = tokens[i];
    }
    vocab_size_ = static_cast<int>(tokens.size());

    try {
        auto merges = parser.get_metadata<std::vector<std::string>>("tokenizer.ggml.merges");
        int rank = 0;
        for (const auto& merge : merges) {
            size_t sp = merge.find(' ');
            if (sp != std::string::npos) {
                std::string key = merge.substr(0, sp) + '\x01' + merge.substr(sp + 1);
                merge_rank_.emplace(key, rank);
            }
            ++rank;
        }
    } catch (...) {
        // No merges: still usable as a longest-match-ish fallback
    }

    auto special = [&](const std::string& key, int def) -> int {
        try {
            return static_cast<int>(parser.get_metadata<uint32_t>(key));
        } catch (...) {
            return def;
        }
    };
    bos_id_ = special("tokenizer.ggml.bos_token_id", 1);
    eos_id_ = special("tokenizer.ggml.eos_token_id", 2);
    unk_id_ = special("tokenizer.ggml.unknown_token_id", 0);
    pad_id_ = special("tokenizer.ggml.padding_token_id", -1);

    try {
        chat_template_ = parser.get_metadata<std::string>("tokenizer.chat_template");
    } catch (...) {
        chat_template_.clear();
    }

    collect_special_pieces();
}

void Tokenizer::load_huggingface(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open tokenizer: " + path);
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // ---- Parse "vocab": { token: id, ... } ----
    size_t vocab_key = content.find("\"vocab\"");
    if (vocab_key == std::string::npos) {
        throw std::runtime_error("tokenizer.json: no \"vocab\" found");
    }
    size_t vopen = content.find('{', vocab_key);
    if (vopen == std::string::npos) throw std::runtime_error("tokenizer.json: bad vocab");
    size_t vclose = match_bracket(content, vopen);

    size_t pos = vopen + 1;
    while (pos < vclose) {
        // skip separators/whitespace
        while (pos < vclose && (content[pos] == ',' || std::isspace(static_cast<unsigned char>(content[pos])))) ++pos;
        if (pos >= vclose) break;
        if (content[pos] != '"') { ++pos; continue; }

        std::string token = read_json_string(content, pos);
        while (pos < vclose && std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
        if (pos >= vclose || content[pos] != ':') continue;
        ++pos;
        while (pos < vclose && std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
        int id = 0;
        bool neg = false;
        if (pos < vclose && content[pos] == '-') { neg = true; ++pos; }
        while (pos < vclose && content[pos] >= '0' && content[pos] <= '9') {
            id = id * 10 + (content[pos] - '0');
            ++pos;
        }
        if (neg) id = -id;

        vocab_[token] = id;
        id_to_piece_[id] = token;
    }

    vocab_size_ = static_cast<int>(vocab_.size());
    if (vocab_size_ == 0) throw std::runtime_error("tokenizer.json: empty vocab");

    // ---- Parse "merges": ["a b", ...] ----
    size_t merges_key = content.find("\"merges\"");
    if (merges_key != std::string::npos) {
        size_t mopen = content.find('[', merges_key);
        if (mopen != std::string::npos) {
            size_t mclose = match_bracket(content, mopen);
            size_t p = mopen + 1;
            int rank = 0;
            while (p < mclose) {
                while (p < mclose && content[p] != '"') ++p;
                if (p >= mclose) break;
                std::string merge = read_json_string(content, p);
                size_t sp = merge.find(' ');
                if (sp != std::string::npos) {
                    std::string key = merge.substr(0, sp) + '\x01' + merge.substr(sp + 1);
                    merge_rank_.emplace(key, rank);
                }
                ++rank;
            }
        }
    }

    // ---- Special token ids (top-level added_tokens / *_token_id) ----
    auto find_special = [&](const std::string& key, int default_val) -> int {
        size_t p = content.find("\"" + key + "\"");
        if (p == std::string::npos) return default_val;
        size_t colon = content.find(':', p);
        if (colon == std::string::npos) return default_val;
        size_t q = colon + 1;
        while (q < content.size() && std::isspace(static_cast<unsigned char>(content[q]))) ++q;
        bool neg = false;
        if (q < content.size() && content[q] == '-') { neg = true; ++q; }
        int val = 0;
        size_t start = q;
        while (q < content.size() && content[q] >= '0' && content[q] <= '9') {
            val = val * 10 + (content[q] - '0');
            ++q;
        }
        if (q == start) return default_val;
        return neg ? -val : val;
    };

    bos_id_ = find_special("bos_token_id", 1);
    eos_id_ = find_special("eos_token_id", 2);
    pad_id_ = find_special("pad_token_id", -1);
    unk_id_ = find_special("unk_token_id", 0);

    // Optional chat template (same string HuggingFace exposes).
    size_t ct = content.find("\"chat_template\"");
    if (ct != std::string::npos) {
        size_t colon = content.find(':', ct);
        if (colon != std::string::npos) {
            size_t q = content.find('"', colon + 1);
            if (q != std::string::npos) {
                try {
                    chat_template_ = read_json_string(content, q);
                } catch (...) {
                    chat_template_.clear();
                }
            }
        }
    }

    collect_special_pieces();
}

void Tokenizer::collect_special_pieces() {
    special_pieces_.clear();
    for (const auto& entry : vocab_) {
        const std::string& piece = entry.first;
        if (piece.size() >= 3 && piece.front() == '<' && piece.back() == '>') {
            special_pieces_.emplace_back(piece, entry.second);
        }
    }
    // Longest first so <|start_header_id|> wins over shorter prefixes.
    std::sort(special_pieces_.begin(), special_pieces_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

std::vector<std::string> Tokenizer::utf8_symbols(const std::string& text) const {
    // Split into UTF-8 codepoints.
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) len = 1;
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

std::vector<int> Tokenizer::bpe_encode_symbols(const std::vector<std::string>& input) const {
    if (input.empty()) return {};

    std::vector<std::string> symbols = input;

    // Repeatedly merge the adjacent pair with the best (lowest) rank.
    while (symbols.size() > 1) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = symbols[i] + '\x01' + symbols[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank == INT32_MAX) break;

        // Merge all occurrences of the winning pair in one pass.
        const std::string left = symbols[best_i];
        const std::string right = symbols[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
                merged.push_back(left + right);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                ++i;
            }
        }
        symbols.swap(merged);
    }

    // Map symbols to ids, with byte fallback for unknown symbols.
    std::vector<int> ids;
    for (const auto& sym : symbols) {
        auto it = vocab_.find(sym);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            continue;
        }
        bool all_bytes = true;
        std::vector<int> byte_ids;
        for (unsigned char b : sym) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto bit = vocab_.find(buf);
            if (bit == vocab_.end()) { all_bytes = false; break; }
            byte_ids.push_back(bit->second);
        }
        if (all_bytes) {
            ids.insert(ids.end(), byte_ids.begin(), byte_ids.end());
        } else {
            ids.push_back(unk_id_);
        }
    }
    return ids;
}

std::vector<int> Tokenizer::encode_plain(const std::string& text) const {
    // SentencePiece-style: spaces become '▁', and a dummy prefix is added.
    std::string normalized;
    normalized.reserve(text.size() + 4);
    normalized += "\xE2\x96\x81";  // ▁ (U+2581)
    for (char c : text) {
        if (c == ' ') normalized += "\xE2\x96\x81";
        else normalized += c;
    }

    // BPE per whitespace-marked chunk. Merging across ▁ boundaries is valid
    // for Llama-style vocabularies, but chunking keeps behavior predictable.
    std::vector<int> tokens;
    std::vector<std::string> chunk;
    for (const auto& sym : utf8_symbols(normalized)) {
        chunk.push_back(sym);
        if (sym == "\xE2\x96\x81" && chunk.size() > 1) {
            std::string last = chunk.back();
            chunk.pop_back();
            std::vector<int> part = bpe_encode_symbols(chunk);
            tokens.insert(tokens.end(), part.begin(), part.end());
            chunk.clear();
            chunk.push_back(last);
        }
    }
    if (!chunk.empty()) {
        std::vector<int> part = bpe_encode_symbols(chunk);
        tokens.insert(tokens.end(), part.begin(), part.end());
    }
    return tokens;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int> tokens;
    if (add_bos && bos_id_ >= 0) tokens.push_back(bos_id_);

    auto flush_plain = [&](std::string& plain) {
        if (plain.empty()) return;
        std::vector<int> part = encode_plain(plain);
        tokens.insert(tokens.end(), part.begin(), part.end());
        plain.clear();
    };

    // Emit known special markers verbatim (<|user|>, <s>, <start_of_turn>, ...)
    // instead of letting BPE shred them into characters.
    std::string plain;
    for (size_t i = 0; i < text.size();) {
        bool matched = false;
        if (text[i] == '<') {
            for (const auto& [piece, id] : special_pieces_) {
                if (text.size() - i >= piece.size() &&
                    text.compare(i, piece.size(), piece) == 0) {
                    flush_plain(plain);
                    tokens.push_back(id);
                    i += piece.size();
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            plain.push_back(text[i]);
            ++i;
        }
    }
    flush_plain(plain);

    if (add_eos && eos_id_ >= 0) tokens.push_back(eos_id_);
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    std::string pending_bytes;

    auto flush_bytes = [&]() {
        out += pending_bytes;
        pending_bytes.clear();
    };

    for (int id : tokens) {
        if (id == bos_id_ || id == eos_id_ || id == pad_id_) continue;
        auto it = id_to_piece_.find(id);
        if (it == id_to_piece_.end()) continue;

        const std::string& piece = it->second;
        if (piece.size() == 6 && piece.rfind("<0x", 0) == 0 && piece.back() == '>') {
            unsigned val = 0;
            for (size_t k = 3; k <= 4; ++k) {
                char h = piece[k];
                val <<= 4;
                if (h >= '0' && h <= '9') val |= (h - '0');
                else if (h >= 'a' && h <= 'f') val |= (h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') val |= (h - 'A' + 10);
            }
            pending_bytes += static_cast<char>(val);
            continue;
        }

        flush_bytes();

        // Replace SentencePiece/GPT-2 space markers with real spaces.
        for (size_t i = 0; i < piece.size();) {
            if (i + 2 < piece.size() &&
                static_cast<unsigned char>(piece[i]) == 0xE2 &&
                static_cast<unsigned char>(piece[i + 1]) == 0x96 &&
                static_cast<unsigned char>(piece[i + 2]) == 0x81) {
                out += ' ';
                i += 3;
            } else if (i + 1 < piece.size() &&
                       static_cast<unsigned char>(piece[i]) == 0xC4 &&
                       static_cast<unsigned char>(piece[i + 1]) == 0xA0) {
                out += ' ';  // Ġ
                i += 2;
            } else {
                out += piece[i];
                ++i;
            }
        }
    }
    flush_bytes();
    return out;
}

std::string Tokenizer::eos_piece() const {
    auto it = id_to_piece_.find(eos_id_);
    if (it != id_to_piece_.end()) return it->second;
    return "</s>";
}

ChatFormat Tokenizer::chat_format() const {
    if (chat_template_.empty()) return ChatFormat::Generic;
    auto has = [&](const char* s) { return chat_template_.find(s) != std::string::npos; };
    if (has("<|start_header_id|>")) return ChatFormat::Llama3;
    if (has("<|im_start|>")) return ChatFormat::ChatML;
    if (has("<|user|>") && has("<|assistant|>")) return ChatFormat::Zephyr;
    if (has("[INST]")) return ChatFormat::Mistral;
    if (has("<start_of_turn>")) return ChatFormat::Gemma;
    return ChatFormat::Generic;
}

bool Tokenizer::chat_needs_bos() const {
    switch (chat_format()) {
        case ChatFormat::Zephyr:
        case ChatFormat::Gemma:
        case ChatFormat::Generic:
            return true;
        case ChatFormat::Llama3:
        case ChatFormat::ChatML:
        case ChatFormat::Mistral:
            return false;
    }
    return true;
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages,
                                           bool add_generation_prompt) const {
    const std::string eos = eos_piece();

    std::string system;
    std::vector<ChatMessage> turns;
    for (const auto& m : messages) {
        if (m.role == "system") {
            if (!system.empty()) system += "\n";
            system += m.content;
        } else {
            turns.push_back(m);
        }
    }

    const ChatFormat format = chat_format();
    std::string out;

    auto role_or = [](const ChatMessage& m, const char* fallback) {
        return m.role.empty() ? std::string(fallback) : m.role;
    };

    switch (format) {
        case ChatFormat::Zephyr: {
            if (!system.empty()) out += "<|system|>\n" + system + eos + "\n";
            for (const auto& m : turns) {
                if (m.role == "user") out += "<|user|>\n" + m.content + eos + "\n";
                else if (m.role == "assistant") out += "<|assistant|>\n" + m.content + eos + "\n";
            }
            if (add_generation_prompt) out += "<|assistant|>\n";
            break;
        }
        case ChatFormat::Llama3: {
            out += "<|begin_of_text|>";
            if (!system.empty()) {
                out += "<|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>";
            }
            for (const auto& m : turns) {
                out += "<|start_header_id|>" + role_or(m, "user") +
                       "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
            }
            if (add_generation_prompt) {
                out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }
            break;
        }
        case ChatFormat::ChatML: {
            if (!system.empty()) out += "<|im_start|>system\n" + system + "<|im_end|>\n";
            for (const auto& m : turns) {
                out += "<|im_start|>" + role_or(m, "user") + "\n" +
                       m.content + "<|im_end|>\n";
            }
            if (add_generation_prompt) out += "<|im_start|>assistant\n";
            break;
        }
        case ChatFormat::Mistral: {
            out += "<s>";
            bool first_user = true;
            for (const auto& m : turns) {
                if (m.role == "assistant") {
                    out += " " + m.content + "</s>";
                } else {
                    out += "[INST] ";
                    if (first_user && !system.empty()) out += system + "\n";
                    out += m.content + " [/INST]";
                    first_user = false;
                }
            }
            // Generation starts right after the last [/INST]; nothing to add.
            (void)add_generation_prompt;
            break;
        }
        case ChatFormat::Gemma: {
            bool first_user = true;
            for (const auto& m : turns) {
                if (m.role == "assistant") {
                    out += "<start_of_turn>model\n" + m.content + "<end_of_turn>\n";
                } else {
                    out += "<start_of_turn>user\n";
                    if (first_user && !system.empty()) out += system + "\n";
                    out += m.content + "<end_of_turn>\n";
                    first_user = false;
                }
            }
            if (add_generation_prompt) out += "<start_of_turn>model\n";
            break;
        }
        case ChatFormat::Generic:
        default: {
            if (!system.empty()) out += "System: " + system + "\n";
            for (const auto& m : turns) {
                if (m.role == "assistant") out += "Assistant: " + m.content + "\n";
                else out += "User: " + m.content + "\n";
            }
            if (add_generation_prompt) out += "Assistant:";
            break;
        }
    }
    return out;
}

}  // namespace inference