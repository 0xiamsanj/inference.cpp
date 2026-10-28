#pragma once

#include "core/tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <fstream>
#include <memory>

namespace inference {

// GGUF metadata value types
using MetadataValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
    uint64_t, int64_t, float, double, bool, std::string,
    std::vector<uint8_t>, std::vector<int8_t>, std::vector<uint16_t>,
    std::vector<int16_t>, std::vector<uint32_t>, std::vector<int32_t>,
    std::vector<uint64_t>, std::vector<int64_t>, std::vector<float>,
    std::vector<double>, std::vector<bool>, std::vector<std::string>
>;

struct GGUFMetadata {
    std::string key;
    MetadataValue value;
};

struct GGUFTensorInfo {
    std::string name;
    DataType dtype;
    Shape shape;
    size_t offset;      // Offset in file (after metadata)
    size_t data_size;   // Size in bytes
};

class GGUFParser {
public:
    explicit GGUFParser(const std::string& path);
    ~GGUFParser();

    // Metadata access
    const std::vector<GGUFMetadata>& metadata() const { return metadata_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    // Get metadata by key (with type checking)
    template<typename T>
    T get_metadata(const std::string& key) const;

    // Load tensor by name into pre-allocated Tensor
    void load_tensor(const std::string& name, Tensor& out) const;

    // Load all tensors (returns map of name -> Tensor)
    std::unordered_map<std::string, Tensor> load_all_tensors(Device device = Device::cpu()) const;

    // Raw read of a tensor's bytes into dst (exactly data_size bytes)
    void read_tensor_data(const GGUFTensorInfo& info, void* dst) const;

    // Find tensor info by name, returns nullptr if absent
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    // Model info helpers
    int get_block_count() const;
    int get_embedding_dim() const;
    int get_num_heads() const;
    int get_head_dim() const;
    int get_vocab_size() const;
    std::string get_arch_name() const;

private:
    void parse_header();
    void parse_metadata();
    void parse_tensor_infos();

    std::string path_;
    std::ifstream file_;
    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_kv_count_ = 0;
    std::vector<GGUFMetadata> metadata_;
    std::vector<GGUFTensorInfo> tensors_;
    size_t tensor_data_offset_ = 0;
};

// Helper to extract metadata with default
template<typename T>
T get_metadata_or(const GGUFParser& parser, const std::string& key, T default_value) {
    try {
        return parser.get_metadata<T>(key);
    } catch (...) {
        return default_value;
    }
}

}  // namespace inference