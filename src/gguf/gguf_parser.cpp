#include "gguf/gguf_parser.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <type_traits>
#include <endian.h>

namespace inference {

// Endian helpers (GGUF is little-endian)
template<typename T>
T read_le(std::ifstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) throw std::runtime_error("Failed to read from file");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // Byte swap for big-endian systems
    char* bytes = reinterpret_cast<char*>(&value);
    std::reverse(bytes, bytes + sizeof(T));
#endif
    return value;
}

template<typename T>
std::vector<T> read_vector_le(std::ifstream& file, size_t count) {
    std::vector<T> vec(count);
    file.read(reinterpret_cast<char*>(vec.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("Failed to read vector from file");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (auto& v : vec) {
        char* bytes = reinterpret_cast<char*>(&v);
        std::reverse(bytes, bytes + sizeof(T));
    }
#endif
    return vec;
}

std::string read_string(std::ifstream& file) {
    uint64_t len = read_le<uint64_t>(file);
    std::string str(len, '\0');
    file.read(&str[0], len);
    if (!file) throw std::runtime_error("Failed to read string");
    return str;
}

GGUFParser::GGUFParser(const std::string& path) : path_(path) {
    file_.open(path, std::ios::binary);
    if (!file_) {
        throw std::runtime_error("Cannot open GGUF file: " + path);
    }
    parse_header();
    parse_metadata();
    parse_tensor_infos();
}

GGUFParser::~GGUFParser() {
    if (file_.is_open()) file_.close();
}

void GGUFParser::parse_header() {
    // Magic: "GGUF" (0x46554747)
    uint32_t magic = read_le<uint32_t>(file_);
    if (magic != 0x46554747) {
        throw std::runtime_error("Invalid GGUF magic: " + std::to_string(magic));
    }

    version_ = read_le<uint32_t>(file_);
    tensor_count_ = read_le<uint64_t>(file_);
    metadata_kv_count_ = read_le<uint64_t>(file_);

    if (version_ > 3) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version_));
    }
}

void GGUFParser::parse_metadata() {
    metadata_.reserve(metadata_kv_count_);

    for (uint64_t i = 0; i < metadata_kv_count_; ++i) {
        std::string key = read_string(file_);
        uint32_t value_type = read_le<uint32_t>(file_);

        MetadataValue value;

        // GGUF metadata value types
        // 0:UINT8 1:INT8 2:UINT16 3:INT16 4:UINT32 5:INT32 6:FLOAT32
        // 7:BOOL 8:STRING 9:ARRAY 10:UINT64 11:INT64 12:FLOAT64
        switch (value_type) {
            case 0:  value = read_le<uint8_t>(file_); break;
            case 1:  value = read_le<int8_t>(file_); break;
            case 2:  value = read_le<uint16_t>(file_); break;
            case 3:  value = read_le<int16_t>(file_); break;
            case 4:  value = read_le<uint32_t>(file_); break;
            case 5:  value = read_le<int32_t>(file_); break;
            case 6:  value = read_le<float>(file_); break;
            case 7:  value = static_cast<bool>(read_le<uint8_t>(file_)); break;
            case 8:  value = read_string(file_); break;

            case 9: {  // Array
                uint32_t array_type = read_le<uint32_t>(file_);
                uint64_t array_len = read_le<uint64_t>(file_);
                switch (array_type) {
                    case 0: value = read_vector_le<uint8_t>(file_, array_len); break;
                    case 1: value = read_vector_le<int8_t>(file_, array_len); break;
                    case 2: value = read_vector_le<uint16_t>(file_, array_len); break;
                    case 3: value = read_vector_le<int16_t>(file_, array_len); break;
                    case 4: value = read_vector_le<uint32_t>(file_, array_len); break;
                    case 5: value = read_vector_le<int32_t>(file_, array_len); break;
                    case 6: value = read_vector_le<float>(file_, array_len); break;
                    case 7: value = read_vector_le<uint8_t>(file_, array_len); break;
                    case 10: value = read_vector_le<uint64_t>(file_, array_len); break;
                    case 11: value = read_vector_le<int64_t>(file_, array_len); break;
                    case 12: value = read_vector_le<double>(file_, array_len); break;
                    case 8: {
                        std::vector<std::string> arr;
                        arr.reserve(array_len);
                        for (uint64_t j = 0; j < array_len; ++j) arr.push_back(read_string(file_));
                        value = arr;
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported array type: " + std::to_string(array_type));
                }
                break;
            }

            case 10: value = read_le<uint64_t>(file_); break;
            case 11: value = read_le<int64_t>(file_); break;
            case 12: value = read_le<double>(file_); break;

            default:
                throw std::runtime_error("Unknown metadata value type: " + std::to_string(value_type));
        }

        metadata_.push_back({key, value});
    }
}

void GGUFParser::parse_tensor_infos() {
    tensors_.reserve(tensor_count_);

    for (uint64_t i = 0; i < tensor_count_; ++i) {
        std::string name = read_string(file_);
        // GGUF tensor info order: name, n_dims, dims[n_dims], type, offset
        uint32_t n_dims = read_le<uint32_t>(file_);

        Shape shape(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            shape[d] = static_cast<int64_t>(read_le<uint64_t>(file_));
        }

        uint32_t dtype = read_le<uint32_t>(file_);
        uint64_t offset = read_le<uint64_t>(file_);

        GGUFTensorInfo info;
        info.name = name;
        info.dtype = static_cast<DataType>(dtype);
        info.shape = shape;
        info.offset = offset;
        info.data_size = type_nbytes(info.dtype, shape_numel(shape));

        tensors_.push_back(std::move(info));
    }

    // Tensor data begins after the descriptors, padded to `general.alignment`
    // (default 32 bytes in the GGUF spec).
    tensor_data_offset_ = static_cast<size_t>(file_.tellg());
    uint32_t alignment = 32;
    try {
        alignment = get_metadata<uint32_t>("general.alignment");
    } catch (...) {}
    if (alignment == 0) alignment = 32;
    tensor_data_offset_ = (tensor_data_offset_ + alignment - 1) / alignment * alignment;
}

void GGUFParser::read_tensor_data(const GGUFTensorInfo& info, void* dst) const {
    std::ifstream data_file(path_, std::ios::binary);
    if (!data_file) throw std::runtime_error("Cannot reopen file for tensor data");

    data_file.seekg(static_cast<std::streamoff>(tensor_data_offset_ + info.offset));
    if (!data_file) throw std::runtime_error("Seek failed for tensor: " + info.name);

    data_file.read(static_cast<char*>(dst), info.data_size);
    if (!data_file) throw std::runtime_error("Failed to read tensor data: " + info.name);
}

template<typename T>
T GGUFParser::get_metadata(const std::string& key) const {
    for (const auto& meta : metadata_) {
        if (meta.key == key) {
            // Exact type match first
            if (auto* val = std::get_if<T>(&meta.value)) {
                return *val;
            }
            // Allow lossless-ish numeric conversions between integer widths,
            // and between float/double. GGUF writers are inconsistent here.
            if constexpr (std::is_integral_v<T>) {
                if (auto* v = std::get_if<uint32_t>(&meta.value)) return static_cast<T>(*v);
                if (auto* v = std::get_if<int32_t>(&meta.value))  return static_cast<T>(*v);
                if (auto* v = std::get_if<uint64_t>(&meta.value)) return static_cast<T>(*v);
                if (auto* v = std::get_if<int64_t>(&meta.value))  return static_cast<T>(*v);
                if (auto* v = std::get_if<uint16_t>(&meta.value)) return static_cast<T>(*v);
                if (auto* v = std::get_if<int16_t>(&meta.value))  return static_cast<T>(*v);
                if (auto* v = std::get_if<uint8_t>(&meta.value))  return static_cast<T>(*v);
                if (auto* v = std::get_if<int8_t>(&meta.value))   return static_cast<T>(*v);
                if (auto* v = std::get_if<bool>(&meta.value))     return static_cast<T>(*v);
            }
            if constexpr (std::is_floating_point_v<T>) {
                if (auto* v = std::get_if<float>(&meta.value))  return static_cast<T>(*v);
                if (auto* v = std::get_if<double>(&meta.value)) return static_cast<T>(*v);
            }
            throw std::runtime_error("Metadata type mismatch for key: " + key);
        }
    }
    throw std::runtime_error("Metadata key not found: " + key);
}

// Explicit instantiations for common types
template std::string GGUFParser::get_metadata<std::string>(const std::string&) const;
template uint32_t GGUFParser::get_metadata<uint32_t>(const std::string&) const;
template uint64_t GGUFParser::get_metadata<uint64_t>(const std::string&) const;
template int32_t GGUFParser::get_metadata<int32_t>(const std::string&) const;
template int64_t GGUFParser::get_metadata<int64_t>(const std::string&) const;
template float GGUFParser::get_metadata<float>(const std::string&) const;
template bool GGUFParser::get_metadata<bool>(const std::string&) const;
template std::vector<uint32_t> GGUFParser::get_metadata<std::vector<uint32_t>>(const std::string&) const;
template std::vector<std::string> GGUFParser::get_metadata<std::vector<std::string>>(const std::string&) const;

void GGUFParser::load_tensor(const std::string& name, Tensor& out) const {
    const GGUFTensorInfo* info = find_tensor(name);
    if (!info) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    if (out.dtype() != info->dtype) {
        throw std::runtime_error("Tensor dtype mismatch for " + name);
    }
    if (out.numel() != shape_numel(info->shape)) {
        throw std::runtime_error("Tensor shape mismatch for " + name);
    }
    read_tensor_data(*info, out.data());
}

const GGUFTensorInfo* GGUFParser::find_tensor(const std::string& name) const {
    for (const auto& info : tensors_) {
        if (info.name == name) return &info;
    }
    return nullptr;
}

std::unordered_map<std::string, Tensor> GGUFParser::load_all_tensors(Device device) const {
    std::unordered_map<std::string, Tensor> result;
    result.reserve(tensors_.size());

    for (const auto& info : tensors_) {
        Tensor t(info.dtype, info.shape, device);
        read_tensor_data(info, t.data());
        result.emplace(info.name, std::move(t));
    }
    return result;
}

int GGUFParser::get_block_count() const {
    return get_metadata_or(*this, "llama.block_count", 0);
}

int GGUFParser::get_embedding_dim() const {
    return get_metadata_or(*this, "llama.embedding_length", 0);
}

int GGUFParser::get_num_heads() const {
    return get_metadata_or(*this, "llama.attention.head_count", 0);
}

int GGUFParser::get_head_dim() const {
    int n_heads = get_num_heads();
    int embed = get_embedding_dim();
    if (n_heads > 0) return embed / n_heads;
    return get_metadata_or(*this, "llama.attention.head_dim", 0);
}

int GGUFParser::get_vocab_size() const {
    try {
        return get_metadata<uint32_t>("llama.vocab_size");
    } catch (...) {}
    try {
        return static_cast<int>(get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens").size());
    } catch (...) {}
    return 0;
}

std::string GGUFParser::get_arch_name() const {
    return get_metadata_or(*this, "general.architecture", std::string("unknown"));
}

}  // namespace inference