#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace inference {

// Data types matching GGUF
enum class DataType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 21,
};

// Device abstraction
enum class DeviceType {
    CPU,
    CUDA,
};

struct Device {
    DeviceType type;
    int id;  // For multi-GPU

    static Device cpu() { return {DeviceType::CPU, 0}; }
    static Device cuda(int id = 0) { return {DeviceType::CUDA, id}; }

    bool operator==(const Device& other) const {
        return type == other.type && id == other.id;
    }
};

// Shape utilities
using Shape = std::vector<int64_t>;

// Forward declarations for dtype helpers used by Tensor
inline size_t type_size(DataType dt);
inline int block_size(DataType dt);

inline int64_t shape_numel(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<>());
}

inline Shape shape_broadcast(const Shape& a, const Shape& b) {
    Shape result(std::max(a.size(), b.size()));
    auto it_a = a.rbegin();
    auto it_b = b.rbegin();
    auto it_r = result.rbegin();
    for (; it_a != a.rend() && it_b != b.rend(); ++it_a, ++it_b, ++it_r) {
        if (*it_a == *it_b) *it_r = *it_a;
        else if (*it_a == 1) *it_r = *it_b;
        else if (*it_b == 1) *it_r = *it_a;
        else throw std::runtime_error("Incompatible shapes for broadcast");
    }
    for (; it_a != a.rend(); ++it_a, ++it_r) *it_r = *it_a;
    for (; it_b != b.rend(); ++it_b, ++it_r) *it_r = *it_b;
    return result;
}

// Tensor: owns data, knows shape, dtype, device
class Tensor {
public:
    // Constructors
    Tensor() = default;

    Tensor(DataType dtype, const Shape& shape, Device device = Device::cpu())
        : dtype_(dtype), shape_(shape), device_(device) {
        allocate();
    }

    Tensor(DataType dtype, std::initializer_list<int64_t> shape, Device device = Device::cpu())
        : Tensor(dtype, Shape(shape), device) {}

    // Move semantics
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Copy semantics: deep copy (simple, predictable ownership)
    Tensor(const Tensor& other)
        : dtype_(other.dtype_), shape_(other.shape_), device_(other.device_) {
        allocate();
        if (other.data_ && byte_size() > 0) {
            std::memcpy(data_.get(), other.data_.get(), byte_size());
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        dtype_  = other.dtype_;
        shape_  = other.shape_;
        device_ = other.device_;
        allocate();
        if (other.data_ && byte_size() > 0) {
            std::memcpy(data_.get(), other.data_.get(), byte_size());
        }
        return *this;
    }

    // Accessors
    DataType dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    int64_t numel() const { return shape_numel(shape_); }
    Device device() const { return device_; }
    size_t byte_size() const { return numel() * type_size(dtype_); }
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }

    // Typed access (for F32/F16)
    template<typename T>
    T* as() { return static_cast<T*>(data()); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(data()); }

    // Reshape (no copy, checks compatibility)
    void reshape(const Shape& new_shape) {
        if (shape_numel(new_shape) != numel()) {
            throw std::runtime_error("Reshape size mismatch");
        }
        shape_ = new_shape;
    }

    // Squeeze/unsqueeze
    void squeeze(int dim = -1) {
        if (dim < 0) dim += shape_.size();
        if (shape_[dim] != 1) throw std::runtime_error("Cannot squeeze non-1 dimension");
        shape_.erase(shape_.begin() + dim);
    }

    void unsqueeze(int dim) {
        if (dim < 0) dim += shape_.size() + 1;
        shape_.insert(shape_.begin() + dim, 1);
    }

    // Clone (deep copy)
    Tensor clone() const {
        Tensor result(dtype_, shape_, device_);
        std::memcpy(result.data(), data(), byte_size());
        return result;
    }

    // To device (CPU <-> CUDA)
    Tensor to(Device target_device) const;

    // Fill with scalar
    void fill(float value);

    // Debug
    std::string debug_string() const;

private:
    void allocate();

    DataType dtype_;
    Shape shape_;
    Device device_;
    std::unique_ptr<uint8_t[]> data_;
};

// Type size lookup
inline size_t type_size(DataType dt) {
    switch (dt) {
        case DataType::F32: return 4;
        case DataType::F16: return 2;
        case DataType::BF16: return 2;
        case DataType::Q4_0: case DataType::Q4_1: case DataType::Q4_K: return 1;
        case DataType::Q5_0: case DataType::Q5_1: case DataType::Q5_K: return 1;
        case DataType::Q8_0: case DataType::Q8_1: case DataType::Q8_K: return 1;
        case DataType::Q2_K: case DataType::Q3_K: case DataType::Q6_K: return 1;
        case DataType::I8: return 1;
        case DataType::I16: return 2;
        case DataType::I32: return 4;
        case DataType::I64: return 8;
        case DataType::F64: return 8;
        default: return 1;
    }
}

// Bytes per quantization block (exact on-disk size for GGUF quant types)
inline size_t type_nbytes_per_block(DataType dt) {
    switch (dt) {
        case DataType::F32: return 4;     // per element
        case DataType::F16: return 2;     // per element
        case DataType::BF16: return 2;    // per element
        case DataType::Q4_0: return 18;   // 2 (f16 d) + 16
        case DataType::Q4_1: return 20;   // 2+2 + 16
        case DataType::Q5_0: return 22;   // 2 + 4(qh) + 16
        case DataType::Q5_1: return 24;   // 2+2 + 4 + 16
        case DataType::Q8_0: return 34;   // 2 + 32
        case DataType::Q8_1: return 36;   // 2+2 + 32
        case DataType::Q2_K: return 84;   // 16 scales + 64 qs + 2+2
        case DataType::Q3_K: return 110;  // 32 hm + 64 qs + 12 scales + 2
        case DataType::Q4_K: return 144;  // 2+2 + 12 scales + 128 qs
        case DataType::Q5_K: return 176;  // 2+2 + 12 + 32 qh + 128 qs
        case DataType::Q6_K: return 210;  // 128 ql + 64 qh + 16 scales + 2
        case DataType::Q8_K: return 292;  // 4 + 256 qs + 32 bsums
        default: return 1;
    }
}

// Exact byte size of `n` elements of dtype (handles quantized blocks)
inline int block_size(DataType dt);  // forward declaration

inline size_t type_nbytes(DataType dt, int64_t n) {
    switch (dt) {
        case DataType::F32: return static_cast<size_t>(n) * 4;
        case DataType::F16: return static_cast<size_t>(n) * 2;
        case DataType::BF16: return static_cast<size_t>(n) * 2;
        case DataType::F64: return static_cast<size_t>(n) * 8;
        case DataType::I8: return static_cast<size_t>(n);
        case DataType::I16: return static_cast<size_t>(n) * 2;
        case DataType::I32: return static_cast<size_t>(n) * 4;
        case DataType::I64: return static_cast<size_t>(n) * 8;
        default: {
            int bs = block_size(dt);
            return (static_cast<size_t>(n) / bs) * type_nbytes_per_block(dt);
        }
    }
}

// Quantized tensor block sizes
inline int block_size(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: return 32;
        case DataType::Q4_1: return 32;
        case DataType::Q5_0: return 32;
        case DataType::Q5_1: return 32;
        case DataType::Q8_0: return 32;
        case DataType::Q8_1: return 32;
        case DataType::Q2_K: return 256;
        case DataType::Q3_K: return 256;
        case DataType::Q4_K: return 256;
        case DataType::Q5_K: return 256;
        case DataType::Q6_K: return 256;
        case DataType::Q8_K: return 256;
        default: return 1;
    }
}

// Factory functions
Tensor zeros(DataType dtype, const Shape& shape, Device device = Device::cpu());
Tensor ones(DataType dtype, const Shape& shape, Device device = Device::cpu());
Tensor randn(DataType dtype, const Shape& shape, Device device = Device::cpu());
Tensor from_blob(DataType dtype, const Shape& shape, void* data, Device device = Device::cpu());

}  // namespace inference