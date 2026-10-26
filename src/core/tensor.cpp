#include "core/tensor.hpp"
#include <cstring>
#include <random>
#include <sstream>
#include <iomanip>

namespace inference {

void Tensor::allocate() {
    size_t bytes = byte_size();
    if (bytes > 0) {
        data_ = std::make_unique<uint8_t[]>(bytes);
    }
}

Tensor Tensor::to(Device target_device) const {
    if (device_ == target_device) return clone();

    Tensor result(dtype_, shape_, target_device);
    // TODO: Implement CPU<->CUDA copy when CUDA is added
    // For now, only CPU->CPU works
    if (device_.type == DeviceType::CPU && target_device.type == DeviceType::CPU) {
        std::memcpy(result.data(), data(), byte_size());
    } else {
        throw std::runtime_error("Cross-device copy not implemented yet");
    }
    return result;
}

void Tensor::fill(float value) {
    if (dtype_ == DataType::F32) {
        std::fill_n(as<float>(), numel(), value);
    } else if (dtype_ == DataType::F16) {
        // IEEE 754 half-precision conversion
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));

        uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
        int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = bits & 0x7FFFFF;

        uint16_t h;
        if (exponent > 15) {
            h = sign | 0x7C00;  // Inf
        } else if (exponent < -24) {
            h = sign;  // Zero
        } else if (exponent < -14) {
            // Subnormal
            uint32_t m = (mantissa | 0x800000) >> (-exponent - 14);
            h = sign | static_cast<uint16_t>(m >> 13);
        } else {
            h = sign | static_cast<uint16_t>((exponent + 15) << 10) |
                static_cast<uint16_t>(mantissa >> 13);
        }

        for (int64_t i = 0; i < numel(); ++i) {
            as<uint16_t>()[i] = h;
        }
    } else {
        throw std::runtime_error("fill() not implemented for this dtype");
    }
}

std::string Tensor::debug_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=";
    switch (dtype_) {
        case DataType::F32: oss << "F32"; break;
        case DataType::F16: oss << "F16"; break;
        case DataType::Q4_0: oss << "Q4_0"; break;
        case DataType::Q4_K: oss << "Q4_K"; break;
        case DataType::Q8_0: oss << "Q8_0"; break;
        default: oss << static_cast<int>(dtype_);
    }
    oss << ", device=" << (device_.type == DeviceType::CPU ? "CPU" : "CUDA") << ")";
    return oss.str();
}

Tensor zeros(DataType dtype, const Shape& shape, Device device) {
    Tensor t(dtype, shape, device);
    std::memset(t.data(), 0, t.byte_size());
    return t;
}

Tensor ones(DataType dtype, const Shape& shape, Device device) {
    Tensor t(dtype, shape, device);
    if (dtype == DataType::F32) {
        std::fill_n(t.as<float>(), t.numel(), 1.0f);
    } else {
        t.fill(1.0f);
    }
    return t;
}

Tensor randn(DataType dtype, const Shape& shape, Device device) {
    Tensor t(dtype, shape, device);
    if (dtype == DataType::F32) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int64_t i = 0; i < t.numel(); ++i) {
            t.as<float>()[i] = dist(gen);
        }
    } else {
        throw std::runtime_error("randn() only implemented for F32");
    }
    return t;
}

Tensor from_blob(DataType dtype, const Shape& shape, void* data, Device device) {
    Tensor t(dtype, shape, device);
    std::memcpy(t.data(), data, t.byte_size());
    return t;
}

}  // namespace inference