#pragma once

#include "core/tensor.hpp"
#include <cstdint>
#include <vector>

namespace inference {

// Quantization utilities for GGUF formats
// Based on llama.cpp quantization implementation

// Q4_0: 32 values/block, 18 bytes (fp16 scale + 16 packed 4-bit values)
struct BlockQ4_0 {
    uint16_t d;        // fp16 scale
    uint8_t qs[16];    // low nibble = element j, high nibble = element j+16
};
static_assert(sizeof(BlockQ4_0) == 18, "GGUF Q4_0 block must be 18 bytes");

// Q4_K: 256 values/super-block, 144 bytes
// Order on disk: d, dmin, 12 scale bytes, 128 quant bytes
struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "GGUF Q4_K block must be 144 bytes");

// Q8_0: 32 values/block, 34 bytes (fp16 scale + 32 int8 values)
struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "GGUF Q8_0 block must be 34 bytes");

// Dequantize functions - convert to FP32
void dequantize_q4_0(const void* src, float* dst, int64_t n);
void dequantize_q4_k(const void* src, float* dst, int64_t n);
void dequantize_q8_0(const void* src, float* dst, int64_t n);
void dequantize_q5_0(const void* src, float* dst, int64_t n);
void dequantize_q5_1(const void* src, float* dst, int64_t n);

// Quantize functions (for weight preparation if needed)
void quantize_q4_0(const float* src, void* dst, int64_t n);
void quantize_q8_0(const float* src, void* dst, int64_t n);

// Generic dispatch
void dequantize(DataType dtype, const void* src, float* dst, int64_t n);
void quantize(DataType dtype, const float* src, void* dst, int64_t n);

// K-quants dequantize (more complex, used by modern models)
void dequantize_q2_k(const void*, float*, int64_t);
void dequantize_q3_k(const void*, float*, int64_t);
void dequantize_q4_k(const void*, float*, int64_t);
void dequantize_q5_k(const void*, float*, int64_t);
void dequantize_q6_k(const void*, float*, int64_t);
void dequantize_q8_k(const void*, float*, int64_t);

// Block sizes
constexpr int QK_K = 256;  // Super-block size for K-quants

}  // namespace inference