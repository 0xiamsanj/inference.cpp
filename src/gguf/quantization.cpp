#include "gguf/quantization.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace inference {

// IEEE 754 half-precision -> float (GGUF stores scales as fp16)
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        // Zero or subnormal. Subnormal halfs are common in quant scales.
        float v = std::ldexp(static_cast<float>(mant), -24);
        return (h & 0x8000u) ? -v : v;
    }
    uint32_t f;
    if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// ============================================================================
// Q4_0: 32 values/block, 18 bytes = fp16 scale + 16 packed bytes
// Packing: low nibble of qs[j] is element j; high nibble is element j+16.
// ============================================================================
void dequantize_q4_0(const void* src, float* dst, int64_t n) {
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t n_blocks = (n + 31) / 32;
    const size_t block_bytes = 18;

    for (int64_t i = 0; i < n_blocks; ++i) {
        const uint8_t* blk = base + i * block_bytes;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float d = fp16_to_fp32(d_bits);
        const uint8_t* qs = blk + 2;

        float* y = dst + i * 32;
        int64_t remaining = n - i * 32;
        for (int j = 0; j < 16; ++j) {
            int x0 = (qs[j] & 0x0F) - 8;
            int x1 = (qs[j] >> 4) - 8;
            if (j < remaining)     y[j] = x0 * d;
            if (j + 16 < remaining) y[j + 16] = x1 * d;
        }
    }
}

// ============================================================================
// Q8_0: 32 values/block, 34 bytes = fp16 scale + 32 int8 values
// ============================================================================
void dequantize_q8_0(const void* src, float* dst, int64_t n) {
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t n_blocks = (n + 31) / 32;
    const size_t block_bytes = 34;

    for (int64_t i = 0; i < n_blocks; ++i) {
        const uint8_t* blk = base + i * block_bytes;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float d = fp16_to_fp32(d_bits);
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);

        float* y = dst + i * 32;
        int64_t remaining = n - i * 32;
        for (int j = 0; j < 32 && j < remaining; ++j) {
            y[j] = qs[j] * d;
        }
    }
}

// ============================================================================
// Q5_0: 32 values/block, 22 bytes = fp16 d + 4 bytes qh + 16 bytes qs
// Each value: 4 low bits from qs, 1 high bit from qh, bias 16.
// ============================================================================
void dequantize_q5_0(const void* src, float* dst, int64_t n) {
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t n_blocks = (n + 31) / 32;
    const size_t block_bytes = 22;

    for (int64_t i = 0; i < n_blocks; ++i) {
        const uint8_t* blk = base + i * block_bytes;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float d = fp16_to_fp32(d_bits);
        uint32_t qh;
        std::memcpy(&qh, blk + 2, 4);
        const uint8_t* qs = blk + 6;

        float* y = dst + i * 32;
        int64_t remaining = n - i * 32;
        for (int j = 0; j < 16; ++j) {
            int x0 = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
            int x1 = (qs[j] >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            if (j < remaining)          y[j] = (x0 - 16) * d;
            if (j + 16 < remaining)     y[j + 16] = (x1 - 16) * d;
        }
    }
}

// ============================================================================
// Q5_1: 32 values/block, 24 bytes = fp16 d + fp16 m + 4 qh + 16 qs
// value = q*d + m, q in [0,31]
// ============================================================================
void dequantize_q5_1(const void* src, float* dst, int64_t n) {
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t n_blocks = (n + 31) / 32;
    const size_t block_bytes = 24;

    for (int64_t i = 0; i < n_blocks; ++i) {
        const uint8_t* blk = base + i * block_bytes;
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&m_bits, blk + 2, 2);
        const float d = fp16_to_fp32(d_bits);
        const float m = fp16_to_fp32(m_bits);
        uint32_t qh;
        std::memcpy(&qh, blk + 4, 4);
        const uint8_t* qs = blk + 8;

        float* y = dst + i * 32;
        int64_t remaining = n - i * 32;
        for (int j = 0; j < 16; ++j) {
            int x0 = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
            int x1 = (qs[j] >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            if (j < remaining)      y[j] = x0 * d + m;
            if (j + 16 < remaining) y[j + 16] = x1 * d + m;
        }
    }
}

// ============================================================================
// K-quants (Q4_K, Q5_K, Q6_K, Q8_K) - more complex
// These use 256-element super-blocks with sub-blocks
// ============================================================================

// Q4_K: 256 values per super-block.
// Layout (llama.cpp ggml-common.h, K_SCALE_SIZE=12):
//   ggml_half d;      // 2 bytes: super-block scale for scales
//   ggml_half dmin;   // 2 bytes: super-block scale for mins
//   uint8_t scales[12];
//   uint8_t qs[128];  // 4-bit quants
// Total 144 bytes.
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void dequantize_q4_k(const void* src, float* dst, int64_t n) {
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t n_superblocks = n / QK_K;
    const size_t block_bytes = 144;

    for (int64_t sb = 0; sb < n_superblocks; ++sb) {
        const uint8_t* blk = base + sb * block_bytes;

        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, blk + 0, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        const float d    = fp16_to_fp32(d_bits);
        const float dmin = fp16_to_fp32(dmin_bits);

        const uint8_t* scales = blk + 4;    // 12 bytes
        const uint8_t* q = blk + 16;        // 128 bytes

        float* y = dst + sb * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;

            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;

            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;

            q += 32;
            is += 2;
        }
    }
}

// Placeholder for other K-quants
void dequantize_q2_k(const void*, float*, int64_t) { throw std::runtime_error("Not implemented"); }
void dequantize_q3_k(const void*, float*, int64_t) { throw std::runtime_error("Not implemented"); }
void dequantize_q5_k(const void*, float*, int64_t) { throw std::runtime_error("Not implemented"); }
void dequantize_q8_k(const void*, float*, int64_t) { throw std::runtime_error("Not implemented"); }

// Q6_K: 256 values per super-block (210 bytes)
//   uint8_t ql[128];   // lower 4 bits
//   uint8_t qh[64];    // upper 2 bits
//   int8_t  scales[16];
//   ggml_half d;       // 2 bytes
void dequantize_q6_k(const void* src, float* dst, int64_t n) {
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t n_superblocks = n / QK_K;
    const size_t block_bytes = 210;

    for (int64_t sb = 0; sb < n_superblocks; ++sb) {
        const uint8_t* blk = base + sb * block_bytes;
        const uint8_t* ql = blk + 0;      // 128
        const uint8_t* qh = blk + 128;    // 64
        const int8_t*  sc = reinterpret_cast<const int8_t*>(blk + 192);  // 16
        uint16_t d_bits;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = fp16_to_fp32(d_bits);

        float* y = dst + sb * QK_K;
        for (int chunk = 0; chunk < 2; ++chunk) {  // 2 x 128 values
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF)  | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ============================================================================
// Generic dispatch
// ============================================================================
void dequantize(DataType dtype, const void* src, float* dst, int64_t n) {
    switch (dtype) {
        case DataType::Q4_0: dequantize_q4_0(src, dst, n); break;
        case DataType::Q4_1: throw std::runtime_error("Q4_1 not implemented");
        case DataType::Q5_0: dequantize_q5_0(src, dst, n); break;
        case DataType::Q5_1: dequantize_q5_1(src, dst, n); break;
        case DataType::Q8_0: dequantize_q8_0(src, dst, n); break;
        case DataType::Q8_1: throw std::runtime_error("Q8_1 not implemented");
        case DataType::Q2_K: dequantize_q2_k(src, dst, n); break;
        case DataType::Q3_K: dequantize_q3_k(src, dst, n); break;
        case DataType::Q4_K: dequantize_q4_k(src, dst, n); break;
        case DataType::Q5_K: dequantize_q5_k(src, dst, n); break;
        case DataType::Q6_K: dequantize_q6_k(src, dst, n); break;
        case DataType::Q8_K: dequantize_q8_k(src, dst, n); break;
        default:
            throw std::runtime_error("Unsupported quantization type for dequantize");
    }
}

// Quantize (produces GGUF-compatible blocks)
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void quantize_q4_0(const float* src, void* dst, int64_t n) {
    uint8_t* base = static_cast<uint8_t*>(dst);
    int64_t n_blocks = (n + 31) / 32;
    const size_t block_bytes = 18;

    for (int64_t i = 0; i < n_blocks; ++i) {
        uint8_t* blk = base + i * block_bytes;

        float max_abs = 0;
        for (int j = 0; j < 32; ++j) {
            int idx = i * 32 + j;
            if (idx < n) max_abs = std::max(max_abs, std::abs(src[idx]));
        }

        const float d = max_abs / 7.0f;
        const float inv = (d != 0) ? 1.0f / d : 0;
        uint16_t d_bits = fp32_to_fp16(d);
        std::memcpy(blk, &d_bits, 2);

        uint8_t* qs = blk + 2;
        for (int j = 0; j < 16; ++j) {
            int lo = 0, hi = 0;
            int i0 = i * 32 + j;
            int i1 = i * 32 + j + 16;
            if (i0 < n) {
                lo = static_cast<int>(std::round(src[i0] * inv));
                lo = std::clamp(lo, -8, 7);
            }
            if (i1 < n) {
                hi = static_cast<int>(std::round(src[i1] * inv));
                hi = std::clamp(hi, -8, 7);
            }
            qs[j] = static_cast<uint8_t>(((lo + 8) & 0x0F) | (((hi + 8) & 0x0F) << 4));
        }
    }
}

void quantize_q8_0(const float* src, void* dst, int64_t n) {
    uint8_t* base = static_cast<uint8_t*>(dst);
    int64_t n_blocks = (n + 31) / 32;
    const size_t block_bytes = 34;

    for (int64_t i = 0; i < n_blocks; ++i) {
        uint8_t* blk = base + i * block_bytes;

        float max_abs = 0;
        for (int j = 0; j < 32; ++j) {
            int idx = i * 32 + j;
            if (idx < n) max_abs = std::max(max_abs, std::abs(src[idx]));
        }

        const float d = max_abs / 127.0f;
        const float inv = (d != 0) ? 1.0f / d : 0;
        uint16_t d_bits = fp32_to_fp16(d);
        std::memcpy(blk, &d_bits, 2);

        int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
        for (int j = 0; j < 32; ++j) {
            int idx = i * 32 + j;
            if (idx < n) {
                int q = static_cast<int>(std::round(src[idx] * inv));
                qs[j] = static_cast<int8_t>(std::clamp(q, -128, 127));
            } else {
                qs[j] = 0;
            }
        }
    }
}

void quantize(DataType dtype, const float* src, void* dst, int64_t n) {
    switch (dtype) {
        case DataType::Q4_0: quantize_q4_0(src, dst, n); break;
        case DataType::Q8_0: quantize_q8_0(src, dst, n); break;
        default: throw std::runtime_error("Quantize not implemented for this type");
    }
}

}  // namespace inference