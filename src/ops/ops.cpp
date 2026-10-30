#include "ops/ops.hpp"
#include "gguf/quantization.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace inference {

// ============================================================================
// GEMM: General Matrix Multiply
// A: [M, K], B: [N, K] (trans_b=true) or [K, N] (trans_b=false)
// C: [M, N]
// ============================================================================
void gemm(const Tensor& A, const Tensor& B, Tensor& C, bool trans_b) {
    // Validate shapes
    int64_t M = A.shape()[0];
    int64_t K_A = A.shape()[1];

    int64_t K_B, N;
    if (trans_b) {
        // B is [N, K]
        N = B.shape()[0];
        K_B = B.shape()[1];
    } else {
        // B is [K, N]
        K_B = B.shape()[0];
        N = B.shape()[1];
    }

    if (K_A != K_B) {
        throw std::runtime_error("GEMM: K dimension mismatch");
    }

    // Output shape
    Shape out_shape = {M, N};
    if (C.shape() != out_shape) {
        C.reshape(out_shape);
    }

    // For now, implement FP32 only. Quantized weights need dequantization first.
    if (A.dtype() != DataType::F32 || C.dtype() != DataType::F32) {
        throw std::runtime_error("GEMM: Only F32 supported for now");
    }

    const float* a_ptr = A.as<float>();
    float* c_ptr = C.as<float>();

    if (B.dtype() == DataType::F32) {
        const float* b_ptr = B.as<float>();

        // Simple GEMM with basic loop ordering (can optimize with blocking/SIMD later)
        if (trans_b) {
            // C = A @ B^T
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K_A; ++k) {
                        sum += a_ptr[m * K_A + k] * b_ptr[n * K_B + k];
                    }
                    c_ptr[m * N + n] = sum;
                }
            }
        } else {
            // C = A @ B
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K_A; ++k) {
                        sum += a_ptr[m * K_A + k] * b_ptr[k * N + n];
                    }
                    c_ptr[m * N + n] = sum;
                }
            }
        }
    } else if (B.dtype() == DataType::Q4_0 || B.dtype() == DataType::Q8_0) {
        // Dequantize B on-the-fly during GEMM (memory efficient)
        // For simplicity, dequantize entire B first
        Tensor B_f32(DataType::F32, trans_b ? Shape{N, K_B} : Shape{K_B, N});
        dequantize(B.dtype(), B.data(), B_f32.as<float>(), B_f32.numel());
        gemm(A, B_f32, C, trans_b);
    } else {
        throw std::runtime_error("GEMM: Unsupported B dtype");
    }
}

// Matrix-vector: y = A @ x
// A: [M, K], x: [K], y: [M]
void gemv(const Tensor& A, const Tensor& x, Tensor& y) {
    if (A.shape().size() != 2) throw std::runtime_error("gemv: A must be 2D");
    if (x.shape().size() != 1) throw std::runtime_error("gemv: x must be 1D");

    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];

    if (x.shape()[0] != K) throw std::runtime_error("gemv: dimension mismatch");
    if (y.shape() != Shape{M}) y.reshape({M});

    const float* a = A.as<float>();
    const float* x_ptr = x.as<float>();
    float* y_ptr = y.as<float>();

    for (int64_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += a[m * K + k] * x_ptr[k];
        }
        y_ptr[m] = sum;
    }
}

// ============================================================================
// Element-wise operations
// ============================================================================
void add_(Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("add_: shape mismatch");
    if (a.dtype() != DataType::F32 || b.dtype() != DataType::F32) {
        throw std::runtime_error("add_: only F32 supported");
    }

    float* a_ptr = a.as<float>();
    const float* b_ptr = b.as<float>();
    int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        a_ptr[i] += b_ptr[i];
    }
}

void mul_(Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("mul_: shape mismatch");
    if (a.dtype() != DataType::F32 || b.dtype() != DataType::F32) {
        throw std::runtime_error("mul_: only F32 supported");
    }

    float* a_ptr = a.as<float>();
    const float* b_ptr = b.as<float>();
    int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        a_ptr[i] *= b_ptr[i];
    }
}

void silu_(Tensor& x) {
    if (x.dtype() != DataType::F32) throw std::runtime_error("silu_: only F32");
    float* ptr = x.as<float>();
    int64_t n = x.numel();

    for (int64_t i = 0; i < n; ++i) {
        float v = ptr[i];
        float sig = 1.0f / (1.0f + std::exp(-v));
        ptr[i] = v * sig;
    }
}

void gelu_(Tensor& x) {
    if (x.dtype() != DataType::F32) throw std::runtime_error("gelu_: only F32");
    float* ptr = x.as<float>();
    int64_t n = x.numel();

    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        float v = ptr[i];
        float cube = v * v * v;
        float inner = sqrt_2_over_pi * (v + coef * cube);
        float tanh_v = std::tanh(inner);
        ptr[i] = 0.5f * v * (1.0f + tanh_v);
    }
}

void relu_(Tensor& x) {
    if (x.dtype() != DataType::F32) throw std::runtime_error("relu_: only F32");
    float* ptr = x.as<float>();
    int64_t n = x.numel();

    for (int64_t i = 0; i < n; ++i) {
        ptr[i] = std::max(0.0f, ptr[i]);
    }
}

void softmax_(Tensor& x, int dim) {
    if (x.dtype() != DataType::F32) throw std::runtime_error("softmax_: only F32");

    Shape shape = x.shape();
    if (dim < 0) dim += shape.size();
    if (dim >= (int)shape.size()) throw std::runtime_error("softmax_: invalid dim");

    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= shape[i];

    int64_t inner = 1;
    for (int i = dim + 1; i < (int)shape.size(); ++i) inner *= shape[i];

    int64_t dim_size = shape[dim];

    float* ptr = x.as<float>();

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            // Find max for numerical stability
            float max_val = -INFINITY;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + i;
                max_val = std::max(max_val, ptr[idx]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + i;
                float v = std::exp(ptr[idx] - max_val);
                ptr[idx] = v;
                sum += v;
            }

            // Normalize
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + i;
                ptr[idx] /= sum;
            }
        }
    }
}

// RoPE (Rotary Positional Embedding)
// x: [*, seq_len, head_dim] - applies to last two dims
void rope_(Tensor& x, int pos, int /*dim*/, float theta) {
    if (x.dtype() != DataType::F32) throw std::runtime_error("rope_: only F32");
    if (x.shape().size() < 2) throw std::runtime_error("rope_: need at least 2D");

    Shape shape = x.shape();
    int64_t seq_len = shape[shape.size() - 2];
    int64_t head_dim = shape[shape.size() - 1];

    if (head_dim % 2 != 0) throw std::runtime_error("rope_: head_dim must be even");

    float* ptr = x.as<float>();
    int64_t outer = x.numel() / (seq_len * head_dim);

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t s = 0; s < seq_len; ++s) {
            int position = pos + s;
            for (int64_t d = 0; d < head_dim; d += 2) {
                float freq = 1.0f / std::pow(theta, static_cast<float>(d) / head_dim);
                float angle = position * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int64_t idx1 = ((o * seq_len + s) * head_dim + d);
                int64_t idx2 = idx1 + 1;

                float x1 = ptr[idx1];
                float x2 = ptr[idx2];

                ptr[idx1] = x1 * cos_a - x2 * sin_a;
                ptr[idx2] = x1 * sin_a + x2 * cos_a;
            }
        }
    }
}



// ============================================================================
// Attention: out = softmax(Q @ K^T / sqrt(d)) @ V
// Q: [B, H, M, D], K: [B, H, N, D], V: [B, H, N, D]
// out: [B, H, M, D]
// ============================================================================
void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
               const Tensor* mask, float scale) {
    // Simplified: assume 4D tensors [batch, heads, seq, head_dim]
    if (q.shape().size() != 4 || k.shape().size() != 4 || v.shape().size() != 4) {
        throw std::runtime_error("attention: expected 4D tensors");
    }

    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t M = q.shape()[2];  // query seq len
    int64_t D = q.shape()[3];  // head dim
    int64_t N = k.shape()[2];  // key/value seq len

    if (k.shape() != Shape{B, H, N, D} || v.shape() != Shape{B, H, N, D}) {
        throw std::runtime_error("attention: shape mismatch");
    }

    Shape out_shape = {B, H, M, D};
    if (out.shape() != out_shape) out.reshape(out_shape);

    // For each batch and head, compute attention
    Tensor scores(DataType::F32, {M, N});

    const float* q_ptr = q.as<float>();
    const float* k_ptr = k.as<float>();
    const float* v_ptr = v.as<float>();
    float* out_ptr = out.as<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* q_bh = q_ptr + ((b * H + h) * M * D);
            const float* k_bh = k_ptr + ((b * H + h) * N * D);
            const float* v_bh = v_ptr + ((b * H + h) * N * D);
            float* out_bh = out_ptr + ((b * H + h) * M * D);

            // scores = Q @ K^T * scale
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        sum += q_bh[m * D + d] * k_bh[n * D + d];
                    }
                    scores.as<float>()[m * N + n] = sum * scale;
                }
            }

            // Apply mask if provided
            if (mask) {
                const float* mask_ptr = mask->as<float>();
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t n = 0; n < N; ++n) {
                        scores.as<float>()[m * N + n] += mask_ptr[m * N + n];
                    }
                }
            }

            // Softmax
            softmax_(scores, -1);

            // out = scores @ V
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int64_t n = 0; n < N; ++n) {
                        sum += scores.as<float>()[m * N + n] * v_bh[n * D + d];
                    }
                    out_bh[m * D + d] = sum;
                }
            }
        }
    }
}

// ============================================================================
// Sampling
// ============================================================================
int sample_top_k_top_p(const Tensor& logits, int top_k, float top_p, float temperature,
                       int seed) {
    if (logits.dtype() != DataType::F32) throw std::runtime_error("sample: only F32");
    if (logits.shape().size() != 1) throw std::runtime_error("sample: logits must be 1D");

    // Greedy when temperature is zero (avoids dividing by zero).
    if (temperature <= 0.0f) return sample_argmax(logits);

    int vocab_size = logits.shape()[0];
    const float* logits_ptr = logits.as<float>();

    // Create index array
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    // Apply temperature
    std::vector<float> probs(vocab_size);
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits_ptr[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits_ptr[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum_exp;
    }

    // Top-k: keep only top_k tokens
    if (top_k > 0 && top_k < vocab_size) {
        std::nth_element(indices.begin(), indices.begin() + top_k, indices.end(),
                        [&](int a, int b) { return probs[a] > probs[b]; });
        indices.resize(top_k);
    }

    // Top-p (nucleus): keep smallest set with cumulative prob >= top_p
    if (top_p < 1.0f && top_p > 0.0f) {
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        size_t cutoff = indices.size();
        for (size_t i = 0; i < indices.size(); ++i) {
            cumsum += probs[indices[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        indices.resize(cutoff);
    }

    // Renormalize
    float sum = 0.0f;
    for (int idx : indices) sum += probs[idx];
    for (int idx : indices) probs[idx] /= sum;

    // Sample
    std::mt19937 gen(seed >= 0 ? static_cast<uint32_t>(seed)
                               : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen);

    float cumsum = 0.0f;
    for (int idx : indices) {
        cumsum += probs[idx];
        if (r < cumsum) return idx;
    }

    return indices.back();  // Fallback
}

int sample_argmax(const Tensor& logits) {
    if (logits.dtype() != DataType::F32) throw std::runtime_error("argmax: only F32");
    const float* ptr = logits.as<float>();
    int n = logits.numel();

    int best = 0;
    float best_val = ptr[0];
    for (int i = 1; i < n; ++i) {
        if (ptr[i] > best_val) {
            best_val = ptr[i];
            best = i;
        }
    }
    return best;
}

Tensor contiguous(const Tensor& t) {
    if (t.shape().size() <= 1) return t.clone();
    // For now just clone - proper implementation would check strides
    return t.clone();
}

}  // namespace inference