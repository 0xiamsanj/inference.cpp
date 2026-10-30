#pragma once

#include "core/tensor.hpp"

namespace inference {

// Matrix multiplication: C = A @ B^T (or A @ B)
// Supports: [M, K] @ [N, K]^T -> [M, N]  (common in attention)
//           [M, K] @ [K, N] -> [M, N]     (common in FFN)
// Handles quantized weights (dequantize on-the-fly)
void gemm(const Tensor& A, const Tensor& B, Tensor& C, bool trans_b = false);
void gemv(const Tensor& A, const Tensor& x, Tensor& y);  // Matrix-vector: y = A @ x

// Batched GEMM for attention: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N]
void batched_gemm(const Tensor& A, const Tensor& B, Tensor& C);

// Element-wise operations
void add_(Tensor& a, const Tensor& b);           // a += b
void mul_(Tensor& a, const Tensor& b);           // a *= b
void silu_(Tensor& x);                           // x = x * sigmoid(x)
void gelu_(Tensor& x);                           // GELU activation
void relu_(Tensor& x);                           // ReLU
void softmax_(Tensor& x, int dim = -1);          // Softmax along dimension
void rope_(Tensor& x, int pos, int dim, float theta = 10000.0f);  // RoPE

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps = 1e-6f);
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out, float eps = 1e-5f);

// Attention
void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
               const Tensor* mask = nullptr, float scale = 1.0f);

// Sampling
int sample_top_k_top_p(const Tensor& logits, int top_k, float top_p, float temperature,
                       int seed = -1);
int sample_argmax(const Tensor& logits);

// Utility
Tensor contiguous(const Tensor& t);  // Ensure contiguous memory

}  // namespace inference