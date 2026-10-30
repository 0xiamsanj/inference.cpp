#pragma once

#include "core/tensor.hpp"

namespace inference {

// Multi-head attention with optional KV cache
void multi_head_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& out,
    const Tensor* mask = nullptr,
    float scale = 1.0f
);

// Grouped Query Attention (GQA)
void grouped_query_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& out,
    int num_kv_groups,
    const Tensor* mask = nullptr,
    float scale = 1.0f
);

// Flash Attention (simplified - no tiling)
void flash_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& out,
    float scale = 1.0f
);

}  // namespace inference