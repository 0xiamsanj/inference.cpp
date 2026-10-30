#pragma once

#include "core/tensor.hpp"

namespace inference {

// RMSNorm: x * weight / rms(x)
void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps = 1e-6f);

// LayerNorm: (x - mean) / std * weight + bias
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out, float eps = 1e-5f);

// GroupNorm (for vision models)
void group_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out, int num_groups, float eps = 1e-5f);

}  // namespace inference