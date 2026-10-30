#include "ops/normalization.hpp"
#include <cmath>

namespace inference {

void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
    if (x.dtype() != DataType::F32 || weight.dtype() != DataType::F32) {
        throw std::runtime_error("rms_norm: only F32 supported");
    }

    Shape shape = x.shape();
    int64_t last_dim = shape.back();
    int64_t outer = x.numel() / last_dim;

    if (weight.shape() != Shape{last_dim}) {
        throw std::runtime_error("rms_norm: weight shape mismatch");
    }

    if (out.shape() != shape) out.reshape(shape);

    const float* x_ptr = x.as<float>();
    const float* w_ptr = weight.as<float>();
    float* out_ptr = out.as<float>();

    for (int64_t o = 0; o < outer; ++o) {
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < last_dim; ++i) {
            float v = x_ptr[o * last_dim + i];
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / last_dim + eps);
        float scale = 1.0f / rms;

        for (int64_t i = 0; i < last_dim; ++i) {
            out_ptr[o * last_dim + i] = x_ptr[o * last_dim + i] * scale * w_ptr[i];
        }
    }
}

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out, float eps) {
    if (x.dtype() != DataType::F32 || weight.dtype() != DataType::F32 || bias.dtype() != DataType::F32) {
        throw std::runtime_error("layer_norm: only F32 supported");
    }

    Shape shape = x.shape();
    int64_t last_dim = shape.back();
    int64_t outer = x.numel() / last_dim;

    if (weight.shape() != Shape{last_dim} || bias.shape() != Shape{last_dim}) {
        throw std::runtime_error("layer_norm: weight/bias shape mismatch");
    }

    if (out.shape() != shape) out.reshape(shape);

    const float* x_ptr = x.as<float>();
    const float* w_ptr = weight.as<float>();
    const float* b_ptr = bias.as<float>();
    float* out_ptr = out.as<float>();

    for (int64_t o = 0; o < outer; ++o) {
        float mean = 0.0f;
        for (int64_t i = 0; i < last_dim; ++i) {
            mean += x_ptr[o * last_dim + i];
        }
        mean /= last_dim;

        float var = 0.0f;
        for (int64_t i = 0; i < last_dim; ++i) {
            float diff = x_ptr[o * last_dim + i] - mean;
            var += diff * diff;
        }
        var /= last_dim;
        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int64_t i = 0; i < last_dim; ++i) {
            float normalized = (x_ptr[o * last_dim + i] - mean) * inv_std;
            out_ptr[o * last_dim + i] = normalized * w_ptr[i] + b_ptr[i];
        }
    }
}

void group_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out, int num_groups, float eps) {
    if (x.dtype() != DataType::F32) throw std::runtime_error("group_norm: only F32");

    Shape shape = x.shape();
    if (shape.size() != 4) throw std::runtime_error("group_norm: expected 4D [N,C,H,W]");

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    if (C % num_groups != 0) throw std::runtime_error("group_norm: C must be divisible by num_groups");

    if (out.shape() != shape) out.reshape(shape);

    const float* x_ptr = x.as<float>();
    const float* w_ptr = weight.as<float>();
    const float* b_ptr = bias.as<float>();
    float* out_ptr = out.as<float>();

    int64_t channels_per_group = C / num_groups;
    int64_t group_size = channels_per_group * H * W;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            // Compute mean/var for this group
            float mean = 0.0f;
            for (int64_t c = 0; c < channels_per_group; ++c) {
                int64_t ch = g * channels_per_group + c;
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t idx = ((n * C + ch) * H + h) * W + w;
                        mean += x_ptr[idx];
                    }
                }
            }
            mean /= group_size;

            float var = 0.0f;
            for (int64_t c = 0; c < channels_per_group; ++c) {
                int64_t ch = g * channels_per_group + c;
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t idx = ((n * C + ch) * H + h) * W + w;
                        float diff = x_ptr[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= group_size;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize
            for (int64_t c = 0; c < channels_per_group; ++c) {
                int64_t ch = g * channels_per_group + c;
                float w_scale = w_ptr[ch];
                float b_shift = b_ptr[ch];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t idx = ((n * C + ch) * H + h) * W + w;
                        out_ptr[idx] = (x_ptr[idx] - mean) * inv_std * w_scale + b_shift;
                    }
                }
            }
        }
    }
}

}  // namespace inference