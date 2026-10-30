#include "ops/attention.hpp"
#include "ops/ops.hpp"
#include <cmath>
#include <algorithm>

namespace inference {

void multi_head_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& out,
    const Tensor* mask,
    float scale
) {
    // Delegate to generic attention
    attention(q, k, v, out, mask, scale);
}

void grouped_query_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& out,
    int num_kv_groups,
    const Tensor* mask,
    float scale
) {
    // q: [B, H_q, T, D], k: [B, H_kv, T, D], v: [B, H_kv, T, D]
    // H_q = H_kv * num_kv_groups
    (void)mask;  // Causal masking applied in loop

    if (q.shape().size() != 4 || k.shape().size() != 4 || v.shape().size() != 4) {
        throw std::runtime_error("GQA: expected 4D tensors");
    }

    int64_t B = q.shape()[0];
    int64_t H_q = q.shape()[1];
    int64_t T = q.shape()[2];
    int64_t D = q.shape()[3];
    int64_t H_kv = k.shape()[1];

    if (H_q != H_kv * num_kv_groups) {
        throw std::runtime_error("GQA: head count mismatch");
    }

    if (out.shape() != Shape{B, H_q, T, D}) out.reshape({B, H_q, T, D});

    // For each query head group, use corresponding KV head
    const float* q_ptr = q.as<float>();
    const float* k_ptr = k.as<float>();
    const float* v_ptr = v.as<float>();
    float* out_ptr = out.as<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h_q = 0; h_q < H_q; ++h_q) {
            int64_t h_kv = h_q / num_kv_groups;

            const float* q_bh = q_ptr + ((b * H_q + h_q) * T * D);
            const float* k_bh = k_ptr + ((b * H_kv + h_kv) * T * D);
            const float* v_bh = v_ptr + ((b * H_kv + h_kv) * T * D);
            float* out_bh = out_ptr + ((b * H_q + h_q) * T * D);

            // Compute attention for this head
            for (int64_t t_q = 0; t_q < T; ++t_q) {
                // Scores
                float max_score = -INFINITY;
                for (int64_t t_k = 0; t_k <= t_q; ++t_k) {  // Causal
                    float score = 0;
                    for (int64_t d = 0; d < D; ++d) {
                        score += q_bh[t_q * D + d] * k_bh[t_k * D + d];
                    }
                    score *= scale;
                    max_score = std::max(max_score, score);
                }

                // Softmax
                float sum_exp = 0;
                float scores[2048];  // Max seq len
                for (int64_t t_k = 0; t_k <= t_q; ++t_k) {
                    float score = 0;
                    for (int64_t d = 0; d < D; ++d) {
                        score += q_bh[t_q * D + d] * k_bh[t_k * D + d];
                    }
                    score = std::exp(score * scale - max_score);
                    scores[t_k] = score;
                    sum_exp += score;
                }
                for (int64_t t_k = 0; t_k <= t_q; ++t_k) {
                    scores[t_k] /= sum_exp;
                }

                // Weighted sum
                for (int64_t d = 0; d < D; ++d) {
                    float sum = 0;
                    for (int64_t t_k = 0; t_k <= t_q; ++t_k) {
                        sum += scores[t_k] * v_bh[t_k * D + d];
                    }
                    out_bh[t_q * D + d] = sum;
                }
            }
        }
    }
}

void flash_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& out,
    float scale
) {
    // Simplified flash attention - no tiling, just memory-efficient
    // Real flash attention uses tiling to fit in SRAM
    multi_head_attention(q, k, v, out, nullptr, scale);
}

}  // namespace inference