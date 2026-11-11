#include "models/model.hpp"
#include "ops/ops.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdlib>

namespace inference {

Model::Model(ModelWeights weights, Device device)
    : weights_(std::move(weights)), device_(device) {
    // Weights are moved in; no copy of the (large) tensors is made.
}

void Model::apply_rope(Tensor& q, Tensor& k, int pos) const {
    // q, k: [B, H, T, D]
    // RoPE applies to last dimension (head_dim)
    rope_(q, pos, -1, weights_.rope_theta);
    rope_(k, pos, -1, weights_.rope_theta);
}

void Model::attention_forward(int layer_idx, const Tensor& x, Tensor& out,
                              KVCache& kv_cache, int start_pos) {
    const auto& layer = weights_.layers[layer_idx];
    int B = x.shape()[0];
    int T = x.shape()[1];
    int D = weights_.embed_dim;
    int H = weights_.n_heads;
    int H_kv = weights_.n_kv_heads;
    int head_dim = weights_.head_dim;

    // Ensure output shape
    if (out.shape() != Shape{B, T, D}) out.reshape({B, T, D});

    // NOTE: educational, unoptimized implementation that processes each
    // token with explicit loops so every step is readable. See README for
    // the optimization roadmap (blocked GEMM, SIMD, CUDA).

    float* out_ptr = out.as<float>();
    const float* x_ptr = x.as<float>();
    const float* attn_norm_ptr = layer.attn_norm.data.as<float>();

    // For each batch and token
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int pos = start_pos + t;

            // 1. RMSNorm
            float x_local[4096];  // Max embed_dim
            for (int d = 0; d < D; ++d) x_local[d] = x_ptr[(b * T + t) * D + d];

            // RMSNorm
            float sum_sq = 0;
            for (int d = 0; d < D; ++d) sum_sq += x_local[d] * x_local[d];
            float rms = std::sqrt(sum_sq / D + weights_.rms_eps);
            for (int d = 0; d < D; ++d) x_local[d] = x_local[d] / rms * attn_norm_ptr[d];

            // 2. Q, K, V projections
            float q_local[4096];  // H * head_dim
            float k_local[4096];  // H_kv * head_dim
            float v_local[4096];  // H_kv * head_dim

            matvec(layer.wq, x_local, q_local, H * head_dim, D);
            matvec(layer.wk, x_local, k_local, H_kv * head_dim, D);
            matvec(layer.wv, x_local, v_local, H_kv * head_dim, D);

            // 3. RoPE on Q and K
            for (int h = 0; h < H; ++h) {
                for (int d = 0; d < head_dim; d += 2) {
                    float freq = 1.0f / std::pow(weights_.rope_theta, float(d) / head_dim);
                    float angle = pos * freq;
                    float cos_a = std::cos(angle);
                    float sin_a = std::sin(angle);

                    int idx1 = h * head_dim + d;
                    int idx2 = idx1 + 1;
                    float q1 = q_local[idx1];
                    float q2 = q_local[idx2];
                    q_local[idx1] = q1 * cos_a - q2 * sin_a;
                    q_local[idx2] = q1 * sin_a + q2 * cos_a;
                }
            }

            for (int h = 0; h < H_kv; ++h) {
                for (int d = 0; d < head_dim; d += 2) {
                    float freq = 1.0f / std::pow(weights_.rope_theta, float(d) / head_dim);
                    float angle = pos * freq;
                    float cos_a = std::cos(angle);
                    float sin_a = std::sin(angle);

                    int idx1 = h * head_dim + d;
                    int idx2 = idx1 + 1;
                    float k1 = k_local[idx1];
                    float k2 = k_local[idx2];
                    k_local[idx1] = k1 * cos_a - k2 * sin_a;
                    k_local[idx2] = k1 * sin_a + k2 * cos_a;
                }
            }

            // 4. Store K, V in cache at absolute position `pos`
            // Cache: [1, H_kv, max_seq, head_dim]
            float* cache_k = kv_cache.key(layer_idx).as<float>();
            float* cache_v = kv_cache.value(layer_idx).as<float>();
            const int max_cache = static_cast<int>(kv_cache.key(layer_idx).shape()[2]);
            if (pos >= max_cache) {
                throw std::runtime_error("Model: sequence longer than KV cache");
            }

            for (int h = 0; h < H_kv; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    int cache_idx = h * max_cache * head_dim + pos * head_dim + d;
                    cache_k[cache_idx] = k_local[h * head_dim + d];
                    cache_v[cache_idx] = v_local[h * head_dim + d];
                }
            }

            // 5. Causal attention over positions 0..pos inclusive
            // For GQA: several query heads share one KV head
            int heads_per_group = H / H_kv;

            float attn_out[4096];  // H * head_dim

            for (int h = 0; h < H; ++h) {
                int h_kv = h / heads_per_group;

                // Compute attention scores for this head
                float scores[4096];
                int seq_len = pos + 1;  // positions 0..pos

                float max_score = -INFINITY;
                for (int s = 0; s < seq_len; ++s) {
                    float score = 0;
                    for (int d = 0; d < head_dim; ++d) {
                        float q_val = q_local[h * head_dim + d];
                        int cache_idx = h_kv * max_cache * head_dim + s * head_dim + d;
                        score += q_val * cache_k[cache_idx];
                    }
                    score /= std::sqrt(static_cast<float>(head_dim));
                    scores[s] = score;
                    max_score = std::max(max_score, score);
                }

                // Softmax
                float sum_exp = 0;
                for (int s = 0; s < seq_len; ++s) {
                    scores[s] = std::exp(scores[s] - max_score);
                    sum_exp += scores[s];
                }
                for (int s = 0; s < seq_len; ++s) {
                    scores[s] /= sum_exp;
                }

                // Weighted sum of V (all from cache, including current pos)
                for (int d = 0; d < head_dim; ++d) {
                    float sum = 0;
                    for (int s = 0; s < seq_len; ++s) {
                        int cache_idx = h_kv * max_cache * head_dim + s * head_dim + d;
                        sum += scores[s] * cache_v[cache_idx];
                    }
                    attn_out[h * head_dim + d] = sum;
                }
            }

            // 6. Output projection: attn_out @ Wo^T
            // Wo: [D, H*head_dim]
            matvec(layer.wo, attn_out, out_ptr + (b * T + t) * D, D, H * head_dim);
        }
    }
}

void Model::ffn_forward(int layer_idx, const Tensor& x, Tensor& out) {
    const auto& layer = weights_.layers[layer_idx];
    int B = x.shape()[0];
    int T = x.shape()[1];
    int D = weights_.embed_dim;
    int hidden_dim = static_cast<int>(layer.w1.rows);  // Usually 4*D or 8/3*D

    if (out.shape() != Shape{B, T, D}) out.reshape({B, T, D});

    const float* x_ptr = x.as<float>();
    const float* ffn_norm_ptr = layer.ffn_norm.data.as<float>();
    float* out_ptr = out.as<float>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // RMSNorm
            float x_local[4096];
            for (int d = 0; d < D; ++d) x_local[d] = x_ptr[(b * T + t) * D + d];

            float sum_sq = 0;
            for (int d = 0; d < D; ++d) sum_sq += x_local[d] * x_local[d];
            float rms = std::sqrt(sum_sq / D + weights_.rms_eps);
            for (int d = 0; d < D; ++d) x_local[d] = x_local[d] / rms * ffn_norm_ptr[d];

            // FFN: SiLU(x @ W1^T) * (x @ W3^T) @ W2^T
            float gate[16384];  // Max hidden_dim
            float up[16384];

            matvec(layer.w1, x_local, gate, hidden_dim, D);
            for (int i = 0; i < hidden_dim; ++i) {
                float sig = 1.0f / (1.0f + std::exp(-gate[i]));
                gate[i] *= sig;  // SiLU
            }

            matvec(layer.w3, x_local, up, hidden_dim, D);
            for (int i = 0; i < hidden_dim; ++i) gate[i] *= up[i];

            matvec(layer.w2, gate, out_ptr + (b * T + t) * D, D, hidden_dim);
        }
    }
}

void Model::transformer_block(int layer_idx, const Tensor& x, Tensor& out,
                              KVCache& kv_cache, int start_pos) {
    // out = x + Attention(RMSNorm(x))
    // out = out + FFN(RMSNorm(out))

    // First sub-layer: Attention with residual
    Tensor attn_out(DataType::F32, x.shape(), device_);
    attention_forward(layer_idx, x, attn_out, kv_cache, start_pos);

    // Residual: out = x + attn_out
    if (out.shape() != x.shape()) out.reshape(x.shape());
    const float* x_ptr = x.as<float>();
    const float* a_ptr = attn_out.as<float>();
    float* out_ptr = out.as<float>();
    int n = x.numel();
    for (int i = 0; i < n; ++i) {
        out_ptr[i] = x_ptr[i] + a_ptr[i];
    }

    // Second sub-layer: FFN with residual
    Tensor ffn_out(DataType::F32, x.shape(), device_);
    ffn_forward(layer_idx, out, ffn_out);

    // Residual: out = out + ffn_out
    const float* f_ptr = ffn_out.as<float>();
    for (int i = 0; i < n; ++i) {
        out_ptr[i] += f_ptr[i];
    }
}

Tensor Model::forward(const Tensor& input_ids, KVCache* kv_cache, int start_pos) {
    int B = input_ids.shape()[0];
    int T = input_ids.shape()[1];
    int D = weights_.embed_dim;

    // Embedding lookup
    Tensor x(DataType::F32, {B, T, D}, device_);
    const int32_t* ids = input_ids.as<int32_t>();
    float* x_ptr = x.as<float>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = ids[b * T + t];
            if (id < 0 || id >= weights_.vocab_size) id = 0;
            embedding_row(weights_.tok_embeddings, id, x_ptr + (b * T + t) * D, D);
        }
    }

    // Transformer layers
    KVCache* cache = kv_cache;
    if (!cache) {
        // Create temporary cache
        static thread_local KVCache temp_cache;
        if (temp_cache.seq_len() == 0) {
            temp_cache = KVCache(weights_.n_layers, weights_.max_seq_len,
                                 weights_.n_kv_heads, weights_.head_dim, device_);
        }
        temp_cache.clear();
        cache = &temp_cache;
    }

    Tensor layer_out(DataType::F32, {B, T, D}, device_);

    for (int l = 0; l < weights_.n_layers; ++l) {
        if (l == 0) {
            transformer_block(l, x, layer_out, *cache, start_pos);
        } else {
            transformer_block(l, layer_out, x, *cache, start_pos);
            std::swap(layer_out, x);  // Ping-pong buffers; layer_out always holds latest
        }
    }

    // Keep the cache length in sync with absolute positions processed so far.
    cache->set_seq_len(start_pos + T);

    // After every iteration layer_out holds the newest block output.
    Tensor& final_out = layer_out;
    const float* final_norm_ptr = weights_.output_norm.data.as<float>();
    float* final_ptr = final_out.as<float>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* token_ptr = final_ptr + (b * T + t) * D;
            float sum_sq = 0;
            for (int d = 0; d < D; ++d) sum_sq += token_ptr[d] * token_ptr[d];
            float rms = std::sqrt(sum_sq / D + weights_.rms_eps);
            for (int d = 0; d < D; ++d) {
                final_ptr[(b * T + t) * D + d] = token_ptr[d] / rms * final_norm_ptr[d];
            }
        }
    }

    // Output projection (tied to the embedding when output.weight is absent).
    Tensor logits(DataType::F32, {B, T, weights_.vocab_size}, device_);
    const HostWeight& out_w =
        weights_.output_tied ? weights_.tok_embeddings : weights_.output;
    float* logits_ptr = logits.as<float>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* token_ptr = final_ptr + (b * T + t) * D;
            float* row = logits_ptr + (b * T + t) * weights_.vocab_size;
            matvec(out_w, token_ptr, row, weights_.vocab_size, D);
        }
    }

    return logits;
}

std::vector<int> Model::generate(const std::vector<int>& prompt_tokens,
                                 const InferenceConfig& config,
                                 KVCache& kv_cache) {
    std::vector<int> generated = prompt_tokens;
    int start_pos = 0;

    KVCache* cache_ptr = &kv_cache;
    kv_cache.clear();

    // Process prompt in chunks (for long prompts)
    // For simplicity, process all at once
    Tensor input(DataType::I32, {1, (int)prompt_tokens.size()}, device_);
    int32_t* input_ptr = input.as<int32_t>();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        input_ptr[i] = prompt_tokens[i];
    }

    // Forward on prompt
    Tensor logits = forward(input, cache_ptr, start_pos);
    start_pos = prompt_tokens.size();

    // Get last token logits
    const float* logits_ptr = logits.as<float>();
    const float* last_logits = logits_ptr + (start_pos - 1) * weights_.vocab_size;

    Tensor last_logits_tensor(DataType::F32, {weights_.vocab_size}, device_);
    std::memcpy(last_logits_tensor.data(), last_logits, weights_.vocab_size * sizeof(float));

    // Helper: apply repetition penalty in place to a 1-D logits tensor
    auto apply_repetition_penalty = [&](Tensor& lg, const std::vector<int>& context) {
        if (config.repetition_penalty <= 1.0f) return;
        float* p = lg.as<float>();
        int n = static_cast<int>(lg.numel());
        for (int tok : context) {
            if (tok < 0 || tok >= n) continue;
            if (p[tok] > 0.0f) p[tok] /= config.repetition_penalty;
            else p[tok] *= config.repetition_penalty;
        }
    };

    // Sample first token
    int next_token;
    apply_repetition_penalty(last_logits_tensor, generated);
    if (config.temperature <= 0.0f) {
        next_token = sample_argmax(last_logits_tensor);
    } else {
        next_token = sample_top_k_top_p(last_logits_tensor, config.top_k, config.top_p, config.temperature, config.seed >= 0 ? config.seed + (int)generated.size() : -1);
    }

    generated.push_back(next_token);

    // Generate remaining tokens
    for (int step = 1; step < config.max_tokens; ++step) {
        if (config.eos_token_id >= 0 && next_token == config.eos_token_id) break;

        // Input is single token
        Tensor single_input(DataType::I32, {1, 1}, device_);
        single_input.as<int32_t>()[0] = next_token;

        logits = forward(single_input, cache_ptr, start_pos);
        start_pos = static_cast<int>(generated.size());

        last_logits = logits.as<float>();  // Shape [1, 1, vocab]
        std::memcpy(last_logits_tensor.data(), last_logits, weights_.vocab_size * sizeof(float));

        apply_repetition_penalty(last_logits_tensor, generated);
        if (config.temperature <= 0.0f) {
            next_token = sample_argmax(last_logits_tensor);
        } else {
            next_token = sample_top_k_top_p(last_logits_tensor, config.top_k, config.top_p, config.temperature, config.seed >= 0 ? config.seed + (int)generated.size() : -1);
        }

        generated.push_back(next_token);
    }

    return generated;
}

}  // namespace inference