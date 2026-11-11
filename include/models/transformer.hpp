#pragma once

#include "core/tensor.hpp"
#include <vector>
#include <memory>

namespace inference {

// Key-Value Cache for efficient autoregressive generation
class KVCache {
public:
    KVCache() = default;
    KVCache(int num_layers, int max_seq_len, int num_heads, int head_dim, Device device = Device::cpu());

    // Get cache for a specific layer
    Tensor& key(int layer) { return keys_[layer]; }
    Tensor& value(int layer) { return values_[layer]; }
    const Tensor& key(int layer) const { return keys_[layer]; }
    const Tensor& value(int layer) const { return values_[layer]; }

    // Current sequence length in cache
    int seq_len() const { return seq_len_; }

    // Explicitly set/advance the sequence length (used by the model)
    void set_seq_len(int n) { seq_len_ = n; }
    void advance(int n) { seq_len_ += n; }

    // Append new K,V for a layer (shapes: [B, H, 1, D] or [B, H, T, D])
    void append(int layer, const Tensor& k, const Tensor& v);

    // Clear cache
    void clear();

    // Get view of cache up to current seq_len
    Tensor key_view(int layer) const;
    Tensor value_view(int layer) const;

private:
    std::vector<Tensor> keys_;
    std::vector<Tensor> values_;
    int num_layers_ = 0;
    int max_seq_len_ = 0;
    int num_heads_ = 0;
    int head_dim_ = 0;
    int seq_len_ = 0;
    Device device_;
};

// A host weight can be stored as FP32 or symmetric INT16 with one scale per row.
struct HostWeight {
    Tensor data;    // F32 or I16, shape [rows, cols] (norms are [1, n])
    Tensor scale;   // F32 [rows] when int16 == true
    bool int16 = false;
    int64_t rows = 0;
    int64_t cols = 0;

    bool empty() const { return rows == 0 || cols == 0; }
    size_t gpu_bytes(bool gpu_int16) const {
        const size_t n = static_cast<size_t>(rows) * cols;
        if (gpu_int16) return n * sizeof(int16_t) + static_cast<size_t>(rows) * sizeof(float);
        return n * sizeof(float);
    }
};

// y = W @ x, where W is [rows, cols] and x is [cols].
void matvec(const HostWeight& w, const float* x, float* y, int64_t rows, int64_t cols);

// out = W[token, :] dequantized into FP32.
void embedding_row(const HostWeight& w, int token, float* out, int64_t dim);

// Single transformer block weights
struct TransformerBlockWeights {
    // Attention
    HostWeight wq, wk, wv, wo;
    HostWeight q_norm, k_norm;

    // FFN
    HostWeight w1, w2, w3;

    // Norms
    HostWeight attn_norm;
    HostWeight ffn_norm;
};

// Full model weights
struct ModelWeights {
    // Embedding
    HostWeight tok_embeddings;   // [vocab, embed]
    HostWeight output_norm;      // final RMSNorm
    HostWeight output;           // output projection [vocab, embed] (or empty if tied)

    // Layers
    std::vector<TransformerBlockWeights> layers;

    // Config
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;          // For GQA
    int embed_dim = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    float rope_theta = 10000.0f;
    float rope_scaling = 1.0f;
    float rms_eps = 1e-6f;
    bool output_tied = false;    // output projection shares the embedding
};

// Load weights from GGUF parser. host_int16 halves host RAM.
ModelWeights load_model_weights(const class GGUFParser& parser,
                                Device device = Device::cpu(),
                                bool host_int16 = true);

}  // namespace inference