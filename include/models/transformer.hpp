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

// Single transformer block weights
struct TransformerBlockWeights {
    // Attention
    Tensor wq, wk, wv, wo;      // [embed, embed] or [embed, n_heads * head_dim]
    Tensor q_norm, k_norm;       // Optional QK norm (for some models)

    // FFN
    Tensor w1, w2, w3;           // Gate, Down, Up projections

    // Norms
    Tensor attn_norm;            // RMSNorm weight
    Tensor ffn_norm;             // RMSNorm weight

    // Quantization scales (if weights are quantized)
    // For simplicity, we'll dequantize on load
};

// Full model weights
struct ModelWeights {
    // Embedding
    Tensor tok_embeddings;       // [vocab, embed]
    Tensor output_norm;          // Final RMSNorm
    Tensor output;               // Output projection [embed, vocab] (often tied to tok_embeddings)

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

// Load weights from GGUF parser
ModelWeights load_model_weights(const class GGUFParser& parser, Device device = Device::cpu());

}  // namespace inference