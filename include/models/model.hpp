#pragma once

#include "models/transformer.hpp"
#include "core/tensor.hpp"
#include <vector>
#include <string>

namespace inference {

// Inference configuration
struct InferenceConfig {
    int max_tokens = 512;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    float repetition_penalty = 1.1f;
    int seed = -1;  // -1 = random
    int eos_token_id = -1;  // stop generation when produced (if >= 0)
    bool gpu_int16 = true;  // store GPU weights as INT16 (half the VRAM)
    bool host_int16 = true; // store host weights as INT16 (half the RAM)
};

// Main model class for inference
class Model {
public:
    explicit Model(ModelWeights weights, Device device = Device::cpu());

    // Run inference: generate tokens given prompt
    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                              const InferenceConfig& config,
                              KVCache& kv_cache);

    // Single forward pass (for embeddings, etc.)
    Tensor forward(const Tensor& input_ids, KVCache* kv_cache = nullptr, int start_pos = 0);

    // Access weights
    const ModelWeights& weights() const { return weights_; }
    const ModelWeights& mutable_weights() { return weights_; }

    // Config
    int vocab_size() const { return weights_.vocab_size; }
    int max_seq_len() const { return weights_.max_seq_len; }
    int n_layers() const { return weights_.n_layers; }
    int embed_dim() const { return weights_.embed_dim; }
    int n_heads() const { return weights_.n_heads; }
    int head_dim() const { return weights_.head_dim; }

private:
    // Single transformer block forward
    void transformer_block(int layer_idx, const Tensor& x, Tensor& out,
                           KVCache& kv_cache, int start_pos);

    // Attention forward
    void attention_forward(int layer_idx, const Tensor& x, Tensor& out,
                           KVCache& kv_cache, int start_pos);

    // FFN forward
    void ffn_forward(int layer_idx, const Tensor& x, Tensor& out);

    // Apply RoPE to q, k
    void apply_rope(Tensor& q, Tensor& k, int pos) const;

    ModelWeights weights_;
    Device device_;

    // Scratch buffers (reused across forward passes)
    mutable std::vector<Tensor> scratch_;
};

}  // namespace inference