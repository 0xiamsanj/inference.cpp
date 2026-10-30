#include "models/transformer.hpp"
#include "gguf/gguf_parser.hpp"
#include "gguf/quantization.hpp"
#include "ops/ops.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace inference {

// ============================================================================
// KVCache Implementation
// ============================================================================
KVCache::KVCache(int num_layers, int max_seq_len, int num_heads, int head_dim, Device device)
    : num_layers_(num_layers), max_seq_len_(max_seq_len),
      num_heads_(num_heads), head_dim_(head_dim), device_(device) {

    keys_.reserve(num_layers);
    values_.reserve(num_layers);

    // Pre-allocate cache: [1, num_heads, max_seq_len, head_dim]
    // Using batch=1 for simplicity (can extend to multi-batch)
    Shape cache_shape = {1, num_heads, max_seq_len, head_dim};

    for (int i = 0; i < num_layers; ++i) {
        keys_.emplace_back(DataType::F32, cache_shape, device);
        values_.emplace_back(DataType::F32, cache_shape, device);
    }
}

void KVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::runtime_error("KVCache: invalid layer");
    }

    // k, v shape: [1, num_heads, T, head_dim]
    // We need to copy into cache at position seq_len_
    int T = k.shape()[2];

    if (seq_len_ + T > max_seq_len_) {
        throw std::runtime_error("KVCache: exceeded max sequence length");
    }

    // Simple copy for now (could optimize with batched copy)
    // Cache: [1, H, max_seq, D]
    // Input: [1, H, T, D]
    const float* k_src = k.as<float>();
    const float* v_src = v.as<float>();
    float* k_dst = keys_[layer].as<float>();
    float* v_dst = values_[layer].as<float>();

    int64_t H = num_heads_;
    int64_t D = head_dim_;

    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                int64_t cache_idx = h * max_seq_len_ * D + (seq_len_ + t) * D + d;
                int64_t src_idx = h * T * D + t * D + d;
                k_dst[cache_idx] = k_src[src_idx];
                v_dst[cache_idx] = v_src[src_idx];
            }
        }
    }

    seq_len_ += T;
}

void KVCache::clear() {
    seq_len_ = 0;
}

Tensor KVCache::key_view(int layer) const {
    if (layer < 0 || layer >= num_layers_) {
        throw std::runtime_error("KVCache: invalid layer");
    }

    // Return view: [1, H, seq_len_, D]
    Tensor view = keys_[layer];
    Shape view_shape = {1, num_heads_, seq_len_, head_dim_};
    view.reshape(view_shape);
    return view;
}

Tensor KVCache::value_view(int layer) const {
    if (layer < 0 || layer >= num_layers_) {
        throw std::runtime_error("KVCache: invalid layer");
    }

    Tensor view = values_[layer];
    Shape view_shape = {1, num_heads_, seq_len_, head_dim_};
    view.reshape(view_shape);
    return view;
}

// ============================================================================
// Model Weight Loading from GGUF
// ============================================================================
ModelWeights load_model_weights(const GGUFParser& parser, Device device) {
    ModelWeights weights;

    // Read config from metadata
    weights.n_layers = parser.get_block_count();
    weights.embed_dim = parser.get_embedding_dim();
    weights.n_heads = parser.get_num_heads();
    weights.head_dim = parser.get_head_dim();
    weights.vocab_size = parser.get_vocab_size();

    // GQA: check for n_kv_heads
    try {
        weights.n_kv_heads = parser.get_metadata<uint32_t>("llama.attention.head_count_kv");
    } catch (...) {
        weights.n_kv_heads = weights.n_heads;  // Default: MHA
    }

    weights.max_seq_len = get_metadata_or<uint32_t>(parser, "llama.context_length", 2048);
    weights.rope_theta = get_metadata_or<float>(parser, "llama.rope.freq_base", 10000.0f);
    weights.rms_eps = get_metadata_or<float>(parser, "llama.attention.layer_norm_rms_epsilon", 1e-6f);

    weights.layers.resize(weights.n_layers);

    // Helper to load a tensor and dequantize to FP32.
    //
    // GGML stores ne[0] contiguously, and GGUF writes ggml dims directly.
    // For a linear weight with dims [in, out], the byte layout is therefore
    // row-major [out][in] (each output row of `in` values is contiguous).
    // So we keep the bytes in file order and simply reverse the shape:
    //   file dims [in, out]  ->  Tensor shape [out, in]
    // This matches the model code, which indexes w[row_out * in + i_in].
    auto load_tensor = [&](const std::string& name) -> Tensor {
        const GGUFTensorInfo* info = parser.find_tensor(name);
        if (!info) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        const int64_t numel = shape_numel(info->shape);
        Tensor flat(DataType::F32, {numel}, device);

        // Read raw (possibly quantized) bytes
        std::vector<uint8_t> raw(info->data_size);
        parser.read_tensor_data(*info, raw.data());

        // Convert to FP32 (linear layout, same order as file)
        if (info->dtype == DataType::F32) {
            std::memcpy(flat.data(), raw.data(), raw.size());
        } else if (info->dtype == DataType::F16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
            float* dst = flat.as<float>();
            for (int64_t i = 0; i < numel; ++i) {
                uint16_t h = src[i];
                uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
                uint32_t exp = (h >> 10) & 0x1Fu;
                uint32_t mant = h & 0x3FFu;
                float fv;
                if (exp == 0) {
                    fv = std::ldexp(static_cast<float>(mant), -24);
                    if (h & 0x8000u) fv = -fv;
                } else {
                    uint32_t f;
                    if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
                    else f = sign | ((exp + 112u) << 23) | (mant << 13);
                    std::memcpy(&fv, &f, sizeof(fv));
                }
                dst[i] = fv;
            }
        } else {
            dequantize(info->dtype, raw.data(), flat.as<float>(), numel);
        }

        // Reverse dims to get C row-major shape (ne[0] is contiguous on disk).
        Shape shape(info->shape.rbegin(), info->shape.rend());
        Tensor out(DataType::F32, shape, device);
        std::memcpy(out.data(), flat.data(), flat.byte_size());
        return out;
    };

    // Load embeddings (GGUF [embed, vocab] -> model [vocab, embed])
    try {
        weights.tok_embeddings = load_tensor("token_embd.weight");
    } catch (...) {
        weights.tok_embeddings = load_tensor("model.embed_tokens.weight");
    }
    weights.vocab_size = static_cast<int>(weights.tok_embeddings.shape()[0]);
    weights.embed_dim = static_cast<int>(weights.tok_embeddings.shape()[1]);

    // Final norm
    try {
        weights.output_norm = load_tensor("output_norm.weight");
    } catch (...) {
        weights.output_norm = load_tensor("model.norm.weight");
    }

    // Output projection (often tied to embedding)
    const GGUFTensorInfo* out_info = parser.find_tensor("output.weight");
    if (out_info) {
        weights.output = load_tensor("output.weight");
    } else {
        weights.output = weights.tok_embeddings;  // tied weights
        weights.output_tied = true;
    }

    // Load each layer
    for (int l = 0; l < weights.n_layers; ++l) {
        auto& layer = weights.layers[l];
        std::string prefix = "blk." + std::to_string(l) + ".";
        // Alternative naming
        std::string prefix2 = "model.layers." + std::to_string(l) + ".";

        auto try_load = [&](const std::vector<std::string>& names) -> Tensor {
            for (const auto& name : names) {
                try {
                    return load_tensor(name);
                } catch (...) {}
            }
            throw std::runtime_error("Could not load any of: " + names[0]);
        };

        // Attention projections
        layer.wq = try_load({prefix + "attn_q.weight", prefix2 + "self_attn.q_proj.weight"});
        layer.wk = try_load({prefix + "attn_k.weight", prefix2 + "self_attn.k_proj.weight"});
        layer.wv = try_load({prefix + "attn_v.weight", prefix2 + "self_attn.v_proj.weight"});
        layer.wo = try_load({prefix + "attn_output.weight", prefix2 + "self_attn.o_proj.weight"});

        // FFN projections
        layer.w1 = try_load({prefix + "ffn_gate.weight", prefix2 + "mlp.gate_proj.weight"});
        layer.w2 = try_load({prefix + "ffn_down.weight", prefix2 + "mlp.down_proj.weight"});
        layer.w3 = try_load({prefix + "ffn_up.weight", prefix2 + "mlp.up_proj.weight"});

        // Norms
        layer.attn_norm = try_load({prefix + "attn_norm.weight", prefix2 + "input_layernorm.weight"});
        layer.ffn_norm = try_load({prefix + "ffn_norm.weight", prefix2 + "post_attention_layernorm.weight"});

        // Optional QK norm
        try {
            layer.q_norm = load_tensor(prefix + "attn_q_norm.weight");
            layer.k_norm = load_tensor(prefix + "attn_k_norm.weight");
        } catch (...) {
            // Not present in all models
        }
    }

    return weights;
}

}  // namespace inference