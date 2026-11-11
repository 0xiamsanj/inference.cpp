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
// Host-side matvec / embedding for FP32 or INT16 weights
// ============================================================================
void matvec(const HostWeight& w, const float* x, float* y, int64_t rows, int64_t cols) {
    if (!w.int16) {
        const float* p = w.data.as<float>();
        for (int64_t r = 0; r < rows; ++r) {
            const float* row = p + r * cols;
            float sum = 0.0f;
            for (int64_t c = 0; c < cols; ++c) sum += row[c] * x[c];
            y[r] = sum;
        }
    } else {
        const int16_t* p = w.data.as<int16_t>();
        const float* scale = w.scale.as<float>();
        for (int64_t r = 0; r < rows; ++r) {
            const int16_t* row = p + r * cols;
            float sum = 0.0f;
            for (int64_t c = 0; c < cols; ++c) sum += static_cast<float>(row[c]) * x[c];
            y[r] = sum * scale[r];
        }
    }
}

void embedding_row(const HostWeight& w, int token, float* out, int64_t dim) {
    if (!w.int16) {
        const float* row = w.data.as<float>() + static_cast<int64_t>(token) * dim;
        std::memcpy(out, row, static_cast<size_t>(dim) * sizeof(float));
    } else {
        const int16_t* row = w.data.as<int16_t>() + static_cast<int64_t>(token) * dim;
        const float s = w.scale.as<float>()[token];
        for (int64_t i = 0; i < dim; ++i) out[i] = static_cast<float>(row[i]) * s;
    }
}

// ============================================================================
// Model Weight Loading from GGUF
// ============================================================================
ModelWeights load_model_weights(const GGUFParser& parser, Device device, bool host_int16) {
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

    // Load a tensor and dequantize it to FP32.
    //
    // GGML stores ne[0] contiguously, so file dims [in, out] are already
    // row-major [out][in]. We reverse the shape metadata and keep the bytes.
    auto load_tensor = [&](const std::string& name) -> Tensor {
        const GGUFTensorInfo* info = parser.find_tensor(name);
        if (!info) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        const int64_t numel = shape_numel(info->shape);
        Tensor flat(DataType::F32, {numel}, device);

        std::vector<uint8_t> raw(info->data_size);
        parser.read_tensor_data(*info, raw.data());

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

        Shape shape(info->shape.rbegin(), info->shape.rend());
        Tensor out(DataType::F32, shape, device);
        std::memcpy(out.data(), flat.data(), flat.byte_size());
        return out;
    };

    // Convert an FP32 tensor to a host weight. 2-D tensors can be quantized to
    // symmetric INT16 with one float scale per output row; norms stay FP32.
    auto make_weight = [&](Tensor t, bool quantize) -> HostWeight {
        HostWeight hw;
        if (t.shape().size() == 2) {
            hw.rows = t.shape()[0];
            hw.cols = t.shape()[1];
        } else {
            hw.rows = 1;
            hw.cols = t.numel();
        }

        if (!quantize || t.shape().size() != 2) {
            hw.data = std::move(t);
            hw.int16 = false;
            return hw;
        }

        hw.int16 = true;
        hw.data = Tensor(DataType::I16, {hw.rows, hw.cols}, device);
        hw.scale = Tensor(DataType::F32, {hw.rows}, device);

        const float* src = t.as<float>();
        int16_t* q = hw.data.as<int16_t>();
        float* scales = hw.scale.as<float>();

        for (int64_t r = 0; r < hw.rows; ++r) {
            const float* row = src + r * hw.cols;
            float max_abs = 0.0f;
            for (int64_t c = 0; c < hw.cols; ++c) {
                float a = std::fabs(row[c]);
                if (a > max_abs) max_abs = a;
            }
            const float scale = max_abs > 0.0f ? max_abs / 32767.0f : 1.0f;
            const float inv = 1.0f / scale;
            scales[r] = scale;
            int16_t* qrow = q + r * hw.cols;
            for (int64_t c = 0; c < hw.cols; ++c) {
                long v = std::lround(row[c] * inv);
                if (v > 32767) v = 32767;
                if (v < -32767) v = -32767;
                qrow[c] = static_cast<int16_t>(v);
            }
        }
        return hw;
    };

    auto load_weight = [&](const std::string& name, bool quantize) -> HostWeight {
        return make_weight(load_tensor(name), quantize && host_int16);
    };

    auto try_load = [&](const std::vector<std::string>& names, bool quantize) -> HostWeight {
        for (const auto& name : names) {
            try {
                return load_weight(name, quantize);
            } catch (...) {}
        }
        throw std::runtime_error("Could not load any of: " + names[0]);
    };

    // Embeddings (GGUF [embed, vocab] -> model [vocab, embed])
    weights.tok_embeddings = try_load(
        {"token_embd.weight", "model.embed_tokens.weight"}, true);
    weights.vocab_size = static_cast<int>(weights.tok_embeddings.rows);
    weights.embed_dim = static_cast<int>(weights.tok_embeddings.cols);

    weights.output_norm = try_load({"output_norm.weight", "model.norm.weight"}, false);

    // Output projection (often tied to the embedding).
    const GGUFTensorInfo* out_info = parser.find_tensor("output.weight");
    if (out_info) {
        weights.output = load_weight("output.weight", true);
    } else {
        weights.output_tied = true;  // reuse tok_embeddings
    }

    for (int l = 0; l < weights.n_layers; ++l) {
        auto& layer = weights.layers[l];
        std::string prefix = "blk." + std::to_string(l) + ".";
        std::string prefix2 = "model.layers." + std::to_string(l) + ".";

        layer.wq = try_load({prefix + "attn_q.weight", prefix2 + "self_attn.q_proj.weight"}, true);
        layer.wk = try_load({prefix + "attn_k.weight", prefix2 + "self_attn.k_proj.weight"}, true);
        layer.wv = try_load({prefix + "attn_v.weight", prefix2 + "self_attn.v_proj.weight"}, true);
        layer.wo = try_load({prefix + "attn_output.weight", prefix2 + "self_attn.o_proj.weight"}, true);

        layer.w1 = try_load({prefix + "ffn_gate.weight", prefix2 + "mlp.gate_proj.weight"}, true);
        layer.w2 = try_load({prefix + "ffn_down.weight", prefix2 + "mlp.down_proj.weight"}, true);
        layer.w3 = try_load({prefix + "ffn_up.weight", prefix2 + "mlp.up_proj.weight"}, true);

        layer.attn_norm = try_load(
            {prefix + "attn_norm.weight", prefix2 + "input_layernorm.weight"}, false);
        layer.ffn_norm = try_load(
            {prefix + "ffn_norm.weight", prefix2 + "post_attention_layernorm.weight"}, false);

        try {
            layer.q_norm = load_weight(prefix + "attn_q_norm.weight", false);
            layer.k_norm = load_weight(prefix + "attn_k_norm.weight", false);
        } catch (...) {
            // Not present in all models
        }
    }

    return weights;
}

}  // namespace inference
