#include "models/cuda_model.hpp"
#include "ops/cuda_backend.hpp"
#include "ops/ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace inference {

#ifdef WITH_CUDA

namespace {

using cuda::Buffer;

// A linear weight, either FP32 or INT16 with one float scale per output row.
struct Weight {
    Buffer data;
    Buffer scale;      // valid only when int16 == true
    bool int16 = false;
};

struct GpuLayer {
    Weight wq, wk, wv, wo;
    Weight w1, w2, w3;
    Buffer attn_norm, ffn_norm;
};

struct GpuState {
    Weight tok_emb, out_w;
    Buffer out_norm, rope_table;
    bool int16_weights = true;
    std::vector<GpuLayer> layers;
    std::vector<Buffer> kc, vc;  // per layer

    // activations
    Buffer x, xn, q, k, v, attn, proj, gate, up, logits;

    int D = 0, H = 0, Hkv = 0, HD = 0, V = 0, FFN = 0, MAXSEQ = 0;
    float eps = 1e-6f;
    std::vector<float> host_logits;
};

std::mutex g_mutex;
std::unordered_map<const Model*, std::unique_ptr<GpuState>> g_states;

Buffer upload_tensor(const Tensor& t) {
    Buffer b = cuda::alloc(t.byte_size());
    cuda::upload(b, t.data(), t.byte_size());
    return b;
}

// Symmetric per-row INT16 quantization: w ~= q * scale, scale = max|w| / 32767.
Weight quantize_i16(const Tensor& t) {
    if (t.dtype() != DataType::F32 || t.shape().size() != 2) {
        throw std::runtime_error("INT16 quantization requires a 2-D FP32 tensor");
    }
    const int64_t rows = t.shape()[0];
    const int64_t cols = t.shape()[1];
    const float* src = t.as<float>();

    std::vector<int16_t> q(static_cast<size_t>(rows) * cols);
    std::vector<float> scales(static_cast<size_t>(rows));

    for (int64_t r = 0; r < rows; ++r) {
        const float* row = src + r * cols;
        float max_abs = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float a = std::fabs(row[c]);
            if (a > max_abs) max_abs = a;
        }
        const float scale = max_abs > 0.0f ? max_abs / 32767.0f : 1.0f;
        const float inv = 1.0f / scale;
        scales[static_cast<size_t>(r)] = scale;
        int16_t* qrow = q.data() + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            long v = std::lround(row[c] * inv);
            if (v > 32767) v = 32767;
            if (v < -32767) v = -32767;
            qrow[c] = static_cast<int16_t>(v);
        }
    }

    Weight w;
    w.int16 = true;
    w.data = cuda::alloc(q.size() * sizeof(int16_t));
    cuda::upload(w.data, q.data(), q.size() * sizeof(int16_t));
    w.scale = cuda::alloc(scales.size() * sizeof(float));
    cuda::upload(w.scale, scales.data(), scales.size() * sizeof(float));
    return w;
}

Weight upload_weight(const HostWeight& hw, bool int16) {
    Weight w;
    if (int16) {
        if (hw.int16) {
            w.int16 = true;
            w.data = cuda::alloc(hw.data.byte_size());
            cuda::upload(w.data, hw.data.data(), hw.data.byte_size());
            w.scale = cuda::alloc(hw.scale.byte_size());
            cuda::upload(w.scale, hw.scale.data(), hw.scale.byte_size());
            return w;
        }
        return quantize_i16(hw.data);
    }

    if (!hw.int16) {
        w.int16 = false;
        w.data = upload_tensor(hw.data);
        return w;
    }

    // Host INT16 -> GPU FP32.
    const int64_t n = hw.rows * hw.cols;
    std::vector<float> tmp(static_cast<size_t>(n));
    const int16_t* q = hw.data.as<int16_t>();
    const float* scales = hw.scale.as<float>();
    for (int64_t r = 0; r < hw.rows; ++r) {
        for (int64_t c = 0; c < hw.cols; ++c) {
            tmp[static_cast<size_t>(r * hw.cols + c)] =
                static_cast<float>(q[r * hw.cols + c]) * scales[r];
        }
    }
    w.int16 = false;
    w.data = cuda::alloc(tmp.size() * sizeof(float));
    cuda::upload(w.data, tmp.data(), tmp.size() * sizeof(float));
    return w;
}

void upload_weights(GpuState& s, const ModelWeights& w, bool int16) {
    s.int16_weights = int16;
    s.D = w.embed_dim;
    s.H = w.n_heads;
    s.Hkv = w.n_kv_heads;
    s.HD = w.head_dim;
    s.V = w.vocab_size;
    s.FFN = static_cast<int>(w.layers[0].w1.rows);
    s.MAXSEQ = w.max_seq_len;
    s.eps = w.rms_eps;

    s.tok_emb = upload_weight(w.tok_embeddings, int16);
    s.out_norm = upload_tensor(w.output_norm.data);

    // Output projection may be tied to the token embedding: share the buffer.
    if (w.output_tied) {
        s.out_w = s.tok_emb;
    } else {
        s.out_w = upload_weight(w.output, int16);
    }

    s.layers.resize(w.layers.size());
    for (size_t i = 0; i < w.layers.size(); ++i) {
        const auto& src = w.layers[i];
        auto& dst = s.layers[i];
        dst.wq = upload_weight(src.wq, int16);
        dst.wk = upload_weight(src.wk, int16);
        dst.wv = upload_weight(src.wv, int16);
        dst.wo = upload_weight(src.wo, int16);
        dst.w1 = upload_weight(src.w1, int16);
        dst.w2 = upload_weight(src.w2, int16);
        dst.w3 = upload_weight(src.w3, int16);
        dst.attn_norm = upload_tensor(src.attn_norm.data);
        dst.ffn_norm = upload_tensor(src.ffn_norm.data);
    }

    // Precompute the RoPE cos/sin table on the CPU (matches the CPU path).
    const int half = s.HD / 2;
    std::vector<float> table(static_cast<size_t>(s.MAXSEQ) * half * 2);
    for (int pos = 0; pos < s.MAXSEQ; ++pos) {
        for (int j = 0; j < half; ++j) {
            float freq = 1.0f / std::pow(w.rope_theta, static_cast<float>(2 * j) / s.HD);
            float angle = pos * freq;
            size_t idx = (static_cast<size_t>(pos) * half + j) * 2;
            table[idx] = std::cos(angle);
            table[idx + 1] = std::sin(angle);
        }
    }
    s.rope_table = cuda::alloc(table.size() * sizeof(float));
    cuda::upload(s.rope_table, table.data(), table.size() * sizeof(float));

    // Allocate activations
    auto alloc_floats = [](int n) { return cuda::alloc(static_cast<size_t>(n) * sizeof(float)); };
    s.x = alloc_floats(s.D);
    s.xn = alloc_floats(s.D);
    s.q = alloc_floats(s.H * s.HD);
    s.k = alloc_floats(s.Hkv * s.HD);
    s.v = alloc_floats(s.Hkv * s.HD);
    s.attn = alloc_floats(s.H * s.HD);
    s.proj = alloc_floats(s.D);
    s.gate = alloc_floats(s.FFN);
    s.up = alloc_floats(s.FFN);
    s.logits = alloc_floats(s.V);
    s.host_logits.resize(s.V);

    // KV cache
    const int n_layers = static_cast<int>(w.layers.size());
    s.kc.resize(n_layers);
    s.vc.resize(n_layers);
    const size_t kv_floats = static_cast<size_t>(s.Hkv) * s.MAXSEQ * s.HD;
    for (int i = 0; i < n_layers; ++i) {
        s.kc[i] = cuda::alloc(kv_floats * sizeof(float));
        s.vc[i] = cuda::alloc(kv_floats * sizeof(float));
    }
}

size_t required_bytes(const ModelWeights& w, bool int16) {
    size_t n = 0;
    auto add = [&](const HostWeight& hw) { n += hw.gpu_bytes(int16); };
    add(w.tok_embeddings);
    add(w.output_norm);
    if (!w.output_tied) add(w.output);
    for (const auto& l : w.layers) {
        add(l.wq); add(l.wk); add(l.wv); add(l.wo);
        add(l.w1); add(l.w2); add(l.w3);
        add(l.attn_norm); add(l.ffn_norm);
    }
    // KV cache + activations (rough upper bound)
    const size_t kv_per_layer = static_cast<size_t>(w.n_kv_heads) * w.max_seq_len * w.head_dim;
    n += kv_per_layer * 2 * w.n_layers * sizeof(float);
    n += static_cast<size_t>(w.max_seq_len) * (w.head_dim / 2) * 2 * sizeof(float);
    n += static_cast<size_t>(w.embed_dim * 4 + w.vocab_size +
                             static_cast<int64_t>(w.layers[0].w1.rows) * 2) * sizeof(float);
    return n;
}

GpuState& get_state(const Model& model, bool int16) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_states.find(&model);
    if (it != g_states.end()) {
        if (it->second->int16_weights == int16) return *it->second;
        g_states.erase(it);  // precision changed: rebuild
    }

    const ModelWeights& w = model.weights();
    const size_t need = required_bytes(w, int16);
    const size_t free_b = cuda::free_memory();
    const size_t margin = 64ull * 1024 * 1024;
    if (need + margin > free_b) {
        throw std::runtime_error(
            "CUDA: model needs ~" + std::to_string(need / (1024 * 1024)) +
            " MB VRAM (" + (int16 ? "int16" : "fp32") + " weights) but only " +
            std::to_string(free_b / (1024 * 1024)) +
            " MB is free. Try int16, a smaller model, or the CPU backend.");
    }

    auto state = std::make_unique<GpuState>();
    upload_weights(*state, w, int16);
    GpuState& ref = *state;
    g_states.emplace(&model, std::move(state));
    return ref;
}

// Dispatch a linear projection using either FP32 or INT16 weights.
inline void linear(const Weight& w, const Buffer& x, const Buffer& y, int M, int N) {
    if (w.int16) cuda::gemv_i16(w.data, w.scale, x, y, M, N);
    else         cuda::gemv(w.data, x, y, M, N);
}

inline void linear_batch(const Weight& w, const Buffer& X, const Buffer& Y,
                         int T, int M, int K) {
    if (w.int16) cuda::gemm_i16(w.data, w.scale, X, Y, T, M, K);
    else         cuda::gemm(w.data, X, Y, T, M, K);
}

// Frees prefill scratch buffers on scope exit.
struct ScopedBuffers {
    std::vector<Buffer> buffers;
    ~ScopedBuffers() {
        for (auto& b : buffers) {
            if (b.valid()) cuda::release(b);
        }
    }
    Buffer add_bytes(size_t bytes) {
        Buffer b = cuda::alloc(bytes);
        buffers.push_back(b);
        return b;
    }
    Buffer add_floats(int64_t n) {
        return add_bytes(static_cast<size_t>(n) * sizeof(float));
    }
};

// Batched prompt prefill. Processes all prompt tokens through each layer with a
// tiled GEMM, then computes logits for the last token only.
void prefill(GpuState& s, const std::vector<int>& tokens, int start) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0) return;

    const int D = s.D, H = s.H, Hkv = s.Hkv, HD = s.HD, F = s.FFN;

    const size_t floats =
        static_cast<size_t>(T) * (2 * D + 2 * H * HD + 2 * Hkv * HD + 2 * F) +
        static_cast<size_t>(s.V) + static_cast<size_t>(T);
    const size_t bytes = floats * sizeof(float);
    const size_t free_b = cuda::free_memory();
    if (bytes + (32ull << 20) > free_b) {
        throw std::runtime_error(
            "CUDA prefill: prompt needs ~" + std::to_string(bytes >> 20) +
            " MB scratch but only " + std::to_string(free_b >> 20) +
            " MB VRAM is free. Shorten the prompt or use the CPU backend.");
    }

    ScopedBuffers sb;
    Buffer x = sb.add_floats(static_cast<int64_t>(T) * D);
    Buffer xn = sb.add_floats(static_cast<int64_t>(T) * D);
    Buffer q = sb.add_floats(static_cast<int64_t>(T) * H * HD);
    Buffer k = sb.add_floats(static_cast<int64_t>(T) * Hkv * HD);
    Buffer v = sb.add_floats(static_cast<int64_t>(T) * Hkv * HD);
    Buffer attn = sb.add_floats(static_cast<int64_t>(T) * H * HD);
    Buffer proj = sb.add_floats(static_cast<int64_t>(T) * D);
    Buffer gate = sb.add_floats(static_cast<int64_t>(T) * F);
    Buffer up = sb.add_floats(static_cast<int64_t>(T) * F);
    Buffer ids = sb.add_bytes(static_cast<size_t>(T) * sizeof(int32_t));
    Buffer last_xn = sb.add_floats(D);

    std::vector<int32_t> host_ids(tokens.begin(), tokens.end());
    cuda::upload(ids, host_ids.data(), host_ids.size() * sizeof(int32_t));

    if (s.tok_emb.int16) {
        cuda::embedding_batch_i16(s.tok_emb.data, s.tok_emb.scale, ids, x, T, D);
    } else {
        cuda::embedding_batch(s.tok_emb.data, ids, x, T, D);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    for (size_t l = 0; l < s.layers.size(); ++l) {
        const auto& L = s.layers[l];

        cuda::rmsnorm(x, L.attn_norm, xn, T, D, s.eps);
        linear_batch(L.wq, xn, q, T, H * HD, D);
        linear_batch(L.wk, xn, k, T, Hkv * HD, D);
        linear_batch(L.wv, xn, v, T, Hkv * HD, D);

        cuda::rope_batch(q, k, s.rope_table, H, Hkv, HD, start, T);
        cuda::kv_write_batch(s.kc[l], s.vc[l], k, v, Hkv, HD, s.MAXSEQ, start, T);
        cuda::attention_batch(q, s.kc[l], s.vc[l], attn,
                              H, Hkv, HD, s.MAXSEQ, start, T, scale);

        linear_batch(L.wo, attn, proj, T, D, H * HD);
        cuda::add(x, proj, T * D);

        cuda::rmsnorm(x, L.ffn_norm, xn, T, D, s.eps);
        linear_batch(L.w1, xn, gate, T, F, D);
        cuda::silu(gate, T * F);
        linear_batch(L.w3, xn, up, T, F, D);
        cuda::mul(gate, up, T * F);
        linear_batch(L.w2, gate, proj, T, D, F);
        cuda::add(x, proj, T * D);
    }

    // Only the last prompt position is needed to start decoding.
    Buffer last_x{x.ptr + static_cast<uint64_t>(T - 1) * D * sizeof(float),
                  static_cast<size_t>(D) * sizeof(float)};
    cuda::rmsnorm(last_x, s.out_norm, last_xn, 1, D, s.eps);
    linear(s.out_w, last_xn, s.logits, s.V, D);
    cuda::sync();
    cuda::download(s.host_logits.data(), s.logits,
                   static_cast<size_t>(s.V) * sizeof(float));
}

// Run one token through the network at absolute position `pos`. The resulting
// logits are downloaded into s.host_logits.
void forward_token(GpuState& s, int token, int pos) {
    if (s.tok_emb.int16) cuda::embedding_i16(s.tok_emb.data, s.tok_emb.scale, token, s.x, s.D);
    else                 cuda::embedding(s.tok_emb.data, token, s.x, s.D);

    const float scale = 1.0f / std::sqrt(static_cast<float>(s.HD));

    for (size_t l = 0; l < s.layers.size(); ++l) {
        const auto& L = s.layers[l];

        cuda::rmsnorm(s.x, L.attn_norm, s.xn, 1, s.D, s.eps);
        linear(L.wq, s.xn, s.q, s.H * s.HD, s.D);
        linear(L.wk, s.xn, s.k, s.Hkv * s.HD, s.D);
        linear(L.wv, s.xn, s.v, s.Hkv * s.HD, s.D);

        cuda::rope(s.q, s.k, s.rope_table, s.H, s.Hkv, s.HD, pos);
        cuda::kv_write(s.kc[l], s.vc[l], s.k, s.v, s.Hkv, s.HD, s.MAXSEQ, pos);
        cuda::attention(s.q, s.kc[l], s.vc[l], s.attn,
                        s.H, s.Hkv, s.HD, s.MAXSEQ, pos, scale);
        linear(L.wo, s.attn, s.proj, s.D, s.H * s.HD);
        cuda::add(s.x, s.proj, s.D);

        cuda::rmsnorm(s.x, L.ffn_norm, s.xn, 1, s.D, s.eps);
        linear(L.w1, s.xn, s.gate, s.FFN, s.D);
        cuda::silu(s.gate, s.FFN);
        linear(L.w3, s.xn, s.up, s.FFN, s.D);
        cuda::mul(s.gate, s.up, s.FFN);
        linear(L.w2, s.gate, s.proj, s.D, s.FFN);
        cuda::add(s.x, s.proj, s.D);
    }

    cuda::rmsnorm(s.x, s.out_norm, s.xn, 1, s.D, s.eps);
    linear(s.out_w, s.xn, s.logits, s.V, s.D);
    cuda::sync();
    cuda::download(s.host_logits.data(), s.logits,
                   static_cast<size_t>(s.V) * sizeof(float));
}

int pick_token(const GpuState& s, const InferenceConfig& config,
               const std::vector<int>& context) {
    Tensor logits(DataType::F32, {s.V});
    std::memcpy(logits.data(), s.host_logits.data(),
                static_cast<size_t>(s.V) * sizeof(float));

    if (config.repetition_penalty > 1.0f) {
        float* p = logits.as<float>();
        for (int tok : context) {
            if (tok < 0 || tok >= s.V) continue;
            if (p[tok] > 0.0f) p[tok] /= config.repetition_penalty;
            else p[tok] *= config.repetition_penalty;
        }
    }

    if (config.temperature <= 0.0f) return sample_argmax(logits);
    const int seed = config.seed >= 0 ? config.seed + (int)context.size() : -1;
    return sample_top_k_top_p(logits, config.top_k, config.top_p, config.temperature, seed);
}

}  // namespace

bool cuda_model_available() { return cuda::available(); }

size_t cuda_required_bytes(const Model& model, bool int16_weights) {
    return required_bytes(model.weights(), int16_weights);
}

void release_cuda_model(const Model& model) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_states.erase(&model);
}

std::vector<int> generate_cuda(const Model& model,
                               const std::vector<int>& prompt_tokens,
                               const InferenceConfig& config,
                               GenerationStats* stats) {
    if (prompt_tokens.empty()) {
        throw std::runtime_error("CUDA generate: empty prompt");
    }

    auto clock_ms = [] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    };

    const double t0 = clock_ms();
    GpuState& s = get_state(model, config.gpu_int16);
    const double t1 = clock_ms();

    if (static_cast<int>(prompt_tokens.size()) > s.MAXSEQ) {
        throw std::runtime_error("CUDA generate: prompt longer than context");
    }

    std::vector<int> generated = prompt_tokens;

    // Batched prompt prefill (also fills the KV cache).
    prefill(s, prompt_tokens, 0);

    const double t2 = clock_ms();

    int pos = static_cast<int>(prompt_tokens.size());
    int next = pick_token(s, config, generated);
    generated.push_back(next);

    for (int step = 1; step < config.max_tokens; ++step) {
        if (config.eos_token_id >= 0 && next == config.eos_token_id) break;
        if (pos >= s.MAXSEQ) break;

        forward_token(s, next, pos);
        ++pos;
        next = pick_token(s, config, generated);
        generated.push_back(next);
    }

    const double t3 = clock_ms();

    if (stats) {
        stats->init_ms = t1 - t0;
        stats->prefill_ms = t2 - t1;
        stats->decode_ms = t3 - t2;
        stats->generated = static_cast<int>(generated.size() - prompt_tokens.size());
    }

    return generated;
}

#else  // !WITH_CUDA

bool cuda_model_available() { return false; }

size_t cuda_required_bytes(const Model&, bool) { return 0; }

void release_cuda_model(const Model&) {}

std::vector<int> generate_cuda(const Model&,
                               const std::vector<int>&,
                               const InferenceConfig&,
                               GenerationStats*) {
    throw std::runtime_error(
        "CUDA backend not built. Configure with -DWITH_CUDA=ON (requires clang++).");
}

#endif

}  // namespace inference
