#include "models/model.hpp"
#include "gguf/gguf_parser.hpp"
#include "tokenizer/tokenizer.hpp"
#include "core/tensor.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace inference;

static void top_tokens(const Tokenizer& tok, const float* logits, int vocab, int k) {
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    for (int i = 0; i < k; ++i) {
        int id = idx[i];
        std::string piece = tok.decode({id});
        std::printf("  #%d id=%d logit=%.4f piece='%s'\n", i, id, logits[id], piece.c_str());
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [prompt]\n", argv[0]); return 1; }
    std::string prompt = argc > 2 ? argv[2] : "Once upon a time";

    GGUFParser parser(argv[1]);
    auto weights = load_model_weights(parser, Device::cpu());

    auto stats = [](const char* name, const Tensor& t) {
        const float* p = t.as<float>();
        float mn = 1e30f, mx = -1e30f;
        double sum = 0, sumsq = 0;
        int64_t n = t.numel();
        bool nan = false;
        for (int64_t i = 0; i < n; ++i) {
            float v = p[i];
            if (std::isnan(v) || std::isinf(v)) { nan = true; if (nan) break; }
            mn = std::min(mn, v); mx = std::max(mx, v);
            sum += v; sumsq += double(v) * v;
        }
        double mean = sum / n;
        double var = sumsq / n - mean * mean;
        std::printf("%-24s shape=[", name);
        for (size_t i = 0; i < t.shape().size(); ++i) std::printf("%s%lld", i?",":"", (long long)t.shape()[i]);
        std::printf("] min=%.4f max=%.4f mean=%.5f std=%.5f%s\n", mn, mx, mean, std::sqrt(std::max(0.0,var)), nan?" NAN!":"");
    };

    stats("tok_embeddings", weights.tok_embeddings);
    stats("output", weights.output);
    stats("output_norm", weights.output_norm);
    stats("blk0.wq", weights.layers[0].wq);
    stats("blk0.w1", weights.layers[0].w1);
    stats("blk0.attn_norm", weights.layers[0].attn_norm);

    Model model(std::move(weights));
    Tokenizer tok;
    tok.load_from_gguf(parser);

    auto ids = tok.encode(prompt, true, false);
    std::printf("\nprompt='%s' -> %zu tokens:", prompt.c_str(), ids.size());
    for (int id : ids) std::printf(" %d", id);
    std::printf("\ndecoded back='%s'\n", tok.decode(ids).c_str());

    Tensor in(DataType::I32, {1, (int)ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) in.as<int32_t>()[i] = ids[i];

    KVCache cache(model.n_layers(), model.max_seq_len(), model.n_heads(), model.head_dim());
    Tensor logits = model.forward(in, &cache, 0);
    int V = model.vocab_size();
    const float* last = logits.as<float>() + (ids.size() - 1) * V;

    float mn = 1e30f, mx = -1e30f; double sum=0;
    for (int i = 0; i < V; ++i) { mn=std::min(mn,last[i]); mx=std::max(mx,last[i]); sum+=last[i]; }
    std::printf("\nlogits: min=%.4f max=%.4f mean=%.5f\n", mn, mx, sum/V);
    std::printf("top 10 at last prompt position:\n");
    top_tokens(tok, last, V, 10);
    return 0;
}