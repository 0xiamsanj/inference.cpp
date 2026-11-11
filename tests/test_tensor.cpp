#include "core/tensor.hpp"
#include "gguf/quantization.hpp"
#include "ops/ops.hpp"
#include "ops/cuda_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace inference;

TEST(TensorTest, BasicCreation) {
    Tensor t(DataType::F32, {2, 3});
    EXPECT_EQ(t.shape(), (Shape{2, 3}));
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dtype(), DataType::F32);
}

TEST(TensorTest, FillAndAccess) {
    Tensor t(DataType::F32, {3});
    t.fill(5.0f);
    EXPECT_FLOAT_EQ(t.as<float>()[0], 5.0f);
    EXPECT_FLOAT_EQ(t.as<float>()[2], 5.0f);
}

TEST(TensorTest, Reshape) {
    Tensor t(DataType::F32, {2, 3});
    t.reshape({3, 2});
    EXPECT_EQ(t.shape(), (Shape{3, 2}));
    EXPECT_EQ(t.numel(), 6);
}

TEST(TensorTest, Clone) {
    Tensor t1(DataType::F32, {2, 2});
    t1.as<float>()[0] = 1;
    t1.as<float>()[1] = 2;
    t1.as<float>()[2] = 3;
    t1.as<float>()[3] = 4;

    Tensor t2 = t1.clone();
    EXPECT_EQ(t2.as<float>()[0], 1);
    EXPECT_EQ(t2.as<float>()[3], 4);

    // Modify clone, original unchanged
    t2.as<float>()[0] = 99;
    EXPECT_EQ(t1.as<float>()[0], 1);
}

TEST(QuantizationTest, Q4_0_Roundtrip) {
    // Test data
    float original[32];
    for (int i = 0; i < 32; ++i) original[i] = (i - 16) * 0.5f;

    // GGUF Q4_0 block is 18 bytes (fp16 scale + 16 packed bytes)
    uint8_t block[18];
    quantize_q4_0(original, block, 32);

    // Dequantize
    float result[32];
    dequantize_q4_0(block, result, 32);

    // Check approximate recovery
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(original[i], result[i], 0.6f) << "Index " << i;
    }
}

TEST(QuantizationTest, Q8_0_Roundtrip) {
    float original[32];
    for (int i = 0; i < 32; ++i) original[i] = (i - 16) * 0.1f;

    // GGUF Q8_0 block is 34 bytes (fp16 scale + 32 int8)
    uint8_t block[34];
    quantize_q8_0(original, block, 32);

    float result[32];
    dequantize_q8_0(block, result, 32);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(original[i], result[i], 0.05f) << "Index " << i;
    }
}

TEST(OpsTest, Gemm) {
    Tensor A(DataType::F32, {2, 3});
    Tensor B(DataType::F32, {3, 4});
    Tensor C(DataType::F32, {2, 4});

    // A = [[1,2,3],[4,5,6]]
    float a_data[] = {1,2,3, 4,5,6};
    std::memcpy(A.data(), a_data, sizeof(a_data));

    // B = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    float b_data[] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    std::memcpy(B.data(), b_data, sizeof(b_data));

    gemm(A, B, C, false);

    // C should equal A[:,:3] (first 3 cols of A)
    EXPECT_FLOAT_EQ(C.as<float>()[0], 1);
    EXPECT_FLOAT_EQ(C.as<float>()[1], 2);
    EXPECT_FLOAT_EQ(C.as<float>()[2], 3);
    EXPECT_FLOAT_EQ(C.as<float>()[3], 0);

    EXPECT_FLOAT_EQ(C.as<float>()[4], 4);
    EXPECT_FLOAT_EQ(C.as<float>()[5], 5);
    EXPECT_FLOAT_EQ(C.as<float>()[6], 6);
    EXPECT_FLOAT_EQ(C.as<float>()[7], 0);
}

TEST(OpsTest, GemmTransB) {
    Tensor A(DataType::F32, {2, 3});
    Tensor B(DataType::F32, {4, 3});  // Transposed
    Tensor C(DataType::F32, {2, 4});

    float a_data[] = {1,2,3, 4,5,6};
    std::memcpy(A.data(), a_data, sizeof(a_data));

    // B^T = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    // So B = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    float b_data[] = {1,0,0, 0,1,0, 0,0,1, 0,0,0};
    std::memcpy(B.data(), b_data, sizeof(b_data));

    gemm(A, B, C, true);  // C = A @ B^T

    EXPECT_FLOAT_EQ(C.as<float>()[0], 1);
    EXPECT_FLOAT_EQ(C.as<float>()[1], 2);
    EXPECT_FLOAT_EQ(C.as<float>()[2], 3);
    EXPECT_FLOAT_EQ(C.as<float>()[3], 0);
}

TEST(OpsTest, Softmax) {
    Tensor x(DataType::F32, {3});
    x.as<float>()[0] = 1.0f;
    x.as<float>()[1] = 2.0f;
    x.as<float>()[2] = 3.0f;

    softmax_(x);

    float sum = 0;
    for (int i = 0; i < 3; ++i) sum += x.as<float>()[i];
    EXPECT_NEAR(sum, 1.0f, 1e-6);

    // Larger input -> larger probability
    EXPECT_GT(x.as<float>()[2], x.as<float>()[1]);
    EXPECT_GT(x.as<float>()[1], x.as<float>()[0]);
}

TEST(OpsTest, RMSNorm) {
    Tensor x(DataType::F32, {2, 4});
    Tensor weight(DataType::F32, {4});
    Tensor out(DataType::F32, {2, 4});

    // x = [[1,2,3,4], [2,4,6,8]]
    float x_data[] = {1,2,3,4, 2,4,6,8};
    std::memcpy(x.data(), x_data, sizeof(x_data));

    // weight = ones
    weight.fill(1.0f);

    rms_norm(x, weight, out);

    // Check normalization
    for (int b = 0; b < 2; ++b) {
        float sum_sq = 0;
        for (int d = 0; d < 4; ++d) {
            float v = out.as<float>()[b * 4 + d];
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / 4);
        EXPECT_NEAR(rms, 1.0f, 1e-5);
    }
}

TEST(OpsTest, RoPE) {
    Tensor x(DataType::F32, {1, 2, 4});  // [B=1, T=2, D=4]
    for (int i = 0; i < 8; ++i) x.as<float>()[i] = 1.0f;

    rope_(x, 0, -1, 10000.0f);

    // After RoPE, values should be rotated
    // Just check it runs and values changed
    bool changed = false;
    for (int i = 0; i < 8; ++i) {
        if (x.as<float>()[i] != 1.0f) changed = true;
    }
    EXPECT_TRUE(changed);
}

TEST(OpsTest, SiLU) {
    Tensor x(DataType::F32, {4});
    x.as<float>()[0] = -1;
    x.as<float>()[1] = 0;
    x.as<float>()[2] = 1;
    x.as<float>()[3] = 2;

    silu_(x);

    // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.269
    EXPECT_NEAR(x.as<float>()[0], -0.2689, 1e-3);
    // SiLU(0) = 0
    EXPECT_FLOAT_EQ(x.as<float>()[1], 0);
    // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
    EXPECT_NEAR(x.as<float>()[2], 0.7311, 1e-3);
}

TEST(OpsTest, Sampling) {
    Tensor logits(DataType::F32, {10});
    for (int i = 0; i < 10; ++i) logits.as<float>()[i] = i * 0.1f;

    // Argmax should pick last (highest logit)
    int idx = sample_argmax(logits);
    EXPECT_EQ(idx, 9);

    // Top-k with k=1 should also pick last
    idx = sample_top_k_top_p(logits, 1, 1.0f, 1.0f);
    EXPECT_EQ(idx, 9);

    // Temperature 0 = argmax
    idx = sample_top_k_top_p(logits, 10, 1.0f, 0.0f);
    EXPECT_EQ(idx, 9);
}

TEST(ChatTemplateTest, Zephyr) {
    Tokenizer tok;
    tok.set_chat_template(
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + eos_token }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}");
    EXPECT_EQ(tok.chat_format(), ChatFormat::Zephyr);
    EXPECT_TRUE(tok.chat_needs_bos());
    std::string prompt = tok.apply_chat_template({{"user", "hi"}}, true);
    EXPECT_EQ(prompt, "<|user|>\nhi</s>\n<|assistant|>\n");
}

TEST(ChatTemplateTest, Llama3) {
    Tokenizer tok;
    tok.set_chat_template(
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
        " + message['content'] + '<|eot_id|>' }}{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}");
    EXPECT_EQ(tok.chat_format(), ChatFormat::Llama3);
    EXPECT_FALSE(tok.chat_needs_bos());
    std::string prompt = tok.apply_chat_template({{"user", "hi"}}, true);
    EXPECT_NE(prompt.find("<|begin_of_text|>"), std::string::npos);
    EXPECT_NE(prompt.find("<|start_header_id|>user<|end_header_id|>"), std::string::npos);
    EXPECT_NE(prompt.find("hi<|eot_id|>"), std::string::npos);
    EXPECT_NE(prompt.find("<|start_header_id|>assistant<|end_header_id|>"), std::string::npos);
}

TEST(ChatTemplateTest, ChatML) {
    Tokenizer tok;
    tok.set_chat_template(
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n'"
        " + message['content'] + '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}");
    EXPECT_EQ(tok.chat_format(), ChatFormat::ChatML);
    std::string prompt = tok.apply_chat_template(
        {{"system", "be nice"}, {"user", "hi"}}, true);
    EXPECT_EQ(prompt,
              "<|im_start|>system\nbe nice<|im_end|>\n"
              "<|im_start|>user\nhi<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(ChatTemplateTest, MistralAndGemma) {
    Tokenizer mistral;
    mistral.set_chat_template(
        "{% for message in messages %}{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% endfor %}");
    EXPECT_EQ(mistral.chat_format(), ChatFormat::Mistral);
    EXPECT_EQ(mistral.apply_chat_template({{"user", "hi"}}, true),
              "<s>[INST] hi [/INST]");

    Tokenizer gemma;
    gemma.set_chat_template(
        "{% for message in messages %}{% if message['role'] == 'user' %}"
        "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}");
    EXPECT_EQ(gemma.chat_format(), ChatFormat::Gemma);
    EXPECT_EQ(gemma.apply_chat_template({{"user", "hi"}}, true),
              "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n");
}

TEST(ChatTemplateTest, GenericFallback) {
    Tokenizer tok;
    EXPECT_EQ(tok.chat_format(), ChatFormat::Generic);
    EXPECT_EQ(tok.apply_chat_template({{"user", "hi"}}, true),
              "User: hi\nAssistant:");
}

#ifdef WITH_CUDA
TEST(CudaTest, GemvSmall) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    // W = [[1,2],[3,4]], x = [5,6]  =>  y = [17, 39]
    Buffer W = alloc(4 * sizeof(float));
    Buffer x = alloc(2 * sizeof(float));
    Buffer y = alloc(2 * sizeof(float));
    float w[4] = {1, 2, 3, 4};
    float xv[2] = {5, 6};
    upload(W, w, sizeof(w));
    upload(x, xv, sizeof(xv));

    gemv(W, x, y, 2, 2);
    inference::cuda::sync();

    float out[2] = {0, 0};
    download(out, y, sizeof(out));
    EXPECT_FLOAT_EQ(out[0], 17.0f);
    EXPECT_FLOAT_EQ(out[1], 39.0f);

    release(W);
    release(x);
    release(y);
}

TEST(CudaTest, RmsNormSmall) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    // x = [1,2,3,4], w = ones, eps = 0 -> out = x / rms(x)
    Buffer x = alloc(4 * sizeof(float));
    Buffer w = alloc(4 * sizeof(float));
    Buffer out_b = alloc(4 * sizeof(float));
    float xv[4] = {1, 2, 3, 4};
    float wv[4] = {1, 1, 1, 1};
    upload(x, xv, sizeof(xv));
    upload(w, wv, sizeof(wv));

    rmsnorm(x, w, out_b, 1, 4, 0.0f);
    inference::cuda::sync();

    float out[4] = {0, 0, 0, 0};
    download(out, out_b, sizeof(out));
    const float rms = std::sqrt(30.0f / 4.0f);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], xv[i] / rms, 1e-4f);
    }

    release(x);
    release(w);
    release(out_b);
}
TEST(CudaTest, AttentionLongGQA) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    const int H = 2, Hkv = 1, HD = 4, MAX = 256, POS = 150;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    std::vector<float> q(H * HD), kc(Hkv * MAX * HD), vc(Hkv * MAX * HD);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(0.1f * i);
    for (size_t i = 0; i < kc.size(); ++i) kc[i] = std::cos(0.07f * i);
    for (size_t i = 0; i < vc.size(); ++i) vc[i] = std::sin(0.03f * i) - 0.2f;

    // CPU reference (GQA: both query heads use kv head 0)
    std::vector<float> ref(H * HD, 0.0f);
    for (int h = 0; h < H; ++h) {
        std::vector<float> scores(POS + 1);
        float mx = -1e30f;
        for (int s = 0; s <= POS; ++s) {
            float dot = 0;
            for (int d = 0; d < HD; ++d) dot += q[h * HD + d] * kc[s * HD + d];
            scores[s] = dot * scale;
            mx = std::max(mx, scores[s]);
        }
        float sum = 0;
        for (int s = 0; s <= POS; ++s) { scores[s] = std::exp(scores[s] - mx); sum += scores[s]; }
        for (int s = 0; s <= POS; ++s) scores[s] /= sum;
        for (int d = 0; d < HD; ++d) {
            float acc = 0;
            for (int s = 0; s <= POS; ++s) acc += scores[s] * vc[s * HD + d];
            ref[h * HD + d] = acc;
        }
    }

    Buffer qb = alloc(q.size() * sizeof(float));
    Buffer kb = alloc(kc.size() * sizeof(float));
    Buffer vb = alloc(vc.size() * sizeof(float));
    Buffer ob = alloc(ref.size() * sizeof(float));
    upload(qb, q.data(), q.size() * sizeof(float));
    upload(kb, kc.data(), kc.size() * sizeof(float));
    upload(vb, vc.data(), vc.size() * sizeof(float));

    attention(qb, kb, vb, ob, H, Hkv, HD, MAX, POS, scale);
    inference::cuda::sync();

    std::vector<float> out(ref.size(), 0.0f);
    download(out.data(), ob, out.size() * sizeof(float));
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 2e-4f) << "index " << i;
    }

    release(qb); release(kb); release(vb); release(ob);
}
TEST(CudaTest, GemvInt16) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    // W_effective = [[1,2],[3,4]] encoded as q=[[1,2],[6,8]] with scales [1, 0.5]
    // x = [5,6] -> y = [17, 39]
    int16_t q[4] = {1, 2, 6, 8};
    float scales[2] = {1.0f, 0.5f};
    float xv[2] = {5, 6};

    Buffer W = alloc(4 * sizeof(int16_t));
    Buffer sc = alloc(2 * sizeof(float));
    Buffer x = alloc(2 * sizeof(float));
    Buffer y = alloc(2 * sizeof(float));
    upload(W, q, sizeof(q));
    upload(sc, scales, sizeof(scales));
    upload(x, xv, sizeof(xv));

    gemv_i16(W, sc, x, y, 2, 2);
    inference::cuda::sync();

    float out[2] = {0, 0};
    download(out, y, sizeof(out));
    EXPECT_FLOAT_EQ(out[0], 17.0f);
    EXPECT_FLOAT_EQ(out[1], 39.0f);

    release(W); release(sc); release(x); release(y);
}

TEST(CudaTest, GemmInt16Batch) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    // W_eff = [[1,2],[3,4]] as q=[[1,2],[6,8]] with scales [1, 0.5]
    // X = [[1,0],[0,1]]  ->  Y = [[1,3],[2,4]]
    int16_t q[4] = {1, 2, 6, 8};
    float scales[2] = {1.0f, 0.5f};
    float xv[4] = {1, 0, 0, 1};

    Buffer W = alloc(4 * sizeof(int16_t));
    Buffer sc = alloc(2 * sizeof(float));
    Buffer X = alloc(4 * sizeof(float));
    Buffer Y = alloc(4 * sizeof(float));
    upload(W, q, sizeof(q));
    upload(sc, scales, sizeof(scales));
    upload(X, xv, sizeof(xv));

    gemm_i16(W, sc, X, Y, /*T=*/2, /*M=*/2, /*K=*/2);
    inference::cuda::sync();

    float out[4] = {0, 0, 0, 0};
    download(out, Y, sizeof(out));
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);

    release(W); release(sc); release(X); release(Y);
}

TEST(CudaTest, AttentionBatch) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    const int H = 2, Hkv = 1, HD = 4, MAX = 8, T = 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    std::vector<float> q(T * H * HD), kc(Hkv * MAX * HD), vc(Hkv * MAX * HD);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(0.11f * i);
    for (size_t i = 0; i < kc.size(); ++i) kc[i] = std::cos(0.07f * i);
    for (size_t i = 0; i < vc.size(); ++i) vc[i] = std::sin(0.03f * i) - 0.2f;

    std::vector<float> ref(T * H * HD, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            std::vector<float> scores(t + 1);
            float mx = -1e30f;
            for (int s = 0; s <= t; ++s) {
                float dot = 0;
                for (int d = 0; d < HD; ++d) {
                    dot += q[(t * H + h) * HD + d] * kc[s * HD + d];
                }
                scores[s] = dot * scale;
                mx = std::max(mx, scores[s]);
            }
            float sum = 0;
            for (int s = 0; s <= t; ++s) { scores[s] = std::exp(scores[s] - mx); sum += scores[s]; }
            for (int s = 0; s <= t; ++s) scores[s] /= sum;
            for (int d = 0; d < HD; ++d) {
                float acc = 0;
                for (int s = 0; s <= t; ++s) acc += scores[s] * vc[s * HD + d];
                ref[(t * H + h) * HD + d] = acc;
            }
        }
    }

    Buffer qb = alloc(q.size() * sizeof(float));
    Buffer kb = alloc(kc.size() * sizeof(float));
    Buffer vb = alloc(vc.size() * sizeof(float));
    Buffer ob = alloc(ref.size() * sizeof(float));
    upload(qb, q.data(), q.size() * sizeof(float));
    upload(kb, kc.data(), kc.size() * sizeof(float));
    upload(vb, vc.data(), vc.size() * sizeof(float));

    attention_batch(qb, kb, vb, ob, H, Hkv, HD, MAX, /*start=*/0, T, scale);
    inference::cuda::sync();

    std::vector<float> out(ref.size(), 0.0f);
    download(out.data(), ob, out.size() * sizeof(float));
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 2e-4f) << "index " << i;
    }

    release(qb); release(kb); release(vb); release(ob);
}

TEST(CudaTest, EmbeddingInt16) {
    if (!inference::cuda::available()) GTEST_SKIP() << "no CUDA device";
    using namespace inference::cuda;

    // Two rows of 3 values, scales [1, 2]; row 1 -> [2,4,6]
    int16_t table[6] = {1, 2, 3, 1, 2, 3};
    float scales[2] = {1.0f, 2.0f};

    Buffer t = alloc(sizeof(table));
    Buffer sc = alloc(sizeof(scales));
    Buffer out_b = alloc(3 * sizeof(float));
    upload(t, table, sizeof(table));
    upload(sc, scales, sizeof(scales));

    embedding_i16(t, sc, 1, out_b, 3);
    inference::cuda::sync();

    float out[3] = {0, 0, 0};
    download(out, out_b, sizeof(out));
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 6.0f);

    release(t); release(sc); release(out_b);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}