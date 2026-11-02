#pragma once

#include "core/tensor.hpp"
#include <vector>

namespace inference {

struct SamplingParams {
    float temperature = 1.0f;
    int top_k = 0;        // 0 = disabled
    float top_p = 1.0f;   // 1.0 = disabled
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    int seed = -1;
};

// Sampler interface
class Sampler {
public:
    explicit Sampler(const SamplingParams& params);
    virtual ~Sampler() = default;

    // Sample next token from logits
    virtual int sample(const Tensor& logits, const std::vector<int>& previous_tokens) = 0;

    void set_params(const SamplingParams& params) { params_ = params; }

protected:
    SamplingParams params_;
};

// Greedy sampler (argmax)
class GreedySampler : public Sampler {
public:
    using Sampler::Sampler;
    int sample(const Tensor& logits, const std::vector<int>&) override;
};

// Top-k + Top-p (nucleus) sampler
class TopKTopPSampler : public Sampler {
public:
    using Sampler::Sampler;
    int sample(const Tensor& logits, const std::vector<int>& previous_tokens) override;
};

// Factory
std::unique_ptr<Sampler> create_sampler(const SamplingParams& params);

}  // namespace inference