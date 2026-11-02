#include "sampling/sampler.hpp"
#include "ops/ops.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <memory>

namespace inference {

Sampler::Sampler(const SamplingParams& params) : params_(params) {}

int GreedySampler::sample(const Tensor& logits, const std::vector<int>&) {
    return sample_argmax(logits);
}

int TopKTopPSampler::sample(const Tensor& logits, const std::vector<int>& previous_tokens) {
    if (logits.dtype() != DataType::F32) throw std::runtime_error("sample: only F32");
    if (logits.shape().size() != 1) throw std::runtime_error("sample: logits must be 1D");

    int vocab_size = logits.shape()[0];
    const float* logits_ptr = logits.as<float>();

    // Create index array
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    // Apply temperature
    std::vector<float> probs(vocab_size);
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits_ptr[i]);
    }

    // Repetition penalty
    if (params_.repetition_penalty != 1.0f) {
        std::vector<float> modified_logits(logits_ptr, logits_ptr + vocab_size);
        for (int token : previous_tokens) {
            if (token >= 0 && token < vocab_size) {
                if (modified_logits[token] > 0) {
                    modified_logits[token] /= params_.repetition_penalty;
                } else {
                    modified_logits[token] *= params_.repetition_penalty;
                }
            }
        }
        // Recompute max
        max_logit = -INFINITY;
        for (int i = 0; i < vocab_size; ++i) {
            max_logit = std::max(max_logit, modified_logits[i]);
        }
        // Softmax with modified logits
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp((modified_logits[i] - max_logit) / params_.temperature);
            sum_exp += probs[i];
        }
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum_exp;
        }
    } else {
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp((logits_ptr[i] - max_logit) / params_.temperature);
            sum_exp += probs[i];
        }
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum_exp;
        }
    }

    // Min-p: filter tokens with prob < min_p * max_prob
    if (params_.min_p > 0.0f) {
        float max_prob = *std::max_element(probs.begin(), probs.end());
        float threshold = params_.min_p * max_prob;
        for (int i = 0; i < vocab_size; ++i) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
        // Renormalize
        float sum = 0.0f;
        for (float p : probs) sum += p;
        if (sum > 0) {
            for (float& p : probs) p /= sum;
        }
    }

    // Top-k
    if (params_.top_k > 0 && params_.top_k < vocab_size) {
        std::nth_element(indices.begin(), indices.begin() + params_.top_k, indices.end(),
                        [&](int a, int b) { return probs[a] > probs[b]; });
        indices.resize(params_.top_k);
    }

    // Top-p (nucleus)
    if (params_.top_p < 1.0f && params_.top_p > 0.0f) {
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        size_t cutoff = indices.size();
        for (size_t i = 0; i < indices.size(); ++i) {
            cumsum += probs[indices[i]];
            if (cumsum >= params_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        indices.resize(cutoff);
    }

    // Renormalize
    float sum = 0.0f;
    for (int idx : indices) sum += probs[idx];
    if (sum > 0) {
        for (int idx : indices) probs[idx] /= sum;
    }

    // Sample
    std::mt19937 gen(params_.seed >= 0 ? params_.seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen);

    float cumsum = 0.0f;
    for (int idx : indices) {
        cumsum += probs[idx];
        if (r < cumsum) return idx;
    }

    return indices.back();
}

std::unique_ptr<Sampler> create_sampler(const SamplingParams& params) {
    if (params.temperature <= 0.0f) {
        return std::make_unique<GreedySampler>(params);
    }
    return std::make_unique<TopKTopPSampler>(params);
}

}  // namespace inference