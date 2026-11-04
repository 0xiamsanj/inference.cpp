#pragma once

#include "models/model.hpp"
#include <vector>

namespace inference {

// True if the CUDA backend was built and a device is usable.
bool cuda_model_available();

// Generation timing breakdown (CUDA path). All times in milliseconds.
struct GenerationStats {
    double init_ms = 0;     // weight quantization + upload (first call)
    double prefill_ms = 0;  // processing the prompt tokens
    double decode_ms = 0;   // generating new tokens
    int generated = 0;      // number of new tokens
};

// Generate tokens entirely on the GPU. Weights are uploaded once per Model and
// cached for subsequent calls. Throws if memory is insufficient or CUDA is
// unavailable.
std::vector<int> generate_cuda(const Model& model,
                               const std::vector<int>& prompt_tokens,
                               const InferenceConfig& config,
                               GenerationStats* stats = nullptr);

// Estimated VRAM needed to run `model` on the GPU. int16_weights halves the
// weight footprint and adds one float scale per output row.
size_t cuda_required_bytes(const Model& model, bool int16_weights);

// Free the cached GPU state for a model (optional; also freed at exit).
void release_cuda_model(const Model& model);

}  // namespace inference
