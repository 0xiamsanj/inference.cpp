#pragma once

#include "core/tensor.hpp"

namespace inference {

// Device-specific operations
// Currently CPU-only, CUDA support planned

// Memory copy between devices
void copy_host_to_device(const void* src, void* dst, size_t bytes);
void copy_device_to_host(const void* src, void* dst, size_t bytes);
void copy_device_to_device(const void* src, void* dst, size_t bytes);

// Device synchronization
void synchronize(Device device);

}  // namespace inference