#include "core/device.hpp"
#include <cstring>

namespace inference {

void copy_host_to_device(const void* src, void* dst, size_t bytes) {
    // CPU only for now
    std::memcpy(dst, src, bytes);
}

void copy_device_to_host(const void* src, void* dst, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void copy_device_to_device(const void* src, void* dst, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void synchronize(Device device) {
    // No-op for CPU
    (void)device;
}

}  // namespace inference