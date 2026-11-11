#include "ops/cuda_backend.hpp"

#include <stdexcept>
#include <string>
#include <mutex>
#include <unordered_map>
#include <cstring>

#ifdef WITH_CUDA
#include "cuda_ptx_data.h"  // generated at build time: static const char kCudaPtx[]

#include <dlfcn.h>
#endif

namespace inference::cuda {

#ifdef WITH_CUDA

namespace {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;

constexpr CUresult CUDA_SUCCESS = 0;

using PFN_cuInit = CUresult (*)(unsigned);
using PFN_cuDeviceGet = CUresult (*)(CUdevice*, int);
using PFN_cuDeviceGetName = CUresult (*)(char*, int, CUdevice);
using PFN_cuDeviceTotalMem = CUresult (*)(size_t*, CUdevice);
using PFN_cuMemGetInfo = CUresult (*)(size_t*, size_t*);
using PFN_cuCtxCreate = CUresult (*)(CUcontext*, unsigned, CUdevice);
using PFN_cuCtxSetCurrent = CUresult (*)(CUcontext);
using PFN_cuModuleLoadData = CUresult (*)(CUmodule*, const void*);
using PFN_cuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using PFN_cuMemAlloc = CUresult (*)(CUdeviceptr*, size_t);
using PFN_cuMemFree = CUresult (*)(CUdeviceptr);
using PFN_cuMemcpyHtoD = CUresult (*)(CUdeviceptr, const void*, size_t);
using PFN_cuMemcpyDtoH = CUresult (*)(void*, CUdeviceptr, size_t);
using PFN_cuLaunchKernel = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, unsigned,
                                        CUstream, void**, void**);
using PFN_cuCtxSynchronize = CUresult (*)();
using PFN_cuGetErrorString = CUresult (*)(CUresult, const char**);

struct Driver {
    bool ok = false;
    void* lib = nullptr;
    CUcontext ctx = nullptr;
    std::string name;
    size_t total_mem = 0;

    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceGetName cuDeviceGetName = nullptr;
    PFN_cuDeviceTotalMem cuDeviceTotalMem = nullptr;
    PFN_cuMemGetInfo cuMemGetInfo = nullptr;
    PFN_cuCtxCreate cuCtxCreate = nullptr;
    PFN_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuMemAlloc cuMemAlloc = nullptr;
    PFN_cuMemFree cuMemFree = nullptr;
    PFN_cuMemcpyHtoD cuMemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH cuMemcpyDtoH = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;
    PFN_cuCtxSynchronize cuCtxSynchronize = nullptr;
    PFN_cuGetErrorString cuGetErrorString = nullptr;

    std::unordered_map<std::string, CUfunction> kernels;

    [[noreturn]] void fail(CUresult r, const char* what) const {
        const char* msg = "unknown";
        if (cuGetErrorString) cuGetErrorString(r, &msg);
        throw std::runtime_error(std::string("CUDA: ") + what + " failed: " +
                                 (msg ? msg : "?"));
    }

    void check(CUresult r, const char* what) const {
        if (r != CUDA_SUCCESS) fail(r, what);
    }

    template <typename Fn>
    Fn sym(const char* primary, const char* fallback = nullptr) {
        void* p = dlsym(lib, primary);
        if (!p && fallback) p = dlsym(lib, fallback);
        if (!p) {
            throw std::runtime_error(std::string("CUDA: missing symbol ") + primary);
        }
        return reinterpret_cast<Fn>(p);
    }
};

Driver& driver_singleton();

void init_driver(Driver& d) {
    d.lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!d.lib) throw std::runtime_error("CUDA: cannot load libcuda.so.1 (no NVIDIA driver?)");

    d.cuInit = d.sym<PFN_cuInit>("cuInit");
    d.cuDeviceGet = d.sym<PFN_cuDeviceGet>("cuDeviceGet");
    d.cuDeviceGetName = d.sym<PFN_cuDeviceGetName>("cuDeviceGetName");
    d.cuDeviceTotalMem = d.sym<PFN_cuDeviceTotalMem>("cuDeviceTotalMem_v2", "cuDeviceTotalMem");
    d.cuMemGetInfo = d.sym<PFN_cuMemGetInfo>("cuMemGetInfo_v2", "cuMemGetInfo");
    d.cuCtxCreate = d.sym<PFN_cuCtxCreate>("cuCtxCreate_v2", "cuCtxCreate");
    d.cuCtxSetCurrent = d.sym<PFN_cuCtxSetCurrent>("cuCtxSetCurrent");
    d.cuModuleLoadData = d.sym<PFN_cuModuleLoadData>("cuModuleLoadData");
    d.cuModuleGetFunction = d.sym<PFN_cuModuleGetFunction>("cuModuleGetFunction");
    d.cuMemAlloc = d.sym<PFN_cuMemAlloc>("cuMemAlloc_v2", "cuMemAlloc");
    d.cuMemFree = d.sym<PFN_cuMemFree>("cuMemFree_v2", "cuMemFree");
    d.cuMemcpyHtoD = d.sym<PFN_cuMemcpyHtoD>("cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    d.cuMemcpyDtoH = d.sym<PFN_cuMemcpyDtoH>("cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    d.cuLaunchKernel = d.sym<PFN_cuLaunchKernel>("cuLaunchKernel");
    d.cuCtxSynchronize = d.sym<PFN_cuCtxSynchronize>("cuCtxSynchronize");
    d.cuGetErrorString = d.sym<PFN_cuGetErrorString>("cuGetErrorString");

    d.check(d.cuInit(0), "cuInit");

    CUdevice dev = 0;
    d.check(d.cuDeviceGet(&dev, 0), "cuDeviceGet");

    char name[256] = {0};
    d.check(d.cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    d.name = name;

    d.check(d.cuDeviceTotalMem(&d.total_mem, dev), "cuDeviceTotalMem");

    // Note: cuDevicePrimaryCtxRetain is broken on some drivers (returns success
    // with a null context). An explicit context is reliable.
    d.check(d.cuCtxCreate(&d.ctx, 0, dev), "cuCtxCreate");
    d.check(d.cuCtxSetCurrent(d.ctx), "cuCtxSetCurrent");

    CUmodule mod = nullptr;
    d.check(d.cuModuleLoadData(&mod, kCudaPtx), "cuModuleLoadData");

    const char* names[] = {
        "gemv_kernel", "gemv_i16_kernel", "gemm_kernel", "gemm_i16_kernel",
        "rmsnorm_kernel", "silu_kernel", "mul_kernel", "add_kernel",
        "embedding_kernel", "embedding_i16_kernel",
        "embedding_batch_kernel", "embedding_batch_i16_kernel",
        "kv_write_kernel", "kv_write_batch_kernel", "rope_kernel",
        "rope_batch_kernel", "attention_kernel", "attention_batch_kernel"
    };
    for (const char* n : names) {
        CUfunction fn = nullptr;
        d.check(d.cuModuleGetFunction(&fn, mod, n), n);
        d.kernels.emplace(n, fn);
    }

    d.ok = true;
}

Driver& driver_singleton() {
    static Driver d;
    static std::once_flag once;
    std::call_once(once, [&] { init_driver(d); });
    return d;
}

Driver& drv() { return driver_singleton(); }

void make_current(Driver& d) {
    d.check(d.cuCtxSetCurrent(d.ctx), "cuCtxSetCurrent");
}

CUfunction kernel(Driver& d, const char* name) {
    auto it = d.kernels.find(name);
    if (it == d.kernels.end()) throw std::runtime_error(std::string("CUDA: unknown kernel ") + name);
    return it->second;
}

void launch3(Driver& d, const char* name,
             unsigned gx, unsigned gy, unsigned gz,
             unsigned bx, unsigned shared_bytes, void** args) {
    make_current(d);
    d.check(d.cuLaunchKernel(kernel(d, name), gx, gy, gz, bx, 1, 1,
                             shared_bytes, nullptr, args, nullptr),
            name);
}

void launch(Driver& d, const char* name,
            unsigned gx, unsigned bx, unsigned shared_bytes,
            void** args) {
    launch3(d, name, gx, 1, 1, bx, shared_bytes, args);
}

CUdeviceptr ptr_of(const Buffer& b) {
    return static_cast<CUdeviceptr>(b.ptr);
}

}  // namespace

bool available() {
    try {
        return drv().ok;
    } catch (...) {
        return false;
    }
}

std::string device_name() {
    auto& d = drv();
    return d.name;
}

size_t total_memory() { return drv().total_mem; }

size_t free_memory() {
    auto& d = drv();
    make_current(d);
    size_t free_b = 0, total_b = 0;
    d.check(d.cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo");
    return free_b;
}

void sync() {
    auto& d = drv();
    make_current(d);
    d.check(d.cuCtxSynchronize(), "cuCtxSynchronize");
}

void shutdown() {
    // Context is intentionally left alive until process exit. Freeing it while
    // buffers are still referenced would be unsafe.
}

Buffer alloc(size_t bytes) {
    auto& d = drv();
    make_current(d);
    Buffer b;
    CUdeviceptr p = 0;
    d.check(d.cuMemAlloc(&p, bytes), "cuMemAlloc");
    b.ptr = p;
    b.bytes = bytes;
    return b;
}

void release(Buffer& b) {
    if (!b.ptr) return;
    auto& d = drv();
    make_current(d);
    d.check(d.cuMemFree(static_cast<CUdeviceptr>(b.ptr)), "cuMemFree");
    b.ptr = 0;
    b.bytes = 0;
}

void upload(const Buffer& dst, const void* src, size_t bytes) {
    auto& d = drv();
    make_current(d);
    d.check(d.cuMemcpyHtoD(ptr_of(dst), src, bytes), "cuMemcpyHtoD");
}

void download(void* dst, const Buffer& src, size_t bytes) {
    auto& d = drv();
    make_current(d);
    d.check(d.cuMemcpyDtoH(dst, ptr_of(src), bytes), "cuMemcpyDtoH");
}

void gemv(const Buffer& W, const Buffer& x, const Buffer& y, int M, int N) {
    auto& d = drv();
    CUdeviceptr wp = ptr_of(W), xp = ptr_of(x), yp = ptr_of(y);
    void* args[] = {&wp, &xp, &yp, &M, &N};
    launch(d, "gemv_kernel", (unsigned)M, 256, 0, args);
}

void gemv_i16(const Buffer& W, const Buffer& row_scale, const Buffer& x,
              const Buffer& y, int M, int N) {
    auto& d = drv();
    CUdeviceptr wp = ptr_of(W), sp = ptr_of(row_scale);
    CUdeviceptr xp = ptr_of(x), yp = ptr_of(y);
    void* args[] = {&wp, &sp, &xp, &yp, &M, &N};
    launch(d, "gemv_i16_kernel", (unsigned)M, 256, 0, args);
}

void gemm(const Buffer& W, const Buffer& X, const Buffer& Y, int T, int M, int K) {
    auto& d = drv();
    CUdeviceptr wp = ptr_of(W), xp = ptr_of(X), yp = ptr_of(Y);
    void* args[] = {&wp, &xp, &yp, &T, &M, &K};
    launch3(d, "gemm_kernel",
            (unsigned)((M + 15) / 16), (unsigned)((T + 31) / 32), 1,
            512, 0, args);
}

void gemm_i16(const Buffer& W, const Buffer& row_scale, const Buffer& X,
              const Buffer& Y, int T, int M, int K) {
    auto& d = drv();
    CUdeviceptr wp = ptr_of(W), sp = ptr_of(row_scale);
    CUdeviceptr xp = ptr_of(X), yp = ptr_of(Y);
    void* args[] = {&wp, &sp, &xp, &yp, &T, &M, &K};
    launch3(d, "gemm_i16_kernel",
            (unsigned)((M + 15) / 16), (unsigned)((T + 31) / 32), 1,
            512, 0, args);
}

void rmsnorm(const Buffer& x, const Buffer& w, const Buffer& out,
             int rows, int dim, float eps) {
    auto& d = drv();
    CUdeviceptr xp = ptr_of(x), wp = ptr_of(w), op = ptr_of(out);
    void* args[] = {&xp, &wp, &op, &rows, &dim, &eps};
    launch(d, "rmsnorm_kernel", (unsigned)rows, 256, 0, args);
}

void silu(const Buffer& x, int n) {
    auto& d = drv();
    CUdeviceptr xp = ptr_of(x);
    void* args[] = {&xp, &n};
    unsigned blocks = (unsigned)((n + 255) / 256);
    launch(d, "silu_kernel", blocks, 256, 0, args);
}

void mul(const Buffer& a, const Buffer& b, int n) {
    auto& d = drv();
    CUdeviceptr ap = ptr_of(a), bp = ptr_of(b);
    void* args[] = {&ap, &bp, &n};
    unsigned blocks = (unsigned)((n + 255) / 256);
    launch(d, "mul_kernel", blocks, 256, 0, args);
}

void add(const Buffer& a, const Buffer& b, int n) {
    auto& d = drv();
    CUdeviceptr ap = ptr_of(a), bp = ptr_of(b);
    void* args[] = {&ap, &bp, &n};
    unsigned blocks = (unsigned)((n + 255) / 256);
    launch(d, "add_kernel", blocks, 256, 0, args);
}

void embedding(const Buffer& table, int id, const Buffer& out, int dim) {
    auto& d = drv();
    CUdeviceptr tp = ptr_of(table), op = ptr_of(out);
    void* args[] = {&tp, &id, &op, &dim};
    launch(d, "embedding_kernel", 1, 256, 0, args);
}

void embedding_i16(const Buffer& table, const Buffer& row_scale, int id,
                   const Buffer& out, int dim) {
    auto& d = drv();
    CUdeviceptr tp = ptr_of(table), sp = ptr_of(row_scale), op = ptr_of(out);
    void* args[] = {&tp, &sp, &id, &op, &dim};
    launch(d, "embedding_i16_kernel", 1, 256, 0, args);
}

void embedding_batch(const Buffer& table, const Buffer& ids, const Buffer& out,
                     int T, int dim) {
    auto& d = drv();
    CUdeviceptr tp = ptr_of(table), ip = ptr_of(ids), op = ptr_of(out);
    void* args[] = {&tp, &ip, &op, &T, &dim};
    launch(d, "embedding_batch_kernel", (unsigned)T, 256, 0, args);
}

void embedding_batch_i16(const Buffer& table, const Buffer& row_scale,
                         const Buffer& ids, const Buffer& out, int T, int dim) {
    auto& d = drv();
    CUdeviceptr tp = ptr_of(table), sp = ptr_of(row_scale);
    CUdeviceptr ip = ptr_of(ids), op = ptr_of(out);
    void* args[] = {&tp, &sp, &ip, &op, &T, &dim};
    launch(d, "embedding_batch_i16_kernel", (unsigned)T, 256, 0, args);
}

void kv_write(const Buffer& kc, const Buffer& vc, const Buffer& k, const Buffer& v,
              int n_heads, int head_dim, int max_seq, int pos) {
    auto& d = drv();
    CUdeviceptr kcp = ptr_of(kc), vcp = ptr_of(vc);
    CUdeviceptr kp = ptr_of(k), vp = ptr_of(v);
    void* args[] = {&kcp, &vcp, &kp, &vp, &n_heads, &head_dim, &max_seq, &pos};
    unsigned blocks = (unsigned)((n_heads * head_dim + 255) / 256);
    launch(d, "kv_write_kernel", blocks, 256, 0, args);
}

void kv_write_batch(const Buffer& kc, const Buffer& vc, const Buffer& k, const Buffer& v,
                    int n_heads, int head_dim, int max_seq, int start, int T) {
    auto& d = drv();
    CUdeviceptr kcp = ptr_of(kc), vcp = ptr_of(vc);
    CUdeviceptr kp = ptr_of(k), vp = ptr_of(v);
    void* args[] = {&kcp, &vcp, &kp, &vp, &n_heads, &head_dim, &max_seq, &start, &T};
    unsigned blocks = (unsigned)((n_heads * head_dim + 255) / 256);
    launch3(d, "kv_write_batch_kernel", blocks, (unsigned)T, 1, 256, 0, args);
}

void rope(const Buffer& q, const Buffer& k, const Buffer& table,
          int n_heads, int n_kv_heads, int head_dim, int pos) {
    auto& d = drv();
    CUdeviceptr qp = ptr_of(q), kp = ptr_of(k), tp = ptr_of(table);
    void* args[] = {&qp, &kp, &tp, &n_heads, &n_kv_heads, &head_dim, &pos};
    launch(d, "rope_kernel", (unsigned)(n_heads + n_kv_heads),
           (unsigned)(head_dim / 2), 0, args);
}

void rope_batch(const Buffer& q, const Buffer& k, const Buffer& table,
                int n_heads, int n_kv_heads, int head_dim, int start, int T) {
    auto& d = drv();
    CUdeviceptr qp = ptr_of(q), kp = ptr_of(k), tp = ptr_of(table);
    void* args[] = {&qp, &kp, &tp, &n_heads, &n_kv_heads, &head_dim, &start, &T};
    launch3(d, "rope_batch_kernel", (unsigned)(n_heads + n_kv_heads),
            (unsigned)T, 1, (unsigned)(head_dim / 2), 0, args);
}

void attention(const Buffer& q, const Buffer& kc, const Buffer& vc, const Buffer& out,
               int n_heads, int n_kv_heads, int head_dim, int max_seq, int pos,
               float scale) {
    auto& d = drv();
    CUdeviceptr qp = ptr_of(q), kcp = ptr_of(kc), vcp = ptr_of(vc), op = ptr_of(out);
    void* args[] = {&qp, &kcp, &vcp, &op, &n_heads, &n_kv_heads,
                    &head_dim, &max_seq, &pos, &scale};
    unsigned shared = (unsigned)(pos + 1) * sizeof(float);
    launch(d, "attention_kernel", (unsigned)n_heads, 128, shared, args);
}

void attention_batch(const Buffer& q, const Buffer& kc, const Buffer& vc,
                     const Buffer& out, int n_heads, int n_kv_heads, int head_dim,
                     int max_seq, int start, int T, float scale) {
    auto& d = drv();
    CUdeviceptr qp = ptr_of(q), kcp = ptr_of(kc), vcp = ptr_of(vc), op = ptr_of(out);
    void* args[] = {&qp, &kcp, &vcp, &op, &n_heads, &n_kv_heads,
                    &head_dim, &max_seq, &start, &T, &scale};
    unsigned shared = (unsigned)(start + T) * sizeof(float);
    launch3(d, "attention_batch_kernel", (unsigned)n_heads, (unsigned)T, 1,
            128, shared, args);
}

#else  // !WITH_CUDA

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "CUDA backend was not built (configure with -DWITH_CUDA=ON and clang++).");
}
}  // namespace

bool available() { return false; }
std::string device_name() { return "CUDA (not built)"; }
size_t total_memory() { return 0; }
size_t free_memory() { return 0; }
void sync() { unavailable(); }
void shutdown() {}
Buffer alloc(size_t) { unavailable(); }
void release(Buffer&) {}
void upload(const Buffer&, const void*, size_t) { unavailable(); }
void download(void*, const Buffer&, size_t) { unavailable(); }
void gemv(const Buffer&, const Buffer&, const Buffer&, int, int) { unavailable(); }
void gemv_i16(const Buffer&, const Buffer&, const Buffer&, const Buffer&, int, int) { unavailable(); }
void gemm(const Buffer&, const Buffer&, const Buffer&, int, int, int) { unavailable(); }
void gemm_i16(const Buffer&, const Buffer&, const Buffer&, const Buffer&, int, int, int) { unavailable(); }
void rmsnorm(const Buffer&, const Buffer&, const Buffer&, int, int, float) { unavailable(); }
void silu(const Buffer&, int) { unavailable(); }
void mul(const Buffer&, const Buffer&, int) { unavailable(); }
void add(const Buffer&, const Buffer&, int) { unavailable(); }
void embedding(const Buffer&, int, const Buffer&, int) { unavailable(); }
void embedding_i16(const Buffer&, const Buffer&, int, const Buffer&, int) { unavailable(); }
void embedding_batch(const Buffer&, const Buffer&, const Buffer&, int, int) { unavailable(); }
void embedding_batch_i16(const Buffer&, const Buffer&, const Buffer&, const Buffer&, int, int) { unavailable(); }
void kv_write(const Buffer&, const Buffer&, const Buffer&, const Buffer&, int, int, int, int) { unavailable(); }
void kv_write_batch(const Buffer&, const Buffer&, const Buffer&, const Buffer&, int, int, int, int, int) { unavailable(); }
void rope(const Buffer&, const Buffer&, const Buffer&, int, int, int, int) { unavailable(); }
void rope_batch(const Buffer&, const Buffer&, const Buffer&, int, int, int, int, int) { unavailable(); }
void attention(const Buffer&, const Buffer&, const Buffer&, const Buffer&,
               int, int, int, int, int, float) { unavailable(); }
void attention_batch(const Buffer&, const Buffer&, const Buffer&, const Buffer&,
                     int, int, int, int, int, int, float) { unavailable(); }

#endif

}  // namespace inference::cuda
