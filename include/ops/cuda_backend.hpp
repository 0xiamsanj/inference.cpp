#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace inference::cuda {

// Opaque device buffer handle (no CUDA headers needed by callers).
struct Buffer {
    uint64_t ptr = 0;
    size_t bytes = 0;
    bool valid() const { return ptr != 0; }
};

// Backend lifecycle / info
bool available();                 // attempts one-time init
std::string device_name();
size_t total_memory();
size_t free_memory();
void sync();
void shutdown();

// Memory
Buffer alloc(size_t bytes);
void release(Buffer& b);
void upload(const Buffer& dst, const void* src, size_t bytes);
void download(void* dst, const Buffer& src, size_t bytes);

// Ops (all tensors are FP32 and row-major)
void gemv(const Buffer& W, const Buffer& x, const Buffer& y, int M, int N);
// INT16 weights with one float scale per output row.
void gemv_i16(const Buffer& W, const Buffer& row_scale, const Buffer& x,
              const Buffer& y, int M, int N);
// Batched: Y[T, M] = X[T, K] @ W[M, K]^T
void gemm(const Buffer& W, const Buffer& X, const Buffer& Y, int T, int M, int K);
void gemm_i16(const Buffer& W, const Buffer& row_scale, const Buffer& X,
              const Buffer& Y, int T, int M, int K);
void rmsnorm(const Buffer& x, const Buffer& w, const Buffer& out,
             int rows, int dim, float eps);
void silu(const Buffer& x, int n);
void mul(const Buffer& a, const Buffer& b, int n);
void add(const Buffer& a, const Buffer& b, int n);
void embedding(const Buffer& table, int id, const Buffer& out, int dim);
void embedding_i16(const Buffer& table, const Buffer& row_scale, int id,
                   const Buffer& out, int dim);
void embedding_batch(const Buffer& table, const Buffer& ids, const Buffer& out,
                     int T, int dim);
void embedding_batch_i16(const Buffer& table, const Buffer& row_scale,
                         const Buffer& ids, const Buffer& out, int T, int dim);
void kv_write(const Buffer& kc, const Buffer& vc, const Buffer& k, const Buffer& v,
              int n_heads, int head_dim, int max_seq, int pos);
void kv_write_batch(const Buffer& kc, const Buffer& vc, const Buffer& k, const Buffer& v,
                    int n_heads, int head_dim, int max_seq, int start, int T);
void rope(const Buffer& q, const Buffer& k, const Buffer& table,
          int n_heads, int n_kv_heads, int head_dim, int pos);
void rope_batch(const Buffer& q, const Buffer& k, const Buffer& table,
                int n_heads, int n_kv_heads, int head_dim, int start, int T);
void attention(const Buffer& q, const Buffer& kc, const Buffer& vc, const Buffer& out,
               int n_heads, int n_kv_heads, int head_dim, int max_seq, int pos,
               float scale);
void attention_batch(const Buffer& q, const Buffer& kc, const Buffer& vc,
                     const Buffer& out, int n_heads, int n_kv_heads, int head_dim,
                     int max_seq, int start, int T, float scale);

}  // namespace inference::cuda
