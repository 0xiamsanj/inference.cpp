// Self-contained CUDA device kernels.
//
// Compiled to PTX by clang (no CUDA toolkit/headers required):
//   clang++ -x cuda --cuda-device-only -nocudainc -nocudalib \
//           --cuda-gpu-arch=sm_75 -O3 -S cuda_kernels.cu -o cuda_kernels.ptx
//
// The PTX is loaded at runtime through the CUDA driver API (libcuda.so.1).

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define DEV_INLINE __device__ __attribute__((always_inline)) inline

// ---- Minimal math built on PTX intrinsics (no libdevice, no headers) ----
DEV_INLINE float f_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
DEV_INLINE float f_exp(float x) { return f_exp2(x * 1.4426950408889634f); }
DEV_INLINE float f_sqrt(float x) {
    float r;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Hardware special registers (clang NVVM builtins).
DEV_INLINE unsigned tid_x() { return (unsigned)__nvvm_read_ptx_sreg_tid_x(); }
DEV_INLINE unsigned ntid_x() { return (unsigned)__nvvm_read_ptx_sreg_ntid_x(); }
DEV_INLINE unsigned ctaid_x() { return (unsigned)__nvvm_read_ptx_sreg_ctaid_x(); }
DEV_INLINE unsigned ctaid_y() { return (unsigned)__nvvm_read_ptx_sreg_ctaid_y(); }
DEV_INLINE unsigned gridDim_x_helper() { return (unsigned)__nvvm_read_ptx_sreg_nctaid_x(); }

DEV_INLINE float block_reduce_sum(float val, float* sh) {
    unsigned tid = tid_x();
    unsigned nt = ntid_x();
    sh[tid] = val;
    __syncthreads();
    for (unsigned off = nt / 2; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }
    return sh[0];
}

DEV_INLINE float block_reduce_max(float val, float* sh) {
    unsigned tid = tid_x();
    unsigned nt = ntid_x();
    sh[tid] = val;
    __syncthreads();
    for (unsigned off = nt / 2; off > 0; off >>= 1) {
        if (tid < off) sh[tid] = sh[tid] > sh[tid + off] ? sh[tid] : sh[tid + off];
        __syncthreads();
    }
    return sh[0];
}

// ============================================================================
// y[m] = dot(W[m, :], x)   W: [M, N] row-major
// One block per output row.
// ============================================================================
extern "C" __global__ void gemv_kernel(const float* W, const float* x, float* y,
                                       int M, int N) {
    __shared__ float red[256];
    int row = (int)ctaid_x();
    if (row >= M) return;
    unsigned tid = tid_x();
    float sum = 0.0f;
    const float* w = W + (long long)row * N;
    for (int k = (int)tid; k < N; k += (int)ntid_x()) {
        sum += w[k] * x[k];
    }
    sum = block_reduce_sum(sum, red);
    if (tid == 0) y[row] = sum;
}

// ============================================================================
// INT16 GEMV: y[m] = scale[m] * dot(W[m, :], x)
// W is int16 with one float scale per output row.
// ============================================================================
extern "C" __global__ void gemv_i16_kernel(const short* W, const float* row_scale,
                                           const float* x, float* y,
                                           int M, int N) {
    __shared__ float red[256];
    int row = (int)ctaid_x();
    if (row >= M) return;
    unsigned tid = tid_x();
    const short* w = W + (long long)row * N;
    float sum = 0.0f;
    for (int k = (int)tid; k < N; k += (int)ntid_x()) {
        sum += (float)w[k] * x[k];
    }
    sum = block_reduce_sum(sum, red);
    if (tid == 0) y[row] = sum * row_scale[row];
}

// ============================================================================
// RMSNorm over the last dimension. One block per row.
// out = x * rsqrt(mean(x^2)+eps) * w
// ============================================================================
extern "C" __global__ void rmsnorm_kernel(const float* x, const float* w,
                                          float* out, int rows, int dim, float eps) {
    __shared__ float red[256];
    int row = (int)ctaid_x();
    if (row >= rows) return;
    unsigned tid = tid_x();
    const float* xr = x + (long long)row * dim;
    float* orow = out + (long long)row * dim;
    float ss = 0.0f;
    for (int i = (int)tid; i < dim; i += (int)ntid_x()) ss += xr[i] * xr[i];
    ss = block_reduce_sum(ss, red);
    float inv = 1.0f / f_sqrt(ss / (float)dim + eps);
    for (int i = (int)tid; i < dim; i += (int)ntid_x()) {
        orow[i] = xr[i] * inv * w[i];
    }
}

// ============================================================================
// Element-wise kernels
// ============================================================================
extern "C" __global__ void silu_kernel(float* x, int n) {
    for (int i = (int)(ctaid_x() * ntid_x() + tid_x()); i < n;
         i += (int)(ntid_x() * (unsigned)gridDim_x_helper())) {
        x[i] = x[i] / (1.0f + f_exp(-x[i]));
    }
}

extern "C" __global__ void mul_kernel(float* a, const float* b, int n) {
    for (int i = (int)(ctaid_x() * ntid_x() + tid_x()); i < n;
         i += (int)(ntid_x() * gridDim_x_helper())) {
        a[i] *= b[i];
    }
}

extern "C" __global__ void add_kernel(float* a, const float* b, int n) {
    for (int i = (int)(ctaid_x() * ntid_x() + tid_x()); i < n;
         i += (int)(ntid_x() * gridDim_x_helper())) {
        a[i] += b[i];
    }
}

// ============================================================================
// Embedding lookup: out = table[id, :]
// ============================================================================
extern "C" __global__ void embedding_kernel(const float* table, int id,
                                            float* out, int dim) {
    const float* row = table + (long long)id * dim;
    for (int i = (int)tid_x(); i < dim; i += (int)ntid_x()) out[i] = row[i];
}

// INT16 embedding lookup with one scale per vocabulary row.
extern "C" __global__ void embedding_i16_kernel(const short* table, const float* row_scale,
                                                int id, float* out, int dim) {
    const short* row = table + (long long)id * dim;
    float s = row_scale[id];
    for (int i = (int)tid_x(); i < dim; i += (int)ntid_x()) out[i] = (float)row[i] * s;
}

// ============================================================================
// Write k/v into the KV cache at position `pos`.
// cache layout: [n_heads][max_seq][head_dim]
// ============================================================================
extern "C" __global__ void kv_write_kernel(float* kc, float* vc,
                                           const float* k, const float* v,
                                           int n_heads, int head_dim,
                                           int max_seq, int pos) {
    int total = n_heads * head_dim;
    for (int i = (int)(ctaid_x() * ntid_x() + tid_x()); i < total;
         i += (int)(ntid_x() * gridDim_x_helper())) {
        int h = i / head_dim;
        int d = i - h * head_dim;
        int ci = (h * max_seq + pos) * head_dim + d;
        kc[ci] = k[i];
        vc[ci] = v[i];
    }
}

// ============================================================================
// RoPE using a precomputed cos/sin table (built on the CPU).
// table[(pos*half + j)*2 + 0/1] = cos/sin for pair j.
// grid = n_heads + n_kv_heads, block = head_dim/2
// ============================================================================
extern "C" __global__ void rope_kernel(float* q, float* k, const float* table,
                                       int n_heads, int n_kv_heads,
                                       int head_dim, int pos) {
    unsigned j = tid_x();
    int half = head_dim / 2;
    if ((int)j >= half) return;

    int base = (pos * half + (int)j) * 2;
    float c = table[base];
    float s = table[base + 1];

    int b = (int)ctaid_x();
    float* vec;
    if (b < n_heads) {
        vec = q + (long long)b * head_dim;
    } else {
        int hk = b - n_heads;
        if (hk >= n_kv_heads) return;
        vec = k + (long long)hk * head_dim;
    }

    float x0 = vec[2 * j];
    float x1 = vec[2 * j + 1];
    vec[2 * j]     = x0 * c - x1 * s;
    vec[2 * j + 1] = x0 * s + x1 * c;
}

// ============================================================================
// Causal attention for ONE query position.
// q: [n_heads][head_dim];  kc/vc: [n_kv_heads][max_seq][head_dim]
// out: [n_heads][head_dim]
// grid = n_heads, block = 128, dynamic shared = (pos+1) floats
// ============================================================================
extern "C" __global__ void attention_kernel(const float* q,
                                            const float* kc, const float* vc,
                                            float* out,
                                            int n_heads, int n_kv_heads,
                                            int head_dim, int max_seq,
                                            int pos, float scale) {
    extern __shared__ float scores[];
    __shared__ float scratch[128];

    int h = (int)ctaid_x();
    unsigned tid = tid_x();
    int group = n_heads / n_kv_heads;
    int hkv = h / group;
    int seq = pos + 1;

    const float* qh = q + (long long)h * head_dim;
    const float* kh = kc + (long long)hkv * max_seq * head_dim;
    const float* vh = vc + (long long)hkv * max_seq * head_dim;

    // scores[s] = q . k_s * scale
    for (int s = (int)tid; s < seq; s += (int)ntid_x()) {
        const float* ks = kh + (long long)s * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) sum += qh[d] * ks[d];
        scores[s] = sum * scale;
    }
    __syncthreads();

    // softmax
    float mx = -3.402823466e38f;
    for (int s = (int)tid; s < seq; s += (int)ntid_x())
        mx = scores[s] > mx ? scores[s] : mx;
    mx = block_reduce_max(mx, scratch);
    __syncthreads();

    float sum = 0.0f;
    for (int s = (int)tid; s < seq; s += (int)ntid_x()) {
        float e = f_exp(scores[s] - mx);
        scores[s] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum, scratch);
    float inv = 1.0f / sum;

    // out[d] = sum_s p[s] * v_s[d]
    float* oh = out + (long long)h * head_dim;
    for (int d = (int)tid; d < head_dim; d += (int)ntid_x()) {
        float acc = 0.0f;
        for (int s = 0; s < seq; ++s) acc += scores[s] * vh[(long long)s * head_dim + d];
        oh[d] = acc * inv;
    }
}

// ============================================================================
// Batched prefill kernels
// ============================================================================

// Tiled GEMM: Y[T, M] = X[T, K] @ W[M, K]^T
// One block computes a 16-token x 16-output tile.
#define GEMM_BM 32
#define GEMM_BN 16
#define GEMM_BK 32

extern "C" __global__ void gemm_i16_kernel(const short* W, const float* row_scale,
                                           const float* X, float* Y,
                                           int T, int M, int K) {
    __shared__ float xs[GEMM_BM][GEMM_BK + 1];
    __shared__ float ws[GEMM_BN][GEMM_BK + 1];

    const int t0 = (int)ctaid_y() * GEMM_BM;
    const int m0 = (int)ctaid_x() * GEMM_BN;
    const unsigned tid = tid_x();
    const int ti = (int)tid / GEMM_BN;
    const int mi = (int)tid % GEMM_BN;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        for (int idx = (int)tid; idx < GEMM_BM * GEMM_BK; idx += (int)ntid_x()) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int tt = t0 + r, kk = k0 + c;
            xs[r][c] = (tt < T && kk < K) ? X[(long long)tt * K + kk] : 0.0f;
        }
        for (int idx = (int)tid; idx < GEMM_BN * GEMM_BK; idx += (int)ntid_x()) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int mm = m0 + r, kk = k0 + c;
            ws[r][c] = (mm < M && kk < K) ? (float)W[(long long)mm * K + kk] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < GEMM_BK; ++k) acc += xs[ti][k] * ws[mi][k];
        __syncthreads();
    }

    int tt = t0 + ti, mm = m0 + mi;
    if (tt < T && mm < M) Y[(long long)tt * M + mm] = acc * row_scale[mm];
}

extern "C" __global__ void gemm_kernel(const float* W, const float* X, float* Y,
                                       int T, int M, int K) {
    __shared__ float xs[GEMM_BM][GEMM_BK + 1];
    __shared__ float ws[GEMM_BN][GEMM_BK + 1];

    const int t0 = (int)ctaid_y() * GEMM_BM;
    const int m0 = (int)ctaid_x() * GEMM_BN;
    const unsigned tid = tid_x();
    const int ti = (int)tid / GEMM_BN;
    const int mi = (int)tid % GEMM_BN;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        for (int idx = (int)tid; idx < GEMM_BM * GEMM_BK; idx += (int)ntid_x()) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int tt = t0 + r, kk = k0 + c;
            xs[r][c] = (tt < T && kk < K) ? X[(long long)tt * K + kk] : 0.0f;
        }
        for (int idx = (int)tid; idx < GEMM_BN * GEMM_BK; idx += (int)ntid_x()) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int mm = m0 + r, kk = k0 + c;
            ws[r][c] = (mm < M && kk < K) ? W[(long long)mm * K + kk] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < GEMM_BK; ++k) acc += xs[ti][k] * ws[mi][k];
        __syncthreads();
    }

    int tt = t0 + ti, mm = m0 + mi;
    if (tt < T && mm < M) Y[(long long)tt * M + mm] = acc;
}

// Embedding for a batch of token ids.
extern "C" __global__ void embedding_batch_kernel(const float* table, const int* ids,
                                                  float* out, int T, int dim) {
    int t = (int)ctaid_x();
    if (t >= T) return;
    int id = ids[t];
    const float* row = table + (long long)id * dim;
    for (int i = (int)tid_x(); i < dim; i += (int)ntid_x()) out[(long long)t * dim + i] = row[i];
}

extern "C" __global__ void embedding_batch_i16_kernel(const short* table,
                                                      const float* row_scale,
                                                      const int* ids, float* out,
                                                      int T, int dim) {
    int t = (int)ctaid_x();
    if (t >= T) return;
    int id = ids[t];
    const short* row = table + (long long)id * dim;
    float s = row_scale[id];
    for (int i = (int)tid_x(); i < dim; i += (int)ntid_x())
        out[(long long)t * dim + i] = (float)row[i] * s;
}

// RoPE for a batch of positions. q: [T, H, HD], k: [T, Hkv, HD].
extern "C" __global__ void rope_batch_kernel(float* q, float* k, const float* table,
                                             int n_heads, int n_kv_heads,
                                             int head_dim, int start, int T) {
    int b = (int)ctaid_x();
    int t = (int)ctaid_y();
    if (t >= T) return;
    unsigned j = tid_x();
    int half = head_dim / 2;
    if ((int)j >= half) return;

    int pos = start + t;
    int base = (pos * half + (int)j) * 2;
    float c = table[base];
    float s = table[base + 1];

    float* vec;
    if (b < n_heads) {
        vec = q + ((long long)t * n_heads + b) * head_dim;
    } else {
        int hk = b - n_heads;
        if (hk >= n_kv_heads) return;
        vec = k + ((long long)t * n_kv_heads + hk) * head_dim;
    }

    float x0 = vec[2 * j];
    float x1 = vec[2 * j + 1];
    vec[2 * j]     = x0 * c - x1 * s;
    vec[2 * j + 1] = x0 * s + x1 * c;
}

// Write K/V for a batch of positions. k/v: [T, Hkv, HD].
extern "C" __global__ void kv_write_batch_kernel(float* kc, float* vc,
                                                 const float* k, const float* v,
                                                 int n_heads, int head_dim,
                                                 int max_seq, int start, int T) {
    int t = (int)ctaid_y();
    if (t >= T) return;
    int total = n_heads * head_dim;
    for (int i = (int)(ctaid_x() * ntid_x() + tid_x()); i < total;
         i += (int)(ntid_x() * gridDim_x_helper())) {
        int h = i / head_dim;
        int d = i - h * head_dim;
        int ci = (h * max_seq + start + t) * head_dim + d;
        int si = (t * n_heads + h) * head_dim + d;
        kc[ci] = k[si];
        vc[ci] = v[si];
    }
}

// Causal attention for a batch of query positions.
// q: [T, H, HD], out: [T, H, HD], kc/vc: [Hkv, max_seq, HD].
extern "C" __global__ void attention_batch_kernel(const float* q,
                                                  const float* kc, const float* vc,
                                                  float* out,
                                                  int n_heads, int n_kv_heads,
                                                  int head_dim, int max_seq,
                                                  int start, int T, float scale) {
    extern __shared__ float scores[];
    __shared__ float scratch[128];

    int h = (int)ctaid_x();
    int t = (int)ctaid_y();
    if (t >= T) return;

    unsigned tid = tid_x();
    int group = n_heads / n_kv_heads;
    int hkv = h / group;
    int pos = start + t;
    int seq = pos + 1;

    const float* qh = q + ((long long)t * n_heads + h) * head_dim;
    const float* kh = kc + (long long)hkv * max_seq * head_dim;
    const float* vh = vc + (long long)hkv * max_seq * head_dim;

    for (int s = (int)tid; s < seq; s += (int)ntid_x()) {
        const float* ks = kh + (long long)s * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) sum += qh[d] * ks[d];
        scores[s] = sum * scale;
    }
    __syncthreads();

    float mx = -3.402823466e38f;
    for (int s = (int)tid; s < seq; s += (int)ntid_x())
        mx = scores[s] > mx ? scores[s] : mx;
    mx = block_reduce_max(mx, scratch);
    __syncthreads();

    float sum = 0.0f;
    for (int s = (int)tid; s < seq; s += (int)ntid_x()) {
        float e = f_exp(scores[s] - mx);
        scores[s] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum, scratch);
    float inv = 1.0f / sum;

    float* oh = out + ((long long)t * n_heads + h) * head_dim;
    for (int d = (int)tid; d < head_dim; d += (int)ntid_x()) {
        float acc = 0.0f;
        for (int s = 0; s < seq; ++s) acc += scores[s] * vh[(long long)s * head_dim + d];
        oh[d] = acc * inv;
    }
}
