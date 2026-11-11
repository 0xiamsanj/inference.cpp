# Build, Test, Debug, and Extend

This is the stuff I actually do when I work on the engine. Building, testing,
the debugging order that saves me time, and the conventions that keep me from
breaking the tensor layout again.

## What you need

| Thing | Why | Notes |
|---|---|---|
| C++20 compiler | everything | g++ 12+ or clang 15+ |
| CMake 3.20+ | build | |
| `libspdlog-dev` | logging dependency | |
| `clang++` | CUDA path | turns kernels into PTX; no CUDA toolkit |
| NVIDIA driver | CUDA runtime | must expose `libcuda.so.1` |
| Python + `gguf`/`numpy` | optional | reference dequantization checks |

On Ubuntu:

```bash
sudo apt update
sudo apt install -y cmake g++ libspdlog-dev clang
```

No `nvcc`, no headers, no `cudart`, no `cuBLAS`. That's on purpose.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The options I actually use:

| Option | Default | What it does |
|---|---|---|
| `WITH_CUDA` | `ON` | build the clang→PTX GPU backend |
| `CUDA_GPU_ARCH` | `sm_75` | PTX target; GTX 1650 is sm_75 |
| `WITH_AVX2` | `ON` | `-mavx2 -mfma` for the CPU build |

A few real-world notes:

- If `WITH_CUDA` was cached as `OFF`, rebuilding won't turn it back on. Run
  `cmake -B build -DWITH_CUDA=ON`.
- If you're on a different GPU, set the arch. `sm_86` for RTX 30-series,
  `sm_89` for 40-series, and so on.
- The build generates `build/generated/cuda_kernels.ptx` and
  `build/generated/cuda_ptx_data.h`. If the PTX build fails, you'll see the
  `clang++` error in the middle of the CMake output.

The three binaries:

```
build/inference
build/debug_logits
build/tests/run_tests
```

## Running things

```bash
# tiny model, GPU
./build/inference --device cuda --gpu-precision int16 \
  --tokens 100 --temp 0.8 models/stories15M-q4_0.gguf "Once upon a time"

# TinyLlama on the GPU
./build/inference --device cuda --gpu-precision int16 \
  --tokens 64 --temp 0.7 models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "The capital of France is"

# CPU reference
./build/inference --device cpu --tokens 20 \
  models/stories15M-q4_0.gguf "Hello"

# interactive
./build/inference -i --device auto \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# logits and weight stats
./build/debug_logits models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "The capital of France is"
```

CLI flags, quickly:

```
-t N               max new tokens (default 512)
--temp F           temperature; 0 = greedy
--top-k N          default 40
--top-p F          default 0.95
--seed N           reproducible sampling
--device D         auto | cpu | cuda
--gpu-precision P  int16 | fp32
--chat             apply the model's chat template
--no-chat          force raw completion
--system TEXT      system message (implies --chat)
--tokenizer PATH   tokenizer.json (GGUF fallback is automatic)
-i                 interactive
```

## Tests

```bash
./build/tests/run_tests
./build/tests/run_tests --gtest_filter=CudaTest.*
./build/tests/run_tests --gtest_filter=QuantizationTest.*
```

The suites:

- `TensorTest`: creation, fill, reshape, deep copy.
- `QuantizationTest`: Q4_0/Q8_0 round-trip.
- `OpsTest`: GEMV/GEMM, softmax, RMSNorm, RoPE, SiLU, sampling.
- `CudaTest`: small GEMV, RMSNorm, long GQA attention, INT16 GEMV, INT16
  embedding.

The CUDA tests skip themselves if there's no device. The CPU-only build excludes
them entirely.

When I add an op, I add two tests: one numerical (against a hand calculation or a
CPU reference), and one shape/edge case. The numerical test is the one that
catches the bug; the edge case is the one that catches the crash.

## Debugging, in the order I do it

### 1. Did it load?

```
Architecture: llama
Layers: 22
Embed dim: 2048
Heads: 32
Vocab size: 32000
```

If any of those are zero, stop. The metadata keys are wrong, or the architecture
prefix is wrong. No point looking at the math.

### 2. Do the weights look sane?

`debug_logits` prints min/max/mean/std for embeddings, output, and the first
layer. That's usually enough.

What I look for:

- Values around 1e4 → scale interpreted wrong (fp16 vs fp32, or wrong stride).
- Std about 10× too small → zeroed subnormal fp16 scales.
- Plausible min/max but garbage output → transposed dims or wrong tensor.

For a real reference, dequantize the same tensor with the `gguf` package and
compare. The recipe is in
[gguf-quantization.md](gguf-quantization.md#checking-your-dequantizer).

### 3. Do the logits make sense?

For `"The capital of France is"`, TinyLlama should put `" Paris"` on top. If the
top tokens are multilingual junk, the output projection or the final hidden state
is wrong. The sampler is almost never the problem.

### 4. CPU vs GPU

Greedy is the clean comparison:

```bash
./build/inference --device cpu  --tokens 20 --temp 0 model.gguf "prompt"
./build/inference --device cuda --gpu-precision fp32 --tokens 20 --temp 0 model.gguf "prompt"
```

Use FP32 on the GPU for parity. INT16 can drift after enough tokens; that's
quantization noise, not a kernel bug.

### 5. Is it slow, or does it just look slow?

The CLI reports:

```
[N tokens, X tok/s decode, GPU init Y ms, prefill Z ms]
```

`decode` excludes init and prefill. If that number is bad, compare against the
isolated GEMV benchmark in the next section before profiling the whole model.

### Symptoms I've actually hit

| Symptom | What it was |
|---|---|
| `Tensor not found` | architecture naming |
| `CUDA backend unavailable` | no `clang++` at build, or no driver at runtime |
| `model needs ~X MB VRAM` | model too big; use INT16/CPU/smaller model |
| Garbage from token 0 | tokenizer ids, dequant, or dim reversal |
| Garbage only on some models | base quant format, often Q6_K |
| Right first token, drift later | KV positions, RoPE table, or seed |
| Interactive aborts | prompt + generated exceeded `max_seq_len` |
| Instruct model gives mediocre answers | wrong or missing chat template; try `--chat` / `--no-chat` |

## Profiling

On the CPU:

```bash
perf stat ./build/inference --device cpu --tokens 16 model.gguf "Hello"
perf record -g ./build/inference --device cpu --tokens 16 model.gguf "Hello"
```

It'll be dominated by the GEMVs inside `attention_forward`, `ffn_forward`, and
the output projection.

On the GPU, the CLI timing is usually enough. If you want kernel numbers, write
a tiny benchmark against `cuda_backend.hpp`:

```cpp
Buffer W = alloc(M*N*sizeof(int16_t)), sc = alloc(M*sizeof(float)), ...;
for (int i = 0; i < 5; ++i) gemv_i16(W, sc, x, y, M, N);
sync();
auto t0 = now();
for (int i = 0; i < 100; ++i) gemv_i16(W, sc, x, y, M, N);
sync();
```

Numbers I got on the GTX 1650:

```
int16 GEMV 5632×2048: ~0.53 ms/pass, ~43 GB/s
fp32  GEMV 5632×2048: ~0.59 ms/pass, ~78 GB/s
```

The INT16 kernel moves half the bytes but spends time converting, so it's not
faster per pass on a matrix that already fits. It's what makes the model fit at
all.

Where the 1.1B time goes:

```
~2.2 GB of weight traffic per token
~43 GB/s
→ ~50 ms/token floor (~20 tok/s)
measured: ~29 tok/s
```

That's close enough to the bandwidth limit that the next big win is reading fewer
bytes, not writing a fancier kernel.

For the 15M model, weights are tiny and launch overhead dominates. ~459 tok/s is
what that looks like.

## Adding a CPU op

1. Declare it in `include/ops/ops.hpp`.
2. Write it in `src/ops/ops.cpp` with explicit loops.
3. Check dtype, shape, and numel at the top.
4. Add a numerical test.
5. Call it from `model.cpp`.

I stick to FP32 unless I'm also adding a conversion. It keeps the code honest.

## Adding a GPU kernel

1. Write it in `src/ops/cuda_kernels.cu` as `extern "C" __global__`.
2. Get indices from the NVVM builtins, not from CUDA headers.
3. Use inline PTX for math (`ex2.approx.f32`, `sqrt.rn.f32`).
4. Declare a wrapper in `cuda_backend.hpp`.
5. Implement the wrapper in `cuda_backend.cpp`.
6. **Add the kernel name to `names[]` in `init_driver`.** I forget this almost
   every time, and the error is just "unknown kernel".
7. Add a CUDA test.
8. Call it from `model_cuda.cpp`.

To check the PTX by itself:

```bash
clang++ -x cuda --cuda-device-only -nocudainc -nocudalib \
  --cuda-gpu-arch=sm_75 -O3 -std=c++17 -S \
  src/ops/cuda_kernels.cu -o /tmp/kernels.ptx
grep "\.entry" /tmp/kernels.ptx
```

Two gotchas:

- Static shared arrays are sized at compile time (`red[256]`, `scratch[128]`).
  Match the block size or resize the array.
- Attention uses dynamic shared memory of `(pos+1) * 4` bytes. Pass it at launch.

And one driver detail: use `cuCtxCreate_v2`. `cuDevicePrimaryCtxRetain` gave me
a null context with a success code on this driver.

## Adding a sampler

For something simple, add a function in `ops.cpp` and call it from `generate`.

For something structured, implement the `Sampler` interface in
`sampler.cpp`. There's already a `SamplingParams` struct with temperature, top-k,
top-p, min-p, and repetition penalty.

Pass the seed through. The convention here is `config.seed + generated.size()`,
which gives reproducible output without reusing one random draw for every token.

## Adding a quant format

Six steps, all in `tensor.hpp` and `quantization.cpp`:

1. enum value,
2. `type_nbytes_per_block`,
3. `block_size`,
4. `dequantize_<type>` with explicit byte offsets,
5. dispatch case,
6. test.

Don't define a `struct` with `float` fields for an on-disk block. Padding will
not match. The current code uses byte offsets for a reason.

## Adding an architecture

The Qwen2 recipe in [models.md](models.md#if-i-were-adding-a-new-family) is the
one I'd follow:

1. dynamic metadata prefix,
2. tensor-name mapping (and optional biases),
3. missing ops,
4. block forward changes,
5. verify logits against a reference,
6. document the approximations.

The verification step is not optional. "It generated a sentence" is not a test.

## The rules I don't break

These are the invariants that prevent silent corruption:

1. **Row-major `[out, in]`.** Reverse GGUF dims; never transpose bytes.
2. **Exact block sizes.** Use `type_nbytes`, not `numel × effective_size`, for
   file reads.
3. **fp16 subnormals are real numbers.** `exp == 0 && mant != 0` is not zero.
4. **Tensor data starts after alignment.** Use `tensor_data_offset_`.
5. **KV positions are absolute.** RoPE and cache writes use `start_pos + t`.
6. **GQA maps head `h` to `h / (H/Hkv)`.**
7. **Seeds are per step.** `seed + step`, not one draw for the whole run.
8. **No CUDA types outside `cuda_backend.cpp`.** If a header needs one, the
   abstraction is wrong.

## Build errors I've seen

| Error | Fix |
|---|---|
| `FindCUDA` / `CUDA_TOOLKIT_ROOT_DIR` | stale cache from the old build; delete `build/` |
| CUDA silently disabled | no `clang++`, or `WITH_CUDA` cached `OFF` |
| `cuModuleLoadData failed` | driver too old for the PTX version |
| `libcuda.so.1` not found | driver not installed or not loaded |
| `cuMemAlloc` invalid context | use `cuCtxCreate_v2` (already done) |
| `unknown kernel` | forgot to add the name to `names[]` |
| Tests skip CUDA | no GPU/driver in that environment |
| `-nocudainc` compile error | kernel pulled in a CUDA header |

## What I'd do next

In order:

1. **On-GPU Q4_K/Q6_K matmul.** Keep the original blocks in VRAM and dequantize
   inside the kernel. This is the difference between "1.1B on a 4 GB card" and
   "2–3B on a 4 GB card".
2. **Quantized host storage or mmap.** The FP32 host copy is the real RAM wall.
3. **CPU SIMD and threads.** 5–10× on the fallback path.
4. **Arch generalization, biases, Qwen2.**
5. **RoPE scaling.**
6. **BERT embeddings** as a second task.

Every one of those should come with a test and a before/after number. That's the
only way to know it actually helped.
