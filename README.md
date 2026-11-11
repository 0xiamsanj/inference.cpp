# inference.cpp

A from-scratch LLM inference engine in C++20.

I built this because I wanted to see what actually happens inside an LLM
inference engine. Not the diagram version. The bytes-and-loops version.

So: no llama.cpp, no PyTorch, no BLAS. Just C++20, a GGUF file, and a GPU path
that works without installing the CUDA toolkit.

It runs TinyLlama 1.1B on a 4 GB GTX 1650. That's the part I'm happiest about.

```
CPU:  ~1 tok/s for TinyLlama
GPU:  ~29 tok/s, INT16 weights, ~2.2 GB VRAM
```

## What it does

- Loads GGUF models that use the Llama architecture.
- Runs them on the CPU with plain loops, so you can read the math.
- Runs them on an NVIDIA GPU through PTX + the driver, no `nvcc` required.
- Stores GPU weights as INT16 by default. That's how TinyLlama fits in 4 GB.
- Supports Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, and Q6_K tensors.
- Does RMSNorm, RoPE, GQA, SwiGLU, KV caching, top-k/top-p sampling, and
  interactive generation.

What it doesn't do: batching, chat templates, other architecture families, or
several GGUF quant formats. Those are known gaps, not surprises. The details are
in [docs/models.md](docs/models.md).

## Docs

The full study material is in [`docs/`](docs/README.md).

| Doc | What's in it |
|---|---|
| [docs/README.md](docs/README.md) | Index and one-page architecture |
| [docs/architecture.md](docs/architecture.md) | How it's put together, memory layout, CPU/GPU paths |
| [docs/walkthrough.md](docs/walkthrough.md) | One prompt traced through every function |
| [docs/gguf-quantization.md](docs/gguf-quantization.md) | File format, quant formats, the bugs they caused |
| [docs/models.md](docs/models.md) | What runs today and what to add next |
| [docs/development.md](docs/development.md) | Build, test, profile, extend |
| [TUTORIAL.md](TUTORIAL.md) | Beginner-friendly, concept-first course |

If you only read two, make them `architecture.md` and `walkthrough.md`.

## Hardware

The reference machine is an i5-9300HF, 15 GB RAM, and a GTX 1650 with 4 GB.

| Part | Minimum | Comfortable |
|---|---|---|
| CPU | AVX2 | AVX2 + FMA |
| RAM | 8 GB | 16 GB |
| GPU | optional | 4 GB+ and a recent driver |
| Compiler | C++20 | g++ 12+ / clang 15+ |

The GPU path needs `clang++` at build time and the NVIDIA driver at runtime.
Nothing else from CUDA.

## Models that fit

| Model | GGUF size | GPU INT16 | Notes |
|---|---:|---|---|
| stories15M | 19 MB | yes, ~30 MB VRAM | the smoke test |
| TinyLlama-1.1B | ~670 MB | yes, ~2.3 GB | verified here |
| Llama-3.2-1B | ~800 MB | yes, ~2.4 GB | |
| Phi-2-2.7B | ~1.6 GB | no, ~5.4 GB | CPU only |
| Llama-3.2-3B | ~2 GB | no, ~6 GB | CPU only |
| Mistral-7B | ~4.1 GB | no | CPU only, and RAM is a problem |

Remember that the loader dequantizes to FP32 in host RAM. TinyLlama needs about
4.4 GB of RAM. The GPU is not always the limiting factor.

## Build

```bash
sudo apt update
sudo apt install -y cmake g++ libspdlog-dev clang

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/tests/run_tests
```

CUDA is on by default when `clang++` is available. To target a different GPU:

```bash
cmake -B build -DCUDA_GPU_ARCH=sm_86   # RTX 30-series
cmake -B build -DWITH_CUDA=OFF         # CPU only
```

If you previously configured with `WITH_CUDA=OFF`, rebuilding won't turn it back
on. Rerun CMake with `-DWITH_CUDA=ON`.

## Run it

Get the tiny model first. It's 19 MB and the tokenizer is inside the GGUF, so
there's no second download:

```bash
mkdir -p models
curl -L -o models/stories15M-q4_0.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
```

Then:

```bash
# GPU
./build/inference --device cuda --gpu-precision int16 \
  --tokens 100 --temp 0.8 models/stories15M-q4_0.gguf "Once upon a time"

# Let it pick (CUDA if the weights fit)
./build/inference --device auto --tokens 40 \
  models/stories15M-q4_0.gguf "Once upon a time"

# CPU
./build/inference --device cpu --tokens 20 \
  models/stories15M-q4_0.gguf "Hello"
```

For the real thing:

```bash
curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

./build/inference --device cuda --gpu-precision int16 --tokens 64 \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "The capital of France is"

./build/inference -i --device auto \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

The CLI reports decode speed separately from the first GPU upload and prompt
prefill, because the first run looks much worse than later ones:

```
[32 tokens, 27.41 tok/s decode, GPU init 6717 ms, prefill 399 ms]
```

There's also a diagnostics tool. It prints weight statistics and the top logits
for a prompt. It has found every serious bug in this project at least once:

```bash
./build/debug_logits models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "The capital of France is"
```

## How the CUDA path works

This is the part people ask about, so here's the short version.

The machine didn't have `nvcc` or permission to install a 3 GB toolkit. But it
did have `clang++` and the NVIDIA driver. So:

```
cuda_kernels.cu
  → clang++ compiles it to PTX (no CUDA headers)
  → CMake embeds the PTX in the binary as a string
  → at runtime, libcuda.so.1 JITs it
```

The kernels use the compiler's NVVM builtins for `threadIdx`/`blockIdx`, and
inline PTX for `exp2` and `sqrt`. The host side loads `libcuda.so.1` with
`dlopen` and launches kernels through `cuLaunchKernel`.

The full explanation, including the driver context quirk, is in
[docs/architecture.md](docs/architecture.md).

### INT16 weights

GPUs have less memory than you'd like. FP32 TinyLlama is ~4.3 GB. The card has
about 3 GB free. So at upload time each linear weight is quantized to INT16 with
one scale per output row:

```
scale[r] = max|W[r, :]| / 32767
q[r, c]  = round(W[r, c] / scale[r])
y[r]     = scale[r] * Σ q[r, c] * x[c]
```

That's 2 bytes per weight plus a float per row, and TinyLlama lands at ~2.2 GB.
Greedy output matches the FP32 CPU path. Use `--gpu-precision fp32` if you want
full precision back and the model fits.

## Project layout

```
inference.cpp/
├── include/                 # headers
│   ├── core/tensor.hpp
│   ├── gguf/gguf_parser.hpp, quantization.hpp
│   ├── ops/ops.hpp, cuda_backend.hpp
│   ├── models/transformer.hpp, model.hpp, cuda_model.hpp
│   └── tokenizer/tokenizer.hpp
├── src/                     # implementations
│   ├── core/, gguf/, ops/, models/, tokenizer/
│   ├── ops/cuda_kernels.cu  # device code (clang → PTX)
│   └── main.cpp
├── cmake/EmbedPtx.cmake
├── tests/test_tensor.cpp    # 18 tests
├── tools/debug_logits.cpp
├── docs/                    # the study material
├── models/                  # your .gguf files
└── CMakeLists.txt
```

## Key ideas, if you're new to this

The [tutorial](TUTORIAL.md) explains these properly. Quick version:

- **Tensor**: a flat array plus a shape. Row-major. Deep copies.
- **GGUF**: metadata + tensor descriptors + one blob of weights.
- **Quantization**: fewer bits per weight, plus scales. Dequantize once at load.
- **Attention**: Q, K, V projections; scores; softmax; weighted sum of V.
- **KV cache**: store past keys and values so generation doesn't recompute them.
- **RoPE**: rotate Q and K by position. No learned position embeddings.
- **Sampling**: turn logits into one token with temperature/top-k/top-p.
- **BPE tokenizer**: merge frequent symbol pairs, then map to ids.

## Performance notes

Measured here, decode-only:

| Model | CPU | GPU INT16 |
|---|---:|---:|
| stories15M | ~54 tok/s | ~459 tok/s |
| TinyLlama-1.1B | ~1 tok/s | ~29 tok/s |

The GPU path is memory-bound. TinyLlama reads ~2.2 GB of weights per token, and
the card does about 43 GB/s on this kernel. That puts the floor near 20 tok/s, so
29 is reasonable.

The CPU path is compute-bound and scalar. It's the reference, not the fast path.

If you want more:

1. Keep Q4_K/Q6_K blocks in VRAM and dequantize in the matmul. This is the big
   one.
2. Batch the prompt prefill.
3. SIMD + threads for the CPU path.
4. Tiled attention for long contexts.

## Extending it

The short version:

- New quant format: `type_nbytes_per_block` and `dequantize_*`.
- New CPU op: `ops.hpp` / `ops.cpp`.
- New GPU kernel: `cuda_kernels.cu`, a wrapper in `cuda_backend`, and the kernel
  name in `names[]`.
- New architecture: metadata prefix, tensor names, block math.

The longer version, including the Qwen2 recipe, is in
[docs/development.md](docs/development.md) and [docs/models.md](docs/models.md).

## Troubleshooting

| What you see | What to check |
|---|---|
| `Invalid GGUF magic` | wrong file or truncated download |
| `Tensor not found` | model isn't Llama-shaped |
| `CUDA backend unavailable` | `clang++` missing at build, or driver at runtime |
| `model needs ~X MB VRAM` | too big; try INT16, CPU, or a smaller model |
| Garbage from token 0 | tokenizer, dequant, or tensor layout |
| Garbage only on some models | probably a quant format, often Q6_K |
| Slow first run | that's the weight upload; decode speed is reported separately |
| Instruct model is mediocre | no chat template yet |

## Thanks and references

The math follows the Llama implementation in llama.cpp and the GGUF spec. The
quantization layouts are from llama.cpp's `ggml-common.h`. If you find a
discrepancy, the reference implementations win.

- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp quant formats:
  https://github.com/ggml-org/llama.cpp/discussions/2828
- Attention paper: https://arxiv.org/abs/1706.03762

MIT licensed. Use it, break it, learn from it.
