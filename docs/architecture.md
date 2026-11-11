# Architecture

I'll start with the thing I care about most: this engine is small enough that you
can hold the whole data flow in your head. There's no hidden runtime, no graph
compiler, no scheduler. A prompt goes in, tokens come out, and every step is a
function you can put a breakpoint in.

That's also why it's slow on the CPU. I wrote the scalar version first because I
wanted to see the math. The GPU path came later, and it exists mostly because I
wanted to know how much of a 1.1B model I could jam into 4 GB of VRAM.

## What I was going for

Five rules, roughly in the order I cared about them:

1. Get the math right before making it fast.
2. Don't depend on another inference runtime. No llama.cpp, no PyTorch, no BLAS.
3. Keep the layers replaceable. Parsing, quant, math, model logic, sampling, and
   tokenization each get their own module.
4. Make the CPU path the reference. If the GPU disagrees, the GPU is wrong.
5. Make it work on hardware people actually have. The target here is a GTX 1650
   with 4 GB.

What I deliberately left out: batching, multi-GPU, continuous batching, MoE,
multimodal, and training. Those are not accidents. They're just not the thing I
was trying to learn this time.

## How the pieces fit

```
main.cpp
  ├─ GGUFParser            file → metadata + tensor descriptors
  ├─ load_model_weights    descriptors → FP32 tensors
  ├─ Tokenizer             text ↔ ids
  ├─ Model                 CPU forward + KV cache
  └─ generate_cuda         GPU forward + INT16 weights
        ├─ ops             CPU math
        ├─ cuda_backend    buffers + kernel launches
        └─ cuda_kernels    device code
```

The dependency arrows only point down. The model layer never sees a CUDA type,
and the CUDA layer never sees a GGUF header. I've found that if I let those two
leak into each other, the whole thing turns into soup.

## The tensor, and the layout rule that bit me

`Tensor` is about as boring as it sounds:

```cpp
DataType dtype_;                  // F32, Q4_K, ...
Shape shape_;                     // std::vector<int64_t>
Device device_;                   // CPU or CUDA
std::unique_ptr<uint8_t[]> data_; // contiguous bytes
```

Elements are row-major. For `[M, N]`, element `(i, j)` lives at `i*N + j`.
`reshape()` only changes metadata and checks that the element count still
matches. Copying is a deep copy, which is convenient and expensive. That's why
the CLI does `Model model(std::move(weights))` instead of passing by value.

Here's the rule that cost me a day:

> GGML stores `ne[0]` contiguously. A weight written with dims `[in, out]` is
> already C row-major `[out][in]`. Reverse the shape metadata. Do not transpose
> the bytes.

I originally wrote a transpose. It looked reasonable, and it scrambled every
model in a way that still produced finite logits, which is the worst kind of bug.
If you remember one thing from this document, remember that one.

So after loading:

| Tensor | File dims | In-memory shape |
|---|---|---|
| `token_embd.weight` | `[D, V]` | `[V, D]` |
| `attn_q.weight` | `[D, D]` | `[D, D]` |
| `attn_k.weight` | `[D, Hkv·HD]` | `[Hkv·HD, D]` |
| `ffn_gate.weight` | `[D, FFN]` | `[FFN, D]` |
| `output.weight` | `[D, V]` | `[V, D]` |

GPU memory is a separate thing. `cuda::Buffer` is just `{ptr, bytes}`. I kept it
opaque so no CUDA header ever leaks into the model code.

## Getting a GGUF into memory

`GGUFParser` does three things in its constructor:

1. Reads the header: magic, version, tensor count, metadata count.
2. Reads metadata into a `std::variant`. The metadata type enum is different from
   the tensor dtype enum, which is a trap. Type 8 is a string, type 11 is int64,
   and if you mix them up you get string reads at nonsense offsets.
3. Reads tensor descriptors in this exact order:

   ```
   name, n_dims, dims[], type, offset
   ```

Then it does the thing I forgot the first time: the raw tensor data starts at the
next multiple of `general.alignment` (usually 32), not right after the
descriptors. A 4-byte shift there turns scales into quant bytes and weights into
numbers around 10^5.

`type_nbytes()` is the other important helper. Quantized tensors are not one byte
per element. A Q4_0 block is 18 bytes for 32 values; Q4_K is 144 bytes for 256;
Q6_K is 210 for 256. You need the exact block size before you can seek.

Once the descriptors are parsed, `load_model_weights()` walks them:

- read raw bytes,
- dequantize (or convert F16) to FP32,
- reverse the dims,
- optionally quantize each 2-D weight to INT16 with per-row scales,
- store it in a `HostWeight` (FP32 or INT16).

By default host weights are INT16: TinyLlama ends up around 2.2 GB (peak ~3 GB
during load). With `--host-precision fp32` it's the old 4.4 GB layout. The
integer path is a runtime format, not a new GGUF type.

## The CPU path

The CPU code is the reference implementation, so it's written the dumb way. One
token at a time. Explicit loops. Stack arrays for the common sizes.

The block itself is the standard Llama sandwich:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Inside attention, for a token at absolute position `pos`:

```
q = Wq · xn                      [H · HD]
k = Wk · xn                      [Hkv · HD]
v = Wv · xn                      [Hkv · HD]

RoPE(q, k, pos)
cache.k[layer][pos] = k
cache.v[layer][pos] = v

for h in 0..H-1:
    hkv = h / (H / Hkv)
    scores[s] = q_h · cache.k[hkv][s] / sqrt(HD)   for s = 0..pos
    p = softmax(scores)
    out_h = Σ p[s] · cache.v[hkv][s]

x = x + Wo · out
```

If `H == Hkv` this is plain multi-head attention. If `Hkv < H` it's grouped-query
attention. TinyLlama has 32 query heads and 4 KV heads, so eight query heads
share each KV head. I tested that path with a synthetic reference because it's
easy to get the mapping wrong and hard to notice on a tiny model.

The FFN is SwiGLU:

```
gate = SiLU(xn · W1ᵀ)
up   = xn · W3ᵀ
down = (gate ⊙ up) · W2ᵀ
x    = x + down
```

The KV cache is per layer:

```
keys_[layer]   : [1, Hkv, max_seq, HD]  FP32
values_[layer] : [1, Hkv, max_seq, HD]  FP32
```

The formula I use when sizing it:

```
2 × layers × Hkv × max_seq × HD × 4 bytes
```

For TinyLlama that's around 90 MB. Small compared to the weights.

One thing that surprised me: the CPU prefill is not batched. `forward()` takes a
`[1, T]` input and loops over `T` inside each layer. It's simpler, and on a
model this size the difference doesn't matter much. The GPU path now does batch
it; see below.

## The GPU path, and why there's no CUDA toolkit

The machine I was targeting had the NVIDIA driver but no `nvcc`, no CUDA headers,
and no passwordless sudo. Installing a 3 GB toolkit wasn't happening.

So I used a different route:

```
cuda_kernels.cu
    │  clang++ -x cuda --cuda-device-only -nocudainc -nocudalib
    ▼
kernels.ptx
    │  cmake/EmbedPtx.cmake
    ▼
cuda_ptx_data.h   (a C string compiled into the binary)
    │  dlopen("libcuda.so.1")
    ▼
cuModuleLoadData(PTX) → driver JIT → cuLaunchKernel
```

The kernels don't include CUDA headers at all. They use the compiler's NVVM
builtins for the hardware registers:

```cpp
__nvvm_read_ptx_sreg_tid_x()    // threadIdx.x
__nvvm_read_ptx_sreg_ntid_x()   // blockDim.x
__nvvm_read_ptx_sreg_ctaid_x()  // blockIdx.x
```

And inline PTX for math, so there's no libdevice dependency:

```cpp
asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); // exp2
asm("sqrt.rn.f32 %0, %1;"   : "=f"(r) : "f"(x)); // sqrt
```

The host side loads `libcuda.so.1` with `dlopen` and pulls symbols with `dlsym`.
I hit one driver quirk worth writing down: `cuDevicePrimaryCtxRetain` returned
success with a null context on this driver. `cuCtxCreate_v2` works, so that's
what the code uses.

## INT16 weights

The first GPU version uploaded FP32 weights. TinyLlama needs about 4.3 GB, and
the card has roughly 3.1 GB free. No chance.

So at upload time I quantize every 2-D linear weight to INT16, with one float
scale per output row:

```
scale[r] = max|W[r, :]| / 32767
q[r, c]  = clamp(round(W[r, c] / scale[r]), -32767, 32767)
```

The kernel accumulates in FP32 and applies the scale at the end:

```
y[r] = scale[r] * Σ_c q[r, c] * x[c]
```

That halves the weight traffic and pushes TinyLlama to about 2.2 GB, which fits.
Greedy output is identical to the FP32 CPU path. If you want full precision back,
`--gpu-precision fp32` still works for models that fit.

The scheme is not GGUF quantization. It's a runtime format. The original file
stays on disk. Host weights are INT16 by default, and the GPU gets INT16 too;
`--host-precision fp32` keeps the host copy in FP32 if you want the CPU
reference path at full precision.

Prompt prefill is batched. Instead of running one token at a time, it processes
the whole prompt through each layer with a tiled GEMM (32 tokens × 16 outputs per
block, padded shared tiles to avoid bank conflicts), then computes logits for the
last position only. On TinyLlama an 89-token prompt dropped from ~2.0 s to
~0.7 s; short prompts improved from ~0.4 s to ~0.24 s. Decode still uses the
single-token GEMV path.

## Where the memory goes

Host RAM, default (INT16 host weights):

```
2 bytes × parameters
+ 4 bytes × output rows   (scales)
+ temporary FP32 scratch during load
```

TinyLlama: ~2.2 GB steady state, ~3 GB peak during load. With
`--host-precision fp32` it goes back to `4 bytes × parameters`, about 4.4 GB.
That last mode is the real limit on a 15 GB machine with a desktop running.

VRAM with INT16:

```
2 bytes × parameters
+ 4 bytes × output rows       (scales)
+ KV cache
+ activations
```

TinyLlama: ~2.2 GB weights, ~90 MB KV, under a megabyte of activations. The
`--device auto` check adds a 96 MB safety margin.

`cuda_required_bytes(model, int16)` is the function that does this math. If you
change the quantization scheme, change it there too, or the preflight check will
lie to you.

## Tokenizer and sampling

The tokenizer loads BPE data from either `tokenizer.json` or the GGUF metadata
itself. The GGUF route is the default because the file already contains the vocab
and merges.

Encoding is SentencePiece-ish: replace spaces with `▁`, add a dummy prefix, split
into UTF-8 codepoints, then merge adjacent pairs by lowest merge rank. Unknown
symbols fall back to `<0xXX>` byte tokens.

The tokenizer detects the common chat formats from `tokenizer.chat_template`
(Zephyr/TinyLlama, Llama-3, ChatML, Mistral, Gemma) and renders messages with the
right markers. It is not a general Jinja implementation: an unknown template
falls back to a simple `User:`/`Assistant:` format. Interactive mode uses the
template automatically; `--chat` and `--no-chat` override it.

Sampling is straightforward: temperature, softmax, top-k, top-p, renormalize,
draw. `temperature <= 0` short-circuits to argmax. Seeds use
`config.seed + generated.size()` so a fixed seed is reproducible without reusing
the same random draw for every token.

## Threading and lifetime

The CPU path is single-threaded. I know. It's the next obvious optimization.

The CUDA context is process-wide, created once, and launches go on the default
stream. GPU states are cached by model pointer so interactive mode doesn't
re-upload the weights every turn. The mutex only protects the cache lookup;
generation itself is assumed single-threaded.

I intentionally leak GPU buffers at process exit. Freeing them during static
destruction means the driver might already be torn down. If you need to free
explicitly, `release_cuda_model(model)` does it while the process is alive.

## Where the time actually goes

On the CPU, this thing is compute-bound. TinyLlama is about 2.2 GFLOP per token,
and scalar C++ gets a couple of GFLOP/s. Hence around 1 tok/s with FP32 host
weights, or ~0.5 tok/s with the default INT16 host weights. The int16-to-float
conversion is the cost of halving RAM.

On the GPU, it's memory-bound. Every token reads every weight once. For INT16
TinyLlama that's ~2.2 GB per token. I measured the GEMV kernel at about 43 GB/s on
the 1650, which puts the floor around 50 ms/token, or roughly 20 tok/s. Actual
decode is ~29 tok/s. We're close to the bandwidth limit.

The tiny model is a different story. At 15M parameters the weights are small
enough that kernel launch overhead dominates, and it runs at ~459 tok/s.

If you want more speed, the order I'd attack it in:

1. Keep Q4_K/Q6_K blocks in VRAM and dequantize inside the matmul. That cuts
   TinyLlama from 2.2 GB to roughly 0.7 GB and makes 2–3B models possible.
2. Make the CPU path blocked, SIMD, and threaded.
3. Batch the CPU prefill too, and add real attention tiling for long contexts.

## If you're going to change something

- New quant format: start in `type_nbytes_per_block` and `dequantize_*`.
- New CPU op: `ops.hpp` / `ops.cpp`, then a test.
- New GPU kernel: `cuda_kernels.cu`, a wrapper in `cuda_backend`, and add the
  name to the kernel list (that last step is easy to forget).
- New architecture: config keys, tensor names, then the block math.
  [models.md](models.md) walks through it.
- Batching: `Model::forward` and the GPU `forward_token` schedule.

The code isn't precious. Most of it is one obvious function per idea. If
something reads as clever, it's probably a bug.
