# Build Your Own LLM Inference Engine — inference.cpp

I wrote this tutorial the way I wish someone had explained it to me: start with a
tensor, end with a model generating text, and don't skip the parts that are
actually confusing.

You can read it front to back, or use it as a companion while you read the code.
Every step points at real files. The reference material is in
[`docs/`](docs/README.md) if you want the deeper version.

One piece of advice before you start: keep a terminal open and run things as you
go. Reading inference code without watching shapes and numbers is like reading a
recipe without cooking.

---

## 0. The Big Picture

An inference engine does five jobs. That's it. Everything else is detail.

```
.gguf file  →  parse  →  dequantize  →  transformer  →  sample  →  detokenize
             metadata     Q4_0→FP32      matmuls        pick a     ids→text
             tensors                     attention      token
```

The two paths in this repo do the same jobs:

- **CPU**: explicit loops. Correct, readable, slow.
- **GPU**: kernels written in CUDA C, compiled to PTX with clang, JITed by the
  driver. Fast, more moving parts.

Where things live:

| Concern | Header | Source |
|---|---|---|
| Tensor | `include/core/tensor.hpp` | `src/core/tensor.cpp` |
| GGUF parsing | `include/gguf/gguf_parser.hpp` | `src/gguf/gguf_parser.cpp` |
| Quantization | `include/gguf/quantization.hpp` | `src/gguf/quantization.cpp` |
| CPU math | `include/ops/ops.hpp` | `src/ops/ops.cpp` |
| Norms/attention helpers | `include/ops/normalization.hpp`, `attention.hpp` | `src/ops/*.cpp` |
| Weights + KV cache | `include/models/transformer.hpp` | `src/models/transformer.cpp` |
| CPU forward | `include/models/model.hpp` | `src/models/model.cpp` |
| GPU forward | `include/models/cuda_model.hpp` | `src/models/model_cuda.cpp` |
| CUDA backend | `include/ops/cuda_backend.hpp` | `src/ops/cuda_backend.cpp` |
| CUDA kernels | — | `src/ops/cuda_kernels.cu` |
| Sampling | `include/sampling/sampler.hpp` | `src/sampling/sampler.cpp` |
| Tokenizer | `include/tokenizer/tokenizer.hpp` | `src/tokenizer/tokenizer.cpp` |
| CLI | — | `src/main.cpp` |

---

## Step 1 — The Tensor

A tensor is a flat array, a shape, and a type. That's the whole idea.

```cpp
Tensor t(DataType::F32, {2, 3});   // 6 floats
t.as<float>()[4] = 1.0f;           // row 1, col 1
```

The layout is row-major:

```
[2, 3] → [ a00, a01, a02, a10, a11, a12 ]
             (i,j) → i*3 + j
```

The class has the usual stuff: `numel()`, `reshape()`, `clone()`, `fill()`.
Two details matter later:

- **Copying is deep.** `Tensor(const Tensor&)` allocates and copies. Nice for
  ownership, bad for 4 GB weights. The CLI moves weights into the model.
- **Reshape doesn't move data.** It only changes the shape metadata and checks
  that the element count matches.

**Try it:** open `tests/test_tensor.cpp`, predict what `Reshape` does, then run:

```bash
./build/tests/run_tests --gtest_filter=TensorTest.*
```

---

## Step 2 — Reading a GGUF

A GGUF file is a header, a tagged metadata store, a list of tensor descriptors,
some padding, then one big blob of weights.

```
magic "GGUF" | version | tensor_count | metadata_count
metadata key/value pairs
tensor descriptors: name, n_dims, dims[], type, offset
padding to general.alignment (32)
raw tensor data
```

The parser is in `src/gguf/gguf_parser.cpp`. Constructor, three passes:

1. `parse_header()`
2. `parse_metadata()`
3. `parse_tensor_infos()`

The metadata is the part that tells you what the model is:

```cpp
int layers = parser.get_block_count();       // llama.block_count
int dim    = parser.get_embedding_dim();     // llama.embedding_length
int heads  = parser.get_num_heads();         // llama.attention.head_count
```

Three things will trip you up here. I know because all three got me.

**Metadata types are not tensor types.** A metadata string is type 8. An int64 is
type 11. The tensor dtype enum is separate. If you mix them, you read garbage
strings at wild offsets.

**Descriptor order is name, n_dims, dims, type, offset.** The type comes after
the dims. Not before.

**The data section is aligned.** It starts at the next multiple of
`general.alignment` (32), not immediately after the descriptors. Forget that and
every block is shifted by four bytes.

**Try it:** add a `--info` flag that dumps `parser.tensors()` and compare the
shapes to the llama.cpp output.

---

## Step 3 — Dequantization

Quantization stores weights in fewer bits plus a scale. To run the math simply,
we dequantize to FP32 once at load.

### Q4_0

32 values, 18 bytes:

```
[ d: fp16 ][ qs: 16 bytes ]
```

```cpp
value[j]      = ((qs[j] & 0xF) - 8) * d;  // low nibble
value[j + 16] = ((qs[j] >> 4)   - 8) * d;  // high nibble
```

The scale is fp16, and the nibbles are 16 apart. Both details matter.

### Q8_0

32 values, 34 bytes. Easy:

```
[ d: fp16 ][ 32 × int8 ]
value[j] = qs[j] * d
```

### Q4_K

This is what modern models actually use. 256 values, 144 bytes:

```
[ d: fp16 ][ dmin: fp16 ][ scales: 12 bytes ][ qs: 128 bytes ]
```

Eight sub-blocks of 32. The scale bytes pack 16 six-bit pairs. You unpack them
with `get_scale_min_k4`, then:

```
for 64 values:
  get two scale/min pairs
  y[0..31]  = d1 * (low nibbles)  - m1
  y[32..63] = d2 * (high nibbles) - m2
```

The code in `quantization.cpp` is the reference. Read it next to the formula.

### The bug that taught me the most

TinyLlama initially produced multilingual garbage. The weights looked fine.
Min/max were fine. The standard deviation was 10× too small.

Why? Q6_K scales are fp16, and TinyLlama's output-projection scales are
**subnormal**. My converter did `if (exp == 0) f = sign`, which turned them into
zero. Whole blocks disappeared. The model still produced confident-looking junk.

The fix is one line of `ldexp`. The lesson is: keep a reference dequantizer
around and compare the stats, not just the min/max.

**Try it:** call `dequantize_q4_0` on a known block and check the first values by
hand. Then try Q4_K. If the numbers look plausible but the model is bad, compare
the standard deviation against `gguf.quants.dequantize`.

---

## Step 4 — Loading Weights

`load_model_weights` turns descriptors into `ModelWeights`.

For each tensor:

1. Find it.
2. Read the raw bytes.
3. Dequantize or convert F16 to FP32.
4. Reverse the dims and copy the bytes.

That last part is the layout rule:

> GGML stores `ne[0]` contiguously, so file dims `[in, out]` are already
> row-major `[out][in]`. Reverse the shape. Don't transpose.

Names look like this:

| Field | llama.cpp name |
|---|---|
| embedding | `token_embd.weight` |
| final norm | `output_norm.weight` |
| output | `output.weight` (or tied) |
| per layer | `blk.N.attn_q.weight`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, `attn_norm`, `ffn_norm` |

If `output.weight` is missing, copy the embedding and set `output_tied`.

TinyLlama ends up as ~4.4 GB of host RAM. That's why the next section exists.

---

## Step 5 — The Transformer Block

A layer is two residual sub-layers:

```
h = x + Attention(RMSNorm(x))
y = h + FFN(RMSNorm(h))
```

### RMSNorm

```
rms = sqrt(mean(x²) + eps)
out = x / rms * weight
```

It normalizes the last dimension and broadcasts over the rest. `eps` comes from
`llama.attention.layer_norm_rms_epsilon`; TinyLlama uses 1e-5.

### Q, K, V

```
q = Wq · xn    [H  · HD]
k = Wk · xn    [Hkv · HD]
v = Wv · xn    [Hkv · HD]
```

If `Hkv < H`, it's grouped-query attention. TinyLlama has 32 query heads and 4 KV
heads, so 8 query heads share each KV head. The mapping is `h / (H / Hkv)`.

### RoPE

Rotate each pair of values in a head by an angle that depends on position:

```
freq  = 1 / theta^(d / HD)
angle = position * freq
x0'   = x0*cos - x1*sin
x1'   = x0*sin + x1*cos
```

GGUF conversion permutes Q/K weights so adjacent pairs are the right thing to
rotate. If you use the "split halves" convention instead, you'll get plausible
but wrong output. The CPU code computes cos/sin directly; the GPU uses a
precomputed table so both paths agree.

### Causal attention

For each head and the current position `pos`:

```
scores[s] = q · k[s] / sqrt(HD)     for s = 0..pos
p         = softmax(scores)
out[d]    = Σ_s p[s] * v[s][d]
```

Only positions up to the current one are visible. That's the "causal" part.

### The FFN (SwiGLU)

```
gate = SiLU(xn · W1ᵀ)
up   = xn · W3ᵀ
down = (gate ⊙ up) · W2ᵀ
y    = h + down
```

`SiLU(v) = v * sigmoid(v)`.

### Residuals and the final norm

Add the sub-layer outputs back. After all layers, do a final RMSNorm and project
to logits:

```
logits = hidden · outputᵀ     [vocab]
```

At this point you have a vector of one score per token. The next step picks one.

---

## Step 6 — KV Cache

Without a cache, generating token `n` recomputes K and V for every previous
token. That's O(n²) work per step.

With a cache:

```cpp
keys_[layer]   : [1, Hkv, max_seq, HD]
values_[layer] : [1, Hkv, max_seq, HD]
```

Each new token writes its K/V at `pos` and attends over `0..pos`. Generation
becomes O(n) per token.

Memory:

```
2 × layers × Hkv × max_seq × HD × 4 bytes
```

TinyLlama: ~90 MB. Cheap compared to the weights.

The subtle part is positions. RoPE and cache writes use `start_pos + t`, and
attention reads `0..pos`. If you get `start_pos` wrong, the first token is fine
and everything after it drifts. That's a classic symptom.

---

## Step 7 — Sampling

Logits aren't tokens. Sampling turns one into the other.

1. **Temperature.** Divide logits by T. T→0 is greedy; T>1 is more random.
2. **Softmax.** Probabilities.
3. **Top-k.** Keep the k highest.
4. **Top-p.** Keep the smallest set whose cumulative probability reaches p.
5. **Repetition penalty.** Down-weight tokens already in the context.
6. **Draw.** Renormalize and sample.

`temperature <= 0` should short-circuit to argmax. I forgot that once and divided
by zero; the test passed by accident because of how NaN comparisons work. Don't
be like me.

Seeds: use `seed + generated.size()` so each step gets a different draw and the
whole run is still reproducible.

---

## Step 8 — Tokenization

The tokenizer loads vocab and merges from `tokenizer.json` or from the GGUF
metadata. GGUF is the default here because the data is already in the file.

Encoding, roughly:

1. Replace spaces with `▁` and add a dummy prefix.
2. Split into UTF-8 codepoints.
3. Repeatedly merge the adjacent pair with the lowest merge rank.
4. Map symbols to ids; unknown symbols become `<0xXX>` bytes.

Decoding reverses it: `▁` → space, byte tokens → raw bytes.

For `"The capital of France is"` you should get:

```
[1, 450, 7483, 310, 3444, 338]
```

If you get something else, the merge ranks or the prefix handling are wrong.

**Chat templates:** the tokenizer reads `tokenizer.chat_template` and recognizes
the common formats (Zephyr/TinyLlama, Llama-3, ChatML, Mistral, Gemma). Pass
`--chat` to wrap a single prompt, or use interactive mode. An unknown template
falls back to a simple `User:`/`Assistant:` format.

---

## Step 9 — The Generation Loop

```
logits = forward(prompt, start_pos=0)
for step in 1..max_tokens:
    token = sample(logits[-1])
    append(token)
    if token == eos: break
    logits = forward([token], start_pos=len(output)-1)
```

The prompt is processed once. After that, one token at a time.

On the CPU, `Model::generate` does exactly that. On the GPU, `generate_cuda`
does the same thing with device buffers and downloads only the logits.

---

## Step 10 — Build, Run, Measure

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/tests/run_tests

curl -L -o models/stories15M-q4_0.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf

./build/inference --device cuda --gpu-precision int16 \
  --tokens 40 --temp 0.8 models/stories15M-q4_0.gguf "Once upon a time"
```

You should see something like:

```
Device: CUDA (NVIDIA GeForce GTX 1650, ... MB free, int16 weights)
...
[40 tokens, ... tok/s decode, GPU init ... ms, prefill ... ms]
```

The decode number is the one that matters. The first run includes the upload.

---

## Step 11 — CUDA Without the Toolkit

This is the part I'm asked about most. You don't need `nvcc`.

The build does:

```
cuda_kernels.cu
  → clang++ -x cuda --cuda-device-only -nocudainc -nocudalib
  → PTX
  → embedded as a C string
  → libcuda.so.1 JITs it at runtime
```

There are no CUDA headers. The kernels use NVVM builtins:

```cpp
unsigned tid = __nvvm_read_ptx_sreg_tid_x();    // threadIdx.x
unsigned nt  = __nvvm_read_ptx_sreg_ntid_x();   // blockDim.x
unsigned bi  = __nvvm_read_ptx_sreg_ctaid_x();  // blockIdx.x
```

And inline PTX for math:

```cpp
asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); // exp2
```

The host side loads `libcuda.so.1` with `dlopen` and calls `cuModuleLoadData`,
`cuModuleGetFunction`, `cuLaunchKernel`. No `cudart`.

One driver quirk: `cuDevicePrimaryCtxRetain` returned a null context with a
success code here. The code uses `cuCtxCreate_v2` instead.

### INT16 weights

FP32 TinyLlama is ~4.3 GB and doesn't fit in 4 GB. So at upload time each linear
weight becomes INT16 with one scale per output row:

```
scale[r] = max|W[r, :]| / 32767
q[r, c]  = round(W[r, c] / scale[r])
y[r]     = scale[r] * Σ q[r, c] * x[c]
```

That's half the VRAM. Greedy output matches the FP32 CPU path. `--gpu-precision
fp32` is still there for models that fit.

The kernels themselves are in `src/ops/cuda_kernels.cu`: GEMV (FP32 and INT16),
RMSNorm, SiLU, mul, add, embedding, KV write, RoPE, and attention. If you want to
add one, the recipe is in [docs/development.md](docs/development.md).

---

## Step 12 — What I'd Optimize Next

In order of payoff:

1. **Quantized GPU matmul.** Keep Q4_K/Q6_K blocks in VRAM and dequantize inside
   the GEMV. This is what lets 2–3B models fit in 4 GB.
2. **Quantized host storage or mmap.** The FP32 host copy is the real RAM wall.
3. **CPU SIMD, blocking, threads.** 5–10× on the fallback path.
4. **Tiled attention** for long contexts.

Every one of these should come with a before/after number. If you can't measure
it, you don't know if it helped.

---

## Debugging Checklist

| Symptom | What it usually is |
|---|---|
| Garbage from token 0 | wrong tensor names, dequant, or dim reversal |
| Garbage only on some models | quant format, often Q6_K/subnormals |
| Right first token, drift later | KV positions or RoPE |
| Very slow first run | weight upload; decode is timed separately |
| `Tensor not found` | model isn't Llama-shaped |
| Instruct model is mediocre | wrong or missing chat template; try `--chat` / `--no-chat` |
| Interactive aborts | context length exceeded |

When I'm stuck, I check in this order:

1. Config (layers, dim, heads, vocab).
2. Tokenizer round-trip.
3. Weight stats against a reference.
4. Logits for a known prompt.
5. CPU vs GPU FP32 parity.
6. Only then the math.

---

## Glossary

- **GGUF** — file format: metadata, tensor descriptors, raw data.
- **Quantization** — fewer bits per weight plus scales.
- **K-quant** — 256-value super-block formats (Q2_K…Q8_K).
- **KV cache** — stored keys/values for past tokens.
- **GQA** — grouped-query attention; several Q heads share one KV head.
- **RoPE** — rotary position embedding applied to Q and K.
- **SwiGLU** — FFN with `SiLU(gate) * up`.
- **Top-p / nucleus** — sample from the smallest high-probability set.
- **BPE** — byte-pair encoding; merges frequent symbol pairs.
- **PTX** — NVIDIA's intermediate assembly; the driver JITs it.

---

## Where to go next

If you want the deeper material:

- [docs/architecture.md](docs/architecture.md) — the design and memory model.
- [docs/walkthrough.md](docs/walkthrough.md) — one prompt, every function.
- [docs/gguf-quantization.md](docs/gguf-quantization.md) — the format and the
  bugs.
- [docs/models.md](docs/models.md) — what to add next.
- [docs/development.md](docs/development.md) — build, test, profile, extend.

The best way to learn this is to change one thing and measure it. Swap the
activation. Add a quant format. Break the tensor layout and watch the logits turn
to noise, then fix it. That's the whole game.
