# End-to-End Walkthrough

Let's follow one prompt all the way through. I'll use a real command and the real
shapes it produces, so you can put breakpoints in the same places and see the
same numbers.

Here's the command:

```bash
./build/inference_cli --device cuda --gpu-precision int16 \
  --tokens 16 --temp 0 \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "The capital of France is"
```

And the output:

```
 Paris.
The capital of France is Paris.
```

The model is TinyLlama 1.1B. Keep these numbers nearby, because I'll refer to
them constantly:

```
layers      22
D           2048       (embedding width)
H           32         (query heads)
Hkv         4          (KV heads, so 8 query heads per KV head)
HD          64         (head dim)
FFN         5632
vocab       32000
context     2048
rope theta  10000
```

## Stage 0: the CLI

Everything starts in `src/main.cpp`. Argument parsing is boring, which is the
point. The defaults that matter:

```cpp
config.max_tokens         = 512;
config.temperature        = 0.7f;
config.top_k              = 40;
config.top_p              = 0.95f;
config.repetition_penalty = 1.1f;
config.gpu_int16          = true;   // from --gpu-precision int16
```

`--temp 0` is what turns this into greedy decoding later.

## Stage 1: opening the GGUF

```cpp
GGUFParser parser(model_path);
```

The constructor reads the file in three passes. First the header: magic, version
3, 201 tensors, 23 metadata entries. Then the metadata, then the tensor
descriptors.

The metadata I actually use for TinyLlama:

```
general.architecture                     llama
llama.block_count                        22
llama.embedding_length                   2048
llama.attention.head_count               32
llama.attention.head_count_kv            4
llama.rope.dimension_count               64
llama.rope.freq_base                     10000.0
llama.attention.layer_norm_rms_epsilon   1e-5
llama.context_length                     2048
tokenizer.ggml.tokens                    32000 strings
tokenizer.ggml.merges                    61249 strings
```

Tensor descriptors come next. Each one is:

```
name, n_dims, dims[], type, offset
```

and the data section starts after padding to `general.alignment` (32 here).
That alignment detail is not optional. Forget it and every weight is shifted by
four bytes. I know from experience.

If you want to poke at this yourself:

```cpp
parser.tensors();
parser.find_tensor("blk.0.attn_q.weight");
parser.get_metadata<uint32_t>("llama.block_count");
```

## Stage 2: weights

```cpp
auto weights = load_model_weights(parser, Device::cpu());
```

This is where the file turns into something the model can use.

TinyLlama is mixed-quant. Some tensors are Q4_K, some are Q6_K. The loader reads
each one, dequantizes it to FP32, reverses the dims, and stores it.

```
token_embd.weight          Q4_K   [2048, 32000] → [32000, 2048]
output.weight              Q6_K   [2048, 32000] → [32000, 2048]
blk.N.attn_q.weight        Q4_K   [2048, 2048]  → [2048, 2048]
blk.N.attn_k.weight        Q4_K   [2048, 256]   → [256, 2048]
blk.N.attn_v.weight        Q6_K   [2048, 256]   → [256, 2048]
blk.N.attn_output.weight   Q4_K   [2048, 2048]  → [2048, 2048]
blk.N.ffn_gate.weight      Q4_K   [2048, 5632]  → [5632, 2048]
blk.N.ffn_up.weight        Q4_K   [2048, 5632]  → [5632, 2048]
blk.N.ffn_down.weight      Q6_K   [5632, 2048]  → [2048, 5632]
norms                      F32    [2048]        → [2048]
```

If `output.weight` is missing, the loader copies the embedding and sets
`output_tied = true`. TinyLlama has a separate output tensor, so that doesn't
trigger here.

At the end of this stage you've got about 4.4 GB of FP32 weights in RAM. Then:

```cpp
Model model(std::move(weights), Device::cpu());
```

The move matters. A copy would double that.

## Stage 3: tokenizer

The CLI tries `tokenizer.json` first, then falls back to the vocab and merges
inside the GGUF. For TinyLlama the fallback is what runs.

```cpp
tokenizer.load_from_gguf(parser);
```

Encoding `"The capital of France is"` with BOS gives:

```
[1, 450, 7483, 310, 3444, 338]
```

That's the correct Llama tokenization, which is a nice sanity check. If you get
something wildly different, the merge ranks or the `▁` handling are off.

The encoder does four things:

1. Replace spaces with `▁` and add a dummy prefix.
2. Split into UTF-8 codepoints.
3. Merge adjacent pairs by lowest merge rank.
4. Map symbols to ids, with `<0xXX>` byte fallback.

Decoding reverses it. `▁` becomes a space, byte tokens get reassembled.

## Stage 4: picking a device

In `main.cpp`:

```cpp
if (device == "cuda" || device == "auto") {
    if (cuda_model_available()) {
        if (device == "auto") {
            size_t need = cuda_required_bytes(model, config.gpu_int16);
            use_cuda = need + 96MB < cuda::free_memory();
        } else {
            use_cuda = true;
        }
    }
}
```

For TinyLlama the estimate is about 2.3 GB and the card has about 3.1 GB free, so
auto picks CUDA.

If you force `--gpu-precision fp32`, the estimate jumps to ~4.3 GB and the run
fails with a message that tells you both numbers. Auto mode would fall back to
CPU instead.

## Stage 5A: the CPU path

I'll do the CPU path first because it's the reference. Select it explicitly with
`--device cpu`.

```cpp
model.generate(prompt_tokens, config, kv_cache);
```

### Prefill

`generate` builds a `[1, 6]` int32 input and calls `forward` once:

```cpp
forward(input, &kv_cache, /*start_pos=*/0);
```

Inside `forward`:

1. Embedding lookup → `x[1, 6, 2048]`.
2. 22 transformer blocks.
3. Final RMSNorm.
4. Output projection → `logits[1, 6, 32000]` (about 750 KB).

### One block

The block is the usual sandwich:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Attention processes one token at a time. For token `t`, position `pos = t`:

```
xn = RMSNorm(x)
q  = Wq · xn                32×64 values
k  = Wk · xn                4×64 values
v  = Wv · xn                4×64 values

RoPE(q, k, pos)
write k, v into the cache at pos

for each of the 32 query heads:
    kv_head = h / 8
    scores[s] = q[h] · k_cache[kv_head][s] / sqrt(64), s = 0..pos
    p = softmax(scores)
    out[h] = Σ p[s] · v_cache[kv_head][s]

x = x + Wo · out
```

The score array is a stack `float scores[4096]`, which is why the CPU path is
effectively capped at 4096 tokens regardless of what the metadata says. If you
want longer contexts on CPU, that array has to move to the heap.

The FFN is SwiGLU, also with stack arrays:

```
gate[5632] = SiLU(W1 · xn)
up[5632]   = W3 · xn
gate      *= up
down[2048] = W2 · gate
x         += down
```

### First token

`generate` takes the last prompt row, `logits[0, 5, :]`, copies it to a 1-D
tensor, applies repetition penalty, and samples. With `--temp 0` that's just
argmax. For this prompt the winner is token `3681`, which decodes to `"▁Paris"`
and prints as `" Paris"`.

### Decode loop

After that, each step is:

```
forward([last_token], start_pos=pos)
sample
append
stop on EOS or max_tokens
```

On this machine the CPU path is roughly 1 tok/s for TinyLlama. Slow, but every
number is visible.

## Stage 5B: the GPU path

`generate_cuda` lives in `src/models/model_cuda.cpp`. The structure is the same,
but the data lives in device buffers.

### First call: building the GPU state

`get_state(model, /*int16=*/true)`:

1. Estimate VRAM and check it against `cuda::free_memory()`.
2. For every linear weight, `quantize_i16`:
   - per output row, `scale = max|w| / 32767`,
   - int16 values into one buffer, float scales into another,
   - upload both.
3. Upload the FP32 norms.
4. Build the RoPE table on the CPU and upload it.
5. Allocate activations and 22 KV caches.
6. Cache the state by `&model`.

This is the ~6.7 s "GPU init" you see in the timing line. It reads 4.4 GB of host
weights and writes 2.2 GB to the card. It only happens once per process; in
interactive mode the next turn skips it.

### Every token

```cpp
embedding_i16(tok_emb, scale, token, x, D)

for layer in 0..21:
    rmsnorm(x, attn_norm, xn)
    gemv_i16(wq, xn, q)       // 2048×2048
    gemv_i16(wk, xn, k)       // 256×2048
    gemv_i16(wv, xn, v)       // 256×2048
    rope(q, k, table, pos)
    kv_write(kc[layer], vc[layer], k, v, pos)
    attention(q, kc, vc, attn)  // 32 blocks, 128 threads each
    gemv_i16(wo, attn, proj)
    add(x, proj)

    rmsnorm(x, ffn_norm, xn)
    gemv_i16(w1, xn, gate)    // 5632×2048
    silu(gate)
    gemv_i16(w3, xn, up)      // 5632×2048
    mul(gate, up)
    gemv_i16(w2, gate, proj)  // 2048×5632
    add(x, proj)

rmsnorm(x, out_norm, xn)
gemv_i16(out_w, xn, logits)   // 32000×2048
sync()
download(logits)
```

Each `gemv_i16` reads a whole matrix. For TinyLlama that's about 2.2 GB of weight
traffic per token, which is why the GPU version is memory-bound rather than
compute-bound.

The RoPE table is built on the CPU for a reason: the GPU's `sin`/`cos` aren't
bit-identical to the CPU's, and I wanted the two paths to agree. The table is
`max_seq × HD/2 × 2` floats, about half a megabyte.

### Sampling on the GPU path

Logits are downloaded, copied into a `Tensor`, penalized, and sampled with the
same function the CPU path uses. There's no separate GPU sampler. Same seed, same
logic, same result.

### Timing

`GenerationStats` records init, prefill, and decode separately. That's why the
first run can report a low number and later runs look much better. The CLI prints
the decode rate, not the total.

## Stage 6: text

```cpp
std::vector<int> new_tokens(generated.begin() + prompt.size(), generated.end());
std::string text = tokenizer.decode(new_tokens);
```

BOS/EOS/PAD are skipped, `▁` becomes a space, byte tokens are reassembled. For
the greedy run above, the first token is `"▁Paris"` and the rest follows.

## Where I'd put breakpoints

| If you want to check… | Stop in | Look at |
|---|---|---|
| Parsing | `GGUFParser::parse_tensor_infos` | `tensors_`, `tensor_data_offset_` |
| Metadata | `GGUFParser::parse_metadata` | the key/value pair |
| Dequantization | `load_model_weights::load_tensor` | first 8 floats, std/min/max |
| Tokenization | `Tokenizer::encode` | ids and the round-trip decode |
| Prefill | `Model::forward` | `x`, `logits` shapes |
| Attention | `Model::attention_forward` | `q_local`, `scores`, cache |
| Sampling | `sample_top_k_top_p` | probabilities and chosen id |
| GPU upload | `quantize_i16` | scale for the first row |
| GPU math | `forward_token` | which buffer goes to which kernel |

When something looks wrong, I go in this order: tokenizer, logits for a known
prompt, weight stats, then the math. It sounds obvious, but it saves a lot of
time compared to staring at the attention kernel.

## A useful sanity check

```bash
./build/debug_logits models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "The capital of France is"
```

After the Q6_K fix, the top tokens are:

```
#0  id=3681  logit=12.92  ' Paris'
#1  id=5982  logit=12.38  ' located'
#2  id=29901 logit=10.62  ':'
```

If you see multilingual junk like `européenne` or `Джерела`, stop and check
dequantization. That's exactly what the subnormal-fp16 bug looked like.
