# Models: What Works, What Doesn't, and What to Add

Short version: this engine is a **Llama-family runtime**. If a GGUF uses
`llama.*` metadata, the Llama block math, and one of the quant types I
implemented, it'll probably run. Anything else needs code.

The interesting part is how little code "anything else" often is. Qwen2 is
mostly a metadata prefix and some bias tensors. Gemma is a different activation
and a couple of extra norms. Whisper is a whole new subsystem. I'll sort them by
how much work they actually are.

## What runs today without touching the code

These all use the same block: RMSNorm → GQA attention with RoPE → SwiGLU FFN.

- **Llama 1, 2, 3, 3.1, 3.2 (1B/3B)**. Short contexts are fine. Long-context
  RoPE scaling is ignored, so don't trust 128k on a model that needs it.
- **TinyLlama 1.1B**. This is the one I verified end to end on the GPU.
- **Mistral 7B**. Same math. I ignore the sliding window, which is a fine
  approximation for short prompts and wrong for long ones.
- **Code Llama**.
- **Vicuna, WizardLM, Zephyr, OpenChat**. They're Llama fine-tunes.
- **SmolLM / SmolLM2**.
- **Yi**. llama.cpp converts it with `llama.*` keys.
- **stories15M / stories110M**. The tiny models I use for smoke tests.

Two caveats that aren't about the model family:

1. **Chat templates cover the common formats.** Zephyr/TinyLlama, Llama-3,
   ChatML, Mistral, and Gemma are detected from `tokenizer.chat_template`.
   Unknown templates fall back to a simple `User:`/`Assistant:` format.
2. **Host RAM is the real limit.** Weights are dequantized to FP32, so you need
   `4 × params` bytes of RAM. TinyLlama is 4.4 GB. A 3B model would need 12 GB
   before you even touch VRAM.

## Why other families fail

There are three walls, and they're independent:

1. **Metadata prefix.** `get_block_count()` and friends hardcode `llama.`.
   Qwen2 says `qwen2.block_count`. Easy fix.
2. **Tensor names.** The loader knows llama.cpp's `blk.N.*` names and Hugging
   Face's `model.layers.N.*` names. Falcon has fused `attn_qkv`, Phi has its own
   naming, etc.
3. **Block math.** Biases, different activations, different norms, ALiBi,
   sliding windows, MoE, cross-attention. This is the real work.

I'd fix them in that order.

## The matrix

Effort is my honest estimate for this codebase with tests: **S** under a day,
**M** a few days, **L** a couple of weeks, **XL** more than that.

The GPU column assumes INT16 weights, ~2 bytes per parameter, and about 3 GB of
free VRAM. It does not account for host RAM, which is usually the tighter
constraint here.

| Family | GGUF arch | What's missing | Effort | Fits 4 GB? |
|---|---|---|---|---|
| Llama / TinyLlama / Mistral | `llama` | nothing | — | yes, up to ~1.4B |
| **Qwen2 / Qwen2.5** | `qwen2` | arch prefix, QKV biases | **S–M** | yes, 0.5B–1.5B |
| Llama 3.x long context | `llama` | RoPE scaling | **S** | yes |
| Mistral sliding window | `llama` | window mask | S | yes |
| **Gemma / Gemma2** | `gemma`, `gemma2` | GeGLU, embedding scale, extra norms, soft-cap | **M** | 2B is ~4 GB, borderline |
| **Phi-2** | `phi2` | fused QKV, biases, LayerNorm, partial RoPE | **M–L** | 2.7B, ~5.4 GB, no |
| **Phi-3 / 3.5** | `phi3` | fused QKV, biases, Su-RoPE | **M–L** | 3.8B, no |
| Falcon 7B | `falcon` | fused QKV, MQA, ALiBi, parallel block | **L** | no |
| GPT-2 | `gpt2` | LayerNorm, GELU, learned positions, conv layout | **M–L** | small ones, yes |
| GPT-NeoX / Pythia | `gptneox` | LayerNorm, partial RoPE, GELU | M–L | up to ~1.4B |
| **BERT / RoBERTa** | `bert` | bidirectional attention, pooling | **M** | yes |
| **E5 / GTE / MiniLM** | `bert` | BERT + pooling + normalize | M | yes |
| Cross-encoder rerankers | `bert` | BERT + classification head | M | yes |
| Mixtral / MoE Llama | `llama` + expert tensors | router, top-k experts | **L** | no |
| DeepSeek-MoE / V2 | `deepseek2` | MLA attention, MoE | **XL** | no |
| Qwen2-VL / LLaVA | `qwen2vl`, `llava` | vision tower + projector | **XL** | no |
| Whisper | `whisper` | audio frontend + encoder-decoder | **XL** | tiny/base maybe |
| CLIP / ViT | `clip` | vision transformer | **L** | yes |
| Mamba / RWKV | `mamba`, `rwkv` | non-attention sequence mixing | **XL** | small ones |

If you only care about running good small chat models on this card, the list is
short: **Qwen2.5-0.5B, Qwen2.5-1.5B (borderline), Llama-3.2-1B, SmolLM2**. Get
Qwen2 working and you've covered the useful part of the market.

## The cheap wins, in order

### 1. Make the metadata prefix dynamic (S)

Replace every hardcoded key with the architecture name:

```cpp
std::string arch = parser.get_arch_name();
std::string key  = arch + ".block_count";
```

Some families have irregular names, so put the exceptions in a small table. This
costs almost nothing and unblocks a lot.

### 2. Add bias tensors (S)

Qwen2, Phi, and several others have `attn_q.bias`, `attn_k.bias`, `attn_v.bias`.
Sometimes FFN biases too.

You need:

- optional bias tensors in `TransformerBlockWeights`,
- a bias add after each projection on the CPU,
- an `add_bias` kernel on the GPU,
- optional loads in `load_model_weights`.

That's it. No math changes.

### 3. Qwen2 / Qwen2.5 (S–M)

Do 1 and 2, then use `qwen2.*` metadata. Tensor names match Llama. RoPE base is
read from metadata already. Some variants have a sliding window; ignore it at
first.

Qwen2.5-0.5B fits easily. 1.5B is around 3 GB of INT16 weights plus KV, so it's
borderline on a 4 GB card with a desktop running.

### 4. Chat templates — done

The tokenizer reads `tokenizer.chat_template`, detects the common formats, and
renders them with the right markers:

- Zephyr/TinyLlama (`<|system|>`, `<|user|>`, `<|assistant|>`)
- Llama-3 (`<|start_header_id|>`)
- ChatML (`<|im_start|>`)
- Mistral (`[INST]`)
- Gemma (`<start_of_turn>`)

Interactive mode applies the template automatically; `--chat` and `--no-chat`
override it, and `--system` adds a system message. A full Jinja interpreter is
still not implemented, so unknown templates fall back to a generic format.

### 5. RoPE scaling (S)

Read:

```
llama.rope.scaling.type
llama.rope.scaling.factor
llama.rope.scaling.original_context_length
```

and adjust the frequency/position when building the cos/sin table. The table is
built on the CPU for both paths, so you only change one place. This is required
for Llama 3.x beyond 8k.

### 6. Sliding-window attention (S)

Add a window to the mask: token `s` is visible only if `s >= pos - window + 1`.
Mistral and some Qwen variants use this. Below the window size it changes
nothing.

### 7. More quant formats (S each)

Q2_K, Q3_K, Q5_K, Q8_K, Q4_1, Q8_1, BF16. Pure mechanical work. See
[gguf-quantization.md](gguf-quantization.md#adding-another-quant-format).

This doesn't unlock new architectures, but it widens the set of downloadable
models that will load.

## The medium stuff

### Gemma / Gemma 2

Four changes: GeGLU (`gelu(gate) * up`), embedding scaling by `sqrt(D)`, extra
norms, and Gemma 2's logit soft-capping plus alternating sliding windows. Gemma
2B is ~4 GB of INT16 weights, so it's not a good fit for this card anyway.

### Phi-2 / Phi-3

Fused QKV (`attn_qkv.weight`) that you split into three, QKV biases, LayerNorm
instead of RMSNorm for Phi-2, and partial/Su-scaled RoPE. The math is still
recognizable, but the loader and block forward both need work.

### Falcon, GPT-2, GPT-NeoX

These need a different block. LayerNorm instead of RMSNorm, GELU instead of
SiLU, Falcon's ALiBi and parallel attention/FFN, GPT-2's learned position
embeddings. I'd add a `BlockType` enum and branch, or write separate forward
functions. Not hard, just more surface area.

### Embeddings and rerankers

This is the second task I'd add, and it's easier than it looks. BERT is the same
kind of stack, but:

- attention is bidirectional (drop the causal mask),
- you need a padding mask,
- you pool (CLS or mean) and optionally L2-normalize,
- the output is a vector, not logits.

BERT-base is ~110M parameters. About 220 MB INT16. It fits anywhere. Add a
`--task embed` mode and you've got a useful second product out of the same
engine.

## The big stuff

### MoE (Mixtral, DeepSeek)

You need the router (`ffn_gate_inp`), expert tensors, softmax over experts,
top-k selection, and a weighted sum of expert FFNs. Memory scales with total
experts rather than active ones, so quantization is mandatory. DeepSeek V2 also
uses MLA attention, which is a different attention block entirely.

### Whisper

Audio decode, resample to 16 kHz, mel spectrogram, a conv stem, a bidirectional
encoder, and a decoder with cross-attention. The tokenizer is a multilingual
BPE. This is a separate pipeline that happens to share tensor and GGUF code.

### Vision-language

Image decode/resize/normalize, a vision encoder, a projector into the token
embedding space, and then the LLM. Qwen2-VL adds dynamic resolution and M-RoPE.
Large, but the LLM part is already here.

### Mamba / RWKV

These replace attention with a recurrence or state-space mixer. New state
management, new kernels, no KV cache in the usual sense. Fun, but it's a
different engine.

## What actually fits on this machine

The GPU limit is mostly `2 × params` for INT16 weights:

| Params | INT16 weights | 4 GB card? |
|---|---:|---|
| 135M | 0.27 GB | easy |
| 360M | 0.72 GB | easy |
| 0.5B | 1.0 GB | yes |
| 1.1B | 2.2 GB | yes |
| 1.5B | 3.0 GB | borderline |
| 2B | 4.0 GB | no |
| 3B | 6.0 GB | no |
| 7B | 14 GB | no |

The host limit is worse because of the FP32 dequantization:

| Params | Host FP32 | ~6 GB free? |
|---|---:|---|
| 1.1B | 4.4 GB | yes |
| 1.5B | 6.0 GB | borderline |
| 3B | 12 GB | no |
| 7B | 28 GB | no |

So on this laptop, **TinyLlama and Llama-3.2-1B are the sweet spot today**.
Qwen2.5-0.5B after the bias/prefix work. If you want 2–3B models, the first
problem to solve isn't VRAM, it's host RAM.

Two ways out:

- keep the GGUF's Q4_K/Q6_K blocks in RAM and dequantize on use,
- or mmap the file and stream/dequantize per matmul.

Either one changes the memory math completely.

## If I were adding a new family

I'd use Qwen2 as the template, because it hits every common problem but none of
the exotic ones.

### Step 1: config

In `gguf_parser.cpp`, derive keys from `general.architecture`. Add a small table
for irregular names.

### Step 2: tensors

In `load_model_weights`, add optional biases:

```cpp
layer.wq      = try_load({prefix + "attn_q.weight", ...});
layer.wq_bias = try_load_optional({prefix + "attn_q.bias"});
```

Keep the empty-tensor convention for "not present".

### Step 3: ops

Bias add on CPU and GPU. If the family needs GeGLU or LayerNorm, those are small
too (CPU LayerNorm already exists; GPU would need a kernel).

### Step 4: block forward

Add the bias after the projection. Branch on the activation if it's not SwiGLU.
Adjust the mask if there's a sliding window.

### Step 5: verification, not vibes

Run `debug_logits` on a prompt. Compare the first few greedy tokens against
llama.cpp or Hugging Face. If the first token matches and the next ten match,
you're probably done. If token zero is wrong, it's loading or tokenization, not
sampling.

### Step 6: document

Add the family to this file and write down the caveats. Future you will forget
which parts are approximated.

## Other tasks worth building

The tensor/GGUF/quant foundation is more reusable than the Llama block:

| Task | Model | New work |
|---|---|---|
| Embeddings | BERT, E5, GTE | non-causal attention, pooling |
| Reranking | cross-encoder | pooling + head |
| Classification | encoder + head | pooling + head |
| Speech | Whisper | audio frontend, encoder-decoder |
| Image-text | CLIP, LLaVA, Qwen2-VL | vision encoder, projector |
| Vision | ViT | patch embedding, pooling |

If you want a second task with the least new code, pick **BERT embeddings**.
It reuses GGUF, tokenization, tensors, and most of the math, and it doesn't need
a new tokenizer.

## My recommendation

Don't try to support everything. Do this:

1. arch-key generalization,
2. bias tensors,
3. Qwen2/2.5,
4. RoPE scaling.

That's a few days of work and it turns the engine from "my TinyLlama thing" into
something you can point at a decent chunk of the small-model ecosystem.

Then, if you want bigger models, go back to the memory problem: quantized host
storage and on-GPU Q4_K/Q6_K matmul. That's the real ceiling, not the math.
