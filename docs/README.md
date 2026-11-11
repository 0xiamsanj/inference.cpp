# inference.cpp docs

I built this thing to answer a question I couldn't shake: what actually happens
between typing a prompt and getting text back? Not the hand-wavy version. The
real one, with bytes moving around.

So there's no llama.cpp, no PyTorch, no BLAS hiding the interesting parts. The
code is small enough to read, and every layer prints something you can check.

Fair warning: the CPU path is slow on purpose. I wanted the math visible before
I made it fast. The GPU path is the opposite — it exists because a 1.1B model on
a 4 GB card is a fun constraint.

## Start here

If you're new to the repo, read in this order. The first three are the ones that
matter.

1. **[architecture.md](architecture.md)** — how the pieces fit, how memory is
   laid out, and why the CPU and GPU paths look different.
2. **[walkthrough.md](walkthrough.md)** — one prompt traced through every
   function. This is the one I'd read with the code open.
3. **[gguf-quantization.md](gguf-quantization.md)** — the file format and the
   quant formats. Also where the best bug stories live.
4. **[models.md](models.md)** — what runs today, what doesn't, and what it takes
   to add the next family.
5. **[development.md](development.md)** — build flags, tests, profiling, and how
   to add an op or a kernel without breaking things.
6. **[../TUTORIAL.md](../TUTORIAL.md)** — the beginner-friendly version. Concepts
   first, math second.

## The 30-second version

```
GGUF file
  → parse metadata + tensor descriptors
  → dequantize to FP32 and reverse the dims
  → tokenize the prompt
  → run the transformer (CPU scalar, or GPU kernels)
  → sample a token
  → detokenize
  → repeat
```

That's it. The rest is details, and the details are where the fun is.

## What this engine actually does

- Loads GGUF models that use the Llama architecture.
- Runs them on the CPU with plain loops.
- Runs them on an NVIDIA GPU without needing the CUDA toolkit: `clang++` turns
  the kernels into PTX, the PTX gets baked into the binary, and the driver JITs
  it at runtime.
- Stores GPU weights as INT16 by default. That's what makes TinyLlama fit on a
  4 GB GTX 1650.
- Handles Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, and Q6_K tensors.
- Does the usual transformer stuff: RMSNorm, RoPE, GQA, SwiGLU, KV cache,
  top-k/top-p sampling.

## What it doesn't do yet

- Only the Llama family. Qwen, Gemma, Phi, and friends need code changes.
- Chat templates cover the common formats (Zephyr/TinyLlama, Llama-3, ChatML,
  Mistral, Gemma), but this is not a general Jinja interpreter.
- No batching. The GPU processes prompts one token at a time.
- The CPU path is single-threaded and scalar.
- Several GGUF quant formats are still unimplemented.

I keep this list honest because half the value of a small engine is knowing
where the edges are. [models.md](models.md) has the detailed version.

## How it fits together

```
main.cpp
  ├─ GGUFParser          reads the file
  ├─ load_model_weights  dequantizes and rearranges
  ├─ Tokenizer           text ↔ ids
  ├─ Model               CPU forward + KV cache
  └─ generate_cuda       GPU forward + INT16 weights
        ├─ ops           math on the CPU
        ├─ cuda_backend  buffers and kernel launches
        └─ cuda_kernels  the actual device code
```

The dependency arrow only points one way. The model layer knows nothing about
CUDA headers, and the CUDA layer knows nothing about GGUF. That separation is
the main reason the code stays readable.

## Numbers from this machine

Intel i5-9300HF, 15 GB RAM, GTX 1650 4 GB:

| Model | CPU decode | GPU INT16 decode | First GPU upload |
|---|---:|---:|---:|
| stories15M | ~54 tok/s | ~459 tok/s | ~190 ms |
| TinyLlama-1.1B | ~1 tok/s | ~29 tok/s | ~6.7 s |

The GPU numbers are decode-only. They don't include the first upload or prompt
prefill, because including those makes the first run look much worse than it is.

## One thing to remember

The tensor layout. GGUF stores `ne[0]` contiguously, so a weight written as
`[in, out]` is already row-major `[out, in]` on disk. Reverse the shape, keep the
bytes. I lost a day to this. Don't transpose.
