# GGUF and Quantization

GGUF is the file format llama.cpp uses, and it's a good one to learn because it's
mostly obvious until it isn't. The header is simple. The metadata is a tagged
key/value store. The tensors are a list of descriptors followed by a blob of
bytes. Then there are about four details that will ruin your afternoon.

I'm going to explain the format, then the quant formats I implemented, then the
four details, because those details are the whole game.

## The file, top to bottom

```
magic "GGUF"           4 bytes
version                uint32
tensor_count           uint64
metadata_kv_count      uint64

metadata pairs
  key:   uint64 length + bytes
  type:  uint32
  value: depends on type (arrays recurse)

tensor descriptors
  name:   uint64 length + bytes
  n_dims: uint32
  dims:   uint64[n_dims]
  type:   uint32
  offset: uint64   (relative to the data section)

padding to general.alignment (usually 32)

raw tensor data
```

That's it. The data section is one big blob; each descriptor's `offset` points
into it.

## Detail #1: metadata types are not tensor types

The metadata value enum is its own thing, and it does not match the tensor dtype
enum. I got this wrong at first and spent an evening reading strings out of
random offsets.

```
0  UINT8      4  UINT32     8  STRING
1  INT8       5  INT32      9  ARRAY
2  UINT16     6  FLOAT32   10  UINT64
3  INT16      7  BOOL      11  INT64
                            12  FLOAT64
```

An array is `element_type (uint32)`, then `length (uint64)`, then the elements.

The tensor dtype numbers are different:

```
0 F32    2 Q4_0   6 Q5_0   8 Q8_0   10 Q2_K   12 Q4_K   14 Q6_K
1 F16    3 Q4_1   7 Q5_1   9 Q8_1   11 Q3_K   13 Q5_K   15 Q8_K
```

Yes, that's confusing. Keep both tables somewhere visible.

## Detail #2: tensor descriptor field order

It's `name, n_dims, dims[], type, offset`. Not name, type, dims. The type comes
after the dims. If you put it before, the first dimension gets interpreted as the
type and everything after that is noise. That was bug number two.

## Detail #3: the data section is aligned

The raw tensor data does **not** start immediately after the last descriptor. It
starts at the next multiple of `general.alignment`, which is 32 in every model
I've seen.

In this repo, `tensor_data_offset_` is computed like this:

```cpp
tensor_data_offset_ = file_.tellg();
uint32_t alignment = 32;
try { alignment = get_metadata<uint32_t>("general.alignment"); } catch (...) {}
tensor_data_offset_ = (tensor_data_offset_ + alignment - 1) / alignment * alignment;
```

If you skip that, every block is shifted by a few bytes. The scales come from
quant bytes, the weights come out around 1e5, and the output is nonsense. Bug
number three.

## Detail #4: dims are reversed, but the bytes are not

This one is the most expensive to get wrong because the result still looks like a
model.

GGML stores `ne[0]` contiguously. A linear weight written with file dims
`[in, out]` is therefore already C row-major `[out][in]`. Each output row of
`in` values is contiguous in the file.

So the loader does this:

```cpp
Shape shape(info->shape.rbegin(), info->shape.rend());
Tensor out(DataType::F32, shape, device);
std::memcpy(out.data(), flat.data(), flat.byte_size());
```

Shape reversed. Bytes untouched. Anything else is a transpose, and a transpose
here scrambles the model while keeping every number finite. I did the transpose
first. Don't.

## Exact sizes, or you'll seek into the wrong place

`type_size()` returns an effective byte size for bookkeeping, and for quantized
types it lies (it says 1 byte per element). For file I/O you need the real block
size. That's what `type_nbytes()` is for.

| Type | Values/block | Bytes/block |
|---|---:|---:|
| Q4_0 | 32 | 18 |
| Q4_1 | 32 | 20 |
| Q5_0 | 32 | 22 |
| Q5_1 | 32 | 24 |
| Q8_0 | 32 | 34 |
| Q8_1 | 32 | 36 |
| Q2_K | 256 | 84 |
| Q3_K | 256 | 110 |
| Q4_K | 256 | 144 |
| Q5_K | 256 | 176 |
| Q6_K | 256 | 210 |
| Q8_K | 256 | 292 |

All quantized element counts are divisible by their block size. If they're not,
the file is broken or your descriptor is.

## What I implemented

Implemented: Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K. Plus F32 and F16 handling in the
loader.

Not implemented: Q4_1, Q8_1, Q2_K, Q3_K, Q5_K, Q8_K. They throw a clear error.

I did the ones TinyLlama and stories15M needed. Adding the rest is mechanical
once you understand the pattern.

## Q4_0: 32 values, 18 bytes

```
[ d: fp16 ][ qs: 16 bytes ]
```

```cpp
value[j]      = ((qs[j] & 0x0F) - 8) * d;  // low nibble
value[j + 16] = ((qs[j] >> 4)   - 8) * d;  // high nibble
```

Two things tripped me up here. The scale is fp16, not fp32. And the two nibbles
are 16 elements apart, not adjacent. If you assume adjacent, the values are still
in a plausible range, which is why it can hide for a while.

## Q8_0: 32 values, 34 bytes

```
[ d: fp16 ][ qs: 32 × int8 ]
value[j] = qs[j] * d
```

The easiest one.

## Q5_0 and Q5_1

Q5_0 is 22 bytes:

```
[ d: fp16 ][ qh: uint32 ][ qs: 16 bytes ]

q_lo = (qs[j] & 0xF) | (((qh >> j)        & 1) << 4)
q_hi = (qs[j] >> 4)  | (((qh >> (j + 16)) & 1) << 4)
value[j]      = (q_lo - 16) * d
value[j + 16] = (q_hi - 16) * d
```

Q5_1 is 24 bytes and adds a minimum:

```
[ d: fp16 ][ m: fp16 ][ qh: uint32 ][ qs: 16 bytes ]
value = q * d + m
```

## Q4_K: 256 values, 144 bytes

This is the one modern models actually use, so it's worth understanding.

```
[ d: fp16 ][ dmin: fp16 ][ scales: 12 bytes ][ qs: 128 bytes ]
```

Eight sub-blocks of 32 values. Each 32-byte chunk of `qs` holds 32 low nibbles,
then 32 high nibbles. The 12 scale bytes pack 16 six-bit scale/min pairs.

Unpacking a pair (this is llama.cpp's `get_scale_min_k4`):

```cpp
if (j < 4) {
    sc = scales[j] & 63;
    m  = scales[j + 4] & 63;
} else {
    sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    m  = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
}
```

Then, for each 64 values:

```cpp
get_scale_min_k4(is + 0, scales, &sc, &m);
float d1 = d * sc, m1 = dmin * m;
get_scale_min_k4(is + 1, scales, &sc, &m);
float d2 = d * sc, m2 = dmin * m;

for (l = 0; l < 32; ++l) y[l]      = d1 * (q[l] & 0xF) - m1;
for (l = 0; l < 32; ++l) y[l + 32] = d2 * (q[l] >> 4)  - m2;

q += 32;
is += 2;
```

Both `d` and `dmin` are fp16. That matters later.

## Q6_K: 256 values, 210 bytes

This is the format that produced the best bug in the project.

```
[ ql: 128 bytes ][ qh: 64 bytes ][ scales: 16 × int8 ][ d: fp16 ]
```

For each 128-value chunk:

```cpp
q1 = ((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
q3 = ((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
q4 = ((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

y[l]      = d * sc[is + 0] * q1;
y[l + 32] = d * sc[is + 2] * q2;
y[l + 64] = d * sc[is + 4] * q3;
y[l + 96] = d * sc[is + 6] * q4;
```

`is = l / 16`, and after each chunk you advance `ql += 64`, `qh += 32`, `sc += 8`.

## The subnormal-fp16 bug

TinyLlama was producing multilingual garbage while stories15M worked fine. Both
CPU and GPU. So it wasn't the CUDA code.

I dumped tensor stats and saw something odd: `output.weight` had the right
min/max, but its standard deviation was about 10× smaller than the reference. The
extremes were right; the bulk of the values were wrong.

The cause was the fp16 converter. Q6_K super-block scales are fp16, and
TinyLlama's output-projection scales are **subnormal** fp16 values. My converter
did this:

```cpp
if (exp == 0) f = sign;   // wrong
```

Subnormals became zero. Whole Q6_K blocks turned into zeros. The output
projection was mostly empty, which is exactly why the model still produced
finite logits and confident-looking multilingual junk.

The fix is boring once you see it:

```cpp
if (exp == 0) {
    float v = std::ldexp((float)mant, -24);
    return (h & 0x8000u) ? -v : v;
}
```

This is why `debug_logits` exists. Min/max alone would have looked fine. The
standard deviation is what gave it away.

## Checking your dequantizer

Don't trust your own formula. Compare it to the reference `gguf` package.

```bash
pip install gguf numpy

python3 - <<'EOF'
import gguf, numpy as np
r = gguf.GGUFReader('models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
t = next(x for x in r.tensors if x.name == 'output.weight')
raw = np.asarray(t.data).view(np.uint8).reshape(-1)
out = gguf.quants.dequantize(raw, t.tensor_type)
print(t.tensor_type.name, out[:8])
print(out.min(), out.max(), out.mean(), out.std())
EOF
```

Then dump the same tensor with this engine and compare. The first eight values,
min, max, mean, and std should match exactly. If they don't:

- first values differ → block layout or field order
- min/max differ → scale interpretation (fp16 vs fp32)
- std is 10× small → zeroed subnormals
- everything differs → wrong tensor offset or alignment

## INT16 on the GPU

INT16 is not a GGUF format. It's a runtime thing I do at upload time because 4 GB
of VRAM is not a lot.

For each 2-D weight:

```
scale[r] = max|W[r, :]| / 32767
q[r, c]  = round(W[r, c] / scale[r])  (clamped to ±32767)
```

Kernel:

```
y[r] = scale[r] * Σ_c q[r, c] * x[c]
```

Two bytes per weight plus one float per row. FP32 accumulation. TinyLlama drops
from ~4.3 GB to ~2.2 GB and fits.

Greedy output matches the FP32 CPU path. If you want full precision back, use
`--gpu-precision fp32`.

## Adding another quant format

It's a six-step recipe:

1. Add the enum value if it's missing.
2. Add the exact block size to `type_nbytes_per_block`.
3. Add the block size to `block_size` if it isn't 1.
4. Write `dequantize_<type>` using explicit byte offsets. Don't use a struct
   with `float` fields; padding will bite you.
5. Add the case to `dequantize()`.
6. Add a test that round-trips values or compares against `gguf`.

Then update [docs/models.md](models.md) so people know it's supported.

## The checklist I wish I'd had

| What you see | What it probably is |
|---|---|
| Weights around 1e4 | fp16 scale read as fp32, or wrong block stride |
| Plausible weights, garbage output | dims transposed, or wrong tensor loaded |
| Some blocks all zero | subnormal fp16 scales flushed to zero |
| Descriptors desync | metadata type enum, or type before n_dims |
| Everything shifted a few bytes | alignment padding before tensor data |
| Only output projection wrong | Q6_K layout or tied-output handling |
| `Tensor not found` | architecture-specific naming |

That's most of what went wrong here. The format is simple; the edge cases are
not.
