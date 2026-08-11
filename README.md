# Llama2 on the Almost Realism HPC Platform

A complete Llama2 inference implementation in roughly 200 lines of Java, with a single
dependency: the [Almost Realism HPC platform](https://github.com/almostrealism/common).

**I am the author of both layers.** This repository is small because the platform beneath
it does the heavy lifting — and I wrote that too: the attention kernels, the KV-cache and
decoding machinery, the tokenizer, and the compiler that turns all of it into native GPU
code. This repo is best read as a demonstration of what the platform makes possible, not
as a standalone project.

Inspired by [llama2.c](https://github.com/karpathy/llama2.c) and
[llama2.java](https://github.com/mukel/llama2.java), and compatible with their tokenizer
and model weight formats.

## Why is this repository so small?

Because `Llama2.java` does not call pre-written GPU kernels — there is no cuBLAS, no
Metal Performance Shaders, no vendor library underneath. It *declares* the transformer as
a graph of mathematical operations (attention, RoPE rotation, RMS norm, SiLU feed-forward),
and the platform **compiles that graph into fused OpenCL or Metal kernels at runtime**,
deciding CPU/GPU placement dynamically. The line where that happens is easy to miss:

```java
transformer.compile(false, profile)
```

Every kernel named in the profiler output below — `softmax2d`, `ropeRotation`,
`attentionKeys`, the fused `rmsnorm` — was generated at startup from the operation graph.
None of them existed as code before the program ran.

## Where the interesting code lives

The substantive machinery sits upstream in [`common`](https://github.com/almostrealism/common),
where it can serve any model architecture, and all of it is mine:

| Component | Location in `common` | What it does |
|---|---|---|
| Attention & transformer construction | [`AttentionFeatures.java`](https://github.com/almostrealism/common/blob/master/engine/ml/src/main/java/org/almostrealism/ml/AttentionFeatures.java) (~2,000 lines) | Builds attention (including grouped-query variants), RoPE, and full transformer blocks as compilable operation graphs. `Llama2` implements this interface — `transformer(...)`, `rmsnorm(...)`, and `dense(...)` in the model definition come from here. |
| Autoregressive decoding & KV cache | [`AutoregressiveModel.java`](https://github.com/almostrealism/common/blob/master/engine/ml/src/main/java/org/almostrealism/ml/AutoregressiveModel.java) | Token-by-token decode loop, position bookkeeping, prompt ingestion. |
| Graph → kernel compilation | [`Model.java`](https://github.com/almostrealism/common/blob/master/domain/graph/src/main/java/org/almostrealism/model/Model.java) and the platform compiler | The layer graph assembled in `Llama2.model(...)` becomes fused native kernels here. |
| BPE tokenizer | [`BPE.java`](https://github.com/almostrealism/common/blob/master/engine/ml/src/main/java/org/almostrealism/ml/BPE.java) | Prompt encoding against the llama2.c tokenizer format. |
| Kernel-level profiling | [`OperationProfile.java`](https://github.com/almostrealism/common/blob/master/base/code/src/main/java/io/almostrealism/profile/OperationProfile.java) | The per-kernel timing report shown below. |

## Usage

1. Compile the project using `mvn package`
2. Download the model weights for [stories110M.bin](https://www.dropbox.com/scl/fi/romns8veg67agl5czmtww/stories110M.bin?rlkey=sbspy97d2j1p3jilgaff190pz&st=kak6t2uo&dl=1)
3. Run with `java -jar target/ar-llama2-0.5.jar` from the directory containing both the
   model weights and `tokenizer.bin` (included in this repository)

To convert other models to this format, see the export instructions in the
[llama2.c](https://github.com/karpathy/llama2.c) README.

## Will this use my GPU?

If your system supports Metal or OpenCL, then yes — the platform decides dynamically what
runs on CPU versus GPU, and the choice can be overridden. CUDA support is in progress
upstream.

## Example output

Running the usage steps above produces the story text followed by a kernel-level profile —
per-kernel invocation counts and timings for code that was generated at startup
(these results are from an Apple M4, fp32, with development instrumentation enabled):

```shell
michael@Mac llama2 % java -jar target/ar-llama2-0.5.jar
Hardware[CL]: Using GPU 0 for kernels
Loaded weights in 548ms
<s>
Once upon a time, there was a little girl named Lily. She loved to play outside
in the sunshine. One day, she saw a big, red apple on a tree...
tokens per second: 16.859504
[08:47.49] OperationProfile: default - 42.62 seconds:
        softmax2d layer (12, 1024)->(12, 1024): 3072 [11.29s tot | 3.675ms avg] 26%
        ropeRotation layer (12, 32, 2)->(12, 32, 2): 6144 [9.972s tot | 1.623ms avg] 23%
        rmsnorm layer (768)->(768): 6400 [6.866s tot | 1.073ms avg] 16%
        dense 768 layer (768)->(768): 12288 [4.231s tot | 0.344ms avg] 9%
        attentionKeys layer (12, 64)->(12, 1024): 3072 [2.232s tot | 0.726ms avg] 5%
        ...
Done
```

The full profile lists every generated kernel; results vary substantially by hardware.

## Author

Michael Murray ([@ashesfall](https://github.com/ashesfall)) — creator and maintainer of
this repository and of the [Almost Realism HPC platform](https://github.com/almostrealism/common)
it is built on.
