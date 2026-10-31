# minivllm

A tiny LLM inference engine built from scratch, one idea at a time, with a benchmark after every step. It is inspired by [vLLM](https://github.com/vllm-project/vllm), a high-throughput, memory-efficient serving engine. **minivllm** aims to be the micrograd of inference engineering: small enough to read in an afternoon, measured well enough to trust.

**The thesis, in one line:** batch-1 decoding is memory-bound and overhead-bound, so serving speed comes from reading fewer bytes per token and sharing every weight read across many requests. Each stage removes one source of waste and measures what it bought.

> **Where the numbers are.** This file and the write-ups explain what each stage teaches and what its results mean. Every measured number lives in [`RESULTS.md`](RESULTS.md), which is from `results/stage*.json`. The headline table below is the one place in this file where figures appear. 

## Headline results

One row per idea, each with the best number the repo measured for it on a Tesla T4 (details and every other number are in [`RESULTS.md`](RESULTS.md)). What I took away from all of it is on one page: [`docs/learned.md`](docs/learned.md).

<!-- headline:start -->
| Idea | What it removes | Best measured result | Stage | Details |
| --- | --- | --- | --- | --- |
| KV cache, `torch.compile` | recomputing every token; launch overhead | Decode at batch 1 reaches 15% of the bandwidth ceiling eagerly and 31% compiled (2.0x) | [1 & 2](docs/stage1_2.md) | [numbers](RESULTS.md#stage1_2-roofline) |
| Static batching | one weight read per request | Throughput rises 61x from batch 1 to batch 64 while a request keeps 95% of its single-request rate | [3](docs/stage3.md) | [numbers](RESULTS.md#stage3-roofline) |
| Continuous batching | padding, finished-but-computed slots, head-of-line blocking | 1.7x to 2.2x the goodput of static batching at the same concurrency | [4](docs/stage4.md) | [numbers](RESULTS.md#stage4-closed) |
| Paged KV cache | KV memory fragmentation | 1.1x to 1.7x the goodput of a contiguous cache in the same memory (most when memory is tight) | [5](docs/stage5.md) | [numbers](RESULTS.md#stage5-memory) |
| N-gram speculation | one token per weight read, when the text repeats | Up to 3.2x on copy, 3.1x on code and 2.2x on open text, with no second model | [6](docs/stage6.md) | [numbers](RESULTS.md#stage6-speculative) |
| Draft-model speculation | the limit that guesses repeat the context | A 0.5B draft step costs this fraction of a target step: 1.5B target 0.84 eager, 0.39 in a CUDA graph; 3B target 0.62 eager, 0.27 in a CUDA graph | [7](docs/stage7.md) | [numbers](RESULTS.md#stage7-cost) |
| Weight-only int8 | half of the bytes read per token | Reads 50% of the fp16 bytes per decoded token with perplexity -0.04% (1.5B) | [8](docs/stage8.md) | [numbers](RESULTS.md#stage8-memory) |
| Fused int8 kernel (Triton) | the dequantize copy; reads weights as bytes | A decode step is 1.4x to 1.7x faster than fp16, both in a CUDA graph (0.5B to 3B) | [9](docs/stage9.md) | [numbers](RESULTS.md#stage9-step) |
| CUDA-graph decode loop | per-step launch overhead | fp16 decodes 1.2x to 2.4x faster in a graph than eagerly (most at the smallest model); int8 in a graph is 2.3x to 3.7x faster than eager fp16 | [10](docs/stage10.md) | [numbers](RESULTS.md#stage10-plain) |
| Best end-to-end stack | what is left once each lever is used where it pays | 3B target, best setup against eager fp16 decoding: copy 6.3x (fp16 graph + lookup), code 2.0x (fp16 graph + lookup), open 1.9x (int8 graph) | [11](docs/stage11.md) | [numbers](RESULTS.md#stage11-speculation) |
| Int8 with the lookup (did not work) | (nothing: the gains do not compose on a T4) | 48% to 65% of the product of the two gains survives on copy and code: an int8 checking pass costs 1.8 to 1.9 plain steps (fp16: 1.0) | [10 & 11](docs/stage11.md) | [numbers](RESULTS.md#stage11-block) |
| The engine (continuous batching, paged cache) | one request or one batch at a time, idle memory bandwidth between them | 1.5B, latency target met up to 4 requests per second by the engine (eager decode); no Hugging Face setup meets it at any tested rate. At 8 requests per second offered it completes 3.8 per second, Hugging Face 0.4 one request at a time and 0.9 in static batches | [12](docs/stage12.md) | [numbers](RESULTS.md#stage12-capacity) |
| Graph decode in the engine | launch overhead | Time per token of a captured decode step against the eager one: alone, 3.1x to 3.3x faster (0.5B) and 1.8x faster (1.5B); with requests arriving over time, at the middle rate its median is 0.56x (0.5B) and 2.28x (1.5B) of the eager step's, so a captured step over padded buckets can lose in a batch | [12](docs/stage12.md) | [numbers](RESULTS.md#stage12-modes) |
| Lookup speculation in the engine | weight reads per token when one request runs alone | One request at a time, time per token against the eager step (graph plus lookup): 0.5B copy 8.0x, code 8.7x, open 5.6x; 1.5B copy 2.9x, code 3.8x, open 2.0x | [12](docs/stage12.md) | [numbers](RESULTS.md#stage12-latency) |
| Against vLLM (same traces) | where the engine is still behind: its batched step, whose cost grows with the load while vLLM's does not | 1.5B, time per token from 1 to 8 requests per second offered: vLLM 17 to 18 ms, the engine's server 24 to 143 ms. At 8 it completes 2.5 per second against vLLM's 5.7 and Hugging Face's 0.9; latency target met up to 8 requests per second by vLLM, 2 requests per second by the engine's server, none of the tested rates by Hugging Face | [12](docs/stage12.md) | [numbers](RESULTS.md#head-to-head) |
| The batched step: fold, fine buckets, fallback | the head expansion and the padding in the gather, which made the batched step's cost grow with the load | 1.5B, at 8 requests per second: time per token 160 ms for the Stage 12 graph step, 84 ms folded, 72 ms with all three levers; capacity 2 requests per second to 8 requests per second; the folded path moves 29% of the attention bytes; over HTTP the best variant takes 66 ms per token at 8 (vLLM, another session: 18 ms) | [13](docs/stage13.md) | [numbers](RESULTS.md#stage13-levers) |
| The in-place paged attention kernel | the gather: every row's blocks copied to the width of its bucket, twice, so a step cost what the padding cost | 1.5B, at 8 requests per second: time per token 139 ms for the Stage 12 graph step, 58 ms for the Stage 13 best, 38 ms with the in-place kernel (best kernel, no fallback); over HTTP 43 ms against vLLM's 18 ms (another session) | [14](docs/stage14.md) | [numbers](RESULTS.md#stage14-load) |
<!-- headline:end -->

## The stages

One notebook per stage, each self-contained (its `%%writefile` cells write the whole `minivllm/` and `bench/` packages). Run them on a Kaggle T4 (GPU on, Internet on).

| Stage | Question | What we found | Notebook | Write-up |
| --- | --- | --- | --- | --- |
| 1 & 2 | What does naive generation cost, and where does the time go? | A KV cache wins only past a crossover length. Decode sits far below the hardware limit, so it is overhead-bound, and the way out is more sequences per weight read. | [`01_baseline_and_kv-cache`](01_baseline_and_kv-cache.ipynb) | [docs](docs/stage1_2.md) |
| 3 | What does batching buy, and what does padding waste? | Extra rows in a decode step are almost free, but padding and finished-but-computed slots waste much of the gain, and short requests wait for long ones. | [`02_static_batching`](02_static_batching.ipynb) | [docs](docs/stage3.md) |
| 4 | How do we stop short requests waiting on long ones? | Continuous batching never computes a finished request: about twice the goodput at the same concurrency, and a much higher capacity under load. | [`03_continuous_batching`](03_continuous_batching.ipynb) | [docs](docs/stage4.md) |
| 5 | How do we stop wasting KV memory? | A paged cache fits about twice as many sequences in the same memory, so goodput rises while memory is the limit. | [`04_paged_kv_cache`](04_paged_kv_cache.ipynb) | [docs](docs/stage5.md) |
| 6 | Can one weight read yield more than one token? | N-gram speculation checks several guessed tokens in one pass (the guesses come from the context, with no second model), since a launch-bound step is nearly free to widen: about two to three times faster on repetitive text with no second model, with a gain that erodes as the batch grows. | [`05_ngram_speculation`](05_ngram_speculation.ipynb) | [docs](docs/stage6.md) |
| 7 | Is a small draft model a better guesser than the context, and cheap enough to pay? | The draft is right more often than the n-gram lookup everywhere, but an eager draft step costs nearly as much as the target's because launches, not weights, set the cost, so it slows decoding down. A CUDA graph over the draft step cuts its cost to about half or less and turns that into a gain, yet the n-gram lookup still wins on every workload at batch size one. In a projection the lookup's gain erodes with batch size and the draft's holds up. | [`06_draft_model_speculation`](06_draft_model_speculation.ipynb) | [docs](docs/stage7.md) |
| 8 | If a decode step were limited by the bytes it reads, how much would reading fewer save? | Weight-only int8 with one scale per output row halves the bytes read per token at every size and keeps perplexity within a fraction of a percent of fp16, but no way of multiplying by an int8 weight that this PyTorch build offers (an eager dequantize, the same under `torch.compile`, the built-in int8 matmul) is faster than fp16 on a T4. Decode steps get slower, more so for larger models, and a quantized draft raises Stage 7's cost ratio. A fused kernel is what the result points at. | [`07_weight_quantization`](07_weight_quantization.ipynb) | [docs](docs/stage8.md) |
| 9 | Can a kernel that reads int8 weights as bytes turn Stage 8's no-go around, and can the int8 copy be a draft for its own model? | A Triton weight-only int8 kernel is correct and, at one token, beats the fp16 layer (most on `lm_head`, not on the narrow layers), and in a CUDA graph or under `torch.compile` the int8 decode step is faster than fp16 at every size, by more for bigger models. The int8 copy is the best draft there is (it is accepted far more often than a 0.5B draft or the n-gram lookup). But anything that feeds several tokens through the int8 model is slow, so an int8 target with speculation loses and the self-draft does not yet beat the n-gram lookup; the cost model says why, and a multi-token kernel is the next lever. | [`08_fused_int8_and_self_speculation`](08_fused_int8_and_self_speculation.ipynb) | [docs](docs/stage9.md) |
| 10 | Do int8 weights and speculative decoding multiply, once the decode loop itself lives in CUDA graphs? | No. A graph decoder (a forward pass captured in a CUDA graph for a fixed number of tokens, with a rollback that is only moving a length) works and gives the n-gram lookup its best results: fp16 with the lookup in a graph is the best setup on repetitive text and code, and int8 plain decoding in a graph on open text. But the new multi-row int8 kernel (each weight read once for the two to eight rows of a checking pass) is a no-go: its time grows with the rows, so an int8 checking pass costs about two plain steps where fp16's costs one, and the whole loss is that kernel. int8 plus the lookup reaches well under the product of the two gains. A better multi-row kernel is the next lever. | [`09_graph_speculation`](09_graph_speculation.ipynb) | [docs](docs/stage10.md) |
| 11 | Was the loss of Stage 10 the multi-row kernel, or the idea? | Both, and the kernel part is smaller than hoped. Two row-exact kernels (one accumulator per row, nothing staged through shared memory in the loop) are better than Stage 10's: an int8 checking pass costs about a fifth fewer plain steps and int8 with the lookup gains a tenth to nearly a fifth. But they are still slower than the fp16 layer from five rows up, a five-token int8 pass still costs close to two plain steps against one for fp16, and both gates are a no-go. With the lookup, int8 is about even with fp16 on repetitive text, behind it on copy at 3B, and ahead only on open text, where int8 plain decoding wins. The tuner's installed winners are sometimes up to a third slower than the best configuration timed in turns, and the offline compile that found the kernels was right about spills and blind to dynamic shared memory. | [`10_multirow_kernel`](10_multirow_kernel.ipynb) | [docs](docs/stage11.md) |
| 12 | What happens when the pieces meet in one engine that serves requests, and how does it compare with Hugging Face and vLLM on the same traces? | One `LLMEngine` (continuous batching on the paged cache, sampling, lookup speculation, a bucketed decode step that can run as CUDA graphs) behind a small HTTP server, driven by an open-loop load generator. Every mode produces the tokens of plain decoding. Against Hugging Face `generate` it is lopsided: Hugging Face is saturated from the lowest rate tested and meets the latency target at no tested rate, the engine does at every rate on the smaller model. One request at a time, the captured step and the lookup are large wins (most on repetitive text), and the server in its own process costs little over the in-process engine. The surprise: under load on the larger model the batched captured step is slower than the eager one, and the uncaptured control is just as slow, so the cost is the buckets and not the graph. Against vLLM on the same traces, both meet the latency target at low load (vLLM already has the lower time per token) and one request at a time on the larger model they are about even, but under load vLLM's time per token stays flat while the engine's rises several times over, so vLLM meets the target at the top rate at both sizes and the engine only on the smaller model. | [`11_engine`](11_engine.ipynb), [`99_vllm_compare`](99_vllm_compare.ipynb) | [docs](docs/stage12.md) |
| 13 | Why does the batched step get slower as the load rises, and how much of the gap to vLLM do three changes close? | The step's cost is counted and timed before anything is changed: the gather copies the keys and values of every row to a padded width, and the head expansion copies them again to every query head. Three changes, each run alone against the Stage 12 traces and then together: fold the query heads into the query rows (nothing is expanded), buckets that pad at most a quarter, and an eager fallback above a batch size read from the step logs. A closed-loop sweep gives the cost per row at a fixed concurrency, and the best variant is served over HTTP next to the vLLM rows. The result: folding is the big win (the time per token under load roughly halves and the larger model's capacity at least doubles), finer buckets help less, the fallback alone only undoes the loss, all three together are best in open loop and the fallback costs time in a closed loop; the best variant over HTTP is much closer to vLLM and still more than twice its time per token at the top rate. | [`12_batched_step`](12_batched_step.ipynb) | [docs](docs/stage13.md) |
| 14 | Can attention that reads the KV blocks where they are, instead of gathering them, close more of the gap to vLLM? | Two Triton kernels (one program per query head, one per key-value head with the group served from registers) walk each row's own blocks through its block table with an online softmax, so nothing is copied, padding blocks are never touched, and the kernels run inside CUDA graphs. Tested on a CPU against a gather-then-softmax reference, with NaN-poisoned memory and wrong kernels caught; gated at the level of one layer, then in the engine. Measured on a T4: the kernels win clearly on partly filled rows and tie or lose at a full bucket, they take a modest share off the engine's time per token at low rates and a large one at the top rate on the larger model, the eager fallback now costs, and the gap to vLLM narrows again without closing. | [`13_paged_kernel`](13_paged_kernel.ipynb) | [docs](docs/stage14.md) |

```text
Stage 1  naive                       every step recomputes the whole sequence
Stage 2  + KV cache                  one token per step, but the GPU reads all weights for a single sequence
Stage 3  + static batching           many sequences share each weight read, but padding wastes slots
Stage 4  + continuous batching       finished sequences leave, new ones join, no idle slots
Stage 5  + paged KV                  the cache is allocated in blocks, so memory is not fragmented
Stage 6  + n-gram speculation        guess several tokens from the context, check them in one pass
Stage 7  + draft-model speculation   a small model guesses instead, behind the same checker
Stage 8  + weight-only int8          every weight stored in one byte plus a row scale, so a step can read fewer bytes
Stage 9  + fused int8 kernel         read those bytes as bytes, and let the int8 copy draft for its own model
Stage 10 + decode loop in graphs     a kernel for a few rows, every model pass a CUDA graph replay, int8 and speculation together
Stage 11 + row-exact int8 kernels    no shared memory in the loop, exact row counts, a fair clock: a no-go again, the loss shrinks and the gains still do not multiply
Stage 12 + the engine                the pieces behind one interface: continuous batching, paged cache, CUDA-graph decode, lookup speculation, sampling, a server, and a head-to-head on the same traces
Stage 13 + the batched step          count the bytes of a batched step, then fold the query heads, pad less and fall back to the eager step above a batch size (folding wins; the gap to vLLM narrows to about a third)
Stage 14 + in-place paged attention  Triton kernels that read each row's own blocks through its block table, no gather (measured: a gain under load, far from the bandwidth; the gap to vLLM narrows to about two to three times at the top rate)
```

The kernel line stops at Stage 11, with a documented negative result. Stage 12 is not another lever: it composes the pieces that paid off into one engine, puts it behind a server, and compares it with Hugging Face `generate` and vLLM on the same traces ([`docs/stage12.md`](docs/stage12.md)).

Every notebook proves its change correct first (same tokens as the stage before, plus a negative control that shows why the tricky part matters), then runs its experiments with a prediction written down before each benchmark.

## Where to read more

- **[`REPORT.md`](REPORT.md):** the detailed write-up of the whole project: every stage's narrative, results discussion, lessons and exercises, in one file.
- **[`docs/exercises.md`](docs/exercises.md):** every stage's exercises on one page, in stage order, for planning what to build or measure next. From the stage docs, like `REPORT.md`.
- **[`docs/learned.md`](docs/learned.md):** what the stages taught, in a few minutes: the one idea, each lever and where it stops, the negative results, how I measured, and the questions I can now answer.
- **[`docs/`](docs/):** the source of that material, one stage at a time, plus [`docs/code.md`](docs/code.md) on how the code works, the repo layout, and the conventions.
- **[`RESULTS.md`](RESULTS.md):** every measured number, tables and plots, from `results/stage*.json`.

## Quickstart

- **Run an experiment:** open a notebook on Kaggle (Accelerator = GPU T4, Internet = On) and run it top to bottom. Set `QUICK = True` in the setup cell for a pass of a few minutes. The last cell can publish the results to GitHub with a token in a Kaggle secret.
- **Serve a model:** `python -m minivllm.server --model Qwen/Qwen2.5-0.5B --graph --warmup --port 8000` (needs a GPU), then `curl -N localhost:8000/v1/completions -d '{"prompt": "Hello", "max_tokens": 32, "stream": true}'`.

## Repo layout

```text
├── README.md, REPORT.md, RESULTS.md   # entry point, detailed report, numerical results
├── docs/                              # one write-up per stage, docs/code.md, and docs/exercises.md 
├── 01_..11_*.ipynb                    # one notebook per stage
├── 99_vllm_compare.ipynb              # appendix: vLLM, the engine's server and Hugging Face on the same traces (not a stage)
├── pyproject.toml                     # package metadata and the `minivllm-server` command
├── minivllm/                          # the engine: model, KV caches, schedulers, block allocator, speculative decoding, quantization, the CUDA-graph, and LLMEngine 
├── bench/                             # workloads, metrics, plots, roofline, publish, the open-loop load generator and the vLLM runner
└── results/                           # stage*.json and plots
```

[`docs/code.md`](docs/code.md) lists every module and tool and explains the conventions every stage follows.

## References

**The model**

- Qwen2.5 technical report: https://arxiv.org/abs/2412.15115

**Serving systems**

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM): https://arxiv.org/abs/2309.06180
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (grouped-query attention, the key-value sharing that Stage 13 folds instead of expanding): https://arxiv.org/abs/2305.13245
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (multi-query attention): https://arxiv.org/abs/1911.02150
- Milakov and Gimelshein, *Online normalizer calculation for softmax* (the running softmax of the in-place attention of Lab 3): https://arxiv.org/abs/1805.02867
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*: https://arxiv.org/abs/2205.14135
- vLLM, the project the paper describes and the system Stage 12 is compared with: https://github.com/vllm-project/vllm
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022), the origin of iteration-level (continuous) scheduling: https://www.usenix.org/conference/osdi22/presentation/yu
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (chunked prefill): https://arxiv.org/abs/2403.02310
- Hugging Face Transformers, text generation (`generate`, the baseline of Stage 12): https://huggingface.co/docs/transformers/main_classes/text_generation

**Speculative decoding**

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*: https://arxiv.org/abs/2211.17192
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*: https://arxiv.org/abs/2302.01318
- Saxena, *Prompt Lookup Decoding* (the n-gram guesser): https://github.com/apoorvumang/prompt-lookup-decoding
- Zhao et al., *QSpec: Speculative Decoding with Complementary Quantization Schemes*: https://arxiv.org/abs/2410.11305

**Quantization**

- Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*: https://arxiv.org/abs/2208.07339
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*: https://arxiv.org/abs/2210.17323
- Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*: https://arxiv.org/abs/2211.10438
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*: https://arxiv.org/abs/2306.00978
- PyTorch, *gpt-fast* (weight-only int8 with `torch.compile`): https://github.com/pytorch-labs/gpt-fast

**Kernels and performance**

- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model*: https://dl.acm.org/doi/10.1145/1498765.1498785
- Tillet, Kung, Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*: https://dl.acm.org/doi/10.1145/3315508.3329973
- PyTorch, *Accelerating PyTorch with CUDA Graphs*: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
