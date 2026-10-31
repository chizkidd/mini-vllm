# Results

Measured numbers, retrieved from `results/stage*.json`. The narrative lives in the stage write-ups in [`docs/`](docs/) (collected in [`REPORT.md`](REPORT.md)), and links here instead of repeating figures.

<a id="headline"></a>

## Headline results

The few numbers worth knowing, one row per idea, read from the results files (the same table is in the README).

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

## Stages 1 & 2: Baseline & KV Cache

Source: [`results/stage1_2.json`](results/stage1_2.json)

<a id="stage1_2-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.10.0+cu128 / 5.0.0 / 3.12.13
- **Model:** Qwen/Qwen2.5-0.5B
- **Model revision:** 060db6499f32faf8b98477b0a26969ef7d8b9987
- **max_pos (RoPE table):** 4096
- **Prompt length for the naive vs cached runs:** 128

<a id="stage1_2-naive-vs-cached"></a>

### Naive vs cached generation

**Short outputs**

| new tokens | naive tok/s | cached tok/s | speedup | TTFT (ms) | ITL p50 (ms) | ITL p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 41.9 | 38.6 | 0.92x | 25.9 | 25.9 | 28.2 |
| 32 | 41.7 | 37.7 | 0.91x | 26.6 | 26.4 | 28.5 |
| 64 | 42.5 | 37.5 | 0.88x | 26.7 | 26.5 | 34.1 |
| 128 | 42.4 | 37.3 | 0.88x | 26.9 | 26.6 | 36.6 |

**Extended sweep**

| new tokens | naive tok/s | cached tok/s | speedup |
| --- | ---: | ---: | ---: |
| 128 | 41.4 | 37.2 | 0.90x |
| 256 | 42.1 | 37.5 | 0.89x |
| 512 | 36.7 | 37.2 | 1.01x |

<a id="stage1_2-crossover"></a>

### Crossover

| Method | Crossover (new tokens) |
| --- | ---: |
| Linear interpolation of the speedup | 483 |
| Log2 interpolation of the speedup | 474 |
| Fit (quadratic naive vs linear cached) | 488 |
| Dense probe, interpolated | 422 |

Across the available estimates the crossover falls between **422** and **488** new tokens.

**Dense probe**

| new tokens | naive tok/s | cached tok/s | speedup |
| --- | ---: | ---: | ---: |
| 300 | 40.9 | 37.0 | 0.91x |
| 340 | 39.7 | 37.0 | 0.93x |
| 380 | 38.9 | 36.9 | 0.95x |
| 400 | 38.1 | 37.2 | 0.98x |
| 440 | 36.5 | 37.2 | 1.02x |

<a id="stage1_2-prefill-decode"></a>

### Prefill and decode

| prompt tokens | TTFT (ms) | ITL p50 (ms) | prefill tok/s | KV cache (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 32 | 27.2 | 26.9 | 1,176 | 0.4 |
| 128 | 26.9 | 26.7 | 4,758 | 1.5 |
| 512 | 26.7 | 26.7 | 19,144 | 6.0 |
| 2048 | 118.0 | 26.9 | 17,363 | 24.0 |

<a id="stage1_2-roofline"></a>

### Roofline

| Quantity | Value |
| --- | ---: |
| Context length (tokens) | 256 |
| Weights (GiB) | 0.92 |
| KV cache at that context (MiB) | 3.0 |
| Bandwidth, spec (GB/s) | 320 |
| Bandwidth, measured copy (GB/s) | 243 |
| Decode ceiling at spec bandwidth (tok/s) | 323 |
| Decode ceiling at measured bandwidth (tok/s) | 245 |
| Eager decode (tok/s) | 37.6 |
| Eager decode, % of measured ceiling | 15% |
| Compiled decode (tok/s) | 76.3 |
| Compiled decode, % of measured ceiling | 31% |
| Compiled vs eager | 2.03x |
| One-time compile warm-up (s) | 88 |
| Decode FLOPs per token (GFLOPs) | 1.01 |
| Decode bytes per token (MiB) | 945 |
| Arithmetic intensity (FLOPs/byte) | 1.02 |
| FP16 compute peak (TFLOP/s) | 65.1 |
| Ridge point, measured bandwidth (FLOPs/byte) | 268 |
| Ridge point, spec bandwidth (FLOPs/byte) | 203 |
| Arithmetic intensity as a share of the measured ridge | 0.38% |

<a id="stage1_2-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage1_2_compile.png` | Eager vs torch.compile decode tok/s |
| `stage1_2_crossover.png` | Naive vs cached tok/s, extended sweep |
| `stage1_2_crossover_annotated.png` | Extended sweep with the interpolated crossover marked |
| `stage1_2_crossover_probe.png` | Zoomed probe window with the crossover marked |
| `stage1_2_crossover_quadratic_fit.png` | Quadratic-naive and linear-cached fits with the fitted crossover |
| `stage1_2_crossover_sweep_probe_fit.png` | Sweep, probe points and fitted curves on one plot |
| `stage1_2_latency.png` | TTFT vs ITL p50, the prefill vs decode split |
| `stage1_2_prefill_throughput.png` | Prefill tok/s vs prompt length |
| `stage1_2_roofline.png` | Decode tok/s vs measured and spec bandwidth ceilings |
| `stage1_2_roofline_model.png` | Decode roofline: TFLOP/s vs arithmetic intensity |
| `stage1_2_throughput.png` | Naive vs cached tok/s, short outputs |
| `stage1_2_ttft_and_throughput.png` | TTFT and prefill throughput vs prompt length (stacked panels) |
| `stage1_2_ttft_itl_vs_prompt.png` | TTFT and decode ITL p50 vs prompt length |
| `stage1_2_ttft_scaling.png` | TTFT vs prompt length |

## Stage 3: Static Batching

Source: [`results/stage3.json`](results/stage3.json)

<a id="stage3-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.10.0+cu128 / 5.0.0 / 3.12.13
- **Model:** Qwen/Qwen2.5-0.5B
- **Model revision:** 060db6499f32faf8b98477b0a26969ef7d8b9987
- **max_pos (RoPE table):** 4096
- **Requests per workload / seed / quick run:** 32 / 0 / False

<a id="stage3-batch-scaling"></a>

### Batch-size scaling

Equal requests: 128-token prompts, 64 new tokens.

| batch size | TTFT (ms) | ITL p50 (ms) | ITL p99 (ms) | aggregate tok/s | per-request tok/s | scaling efficiency | KV cache (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23.8 | 24.2 | 35.5 | 41 | 41.3 | 100% | 2.2 |
| 2 | 23.1 | 22.8 | 25.5 | 88 | 43.8 | 106% | 4.5 |
| 4 | 23.3 | 23.3 | 30.4 | 172 | 43.0 | 104% | 9.0 |
| 8 | 40.1 | 23.5 | 25.4 | 341 | 42.6 | 103% | 17.9 |
| 16 | 90.6 | 23.6 | 32.9 | 677 | 42.3 | 102% | 35.8 |
| 32 | 191.4 | 23.8 | 34.8 | 1,342 | 41.9 | 102% | 71.6 |
| 64 | 390.0 | 25.4 | 30.4 | 2,516 | 39.3 | 95% | 143.2 |

<a id="stage3-roofline"></a>

### Batched roofline

| Quantity | Value |
| --- | ---: |
| Context length (tokens) | 192 |
| Weights (GiB) | 0.92 |
| Bandwidth, spec (GB/s) | 320 |
| Bandwidth, measured copy (GB/s) | 233 |
| FP16 compute peak (TFLOP/s) | 65.1 |
| Ridge point, measured bandwidth (FLOPs/byte) | 279 |

| batch size | arithmetic intensity (FLOPs/byte) | achieved TFLOP/s | aggregate tok/s | bandwidth-bound ceiling (tok/s) | % of ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.01 | 0.041 | 41 | 236 | 18% |
| 2 | 2.02 | 0.088 | 88 | 470 | 19% |
| 4 | 4.03 | 0.173 | 172 | 936 | 18% |
| 8 | 7.98 | 0.342 | 341 | 1,855 | 18% |
| 16 | 15.67 | 0.680 | 677 | 3,641 | 19% |
| 32 | 30.22 | 1.348 | 1,342 | 7,024 | 19% |
| 64 | 56.44 | 2.527 | 2,516 | 13,118 | 19% |

<a id="stage3-padding"></a>

### Padding waste

Batches of 8 requests; prompt lengths spread from 64 tokens up to that times the spread factor; 64 new tokens. The padding share, TTFT and KV columns are for the one batch that was run (the measured batch). The mean padding share over 1000 random batches, with its 5th to 95th percentile range, shows what a typical batch looks like.

| spread factor | longest prompt | real prompt tokens | padding share (measured batch) | mean padding share | 5th to 95th percentile | TTFT (ms) | KV held by padding (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 512 | 0.0% | 0.0% | 0.0% to 0.0% | 24.0 | 0.00 |
| 2 | 108 | 727 | 15.9% | 24.8% | 14.7% to 34.5% | 35.7 | 1.61 |
| 4 | 181 | 992 | 31.5% | 40.4% | 25.3% to 54.1% | 66.5 | 5.34 |
| 8 | 512 | 3056 | 25.4% | 50.4% | 33.2% to 64.8% | 210.9 | 12.19 |

<a id="stage3-idle-decode"></a>

### Idle decode slots

Batches of 8; short outputs of 32 tokens, long outputs of 256 tokens.

| long requests | idle decode slot-steps | goodput tok/s | raw tok/s | makespan (s) |
| --- | ---: | ---: | ---: | ---: |
| 0% | 0.0% | 326 | 326 | 3.1 |
| 10% | 68.5% | 107 | 334 | 13.8 |
| 25% | 63.1% | 125 | 336 | 24.4 |
| 50% | 49.4% | 172 | 339 | 24.2 |

<a id="stage3-goodput"></a>

### Goodput vs batch size

| workload | batch size | goodput tok/s | raw tok/s | padding share | idle decode slot-steps | makespan (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| equal | 1 | 42 | 42 | 0.0% | 0.0% | 49.2 |
| equal | 2 | 87 | 87 | 0.0% | 0.0% | 23.6 |
| equal | 4 | 170 | 170 | 0.0% | 0.0% | 12.0 |
| equal | 8 | 339 | 339 | 0.0% | 0.0% | 6.0 |
| equal | 16 | 654 | 654 | 0.0% | 0.0% | 3.1 |
| equal | 32 | 1,151 | 1,151 | 0.0% | 0.0% | 1.8 |
| mixed | 1 | 42 | 42 | 0.0% | 0.0% | 47.3 |
| mixed | 2 | 55 | 87 | 27.7% | 37.4% | 36.3 |
| mixed | 4 | 66 | 171 | 51.8% | 61.8% | 30.0 |
| mixed | 8 | 90 | 329 | 63.8% | 72.9% | 21.9 |
| mixed | 16 | 156 | 626 | 68.4% | 75.4% | 12.7 |
| mixed | 32 | 193 | 790 | 68.4% | 75.8% | 10.2 |

<a id="stage3-latency"></a>

### Latency of short requests

All 32 requests arrive at once, so the 99th percentile of that many requests is simply the slowest one and equals the makespan; the long mean can sit below the short mean because a request's response time depends on which batch it lands in.

| batch size | makespan (s) | goodput tok/s | latency p50 (s) | latency p99 (s) | short mean (s) | short blocked (s) | short slowdown | long mean (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.3 | 42 | 25.5 | 47.3 | 23.8 | 0.0 | 36.0x | 24.7 |
| 2 | 36.3 | 55 | 20.3 | 36.3 | 19.1 | 1.0 | 28.9x | 19.5 |
| 4 | 30.0 | 66 | 17.1 | 30.0 | 16.8 | 2.7 | 25.7x | 16.4 |
| 8 | 21.9 | 90 | 15.9 | 21.9 | 13.0 | 4.6 | 19.7x | 12.3 |
| 16 | 12.7 | 156 | 12.7 | 12.7 | 9.6 | 5.2 | 14.5x | 8.8 |
| 32 | 10.2 | 193 | 10.2 | 10.2 | 10.2 | 8.4 | 15.6x | 10.2 |

<a id="stage3-kv-memory"></a>

### KV memory over-allocation

**Measured**

| batch size | KV allocated per batch (MiB) | utilization | utilization if every slot reserved the maximum context |
| --- | ---: | ---: | ---: |
| 1 | 6.6 | 100.0% | 10.9% |
| 2 | 13.1 | 69.4% | 10.9% |
| 4 | 35.4 | 45.0% | 10.9% |
| 8 | 71.1 | 33.1% | 10.9% |
| 16 | 143.2 | 29.3% | 10.9% |
| 32 | 286.5 | 29.2% | 10.9% |

**Projection**

Projection (analytic, not measured): a KV budget of 8 GiB, 2048 tokens reserved per sequence vs a mean of 224 tokens actually used.

| model | KV KiB per token | sequences, max context reserved | sequences, actual length | gain |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 12 | 341 | 3122 | 9.2x |
| Qwen2.5-7B | 56 | 73 | 669 | 9.2x |
| Llama-2-7B (MHA) | 512 | 8 | 73 | 9.1x |

<a id="stage3-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage3_goodput_vs_batch.png` | Goodput vs batch size, equal vs mixed requests |
| `stage3_idle_slots.png` | Decode: where the slot-steps go, useful vs idle, by share of long requests |
| `stage3_itl_vs_batch.png` | Decode step latency p50 with a band up to p99 vs batch size |
| `stage3_kv_util.png` | KV cache utilization vs batch size, mixed workload |
| `stage3_latency_cdf.png` | CDF of how long finished short requests wait for their batch, by batch size |
| `stage3_padding_and_prefill_vs_spread.png` | Padding share and prefill time of the measured batch vs prompt-length spread |
| `stage3_prefill_slots.png` | Prefill: where the prompt slots go, mean over many random batches with a 5th to 95th percentile range, by prompt-length spread |
| `stage3_roofline_batch.png` | Batched decode on the roofline, one point per batch size |
| `stage3_throughput_vs_batch.png` | Aggregate decode tok/s vs batch size, with the ideal line |
| `stage3_tradeoff.png` | Goodput vs blocking time of short requests, one point per batch size, mixed workload |
| `stage3_ttft_itl_vs_batch.png` | Prefill (TTFT) and decode (ITL p50) time vs batch size, stacked panels |
| `stage3_waste_vs_batch.png` | Padding and idle shares of slots vs batch size, mixed workload |

## Stage 4: Continuous Batching

Source: [`results/stage4.json`](results/stage4.json)

<a id="stage4-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** Qwen/Qwen2.5-0.5B
- **Model revision:** 060db6499f32faf8b98477b0a26969ef7d8b9987
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage4-closed"></a>

### Closed batch: static vs continuous

64 requests of the mixed workload, all arriving at once, as in Stage 3. Static batching runs waves of the given batch size and continuous batching runs that many slots. Each combination is a single run.

| concurrency | policy | run time (s) | goodput tok/s | mean response (s) | short mean (s) | long mean (s) | blocked mean (s) | slot occupancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | static | 68.1 | 63 | 34.3 | 33.2 | 39.3 | 2.4 | n/a |
| 4 | continuous | 33.8 | 127 | 14.7 | 13.2 | 21.3 | 0.0 | 94% |
| 8 | static | 43.4 | 99 | 24.5 | 23.7 | 27.9 | 3.4 | n/a |
| 8 | continuous | 19.7 | 218 | 7.8 | 6.5 | 13.7 | 0.0 | 83% |
| 16 | static | 25.6 | 168 | 16.5 | 16.0 | 18.5 | 4.1 | n/a |
| 16 | continuous | 12.4 | 345 | 4.8 | 3.6 | 10.0 | 0.0 | 70% |
| 32 | static | 20.9 | 205 | 15.8 | 15.6 | 16.7 | 7.1 | n/a |
| 32 | continuous | 12.1 | 354 | 4.3 | 2.9 | 10.4 | 0.0 | 46% |

<a id="stage4-load"></a>

### Open-loop load

64 requests of the mixed workload arriving as a Poisson process, the same arrival trace for both policies at each rate, concurrency limit 16 (static batches of up to that size, or that many slots). Static batching takes what has arrived when the GPU is free and does not wait for a full batch. The offered load is the rate the random trace actually realized; with so few requests the response time p99 is simply the slowest request.

| nominal rate (req/s) | policy | offered (req/s) | completed (req/s) | GPU busy | queue mean (s) | latency p50 (s) | latency p99 (s) | TTFT p50 (s) | TTFT p99 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | static | 0.54 | 0.51 | 69% | 1.7 | 4.3 | 12.7 | 0.89 | 6.17 |
| 0.5 | continuous | 0.54 | 0.52 | 60% | 0.0 | 1.0 | 7.0 | 0.03 | 0.07 |
| 1.0 | static | 1.07 | 0.95 | 93% | 2.4 | 7.8 | 13.2 | 1.98 | 6.73 |
| 1.0 | continuous | 1.07 | 0.99 | 83% | 0.0 | 1.1 | 7.6 | 0.04 | 0.08 |
| 2.0 | static | 2.15 | 1.63 | 100% | 3.0 | 8.8 | 14.0 | 2.82 | 8.23 |
| 2.0 | continuous | 2.15 | 1.80 | 98% | 0.0 | 1.1 | 7.8 | 0.04 | 0.08 |
| 3.0 | static | 3.22 | 1.93 | 100% | 5.2 | 11.7 | 17.4 | 5.21 | 10.87 |
| 3.0 | continuous | 3.22 | 2.48 | 100% | 0.0 | 1.2 | 7.9 | 0.04 | 0.10 |
| 4.0 | static | 4.30 | 1.90 | 100% | 7.0 | 14.0 | 20.7 | 7.22 | 14.09 |
| 4.0 | continuous | 4.30 | 3.04 | 100% | 0.0 | 1.1 | 7.8 | 0.04 | 0.07 |
| 6.0 | static | 6.44 | 1.89 | 100% | 8.7 | 15.9 | 24.8 | 8.97 | 18.30 |
| 6.0 | continuous | 6.44 | 3.91 | 100% | 0.0 | 1.3 | 8.4 | 0.05 | 0.22 |
| 8.0 | static | 8.59 | 2.10 | 100% | 8.7 | 13.9 | 23.5 | 11.13 | 17.00 |
| 8.0 | continuous | 8.59 | 4.57 | 100% | 0.1 | 1.4 | 8.5 | 0.05 | 0.53 |

<a id="stage4-prefill-budget"></a>

### Prefill interference

One arrival trace at 4.0 requests per second, served by 16 slots with a limit on how many prefills may run per iteration. The gap is what a request that is already running waits between two tokens: a decode step plus any prefills that ran in between.

| prefills per iteration | TTFT p50 (s) | TTFT p99 (s) | decode step p50 (ms) | gap p50 (ms) | gap p99 (ms) | gap max (ms) | GPU time in prefills | latency p50 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.04 | 0.15 | 28.4 | 28.5 | 58.2 | 68.3 | 8% | 1.2 |
| 2 | 0.04 | 0.10 | 28.3 | 28.5 | 58.9 | 81.0 | 8% | 1.2 |
| 4 | 0.04 | 0.09 | 27.9 | 28.0 | 56.6 | 106.7 | 8% | 1.2 |
| all | 0.04 | 0.09 | 28.5 | 28.6 | 57.9 | 86.9 | 8% | 1.2 |

<a id="stage4-kv-slots"></a>

### KV slot reservation

The slot pool of the closed-batch continuous runs. Use is averaged over decode steps. The last two columns are upper bounds on how many times more sequences would fit in the same memory if allocation followed the average or the peak use instead of the worst case.

| slots | tokens per slot | pool (MiB) | pool holding tokens | peak in use (MiB) | capacity gain if paged (average use) | capacity gain if paged (peak use) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 732 | 34 | 31.7% | 20 | 3.2x | 1.7x |
| 8 | 732 | 69 | 28.0% | 35 | 3.6x | 2.0x |
| 16 | 732 | 137 | 23.7% | 56 | 4.2x | 2.4x |
| 32 | 732 | 274 | 15.6% | 81 | 6.4x | 3.4x |

<a id="stage4-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage4_goodput_vs_slots.png` | Goodput vs concurrency, static vs continuous batching, same requests |
| `stage4_kv_slots.png` | What holds tokens in the KV slot pool, by slot count |
| `stage4_latency_vs_load.png` | Response time p50 and p99 vs offered load, static vs continuous |
| `stage4_latency_vs_slots.png` | Mean response time of all and of short requests vs concurrency, static vs continuous |
| `stage4_prefill_budget.png` | TTFT and the p99 gap between tokens vs the prefill budget |
| `stage4_step_timeline.png` | Decode steps, the gaps a running request sees and the prefills between them |
| `stage4_throughput_vs_load.png` | Completed vs offered load, static vs continuous |
| `stage4_ttft_vs_load.png` | Time to first token p50 and p99 vs offered load, static vs continuous |

## Stage 5: Paged KV Cache

Source: [`results/stage5.json`](results/stage5.json)

<a id="stage5-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** Qwen/Qwen2.5-0.5B
- **Model revision:** 060db6499f32faf8b98477b0a26969ef7d8b9987
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage5-memory"></a>

### Same memory: contiguous vs paged

64 requests of the mixed workload, all arriving at once. A memory budget of b is the memory a contiguous cache needs for b sequences of the largest request (732 tokens): the contiguous cache runs b slots, and the paged cache gets the same bytes cut into blocks of 16 tokens and runs up to 32 sequences. Each combination is a single run. Blocked is the share of decode steps in which the first waiting request was held back by memory.

| memory (worst-case slots) | cache | pool (MiB) | run time (s) | goodput tok/s | mean running | mean response (s) | short mean (s) | long mean (s) | TTFT p50 (s) | blocked |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | contiguous | 17 | 54.4 | 79 | 1.9 | 24.6 | 23.0 | 31.6 | 24.6 | n/a |
| 2 | paged | 17 | 32.2 | 133 | 3.8 | 12.6 | 11.2 | 18.7 | 11.0 | 86% |
| 4 | contiguous | 34 | 29.4 | 146 | 3.8 | 12.9 | 11.5 | 18.6 | 11.4 | n/a |
| 4 | paged | 34 | 18.9 | 227 | 6.8 | 6.7 | 5.4 | 12.3 | 3.7 | 64% |
| 8 | contiguous | 69 | 17.2 | 249 | 6.6 | 6.8 | 5.6 | 11.9 | 4.8 | n/a |
| 8 | paged | 69 | 11.3 | 379 | 12.3 | 4.2 | 3.1 | 9.1 | 1.6 | 38% |
| 16 | contiguous | 137 | 10.9 | 395 | 11.2 | 4.1 | 3.1 | 8.7 | 2.0 | n/a |
| 16 | paged | 137 | 10.1 | 423 | 14.8 | 4.2 | 3.1 | 8.9 | 1.4 | 0% |

<a id="stage5-kv-breakdown"></a>

### Where the KV memory goes

Where the KV pool goes, as a share of the pool, averaged over decode steps of the runs above. Stored tokens are real keys and values. Reserved but empty is memory given to a request for tokens it has not generated yet (for a contiguous slot, also for tokens it will never need). Rounding is the unused tail of each sequence's last block. Not handed out is free pool memory. Fill is stored over handed out.

| memory (worst-case slots) | cache | tokens stored | reserved but empty | block rounding | not handed out | fill |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | contiguous | 32.9% | 67.1% | 0.0% | 0.0% | 32.9% |
| 2 | paged | 63.8% | 19.2% | 1.8% | 15.2% | 75.2% |
| 4 | contiguous | 31.7% | 68.3% | 0.0% | 0.0% | 31.7% |
| 4 | paged | 57.6% | 17.3% | 1.6% | 23.5% | 75.2% |
| 8 | contiguous | 28.0% | 72.0% | 0.0% | 0.0% | 28.0% |
| 8 | paged | 52.1% | 15.7% | 1.5% | 30.7% | 75.2% |
| 16 | contiguous | 23.7% | 76.3% | 0.0% | 0.0% | 23.7% |
| 16 | paged | 31.3% | 9.4% | 0.9% | 58.4% | 75.2% |

<a id="stage5-block-size"></a>

### The gather and the block size

16 sequences at once and plentiful memory, so only the block size changes. The reference is the contiguous cache with the same concurrency. Rounding is the unused tail of the last blocks as a share of the memory handed out. The step ratio is the paged decode step p50 over the contiguous one.

| cache | block size (tokens) | goodput tok/s | decode step p50 (ms) | step vs contiguous | block rounding | mean running |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous (reference) | n/a | 392 | 25.0 | n/a | n/a | 11.2 |
| paged | 8 | 355 | 27.4 | 1.10x | 1.2% | 11.2 |
| paged | 16 | 356 | 27.3 | 1.10x | 2.1% | 11.2 |
| paged | 32 | 357 | 27.3 | 1.09x | 4.5% | 11.2 |
| paged | 64 | 357 | 27.2 | 1.09x | 8.6% | 11.2 |
| paged | 128 | 356 | 27.3 | 1.09x | 18.6% | 11.2 |
| paged | 256 | 357 | 27.2 | 1.09x | 33.7% | 11.2 |

<a id="stage5-load"></a>

### Open-loop load at fixed memory

64 requests of the mixed workload arriving as a Poisson process, the same arrival trace for both caches at each rate, and the same KV memory (4 worst-case slots). The paged cache uses blocks of 16 tokens and runs up to 32 sequences. The offered load is the rate the random trace actually realized; with so few requests the response time p99 is simply the slowest request.

| nominal rate (req/s) | cache | offered (req/s) | completed (req/s) | mean running | queue mean (s) | latency p50 (s) | latency p99 (s) | TTFT p50 (s) | TTFT p99 (s) | blocked |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | contiguous | 1.07 | 1.00 | 2.1 | 0.0 | 1.0 | 6.5 | 0.04 | 0.61 | n/a |
| 1 | paged | 1.07 | 0.99 | 2.2 | 0.0 | 1.1 | 7.3 | 0.04 | 0.08 | 0% |
| 2 | contiguous | 2.15 | 1.77 | 3.2 | 0.6 | 1.6 | 7.3 | 0.13 | 2.50 | n/a |
| 2 | paged | 2.15 | 1.80 | 3.5 | 0.0 | 1.1 | 7.3 | 0.04 | 0.70 | 4% |
| 3 | contiguous | 3.22 | 2.12 | 3.6 | 2.3 | 4.1 | 10.7 | 2.28 | 5.62 | n/a |
| 3 | paged | 3.22 | 2.37 | 4.5 | 0.1 | 1.2 | 7.6 | 0.04 | 1.70 | 14% |
| 4 | contiguous | 4.30 | 2.16 | 3.7 | 4.4 | 6.6 | 15.1 | 5.11 | 10.07 | n/a |
| 4 | paged | 4.30 | 2.76 | 5.3 | 0.3 | 1.3 | 8.5 | 0.05 | 2.63 | 23% |
| 6 | contiguous | 6.44 | 2.17 | 3.7 | 6.5 | 8.6 | 19.7 | 6.94 | 14.87 | n/a |
| 6 | paged | 6.44 | 3.22 | 6.3 | 0.9 | 2.0 | 10.1 | 0.50 | 4.29 | 48% |

<a id="stage5-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage5_block_size.png` | Rounding waste and decode step time vs block size, with the contiguous reference |
| `stage5_goodput_vs_memory.png` | Goodput and mean running sequences vs KV memory, contiguous vs paged, same requests |
| `stage5_kv_breakdown.png` | What holds tokens in the KV pool, by cache and memory budget |
| `stage5_kv_contig_breakdown.png` | What holds tokens in the KV pool, contiguous cache only, by memory budget |
| `stage5_kv_paged_breakdown.png` | What holds tokens in the KV pool, paged cache only, by memory budget |
| `stage5_latency_vs_load.png` | Response time p50 and p99 vs offered load at the same KV memory |
| `stage5_latency_vs_memory.png` | Mean response time and TTFT p50 vs KV memory, contiguous vs paged |
| `stage5_throughput_vs_load.png` | Completed vs offered load at the same KV memory, contiguous vs paged |
| `stage5_ttft_vs_load.png` | Time to first token p50 and p99 vs offered load at the same KV memory |

## Stage 6: N-gram Speculation

Source: [`results/stage6.json`](results/stage6.json)

<a id="stage6-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** Qwen/Qwen2.5-0.5B
- **Model revision:** 060db6499f32faf8b98477b0a26969ef7d8b9987
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage6-verify-cost"></a>

### Cost of checking tokens

One forward pass on top of 512 cached tokens, every sequence feeding the given number of tokens (a draft of k tokens plus the last generated one, so k + 1). Relative cost is the pass time over the pass time with one token per sequence at the same batch size, so 1.00 means the extra tokens were free.

| batch size | tokens per sequence | pass time (ms) | relative cost | ms per token fed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 21.8 | 1.00x | 21.79 |
| 1 | 2 | 25.0 | 1.14x | 12.48 |
| 1 | 3 | 24.1 | 1.11x | 8.04 |
| 1 | 5 | 25.3 | 1.16x | 5.06 |
| 1 | 9 | 24.9 | 1.14x | 2.76 |
| 1 | 17 | 25.1 | 1.15x | 1.48 |
| 4 | 1 | 22.1 | 1.00x | 5.54 |
| 4 | 2 | 23.9 | 1.08x | 2.99 |
| 4 | 3 | 24.1 | 1.09x | 2.01 |
| 4 | 5 | 24.7 | 1.11x | 1.23 |
| 4 | 9 | 25.3 | 1.14x | 0.70 |
| 4 | 17 | 24.5 | 1.11x | 0.36 |
| 16 | 1 | 22.5 | 1.00x | 1.41 |
| 16 | 2 | 24.6 | 1.09x | 0.77 |
| 16 | 3 | 25.2 | 1.12x | 0.52 |
| 16 | 5 | 25.0 | 1.11x | 0.31 |
| 16 | 9 | 24.5 | 1.09x | 0.17 |
| 16 | 17 | 27.4 | 1.22x | 0.10 |
| 64 | 1 | 43.3 | 1.00x | 0.68 |
| 64 | 2 | 46.2 | 1.07x | 0.36 |
| 64 | 3 | 49.6 | 1.15x | 0.26 |
| 64 | 5 | 54.5 | 1.26x | 0.17 |
| 64 | 9 | 65.9 | 1.52x | 0.11 |
| 64 | 17 | 90.1 | 2.08x | 0.08 |

<a id="stage6-speculative"></a>

### Plain vs speculative decoding

Greedy decoding of 128 tokens per prompt, plain (k = 0) against speculative decoding with prompt lookup (ngram_max 3, ngram_min 1), on three workloads: copy (repeat a passage), code (repetitive code) and open (free-form writing). Each row is the whole workload, prefill and drafting included. Predicted speedup is tokens per pass times the plain step time over the checking pass time; the measured speedup is lower by the time spent outside the forward passes. Exact is how many outputs equal the plain greedy output.

| workload | draft length k | tokens/s | speedup | predicted | tokens per pass | acceptance | draft found | pass time (ms) | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| copy | 0 | 45.7 | 1.00x | n/a | 1.00 | n/a | n/a | 21.9 | 4 of 4 |
| copy | 2 | 91.0 | 1.99x | 2.02x | 2.18 | 80% | 74% | 23.6 | 3 of 4 |
| copy | 4 | 117.4 | 2.57x | 2.62x | 2.84 | 71% | 66% | 23.7 | 3 of 4 |
| copy | 8 | 147.8 | 3.24x | 3.32x | 3.58 | 60% | 58% | 23.6 | 3 of 4 |
| code | 0 | 45.6 | 1.00x | n/a | 1.00 | n/a | n/a | 21.9 | 4 of 4 |
| code | 2 | 91.4 | 2.01x | 2.04x | 2.19 | 76% | 78% | 23.6 | 4 of 4 |
| code | 4 | 116.8 | 2.56x | 2.62x | 2.82 | 64% | 73% | 23.7 | 4 of 4 |
| code | 8 | 141.3 | 3.10x | 3.18x | 3.41 | 47% | 67% | 23.5 | 4 of 4 |
| open | 0 | 46.3 | 1.00x | n/a | 1.00 | n/a | n/a | 21.6 | 4 of 4 |
| open | 2 | 76.2 | 1.65x | 1.67x | 1.80 | 68% | 59% | 23.4 | 4 of 4 |
| open | 4 | 92.7 | 2.00x | 2.03x | 2.14 | 57% | 52% | 22.8 | 4 of 4 |
| open | 8 | 102.4 | 2.21x | 2.25x | 2.41 | 42% | 46% | 23.1 | 4 of 4 |

<a id="stage6-oracle"></a>

### The ceiling

The ceiling: a draft that is always right (it knows the plain output), on the copy prompts. It shows the best any draft of that length could do on this model and GPU.

| draft length k | speedup |
| ---: | ---: |
| 2 | 2.63x |
| 4 | 4.30x |
| 8 | 7.03x |

<a id="stage6-projection"></a>

### Projection to batching

A projection, not a measurement: tokens per pass from the single-sequence runs divided by the relative cost of a checking pass at each batch size. It assumes every sequence behaves like the single sequence measured, and ignores drafting time and the raggedness of a batch, so it overstates the real gain and is useful for its trend.

| workload | batch size | draft length k | tokens per pass | relative pass cost | projected speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| copy | 1 | 2 | 2.18 | 1.11x | 1.97x |
| copy | 4 | 2 | 2.18 | 1.09x | 2.01x |
| copy | 16 | 2 | 2.18 | 1.12x | 1.95x |
| copy | 64 | 2 | 2.18 | 1.15x | 1.90x |
| copy | 1 | 4 | 2.84 | 1.16x | 2.44x |
| copy | 4 | 4 | 2.84 | 1.11x | 2.55x |
| copy | 16 | 4 | 2.84 | 1.11x | 2.55x |
| copy | 64 | 4 | 2.84 | 1.26x | 2.26x |
| copy | 1 | 8 | 3.58 | 1.14x | 3.13x |
| copy | 4 | 8 | 3.58 | 1.14x | 3.13x |
| copy | 16 | 8 | 3.58 | 1.09x | 3.28x |
| copy | 64 | 8 | 3.58 | 1.52x | 2.35x |
| code | 1 | 2 | 2.19 | 1.11x | 1.98x |
| code | 4 | 2 | 2.19 | 1.09x | 2.01x |
| code | 16 | 2 | 2.19 | 1.12x | 1.95x |
| code | 64 | 2 | 2.19 | 1.15x | 1.91x |
| code | 1 | 4 | 2.82 | 1.16x | 2.43x |
| code | 4 | 4 | 2.82 | 1.11x | 2.53x |
| code | 16 | 4 | 2.82 | 1.11x | 2.54x |
| code | 64 | 4 | 2.82 | 1.26x | 2.24x |
| code | 1 | 8 | 3.41 | 1.14x | 2.99x |
| code | 4 | 8 | 3.41 | 1.14x | 2.98x |
| code | 16 | 8 | 3.41 | 1.09x | 3.12x |
| code | 64 | 8 | 3.41 | 1.52x | 2.24x |
| open | 1 | 2 | 1.80 | 1.11x | 1.63x |
| open | 4 | 2 | 1.80 | 1.09x | 1.66x |
| open | 16 | 2 | 1.80 | 1.12x | 1.61x |
| open | 64 | 2 | 1.80 | 1.15x | 1.57x |
| open | 1 | 4 | 2.14 | 1.16x | 1.84x |
| open | 4 | 4 | 2.14 | 1.11x | 1.92x |
| open | 16 | 4 | 2.14 | 1.11x | 1.93x |
| open | 64 | 4 | 2.14 | 1.26x | 1.70x |
| open | 1 | 8 | 2.41 | 1.14x | 2.11x |
| open | 4 | 8 | 2.41 | 1.14x | 2.10x |
| open | 16 | 8 | 2.41 | 1.09x | 2.21x |
| open | 64 | 8 | 2.41 | 1.52x | 1.58x |

<a id="stage6-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage6_acceptance.png` | Acceptance and tokens per pass vs draft length, by workload |
| `stage6_batch_projection.png` | Projected speedup vs batch size from the measured pass cost and tokens per pass |
| `stage6_speedup.png` | Speedup over plain greedy decoding vs draft length, by workload, with the oracle ceiling |
| `stage6_verify_cost.png` | Forward pass time vs tokens fed per sequence, by batch size |

## Stage 7: Draft-Model Speculation

Source: [`results/stage7.json`](results/stage7.json)

<a id="stage7-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** draft Qwen/Qwen2.5-0.5B; targets Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B
- **Model revision:** Qwen2.5-1.5B 8faed761; Qwen2.5-0.5B 060db649; Qwen2.5-3B 3aab1f19
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage7-cost"></a>

### Cost of a draft step

Cost of one draft step against one plain target step, c, on top of 256 cached tokens, as a median of repeated steps with argmax included. The draft runs in three modes: eager (an ordinary cache), static (a fixed-shape cache, run eagerly) and graph (the single-token step replayed from a CUDA graph; if the capture was not possible it ran as static and the reason is given). The weights ratio is the draft's bytes over the target's and the layers ratio is the draft's layers over the target's.

| target | draft mode requested | ran as | draft step (ms) | target step (ms) | c | weights ratio | layers ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B | eager | eager | 21.29 | 25.48 | 0.84 | 0.32 | 0.86 |
| 1.5B | static | static | 22.31 | 25.48 | 0.88 | 0.32 | 0.86 |
| 1.5B | graph | graph | 9.90 | 25.48 | 0.39 | 0.32 | 0.86 |
| 3B | eager | eager | 21.35 | 34.66 | 0.62 | 0.16 | 0.67 |
| 3B | static | static | 22.34 | 34.66 | 0.64 | 0.16 | 0.67 |
| 3B | graph | graph | 9.52 | 34.66 | 0.27 | 0.16 | 0.67 |

<a id="stage7-gate"></a>

### The go/no-go gate

The go/no-go numbers on the 1.5B target: per-token acceptance a on the open prompts at k = 4 was 0.77 over 81 rounds, which predicts 3.15 tokens per round; the checking pass costs v = 1.11 plain steps. Predicted speedup is tokens per round over k times c plus v.

| draft mode requested | ran as | c | predicted speedup |
| --- | --- | ---: | ---: |
| eager | eager | 0.81 | 0.72x |
| static | static | 0.86 | 0.69x |
| graph | graph | 0.34 | 1.28x |

<a id="stage7-verify-cost"></a>

### Cost of checking tokens and of a draft step by batch size

One forward pass on top of 256 cached tokens, every sequence feeding the given number of tokens (the target: a draft of k tokens plus the last generated one, so k + 1; the draft: one token). Relative cost is the pass time over the pass time with one token per sequence at the same batch size.

| model | target | batch size | tokens per sequence | pass time (ms) | relative cost | ms per token fed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| target | 1.5B | 1 | 1 | 26.2 | 1.00x | 26.24 |
| target | 1.5B | 1 | 3 | 28.9 | 1.10x | 9.62 |
| target | 1.5B | 1 | 5 | 28.7 | 1.10x | 5.75 |
| target | 1.5B | 1 | 9 | 29.0 | 1.11x | 3.22 |
| target | 1.5B | 4 | 1 | 26.2 | 1.00x | 6.56 |
| target | 1.5B | 4 | 3 | 28.6 | 1.09x | 2.39 |
| target | 1.5B | 4 | 5 | 29.6 | 1.13x | 1.48 |
| target | 1.5B | 4 | 9 | 29.6 | 1.13x | 0.82 |
| target | 1.5B | 16 | 1 | 31.4 | 1.00x | 1.96 |
| target | 1.5B | 16 | 3 | 36.4 | 1.16x | 0.76 |
| target | 1.5B | 16 | 5 | 40.3 | 1.28x | 0.50 |
| target | 1.5B | 16 | 9 | 57.7 | 1.84x | 0.40 |
| target | 1.5B | 64 | 1 | 64.7 | 1.00x | 1.01 |
| target | 1.5B | 64 | 3 | 92.9 | 1.44x | 0.48 |
| target | 1.5B | 64 | 5 | 121.3 | 1.88x | 0.38 |
| target | 1.5B | 64 | 9 | 187.6 | 2.90x | 0.33 |
| target | 3B | 1 | 1 | 34.5 | 1.00x | 34.46 |
| target | 3B | 1 | 3 | 37.6 | 1.09x | 12.52 |
| target | 3B | 1 | 5 | 37.6 | 1.09x | 7.51 |
| target | 3B | 1 | 9 | 37.0 | 1.07x | 4.11 |
| target | 3B | 4 | 1 | 37.7 | 1.00x | 9.42 |
| target | 3B | 4 | 3 | 40.0 | 1.06x | 3.33 |
| target | 3B | 4 | 5 | 47.2 | 1.25x | 2.36 |
| target | 3B | 4 | 9 | 48.9 | 1.30x | 1.36 |
| target | 3B | 16 | 1 | 51.7 | 1.00x | 3.23 |
| target | 3B | 16 | 3 | 62.7 | 1.21x | 1.31 |
| target | 3B | 16 | 5 | 71.8 | 1.39x | 0.90 |
| target | 3B | 16 | 9 | 100.1 | 1.94x | 0.70 |
| target | 3B | 64 | 1 | 107.5 | 1.00x | 1.68 |
| target | 3B | 64 | 3 | 151.4 | 1.41x | 0.79 |
| target | 3B | 64 | 5 | 206.8 | 1.92x | 0.65 |
| target | 3B | 64 | 9 | 307.3 | 2.86x | 0.53 |
| draft | 1.5B | 1 | 1 | 21.5 | 1.00x | 21.54 |
| draft | 1.5B | 4 | 1 | 23.8 | 1.00x | 5.95 |
| draft | 1.5B | 16 | 1 | 22.3 | 1.00x | 1.40 |
| draft | 1.5B | 64 | 1 | 34.2 | 1.00x | 0.53 |
| draft | 3B | 1 | 1 | 20.8 | 1.00x | 20.83 |
| draft | 3B | 4 | 1 | 21.5 | 1.00x | 5.37 |
| draft | 3B | 16 | 1 | 21.8 | 1.00x | 1.36 |
| draft | 3B | 64 | 1 | 32.6 | 1.00x | 0.51 |

<a id="stage7-sweep"></a>

### Plain vs n-gram vs draft model

Greedy decoding of 128 tokens per prompt, plain (k = 0) against speculative decoding with each proposer, on three workloads: copy (repeat a passage), code (repetitive code) and open (free-form writing). Every proposer feeds the same checking loop. Each row is the whole workload, prefill and guessing included. a is the per-token acceptance rate (accepted over the positions examined) and expected tokens is what a predicts per round if positions were accepted independently. Predicted is expected tokens over k times c plus v, for the draft model only (the n-gram proposer has no fixed number of guesses). c(run) is c recomputed from the guess time of the run itself. Exact is how many outputs equal the plain greedy output.

**1.5B target**

| workload | proposer | draft length k | tokens/s | speedup | tokens per pass | a | expected tokens | predicted | c (run) | guess (ms/round) | check (ms/round) | exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| copy | plain | 0 | 38.6 | 1.00x | 1.00 | n/a | n/a | n/a | n/a | 0.0 | 25.9 | 4 of 4 |
| copy | n-gram | 2 | 62.9 | 1.63x | 1.77 | 0.72 | 2.23 | n/a | n/a | 0.0 | 27.8 | 4 of 4 |
| copy | n-gram | 4 | 74.5 | 1.93x | 2.07 | 0.74 | 3.01 | n/a | n/a | 0.0 | 27.4 | 4 of 4 |
| copy | n-gram | 8 | 83.5 | 2.17x | 2.33 | 0.76 | 3.83 | n/a | n/a | 0.0 | 27.4 | 4 of 4 |
| copy | draft eager | 2 | 36.2 | 0.94x | 2.62 | 0.87 | 2.63 | 0.95x | 0.85 | 43.9 | 28.2 | 4 of 4 |
| copy | draft eager | 4 | 33.8 | 0.88x | 3.88 | 0.89 | 3.97 | 0.90x | 0.83 | 86.3 | 28.2 | 4 of 4 |
| copy | draft eager | 8 | 27.2 | 0.71x | 5.40 | 0.88 | 5.75 | 0.74x | 0.82 | 169.8 | 29.1 | 4 of 4 |
| copy | draft graph | 2 | 41.4 | 1.07x | 2.62 | 0.87 | 2.63 | 1.40x | 0.64 | 32.9 | 30.0 | 4 of 4 |
| copy | draft graph | 4 | 47.5 | 1.23x | 3.88 | 0.89 | 3.97 | 1.50x | 0.49 | 51.3 | 29.9 | 4 of 4 |
| copy | draft graph | 8 | 45.6 | 1.18x | 5.40 | 0.88 | 5.75 | 1.37x | 0.42 | 87.5 | 30.5 | 4 of 4 |
| code | plain | 0 | 36.9 | 1.00x | 1.00 | n/a | n/a | n/a | n/a | 0.0 | 27.1 | 4 of 4 |
| code | n-gram | 2 | 67.7 | 1.84x | 2.01 | 0.75 | 2.31 | n/a | n/a | 0.0 | 29.2 | 4 of 4 |
| code | n-gram | 4 | 86.1 | 2.34x | 2.53 | 0.77 | 3.19 | n/a | n/a | 0.0 | 28.8 | 4 of 4 |
| code | n-gram | 8 | 99.6 | 2.70x | 2.92 | 0.77 | 3.95 | n/a | n/a | 0.0 | 28.7 | 4 of 4 |
| code | draft eager | 2 | 37.8 | 1.03x | 2.81 | 0.94 | 2.83 | 1.02x | 0.83 | 45.2 | 28.7 | 4 of 4 |
| code | draft eager | 4 | 36.8 | 1.00x | 4.23 | 0.93 | 4.39 | 0.99x | 0.79 | 86.2 | 28.5 | 4 of 4 |
| code | draft eager | 8 | 33.5 | 0.91x | 6.51 | 0.94 | 7.03 | 0.90x | 0.76 | 165.4 | 28.9 | 4 of 4 |
| code | draft graph | 2 | 46.2 | 1.25x | 2.81 | 0.94 | 2.83 | 1.51x | 0.59 | 32.0 | 28.4 | 4 of 4 |
| code | draft graph | 4 | 53.7 | 1.46x | 4.23 | 0.93 | 4.39 | 1.66x | 0.46 | 49.7 | 28.5 | 4 of 4 |
| code | draft graph | 8 | 56.5 | 1.53x | 6.51 | 0.94 | 7.03 | 1.67x | 0.39 | 85.4 | 29.1 | 4 of 4 |
| open | plain | 0 | 38.9 | 1.00x | 1.00 | n/a | n/a | n/a | n/a | 0.0 | 25.7 | 4 of 4 |
| open | n-gram | 2 | 47.4 | 1.22x | 1.30 | 0.49 | 1.72 | n/a | n/a | 0.0 | 27.2 | 4 of 4 |
| open | n-gram | 4 | 50.0 | 1.29x | 1.37 | 0.50 | 1.93 | n/a | n/a | 0.0 | 27.1 | 4 of 4 |
| open | n-gram | 8 | 51.3 | 1.32x | 1.42 | 0.53 | 2.13 | n/a | n/a | 0.0 | 27.3 | 4 of 4 |
| open | draft eager | 2 | 33.2 | 0.85x | 2.47 | 0.82 | 2.50 | 0.90x | 0.87 | 45.0 | 29.1 | 2 of 4 |
| open | draft eager | 4 | 28.9 | 0.74x | 3.32 | 0.80 | 3.34 | 0.75x | 0.84 | 86.0 | 28.7 | 4 of 4 |
| open | draft eager | 8 | 20.7 | 0.53x | 4.03 | 0.80 | 4.27 | 0.55x | 0.81 | 166.1 | 28.8 | 3 of 4 |
| open | draft graph | 2 | 41.0 | 1.05x | 2.47 | 0.82 | 2.50 | 1.33x | 0.61 | 31.5 | 28.4 | 2 of 4 |
| open | draft graph | 4 | 42.1 | 1.08x | 3.32 | 0.80 | 3.34 | 1.26x | 0.49 | 49.9 | 28.7 | 4 of 4 |
| open | draft graph | 8 | 34.8 | 0.90x | 4.03 | 0.80 | 4.27 | 1.01x | 0.42 | 86.5 | 29.0 | 3 of 4 |

**3B target**

| workload | proposer | draft length k | tokens/s | speedup | tokens per pass | a | expected tokens | predicted | c (run) | guess (ms/round) | check (ms/round) | exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| copy | plain | 0 | 24.3 | 1.00x | 1.00 | n/a | n/a | n/a | n/a | 0.0 | 41.0 | 4 of 4 |
| copy | n-gram | 2 | 58.4 | 2.41x | 2.30 | 0.89 | 2.69 | n/a | n/a | 0.0 | 38.2 | 3 of 4 |
| copy | n-gram | 4 | 75.4 | 3.10x | 3.10 | 0.91 | 4.18 | n/a | n/a | 0.0 | 39.7 | 4 of 4 |
| copy | n-gram | 8 | 96.4 | 3.97x | 4.06 | 0.92 | 6.56 | n/a | n/a | 0.0 | 40.3 | 4 of 4 |
| copy | draft eager | 2 | 33.5 | 1.38x | 2.79 | 0.94 | 2.82 | 1.22x | 0.55 | 45.2 | 37.3 | 4 of 4 |
| copy | draft eager | 4 | 34.4 | 1.42x | 4.31 | 0.94 | 4.43 | 1.25x | 0.53 | 86.6 | 37.5 | 3 of 4 |
| copy | draft eager | 8 | 31.6 | 1.30x | 6.43 | 0.94 | 7.03 | 1.17x | 0.50 | 164.6 | 37.6 | 4 of 4 |
| copy | draft graph | 2 | 39.4 | 1.62x | 2.79 | 0.94 | 2.82 | 1.72x | 0.40 | 32.7 | 37.3 | 4 of 4 |
| copy | draft graph | 4 | 47.1 | 1.94x | 4.31 | 0.94 | 4.43 | 2.03x | 0.32 | 52.1 | 37.9 | 3 of 4 |
| copy | draft graph | 8 | 50.2 | 2.07x | 6.43 | 0.94 | 7.03 | 2.15x | 0.27 | 88.2 | 37.9 | 4 of 4 |
| code | plain | 0 | 24.3 | 1.00x | 1.00 | n/a | n/a | n/a | n/a | 0.0 | 40.9 | 4 of 4 |
| code | n-gram | 2 | 50.7 | 2.09x | 1.90 | 0.71 | 2.21 | n/a | n/a | 0.0 | 36.6 | 4 of 4 |
| code | n-gram | 4 | 60.5 | 2.49x | 2.29 | 0.73 | 2.95 | n/a | n/a | 0.0 | 36.8 | 4 of 4 |
| code | n-gram | 8 | 67.2 | 2.77x | 2.59 | 0.74 | 3.55 | n/a | n/a | 0.0 | 37.4 | 4 of 4 |
| code | draft eager | 2 | 31.8 | 1.31x | 2.65 | 0.90 | 2.70 | 1.16x | 0.55 | 44.7 | 37.6 | 4 of 4 |
| code | draft eager | 4 | 32.5 | 1.34x | 4.10 | 0.90 | 4.11 | 1.16x | 0.53 | 87.1 | 37.8 | 4 of 4 |
| code | draft eager | 8 | 29.2 | 1.20x | 6.05 | 0.91 | 6.31 | 1.05x | 0.51 | 168.1 | 37.8 | 4 of 4 |
| code | draft graph | 2 | 37.5 | 1.54x | 2.65 | 0.90 | 2.70 | 1.64x | 0.40 | 32.5 | 37.3 | 4 of 4 |
| code | draft graph | 4 | 45.1 | 1.86x | 4.10 | 0.90 | 4.11 | 1.88x | 0.32 | 51.8 | 37.7 | 4 of 4 |
| code | draft graph | 8 | 47.0 | 1.93x | 6.05 | 0.91 | 6.31 | 1.93x | 0.27 | 88.7 | 38.0 | 4 of 4 |
| open | plain | 0 | 24.7 | 1.00x | 1.00 | n/a | n/a | n/a | n/a | 0.0 | 40.4 | 4 of 4 |
| open | n-gram | 2 | 33.4 | 1.35x | 1.23 | 0.41 | 1.59 | n/a | n/a | 0.0 | 36.5 | 3 of 4 |
| open | n-gram | 4 | 34.8 | 1.41x | 1.29 | 0.44 | 1.74 | n/a | n/a | 0.0 | 36.7 | 3 of 4 |
| open | n-gram | 8 | 36.1 | 1.46x | 1.34 | 0.47 | 1.89 | n/a | n/a | 0.0 | 36.8 | 4 of 4 |
| open | draft eager | 2 | 28.0 | 1.13x | 2.28 | 0.75 | 2.31 | 0.99x | 0.55 | 44.1 | 37.1 | 3 of 4 |
| open | draft eager | 4 | 24.5 | 0.99x | 3.02 | 0.76 | 3.08 | 0.87x | 0.53 | 85.5 | 37.6 | 3 of 4 |
| open | draft eager | 8 | 16.5 | 0.66x | 3.34 | 0.73 | 3.47 | 0.58x | 0.51 | 165.7 | 37.8 | 4 of 4 |
| open | draft graph | 2 | 32.3 | 1.31x | 2.28 | 0.75 | 2.31 | 1.41x | 0.40 | 32.7 | 37.4 | 3 of 4 |
| open | draft graph | 4 | 33.8 | 1.37x | 3.02 | 0.76 | 3.08 | 1.41x | 0.32 | 51.2 | 38.0 | 3 of 4 |
| open | draft graph | 8 | 26.5 | 1.07x | 3.34 | 0.73 | 3.47 | 1.06x | 0.27 | 87.6 | 38.4 | 4 of 4 |

<a id="stage7-head-to-head"></a>

### Head to head

For each target, workload and proposer, the draft length k with the best measured speedup, side by side. Best marks the proposer with the highest speedup for that workload and target.

| target | workload | proposer | best k | speedup | a | tokens per pass | guess (ms/round) | check (ms/round) |  |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.5B | copy | n-gram | 8 | 2.17x | 0.76 | 2.33 | 0.0 | 27.4 | best |
| 1.5B | copy | draft eager | 2 | 0.94x | 0.87 | 2.62 | 43.9 | 28.2 |  |
| 1.5B | copy | draft graph | 4 | 1.23x | 0.89 | 3.88 | 51.3 | 29.9 |  |
| 1.5B | code | n-gram | 8 | 2.70x | 0.77 | 2.92 | 0.0 | 28.7 | best |
| 1.5B | code | draft eager | 2 | 1.03x | 0.94 | 2.81 | 45.2 | 28.7 |  |
| 1.5B | code | draft graph | 8 | 1.53x | 0.94 | 6.51 | 85.4 | 29.1 |  |
| 1.5B | open | n-gram | 8 | 1.32x | 0.53 | 1.42 | 0.0 | 27.3 | best |
| 1.5B | open | draft eager | 2 | 0.85x | 0.82 | 2.47 | 45.0 | 29.1 |  |
| 1.5B | open | draft graph | 4 | 1.08x | 0.80 | 3.32 | 49.9 | 28.7 |  |
| 3B | copy | n-gram | 8 | 3.97x | 0.92 | 4.06 | 0.0 | 40.3 | best |
| 3B | copy | draft eager | 4 | 1.42x | 0.94 | 4.31 | 86.6 | 37.5 |  |
| 3B | copy | draft graph | 8 | 2.07x | 0.94 | 6.43 | 88.2 | 37.9 |  |
| 3B | code | n-gram | 8 | 2.77x | 0.74 | 2.59 | 0.0 | 37.4 | best |
| 3B | code | draft eager | 4 | 1.34x | 0.90 | 4.10 | 87.1 | 37.8 |  |
| 3B | code | draft graph | 8 | 1.93x | 0.91 | 6.05 | 88.7 | 38.0 |  |
| 3B | open | n-gram | 8 | 1.46x | 0.47 | 1.34 | 0.0 | 36.8 | best |
| 3B | open | draft eager | 2 | 1.13x | 0.75 | 2.28 | 44.1 | 37.1 |  |
| 3B | open | draft graph | 4 | 1.37x | 0.76 | 3.02 | 51.2 | 38.0 |  |

<a id="stage7-projection"></a>

### Projection to batching

A projection, not a measurement: tokens per pass from the single-sequence runs divided by k times c(B) plus v(B), where c(B) is the eager draft's step time over the target's at batch size B and v(B) is the relative cost of a checking pass of k + 1 tokens at batch size B (zero draft cost for the n-gram proposer). It assumes every sequence behaves like the single sequence measured and ignores the raggedness of a batch, so it overstates the real gain and is useful for its trend. The CUDA-graph draft is not projected.

| target | workload | proposer | batch size | draft length k | tokens per pass | c(B) | v(B) | projected speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | copy | n-gram | 1 | 2 | 1.77 | 0.00 | 1.10x | 1.61x |
| 1.5B | copy | n-gram | 4 | 2 | 1.77 | 0.00 | 1.09x | 1.62x |
| 1.5B | copy | n-gram | 16 | 2 | 1.77 | 0.00 | 1.16x | 1.53x |
| 1.5B | copy | n-gram | 64 | 2 | 1.77 | 0.00 | 1.44x | 1.23x |
| 1.5B | copy | n-gram | 1 | 4 | 2.07 | 0.00 | 1.10x | 1.89x |
| 1.5B | copy | n-gram | 4 | 4 | 2.07 | 0.00 | 1.13x | 1.84x |
| 1.5B | copy | n-gram | 16 | 4 | 2.07 | 0.00 | 1.28x | 1.61x |
| 1.5B | copy | n-gram | 64 | 4 | 2.07 | 0.00 | 1.88x | 1.10x |
| 1.5B | copy | n-gram | 1 | 8 | 2.33 | 0.00 | 1.11x | 2.11x |
| 1.5B | copy | n-gram | 4 | 8 | 2.33 | 0.00 | 1.13x | 2.06x |
| 1.5B | copy | n-gram | 16 | 8 | 2.33 | 0.00 | 1.84x | 1.27x |
| 1.5B | copy | n-gram | 64 | 8 | 2.33 | 0.00 | 2.90x | 0.80x |
| 1.5B | copy | draft eager | 1 | 2 | 2.62 | 0.82 | 1.10x | 0.96x |
| 1.5B | copy | draft eager | 4 | 2 | 2.62 | 0.91 | 1.09x | 0.90x |
| 1.5B | copy | draft eager | 16 | 2 | 2.62 | 0.71 | 1.16x | 1.01x |
| 1.5B | copy | draft eager | 64 | 2 | 2.62 | 0.53 | 1.44x | 1.05x |
| 1.5B | copy | draft eager | 1 | 4 | 3.88 | 0.82 | 1.10x | 0.89x |
| 1.5B | copy | draft eager | 4 | 4 | 3.88 | 0.91 | 1.13x | 0.81x |
| 1.5B | copy | draft eager | 16 | 4 | 3.88 | 0.71 | 1.28x | 0.94x |
| 1.5B | copy | draft eager | 64 | 4 | 3.88 | 0.53 | 1.88x | 0.97x |
| 1.5B | copy | draft eager | 1 | 8 | 5.40 | 0.82 | 1.11x | 0.70x |
| 1.5B | copy | draft eager | 4 | 8 | 5.40 | 0.91 | 1.13x | 0.64x |
| 1.5B | copy | draft eager | 16 | 8 | 5.40 | 0.71 | 1.84x | 0.72x |
| 1.5B | copy | draft eager | 64 | 8 | 5.40 | 0.53 | 2.90x | 0.76x |
| 1.5B | code | n-gram | 1 | 2 | 2.01 | 0.00 | 1.10x | 1.83x |
| 1.5B | code | n-gram | 4 | 2 | 2.01 | 0.00 | 1.09x | 1.84x |
| 1.5B | code | n-gram | 16 | 2 | 2.01 | 0.00 | 1.16x | 1.73x |
| 1.5B | code | n-gram | 64 | 2 | 2.01 | 0.00 | 1.44x | 1.40x |
| 1.5B | code | n-gram | 1 | 4 | 2.53 | 0.00 | 1.10x | 2.31x |
| 1.5B | code | n-gram | 4 | 4 | 2.53 | 0.00 | 1.13x | 2.24x |
| 1.5B | code | n-gram | 16 | 4 | 2.53 | 0.00 | 1.28x | 1.97x |
| 1.5B | code | n-gram | 64 | 4 | 2.53 | 0.00 | 1.88x | 1.35x |
| 1.5B | code | n-gram | 1 | 8 | 2.92 | 0.00 | 1.11x | 2.64x |
| 1.5B | code | n-gram | 4 | 8 | 2.92 | 0.00 | 1.13x | 2.59x |
| 1.5B | code | n-gram | 16 | 8 | 2.92 | 0.00 | 1.84x | 1.59x |
| 1.5B | code | n-gram | 64 | 8 | 2.92 | 0.00 | 2.90x | 1.01x |
| 1.5B | code | draft eager | 1 | 2 | 2.81 | 0.82 | 1.10x | 1.02x |
| 1.5B | code | draft eager | 4 | 2 | 2.81 | 0.91 | 1.09x | 0.96x |
| 1.5B | code | draft eager | 16 | 2 | 2.81 | 0.71 | 1.16x | 1.09x |
| 1.5B | code | draft eager | 64 | 2 | 2.81 | 0.53 | 1.44x | 1.12x |
| 1.5B | code | draft eager | 1 | 4 | 4.23 | 0.82 | 1.10x | 0.97x |
| 1.5B | code | draft eager | 4 | 4 | 4.23 | 0.91 | 1.13x | 0.89x |
| 1.5B | code | draft eager | 16 | 4 | 4.23 | 0.71 | 1.28x | 1.03x |
| 1.5B | code | draft eager | 64 | 4 | 4.23 | 0.53 | 1.88x | 1.06x |
| 1.5B | code | draft eager | 1 | 8 | 6.51 | 0.82 | 1.11x | 0.85x |
| 1.5B | code | draft eager | 4 | 8 | 6.51 | 0.91 | 1.13x | 0.78x |
| 1.5B | code | draft eager | 16 | 8 | 6.51 | 0.71 | 1.84x | 0.87x |
| 1.5B | code | draft eager | 64 | 8 | 6.51 | 0.53 | 2.90x | 0.91x |
| 1.5B | open | n-gram | 1 | 2 | 1.30 | 0.00 | 1.10x | 1.18x |
| 1.5B | open | n-gram | 4 | 2 | 1.30 | 0.00 | 1.09x | 1.19x |
| 1.5B | open | n-gram | 16 | 2 | 1.30 | 0.00 | 1.16x | 1.12x |
| 1.5B | open | n-gram | 64 | 2 | 1.30 | 0.00 | 1.44x | 0.90x |
| 1.5B | open | n-gram | 1 | 4 | 1.37 | 0.00 | 1.10x | 1.25x |
| 1.5B | open | n-gram | 4 | 4 | 1.37 | 0.00 | 1.13x | 1.21x |
| 1.5B | open | n-gram | 16 | 4 | 1.37 | 0.00 | 1.28x | 1.06x |
| 1.5B | open | n-gram | 64 | 4 | 1.37 | 0.00 | 1.88x | 0.73x |
| 1.5B | open | n-gram | 1 | 8 | 1.42 | 0.00 | 1.11x | 1.28x |
| 1.5B | open | n-gram | 4 | 8 | 1.42 | 0.00 | 1.13x | 1.25x |
| 1.5B | open | n-gram | 16 | 8 | 1.42 | 0.00 | 1.84x | 0.77x |
| 1.5B | open | n-gram | 64 | 8 | 1.42 | 0.00 | 2.90x | 0.49x |
| 1.5B | open | draft eager | 1 | 2 | 2.47 | 0.82 | 1.10x | 0.90x |
| 1.5B | open | draft eager | 4 | 2 | 2.47 | 0.91 | 1.09x | 0.85x |
| 1.5B | open | draft eager | 16 | 2 | 2.47 | 0.71 | 1.16x | 0.96x |
| 1.5B | open | draft eager | 64 | 2 | 2.47 | 0.53 | 1.44x | 0.99x |
| 1.5B | open | draft eager | 1 | 4 | 3.32 | 0.82 | 1.10x | 0.76x |
| 1.5B | open | draft eager | 4 | 4 | 3.32 | 0.91 | 1.13x | 0.70x |
| 1.5B | open | draft eager | 16 | 4 | 3.32 | 0.71 | 1.28x | 0.80x |
| 1.5B | open | draft eager | 64 | 4 | 3.32 | 0.53 | 1.88x | 0.83x |
| 1.5B | open | draft eager | 1 | 8 | 4.03 | 0.82 | 1.11x | 0.53x |
| 1.5B | open | draft eager | 4 | 8 | 4.03 | 0.91 | 1.13x | 0.48x |
| 1.5B | open | draft eager | 16 | 8 | 4.03 | 0.71 | 1.84x | 0.54x |
| 1.5B | open | draft eager | 64 | 8 | 4.03 | 0.53 | 2.90x | 0.57x |
| 3B | copy | n-gram | 1 | 2 | 2.30 | 0.00 | 1.09x | 2.11x |
| 3B | copy | n-gram | 4 | 2 | 2.30 | 0.00 | 1.06x | 2.16x |
| 3B | copy | n-gram | 16 | 2 | 2.30 | 0.00 | 1.21x | 1.89x |
| 3B | copy | n-gram | 64 | 2 | 2.30 | 0.00 | 1.41x | 1.63x |
| 3B | copy | n-gram | 1 | 4 | 3.10 | 0.00 | 1.09x | 2.84x |
| 3B | copy | n-gram | 4 | 4 | 3.10 | 0.00 | 1.25x | 2.47x |
| 3B | copy | n-gram | 16 | 4 | 3.10 | 0.00 | 1.39x | 2.23x |
| 3B | copy | n-gram | 64 | 4 | 3.10 | 0.00 | 1.92x | 1.61x |
| 3B | copy | n-gram | 1 | 8 | 4.06 | 0.00 | 1.07x | 3.78x |
| 3B | copy | n-gram | 4 | 8 | 4.06 | 0.00 | 1.30x | 3.13x |
| 3B | copy | n-gram | 16 | 8 | 4.06 | 0.00 | 1.94x | 2.10x |
| 3B | copy | n-gram | 64 | 8 | 4.06 | 0.00 | 2.86x | 1.42x |
| 3B | copy | draft eager | 1 | 2 | 2.79 | 0.60 | 1.09x | 1.21x |
| 3B | copy | draft eager | 4 | 2 | 2.79 | 0.57 | 1.06x | 1.27x |
| 3B | copy | draft eager | 16 | 2 | 2.79 | 0.42 | 1.21x | 1.36x |
| 3B | copy | draft eager | 64 | 2 | 2.79 | 0.30 | 1.41x | 1.38x |
| 3B | copy | draft eager | 1 | 4 | 4.31 | 0.60 | 1.09x | 1.23x |
| 3B | copy | draft eager | 4 | 4 | 4.31 | 0.57 | 1.25x | 1.22x |
| 3B | copy | draft eager | 16 | 4 | 4.31 | 0.42 | 1.39x | 1.40x |
| 3B | copy | draft eager | 64 | 4 | 4.31 | 0.30 | 1.92x | 1.37x |
| 3B | copy | draft eager | 1 | 8 | 6.43 | 0.60 | 1.07x | 1.09x |
| 3B | copy | draft eager | 4 | 8 | 6.43 | 0.57 | 1.30x | 1.10x |
| 3B | copy | draft eager | 16 | 8 | 6.43 | 0.42 | 1.94x | 1.21x |
| 3B | copy | draft eager | 64 | 8 | 6.43 | 0.30 | 2.86x | 1.22x |
| 3B | code | n-gram | 1 | 2 | 1.90 | 0.00 | 1.09x | 1.74x |
| 3B | code | n-gram | 4 | 2 | 1.90 | 0.00 | 1.06x | 1.78x |
| 3B | code | n-gram | 16 | 2 | 1.90 | 0.00 | 1.21x | 1.56x |
| 3B | code | n-gram | 64 | 2 | 1.90 | 0.00 | 1.41x | 1.35x |
| 3B | code | n-gram | 1 | 4 | 2.29 | 0.00 | 1.09x | 2.10x |
| 3B | code | n-gram | 4 | 4 | 2.29 | 0.00 | 1.25x | 1.83x |
| 3B | code | n-gram | 16 | 4 | 2.29 | 0.00 | 1.39x | 1.65x |
| 3B | code | n-gram | 64 | 4 | 2.29 | 0.00 | 1.92x | 1.19x |
| 3B | code | n-gram | 1 | 8 | 2.59 | 0.00 | 1.07x | 2.41x |
| 3B | code | n-gram | 4 | 8 | 2.59 | 0.00 | 1.30x | 2.00x |
| 3B | code | n-gram | 16 | 8 | 2.59 | 0.00 | 1.94x | 1.34x |
| 3B | code | n-gram | 64 | 8 | 2.59 | 0.00 | 2.86x | 0.91x |
| 3B | code | draft eager | 1 | 2 | 2.65 | 0.60 | 1.09x | 1.15x |
| 3B | code | draft eager | 4 | 2 | 2.65 | 0.57 | 1.06x | 1.20x |
| 3B | code | draft eager | 16 | 2 | 2.65 | 0.42 | 1.21x | 1.29x |
| 3B | code | draft eager | 64 | 2 | 2.65 | 0.30 | 1.41x | 1.31x |
| 3B | code | draft eager | 1 | 4 | 4.10 | 0.60 | 1.09x | 1.17x |
| 3B | code | draft eager | 4 | 4 | 4.10 | 0.57 | 1.25x | 1.16x |
| 3B | code | draft eager | 16 | 4 | 4.10 | 0.42 | 1.39x | 1.33x |
| 3B | code | draft eager | 64 | 4 | 4.10 | 0.30 | 1.92x | 1.31x |
| 3B | code | draft eager | 1 | 8 | 6.05 | 0.60 | 1.07x | 1.02x |
| 3B | code | draft eager | 4 | 8 | 6.05 | 0.57 | 1.30x | 1.03x |
| 3B | code | draft eager | 16 | 8 | 6.05 | 0.42 | 1.94x | 1.14x |
| 3B | code | draft eager | 64 | 8 | 6.05 | 0.30 | 2.86x | 1.14x |
| 3B | open | n-gram | 1 | 2 | 1.23 | 0.00 | 1.09x | 1.13x |
| 3B | open | n-gram | 4 | 2 | 1.23 | 0.00 | 1.06x | 1.16x |
| 3B | open | n-gram | 16 | 2 | 1.23 | 0.00 | 1.21x | 1.01x |
| 3B | open | n-gram | 64 | 2 | 1.23 | 0.00 | 1.41x | 0.87x |
| 3B | open | n-gram | 1 | 4 | 1.29 | 0.00 | 1.09x | 1.18x |
| 3B | open | n-gram | 4 | 4 | 1.29 | 0.00 | 1.25x | 1.03x |
| 3B | open | n-gram | 16 | 4 | 1.29 | 0.00 | 1.39x | 0.93x |
| 3B | open | n-gram | 64 | 4 | 1.29 | 0.00 | 1.92x | 0.67x |
| 3B | open | n-gram | 1 | 8 | 1.34 | 0.00 | 1.07x | 1.24x |
| 3B | open | n-gram | 4 | 8 | 1.34 | 0.00 | 1.30x | 1.03x |
| 3B | open | n-gram | 16 | 8 | 1.34 | 0.00 | 1.94x | 0.69x |
| 3B | open | n-gram | 64 | 8 | 1.34 | 0.00 | 2.86x | 0.47x |
| 3B | open | draft eager | 1 | 2 | 2.28 | 0.60 | 1.09x | 0.99x |
| 3B | open | draft eager | 4 | 2 | 2.28 | 0.57 | 1.06x | 1.03x |
| 3B | open | draft eager | 16 | 2 | 2.28 | 0.42 | 1.21x | 1.11x |
| 3B | open | draft eager | 64 | 2 | 2.28 | 0.30 | 1.41x | 1.13x |
| 3B | open | draft eager | 1 | 4 | 3.02 | 0.60 | 1.09x | 0.86x |
| 3B | open | draft eager | 4 | 4 | 3.02 | 0.57 | 1.25x | 0.86x |
| 3B | open | draft eager | 16 | 4 | 3.02 | 0.42 | 1.39x | 0.98x |
| 3B | open | draft eager | 64 | 4 | 3.02 | 0.30 | 1.92x | 0.96x |
| 3B | open | draft eager | 1 | 8 | 3.34 | 0.60 | 1.07x | 0.57x |
| 3B | open | draft eager | 4 | 8 | 3.34 | 0.57 | 1.30x | 0.57x |
| 3B | open | draft eager | 16 | 8 | 3.34 | 0.42 | 1.94x | 0.63x |
| 3B | open | draft eager | 64 | 8 | 3.34 | 0.30 | 2.86x | 0.63x |

<a id="stage7-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage7_acceptance_1p5b.png` | Acceptance and tokens per pass vs draft length, n-gram and draft model, 1.5B target |
| `stage7_acceptance_3b.png` | Acceptance and tokens per pass vs draft length, n-gram and draft model, 3B target |
| `stage7_batch_projection_1p5b.png` | Projected speedup vs batch size, n-gram and draft model, 1.5B target |
| `stage7_batch_projection_3b.png` | Projected speedup vs batch size, n-gram and draft model, 3B target |
| `stage7_cost_by_target.png` | The cost ratio c by target model and execution mode, with the weights and layers ratios |
| `stage7_cost_ratio.png` | Draft step cost over target step cost: the weights ratio, the layers ratio and each execution mode of the draft |
| `stage7_head_to_head_1p5b.png` | Best speedup by workload for each proposer, 1.5B target |
| `stage7_head_to_head_3b.png` | Best speedup by workload for each proposer, 3B target |
| `stage7_predicted_vs_c.png` | Predicted speedup vs the cost ratio c at the measured acceptance, with the measured modes marked |
| `stage7_predicted_vs_measured.png` | Measured vs predicted speedup of the draft model, both targets |
| `stage7_speedup_1p5b.png` | Draft-model speedup over plain greedy decoding vs draft length, 1.5B target |
| `stage7_speedup_3b.png` | Draft-model speedup over plain greedy decoding vs draft length, 3B target |
| `stage7_time_split_1p5b.png` | Time per round spent guessing and in the target's checking pass, 1.5B target |
| `stage7_time_split_3b.png` | Time per round spent guessing and in the target's checking pass, 3B target |

## Stage 8: Weight-Only Int8 Quantization

Source: [`results/stage8.json`](results/stage8.json)

<a id="stage8-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B
- **Model revision:** Qwen2.5-0.5B 060db649; Qwen2.5-1.5B 8faed761; Qwen2.5-3B 3aab1f19
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage8-gate"></a>

### The go/no-go gate: one linear layer

One linear layer y = x @ W.T at real Qwen2.5 weight shapes, timed as the median of repeated calls (spread is the interquartile range over the median), with the fp16 weight and with the int8 weight through each way of computing it. GB/s is the bytes of the stored weight over the time, against a measured copy bandwidth of 243 GB/s; vs fp16 is the fp16 time over this time, above one means faster than the fp16 layer. A way that does not work in this build is listed with the reason.

| layer | weight shape | batch | way | works | time (ms) | spread | GB/s | vs fp16 | reason it does not |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| mlp gate/up | 8960 x 1536 | 1 | fp | yes | 0.149 | 5% | 185 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 dequant, eager | yes | 0.557 | 1% | 25 | 0.27x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 dequant, compiled | yes | 0.456 | 7% | 30 | 0.33x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 intpack | yes | 1.624 | 7% | 8 | 0.09x |  |
| mlp gate/up | 8960 x 1536 | 16 | fp | yes | 0.179 | 3% | 153 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 dequant, eager | yes | 0.612 | 1% | 23 | 0.29x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 dequant, compiled | yes | 0.457 | 4% | 30 | 0.39x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 intpack | yes | 8.714 | 0% | 2 | 0.02x |  |
| mlp down | 1536 x 8960 | 1 | fp | yes | 0.142 | 4% | 194 | 1.00x |  |
| mlp down | 1536 x 8960 | 1 | int8 dequant, eager | yes | 0.570 | 2% | 24 | 0.25x |  |
| mlp down | 1536 x 8960 | 1 | int8 dequant, compiled | yes | 0.400 | 3% | 34 | 0.35x |  |
| mlp down | 1536 x 8960 | 1 | int8 intpack | yes | 0.778 | 1% | 18 | 0.18x |  |
| mlp down | 1536 x 8960 | 16 | fp | yes | 0.147 | 5% | 188 | 1.00x |  |
| mlp down | 1536 x 8960 | 16 | int8 dequant, eager | yes | 0.474 | 2% | 29 | 0.31x |  |
| mlp down | 1536 x 8960 | 16 | int8 dequant, compiled | yes | 0.415 | 3% | 33 | 0.35x |  |
| mlp down | 1536 x 8960 | 16 | int8 intpack | yes | 5.708 | 0% | 2 | 0.03x |  |
| attn q/o | 1536 x 1536 | 1 | fp | yes | 0.044 | 10% | 107 | 1.00x |  |
| attn q/o | 1536 x 1536 | 1 | int8 dequant, eager | yes | 0.115 | 13% | 21 | 0.38x |  |
| attn q/o | 1536 x 1536 | 1 | int8 dequant, compiled | yes | 0.200 | 21% | 12 | 0.22x |  |
| attn q/o | 1536 x 1536 | 1 | int8 intpack | yes | 0.133 | 3% | 18 | 0.33x |  |
| attn q/o | 1536 x 1536 | 16 | fp | yes | 0.050 | 9% | 95 | 1.00x |  |
| attn q/o | 1536 x 1536 | 16 | int8 dequant, eager | yes | 0.123 | 13% | 19 | 0.40x |  |
| attn q/o | 1536 x 1536 | 16 | int8 dequant, compiled | yes | 0.171 | 18% | 14 | 0.29x |  |
| attn q/o | 1536 x 1536 | 16 | int8 intpack | yes | 0.762 | 0% | 3 | 0.07x |  |
| attn k/v | 256 x 1536 | 1 | fp | yes | 0.030 | 9% | 26 | 1.00x |  |
| attn k/v | 256 x 1536 | 1 | int8 dequant, eager | yes | 0.095 | 16% | 4 | 0.32x |  |
| attn k/v | 256 x 1536 | 1 | int8 dequant, compiled | yes | 0.188 | 17% | 2 | 0.16x |  |
| attn k/v | 256 x 1536 | 1 | int8 intpack | yes | 0.079 | 5% | 5 | 0.38x |  |
| attn k/v | 256 x 1536 | 16 | fp | yes | 0.040 | 27% | 20 | 1.00x |  |
| attn k/v | 256 x 1536 | 16 | int8 dequant, eager | yes | 0.115 | 15% | 3 | 0.35x |  |
| attn k/v | 256 x 1536 | 16 | int8 dequant, compiled | yes | 0.196 | 9% | 2 | 0.20x |  |
| attn k/v | 256 x 1536 | 16 | int8 intpack | yes | 0.201 | 2% | 2 | 0.20x |  |
| lm_head | 151936 x 1536 | 1 | fp | yes | 2.278 | 1% | 205 | 1.00x |  |
| lm_head | 151936 x 1536 | 1 | int8 dequant, eager | yes | 7.309 | 8% | 32 | 0.31x |  |
| lm_head | 151936 x 1536 | 1 | int8 dequant, compiled | yes | 5.449 | 1% | 43 | 0.42x |  |
| lm_head | 151936 x 1536 | 1 | int8 intpack | yes | 8.211 | 5% | 28 | 0.28x |  |
| lm_head | 151936 x 1536 | 16 | fp | yes | 2.037 | 1% | 229 | 1.00x |  |
| lm_head | 151936 x 1536 | 16 | int8 dequant, eager | yes | 7.066 | 3% | 33 | 0.29x |  |
| lm_head | 151936 x 1536 | 16 | int8 dequant, compiled | yes | 5.233 | 1% | 45 | 0.39x |  |
| lm_head | 151936 x 1536 | 16 | int8 intpack | yes | 65.844 | 0% | 4 | 0.03x |  |

<a id="stage8-quantizer"></a>

### The quantizer: error by row

Relative error of each output row of a weight matrix after quantization, median over the rows of every matrix of that kind, with one scale per channel and with one scale for the whole matrix (the control).

| model | matrix | per-channel scales | per-tensor scale | per-tensor / per-channel |
| --- | --- | ---: | ---: | ---: |
| 0.5B | q_w | 0.0088 | 0.0324 | 3.7x |
| 0.5B | k_w | 0.0087 | 0.0211 | 2.4x |
| 0.5B | v_w | 0.0093 | 0.0160 | 1.7x |
| 0.5B | o_w | 0.0092 | 0.0446 | 4.9x |
| 0.5B | gate | 0.0081 | 0.0480 | 5.9x |
| 0.5B | up | 0.0081 | 0.0500 | 6.2x |
| 0.5B | down | 0.0099 | 0.0547 | 5.5x |
| 0.5B | lm_head | 0.0084 | 0.0303 | 3.6x |

<a id="stage8-accuracy"></a>

### Accuracy

**Logits against fp16** (all positions of the evaluation texts)

| model | variant | max abs diff | mean abs diff | KL (nats) | same top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | int8 per-channel | 6.547 | 0.1078 | 0.00417 | 97.6% |
| 0.5B | int8 per-tensor (control) | 17.043 | 0.4739 | 0.09894 | 88.8% |
| 0.5B | int8 wrong scale (control) | 34.719 | 3.3254 | 12.62769 | 0.0% |
| 1.5B | int8 per-channel | 4.262 | 0.0748 | 0.00182 | 98.0% |
| 1.5B | int8 per-tensor (control) | 23.922 | 0.3576 | 0.08093 | 92.3% |
| 1.5B | int8 wrong scale (control) | 37.797 | 4.4191 | 12.87346 | 0.0% |

**Perplexity** on a small fixed text set

| model | variant | perplexity | tokens |
| --- | --- | ---: | ---: |
| 0.5B | fp16 | 5.116 | 748 |
| 0.5B | int8 per-channel | 5.136 | 748 |
| 0.5B | int8 per-tensor (control) | 5.536 | 748 |
| 0.5B | int8 wrong scale (control) | 1777523.622 | 748 |
| 1.5B | fp16 | 4.465 | 748 |
| 1.5B | int8 per-channel | 4.463 | 748 |
| 1.5B | int8 per-tensor (control) | 4.836 | 748 |
| 1.5B | int8 wrong scale (control) | 2203935.623 | 748 |

**Greedy output against fp16 greedy output**

| model | variant | workload | identical outputs | mean matching prefix |
| --- | --- | --- | ---: | ---: |
| 0.5B | int8 per-channel | copy | 0 of 2 | 88% |
| 0.5B | int8 per-channel | code | 2 of 2 | 100% |
| 0.5B | int8 per-channel | open | 0 of 2 | 14% |
| 0.5B | int8 per-tensor (control) | copy | 1 of 2 | 84% |
| 0.5B | int8 per-tensor (control) | code | 0 of 2 | 40% |
| 0.5B | int8 per-tensor (control) | open | 0 of 2 | 2% |
| 0.5B | int8 wrong scale (control) | copy | 0 of 2 | 0% |
| 0.5B | int8 wrong scale (control) | code | 0 of 2 | 0% |
| 0.5B | int8 wrong scale (control) | open | 0 of 2 | 0% |
| 1.5B | int8 per-channel | copy | 1 of 2 | 60% |
| 1.5B | int8 per-channel | code | 2 of 2 | 100% |
| 1.5B | int8 per-channel | open | 0 of 2 | 62% |
| 1.5B | int8 per-tensor (control) | copy | 0 of 2 | 45% |
| 1.5B | int8 per-tensor (control) | code | 0 of 2 | 5% |
| 1.5B | int8 per-tensor (control) | open | 0 of 2 | 2% |
| 1.5B | int8 wrong scale (control) | copy | 0 of 2 | 0% |
| 1.5B | int8 wrong scale (control) | code | 0 of 2 | 0% |
| 1.5B | int8 wrong scale (control) | open | 0 of 2 | 0% |

<a id="stage8-memory"></a>

### Memory

Stored is the weight bytes the model holds. Read per token is what one batch-1 decode step has to read: every matrix once and one embedding row, so a tied embedding table kept for the lookup is stored but not read. Allocated is the GPU memory the loaded model took.

| model | weights | stored (GiB) | read per token (GiB) | GPU memory allocated (GiB) |
| --- | --- | ---: | ---: | ---: |
| 0.5B | fp16 | 0.920 | 0.920 | 0.926 |
| 0.5B | int8 | 0.715 | 0.461 | 0.716 |
| 1.5B | fp16 | 2.875 | 2.875 | 2.877 |
| 1.5B | int8 | 1.874 | 1.439 | 1.876 |
| 3B | fp16 | 5.748 | 5.748 | 5.846 |
| 3B | int8 | 3.456 | 2.876 | 3.459 |

<a id="stage8-speed"></a>

### Decode step time

Decode step time (one token for each sequence, argmax included) on top of 256 cached tokens, median of repeated steps with the spread. Floor is the bytes read per token over the measured copy bandwidth, the time a step could not beat if it did nothing but read its weights. Share is floor over time. Eager runs the ordinary forward pass, compiled runs it under torch.compile with CUDA graphs.

| model | weights | mode | batch | step (ms) | spread | floor (ms) | share of floor |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | fp16 | eager | 1 | 21.91 | 3% | 4.06 | 18.6% |
| 0.5B | fp16 | eager | 4 | 22.04 | 4% | 4.06 | 18.4% |
| 0.5B | fp16 | eager | 16 | 22.67 | 3% | 4.06 | 17.9% |
| 0.5B | fp16 | eager | 64 | 30.39 | 1% | 4.06 | 13.4% |
| 0.5B | fp16 | compiled | 1 | 10.83 | 3% | 4.06 | 37.5% |
| 0.5B | int8 | eager | 1 | 36.45 | 4% | 2.04 | 5.6% |
| 0.5B | int8 | eager | 4 | 36.39 | 3% | 2.04 | 5.6% |
| 0.5B | int8 | eager | 16 | 36.30 | 3% | 2.04 | 5.6% |
| 0.5B | int8 | eager | 64 | 43.15 | 1% | 2.04 | 4.7% |
| 0.5B | int8 | compiled | 1 | 14.48 | 4% | 2.04 | 14.1% |
| 0.5B | int8 intpack | eager | 1 | 29.62 | 3% | 2.04 | 6.9% |
| 0.5B | int8 intpack | eager | 4 | 47.67 | 0% | 2.04 | 4.3% |
| 0.5B | int8 intpack | eager | 16 | 158.82 | 0% | 2.04 | 1.3% |
| 0.5B | int8 intpack | eager | 64 | 596.79 | 1% | 2.04 | 0.3% |
| 1.5B | fp16 | eager | 1 | 27.25 | 4% | 12.70 | 46.6% |
| 1.5B | fp16 | eager | 4 | 27.44 | 2% | 12.70 | 46.3% |
| 1.5B | fp16 | eager | 16 | 29.76 | 1% | 12.70 | 42.7% |
| 1.5B | fp16 | eager | 64 | 59.64 | 1% | 12.70 | 21.3% |
| 1.5B | fp16 | compiled | 1 | 17.25 | 1% | 12.70 | 73.6% |
| 1.5B | int8 | eager | 1 | 54.90 | 1% | 6.36 | 11.6% |
| 1.5B | int8 | eager | 4 | 55.97 | 1% | 6.36 | 11.4% |
| 1.5B | int8 | eager | 16 | 65.28 | 1% | 6.36 | 9.7% |
| 1.5B | int8 | eager | 64 | 95.19 | 1% | 6.36 | 6.7% |
| 1.5B | int8 | compiled | 1 | 37.95 | 1% | 6.36 | 16.8% |
| 1.5B | int8 intpack | eager | 1 | 71.44 | 1% | 6.36 | 8.9% |
| 1.5B | int8 intpack | eager | 4 | 135.30 | 1% | 6.36 | 4.7% |
| 1.5B | int8 intpack | eager | 16 | 468.00 | 0% | 6.36 | 1.4% |
| 1.5B | int8 intpack | eager | 64 | 1839.96 | 2% | 6.36 | 0.3% |
| 3B | fp16 | eager | 1 | 33.99 | 3% | 25.39 | 74.7% |
| 3B | fp16 | eager | 4 | 36.42 | 2% | 25.39 | 69.7% |
| 3B | fp16 | eager | 16 | 48.96 | 1% | 25.39 | 51.8% |
| 3B | fp16 | eager | 64 | 98.35 | 0% | 25.39 | 25.8% |
| 3B | fp16 | compiled | 1 | 31.19 | 1% | 25.39 | 81.4% |
| 3B | int8 | eager | 1 | 100.81 | 0% | 12.70 | 12.6% |
| 3B | int8 | eager | 4 | 104.56 | 1% | 12.70 | 12.2% |
| 3B | int8 | eager | 16 | 118.21 | 1% | 12.70 | 10.7% |
| 3B | int8 | eager | 64 | 169.81 | 1% | 12.70 | 7.5% |
| 3B | int8 | compiled | 1 | 72.08 | 0% | 12.70 | 17.6% |
| 3B | int8 intpack | eager | 1 | 136.49 | 1% | 12.70 | 9.3% |
| 3B | int8 intpack | eager | 4 | 259.30 | 1% | 12.70 | 4.9% |
| 3B | int8 intpack | eager | 16 | 948.55 | 0% | 12.70 | 1.3% |
| 3B | int8 intpack | eager | 64 | 3586.79 | 1% | 12.70 | 0.4% |

Compile warm-up (s): 0.5B fp16 29, 0.5B int8 36, 1.5B fp16 34, 1.5B int8 42, 3B fp16 41, 3B int8 54.

<a id="stage8-draft"></a>

### A quantized draft

The Stage 7 cost ratio c, the draft step time over the target step time, with the 0.5B draft in fp16 and in int8. The target is the fp16 model of the first target size, in eager mode.

| draft weights | mode | draft step (ms) | target step (ms) | c |
| --- | --- | ---: | ---: | ---: |
| fp16 | eager | 22.38 | 26.91 | 0.83 |
| fp16 | graph | 9.23 | 26.91 | 0.34 |
| int8 | eager | 34.90 | 26.91 | 1.30 |
| int8 | graph | 20.95 | 26.91 | 0.78 |

**Speculative output against each target's own plain greedy output** (the first prompt of each workload, drafted by the 0.5B model). A checking pass feeds a block of tokens where a plain step feeds one, and in fp16 the two matmul shapes round differently, so an output may differ where the target's top two tokens are nearly tied. At the first difference the table gives the target's own logit gap between the two tokens (a small gap is a tie, a large one is a real discrepancy). The fp16 target is the control.

| target | workload | first differing token (0-based) | logit gap |
| --- | --- | ---: | ---: |
| fp16 | copy | none (identical) | n/a |
| fp16 | code | none (identical) | n/a |
| fp16 | open | none (identical) | n/a |
| int8 | copy | none (identical) | n/a |
| int8 | code | 2 | 0.008 |
| int8 | open | none (identical) | n/a |

fp16 target: 3 of 3 outputs identical to plain greedy.

int8 target: 2 of 3 outputs identical to plain greedy.

Every difference of the int8 target is a near-tie: yes.

<a id="stage8-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage8_accuracy.png` | Perplexity of fp16 and of int8 with one scale per channel and per matrix, by model (the wrong-scale control is off the chart) |
| `stage8_bandwidth_share.png` | Share of the bandwidth floor each decode step reaches (floor over step time), by model, weights and mode |
| `stage8_draft_cost.png` | Draft step cost ratio c with an fp16 and an int8 draft, eager and graph |
| `stage8_gate_bandwidth.png` | Effective bandwidth of each way against the measured copy bandwidth, by weight shape |
| `stage8_gate_speed.png` | Speed of each int8 way of computing a linear layer against the fp16 layer, by weight shape |
| `stage8_memory.png` | Weight bytes stored and read per decoded token, fp16 against int8, by model |
| `stage8_scale_error.png` | Median relative error of each weight row with one scale per channel and with one scale per matrix |
| `stage8_step_time.png` | Decode step time at batch 1, fp16 and int8, eager and compiled, by model |
| `stage8_step_vs_batch.png` | Eager decode step time against batch size, fp16 and int8, by model |

## Stage 9: A Fused Int8 Kernel and the Quantized Self-Draft

Source: [`results/stage9.json`](results/stage9.json)

<a id="stage9-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Triton:** 3.6.0
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B
- **Model revision:** Qwen2.5-1.5B 8faed761; Qwen2.5-0.5B 060db649; Qwen2.5-3B 3aab1f19
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage9-acceptance"></a>

### Acceptance of an int8 self-draft

Per-token acceptance `alpha` of three guessers against the fp16 target, with a draft length of 4, on the first prompts of each workload. The int8 self-draft is an int8 copy of the target itself, run through the slow eager dequantize in this section, so only `alpha` and the tokens per round are meaningful here, not any time.

| target | workload | guesser | alpha | tokens per round | outputs identical to plain |
| --- | --- | --- | ---: | ---: | ---: |
| 1.5B | copy | n-gram | 0.791 | 2.48 | 3 of 3 |
| 1.5B | copy | 0.5B draft | 0.903 | 4.01 | 3 of 3 |
| 1.5B | copy | int8 self-draft | 0.991 | 4.83 | 3 of 3 |
| 1.5B | code | n-gram | 0.703 | 2.18 | 3 of 3 |
| 1.5B | code | 0.5B draft | 0.923 | 4.13 | 3 of 3 |
| 1.5B | code | int8 self-draft | 0.991 | 4.91 | 3 of 3 |
| 1.5B | open | n-gram | 0.290 | 1.14 | 3 of 3 |
| 1.5B | open | 0.5B draft | 0.753 | 3.06 | 3 of 3 |
| 1.5B | open | int8 self-draft | 0.974 | 4.67 | 3 of 3 |
| 3B | copy | n-gram | 0.987 | 4.52 | 3 of 3 |
| 3B | copy | 0.5B draft | 0.983 | 4.75 | 3 of 3 |
| 3B | copy | int8 self-draft | 1.000 | 5.00 | 3 of 3 |
| 3B | code | n-gram | 0.644 | 1.95 | 3 of 3 |
| 3B | code | 0.5B draft | 0.865 | 3.85 | 3 of 3 |
| 3B | code | int8 self-draft | 0.982 | 4.67 | 3 of 3 |
| 3B | open | n-gram | 0.207 | 1.09 | 2 of 3 |
| 3B | open | 0.5B draft | 0.711 | 2.77 | 2 of 3 |
| 3B | open | int8 self-draft | 0.969 | 4.52 | 2 of 3 |

The int8 self-draft is accepted more often than the 0.5B draft on every workload and target: yes.

<a id="stage9-kernel"></a>

### The fused int8 kernel

The Triton kernel (`minivllm/int8_kernel.py`) with block sizes tuned for every distinct layer shape of the three model sizes. `gemv1` is the one-token kernel, `dot` the tensor-core kernel for a block of tokens. GB/s is the bytes of the stored int8 weight over the time of the best candidate; the share is against the measured copy bandwidth in the gate below. Candidates that did not compile or launch are counted as failed.

| layer | weight shape | rows of x | kernel | tile (output rows x K) | warps | time (ms) | GB/s | share of copy | candidates run / failed |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up | 4864 x 896 | 1 | gemv1 | 8 x 256 | 2 | 0.074 | 59 | 24% | 48 / 0 |
| mlp gate/up | 4864 x 896 | 5 | dot | 128 x 64 | 4 | 0.627 | 7 | 3% | 12 / 0 |
| mlp down | 896 x 4864 | 1 | gemv1 | 8 x 128 | 8 | 0.070 | 62 | 25% | 48 / 0 |
| mlp down | 896 x 4864 | 5 | dot | 64 x 64 | 4 | 0.498 | 9 | 4% | 12 / 0 |
| attn q/o | 896 x 896 | 1 | gemv1 | 8 x 512 | 8 | 0.065 | 12 | 5% | 48 / 0 |
| attn q/o | 896 x 896 | 5 | dot | 64 x 64 | 4 | 0.114 | 7 | 3% | 12 / 0 |
| attn k/v | 128 x 896 | 1 | gemv1 | 2 x 512 | 2 | 0.064 | 2 | 1% | 48 / 0 |
| attn k/v | 128 x 896 | 5 | dot | 32 x 64 | 4 | 0.103 | 1 | 0% | 12 / 0 |
| lm_head | 151936 x 896 | 1 | gemv1 | 16 x 512 | 4 | 0.575 | 237 | 98% | 48 / 0 |
| lm_head | 151936 x 896 | 5 | dot | 128 x 64 | 4 | 6.681 | 20 | 8% | 12 / 0 |
| mlp gate/up | 8960 x 1536 | 1 | gemv1 | 4 x 512 | 4 | 0.103 | 133 | 55% | 48 / 0 |
| mlp gate/up | 8960 x 1536 | 5 | dot | 128 x 64 | 4 | 0.763 | 18 | 7% | 12 / 0 |
| mlp down | 1536 x 8960 | 1 | gemv1 | 2 x 512 | 4 | 0.106 | 130 | 53% | 48 / 0 |
| mlp down | 1536 x 8960 | 5 | dot | 64 x 64 | 4 | 1.119 | 12 | 5% | 12 / 0 |
| attn q/o | 1536 x 1536 | 1 | gemv1 | 2 x 128 | 8 | 0.064 | 37 | 15% | 48 / 0 |
| attn q/o | 1536 x 1536 | 5 | dot | 128 x 64 | 4 | 0.258 | 9 | 4% | 12 / 0 |
| attn k/v | 256 x 1536 | 1 | gemv1 | 2 x 128 | 8 | 0.065 | 6 | 2% | 48 / 0 |
| attn k/v | 256 x 1536 | 5 | dot | 32 x 64 | 4 | 0.133 | 3 | 1% | 12 / 0 |
| lm_head | 151936 x 1536 | 1 | gemv1 | 16 x 512 | 4 | 0.949 | 246 | 101% | 48 / 0 |
| lm_head | 151936 x 1536 | 5 | dot | 128 x 64 | 4 | 11.845 | 20 | 8% | 12 / 0 |
| mlp gate/up | 11008 x 2048 | 1 | gemv1 | 16 x 512 | 4 | 0.138 | 164 | 67% | 48 / 0 |
| mlp gate/up | 11008 x 2048 | 5 | dot | 64 x 64 | 4 | 1.246 | 18 | 7% | 12 / 0 |
| mlp down | 2048 x 11008 | 1 | gemv1 | 2 x 512 | 2 | 0.142 | 159 | 65% | 48 / 0 |
| mlp down | 2048 x 11008 | 5 | dot | 64 x 64 | 4 | 1.404 | 16 | 7% | 12 / 0 |
| attn q/o | 2048 x 2048 | 1 | gemv1 | 8 x 256 | 8 | 0.065 | 65 | 27% | 48 / 0 |
| attn q/o | 2048 x 2048 | 5 | dot | 64 x 64 | 4 | 0.305 | 14 | 6% | 12 / 0 |
| attn k/v | 256 x 2048 | 1 | gemv1 | 8 x 256 | 4 | 0.065 | 8 | 3% | 48 / 0 |
| attn k/v | 256 x 2048 | 5 | dot | 32 x 64 | 4 | 0.161 | 3 | 1% | 12 / 0 |
| lm_head | 151936 x 2048 | 1 | gemv1 | 16 x 512 | 8 | 1.242 | 251 | 103% | 48 / 0 |
| lm_head | 151936 x 2048 | 5 | dot | 128 x 64 | 4 | 16.169 | 19 | 8% | 12 / 0 |

**Correctness** against the dequantize path at every shape and 1, 5, 16 and 37 rows of x (60 cases): the largest relative error is 4.9e-04, and with a wrong scale (each row given its neighbour's) the smallest relative error is 1.5e-01, so the check can fail.

<a id="stage9-gate"></a>

### The go/no-go gate: one linear layer

One linear layer y = x @ W.T at real Qwen2.5 weight shapes, timed as the median of repeated calls (spread is the interquartile range over the median), with the fp16 weight and with the int8 weight through each way of computing it. GB/s is the bytes of the stored weight over the time, against a measured copy bandwidth of 243 GB/s; vs fp16 is the fp16 time over this time, above one means faster than the fp16 layer. A way that does not work in this build is listed with the reason.

| layer | weight shape | batch | way | works | time (ms) | spread | GB/s | vs fp16 | reason it does not |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| mlp gate/up | 8960 x 1536 | 1 | fp | yes | 0.211 | 13% | 131 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 dequant, eager | yes | 0.503 | 11% | 27 | 0.42x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 dequant, compiled | yes | 0.458 | 5% | 30 | 0.46x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 intpack | yes | 1.675 | 5% | 8 | 0.13x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 triton | yes | 0.175 | 43% | 79 | 1.20x |  |
| mlp gate/up | 8960 x 1536 | 5 | fp | yes | 0.167 | 4% | 165 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 dequant, eager | yes | 0.665 | 3% | 21 | 0.25x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 dequant, compiled | yes | 0.447 | 4% | 31 | 0.37x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 intpack | yes | 3.889 | 4% | 4 | 0.04x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 triton | yes | 1.756 | 1% | 8 | 0.10x |  |
| mlp gate/up | 8960 x 1536 | 16 | fp | yes | 0.192 | 6% | 143 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 dequant, eager | yes | 0.663 | 4% | 21 | 0.29x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 dequant, compiled | yes | 0.474 | 13% | 29 | 0.41x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 intpack | yes | 7.912 | 0% | 2 | 0.02x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 triton | yes | 1.407 | 1% | 10 | 0.14x |  |
| mlp down | 1536 x 8960 | 1 | fp | yes | 0.148 | 12% | 186 | 1.00x |  |
| mlp down | 1536 x 8960 | 1 | int8 dequant, eager | yes | 0.439 | 26% | 31 | 0.34x |  |
| mlp down | 1536 x 8960 | 1 | int8 dequant, compiled | yes | 0.425 | 5% | 32 | 0.35x |  |
| mlp down | 1536 x 8960 | 1 | int8 intpack | yes | 0.608 | 3% | 23 | 0.24x |  |
| mlp down | 1536 x 8960 | 1 | int8 triton | yes | 0.119 | 7% | 116 | 1.24x |  |
| mlp down | 1536 x 8960 | 5 | fp | yes | 0.149 | 4% | 185 | 1.00x |  |
| mlp down | 1536 x 8960 | 5 | int8 dequant, eager | yes | 0.427 | 4% | 32 | 0.35x |  |
| mlp down | 1536 x 8960 | 5 | int8 dequant, compiled | yes | 0.423 | 5% | 33 | 0.35x |  |
| mlp down | 1536 x 8960 | 5 | int8 intpack | yes | 1.739 | 4% | 8 | 0.09x |  |
| mlp down | 1536 x 8960 | 5 | int8 triton | yes | 1.160 | 7% | 12 | 0.13x |  |
| mlp down | 1536 x 8960 | 16 | fp | yes | 0.147 | 5% | 188 | 1.00x |  |
| mlp down | 1536 x 8960 | 16 | int8 dequant, eager | yes | 0.425 | 1% | 32 | 0.35x |  |
| mlp down | 1536 x 8960 | 16 | int8 dequant, compiled | yes | 0.443 | 2% | 31 | 0.33x |  |
| mlp down | 1536 x 8960 | 16 | int8 intpack | yes | 7.412 | 0% | 2 | 0.02x |  |
| mlp down | 1536 x 8960 | 16 | int8 triton | yes | 1.897 | 1% | 7 | 0.08x |  |
| attn q/o | 1536 x 1536 | 1 | fp | yes | 0.047 | 3% | 100 | 1.00x |  |
| attn q/o | 1536 x 1536 | 1 | int8 dequant, eager | yes | 0.131 | 1% | 18 | 0.36x |  |
| attn q/o | 1536 x 1536 | 1 | int8 dequant, compiled | yes | 0.209 | 28% | 11 | 0.23x |  |
| attn q/o | 1536 x 1536 | 1 | int8 intpack | yes | 0.134 | 3% | 18 | 0.35x |  |
| attn q/o | 1536 x 1536 | 1 | int8 triton | yes | 0.080 | 25% | 29 | 0.59x |  |
| attn q/o | 1536 x 1536 | 5 | fp | yes | 0.048 | 7% | 97 | 1.00x |  |
| attn q/o | 1536 x 1536 | 5 | int8 dequant, eager | yes | 0.116 | 16% | 20 | 0.42x |  |
| attn q/o | 1536 x 1536 | 5 | int8 dequant, compiled | yes | 0.190 | 20% | 12 | 0.25x |  |
| attn q/o | 1536 x 1536 | 5 | int8 intpack | yes | 0.317 | 2% | 7 | 0.15x |  |
| attn q/o | 1536 x 1536 | 5 | int8 triton | yes | 0.253 | 5% | 9 | 0.19x |  |
| attn q/o | 1536 x 1536 | 16 | fp | yes | 0.049 | 7% | 97 | 1.00x |  |
| attn q/o | 1536 x 1536 | 16 | int8 dequant, eager | yes | 0.109 | 13% | 22 | 0.45x |  |
| attn q/o | 1536 x 1536 | 16 | int8 dequant, compiled | yes | 0.212 | 17% | 11 | 0.23x |  |
| attn q/o | 1536 x 1536 | 16 | int8 intpack | yes | 0.775 | 1% | 3 | 0.06x |  |
| attn q/o | 1536 x 1536 | 16 | int8 triton | yes | 0.258 | 7% | 9 | 0.19x |  |
| attn k/v | 256 x 1536 | 1 | fp | yes | 0.031 | 21% | 25 | 1.00x |  |
| attn k/v | 256 x 1536 | 1 | int8 dequant, eager | yes | 0.113 | 25% | 3 | 0.27x |  |
| attn k/v | 256 x 1536 | 1 | int8 dequant, compiled | yes | 0.209 | 13% | 2 | 0.15x |  |
| attn k/v | 256 x 1536 | 1 | int8 intpack | yes | 0.087 | 6% | 5 | 0.36x |  |
| attn k/v | 256 x 1536 | 1 | int8 triton | yes | 0.095 | 13% | 4 | 0.33x |  |
| attn k/v | 256 x 1536 | 5 | fp | yes | 0.038 | 5% | 20 | 1.00x |  |
| attn k/v | 256 x 1536 | 5 | int8 dequant, eager | yes | 0.122 | 21% | 3 | 0.32x |  |
| attn k/v | 256 x 1536 | 5 | int8 dequant, compiled | yes | 0.210 | 22% | 2 | 0.18x |  |
| attn k/v | 256 x 1536 | 5 | int8 intpack | yes | 0.117 | 8% | 3 | 0.33x |  |
| attn k/v | 256 x 1536 | 5 | int8 triton | yes | 0.147 | 7% | 3 | 0.26x |  |
| attn k/v | 256 x 1536 | 16 | fp | yes | 0.048 | 25% | 16 | 1.00x |  |
| attn k/v | 256 x 1536 | 16 | int8 dequant, eager | yes | 0.114 | 24% | 3 | 0.42x |  |
| attn k/v | 256 x 1536 | 16 | int8 dequant, compiled | yes | 0.217 | 14% | 2 | 0.22x |  |
| attn k/v | 256 x 1536 | 16 | int8 intpack | yes | 0.478 | 2% | 1 | 0.10x |  |
| attn k/v | 256 x 1536 | 16 | int8 triton | yes | 0.290 | 5% | 1 | 0.17x |  |
| lm_head | 151936 x 1536 | 1 | fp | yes | 2.528 | 6% | 185 | 1.00x |  |
| lm_head | 151936 x 1536 | 1 | int8 dequant, eager | yes | 8.450 | 12% | 28 | 0.30x |  |
| lm_head | 151936 x 1536 | 1 | int8 dequant, compiled | yes | 5.474 | 1% | 43 | 0.46x |  |
| lm_head | 151936 x 1536 | 1 | int8 intpack | yes | 8.687 | 5% | 27 | 0.29x |  |
| lm_head | 151936 x 1536 | 1 | int8 triton | yes | 0.956 | 1% | 244 | 2.64x |  |
| lm_head | 151936 x 1536 | 5 | fp | yes | 1.982 | 0% | 235 | 1.00x |  |
| lm_head | 151936 x 1536 | 5 | int8 dequant, eager | yes | 7.678 | 5% | 30 | 0.26x |  |
| lm_head | 151936 x 1536 | 5 | int8 dequant, compiled | yes | 5.121 | 3% | 46 | 0.39x |  |
| lm_head | 151936 x 1536 | 5 | int8 intpack | yes | 25.148 | 1% | 9 | 0.08x |  |
| lm_head | 151936 x 1536 | 5 | int8 triton | yes | 11.665 | 2% | 20 | 0.17x |  |
| lm_head | 151936 x 1536 | 16 | fp | yes | 2.041 | 1% | 229 | 1.00x |  |
| lm_head | 151936 x 1536 | 16 | int8 dequant, eager | yes | 7.983 | 5% | 29 | 0.26x |  |
| lm_head | 151936 x 1536 | 16 | int8 dequant, compiled | yes | 5.218 | 2% | 45 | 0.39x |  |
| lm_head | 151936 x 1536 | 16 | int8 intpack | yes | 66.401 | 0% | 4 | 0.03x |  |
| lm_head | 151936 x 1536 | 16 | int8 triton | yes | 12.309 | 3% | 19 | 0.17x |  |

<a id="stage9-step"></a>

### Decode step time

Decode step time (one token for each sequence, argmax included) on top of 256 cached tokens, median of repeated steps with the spread. Eager runs the ordinary forward pass, graph replays the single-token step captured in a CUDA graph (the draft's step; it has no spread), compiled runs it under `torch.compile`. Floor is the bytes read per token over the measured copy bandwidth, and share is floor over time.

| model | weights | mode | batch | step (ms) | spread | floor (ms) | share of floor |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | fp16 | eager | 1 | 22.25 | 7% | 4.07 | 18.3% |
| 0.5B | fp16 | eager | 4 | 22.62 | 5% | 4.07 | 18.0% |
| 0.5B | fp16 | eager | 16 | 24.59 | 25% | 4.07 | 16.5% |
| 0.5B | fp16 | graph | 1 | 9.35 | n/a | 4.07 | 43.5% |
| 0.5B | fp16 | compiled | 1 | 10.31 | 5% | 4.07 | 39.4% |
| 0.5B | int8 triton | eager | 1 | 32.33 | 2% | 2.04 | 6.3% |
| 0.5B | int8 triton | eager | 4 | 40.43 | 2% | 2.04 | 5.0% |
| 0.5B | int8 triton | eager | 16 | 40.22 | 1% | 2.04 | 5.1% |
| 0.5B | int8 triton | graph | 1 | 6.49 | n/a | 2.04 | 31.4% |
| 0.5B | int8 triton | compiled | 1 | 7.92 | 4% | 2.04 | 25.7% |
| 1.5B | fp16 | eager | 1 | 27.78 | 3% | 12.70 | 45.7% |
| 1.5B | fp16 | eager | 4 | 28.49 | 5% | 12.70 | 44.6% |
| 1.5B | fp16 | eager | 16 | 29.53 | 1% | 12.70 | 43.0% |
| 1.5B | fp16 | graph | 1 | 19.43 | n/a | 12.70 | 65.4% |
| 1.5B | fp16 | compiled | 1 | 18.09 | 4% | 12.70 | 70.2% |
| 1.5B | int8 triton | eager | 1 | 39.89 | 5% | 6.36 | 15.9% |
| 1.5B | int8 triton | eager | 4 | 101.01 | 1% | 6.36 | 6.3% |
| 1.5B | int8 triton | eager | 16 | 111.19 | 1% | 6.36 | 5.7% |
| 1.5B | int8 triton | graph | 1 | 12.72 | n/a | 6.36 | 50.0% |
| 1.5B | int8 triton | compiled | 1 | 11.59 | 8% | 6.36 | 54.8% |
| 3B | fp16 | eager | 1 | 36.72 | 3% | 25.39 | 69.2% |
| 3B | fp16 | eager | 4 | 38.49 | 3% | 25.39 | 66.0% |
| 3B | fp16 | eager | 16 | 50.86 | 1% | 25.39 | 49.9% |
| 3B | fp16 | graph | 1 | 35.04 | n/a | 25.39 | 72.5% |
| 3B | fp16 | compiled | 1 | 31.97 | 1% | 25.39 | 79.4% |
| 3B | int8 triton | eager | 1 | 49.82 | 3% | 12.71 | 25.5% |
| 3B | int8 triton | eager | 4 | 186.82 | 1% | 12.71 | 6.8% |
| 3B | int8 triton | eager | 16 | 204.87 | 1% | 12.71 | 6.2% |
| 3B | int8 triton | graph | 1 | 20.59 | n/a | 12.71 | 61.7% |
| 3B | int8 triton | compiled | 1 | 17.98 | 1% | 12.71 | 70.7% |

Compile warm-up (s): 0.5B fp16 30, 0.5B int8 triton 53, 1.5B fp16 36, 1.5B int8 triton 64, 3B fp16 43, 3B int8 triton 77.

**Accuracy**: the Triton path and the dequantize path against fp16 (all positions of the evaluation texts)

| model | variant | perplexity |
| --- | --- | ---: |
| 0.5B | fp16 | 5.1163 |
| 0.5B | int8 dequant | 5.1362 |
| 0.5B | int8 triton | 5.1402 |
| 1.5B | fp16 | 4.4649 |
| 1.5B | int8 dequant | 4.4632 |
| 1.5B | int8 triton | 4.4626 |

| model | variant | max abs diff | KL (nats) | same top-1 |
| --- | --- | ---: | ---: | ---: |
| 0.5B | int8 dequant | 6.547 | 0.00417 | 97.6% |
| 0.5B | int8 triton | 6.709 | 0.00420 | 98.0% |
| 1.5B | int8 dequant | 4.262 | 0.00182 | 98.0% |
| 1.5B | int8 triton | 4.248 | 0.00181 | 98.3% |

**The checking pass**: the time of one forward pass that feeds 1 or 5 tokens of one sequence, and the ratio to feeding 1 (the `v` of the cost model)

| model | weights | tokens fed | time (ms) | relative to 1 token |
| --- | --- | ---: | ---: | ---: |
| 0.5B | fp16 | 1 | 22.12 | 1.00x |
| 0.5B | fp16 | 5 | 25.03 | 1.13x |
| 0.5B | int8 triton | 1 | 33.70 | 1.00x |
| 0.5B | int8 triton | 5 | 37.26 | 1.11x |
| 1.5B | fp16 | 1 | 28.86 | 1.00x |
| 1.5B | fp16 | 5 | 30.39 | 1.05x |
| 1.5B | int8 triton | 1 | 38.58 | 1.00x |
| 1.5B | int8 triton | 5 | 90.81 | 2.35x |
| 3B | fp16 | 1 | 35.14 | 1.00x |
| 3B | fp16 | 5 | 39.81 | 1.13x |
| 3B | int8 triton | 1 | 49.22 | 1.00x |
| 3B | int8 triton | 5 | 174.12 | 3.54x |

<a id="stage9-speculation"></a>

### Speculation with the int8 model

**Cost ratio** c = the time of one guess step over the time of one plain step of the fp16 target (eager). The int8 self-draft is the int8 copy of the target, run with the Triton kernel; `graph` replays the captured step.

| target | guesser | mode | guess step (ms) | target step (ms) | c |
| --- | --- | --- | ---: | ---: | ---: |
| 1.5B | 0.5B draft | graph | 9.38 | 28.15 | 0.33 |
| 1.5B | int8 self-draft | graph | 12.36 | 28.15 | 0.44 |
| 1.5B | int8 self-draft | eager | 38.75 | 28.15 | 1.38 |
| 3B | 0.5B draft | graph | 9.29 | 35.63 | 0.26 |
| 3B | int8 self-draft | graph | 20.02 | 35.63 | 0.56 |
| 3B | int8 self-draft | eager | 50.30 | 35.63 | 1.41 |

**Speedup.** Every row is a setup (the weights of the target, then the guesser) on three workloads, at each draft length. `own plain` divides by the plain decoding of the same weights, `fp16 plain` by the plain decoding of the fp16 target, so the second is the one that tells whether int8 and speculation compose. Both baselines are the median of several timings. `predicted` is the cost model's speedup from the row's own acceptance, guess time and checking time.

| target | workload | setup | k | alpha | tokens per round | guess (ms) | check or step (ms) | own plain | fp16 plain | predicted | identical to plain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.5B | copy | fp16 plain | n/a | n/a | 1.00 | n/a | 27.7 | 1.00x | 1.00x | n/a | 3 of 3 |
| 1.5B | copy | fp16 n-gram | 4 | 0.79 | 2.48 | 0.0 | 29.2 | 2.30x | 2.30x | 3.13x | 3 of 3 |
| 1.5B | copy | fp16 n-gram | 6 | 0.80 | 2.71 | 0.0 | 30.7 | 2.38x | 2.38x | 3.56x | 3 of 3 |
| 1.5B | copy | fp16 0.5B draft (graph) | 4 | 0.90 | 4.01 | 51.6 | 31.5 | 1.32x | 1.32x | 1.37x | 3 of 3 |
| 1.5B | copy | fp16 0.5B draft (graph) | 6 | 0.90 | 5.00 | 68.7 | 31.9 | 1.36x | 1.36x | 1.44x | 3 of 3 |
| 1.5B | copy | fp16 int8 self-draft (graph) | 4 | 0.99 | 4.83 | 202.9 | 31.9 | 0.57x | 0.57x | 0.58x | 3 of 3 |
| 1.5B | copy | fp16 int8 self-draft (graph) | 6 | 0.99 | 6.63 | 307.9 | 31.8 | 0.54x | 0.54x | 0.56x | 3 of 3 |
| 1.5B | copy | fp16 int8 self-draft (eager) | 4 | 0.99 | 4.83 | 281.6 | 31.0 | 0.43x | 0.43x | 0.43x | 3 of 3 |
| 1.5B | copy | fp16 int8 self-draft (eager) | 6 | 0.99 | 6.63 | 431.0 | 30.4 | 0.40x | 0.40x | 0.41x | 3 of 3 |
| 1.5B | code | fp16 plain | n/a | n/a | 1.00 | n/a | 27.2 | 1.00x | 1.00x | n/a | 3 of 3 |
| 1.5B | code | fp16 n-gram | 4 | 0.70 | 2.18 | 0.0 | 28.9 | 2.00x | 2.00x | 2.62x | 3 of 3 |
| 1.5B | code | fp16 n-gram | 6 | 0.71 | 2.36 | 0.0 | 29.7 | 2.10x | 2.10x | 2.86x | 3 of 3 |
| 1.5B | code | fp16 0.5B draft (graph) | 4 | 0.92 | 4.13 | 50.2 | 30.1 | 1.38x | 1.38x | 1.45x | 3 of 3 |
| 1.5B | code | fp16 0.5B draft (graph) | 6 | 0.94 | 5.70 | 68.6 | 30.6 | 1.54x | 1.54x | 1.59x | 3 of 3 |
| 1.5B | code | fp16 int8 self-draft (graph) | 4 | 0.99 | 4.91 | 227.1 | 31.1 | 0.52x | 0.52x | 0.52x | 3 of 3 |
| 1.5B | code | fp16 int8 self-draft (graph) | 6 | 0.99 | 6.48 | 332.9 | 32.2 | 0.48x | 0.48x | 0.51x | 3 of 3 |
| 1.5B | code | fp16 int8 self-draft (eager) | 4 | 0.99 | 4.91 | 304.6 | 30.6 | 0.40x | 0.40x | 0.40x | 3 of 3 |
| 1.5B | code | fp16 int8 self-draft (eager) | 6 | 0.99 | 6.48 | 456.9 | 30.9 | 0.36x | 0.36x | 0.38x | 3 of 3 |
| 1.5B | open | fp16 plain | n/a | n/a | 1.00 | n/a | 27.5 | 1.00x | 1.00x | n/a | 3 of 3 |
| 1.5B | open | fp16 n-gram | 4 | 0.29 | 1.14 | 0.0 | 28.0 | 1.13x | 1.13x | 1.38x | 3 of 3 |
| 1.5B | open | fp16 n-gram | 6 | 0.30 | 1.15 | 0.0 | 28.9 | 1.10x | 1.10x | 1.36x | 2 of 3 |
| 1.5B | open | fp16 0.5B draft (graph) | 4 | 0.75 | 3.06 | 50.3 | 30.9 | 1.05x | 1.05x | 1.04x | 3 of 3 |
| 1.5B | open | fp16 0.5B draft (graph) | 6 | 0.76 | 3.52 | 67.5 | 31.3 | 0.99x | 0.99x | 1.00x | 3 of 3 |
| 1.5B | open | fp16 int8 self-draft (graph) | 4 | 0.97 | 4.67 | 120.2 | 31.2 | 0.86x | 0.86x | 0.86x | 3 of 3 |
| 1.5B | open | fp16 int8 self-draft (graph) | 6 | 0.98 | 6.20 | 143.7 | 31.6 | 0.98x | 0.98x | 1.02x | 3 of 3 |
| 1.5B | open | fp16 int8 self-draft (eager) | 4 | 0.97 | 4.67 | 198.6 | 30.7 | 0.57x | 0.57x | 0.57x | 3 of 3 |
| 1.5B | open | fp16 int8 self-draft (eager) | 6 | 0.98 | 6.20 | 269.2 | 30.8 | 0.58x | 0.58x | 0.60x | 3 of 3 |
| 1.5B | copy | int8 triton plain | n/a | n/a | 1.00 | n/a | 40.6 | 1.00x | 0.43x | n/a | 3 of 3 |
| 1.5B | copy | int8 triton n-gram | 4 | 0.79 | 2.52 | 0.0 | 85.4 | 1.09x | 0.47x | 1.56x | 3 of 3 |
| 1.5B | copy | int8 triton n-gram | 6 | 0.80 | 2.77 | 0.0 | 83.8 | 1.17x | 0.50x | 1.91x | 3 of 3 |
| 1.5B | code | int8 triton plain | n/a | n/a | 1.00 | n/a | 39.4 | 1.00x | 0.39x | n/a | 3 of 3 |
| 1.5B | code | int8 triton n-gram | 4 | 0.69 | 2.19 | 0.0 | 83.4 | 1.02x | 0.40x | 1.29x | 2 of 3 |
| 1.5B | code | int8 triton n-gram | 6 | 0.69 | 2.34 | 0.0 | 82.5 | 1.06x | 0.42x | 1.42x | 2 of 3 |
| 1.5B | open | int8 triton plain | n/a | n/a | 1.00 | n/a | 38.8 | 1.00x | 0.70x | n/a | 3 of 3 |
| 1.5B | open | int8 triton n-gram | 4 | 0.35 | 1.19 | 0.0 | 63.2 | 0.74x | 0.52x | 0.93x | 3 of 3 |
| 1.5B | open | int8 triton n-gram | 6 | 0.36 | 1.20 | 0.0 | 62.9 | 0.75x | 0.53x | 0.96x | 3 of 3 |
| 3B | copy | fp16 plain | n/a | n/a | 1.00 | n/a | 35.5 | 1.00x | 1.00x | n/a | 3 of 3 |
| 3B | copy | fp16 n-gram | 4 | 0.99 | 4.52 | 0.0 | 39.0 | 3.85x | 3.85x | 4.43x | 3 of 3 |
| 3B | copy | fp16 n-gram | 6 | 0.99 | 5.94 | 0.0 | 39.5 | 4.91x | 4.91x | 6.05x | 3 of 3 |
| 3B | copy | fp16 0.5B draft (graph) | 4 | 0.98 | 4.75 | 51.2 | 41.2 | 1.77x | 1.77x | 1.85x | 3 of 3 |
| 3B | copy | fp16 0.5B draft (graph) | 6 | 0.98 | 6.33 | 67.4 | 42.2 | 1.99x | 1.99x | 2.16x | 3 of 3 |
| 3B | copy | fp16 int8 self-draft (graph) | 4 | 1.00 | 5.00 | 405.5 | 42.1 | 0.39x | 0.39x | 0.40x | 3 of 3 |
| 3B | copy | fp16 int8 self-draft (graph) | 6 | 1.00 | 6.79 | 624.5 | 43.2 | 0.36x | 0.36x | 0.37x | 3 of 3 |
| 3B | copy | fp16 int8 self-draft (eager) | 4 | 1.00 | 5.00 | 499.3 | 41.3 | 0.33x | 0.33x | 0.33x | 3 of 3 |
| 3B | copy | fp16 int8 self-draft (eager) | 6 | 1.00 | 6.79 | 770.1 | 41.7 | 0.30x | 0.30x | 0.31x | 3 of 3 |
| 3B | code | fp16 plain | n/a | n/a | 1.00 | n/a | 34.9 | 1.00x | 1.00x | n/a | 3 of 3 |
| 3B | code | fp16 n-gram | 4 | 0.64 | 1.95 | 0.0 | 38.8 | 1.74x | 1.74x | 2.25x | 3 of 3 |
| 3B | code | fp16 n-gram | 6 | 0.65 | 2.08 | 0.0 | 38.8 | 1.85x | 1.85x | 2.44x | 3 of 3 |
| 3B | code | fp16 0.5B draft (graph) | 4 | 0.86 | 3.85 | 51.8 | 40.0 | 1.46x | 1.46x | 1.45x | 3 of 3 |
| 3B | code | fp16 0.5B draft (graph) | 6 | 0.87 | 4.67 | 70.5 | 41.0 | 1.46x | 1.46x | 1.49x | 3 of 3 |
| 3B | code | fp16 int8 self-draft (graph) | 4 | 0.98 | 4.67 | 430.0 | 40.5 | 0.35x | 0.35x | 0.36x | 3 of 3 |
| 3B | code | fp16 int8 self-draft (graph) | 6 | 0.98 | 6.48 | 665.1 | 41.3 | 0.33x | 0.33x | 0.33x | 3 of 3 |
| 3B | code | fp16 int8 self-draft (eager) | 4 | 0.98 | 4.67 | 516.0 | 40.0 | 0.30x | 0.30x | 0.30x | 3 of 3 |
| 3B | code | fp16 int8 self-draft (eager) | 6 | 0.98 | 6.48 | 809.0 | 40.2 | 0.27x | 0.27x | 0.27x | 3 of 3 |
| 3B | open | fp16 plain | n/a | n/a | 1.00 | n/a | 35.3 | 1.00x | 1.00x | n/a | 3 of 3 |
| 3B | open | fp16 n-gram | 4 | 0.21 | 1.09 | 0.0 | 36.8 | 1.02x | 1.02x | 1.21x | 2 of 3 |
| 3B | open | fp16 n-gram | 6 | 0.21 | 1.09 | 0.0 | 37.3 | 1.01x | 1.01x | 1.19x | 2 of 3 |
| 3B | open | fp16 0.5B draft (graph) | 4 | 0.71 | 2.77 | 49.8 | 39.6 | 1.07x | 1.07x | 1.12x | 2 of 3 |
| 3B | open | fp16 0.5B draft (graph) | 6 | 0.69 | 2.97 | 67.1 | 40.2 | 0.96x | 0.96x | 0.99x | 2 of 3 |
| 3B | open | fp16 int8 self-draft (graph) | 4 | 0.97 | 4.52 | 212.3 | 40.6 | 0.62x | 0.62x | 0.66x | 2 of 3 |
| 3B | open | fp16 int8 self-draft (graph) | 6 | 0.96 | 6.06 | 240.7 | 41.3 | 0.75x | 0.75x | 0.78x | 2 of 3 |
| 3B | open | fp16 int8 self-draft (eager) | 4 | 0.97 | 4.52 | 298.2 | 40.2 | 0.47x | 0.47x | 0.49x | 2 of 3 |
| 3B | open | fp16 int8 self-draft (eager) | 6 | 0.96 | 6.06 | 376.9 | 39.8 | 0.51x | 0.51x | 0.53x | 2 of 3 |
| 3B | copy | int8 triton plain | n/a | n/a | 1.00 | n/a | 50.4 | 1.00x | 0.34x | n/a | 3 of 3 |
| 3B | copy | int8 triton n-gram | 4 | 0.99 | 4.52 | 0.0 | 174.5 | 1.12x | 0.38x | 1.41x | 3 of 3 |
| 3B | copy | int8 triton n-gram | 6 | 0.99 | 5.94 | 0.0 | 171.8 | 1.25x | 0.43x | 1.98x | 3 of 3 |
| 3B | code | int8 triton plain | n/a | n/a | 1.00 | n/a | 50.1 | 1.00x | 0.31x | n/a | 3 of 3 |
| 3B | code | int8 triton n-gram | 4 | 0.62 | 1.85 | 0.0 | 136.3 | 0.83x | 0.26x | 0.87x | 2 of 3 |
| 3B | code | int8 triton n-gram | 6 | 0.62 | 1.91 | 0.0 | 134.2 | 0.85x | 0.27x | 0.94x | 2 of 3 |
| 3B | open | int8 triton plain | n/a | n/a | 1.00 | n/a | 50.9 | 1.00x | 0.67x | n/a | 3 of 3 |
| 3B | open | int8 triton n-gram | 4 | 0.22 | 1.11 | 0.0 | 102.3 | 0.55x | 0.37x | 0.64x | 1 of 3 |
| 3B | open | int8 triton n-gram | 6 | 0.23 | 1.11 | 0.0 | 100.4 | 0.57x | 0.38x | 0.66x | 1 of 3 |

**Exactness**, per setup against the same weights' own plain greedy output (first prompt of each workload, 32 tokens). At the first difference the table gives the target's own logit gap between the two tokens (a small gap is an fp16 tie).

| target | setup | workload | first differing token (0-based) | logit gap |
| --- | --- | --- | ---: | ---: |
| 1.5B | fp16 target, int8 self-draft | copy | none (identical) | n/a |
| 1.5B | fp16 target, int8 self-draft | code | none (identical) | n/a |
| 1.5B | fp16 target, int8 self-draft | open | none (identical) | n/a |
| 1.5B | int8 target, n-gram | copy | none (identical) | n/a |
| 1.5B | int8 target, n-gram | code | 2.0 | 0.008 |
| 1.5B | int8 target, n-gram | open | none (identical) | n/a |
| 3B | fp16 target, int8 self-draft | copy | none (identical) | n/a |
| 3B | fp16 target, int8 self-draft | code | none (identical) | n/a |
| 3B | fp16 target, int8 self-draft | open | 6.0 | 0.000 |
| 3B | int8 target, n-gram | copy | none (identical) | n/a |
| 3B | int8 target, n-gram | code | 28.0 | 0.000 |
| 3B | int8 target, n-gram | open | none (identical) | n/a |

Every difference is a near-tie: yes.

<a id="stage9-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage9_acceptance.png` | Per-token acceptance of the n-gram lookup, the 0.5B draft and the int8 self-draft against the fp16 target, by workload |
| `stage9_bandwidth_share.png` | Share of the bandwidth floor each decode step reaches (floor over step time), by model, weights and mode |
| `stage9_cost_ratio.png` | Cost ratio c of the 0.5B draft and the int8 self-draft against the fp16 target step, by target |
| `stage9_gate_bandwidth.png` | Effective bandwidth of each way, the Triton kernel included, against the measured copy bandwidth, by weight shape |
| `stage9_gate_speed.png` | Speed of each int8 way of computing a linear layer, the Triton kernel included, against the fp16 layer, by weight shape |
| `stage9_speedup_1p5b.png` | Best speedup over fp16 plain decoding by workload and setup, 1.5B target |
| `stage9_speedup_3b.png` | Best speedup over fp16 plain decoding by workload and setup, 3B target |
| `stage9_step_time.png` | Decode step time at one token, fp16 and int8 with the Triton kernel, eager and in a CUDA graph, by model |

## Stage 10: A Decode Loop in CUDA Graphs

Source: [`results/stage10.json`](results/stage10.json)

<a id="stage10-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Triton:** 3.6.0
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B
- **Model revision:** Qwen2.5-3B 3aab1f19; Qwen2.5-0.5B 060db649; Qwen2.5-1.5B 8faed761
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage10-kernel"></a>

### The multi-row kernel

The multi-row Triton kernel and the tuner (`minivllm/int8_kernel.py`): for every distinct layer shape of the three model sizes and each row bucket, the fastest candidate, timed as repeated calls inside one CUDA graph. `gemv1` reads each weight once for one row of x, `gemvm` once for up to 8 rows on the FMA units, `dot` is the tensor-core kernel, `dequant` converts the weights to fp16 and uses the fp16 matmul. The tile is output columns x K (and rows of x first for `dot`). GB/s is the bytes of the stored int8 weight over the time, and the share is against the measured copy bandwidth in the gate below. Candidates that did not compile or launch are counted as failed.

| layer | weight shape | rows of x | kernel | tile | warps | time (ms) | GB/s | share of copy | candidates run / failed |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up | 4864 x 896 | 1 | gemv1 | 16 x 512 | 8 | 0.015 | 283 | 116% | 36 / 0 |
| mlp gate/up | 4864 x 896 | 2 | gemvm | 4 x 128 | 2 | 0.040 | 109 | 45% | 28 / 0 |
| mlp gate/up | 4864 x 896 | 4 | gemvm | 4 x 128 | 2 | 0.059 | 75 | 31% | 28 / 0 |
| mlp gate/up | 4864 x 896 | 8 | gemvm | 4 x 128 | 2 | 0.082 | 53 | 22% | 50 / 2 |
| mlp gate/up | 4864 x 896 | 16 | dequant | n/a | n/a | 0.157 | 28 | 11% | 23 / 2 |
| mlp gate/up | 4864 x 896 | 128 | dequant | n/a | n/a | 0.181 | 24 | 10% | 23 / 2 |
| mlp down | 896 x 4864 | 1 | gemv1 | 8 x 128 | 8 | 0.014 | 307 | 126% | 36 / 0 |
| mlp down | 896 x 4864 | 2 | gemvm | 4 x 256 | 2 | 0.034 | 127 | 52% | 28 / 0 |
| mlp down | 896 x 4864 | 4 | gemvm | 4 x 128 | 2 | 0.046 | 94 | 39% | 28 / 0 |
| mlp down | 896 x 4864 | 8 | gemvm | 4 x 128 | 2 | 0.060 | 73 | 30% | 50 / 2 |
| mlp down | 896 x 4864 | 16 | dequant | n/a | n/a | 0.142 | 31 | 13% | 23 / 2 |
| mlp down | 896 x 4864 | 128 | dequant | n/a | n/a | 0.187 | 23 | 10% | 23 / 2 |
| attn q/o | 896 x 896 | 1 | gemv1 | 8 x 256 | 4 | 0.004 | 182 | 75% | 36 / 0 |
| attn q/o | 896 x 896 | 2 | gemvm | 4 x 128 | 2 | 0.007 | 109 | 45% | 28 / 0 |
| attn q/o | 896 x 896 | 4 | gemvm | 4 x 128 | 2 | 0.010 | 82 | 34% | 28 / 0 |
| attn q/o | 896 x 896 | 8 | gemvm | 4 x 128 | 2 | 0.014 | 58 | 24% | 50 / 2 |
| attn q/o | 896 x 896 | 16 | dequant | n/a | n/a | 0.025 | 33 | 13% | 23 / 2 |
| attn q/o | 896 x 896 | 128 | dequant | n/a | n/a | 0.042 | 19 | 8% | 23 / 2 |
| attn k/v | 128 x 896 | 1 | gemv1 | 8 x 512 | 8 | 0.003 | 35 | 14% | 36 / 0 |
| attn k/v | 128 x 896 | 2 | gemvm | 1 x 64 | 4 | 0.004 | 32 | 13% | 28 / 0 |
| attn k/v | 128 x 896 | 4 | gemvm | 1 x 64 | 8 | 0.005 | 25 | 10% | 28 / 0 |
| attn k/v | 128 x 896 | 8 | gemvm | 4 x 256 | 4 | 0.006 | 20 | 8% | 50 / 2 |
| attn k/v | 128 x 896 | 16 | dequant | n/a | n/a | 0.015 | 8 | 3% | 23 / 2 |
| attn k/v | 128 x 896 | 128 | dequant | n/a | n/a | 0.018 | 6 | 3% | 23 / 2 |
| lm_head | 151936 x 896 | 1 | gemv1 | 16 x 512 | 4 | 0.681 | 200 | 82% | 36 / 0 |
| lm_head | 151936 x 896 | 2 | gemvm | 4 x 256 | 2 | 1.562 | 87 | 36% | 28 / 0 |
| lm_head | 151936 x 896 | 4 | gemvm | 4 x 256 | 2 | 2.088 | 65 | 27% | 28 / 0 |
| lm_head | 151936 x 896 | 8 | gemvm | 4 x 256 | 2 | 2.836 | 48 | 20% | 50 / 2 |
| lm_head | 151936 x 896 | 16 | dequant | n/a | n/a | 5.132 | 27 | 11% | 23 / 2 |
| lm_head | 151936 x 896 | 128 | dequant | n/a | n/a | 6.850 | 20 | 8% | 23 / 2 |
| mlp gate/up | 8960 x 1536 | 1 | gemv1 | 16 x 512 | 4 | 0.061 | 226 | 93% | 36 / 0 |
| mlp gate/up | 8960 x 1536 | 2 | gemvm | 4 x 256 | 2 | 0.142 | 97 | 40% | 28 / 0 |
| mlp gate/up | 8960 x 1536 | 4 | gemvm | 4 x 64 | 2 | 0.194 | 71 | 29% | 28 / 0 |
| mlp gate/up | 8960 x 1536 | 8 | gemvm | 4 x 256 | 2 | 0.224 | 62 | 25% | 50 / 2 |
| mlp gate/up | 8960 x 1536 | 16 | dequant | n/a | n/a | 0.464 | 30 | 12% | 23 / 2 |
| mlp gate/up | 8960 x 1536 | 128 | dequant | n/a | n/a | 0.542 | 25 | 10% | 23 / 2 |
| mlp down | 1536 x 8960 | 1 | gemv1 | 16 x 512 | 8 | 0.061 | 227 | 93% | 36 / 0 |
| mlp down | 1536 x 8960 | 2 | gemvm | 4 x 256 | 4 | 0.114 | 121 | 50% | 28 / 0 |
| mlp down | 1536 x 8960 | 4 | gemvm | 4 x 256 | 4 | 0.144 | 95 | 39% | 28 / 0 |
| mlp down | 1536 x 8960 | 8 | gemvm | 4 x 256 | 4 | 0.196 | 70 | 29% | 50 / 2 |
| mlp down | 1536 x 8960 | 16 | dequant | n/a | n/a | 0.445 | 31 | 13% | 23 / 2 |
| mlp down | 1536 x 8960 | 128 | dequant | n/a | n/a | 0.551 | 25 | 10% | 23 / 2 |
| attn q/o | 1536 x 1536 | 1 | gemv1 | 8 x 256 | 4 | 0.007 | 353 | 145% | 36 / 0 |
| attn q/o | 1536 x 1536 | 2 | gemvm | 4 x 256 | 2 | 0.016 | 146 | 60% | 28 / 0 |
| attn q/o | 1536 x 1536 | 4 | gemvm | 4 x 256 | 2 | 0.023 | 102 | 42% | 28 / 0 |
| attn q/o | 1536 x 1536 | 8 | gemvm | 4 x 256 | 2 | 0.031 | 77 | 32% | 50 / 2 |
| attn q/o | 1536 x 1536 | 16 | dequant | n/a | n/a | 0.085 | 28 | 11% | 23 / 2 |
| attn q/o | 1536 x 1536 | 128 | dequant | n/a | n/a | 0.108 | 22 | 9% | 23 / 2 |
| attn k/v | 256 x 1536 | 1 | gemv1 | 8 x 512 | 8 | 0.004 | 96 | 40% | 36 / 0 |
| attn k/v | 256 x 1536 | 2 | gemvm | 4 x 64 | 8 | 0.005 | 75 | 31% | 28 / 0 |
| attn k/v | 256 x 1536 | 4 | gemvm | 2 x 256 | 2 | 0.007 | 60 | 25% | 28 / 0 |
| attn k/v | 256 x 1536 | 8 | gemvm | 4 x 256 | 2 | 0.009 | 45 | 19% | 50 / 2 |
| attn k/v | 256 x 1536 | 16 | dequant | n/a | n/a | 0.020 | 20 | 8% | 23 / 2 |
| attn k/v | 256 x 1536 | 128 | dequant | n/a | n/a | 0.033 | 12 | 5% | 23 / 2 |
| lm_head | 151936 x 1536 | 1 | gemv1 | 16 x 512 | 2 | 1.159 | 202 | 83% | 36 / 0 |
| lm_head | 151936 x 1536 | 2 | gemvm | 4 x 256 | 2 | 2.407 | 97 | 40% | 28 / 0 |
| lm_head | 151936 x 1536 | 4 | gemvm | 4 x 256 | 2 | 3.081 | 76 | 31% | 28 / 0 |
| lm_head | 151936 x 1536 | 8 | gemvm | 4 x 256 | 2 | 3.836 | 61 | 25% | 50 / 2 |
| lm_head | 151936 x 1536 | 16 | dequant | n/a | n/a | 8.563 | 27 | 11% | 23 / 2 |
| lm_head | 151936 x 1536 | 128 | dequant | n/a | n/a | 11.249 | 21 | 9% | 23 / 2 |
| mlp gate/up | 11008 x 2048 | 1 | gemv1 | 16 x 512 | 4 | 0.097 | 233 | 96% | 36 / 0 |
| mlp gate/up | 11008 x 2048 | 2 | gemvm | 4 x 256 | 4 | 0.230 | 98 | 40% | 28 / 0 |
| mlp gate/up | 11008 x 2048 | 4 | gemvm | 4 x 256 | 4 | 0.284 | 79 | 33% | 28 / 0 |
| mlp gate/up | 11008 x 2048 | 8 | gemvm | 4 x 256 | 2 | 0.368 | 61 | 25% | 50 / 2 |
| mlp gate/up | 11008 x 2048 | 16 | dequant | n/a | n/a | 0.776 | 29 | 12% | 23 / 2 |
| mlp gate/up | 11008 x 2048 | 128 | dequant | n/a | n/a | 1.019 | 22 | 9% | 23 / 2 |
| mlp down | 2048 x 11008 | 1 | gemv1 | 16 x 512 | 8 | 0.101 | 224 | 92% | 36 / 0 |
| mlp down | 2048 x 11008 | 2 | gemvm | 4 x 256 | 4 | 0.174 | 129 | 53% | 28 / 0 |
| mlp down | 2048 x 11008 | 4 | gemvm | 4 x 256 | 4 | 0.230 | 98 | 40% | 28 / 0 |
| mlp down | 2048 x 11008 | 8 | gemvm | 4 x 256 | 4 | 0.308 | 73 | 30% | 50 / 2 |
| mlp down | 2048 x 11008 | 16 | dequant | n/a | n/a | 0.804 | 28 | 12% | 23 / 2 |
| mlp down | 2048 x 11008 | 128 | dequant | n/a | n/a | 1.045 | 22 | 9% | 23 / 2 |
| attn q/o | 2048 x 2048 | 1 | gemv1 | 8 x 512 | 4 | 0.011 | 400 | 164% | 36 / 0 |
| attn q/o | 2048 x 2048 | 2 | gemvm | 4 x 256 | 2 | 0.029 | 143 | 59% | 28 / 0 |
| attn q/o | 2048 x 2048 | 4 | gemvm | 4 x 256 | 2 | 0.040 | 106 | 44% | 28 / 0 |
| attn q/o | 2048 x 2048 | 8 | gemvm | 4 x 128 | 2 | 0.054 | 78 | 32% | 50 / 2 |
| attn q/o | 2048 x 2048 | 16 | dequant | n/a | n/a | 0.150 | 28 | 11% | 23 / 2 |
| attn q/o | 2048 x 2048 | 128 | dequant | n/a | n/a | 0.163 | 26 | 11% | 23 / 2 |
| attn k/v | 256 x 2048 | 1 | gemv1 | 8 x 512 | 8 | 0.004 | 121 | 50% | 36 / 0 |
| attn k/v | 256 x 2048 | 2 | gemvm | 4 x 64 | 8 | 0.006 | 86 | 36% | 28 / 0 |
| attn k/v | 256 x 2048 | 4 | gemvm | 2 x 256 | 2 | 0.008 | 69 | 29% | 28 / 0 |
| attn k/v | 256 x 2048 | 8 | gemvm | 4 x 256 | 2 | 0.011 | 49 | 20% | 50 / 2 |
| attn k/v | 256 x 2048 | 16 | dequant | n/a | n/a | 0.022 | 24 | 10% | 23 / 2 |
| attn k/v | 256 x 2048 | 128 | dequant | n/a | n/a | 0.038 | 14 | 6% | 23 / 2 |
| lm_head | 151936 x 2048 | 1 | gemv1 | 16 x 512 | 2 | 1.466 | 212 | 87% | 36 / 0 |
| lm_head | 151936 x 2048 | 2 | gemvm | 4 x 256 | 2 | 3.134 | 99 | 41% | 28 / 0 |
| lm_head | 151936 x 2048 | 4 | gemvm | 4 x 256 | 2 | 4.022 | 77 | 32% | 28 / 0 |
| lm_head | 151936 x 2048 | 8 | gemvm | 4 x 256 | 2 | 4.862 | 64 | 26% | 50 / 2 |
| lm_head | 151936 x 2048 | 16 | dequant | n/a | n/a | 11.146 | 28 | 11% | 23 / 2 |
| lm_head | 151936 x 2048 | 128 | dequant | n/a | n/a | 14.641 | 21 | 9% | 23 / 2 |

**Correctness** against the dequantize path at every shape and 11 row counts (165 cases): the largest relative error is 7.8e-04, and with a wrong scale (each row given its neighbour's) the smallest relative error is 1.3e-01, so the check can fail. By rows of x:

| rows of x | kernels that ran | largest relative error | smallest error with a wrong scale |
| --- | --- | ---: | ---: |
| 1 | gemv1 | 4.3e-04 | 1.3e-01 |
| 2 | gemvm | 4.6e-04 | 1.7e-01 |
| 3 | gemvm | 4.6e-04 | 1.7e-01 |
| 4 | gemvm | 4.9e-04 | 1.6e-01 |
| 5 | gemvm | 5.0e-04 | 1.7e-01 |
| 6 | gemvm | 4.7e-04 | 2.2e-01 |
| 7 | gemvm | 5.2e-04 | 2.0e-01 |
| 8 | gemvm | 4.4e-04 | 2.3e-01 |
| 9 | dequant | 7.8e-04 | 2.0e-01 |
| 16 | dequant | 7.2e-04 | 2.2e-01 |
| 37 | dequant | 7.5e-04 | 2.4e-01 |

<a id="stage10-gate"></a>

### The go/no-go gate: one linear layer

**In a CUDA graph.** The large layers of the gate model, each way timed as 20 calls inside one graph and divided by 20, so launch cost is gone and what is left is what the kernels take (the way the decoder runs them). GB/s is the bytes of the stored weight over the time, against a measured copy bandwidth of 243 GB/s; vs fp16 is the fp16 layer's time over this time, above one means faster than the fp16 layer.

| layer | weight shape | rows of x | way | timed as | time (ms) | spread | GB/s | vs fp16 layer |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| mlp gate/up | 8960 x 1536 | 1 | fp16 layer | graph | 0.146 | 2% | 189 | 1.00x |
| mlp gate/up | 8960 x 1536 | 1 | int8 tuned kernel | graph | 0.071 | 0% | 194 | 2.06x |
| mlp gate/up | 8960 x 1536 | 1 | int8 tensor-core kernel | graph | 0.806 | 2% | 17 | 0.18x |
| mlp gate/up | 8960 x 1536 | 2 | fp16 layer | graph | 0.117 | 2% | 234 | 1.00x |
| mlp gate/up | 8960 x 1536 | 2 | int8 tuned kernel | graph | 0.144 | 2% | 96 | 0.82x |
| mlp gate/up | 8960 x 1536 | 2 | int8 tensor-core kernel | graph | 0.823 | 3% | 17 | 0.14x |
| mlp gate/up | 8960 x 1536 | 3 | fp16 layer | graph | 0.119 | 2% | 231 | 1.00x |
| mlp gate/up | 8960 x 1536 | 3 | int8 tuned kernel | graph | 0.261 | 0% | 53 | 0.46x |
| mlp gate/up | 8960 x 1536 | 3 | int8 tensor-core kernel | graph | 0.826 | 2% | 17 | 0.14x |
| mlp gate/up | 8960 x 1536 | 5 | fp16 layer | graph | 0.119 | 4% | 231 | 1.00x |
| mlp gate/up | 8960 x 1536 | 5 | int8 tuned kernel | graph | 0.221 | 10% | 62 | 0.54x |
| mlp gate/up | 8960 x 1536 | 5 | int8 tensor-core kernel | graph | 0.830 | 3% | 17 | 0.14x |
| mlp gate/up | 8960 x 1536 | 7 | fp16 layer | graph | 0.120 | 3% | 229 | 1.00x |
| mlp gate/up | 8960 x 1536 | 7 | int8 tuned kernel | graph | 0.244 | 16% | 56 | 0.49x |
| mlp gate/up | 8960 x 1536 | 7 | int8 tensor-core kernel | graph | 0.837 | 3% | 16 | 0.14x |
| mlp gate/up | 8960 x 1536 | 8 | fp16 layer | graph | 0.122 | 2% | 226 | 1.00x |
| mlp gate/up | 8960 x 1536 | 8 | int8 tuned kernel | graph | 0.235 | 11% | 59 | 0.52x |
| mlp gate/up | 8960 x 1536 | 8 | int8 tensor-core kernel | graph | 0.843 | 1% | 16 | 0.14x |
| mlp gate/up | 8960 x 1536 | 16 | fp16 layer | graph | 0.152 | 3% | 181 | 1.00x |
| mlp gate/up | 8960 x 1536 | 16 | int8 tuned kernel | graph | 0.532 | 6% | 26 | 0.29x |
| mlp gate/up | 8960 x 1536 | 16 | int8 tensor-core kernel | graph | 0.852 | 1% | 16 | 0.18x |
| mlp down | 1536 x 8960 | 1 | fp16 layer | graph | 0.113 | 2% | 244 | 1.00x |
| mlp down | 1536 x 8960 | 1 | int8 tuned kernel | graph | 0.066 | 1% | 208 | 1.70x |
| mlp down | 1536 x 8960 | 1 | int8 tensor-core kernel | graph | 1.040 | 1% | 13 | 0.11x |
| mlp down | 1536 x 8960 | 2 | fp16 layer | graph | 0.120 | 2% | 230 | 1.00x |
| mlp down | 1536 x 8960 | 2 | int8 tuned kernel | graph | 0.138 | 2% | 100 | 0.87x |
| mlp down | 1536 x 8960 | 2 | int8 tensor-core kernel | graph | 1.047 | 1% | 13 | 0.11x |
| mlp down | 1536 x 8960 | 3 | fp16 layer | graph | 0.120 | 1% | 229 | 1.00x |
| mlp down | 1536 x 8960 | 3 | int8 tuned kernel | graph | 0.165 | 2% | 84 | 0.73x |
| mlp down | 1536 x 8960 | 3 | int8 tensor-core kernel | graph | 1.048 | 2% | 13 | 0.11x |
| mlp down | 1536 x 8960 | 5 | fp16 layer | graph | 0.119 | 2% | 231 | 1.00x |
| mlp down | 1536 x 8960 | 5 | int8 tuned kernel | graph | 0.210 | 4% | 66 | 0.57x |
| mlp down | 1536 x 8960 | 5 | int8 tensor-core kernel | graph | 1.060 | 1% | 13 | 0.11x |
| mlp down | 1536 x 8960 | 7 | fp16 layer | graph | 0.119 | 2% | 231 | 1.00x |
| mlp down | 1536 x 8960 | 7 | int8 tuned kernel | graph | 0.226 | 14% | 61 | 0.53x |
| mlp down | 1536 x 8960 | 7 | int8 tensor-core kernel | graph | 1.060 | 1% | 13 | 0.11x |
| mlp down | 1536 x 8960 | 8 | fp16 layer | graph | 0.120 | 1% | 230 | 1.00x |
| mlp down | 1536 x 8960 | 8 | int8 tuned kernel | graph | 0.225 | 12% | 61 | 0.53x |
| mlp down | 1536 x 8960 | 8 | int8 tensor-core kernel | graph | 1.061 | 1% | 13 | 0.11x |
| mlp down | 1536 x 8960 | 16 | fp16 layer | graph | 0.121 | 3% | 227 | 1.00x |
| mlp down | 1536 x 8960 | 16 | int8 tuned kernel | graph | 0.507 | 5% | 27 | 0.24x |
| mlp down | 1536 x 8960 | 16 | int8 tensor-core kernel | graph | 1.073 | 1% | 13 | 0.11x |
| lm_head | 151936 x 1536 | 1 | fp16 layer | graph | 5.235 | 0% | 89 | 1.00x |
| lm_head | 151936 x 1536 | 1 | int8 tuned kernel | graph | 1.199 | 0% | 195 | 4.37x |
| lm_head | 151936 x 1536 | 1 | int8 tensor-core kernel | graph | 12.994 | 1% | 18 | 0.40x |
| lm_head | 151936 x 1536 | 2 | fp16 layer | graph | 2.706 | 3% | 172 | 1.00x |
| lm_head | 151936 x 1536 | 2 | int8 tuned kernel | graph | 2.406 | 0% | 97 | 1.12x |
| lm_head | 151936 x 1536 | 2 | int8 tensor-core kernel | graph | 13.053 | 1% | 18 | 0.21x |
| lm_head | 151936 x 1536 | 3 | fp16 layer | graph | 2.754 | 1% | 169 | 1.00x |
| lm_head | 151936 x 1536 | 3 | int8 tuned kernel | graph | 2.972 | 0% | 79 | 0.93x |
| lm_head | 151936 x 1536 | 3 | int8 tensor-core kernel | graph | 13.119 | 0% | 18 | 0.21x |
| lm_head | 151936 x 1536 | 5 | fp16 layer | graph | 2.800 | 0% | 167 | 1.00x |
| lm_head | 151936 x 1536 | 5 | int8 tuned kernel | graph | 3.613 | 0% | 65 | 0.78x |
| lm_head | 151936 x 1536 | 5 | int8 tensor-core kernel | graph | 13.210 | 0% | 18 | 0.21x |
| lm_head | 151936 x 1536 | 7 | fp16 layer | graph | 2.850 | 0% | 164 | 1.00x |
| lm_head | 151936 x 1536 | 7 | int8 tuned kernel | graph | 3.801 | 0% | 61 | 0.75x |
| lm_head | 151936 x 1536 | 7 | int8 tensor-core kernel | graph | 13.387 | 0% | 17 | 0.21x |
| lm_head | 151936 x 1536 | 8 | fp16 layer | graph | 2.856 | 2% | 163 | 1.00x |
| lm_head | 151936 x 1536 | 8 | int8 tuned kernel | graph | 3.833 | 0% | 61 | 0.75x |
| lm_head | 151936 x 1536 | 8 | int8 tensor-core kernel | graph | 13.384 | 0% | 17 | 0.21x |
| lm_head | 151936 x 1536 | 16 | fp16 layer | graph | 2.965 | 0% | 157 | 1.00x |
| lm_head | 151936 x 1536 | 16 | int8 tuned kernel | graph | 8.402 | 0% | 28 | 0.35x |
| lm_head | 151936 x 1536 | 16 | int8 tensor-core kernel | graph | 13.522 | 0% | 17 | 0.22x |

Gate verdict: **no-go**. At 5 rows the tuned int8 kernel is 0.63x the fp16 layer on the large layers (1 row 2.71x, 8 rows 0.60x), mean over the large layers.

**One call per timing** (the Stage 9 way, launch cost included), every way:

One linear layer y = x @ W.T at real Qwen2.5 weight shapes, timed as the median of repeated calls (spread is the interquartile range over the median), with the fp16 weight and with the int8 weight through each way of computing it. GB/s is the bytes of the stored weight over the time, against a measured copy bandwidth of 243 GB/s; vs fp16 is the fp16 time over this time, above one means faster than the fp16 layer. A way that does not work in this build is listed with the reason.

| layer | weight shape | batch | way | works | time (ms) | spread | GB/s | vs fp16 | reason it does not |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| mlp gate/up | 8960 x 1536 | 1 | fp | yes | 0.186 | 18% | 148 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 dequant, eager | yes | 0.476 | 1% | 29 | 0.39x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 intpack | yes | 0.703 | 5% | 20 | 0.26x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 triton | yes | 0.127 | 8% | 109 | 1.47x |  |
| mlp gate/up | 8960 x 1536 | 1 | int8 triton dot | yes | 0.905 | 1% | 15 | 0.21x |  |
| mlp gate/up | 8960 x 1536 | 2 | fp | yes | 0.176 | 5% | 156 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 2 | int8 dequant, eager | yes | 0.502 | 1% | 27 | 0.35x |  |
| mlp gate/up | 8960 x 1536 | 2 | int8 intpack | yes | 0.773 | 5% | 18 | 0.23x |  |
| mlp gate/up | 8960 x 1536 | 2 | int8 triton | yes | 0.145 | 10% | 95 | 1.22x |  |
| mlp gate/up | 8960 x 1536 | 2 | int8 triton dot | yes | 0.897 | 1% | 15 | 0.20x |  |
| mlp gate/up | 8960 x 1536 | 3 | fp | yes | 0.146 | 5% | 188 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 3 | int8 dequant, eager | yes | 0.448 | 1% | 31 | 0.33x |  |
| mlp gate/up | 8960 x 1536 | 3 | int8 intpack | yes | 1.307 | 2% | 11 | 0.11x |  |
| mlp gate/up | 8960 x 1536 | 3 | int8 triton | yes | 0.196 | 5% | 70 | 0.75x |  |
| mlp gate/up | 8960 x 1536 | 3 | int8 triton dot | yes | 0.863 | 1% | 16 | 0.17x |  |
| mlp gate/up | 8960 x 1536 | 5 | fp | yes | 0.144 | 3% | 191 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 dequant, eager | yes | 0.443 | 3% | 31 | 0.33x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 intpack | yes | 1.743 | 3% | 8 | 0.08x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 triton | yes | 0.197 | 4% | 70 | 0.73x |  |
| mlp gate/up | 8960 x 1536 | 5 | int8 triton dot | yes | 0.852 | 2% | 16 | 0.17x |  |
| mlp gate/up | 8960 x 1536 | 7 | fp | yes | 0.147 | 8% | 188 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 7 | int8 dequant, eager | yes | 0.440 | 1% | 31 | 0.33x |  |
| mlp gate/up | 8960 x 1536 | 7 | int8 intpack | yes | 2.365 | 5% | 6 | 0.06x |  |
| mlp gate/up | 8960 x 1536 | 7 | int8 triton | yes | 0.209 | 5% | 66 | 0.70x |  |
| mlp gate/up | 8960 x 1536 | 7 | int8 triton dot | yes | 0.846 | 1% | 16 | 0.17x |  |
| mlp gate/up | 8960 x 1536 | 8 | fp | yes | 0.145 | 4% | 190 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 8 | int8 dequant, eager | yes | 0.438 | 2% | 31 | 0.33x |  |
| mlp gate/up | 8960 x 1536 | 8 | int8 intpack | yes | 2.253 | 0% | 6 | 0.06x |  |
| mlp gate/up | 8960 x 1536 | 8 | int8 triton | yes | 0.200 | 7% | 69 | 0.72x |  |
| mlp gate/up | 8960 x 1536 | 8 | int8 triton dot | yes | 0.839 | 1% | 16 | 0.17x |  |
| mlp gate/up | 8960 x 1536 | 16 | fp | yes | 0.170 | 2% | 162 | 1.00x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 dequant, eager | yes | 0.466 | 2% | 30 | 0.36x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 intpack | yes | 4.425 | 11% | 3 | 0.04x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 triton | yes | 0.475 | 2% | 29 | 0.36x |  |
| mlp gate/up | 8960 x 1536 | 16 | int8 triton dot | yes | 0.800 | 1% | 17 | 0.21x |  |
| mlp down | 1536 x 8960 | 1 | fp | yes | 0.136 | 6% | 202 | 1.00x |  |
| mlp down | 1536 x 8960 | 1 | int8 dequant, eager | yes | 0.437 | 5% | 32 | 0.31x |  |
| mlp down | 1536 x 8960 | 1 | int8 intpack | yes | 0.789 | 1% | 17 | 0.17x |  |
| mlp down | 1536 x 8960 | 1 | int8 triton | yes | 0.112 | 4% | 123 | 1.22x |  |
| mlp down | 1536 x 8960 | 1 | int8 triton dot | yes | 1.398 | 11% | 10 | 0.10x |  |
| mlp down | 1536 x 8960 | 2 | fp | yes | 0.143 | 2% | 192 | 1.00x |  |
| mlp down | 1536 x 8960 | 2 | int8 dequant, eager | yes | 0.445 | 1% | 31 | 0.32x |  |
| mlp down | 1536 x 8960 | 2 | int8 intpack | yes | 0.689 | 2% | 20 | 0.21x |  |
| mlp down | 1536 x 8960 | 2 | int8 triton | yes | 0.133 | 7% | 104 | 1.08x |  |
| mlp down | 1536 x 8960 | 2 | int8 triton dot | yes | 1.229 | 1% | 11 | 0.12x |  |
| mlp down | 1536 x 8960 | 3 | fp | yes | 0.143 | 3% | 192 | 1.00x |  |
| mlp down | 1536 x 8960 | 3 | int8 dequant, eager | yes | 0.431 | 1% | 32 | 0.33x |  |
| mlp down | 1536 x 8960 | 3 | int8 intpack | yes | 1.282 | 2% | 11 | 0.11x |  |
| mlp down | 1536 x 8960 | 3 | int8 triton | yes | 0.145 | 7% | 95 | 0.99x |  |
| mlp down | 1536 x 8960 | 3 | int8 triton dot | yes | 1.230 | 1% | 11 | 0.12x |  |
| mlp down | 1536 x 8960 | 5 | fp | yes | 0.145 | 4% | 190 | 1.00x |  |
| mlp down | 1536 x 8960 | 5 | int8 dequant, eager | yes | 0.433 | 1% | 32 | 0.33x |  |
| mlp down | 1536 x 8960 | 5 | int8 intpack | yes | 1.855 | 3% | 7 | 0.08x |  |
| mlp down | 1536 x 8960 | 5 | int8 triton | yes | 0.177 | 9% | 78 | 0.82x |  |
| mlp down | 1536 x 8960 | 5 | int8 triton dot | yes | 1.191 | 2% | 12 | 0.12x |  |
| mlp down | 1536 x 8960 | 7 | fp | yes | 0.144 | 4% | 192 | 1.00x |  |
| mlp down | 1536 x 8960 | 7 | int8 dequant, eager | yes | 0.434 | 1% | 32 | 0.33x |  |
| mlp down | 1536 x 8960 | 7 | int8 intpack | yes | 2.390 | 3% | 6 | 0.06x |  |
| mlp down | 1536 x 8960 | 7 | int8 triton | yes | 0.170 | 7% | 81 | 0.84x |  |
| mlp down | 1536 x 8960 | 7 | int8 triton dot | yes | 1.171 | 1% | 12 | 0.12x |  |
| mlp down | 1536 x 8960 | 8 | fp | yes | 0.143 | 3% | 193 | 1.00x |  |
| mlp down | 1536 x 8960 | 8 | int8 dequant, eager | yes | 0.432 | 2% | 32 | 0.33x |  |
| mlp down | 1536 x 8960 | 8 | int8 intpack | yes | 2.430 | 2% | 6 | 0.06x |  |
| mlp down | 1536 x 8960 | 8 | int8 triton | yes | 0.178 | 16% | 77 | 0.80x |  |
| mlp down | 1536 x 8960 | 8 | int8 triton dot | yes | 1.168 | 3% | 12 | 0.12x |  |
| mlp down | 1536 x 8960 | 16 | fp | yes | 0.148 | 4% | 186 | 1.00x |  |
| mlp down | 1536 x 8960 | 16 | int8 dequant, eager | yes | 0.430 | 1% | 32 | 0.34x |  |
| mlp down | 1536 x 8960 | 16 | int8 intpack | yes | 4.519 | 7% | 3 | 0.03x |  |
| mlp down | 1536 x 8960 | 16 | int8 triton | yes | 0.449 | 1% | 31 | 0.33x |  |
| mlp down | 1536 x 8960 | 16 | int8 triton dot | yes | 1.204 | 6% | 11 | 0.12x |  |
| attn q/o | 1536 x 1536 | 1 | fp | yes | 0.048 | 8% | 99 | 1.00x |  |
| attn q/o | 1536 x 1536 | 1 | int8 dequant, eager | yes | 0.108 | 14% | 22 | 0.44x |  |
| attn q/o | 1536 x 1536 | 1 | int8 intpack | yes | 0.142 | 4% | 17 | 0.34x |  |
| attn q/o | 1536 x 1536 | 1 | int8 triton | yes | 0.075 | 16% | 32 | 0.64x |  |
| attn q/o | 1536 x 1536 | 1 | int8 triton dot | yes | 0.259 | 1% | 9 | 0.18x |  |
| attn q/o | 1536 x 1536 | 2 | fp | yes | 0.047 | 5% | 100 | 1.00x |  |
| attn q/o | 1536 x 1536 | 2 | int8 dequant, eager | yes | 0.108 | 6% | 22 | 0.44x |  |
| attn q/o | 1536 x 1536 | 2 | int8 intpack | yes | 0.145 | 3% | 16 | 0.33x |  |
| attn q/o | 1536 x 1536 | 2 | int8 triton | yes | 0.078 | 13% | 30 | 0.60x |  |
| attn q/o | 1536 x 1536 | 2 | int8 triton dot | yes | 0.256 | 2% | 9 | 0.18x |  |
| attn q/o | 1536 x 1536 | 3 | fp | yes | 0.047 | 3% | 100 | 1.00x |  |
| attn q/o | 1536 x 1536 | 3 | int8 dequant, eager | yes | 0.106 | 12% | 22 | 0.44x |  |
| attn q/o | 1536 x 1536 | 3 | int8 intpack | yes | 0.237 | 1% | 10 | 0.20x |  |
| attn q/o | 1536 x 1536 | 3 | int8 triton | yes | 0.080 | 11% | 30 | 0.59x |  |
| attn q/o | 1536 x 1536 | 3 | int8 triton dot | yes | 0.249 | 2% | 9 | 0.19x |  |
| attn q/o | 1536 x 1536 | 5 | fp | yes | 0.048 | 6% | 99 | 1.00x |  |
| attn q/o | 1536 x 1536 | 5 | int8 dequant, eager | yes | 0.108 | 8% | 22 | 0.44x |  |
| attn q/o | 1536 x 1536 | 5 | int8 intpack | yes | 0.330 | 2% | 7 | 0.15x |  |
| attn q/o | 1536 x 1536 | 5 | int8 triton | yes | 0.082 | 10% | 29 | 0.58x |  |
| attn q/o | 1536 x 1536 | 5 | int8 triton dot | yes | 0.244 | 4% | 10 | 0.20x |  |
| attn q/o | 1536 x 1536 | 7 | fp | yes | 0.047 | 7% | 100 | 1.00x |  |
| attn q/o | 1536 x 1536 | 7 | int8 dequant, eager | yes | 0.105 | 12% | 22 | 0.45x |  |
| attn q/o | 1536 x 1536 | 7 | int8 intpack | yes | 0.416 | 1% | 6 | 0.11x |  |
| attn q/o | 1536 x 1536 | 7 | int8 triton | yes | 0.082 | 8% | 29 | 0.57x |  |
| attn q/o | 1536 x 1536 | 7 | int8 triton dot | yes | 0.242 | 3% | 10 | 0.19x |  |
| attn q/o | 1536 x 1536 | 8 | fp | yes | 0.047 | 4% | 100 | 1.00x |  |
| attn q/o | 1536 x 1536 | 8 | int8 dequant, eager | yes | 0.107 | 15% | 22 | 0.44x |  |
| attn q/o | 1536 x 1536 | 8 | int8 intpack | yes | 0.418 | 1% | 6 | 0.11x |  |
| attn q/o | 1536 x 1536 | 8 | int8 triton | yes | 0.082 | 7% | 29 | 0.57x |  |
| attn q/o | 1536 x 1536 | 8 | int8 triton dot | yes | 0.247 | 6% | 10 | 0.19x |  |
| attn q/o | 1536 x 1536 | 16 | fp | yes | 0.047 | 7% | 99 | 1.00x |  |
| attn q/o | 1536 x 1536 | 16 | int8 dequant, eager | yes | 0.107 | 15% | 22 | 0.44x |  |
| attn q/o | 1536 x 1536 | 16 | int8 intpack | yes | 0.776 | 1% | 3 | 0.06x |  |
| attn q/o | 1536 x 1536 | 16 | int8 triton | yes | 0.140 | 12% | 17 | 0.34x |  |
| attn q/o | 1536 x 1536 | 16 | int8 triton dot | yes | 0.242 | 3% | 10 | 0.20x |  |
| attn k/v | 256 x 1536 | 1 | fp | yes | 0.034 | 12% | 23 | 1.00x |  |
| attn k/v | 256 x 1536 | 1 | int8 dequant, eager | yes | 0.101 | 12% | 4 | 0.34x |  |
| attn k/v | 256 x 1536 | 1 | int8 intpack | yes | 0.079 | 9% | 5 | 0.44x |  |
| attn k/v | 256 x 1536 | 1 | int8 triton | yes | 0.075 | 15% | 5 | 0.46x |  |
| attn k/v | 256 x 1536 | 1 | int8 triton dot | yes | 0.162 | 4% | 2 | 0.21x |  |
| attn k/v | 256 x 1536 | 2 | fp | yes | 0.037 | 9% | 21 | 1.00x |  |
| attn k/v | 256 x 1536 | 2 | int8 dequant, eager | yes | 0.106 | 16% | 4 | 0.35x |  |
| attn k/v | 256 x 1536 | 2 | int8 intpack | yes | 0.080 | 3% | 5 | 0.46x |  |
| attn k/v | 256 x 1536 | 2 | int8 triton | yes | 0.078 | 13% | 5 | 0.47x |  |
| attn k/v | 256 x 1536 | 2 | int8 triton dot | yes | 0.163 | 4% | 2 | 0.23x |  |
| attn k/v | 256 x 1536 | 3 | fp | yes | 0.036 | 4% | 22 | 1.00x |  |
| attn k/v | 256 x 1536 | 3 | int8 dequant, eager | yes | 0.105 | 4% | 4 | 0.34x |  |
| attn k/v | 256 x 1536 | 3 | int8 intpack | yes | 0.100 | 1% | 4 | 0.36x |  |
| attn k/v | 256 x 1536 | 3 | int8 triton | yes | 0.079 | 10% | 5 | 0.46x |  |
| attn k/v | 256 x 1536 | 3 | int8 triton dot | yes | 0.162 | 4% | 2 | 0.22x |  |
| attn k/v | 256 x 1536 | 5 | fp | yes | 0.043 | 5% | 18 | 1.00x |  |
| attn k/v | 256 x 1536 | 5 | int8 dequant, eager | yes | 0.111 | 16% | 4 | 0.38x |  |
| attn k/v | 256 x 1536 | 5 | int8 intpack | yes | 0.114 | 4% | 3 | 0.37x |  |
| attn k/v | 256 x 1536 | 5 | int8 triton | yes | 0.081 | 19% | 5 | 0.53x |  |
| attn k/v | 256 x 1536 | 5 | int8 triton dot | yes | 0.161 | 3% | 2 | 0.26x |  |
| attn k/v | 256 x 1536 | 7 | fp | yes | 0.041 | 9% | 19 | 1.00x |  |
| attn k/v | 256 x 1536 | 7 | int8 dequant, eager | yes | 0.117 | 14% | 3 | 0.35x |  |
| attn k/v | 256 x 1536 | 7 | int8 intpack | yes | 0.131 | 1% | 3 | 0.31x |  |
| attn k/v | 256 x 1536 | 7 | int8 triton | yes | 0.079 | 12% | 5 | 0.52x |  |
| attn k/v | 256 x 1536 | 7 | int8 triton dot | yes | 0.162 | 3% | 2 | 0.25x |  |
| attn k/v | 256 x 1536 | 8 | fp | yes | 0.041 | 7% | 19 | 1.00x |  |
| attn k/v | 256 x 1536 | 8 | int8 dequant, eager | yes | 0.112 | 10% | 4 | 0.36x |  |
| attn k/v | 256 x 1536 | 8 | int8 intpack | yes | 0.138 | 42% | 3 | 0.29x |  |
| attn k/v | 256 x 1536 | 8 | int8 triton | yes | 0.080 | 10% | 5 | 0.51x |  |
| attn k/v | 256 x 1536 | 8 | int8 triton dot | yes | 0.164 | 5% | 2 | 0.25x |  |
| attn k/v | 256 x 1536 | 16 | fp | yes | 0.041 | 18% | 19 | 1.00x |  |
| attn k/v | 256 x 1536 | 16 | int8 dequant, eager | yes | 0.132 | 10% | 3 | 0.31x |  |
| attn k/v | 256 x 1536 | 16 | int8 intpack | yes | 0.201 | 2% | 2 | 0.20x |  |
| attn k/v | 256 x 1536 | 16 | int8 triton | yes | 0.151 | 9% | 3 | 0.27x |  |
| attn k/v | 256 x 1536 | 16 | int8 triton dot | yes | 0.196 | 42% | 2 | 0.21x |  |
| lm_head | 151936 x 1536 | 1 | fp | yes | 2.319 | 51% | 201 | 1.00x |  |
| lm_head | 151936 x 1536 | 1 | int8 dequant, eager | yes | 9.327 | 16% | 25 | 0.25x |  |
| lm_head | 151936 x 1536 | 1 | int8 intpack | yes | 9.119 | 4% | 26 | 0.25x |  |
| lm_head | 151936 x 1536 | 1 | int8 triton | yes | 0.959 | 1% | 244 | 2.42x |  |
| lm_head | 151936 x 1536 | 1 | int8 triton dot | yes | 12.736 | 1% | 18 | 0.18x |  |
| lm_head | 151936 x 1536 | 2 | fp | yes | 1.960 | 1% | 238 | 1.00x |  |
| lm_head | 151936 x 1536 | 2 | int8 dequant, eager | yes | 8.378 | 5% | 28 | 0.23x |  |
| lm_head | 151936 x 1536 | 2 | int8 intpack | yes | 9.709 | 5% | 24 | 0.20x |  |
| lm_head | 151936 x 1536 | 2 | int8 triton | yes | 1.450 | 1% | 161 | 1.35x |  |
| lm_head | 151936 x 1536 | 2 | int8 triton dot | yes | 12.782 | 2% | 18 | 0.15x |  |
| lm_head | 151936 x 1536 | 3 | fp | yes | 1.962 | 1% | 238 | 1.00x |  |
| lm_head | 151936 x 1536 | 3 | int8 dequant, eager | yes | 8.465 | 5% | 28 | 0.23x |  |
| lm_head | 151936 x 1536 | 3 | int8 intpack | yes | 17.950 | 1% | 13 | 0.11x |  |
| lm_head | 151936 x 1536 | 3 | int8 triton | yes | 1.769 | 46% | 132 | 1.11x |  |
| lm_head | 151936 x 1536 | 3 | int8 triton dot | yes | 12.923 | 2% | 18 | 0.15x |  |
| lm_head | 151936 x 1536 | 5 | fp | yes | 1.976 | 0% | 236 | 1.00x |  |
| lm_head | 151936 x 1536 | 5 | int8 dequant, eager | yes | 8.506 | 5% | 27 | 0.23x |  |
| lm_head | 151936 x 1536 | 5 | int8 intpack | yes | 26.237 | 1% | 9 | 0.08x |  |
| lm_head | 151936 x 1536 | 5 | int8 triton | yes | 2.676 | 40% | 87 | 0.74x |  |
| lm_head | 151936 x 1536 | 5 | int8 triton dot | yes | 13.217 | 2% | 18 | 0.15x |  |
| lm_head | 151936 x 1536 | 7 | fp | yes | 1.993 | 0% | 234 | 1.00x |  |
| lm_head | 151936 x 1536 | 7 | int8 dequant, eager | yes | 8.624 | 5% | 27 | 0.23x |  |
| lm_head | 151936 x 1536 | 7 | int8 intpack | yes | 34.399 | 1% | 7 | 0.06x |  |
| lm_head | 151936 x 1536 | 7 | int8 triton | yes | 3.735 | 34% | 63 | 0.53x |  |
| lm_head | 151936 x 1536 | 7 | int8 triton dot | yes | 13.265 | 1% | 18 | 0.15x |  |
| lm_head | 151936 x 1536 | 8 | fp | yes | 1.996 | 0% | 234 | 1.00x |  |
| lm_head | 151936 x 1536 | 8 | int8 dequant, eager | yes | 8.608 | 4% | 27 | 0.23x |  |
| lm_head | 151936 x 1536 | 8 | int8 intpack | yes | 35.035 | 1% | 7 | 0.06x |  |
| lm_head | 151936 x 1536 | 8 | int8 triton | yes | 3.767 | 35% | 62 | 0.53x |  |
| lm_head | 151936 x 1536 | 8 | int8 triton dot | yes | 13.286 | 1% | 18 | 0.15x |  |
| lm_head | 151936 x 1536 | 16 | fp | yes | 2.042 | 1% | 229 | 1.00x |  |
| lm_head | 151936 x 1536 | 16 | int8 dequant, eager | yes | 8.849 | 3% | 26 | 0.23x |  |
| lm_head | 151936 x 1536 | 16 | int8 intpack | yes | 68.722 | 1% | 3 | 0.03x |  |
| lm_head | 151936 x 1536 | 16 | int8 triton | yes | 8.906 | 4% | 26 | 0.23x |  |
| lm_head | 151936 x 1536 | 16 | int8 triton dot | yes | 13.608 | 1% | 17 | 0.15x |  |

<a id="stage10-block"></a>

### A forward pass in a graph

A forward pass of several tokens of one sequence over the static KV cache, on top of 256 cached tokens, argmax on top, as the median of repeated passes with the spread. Graph replays the pass captured in a CUDA graph, eager runs the same pass launch by launch. Relative to one token is the `v` of the cost model for the same weights and mode. Floor is the bytes read per token over the measured copy bandwidth, and share is floor over time.

| model | weights | mode | tokens fed | time (ms) | spread | relative to 1 token | floor (ms) | share of floor |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | fp16 | graph | 1 | 8.99 | 2% | 1.00x | 4.06 | 45.2% |
| 0.5B | fp16 | graph | 5 | 9.07 | 1% | 1.01x | 4.06 | 44.8% |
| 0.5B | fp16 | graph | 7 | 9.12 | 1% | 1.01x | 4.06 | 44.6% |
| 0.5B | fp16 | eager | 1 | 22.66 | 2% | 1.00x | 4.06 | 17.9% |
| 0.5B | fp16 | eager | 5 | 23.38 | 2% | 1.03x | 4.06 | 17.4% |
| 0.5B | fp16 | eager | 7 | 23.22 | 2% | 1.02x | 4.06 | 17.5% |
| 0.5B | int8 | graph | 1 | 6.80 | 1% | 1.00x | 2.04 | 29.9% |
| 0.5B | int8 | graph | 5 | 11.60 | 1% | 1.71x | 2.04 | 17.5% |
| 0.5B | int8 | graph | 7 | 12.12 | 3% | 1.78x | 2.04 | 16.8% |
| 0.5B | int8 | eager | 1 | 32.40 | 2% | 1.00x | 2.04 | 6.3% |
| 0.5B | int8 | eager | 5 | 34.50 | 1% | 1.06x | 2.04 | 5.9% |
| 0.5B | int8 | eager | 7 | 35.43 | 2% | 1.09x | 2.04 | 5.7% |
| 1.5B | fp16 | graph | 1 | 19.63 | 2% | 1.00x | 12.70 | 64.7% |
| 1.5B | fp16 | graph | 5 | 19.90 | 0% | 1.01x | 12.70 | 63.8% |
| 1.5B | fp16 | graph | 7 | 20.20 | 0% | 1.03x | 12.70 | 62.9% |
| 1.5B | fp16 | eager | 1 | 27.42 | 3% | 1.00x | 12.70 | 46.3% |
| 1.5B | fp16 | eager | 5 | 28.00 | 9% | 1.02x | 12.70 | 45.4% |
| 1.5B | fp16 | eager | 7 | 28.48 | 2% | 1.04x | 12.70 | 44.6% |
| 1.5B | int8 | graph | 1 | 11.53 | 1% | 1.00x | 6.36 | 55.1% |
| 1.5B | int8 | graph | 5 | 26.66 | 2% | 2.31x | 6.36 | 23.8% |
| 1.5B | int8 | graph | 7 | 27.19 | 3% | 2.36x | 6.36 | 23.4% |
| 1.5B | int8 | eager | 1 | 38.39 | 2% | 1.00x | 6.36 | 16.6% |
| 1.5B | int8 | eager | 5 | 40.75 | 1% | 1.06x | 6.36 | 15.6% |
| 1.5B | int8 | eager | 7 | 41.14 | 2% | 1.07x | 6.36 | 15.4% |
| 3B | fp16 | graph | 1 | 36.30 | 1% | 1.00x | 25.38 | 69.9% |
| 3B | fp16 | graph | 5 | 36.36 | 0% | 1.00x | 25.38 | 69.8% |
| 3B | fp16 | graph | 7 | 36.94 | 1% | 1.02x | 25.38 | 68.7% |
| 3B | fp16 | eager | 1 | 38.13 | 3% | 1.00x | 25.38 | 66.6% |
| 3B | fp16 | eager | 5 | 37.18 | 2% | 0.98x | 25.38 | 68.3% |
| 3B | fp16 | eager | 7 | 37.68 | 2% | 0.99x | 25.38 | 67.4% |
| 3B | int8 | graph | 1 | 19.84 | 1% | 1.00x | 12.70 | 64.0% |
| 3B | int8 | graph | 5 | 48.67 | 0% | 2.45x | 12.70 | 26.1% |
| 3B | int8 | graph | 7 | 51.12 | 0% | 2.58x | 12.70 | 24.8% |
| 3B | int8 | eager | 1 | 48.94 | 3% | 1.00x | 12.70 | 26.0% |
| 3B | int8 | eager | 5 | 53.57 | 1% | 1.09x | 12.70 | 23.7% |
| 3B | int8 | eager | 7 | 54.61 | 11% | 1.12x | 12.70 | 23.3% |

**Accuracy of the multi-row kernel inside a model** (Qwen/Qwen2.5-1.5B): the logits of the block after the same prompt through the Triton path, the dequantize path and fp16.

| tokens in the block | compared | max abs diff | mean abs diff | KL (nats) | same top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 | int8 triton vs int8 dequant | 0.0342 | 0.00446 | 0.000020 | 100.0% |
| 5 | int8 dequant vs fp16 | 1.0664 | 0.08043 | 0.001140 | 100.0% |
| 5 | int8 triton vs fp16 | 1.0962 | 0.08180 | 0.001140 | 100.0% |
| 7 | int8 triton vs int8 dequant | 0.0342 | 0.00495 | 0.000020 | 100.0% |
| 7 | int8 dequant vs fp16 | 1.0533 | 0.07874 | 0.001230 | 85.7% |
| 7 | int8 triton vs fp16 | 1.0826 | 0.07981 | 0.001270 | 85.7% |
| 2 | int8 triton vs int8 dequant | 0.0430 | 0.00613 | 0.000020 | 100.0% |
| 2 | int8 dequant vs fp16 | 1.0601 | 0.10609 | 0.001080 | 100.0% |
| 2 | int8 triton vs fp16 | 1.1030 | 0.10893 | 0.001120 | 100.0% |
| 8 | int8 triton vs int8 dequant | 0.0352 | 0.00493 | 0.000020 | 100.0% |
| 8 | int8 dequant vs fp16 | 1.6562 | 0.09690 | 0.001220 | 87.5% |
| 8 | int8 triton vs fp16 | 1.6445 | 0.09775 | 0.001200 | 87.5% |

<a id="stage10-plain"></a>

### Plain decoding in a graph

Plain greedy decoding of 96 tokens, batch 1, median of 5 runs. The decode rate is the tokens after the first over the sum of the per-token times (prefill excluded), the total rate includes the prefill. Eager is the engine of the earlier stages, graph is `GraphDecoder.plain`. The output of each run is compared with the eager output of the same weights; a difference must be an fp16 tie, and the table gives the target's own logit gap there.

| model | weights | mode | decode (tok/s) | spread | whole generation (tok/s) | per token (ms) | output |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.5B | fp16 | eager | 45.7 | 9% | 45.6 | 21.89 | identical |
| 0.5B | fp16 | graph | 110.7 | 0% | 108.6 | 9.03 | identical |
| 0.5B | int8 | eager | 31.5 | 3% | 31.4 | 31.75 | identical |
| 0.5B | int8 | graph | 167.6 | 0% | 158.0 | 5.97 | identical |
| 1.5B | fp16 | eager | 37.6 | 6% | 37.6 | 26.60 | identical |
| 1.5B | fp16 | graph | 46.9 | 5% | 46.7 | 21.30 | identical |
| 1.5B | int8 | eager | 26.8 | 2% | 26.6 | 37.38 | identical |
| 1.5B | int8 | graph | 87.8 | 1% | 84.8 | 11.39 | identical |
| 3B | fp16 | eager | 18.5 | 44% | 18.5 | 53.99 | identical |
| 3B | fp16 | graph | 25.0 | 8% | 25.0 | 39.98 | identical |
| 3B | int8 | eager | 21.0 | 1% | 20.8 | 47.56 | identical |
| 3B | int8 | graph | 50.4 | 1% | 48.2 | 19.86 | identical |

**What each change buys** (ratios of the decode rates above):

| model | fp16 graph / fp16 eager | int8 eager / fp16 eager | int8 graph / fp16 graph | int8 graph / fp16 eager |
| --- | ---: | ---: | ---: | ---: |
| 0.5B | 2.42x | 0.69x | 1.51x | 3.67x |
| 1.5B | 1.25x | 0.71x | 1.87x | 2.33x |
| 3B | 1.35x | 1.14x | 2.01x | 2.72x |

Every graph output equals the eager output of the same weights or differs at a near-tie: yes.

<a id="stage10-speculation"></a>

### Speculation in the graph

**Speedup.** Each row is a setup (the weights, the engine, then the guesser: the n-gram lookup, which costs nothing) on three workloads at each draft length. Engine `eager` is the engine of the earlier stages, `graph` is `GraphDecoder`. Every speedup is a ratio of tokens per second of whole generations (prefill included) against the plain decoding of fp16 in the eager engine, of fp16 in a graph and of int8 in a graph, each the median of several timings. `v` is the mean time of a round over the mean plain step of the same weights and engine (rounds with no draft are one-token passes and count), and `predicted` is the cost model's speedup over the row's own plain decoding from its acceptance, `v` and guess time.

| target | workload | setup | k | alpha | tokens per round | rounds with a draft | v | check or step (ms) | vs fp16 eager | vs fp16 graph | vs int8 graph | predicted | identical to plain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.5B | copy | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 27.8 | 1.00x | 0.73x | 0.44x | n/a | 3 of 3 |
| 1.5B | copy | fp16 eager n-gram | 4 | 0.79 | 2.48 | 72% | 1.05 | 29.1 | 2.27x | 1.67x | 1.01x | 3.15x | 3 of 3 |
| 1.5B | copy | fp16 eager n-gram | 6 | 0.80 | 2.71 | 70% | 1.04 | 29.1 | 2.49x | 1.83x | 1.11x | 3.78x | 3 of 3 |
| 1.5B | code | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 26.8 | 1.00x | 0.75x | 0.45x | n/a | 3 of 3 |
| 1.5B | code | fp16 eager n-gram | 4 | 0.70 | 2.18 | 68% | 1.15 | 30.8 | 1.86x | 1.39x | 0.84x | 2.43x | 3 of 3 |
| 1.5B | code | fp16 eager n-gram | 6 | 0.71 | 2.36 | 66% | 1.11 | 29.7 | 2.08x | 1.55x | 0.94x | 2.82x | 3 of 3 |
| 1.5B | open | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 26.7 | 1.00x | 0.76x | 0.45x | n/a | 3 of 3 |
| 1.5B | open | fp16 eager n-gram | 4 | 0.29 | 1.14 | 37% | 1.05 | 27.9 | 1.09x | 0.82x | 0.49x | 1.34x | 3 of 3 |
| 1.5B | open | fp16 eager n-gram | 6 | 0.30 | 1.15 | 37% | 1.05 | 27.9 | 1.10x | 0.83x | 0.49x | 1.36x | 2 of 3 |
| 1.5B | copy | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.8 | 1.36x | 1.00x | 0.61x | n/a | 3 of 3 |
| 1.5B | copy | fp16 graph n-gram | 4 | 0.79 | 2.48 | 72% | 1.02 | 20.2 | 3.24x | 2.37x | 1.44x | 3.22x | 3 of 3 |
| 1.5B | copy | fp16 graph n-gram | 6 | 0.80 | 2.71 | 70% | 1.03 | 20.3 | 3.52x | 2.58x | 1.56x | 3.86x | 3 of 3 |
| 1.5B | code | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.9 | 1.34x | 1.00x | 0.61x | n/a | 3 of 3 |
| 1.5B | code | fp16 graph n-gram | 4 | 0.72 | 2.18 | 69% | 1.01 | 20.1 | 2.82x | 2.10x | 1.27x | 2.83x | 3 of 3 |
| 1.5B | code | fp16 graph n-gram | 6 | 0.71 | 2.36 | 67% | 1.02 | 20.3 | 3.01x | 2.24x | 1.36x | 3.09x | 3 of 3 |
| 1.5B | open | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 20.1 | 1.32x | 1.00x | 0.59x | n/a | 3 of 3 |
| 1.5B | open | fp16 graph n-gram | 4 | 0.30 | 1.14 | 38% | 1.00 | 20.0 | 1.51x | 1.14x | 0.67x | 1.42x | 3 of 3 |
| 1.5B | open | fp16 graph n-gram | 6 | 0.31 | 1.15 | 37% | 1.00 | 20.1 | 1.52x | 1.15x | 0.68x | 1.44x | 2 of 3 |
| 1.5B | copy | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.6 | 2.25x | 1.65x | 1.00x | n/a | 3 of 3 |
| 1.5B | copy | int8 graph n-gram | 4 | 0.79 | 2.52 | 74% | 1.98 | 22.9 | 2.82x | 2.07x | 1.25x | 1.66x | 3 of 3 |
| 1.5B | copy | int8 graph n-gram | 6 | 0.80 | 2.77 | 72% | 2.00 | 23.1 | 3.05x | 2.23x | 1.35x | 1.97x | 3 of 3 |
| 1.5B | code | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.6 | 2.22x | 1.65x | 1.00x | n/a | 3 of 3 |
| 1.5B | code | int8 graph n-gram | 4 | 0.71 | 2.13 | 66% | 1.85 | 21.4 | 2.52x | 1.88x | 1.14x | 1.54x | 2 of 3 |
| 1.5B | code | int8 graph n-gram | 6 | 0.71 | 2.28 | 64% | 1.87 | 21.6 | 2.65x | 1.98x | 1.20x | 1.66x | 2 of 3 |
| 1.5B | open | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.5 | 2.24x | 1.69x | 1.00x | n/a | 3 of 3 |
| 1.5B | open | int8 graph n-gram | 4 | 0.36 | 1.19 | 40% | 1.50 | 17.3 | 1.78x | 1.35x | 0.80x | 1.04x | 2 of 3 |
| 1.5B | open | int8 graph n-gram | 6 | 0.38 | 1.20 | 39% | 1.53 | 17.6 | 1.77x | 1.34x | 0.79x | 1.06x | 2 of 3 |
| 3B | copy | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 49.3 | 1.00x | 0.83x | 0.43x | n/a | 3 of 3 |
| 3B | copy | fp16 eager n-gram | 4 | 0.99 | 4.52 | 92% | 0.78 | 38.3 | 5.41x | 4.51x | 2.31x | 6.27x | 3 of 3 |
| 3B | copy | fp16 eager n-gram | 6 | 0.99 | 5.94 | 90% | 0.77 | 38.1 | 7.03x | 5.86x | 3.00x | 8.72x | 3 of 3 |
| 3B | code | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 41.8 | 1.00x | 0.97x | 0.50x | n/a | 3 of 3 |
| 3B | code | fp16 eager n-gram | 4 | 0.64 | 1.95 | 67% | 0.88 | 36.9 | 2.16x | 2.09x | 1.07x | 2.82x | 3 of 3 |
| 3B | code | fp16 eager n-gram | 6 | 0.65 | 2.08 | 65% | 0.90 | 37.7 | 2.25x | 2.18x | 1.12x | 3.01x | 3 of 3 |
| 3B | open | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 47.1 | 1.00x | 0.89x | 0.44x | n/a | 3 of 3 |
| 3B | open | fp16 eager n-gram | 4 | 0.21 | 1.09 | 34% | 0.80 | 37.8 | 1.34x | 1.19x | 0.59x | 1.57x | 2 of 3 |
| 3B | open | fp16 eager n-gram | 6 | 0.21 | 1.09 | 34% | 0.81 | 38.3 | 1.33x | 1.18x | 0.58x | 1.55x | 2 of 3 |
| 3B | copy | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 40.4 | 1.20x | 1.00x | 0.51x | n/a | 3 of 3 |
| 3B | copy | fp16 graph n-gram | 4 | 0.99 | 4.52 | 92% | 0.91 | 37.0 | 5.62x | 4.68x | 2.40x | 5.32x | 3 of 3 |
| 3B | copy | fp16 graph n-gram | 6 | 0.98 | 5.94 | 90% | 0.93 | 37.5 | 7.16x | 5.97x | 3.05x | 7.19x | 3 of 3 |
| 3B | code | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 40.7 | 1.03x | 1.00x | 0.51x | n/a | 3 of 3 |
| 3B | code | fp16 graph n-gram | 4 | 0.65 | 1.95 | 67% | 0.91 | 37.0 | 2.16x | 2.09x | 1.07x | 2.76x | 3 of 3 |
| 3B | code | fp16 graph n-gram | 6 | 0.65 | 2.08 | 65% | 0.92 | 37.2 | 2.28x | 2.21x | 1.13x | 2.95x | 3 of 3 |
| 3B | open | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 41.7 | 1.12x | 1.00x | 0.49x | n/a | 3 of 3 |
| 3B | open | fp16 graph n-gram | 4 | 0.21 | 1.09 | 34% | 0.90 | 37.6 | 1.35x | 1.20x | 0.59x | 1.40x | 2 of 3 |
| 3B | open | fp16 graph n-gram | 6 | 0.21 | 1.09 | 34% | 0.91 | 37.8 | 1.35x | 1.20x | 0.59x | 1.39x | 2 of 3 |
| 3B | copy | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.7 | 2.34x | 1.95x | 1.00x | n/a | 3 of 3 |
| 3B | copy | int8 graph n-gram | 4 | 0.99 | 4.52 | 92% | 2.34 | 46.3 | 4.27x | 3.56x | 1.82x | 2.08x | 3 of 3 |
| 3B | copy | int8 graph n-gram | 6 | 0.98 | 5.94 | 90% | 2.41 | 47.5 | 5.28x | 4.40x | 2.25x | 2.77x | 3 of 3 |
| 3B | code | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.7 | 2.02x | 1.95x | 1.00x | n/a | 3 of 3 |
| 3B | code | int8 graph n-gram | 4 | 0.62 | 1.85 | 64% | 1.93 | 37.9 | 1.94x | 1.88x | 0.96x | 1.25x | 2 of 3 |
| 3B | code | int8 graph n-gram | 6 | 0.62 | 1.91 | 63% | 1.98 | 38.9 | 1.96x | 1.89x | 0.97x | 1.29x | 2 of 3 |
| 3B | open | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.7 | 2.28x | 2.03x | 1.00x | n/a | 3 of 3 |
| 3B | open | int8 graph n-gram | 4 | 0.26 | 1.13 | 38% | 1.55 | 30.6 | 1.67x | 1.49x | 0.73x | 0.87x | 1 of 3 |
| 3B | open | int8 graph n-gram | 6 | 0.27 | 1.14 | 38% | 1.59 | 31.3 | 1.65x | 1.47x | 0.73x | 0.86x | 1 of 3 |

**Do int8 and speculation multiply?** All in the graph decoder, against fp16 plain decoding in a graph. `int8 alone` is the gain of int8 plain, `n-gram alone` the gain of the lookup on fp16, `product` their product, `both` what int8 plus the lookup measured, `both / product` how much of the product survived, and `v` the checking cost of each weight format. The last column is int8 plus the lookup against fp16 plain decoding in the eager engine.

| target | workload | k | int8 alone | n-gram alone (fp16) | n-gram alone (int8) | product | both | both / product | v fp16 | v int8 | vs fp16 eager plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | copy | 4 | 1.65x | 2.37x | 1.25x | 3.92x | 2.07x | 53% | 1.02 | 1.98 | 2.82x |
| 1.5B | copy | 6 | 1.65x | 2.58x | 1.35x | 4.25x | 2.23x | 53% | 1.03 | 2.00 | 3.05x |
| 1.5B | code | 4 | 1.65x | 2.10x | 1.14x | 3.47x | 1.88x | 54% | 1.01 | 1.85 | 2.52x |
| 1.5B | code | 6 | 1.65x | 2.24x | 1.20x | 3.71x | 1.98x | 53% | 1.02 | 1.87 | 2.65x |
| 1.5B | open | 4 | 1.69x | 1.14x | 0.80x | 1.93x | 1.35x | 70% | 1.00 | 1.50 | 1.78x |
| 1.5B | open | 6 | 1.69x | 1.15x | 0.79x | 1.95x | 1.34x | 69% | 1.00 | 1.53 | 1.77x |
| 3B | copy | 4 | 1.95x | 4.68x | 1.82x | 9.15x | 3.56x | 39% | 0.91 | 2.34 | 4.27x |
| 3B | copy | 6 | 1.95x | 5.97x | 2.25x | 11.66x | 4.40x | 38% | 0.93 | 2.41 | 5.28x |
| 3B | code | 4 | 1.95x | 2.09x | 0.96x | 4.08x | 1.88x | 46% | 0.91 | 1.93 | 1.94x |
| 3B | code | 6 | 1.95x | 2.21x | 0.97x | 4.30x | 1.89x | 44% | 0.92 | 1.98 | 1.96x |
| 3B | open | 4 | 2.03x | 1.20x | 0.73x | 2.44x | 1.49x | 61% | 0.90 | 1.55 | 1.67x |
| 3B | open | 6 | 2.03x | 1.20x | 0.73x | 2.42x | 1.47x | 61% | 0.91 | 1.59 | 1.65x |

**Exactness**, per setup against the same weights' own plain greedy output (first prompt of each workload, 32 tokens). At the first difference the table gives the target's own logit gap between the two tokens (a small gap is an fp16 tie). The fp16 eager row is the control: near-ties appear without any graph or int8.

| target | setup | workload | first differing token (0-based) | logit gap |
| --- | --- | --- | ---: | ---: |
| 1.5B | fp16 eager n-gram (control) | copy | none (identical) | n/a |
| 1.5B | fp16 eager n-gram (control) | code | none (identical) | n/a |
| 1.5B | fp16 eager n-gram (control) | open | none (identical) | n/a |
| 1.5B | fp16 graph n-gram | copy | none (identical) | n/a |
| 1.5B | fp16 graph n-gram | code | none (identical) | n/a |
| 1.5B | fp16 graph n-gram | open | none (identical) | n/a |
| 1.5B | int8 graph n-gram | copy | none (identical) | n/a |
| 1.5B | int8 graph n-gram | code | 2.0 | 0.008 |
| 1.5B | int8 graph n-gram | open | none (identical) | n/a |
| 1.5B | int8 graph plain | copy | none (identical) | n/a |
| 1.5B | int8 graph plain | code | none (identical) | n/a |
| 1.5B | int8 graph plain | open | none (identical) | n/a |
| 3B | fp16 eager n-gram (control) | copy | none (identical) | n/a |
| 3B | fp16 eager n-gram (control) | code | none (identical) | n/a |
| 3B | fp16 eager n-gram (control) | open | none (identical) | n/a |
| 3B | fp16 graph n-gram | copy | none (identical) | n/a |
| 3B | fp16 graph n-gram | code | none (identical) | n/a |
| 3B | fp16 graph n-gram | open | none (identical) | n/a |
| 3B | int8 graph n-gram | copy | none (identical) | n/a |
| 3B | int8 graph n-gram | code | 28.0 | 0.000 |
| 3B | int8 graph n-gram | open | none (identical) | n/a |
| 3B | int8 graph plain | copy | none (identical) | n/a |
| 3B | int8 graph plain | code | none (identical) | n/a |
| 3B | int8 graph plain | open | none (identical) | n/a |

Every difference is a near-tie: yes.

**A control that must fail**: the same decoder with a mask that lets every row see all keys up to the end of its block (so the rollback and the causal mask are both wrong).

| target | workload | output differs from plain greedy |
| --- | --- | --- |
| 1.5B | copy | yes |
| 1.5B | code | yes |
| 1.5B | open | yes |

The control changes the output (so the exactness check can fail): yes.

<a id="stage10-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage10_block_cost.png` | Cost of a checking pass of several tokens relative to a one-token pass in a CUDA graph, fp16 and int8, by model |
| `stage10_layer_eager.png` | The same comparison with one call per timing (launch cost included), with the dequantize path and the built-in int8 matmul, by rows of x |
| `stage10_layer_graph.png` | Speed of the tuned int8 kernel and the tensor-core kernel against the fp16 layer on the large layers, timed in a CUDA graph, by rows of x |
| `stage10_multiply.png` | Gain of int8 plus the n-gram lookup against the product of the two separate gains, by target and workload |
| `stage10_plain_rate.png` | Decode rate of plain greedy decoding, fp16 and int8, eager and in a graph, by model |
| `stage10_speedup_1p5b.png` | Best speedup over fp16 eager plain decoding by workload and setup, 1.5B target |
| `stage10_speedup_3b.png` | Best speedup over fp16 eager plain decoding by workload and setup, 3B target |

## Stage 11: Row-Exact Multi-Row Int8 Kernels

Source: [`results/stage11.json`](results/stage11.json)

<a id="stage11-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Triton:** 3.6.0
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B
- **Model revision:** Qwen2.5-0.5B 060db649; Qwen2.5-1.5B 8faed761; Qwen2.5-3B 3aab1f19
- **max_pos (RoPE table):** 4096
- **Seed / quick run:** 0 / False

<a id="stage11-gpu"></a>

### GPU state

The state of the GPU at the start and end of each timed section, from `nvidia-smi`. A section that starts with a warm-up ran the GPU under load first, so its clock is the steady one. Throttle reasons are the driver's bit mask (empty or zero when nothing holds the clock down).

| where | warm-up (s) | SM clock (MHz) | max clock (MHz) | temperature (C) | power (W) | power limit (W) | throttle reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| before tuning | 20 | 1170 | 1590 | 70 | 49 | 70 | 0x0000000000000004 |
| after tuning | 0 | 1185 | 1590 | 76 | 66 | 70 | 0x0000000000000004 |
| before the layer gate | 20 | 1005 | 1590 | 81 | 64 | 70 | 0x0000000000000004 |
| after the layer gate | 0 | 1425 | 1590 | 75 | 43 | 70 | 0x0000000000000000 |
| before the block passes | 20 | 1140 | 1590 | 79 | 65 | 70 | 0x0000000000000004 |
| after the block passes | 0 | 1590 | 1590 | 75 | 51 | 70 | 0x0000000000000000 |
| before plain decoding | 20 | 1320 | 1590 | 74 | 66 | 70 | 0x0000000000000004 |
| before the speculation sweeps | 20 | 1125 | 1590 | 76 | 66 | 70 | 0x0000000000000004 |
| after the speculation sweeps | 0 | 1590 | 1590 | 76 | 51 | 70 | 0x0000000000000000 |

At the clock under load and the measured copy bandwidth, 12.3 fused multiply-adds fit in the time one int8 weight takes to arrive (40 multiprocessors of 64 FP32 lanes).

<a id="stage11-kernel"></a>

### The row-exact kernels, the tuner and the compiler

The tuner (`minivllm/int8_kernel.py`, `bench.metrics.tune_int8_kernel`): for every distinct layer shape of the three model sizes and each row count, the fastest candidate, timed as repeated calls inside one CUDA graph, and the best time of each kernel family. `gemv1` reads each weight once for one row of x, `gemvm` is Stage 10's multi-row kernel, `gemvr` and `gemvs` are the row-exact kernels of this stage, `dequant` converts the weights to fp16 and uses the fp16 matmul. The tile is output columns x K. GB/s is the bytes of the stored int8 weight over the time, and the share is against the measured copy bandwidth. Tuning took 18 minutes. Candidates that did not compile or launch are counted as failed.

| layer | weight shape | rows of x | kernel | tile | warps | time (ms) | GB/s | share of copy | gemvm (ms) | gemvr (ms) | gemvs (ms) | candidates run / failed |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up | 4864 x 896 | 1 | gemv1 | 8 x 128 | 2 | 0.021 | 210 | 86% | n/a | n/a | n/a | 36 / 0 |
| mlp gate/up | 4864 x 896 | 2 | gemvs | 32 x 64 | 1 | 0.047 | 92 | 38% | 0.070 | 0.048 | 0.047 | 76 / 0 |
| mlp gate/up | 4864 x 896 | 3 | gemvs | 32 x 64 | 1 | 0.053 | 82 | 34% | 0.096 | 0.059 | 0.053 | 76 / 0 |
| mlp gate/up | 4864 x 896 | 4 | gemvr | 2 x 128 | 1 | 0.046 | 95 | 39% | 0.051 | 0.046 | 0.059 | 68 / 0 |
| mlp gate/up | 4864 x 896 | 5 | gemvs | 32 x 64 | 1 | 0.067 | 65 | 27% | 0.132 | 0.143 | 0.067 | 68 / 0 |
| mlp gate/up | 4864 x 896 | 6 | gemvr | 2 x 128 | 1 | 0.064 | 68 | 28% | 0.073 | 0.064 | 0.078 | 62 / 0 |
| mlp gate/up | 4864 x 896 | 7 | gemvr | 2 x 128 | 1 | 0.074 | 59 | 24% | 0.079 | 0.074 | 0.086 | 62 / 0 |
| mlp gate/up | 4864 x 896 | 8 | gemvm | 4 x 128 | 2 | 0.076 | 57 | 24% | 0.076 | 0.098 | 0.094 | 53 / 0 |
| mlp down | 896 x 4864 | 1 | gemv1 | 16 x 512 | 8 | 0.019 | 225 | 93% | n/a | n/a | n/a | 36 / 0 |
| mlp down | 896 x 4864 | 2 | gemvr | 1 x 256 | 1 | 0.028 | 156 | 64% | 0.029 | 0.028 | 0.063 | 76 / 0 |
| mlp down | 896 x 4864 | 3 | gemvr | 2 x 128 | 1 | 0.038 | 115 | 47% | 0.040 | 0.038 | 0.074 | 76 / 0 |
| mlp down | 896 x 4864 | 4 | gemvm | 4 x 128 | 2 | 0.043 | 102 | 42% | 0.043 | 0.046 | 0.081 | 68 / 0 |
| mlp down | 896 x 4864 | 5 | gemvm | 4 x 128 | 2 | 0.053 | 83 | 34% | 0.053 | 0.056 | 0.092 | 68 / 0 |
| mlp down | 896 x 4864 | 6 | gemvm | 4 x 128 | 2 | 0.057 | 76 | 31% | 0.057 | 0.070 | 0.098 | 62 / 0 |
| mlp down | 896 x 4864 | 7 | gemvm | 4 x 128 | 2 | 0.056 | 78 | 32% | 0.056 | 0.076 | 0.122 | 62 / 0 |
| mlp down | 896 x 4864 | 8 | gemvm | 4 x 128 | 2 | 0.057 | 76 | 31% | 0.057 | 0.084 | 0.121 | 53 / 0 |
| attn q/o | 896 x 896 | 1 | gemv1 | 8 x 128 | 8 | 0.004 | 198 | 81% | n/a | n/a | n/a | 36 / 0 |
| attn q/o | 896 x 896 | 2 | gemvm | 4 x 128 | 2 | 0.007 | 110 | 45% | 0.007 | 0.009 | 0.011 | 76 / 0 |
| attn q/o | 896 x 896 | 3 | gemvm | 4 x 128 | 2 | 0.010 | 82 | 34% | 0.010 | 0.011 | 0.013 | 76 / 0 |
| attn q/o | 896 x 896 | 4 | gemvm | 4 x 128 | 2 | 0.010 | 82 | 34% | 0.010 | 0.013 | 0.015 | 68 / 0 |
| attn q/o | 896 x 896 | 5 | gemvm | 4 x 128 | 2 | 0.013 | 60 | 25% | 0.013 | 0.015 | 0.016 | 68 / 0 |
| attn q/o | 896 x 896 | 6 | gemvm | 4 x 128 | 2 | 0.013 | 61 | 25% | 0.013 | 0.017 | 0.017 | 62 / 0 |
| attn q/o | 896 x 896 | 7 | gemvm | 4 x 128 | 2 | 0.013 | 60 | 25% | 0.013 | 0.019 | 0.019 | 62 / 0 |
| attn q/o | 896 x 896 | 8 | gemvm | 4 x 128 | 2 | 0.014 | 58 | 24% | 0.014 | 0.022 | 0.021 | 53 / 0 |
| attn k/v | 128 x 896 | 1 | gemv1 | 4 x 512 | 4 | 0.003 | 36 | 15% | n/a | n/a | n/a | 36 / 0 |
| attn k/v | 128 x 896 | 2 | gemvm | 1 x 64 | 4 | 0.004 | 32 | 13% | 0.004 | 0.006 | 0.010 | 76 / 0 |
| attn k/v | 128 x 896 | 3 | gemvm | 1 x 64 | 8 | 0.004 | 26 | 11% | 0.004 | 0.007 | 0.012 | 76 / 0 |
| attn k/v | 128 x 896 | 4 | gemvm | 1 x 64 | 8 | 0.004 | 26 | 11% | 0.004 | 0.007 | 0.013 | 68 / 0 |
| attn k/v | 128 x 896 | 5 | gemvm | 4 x 256 | 4 | 0.006 | 21 | 8% | 0.006 | 0.007 | 0.014 | 68 / 0 |
| attn k/v | 128 x 896 | 6 | gemvm | 1 x 256 | 2 | 0.006 | 20 | 8% | 0.006 | 0.008 | 0.016 | 62 / 0 |
| attn k/v | 128 x 896 | 7 | gemvm | 2 x 256 | 2 | 0.006 | 21 | 8% | 0.006 | 0.008 | 0.017 | 62 / 0 |
| attn k/v | 128 x 896 | 8 | gemvm | 2 x 256 | 2 | 0.006 | 20 | 8% | 0.006 | 0.009 | 0.019 | 53 / 0 |
| lm_head | 151936 x 896 | 1 | gemv1 | 16 x 512 | 2 | 0.859 | 159 | 65% | n/a | n/a | n/a | 36 / 0 |
| lm_head | 151936 x 896 | 2 | gemvr | 32 x 128 | 4 | 1.118 | 122 | 50% | 1.619 | 1.118 | 1.316 | 76 / 0 |
| lm_head | 151936 x 896 | 3 | gemvs | 32 x 64 | 1 | 1.305 | 105 | 43% | 1.892 | 1.344 | 1.305 | 76 / 0 |
| lm_head | 151936 x 896 | 4 | gemvs | 32 x 64 | 1 | 1.527 | 89 | 37% | 2.008 | 1.891 | 1.527 | 68 / 0 |
| lm_head | 151936 x 896 | 5 | gemvs | 64 x 64 | 2 | 1.644 | 83 | 34% | 2.592 | 2.327 | 1.644 | 68 / 0 |
| lm_head | 151936 x 896 | 6 | gemvs | 32 x 32 | 1 | 1.784 | 76 | 31% | 2.593 | 2.599 | 1.784 | 62 / 0 |
| lm_head | 151936 x 896 | 7 | gemvs | 32 x 32 | 1 | 1.953 | 70 | 29% | 2.627 | 2.933 | 1.953 | 62 / 0 |
| lm_head | 151936 x 896 | 8 | gemvs | 128 x 32 | 4 | 2.076 | 66 | 27% | 2.707 | 3.347 | 2.076 | 53 / 0 |
| mlp gate/up | 8960 x 1536 | 1 | gemv1 | 16 x 512 | 8 | 0.060 | 231 | 95% | n/a | n/a | n/a | 36 / 0 |
| mlp gate/up | 8960 x 1536 | 2 | gemvr | 8 x 256 | 2 | 0.092 | 150 | 62% | 0.131 | 0.092 | 0.106 | 76 / 0 |
| mlp gate/up | 8960 x 1536 | 3 | gemvs | 32 x 64 | 1 | 0.107 | 128 | 53% | 0.166 | 0.123 | 0.107 | 76 / 0 |
| mlp gate/up | 8960 x 1536 | 4 | gemvs | 32 x 64 | 1 | 0.109 | 127 | 52% | 0.180 | 0.159 | 0.109 | 68 / 0 |
| mlp gate/up | 8960 x 1536 | 5 | gemvs | 32 x 64 | 1 | 0.111 | 124 | 51% | 0.208 | 0.186 | 0.111 | 68 / 0 |
| mlp gate/up | 8960 x 1536 | 6 | gemvs | 32 x 32 | 1 | 0.121 | 114 | 47% | 0.209 | 0.218 | 0.121 | 62 / 0 |
| mlp gate/up | 8960 x 1536 | 7 | gemvs | 32 x 32 | 1 | 0.129 | 107 | 44% | 0.212 | 0.253 | 0.129 | 62 / 0 |
| mlp gate/up | 8960 x 1536 | 8 | gemvs | 64 x 32 | 2 | 0.131 | 105 | 43% | 0.217 | 0.304 | 0.131 | 53 / 0 |
| mlp down | 1536 x 8960 | 1 | gemv1 | 8 x 512 | 4 | 0.060 | 230 | 94% | n/a | n/a | n/a | 36 / 0 |
| mlp down | 1536 x 8960 | 2 | gemvr | 8 x 128 | 4 | 0.089 | 154 | 63% | 0.108 | 0.089 | 0.164 | 76 / 0 |
| mlp down | 1536 x 8960 | 3 | gemvr | 4 x 256 | 4 | 0.127 | 108 | 44% | 0.131 | 0.127 | 0.176 | 76 / 0 |
| mlp down | 1536 x 8960 | 4 | gemvm | 4 x 256 | 4 | 0.136 | 101 | 42% | 0.136 | 0.157 | 0.184 | 68 / 0 |
| mlp down | 1536 x 8960 | 5 | gemvm | 4 x 256 | 4 | 0.168 | 82 | 34% | 0.168 | 0.180 | 0.205 | 68 / 0 |
| mlp down | 1536 x 8960 | 6 | gemvm | 4 x 256 | 4 | 0.170 | 81 | 33% | 0.170 | 0.218 | 0.218 | 62 / 0 |
| mlp down | 1536 x 8960 | 7 | gemvm | 4 x 256 | 4 | 0.182 | 76 | 31% | 0.182 | 0.241 | 0.253 | 62 / 0 |
| mlp down | 1536 x 8960 | 8 | gemvm | 4 x 256 | 4 | 0.176 | 78 | 32% | 0.176 | 0.309 | 0.270 | 53 / 0 |
| attn q/o | 1536 x 1536 | 1 | gemv1 | 8 x 256 | 4 | 0.007 | 358 | 147% | n/a | n/a | n/a | 36 / 0 |
| attn q/o | 1536 x 1536 | 2 | gemvm | 4 x 256 | 2 | 0.016 | 148 | 61% | 0.016 | 0.016 | 0.018 | 76 / 0 |
| attn q/o | 1536 x 1536 | 3 | gemvs | 16 x 64 | 1 | 0.019 | 122 | 50% | 0.021 | 0.020 | 0.019 | 76 / 0 |
| attn q/o | 1536 x 1536 | 4 | gemvm | 4 x 256 | 2 | 0.022 | 106 | 44% | 0.022 | 0.028 | 0.022 | 68 / 0 |
| attn q/o | 1536 x 1536 | 5 | gemvs | 16 x 64 | 1 | 0.026 | 91 | 37% | 0.029 | 0.034 | 0.026 | 68 / 0 |
| attn q/o | 1536 x 1536 | 6 | gemvm | 4 x 256 | 2 | 0.029 | 83 | 34% | 0.029 | 0.037 | 0.029 | 62 / 0 |
| attn q/o | 1536 x 1536 | 7 | gemvm | 4 x 256 | 2 | 0.030 | 79 | 33% | 0.030 | 0.042 | 0.032 | 62 / 0 |
| attn q/o | 1536 x 1536 | 8 | gemvm | 4 x 256 | 2 | 0.030 | 79 | 32% | 0.030 | 0.049 | 0.035 | 53 / 0 |
| attn k/v | 256 x 1536 | 1 | gemv1 | 8 x 512 | 8 | 0.004 | 101 | 42% | n/a | n/a | n/a | 36 / 0 |
| attn k/v | 256 x 1536 | 2 | gemvm | 4 x 64 | 8 | 0.005 | 76 | 31% | 0.005 | 0.008 | 0.014 | 76 / 0 |
| attn k/v | 256 x 1536 | 3 | gemvm | 2 x 256 | 2 | 0.006 | 61 | 25% | 0.006 | 0.008 | 0.017 | 76 / 0 |
| attn k/v | 256 x 1536 | 4 | gemvm | 2 x 256 | 2 | 0.006 | 61 | 25% | 0.006 | 0.009 | 0.020 | 68 / 0 |
| attn k/v | 256 x 1536 | 5 | gemvm | 2 x 256 | 2 | 0.008 | 48 | 20% | 0.008 | 0.010 | 0.022 | 68 / 0 |
| attn k/v | 256 x 1536 | 6 | gemvm | 2 x 256 | 2 | 0.008 | 48 | 20% | 0.008 | 0.011 | 0.025 | 62 / 0 |
| attn k/v | 256 x 1536 | 7 | gemvm | 4 x 256 | 2 | 0.009 | 46 | 19% | 0.009 | 0.012 | 0.027 | 62 / 0 |
| attn k/v | 256 x 1536 | 8 | gemvm | 2 x 256 | 2 | 0.008 | 47 | 19% | 0.008 | 0.013 | 0.028 | 53 / 0 |
| lm_head | 151936 x 1536 | 1 | gemv1 | 16 x 512 | 2 | 1.100 | 212 | 87% | n/a | n/a | n/a | 36 / 0 |
| lm_head | 151936 x 1536 | 2 | gemvr | 16 x 256 | 4 | 1.724 | 136 | 56% | 2.260 | 1.724 | 2.365 | 76 / 0 |
| lm_head | 151936 x 1536 | 3 | gemvr | 16 x 256 | 4 | 2.310 | 101 | 42% | 2.920 | 2.310 | 2.484 | 76 / 0 |
| lm_head | 151936 x 1536 | 4 | gemvs | 64 x 64 | 2 | 2.649 | 88 | 36% | 2.921 | 3.124 | 2.649 | 68 / 0 |
| lm_head | 151936 x 1536 | 5 | gemvs | 64 x 64 | 2 | 2.855 | 82 | 34% | 3.495 | 3.778 | 2.855 | 68 / 0 |
| lm_head | 151936 x 1536 | 6 | gemvs | 32 x 32 | 1 | 3.222 | 73 | 30% | 3.569 | 4.291 | 3.222 | 62 / 0 |
| lm_head | 151936 x 1536 | 7 | gemvs | 32 x 32 | 1 | 3.425 | 68 | 28% | 3.617 | 4.887 | 3.425 | 62 / 0 |
| lm_head | 151936 x 1536 | 8 | gemvs | 32 x 32 | 1 | 3.614 | 65 | 27% | 3.712 | 5.450 | 3.614 | 53 / 0 |
| mlp gate/up | 11008 x 2048 | 1 | gemv1 | 16 x 512 | 8 | 0.095 | 238 | 98% | n/a | n/a | n/a | 36 / 0 |
| mlp gate/up | 11008 x 2048 | 2 | gemvr | 16 x 256 | 4 | 0.149 | 152 | 62% | 0.197 | 0.149 | 0.165 | 76 / 0 |
| mlp gate/up | 11008 x 2048 | 3 | gemvs | 32 x 64 | 1 | 0.166 | 136 | 56% | 0.266 | 0.170 | 0.166 | 76 / 0 |
| mlp gate/up | 11008 x 2048 | 4 | gemvs | 32 x 64 | 1 | 0.172 | 131 | 54% | 0.284 | 0.241 | 0.172 | 68 / 0 |
| mlp gate/up | 11008 x 2048 | 5 | gemvs | 32 x 64 | 1 | 0.181 | 125 | 51% | 0.320 | 0.289 | 0.181 | 68 / 0 |
| mlp gate/up | 11008 x 2048 | 6 | gemvs | 32 x 32 | 1 | 0.211 | 107 | 44% | 0.322 | 0.347 | 0.211 | 62 / 0 |
| mlp gate/up | 11008 x 2048 | 7 | gemvs | 64 x 32 | 2 | 0.223 | 101 | 42% | 0.328 | 0.410 | 0.223 | 62 / 0 |
| mlp gate/up | 11008 x 2048 | 8 | gemvs | 32 x 32 | 1 | 0.242 | 93 | 38% | 0.343 | 0.530 | 0.242 | 53 / 0 |
| mlp down | 2048 x 11008 | 1 | gemv1 | 16 x 512 | 8 | 0.098 | 230 | 94% | n/a | n/a | n/a | 36 / 0 |
| mlp down | 2048 x 11008 | 2 | gemvr | 2 x 256 | 2 | 0.140 | 161 | 66% | 0.163 | 0.140 | 0.243 | 76 / 0 |
| mlp down | 2048 x 11008 | 3 | gemvr | 4 x 256 | 4 | 0.184 | 123 | 50% | 0.214 | 0.184 | 0.254 | 76 / 0 |
| mlp down | 2048 x 11008 | 4 | gemvm | 4 x 256 | 4 | 0.215 | 105 | 43% | 0.215 | 0.241 | 0.280 | 68 / 0 |
| mlp down | 2048 x 11008 | 5 | gemvm | 4 x 256 | 4 | 0.265 | 85 | 35% | 0.265 | 0.307 | 0.316 | 68 / 0 |
| mlp down | 2048 x 11008 | 6 | gemvm | 4 x 256 | 4 | 0.281 | 80 | 33% | 0.281 | 0.349 | 0.340 | 62 / 0 |
| mlp down | 2048 x 11008 | 7 | gemvm | 4 x 256 | 4 | 0.282 | 80 | 33% | 0.282 | 0.447 | 0.377 | 62 / 0 |
| mlp down | 2048 x 11008 | 8 | gemvm | 4 x 256 | 4 | 0.285 | 79 | 33% | 0.285 | 0.529 | 0.415 | 53 / 0 |
| attn q/o | 2048 x 2048 | 1 | gemv1 | 8 x 512 | 4 | 0.009 | 453 | 186% | n/a | n/a | n/a | 36 / 0 |
| attn q/o | 2048 x 2048 | 2 | gemvm | 4 x 256 | 2 | 0.027 | 155 | 64% | 0.027 | 0.027 | 0.029 | 76 / 0 |
| attn q/o | 2048 x 2048 | 3 | gemvr | 16 x 64 | 1 | 0.033 | 128 | 53% | 0.038 | 0.033 | 0.033 | 76 / 0 |
| attn q/o | 2048 x 2048 | 4 | gemvs | 16 x 64 | 1 | 0.040 | 105 | 43% | 0.040 | 0.046 | 0.040 | 68 / 0 |
| attn q/o | 2048 x 2048 | 5 | gemvs | 16 x 64 | 1 | 0.046 | 91 | 37% | 0.050 | 0.056 | 0.046 | 68 / 0 |
| attn q/o | 2048 x 2048 | 6 | gemvm | 4 x 128 | 2 | 0.049 | 85 | 35% | 0.049 | 0.064 | 0.050 | 62 / 0 |
| attn q/o | 2048 x 2048 | 7 | gemvm | 4 x 128 | 2 | 0.050 | 84 | 35% | 0.050 | 0.072 | 0.056 | 62 / 0 |
| attn q/o | 2048 x 2048 | 8 | gemvm | 4 x 128 | 2 | 0.054 | 78 | 32% | 0.054 | 0.095 | 0.065 | 53 / 0 |
| attn k/v | 256 x 2048 | 1 | gemv1 | 8 x 128 | 8 | 0.004 | 125 | 52% | n/a | n/a | n/a | 36 / 0 |
| attn k/v | 256 x 2048 | 2 | gemvm | 4 x 64 | 8 | 0.006 | 87 | 36% | 0.006 | 0.008 | 0.017 | 76 / 0 |
| attn k/v | 256 x 2048 | 3 | gemvm | 2 x 256 | 2 | 0.007 | 72 | 29% | 0.007 | 0.009 | 0.021 | 76 / 0 |
| attn k/v | 256 x 2048 | 4 | gemvm | 2 x 256 | 2 | 0.008 | 69 | 28% | 0.008 | 0.011 | 0.025 | 68 / 0 |
| attn k/v | 256 x 2048 | 5 | gemvm | 4 x 256 | 2 | 0.010 | 53 | 22% | 0.010 | 0.012 | 0.027 | 68 / 0 |
| attn k/v | 256 x 2048 | 6 | gemvm | 2 x 256 | 2 | 0.010 | 54 | 22% | 0.010 | 0.013 | 0.031 | 62 / 0 |
| attn k/v | 256 x 2048 | 7 | gemvm | 4 x 256 | 2 | 0.010 | 53 | 22% | 0.010 | 0.014 | 0.035 | 62 / 0 |
| attn k/v | 256 x 2048 | 8 | gemvm | 4 x 256 | 2 | 0.010 | 52 | 21% | 0.010 | 0.015 | 0.035 | 53 / 0 |
| lm_head | 151936 x 2048 | 1 | gemv1 | 16 x 512 | 2 | 1.414 | 220 | 91% | n/a | n/a | n/a | 36 / 0 |
| lm_head | 151936 x 2048 | 2 | gemvr | 16 x 256 | 4 | 2.274 | 137 | 56% | 2.940 | 2.274 | 3.052 | 76 / 0 |
| lm_head | 151936 x 2048 | 3 | gemvr | 16 x 256 | 4 | 3.028 | 103 | 42% | 3.725 | 3.028 | 3.165 | 76 / 0 |
| lm_head | 151936 x 2048 | 4 | gemvs | 32 x 64 | 1 | 3.558 | 88 | 36% | 3.803 | 4.100 | 3.558 | 68 / 0 |
| lm_head | 151936 x 2048 | 5 | gemvs | 32 x 64 | 1 | 3.658 | 85 | 35% | 4.352 | 4.895 | 3.658 | 68 / 0 |
| lm_head | 151936 x 2048 | 6 | gemvs | 32 x 32 | 1 | 4.039 | 77 | 32% | 4.533 | 5.719 | 4.039 | 62 / 0 |
| lm_head | 151936 x 2048 | 7 | gemvs | 128 x 32 | 4 | 4.227 | 74 | 30% | 4.577 | 6.539 | 4.227 | 62 / 0 |
| lm_head | 151936 x 2048 | 8 | gemvs | 128 x 32 | 4 | 4.465 | 70 | 29% | 4.698 | 7.451 | 4.465 | 53 / 0 |

**The kernel families against each other.** The best time of each family, averaged over the large layers (`mlp gate/up`, `mlp down`, `lm_head`) of every model size, and how many times faster the better of the two new kernels is than `gemvm`.

| rows of x | gemvm (ms) | gemvr (ms) | gemvs (ms) | better new kernel over gemvm |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.835 | 0.629 | 0.836 | 1.33x |
| 3 | 1.050 | 0.820 | 0.865 | 1.29x |
| 4 | 1.071 | 1.112 | 0.958 | 1.13x |
| 5 | 1.287 | 1.351 | 1.014 | 1.28x |
| 6 | 1.312 | 1.542 | 1.124 | 1.17x |
| 7 | 1.329 | 1.762 | 1.200 | 1.12x |
| 8 | 1.364 | 2.011 | 1.270 | 1.08x |

**What the compiler made of the configurations** (registers per thread, registers spilled to local memory and the shared memory a kernel allocates, as the GPU compiler reports them for the largest layer of the gate model). Kept and removed are the configurations the `fits` rule keeps and drops.

| kernel | rows of x | rule | configurations | fewest registers | most registers | with spills | with shared memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| gemvr | 2 | kept | 26 | 68 | 127 | 0 | 17 |
| gemvr | 2 | removed | 14 | 33 | 255 | 2 | 12 |
| gemvr | 5 | kept | 18 | 80 | 146 | 0 | 12 |
| gemvr | 5 | removed | 12 | 64 | 255 | 5 | 10 |
| gemvr | 8 | kept | 9 | 127 | 128 | 0 | 6 |
| gemvr | 8 | removed | 15 | 64 | 178 | 5 | 12 |
| gemvs | 2 | kept | 22 | 60 | 120 | 0 | 15 |
| gemvs | 2 | removed | 14 | 25 | 168 | 0 | 13 |
| gemvs | 5 | kept | 22 | 72 | 128 | 0 | 15 |
| gemvs | 5 | removed | 14 | 38 | 255 | 1 | 13 |
| gemvs | 8 | kept | 16 | 71 | 144 | 0 | 9 |
| gemvs | 8 | removed | 20 | 56 | 255 | 2 | 19 |

**Correctness** of every kernel variant at its tuned configuration against the dequantize path, at every shape and 11 row counts (495 cases): the largest relative error is 9.5e-04, and with a wrong scale (each row given its neighbour's) the smallest relative error is 1.5e-01, so the check can fail. By rows of x:

| rows of x | kernels that ran | cases | largest relative error | smallest error with a wrong scale |
| --- | --- | ---: | ---: | ---: |
| 1 | gemv1 | 30 | 4.2e-04 | 1.5e-01 |
| 2 | gemvm, gemvr, gemvs | 60 | 4.8e-04 | 1.8e-01 |
| 3 | gemvm, gemvr, gemvs | 60 | 4.5e-04 | 1.6e-01 |
| 4 | gemvm, gemvr, gemvs | 60 | 4.5e-04 | 1.7e-01 |
| 5 | gemvm, gemvr, gemvs | 60 | 5.0e-04 | 2.0e-01 |
| 6 | gemvm, gemvr, gemvs | 60 | 4.4e-04 | 2.1e-01 |
| 7 | gemvm, gemvr, gemvs | 60 | 4.9e-04 | 2.2e-01 |
| 8 | gemvm, gemvr, gemvs | 60 | 4.3e-04 | 1.9e-01 |
| 9 | dequant | 15 | 6.8e-04 | 2.1e-01 |
| 16 | dequant | 15 | 7.6e-04 | 2.3e-01 |
| 37 | dequant | 15 | 9.5e-04 | 1.9e-01 |

<a id="stage11-gate"></a>

### The one-layer gate

The large layers of the gate model, each way timed as 20 calls inside one CUDA graph divided by 20, with the ways taking turns over several rounds (the order rotated each round) so that a drifting clock hits all alike. Spread is the range of the rounds over their median and noisy marks a spread above a tenth. GB/s is the bytes of the stored weight over the time, against a measured copy bandwidth of 243 GB/s; vs fp16 is the fp16 layer's time over this time, above one means faster than the fp16 layer.

| layer | weight shape | rows of x | way | time (ms) | spread | noisy | GB/s | vs fp16 layer |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| mlp gate/up | 8960 x 1536 | 1 | fp16 layer | 0.241 | 57% | yes | 114 | 1.00x |
| mlp gate/up | 8960 x 1536 | 1 | int8 one-row kernel (gemv1) | 0.065 | 26% | yes | 211 | 3.69x |
| mlp gate/up | 8960 x 1536 | 1 | int8 tuned winner | 0.071 | 20% | yes | 195 | 3.40x |
| mlp gate/up | 8960 x 1536 | 2 | fp16 layer | 0.122 | 37% | yes | 226 | 1.00x |
| mlp gate/up | 8960 x 1536 | 2 | int8 Stage 10 kernel (gemvm) | 0.149 | 31% | yes | 92 | 0.82x |
| mlp gate/up | 8960 x 1536 | 2 | int8 gemvr | 0.124 | 13% | yes | 111 | 0.99x |
| mlp gate/up | 8960 x 1536 | 2 | int8 gemvs | 0.109 | 5% | no | 126 | 1.12x |
| mlp gate/up | 8960 x 1536 | 2 | int8 tuned winner | 0.135 | 25% | yes | 102 | 0.91x |
| mlp gate/up | 8960 x 1536 | 3 | fp16 layer | 0.148 | 19% | yes | 186 | 1.00x |
| mlp gate/up | 8960 x 1536 | 3 | int8 Stage 10 kernel (gemvm) | 0.190 | 5% | no | 73 | 0.78x |
| mlp gate/up | 8960 x 1536 | 3 | int8 gemvr | 0.139 | 13% | yes | 99 | 1.07x |
| mlp gate/up | 8960 x 1536 | 3 | int8 gemvs | 0.108 | 1% | no | 127 | 1.37x |
| mlp gate/up | 8960 x 1536 | 3 | int8 tuned winner | 0.110 | 17% | yes | 125 | 1.34x |
| mlp gate/up | 8960 x 1536 | 4 | fp16 layer | 0.149 | 7% | no | 185 | 1.00x |
| mlp gate/up | 8960 x 1536 | 4 | int8 Stage 10 kernel (gemvm) | 0.189 | 5% | no | 73 | 0.79x |
| mlp gate/up | 8960 x 1536 | 4 | int8 gemvr | 0.211 | 4% | no | 65 | 0.70x |
| mlp gate/up | 8960 x 1536 | 4 | int8 gemvs | 0.109 | 3% | no | 126 | 1.36x |
| mlp gate/up | 8960 x 1536 | 4 | int8 tuned winner | 0.140 | 25% | yes | 98 | 1.06x |
| mlp gate/up | 8960 x 1536 | 5 | fp16 layer | 0.140 | 5% | no | 196 | 1.00x |
| mlp gate/up | 8960 x 1536 | 5 | int8 Stage 10 kernel (gemvm) | 0.217 | 5% | no | 63 | 0.65x |
| mlp gate/up | 8960 x 1536 | 5 | int8 gemvr | 0.236 | 8% | no | 58 | 0.59x |
| mlp gate/up | 8960 x 1536 | 5 | int8 gemvs | 0.111 | 1% | no | 125 | 1.27x |
| mlp gate/up | 8960 x 1536 | 5 | int8 tuned winner | 0.150 | 31% | yes | 92 | 0.93x |
| mlp gate/up | 8960 x 1536 | 6 | fp16 layer | 0.135 | 6% | no | 204 | 1.00x |
| mlp gate/up | 8960 x 1536 | 6 | int8 Stage 10 kernel (gemvm) | 0.222 | 5% | no | 62 | 0.61x |
| mlp gate/up | 8960 x 1536 | 6 | int8 gemvr | 0.275 | 6% | no | 50 | 0.49x |
| mlp gate/up | 8960 x 1536 | 6 | int8 gemvs | 0.141 | 17% | yes | 98 | 0.96x |
| mlp gate/up | 8960 x 1536 | 6 | int8 tuned winner | 0.167 | 23% | yes | 82 | 0.81x |
| mlp gate/up | 8960 x 1536 | 7 | fp16 layer | 0.137 | 5% | no | 200 | 1.00x |
| mlp gate/up | 8960 x 1536 | 7 | int8 Stage 10 kernel (gemvm) | 0.225 | 6% | no | 61 | 0.61x |
| mlp gate/up | 8960 x 1536 | 7 | int8 gemvr | 0.306 | 3% | no | 45 | 0.45x |
| mlp gate/up | 8960 x 1536 | 7 | int8 gemvs | 0.144 | 18% | yes | 96 | 0.96x |
| mlp gate/up | 8960 x 1536 | 7 | int8 tuned winner | 0.180 | 33% | yes | 76 | 0.76x |
| mlp gate/up | 8960 x 1536 | 8 | fp16 layer | 0.132 | 5% | no | 209 | 1.00x |
| mlp gate/up | 8960 x 1536 | 8 | int8 Stage 10 kernel (gemvm) | 0.223 | 5% | no | 62 | 0.59x |
| mlp gate/up | 8960 x 1536 | 8 | int8 gemvr | 0.350 | 3% | no | 39 | 0.38x |
| mlp gate/up | 8960 x 1536 | 8 | int8 gemvs | 0.155 | 23% | yes | 89 | 0.85x |
| mlp gate/up | 8960 x 1536 | 8 | int8 tuned winner | 0.181 | 16% | yes | 76 | 0.73x |
| mlp down | 1536 x 8960 | 1 | fp16 layer | 0.126 | 32% | yes | 219 | 1.00x |
| mlp down | 1536 x 8960 | 1 | int8 one-row kernel (gemv1) | 0.064 | 41% | yes | 216 | 1.98x |
| mlp down | 1536 x 8960 | 1 | int8 tuned winner | 0.063 | 21% | yes | 218 | 1.99x |
| mlp down | 1536 x 8960 | 2 | fp16 layer | 0.121 | 32% | yes | 227 | 1.00x |
| mlp down | 1536 x 8960 | 2 | int8 Stage 10 kernel (gemvm) | 0.121 | 30% | yes | 114 | 1.00x |
| mlp down | 1536 x 8960 | 2 | int8 gemvr | 0.118 | 9% | no | 116 | 1.02x |
| mlp down | 1536 x 8960 | 2 | int8 gemvs | 0.168 | 2% | no | 82 | 0.72x |
| mlp down | 1536 x 8960 | 2 | int8 tuned winner | 0.107 | 12% | yes | 129 | 1.13x |
| mlp down | 1536 x 8960 | 3 | fp16 layer | 0.120 | 0% | no | 229 | 1.00x |
| mlp down | 1536 x 8960 | 3 | int8 Stage 10 kernel (gemvm) | 0.144 | 20% | yes | 95 | 0.83x |
| mlp down | 1536 x 8960 | 3 | int8 gemvr | 0.165 | 2% | no | 83 | 0.73x |
| mlp down | 1536 x 8960 | 3 | int8 gemvs | 0.178 | 7% | no | 77 | 0.68x |
| mlp down | 1536 x 8960 | 3 | int8 tuned winner | 0.149 | 4% | no | 92 | 0.81x |
| mlp down | 1536 x 8960 | 4 | fp16 layer | 0.121 | 4% | no | 228 | 1.00x |
| mlp down | 1536 x 8960 | 4 | int8 Stage 10 kernel (gemvm) | 0.152 | 24% | yes | 91 | 0.80x |
| mlp down | 1536 x 8960 | 4 | int8 gemvr | 0.201 | 8% | no | 68 | 0.60x |
| mlp down | 1536 x 8960 | 4 | int8 gemvs | 0.194 | 9% | no | 71 | 0.62x |
| mlp down | 1536 x 8960 | 4 | int8 tuned winner | 0.125 | 18% | yes | 110 | 0.97x |
| mlp down | 1536 x 8960 | 5 | fp16 layer | 0.122 | 2% | no | 226 | 1.00x |
| mlp down | 1536 x 8960 | 5 | int8 Stage 10 kernel (gemvm) | 0.177 | 8% | no | 78 | 0.69x |
| mlp down | 1536 x 8960 | 5 | int8 gemvr | 0.232 | 3% | no | 59 | 0.53x |
| mlp down | 1536 x 8960 | 5 | int8 gemvs | 0.210 | 6% | no | 65 | 0.58x |
| mlp down | 1536 x 8960 | 5 | int8 tuned winner | 0.148 | 8% | no | 93 | 0.82x |
| mlp down | 1536 x 8960 | 6 | fp16 layer | 0.122 | 2% | no | 226 | 1.00x |
| mlp down | 1536 x 8960 | 6 | int8 Stage 10 kernel (gemvm) | 0.179 | 9% | no | 77 | 0.68x |
| mlp down | 1536 x 8960 | 6 | int8 gemvr | 0.266 | 4% | no | 52 | 0.46x |
| mlp down | 1536 x 8960 | 6 | int8 gemvs | 0.229 | 5% | no | 60 | 0.53x |
| mlp down | 1536 x 8960 | 6 | int8 tuned winner | 0.150 | 14% | yes | 92 | 0.81x |
| mlp down | 1536 x 8960 | 7 | fp16 layer | 0.122 | 3% | no | 225 | 1.00x |
| mlp down | 1536 x 8960 | 7 | int8 Stage 10 kernel (gemvm) | 0.176 | 6% | no | 78 | 0.69x |
| mlp down | 1536 x 8960 | 7 | int8 gemvr | 0.295 | 2% | no | 47 | 0.41x |
| mlp down | 1536 x 8960 | 7 | int8 gemvs | 0.249 | 7% | no | 55 | 0.49x |
| mlp down | 1536 x 8960 | 7 | int8 tuned winner | 0.165 | 10% | no | 84 | 0.74x |
| mlp down | 1536 x 8960 | 8 | fp16 layer | 0.123 | 3% | no | 224 | 1.00x |
| mlp down | 1536 x 8960 | 8 | int8 Stage 10 kernel (gemvm) | 0.194 | 9% | no | 71 | 0.63x |
| mlp down | 1536 x 8960 | 8 | int8 gemvr | 0.331 | 2% | no | 42 | 0.37x |
| mlp down | 1536 x 8960 | 8 | int8 gemvs | 0.295 | 1% | no | 47 | 0.42x |
| mlp down | 1536 x 8960 | 8 | int8 tuned winner | 0.161 | 21% | yes | 86 | 0.76x |
| lm_head | 151936 x 1536 | 1 | fp16 layer | 4.364 | 0% | no | 107 | 1.00x |
| lm_head | 151936 x 1536 | 1 | int8 one-row kernel (gemv1) | 1.041 | 0% | no | 224 | 4.19x |
| lm_head | 151936 x 1536 | 1 | int8 tuned winner | 1.043 | 0% | no | 224 | 4.19x |
| lm_head | 151936 x 1536 | 2 | fp16 layer | 2.411 | 7% | no | 194 | 1.00x |
| lm_head | 151936 x 1536 | 2 | int8 Stage 10 kernel (gemvm) | 2.171 | 8% | no | 108 | 1.11x |
| lm_head | 151936 x 1536 | 2 | int8 gemvr | 1.687 | 4% | no | 139 | 1.43x |
| lm_head | 151936 x 1536 | 2 | int8 gemvs | 2.285 | 12% | yes | 102 | 1.06x |
| lm_head | 151936 x 1536 | 2 | int8 tuned winner | 1.687 | 3% | no | 139 | 1.43x |
| lm_head | 151936 x 1536 | 3 | fp16 layer | 2.843 | 9% | no | 164 | 1.00x |
| lm_head | 151936 x 1536 | 3 | int8 Stage 10 kernel (gemvm) | 3.030 | 6% | no | 77 | 0.94x |
| lm_head | 151936 x 1536 | 3 | int8 gemvr | 2.357 | 4% | no | 99 | 1.21x |
| lm_head | 151936 x 1536 | 3 | int8 gemvs | 3.650 | 77% | yes | 64 | 0.78x |
| lm_head | 151936 x 1536 | 3 | int8 tuned winner | 2.395 | 2% | no | 98 | 1.19x |
| lm_head | 151936 x 1536 | 4 | fp16 layer | 2.606 | 9% | no | 179 | 1.00x |
| lm_head | 151936 x 1536 | 4 | int8 Stage 10 kernel (gemvm) | 2.933 | 7% | no | 80 | 0.89x |
| lm_head | 151936 x 1536 | 4 | int8 gemvr | 3.155 | 3% | no | 74 | 0.83x |
| lm_head | 151936 x 1536 | 4 | int8 gemvs | 2.731 | 8% | no | 86 | 0.95x |
| lm_head | 151936 x 1536 | 4 | int8 tuned winner | 2.691 | 4% | no | 87 | 0.97x |
| lm_head | 151936 x 1536 | 5 | fp16 layer | 2.542 | 1% | no | 184 | 1.00x |
| lm_head | 151936 x 1536 | 5 | int8 Stage 10 kernel (gemvm) | 3.417 | 1% | no | 68 | 0.74x |
| lm_head | 151936 x 1536 | 5 | int8 gemvr | 3.680 | 1% | no | 63 | 0.69x |
| lm_head | 151936 x 1536 | 5 | int8 gemvs | 2.842 | 0% | no | 82 | 0.89x |
| lm_head | 151936 x 1536 | 5 | int8 tuned winner | 2.842 | 0% | no | 82 | 0.89x |
| lm_head | 151936 x 1536 | 6 | fp16 layer | 2.674 | 3% | no | 175 | 1.00x |
| lm_head | 151936 x 1536 | 6 | int8 Stage 10 kernel (gemvm) | 3.583 | 3% | no | 65 | 0.75x |
| lm_head | 151936 x 1536 | 6 | int8 gemvr | 4.357 | 1% | no | 54 | 0.61x |
| lm_head | 151936 x 1536 | 6 | int8 gemvs | 3.238 | 4% | no | 72 | 0.83x |
| lm_head | 151936 x 1536 | 6 | int8 tuned winner | 3.242 | 4% | no | 72 | 0.82x |
| lm_head | 151936 x 1536 | 7 | fp16 layer | 2.708 | 3% | no | 172 | 1.00x |
| lm_head | 151936 x 1536 | 7 | int8 Stage 10 kernel (gemvm) | 3.677 | 2% | no | 64 | 0.74x |
| lm_head | 151936 x 1536 | 7 | int8 gemvr | 4.916 | 1% | no | 48 | 0.55x |
| lm_head | 151936 x 1536 | 7 | int8 gemvs | 3.445 | 3% | no | 68 | 0.79x |
| lm_head | 151936 x 1536 | 7 | int8 tuned winner | 3.425 | 3% | no | 68 | 0.79x |
| lm_head | 151936 x 1536 | 8 | fp16 layer | 2.694 | 0% | no | 173 | 1.00x |
| lm_head | 151936 x 1536 | 8 | int8 Stage 10 kernel (gemvm) | 3.708 | 0% | no | 63 | 0.73x |
| lm_head | 151936 x 1536 | 8 | int8 gemvr | 5.503 | 1% | no | 42 | 0.49x |
| lm_head | 151936 x 1536 | 8 | int8 gemvs | 3.613 | 0% | no | 65 | 0.75x |
| lm_head | 151936 x 1536 | 8 | int8 tuned winner | 3.613 | 0% | no | 65 | 0.75x |

**Mean over the large layers**, by rows of x, against the fp16 layer, and the tuned winner over Stage 10's `gemvm` measured in the same turns:

| rows of x | tuned winner vs fp16 | gemvm vs fp16 | winner over gemvm |
| --- | ---: | ---: | ---: |
| 1 | 3.19x | n/a | n/a |
| 2 | 1.16x | 0.98x | 1.18x |
| 3 | 1.11x | 0.85x | 1.31x |
| 4 | 1.00x | 0.82x | 1.21x |
| 5 | 0.88x | 0.69x | 1.28x |
| 6 | 0.81x | 0.68x | 1.20x |
| 7 | 0.77x | 0.68x | 1.12x |
| 8 | 0.75x | 0.65x | 1.15x |

Gate verdict: **no-go**. At 5 rows the tuned int8 kernel is 0.88x the fp16 layer on the large layers (1 row 3.19x, 8 rows 0.75x), mean over the large layers.

**Does converting x once pay?** For the best new configuration at 5 rows of each large layer, in turns: the call with `x_fp32` (the conversion is a kernel of its own), the same configuration with an fp16 x converted inside the loop, and the conversion alone.

| layer | kernel | tile | with x_fp32 (ms) | fp16 x in the loop (ms) | conversion alone (ms) | conversion share | x_fp32 gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up | gemvs | 32 x 64 | 0.1455 | 0.1130 | 0.0060 | 4.1% | 0.78x |
| mlp down | gemvr | 2 x 256 | 0.2377 | 0.0977 | 0.0058 | 2.4% | 0.41x |
| lm_head | gemvs | 64 x 64 | 2.8400 | 4.7184 | 0.0171 | 0.6% | 1.66x |

<a id="stage11-block"></a>

### A checking pass in a graph

A forward pass of several tokens of one sequence over the static KV cache, on top of 256 cached tokens, argmax on top, replayed from a CUDA graph. The passes of one set of weights take turns over several rounds, so relative to one token (the `v` of the cost model) is a ratio taken at the same clock. The Stage 10 kernel is measured in the same session, as its own set of weights. Floor is the bytes read per token over the measured copy bandwidth, and share is floor over time.

| model | weights | tokens fed | time (ms) | spread | noisy | relative to 1 token | floor (ms) | share of floor |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0.5B | fp16 | 1 | 8.72 | 39% | yes | 1.00x | 4.06 | 46.6% |
| 0.5B | fp16 | 2 | 8.97 | 1% | no | 1.03x | 4.06 | 45.3% |
| 0.5B | fp16 | 3 | 8.89 | 1% | no | 1.02x | 4.06 | 45.7% |
| 0.5B | fp16 | 5 | 9.08 | 1% | no | 1.04x | 4.06 | 44.7% |
| 0.5B | fp16 | 7 | 9.15 | 1% | no | 1.05x | 4.06 | 44.4% |
| 0.5B | fp16, head not padded | 1 | 9.03 | 1% | no | 1.00x | 4.06 | 45.0% |
| 0.5B | int8 Stage 10 kernel | 1 | 6.46 | 23% | yes | 1.00x | 2.04 | 31.5% |
| 0.5B | int8 Stage 10 kernel | 2 | 8.81 | 7% | no | 1.36x | 2.04 | 23.1% |
| 0.5B | int8 Stage 10 kernel | 3 | 9.95 | 2% | no | 1.54x | 2.04 | 20.4% |
| 0.5B | int8 Stage 10 kernel | 5 | 12.16 | 6% | no | 1.88x | 2.04 | 16.7% |
| 0.5B | int8 Stage 10 kernel | 7 | 12.51 | 2% | no | 1.94x | 2.04 | 16.3% |
| 0.5B | int8 this stage | 1 | 6.43 | 2% | no | 1.00x | 2.04 | 31.6% |
| 0.5B | int8 this stage | 2 | 7.77 | 3% | no | 1.21x | 2.04 | 26.2% |
| 0.5B | int8 this stage | 3 | 8.60 | 4% | no | 1.34x | 2.04 | 23.7% |
| 0.5B | int8 this stage | 5 | 9.31 | 3% | no | 1.45x | 2.04 | 21.9% |
| 0.5B | int8 this stage | 7 | 11.99 | 2% | no | 1.86x | 2.04 | 17.0% |
| 1.5B | fp16 | 1 | 19.41 | 1% | no | 1.00x | 12.69 | 65.4% |
| 1.5B | fp16 | 2 | 19.58 | 1% | no | 1.01x | 12.69 | 64.8% |
| 1.5B | fp16 | 3 | 19.89 | 1% | no | 1.02x | 12.69 | 63.8% |
| 1.5B | fp16 | 5 | 20.04 | 1% | no | 1.03x | 12.69 | 63.3% |
| 1.5B | fp16 | 7 | 20.20 | 0% | no | 1.04x | 12.69 | 62.8% |
| 1.5B | fp16, head not padded | 1 | 19.81 | 3% | no | 1.00x | 12.69 | 64.1% |
| 1.5B | int8 Stage 10 kernel | 1 | 11.83 | 2% | no | 1.00x | 6.35 | 53.7% |
| 1.5B | int8 Stage 10 kernel | 2 | 17.04 | 3% | no | 1.44x | 6.35 | 37.3% |
| 1.5B | int8 Stage 10 kernel | 3 | 21.44 | 3% | no | 1.81x | 6.35 | 29.6% |
| 1.5B | int8 Stage 10 kernel | 5 | 26.91 | 4% | no | 2.27x | 6.35 | 23.6% |
| 1.5B | int8 Stage 10 kernel | 7 | 27.70 | 2% | no | 2.34x | 6.35 | 22.9% |
| 1.5B | int8 this stage | 1 | 11.80 | 1% | no | 1.00x | 6.35 | 53.8% |
| 1.5B | int8 this stage | 2 | 16.04 | 3% | no | 1.36x | 6.35 | 39.6% |
| 1.5B | int8 this stage | 3 | 20.02 | 3% | no | 1.70x | 6.35 | 31.7% |
| 1.5B | int8 this stage | 5 | 21.56 | 2% | no | 1.83x | 6.35 | 29.5% |
| 1.5B | int8 this stage | 7 | 23.64 | 6% | no | 2.00x | 6.35 | 26.9% |
| 3B | fp16 | 1 | 35.61 | 13% | yes | 1.00x | 25.37 | 71.3% |
| 3B | fp16 | 2 | 35.82 | 3% | no | 1.01x | 25.37 | 70.8% |
| 3B | fp16 | 3 | 35.94 | 4% | no | 1.01x | 25.37 | 70.6% |
| 3B | fp16 | 5 | 36.07 | 3% | no | 1.01x | 25.37 | 70.4% |
| 3B | fp16 | 7 | 36.57 | 2% | no | 1.03x | 25.37 | 69.4% |
| 3B | fp16, head not padded | 1 | 36.03 | 13% | yes | 1.00x | 25.37 | 70.4% |
| 3B | int8 Stage 10 kernel | 1 | 20.08 | 4% | no | 1.00x | 12.70 | 63.2% |
| 3B | int8 Stage 10 kernel | 2 | 32.12 | 11% | yes | 1.60x | 12.70 | 39.5% |
| 3B | int8 Stage 10 kernel | 3 | 42.11 | 8% | no | 2.10x | 12.70 | 30.2% |
| 3B | int8 Stage 10 kernel | 5 | 49.64 | 9% | no | 2.47x | 12.70 | 25.6% |
| 3B | int8 Stage 10 kernel | 7 | 51.27 | 6% | no | 2.55x | 12.70 | 24.8% |
| 3B | int8 this stage | 1 | 20.27 | 3% | no | 1.00x | 12.70 | 62.6% |
| 3B | int8 this stage | 2 | 29.87 | 6% | no | 1.47x | 12.70 | 42.5% |
| 3B | int8 this stage | 3 | 35.86 | 6% | no | 1.77x | 12.70 | 35.4% |
| 3B | int8 this stage | 5 | 39.38 | 6% | no | 1.94x | 12.70 | 32.2% |
| 3B | int8 this stage | 7 | 44.74 | 10% | no | 2.21x | 12.70 | 28.4% |

**The padded fp16 head.** The one-token pass of fp16 with the head computed on two rows and as one row:

| model | head padded (ms) | head not padded (ms) | padding saves |
| --- | ---: | ---: | ---: |
| 0.5B | 8.719 | 9.033 | 3.5% |
| 1.5B | 19.410 | 19.806 | 2.0% |
| 3B | 35.608 | 36.028 | 1.2% |

Gate verdict (written before the run: go is a worst `v` of at most 1.30 for a pass of 5 tokens at the targets, partial is below 1.70): **no-go**. The worst `v` of the new kernel is 1.94; per target, fp16 1.5B 1.03, 3B 1.01; int8 with the Stage 10 kernel 1.5B 2.27, 3B 2.47; int8 with the new kernel 1.5B 1.83, 3B 1.94.

<a id="stage11-plain"></a>

### Plain decoding in a graph

Plain greedy decoding of 96 tokens, batch 1, in the graph decoder, median of 5 runs taken in turns. The decode rate is the tokens after the first over the sum of the per-token times (prefill excluded). The padded head computes a one-row fp16 output projection on two rows. The output of each run is compared with the eager output of the same weights; a difference must be an fp16 tie, and the table gives the target's own logit gap there.

| model | setup | decode (tok/s) | spread | per token (ms) | output |
| --- | --- | ---: | ---: | ---: | --- |
| 0.5B | fp16 graph, head not padded | 112.7 | 0% | 8.88 | identical |
| 0.5B | fp16 graph | 117.0 | 1% | 8.55 | identical |
| 0.5B | int8 graph | 163.0 | 0% | 6.13 | identical |
| 1.5B | fp16 graph, head not padded | 49.7 | 1% | 20.11 | identical |
| 1.5B | fp16 graph | 51.6 | 1% | 19.38 | identical |
| 1.5B | int8 graph | 86.0 | 0% | 11.63 | identical |
| 3B | fp16 graph, head not padded | 27.8 | 7% | 35.95 | identical |
| 3B | fp16 graph | 28.1 | 3% | 35.56 | identical |
| 3B | int8 graph | 50.6 | 0% | 19.78 | identical |

**What the fair baseline changes** (ratios of the decode rates above):

| model | padded head / not padded (fp16) | int8 graph / fp16 graph | int8 graph / fp16 graph, head not padded |
| --- | ---: | ---: | ---: |
| 0.5B | 1.04x | 1.39x | 1.45x |
| 1.5B | 1.04x | 1.67x | 1.73x |
| 3B | 1.01x | 1.80x | 1.82x |

Every graph output equals the eager output of the same weights or differs at a near-tie: yes.

<a id="stage11-speculation"></a>

### Speculation in the graph

**Speedup.** Each row is a setup (the weights, the engine, then the guesser: the n-gram lookup, which costs nothing) on three workloads at each draft length. Engine `eager` is the engine of the earlier stages, `graph` is `GraphDecoder`. Every speedup is a ratio of tokens per second of whole generations (prefill included) against the plain decoding of fp16 in the eager engine, of fp16 in a graph and of int8 in a graph, each the median of several timings. `v` is the mean time of a round over the mean plain step of the same weights and engine (rounds with no draft are one-token passes and count), and `predicted` is the cost model's speedup over the row's own plain decoding from its acceptance, `v` and guess time.

| target | workload | setup | k | alpha | tokens per round | rounds with a draft | v | check or step (ms) | vs fp16 eager | vs fp16 graph | vs int8 graph | predicted | identical to plain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.5B | copy | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 26.4 | 1.00x | 0.72x | 0.46x | n/a | 3 of 3 |
| 1.5B | copy | fp16 eager n-gram | 4 | 0.79 | 2.48 | 72% | 1.10 | 29.0 | 2.22x | 1.60x | 1.01x | 3.00x | 3 of 3 |
| 1.5B | copy | fp16 eager n-gram | 6 | 0.80 | 2.71 | 70% | 1.09 | 28.8 | 2.44x | 1.76x | 1.11x | 3.62x | 3 of 3 |
| 1.5B | code | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 26.7 | 1.00x | 0.72x | 0.45x | n/a | 3 of 3 |
| 1.5B | code | fp16 eager n-gram | 4 | 0.70 | 2.18 | 68% | 1.07 | 28.7 | 1.98x | 1.43x | 0.90x | 2.59x | 3 of 3 |
| 1.5B | code | fp16 eager n-gram | 6 | 0.71 | 2.36 | 66% | 1.06 | 28.4 | 2.16x | 1.56x | 0.98x | 2.94x | 3 of 3 |
| 1.5B | open | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 26.5 | 1.00x | 0.72x | 0.45x | n/a | 3 of 3 |
| 1.5B | open | fp16 eager n-gram | 4 | 0.29 | 1.14 | 37% | 1.05 | 27.8 | 1.09x | 0.79x | 0.49x | 1.34x | 3 of 3 |
| 1.5B | open | fp16 eager n-gram | 6 | 0.30 | 1.15 | 37% | 1.07 | 28.4 | 1.07x | 0.77x | 0.48x | 1.33x | 2 of 3 |
| 1.5B | copy | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.0 | 1.39x | 1.00x | 0.63x | n/a | 3 of 3 |
| 1.5B | copy | fp16 graph n-gram | 4 | 0.79 | 2.48 | 72% | 1.04 | 19.7 | 3.21x | 2.31x | 1.47x | 3.17x | 3 of 3 |
| 1.5B | copy | fp16 graph n-gram | 6 | 0.80 | 2.71 | 70% | 1.05 | 19.8 | 3.49x | 2.52x | 1.60x | 3.78x | 3 of 3 |
| 1.5B | code | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.0 | 1.39x | 1.00x | 0.63x | n/a | 3 of 3 |
| 1.5B | code | fp16 graph n-gram | 4 | 0.72 | 2.18 | 69% | 1.04 | 19.8 | 2.84x | 2.04x | 1.29x | 2.75x | 3 of 3 |
| 1.5B | code | fp16 graph n-gram | 6 | 0.71 | 2.36 | 67% | 1.05 | 19.9 | 3.05x | 2.20x | 1.39x | 3.02x | 3 of 3 |
| 1.5B | open | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.1 | 1.38x | 1.00x | 0.62x | n/a | 3 of 3 |
| 1.5B | open | fp16 graph n-gram | 4 | 0.30 | 1.14 | 38% | 1.02 | 19.5 | 1.54x | 1.11x | 0.69x | 1.39x | 3 of 3 |
| 1.5B | open | fp16 graph n-gram | 6 | 0.31 | 1.15 | 37% | 1.02 | 19.5 | 1.55x | 1.12x | 0.70x | 1.41x | 2 of 3 |
| 1.5B | copy | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.6 | 2.19x | 1.58x | 1.00x | n/a | 3 of 3 |
| 1.5B | copy | int8 graph n-gram | 4 | 0.79 | 2.52 | 74% | 1.63 | 18.9 | 3.27x | 2.36x | 1.50x | 2.01x | 3 of 3 |
| 1.5B | copy | int8 graph n-gram | 6 | 0.80 | 2.77 | 72% | 1.71 | 19.8 | 3.40x | 2.45x | 1.55x | 2.30x | 3 of 3 |
| 1.5B | code | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.6 | 2.20x | 1.58x | 1.00x | n/a | 3 of 3 |
| 1.5B | code | int8 graph n-gram | 4 | 0.71 | 2.13 | 66% | 1.56 | 18.0 | 2.93x | 2.11x | 1.33x | 1.83x | 2 of 3 |
| 1.5B | code | int8 graph n-gram | 6 | 0.71 | 2.28 | 64% | 1.63 | 18.9 | 2.99x | 2.15x | 1.36x | 1.91x | 2 of 3 |
| 1.5B | open | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.5 | 2.22x | 1.61x | 1.00x | n/a | 3 of 3 |
| 1.5B | open | int8 graph n-gram | 4 | 0.36 | 1.19 | 40% | 1.33 | 15.4 | 1.98x | 1.43x | 0.89x | 1.17x | 2 of 3 |
| 1.5B | open | int8 graph n-gram | 6 | 0.38 | 1.20 | 39% | 1.38 | 15.9 | 1.94x | 1.40x | 0.87x | 1.18x | 3 of 3 |
| 1.5B | copy | int8 Stage 10 kernel graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.6 | 2.19x | 1.58x | 1.00x | n/a | 3 of 3 |
| 1.5B | copy | int8 Stage 10 kernel graph n-gram | 4 | 0.79 | 2.52 | 74% | 1.94 | 22.5 | 2.79x | 2.01x | 1.27x | 1.69x | 3 of 3 |
| 1.5B | copy | int8 Stage 10 kernel graph n-gram | 6 | 0.80 | 2.77 | 72% | 1.96 | 22.7 | 3.01x | 2.17x | 1.38x | 2.01x | 3 of 3 |
| 1.5B | code | int8 Stage 10 kernel graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.6 | 2.20x | 1.58x | 1.00x | n/a | 3 of 3 |
| 1.5B | code | int8 Stage 10 kernel graph n-gram | 4 | 0.71 | 2.13 | 66% | 1.83 | 21.2 | 2.52x | 1.82x | 1.15x | 1.56x | 2 of 3 |
| 1.5B | code | int8 Stage 10 kernel graph n-gram | 6 | 0.71 | 2.28 | 64% | 1.86 | 21.5 | 2.66x | 1.91x | 1.21x | 1.68x | 2 of 3 |
| 1.5B | open | int8 Stage 10 kernel graph plain | n/a | n/a | 1.00 | n/a | n/a | 11.5 | 2.22x | 1.61x | 1.00x | n/a | 3 of 3 |
| 1.5B | open | int8 Stage 10 kernel graph n-gram | 4 | 0.36 | 1.19 | 40% | 1.49 | 17.1 | 1.78x | 1.29x | 0.80x | 1.05x | 2 of 3 |
| 1.5B | open | int8 Stage 10 kernel graph n-gram | 6 | 0.38 | 1.20 | 39% | 1.51 | 17.4 | 1.77x | 1.28x | 0.80x | 1.07x | 2 of 3 |
| 3B | copy | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 45.4 | 1.00x | 0.85x | 0.50x | n/a | 3 of 3 |
| 3B | copy | fp16 eager n-gram | 4 | 0.99 | 4.52 | 92% | 0.84 | 38.1 | 4.70x | 4.00x | 2.33x | 5.81x | 3 of 3 |
| 3B | copy | fp16 eager n-gram | 6 | 0.99 | 5.94 | 90% | 0.84 | 38.3 | 6.03x | 5.12x | 2.98x | 7.99x | 3 of 3 |
| 3B | code | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 35.4 | 1.00x | 0.98x | 0.57x | n/a | 3 of 3 |
| 3B | code | fp16 eager n-gram | 4 | 0.64 | 1.95 | 67% | 1.05 | 37.1 | 1.86x | 1.83x | 1.07x | 2.38x | 3 of 3 |
| 3B | code | fp16 eager n-gram | 6 | 0.65 | 2.08 | 65% | 1.05 | 37.2 | 1.97x | 1.93x | 1.13x | 2.58x | 3 of 3 |
| 3B | open | fp16 eager plain | n/a | n/a | 1.00 | n/a | n/a | 39.3 | 1.00x | 0.91x | 0.52x | n/a | 3 of 3 |
| 3B | open | fp16 eager n-gram | 4 | 0.21 | 1.09 | 34% | 0.92 | 36.2 | 1.17x | 1.06x | 0.61x | 1.37x | 2 of 3 |
| 3B | open | fp16 eager n-gram | 6 | 0.21 | 1.09 | 34% | 0.92 | 36.1 | 1.18x | 1.06x | 0.62x | 1.37x | 2 of 3 |
| 3B | copy | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 35.6 | 1.18x | 1.00x | 0.58x | n/a | 3 of 3 |
| 3B | copy | fp16 graph n-gram | 4 | 0.99 | 4.52 | 92% | 1.03 | 36.5 | 4.93x | 4.19x | 2.44x | 4.74x | 3 of 3 |
| 3B | copy | fp16 graph n-gram | 6 | 0.98 | 5.94 | 90% | 1.04 | 36.9 | 6.27x | 5.33x | 3.10x | 6.41x | 3 of 3 |
| 3B | code | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 35.5 | 1.02x | 1.00x | 0.59x | n/a | 3 of 3 |
| 3B | code | fp16 graph n-gram | 4 | 0.65 | 1.95 | 67% | 1.02 | 36.1 | 1.92x | 1.88x | 1.10x | 2.47x | 3 of 3 |
| 3B | code | fp16 graph n-gram | 6 | 0.65 | 2.08 | 65% | 1.02 | 36.3 | 2.03x | 1.99x | 1.17x | 2.64x | 3 of 3 |
| 3B | open | fp16 graph plain | n/a | n/a | 1.00 | n/a | n/a | 35.5 | 1.10x | 1.00x | 0.58x | n/a | 3 of 3 |
| 3B | open | fp16 graph n-gram | 4 | 0.21 | 1.09 | 34% | 1.00 | 35.7 | 1.19x | 1.08x | 0.63x | 1.25x | 2 of 3 |
| 3B | open | fp16 graph n-gram | 6 | 0.21 | 1.09 | 34% | 1.01 | 35.8 | 1.19x | 1.08x | 0.62x | 1.25x | 2 of 3 |
| 3B | copy | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.8 | 2.02x | 1.72x | 1.00x | n/a | 3 of 3 |
| 3B | copy | int8 graph n-gram | 4 | 0.99 | 4.52 | 92% | 1.87 | 37.0 | 4.47x | 3.80x | 2.21x | 2.60x | 3 of 3 |
| 3B | copy | int8 graph n-gram | 6 | 0.98 | 5.94 | 90% | 2.08 | 41.2 | 5.14x | 4.37x | 2.54x | 3.21x | 3 of 3 |
| 3B | code | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.8 | 1.74x | 1.70x | 1.00x | n/a | 3 of 3 |
| 3B | code | int8 graph n-gram | 4 | 0.64 | 1.89 | 65% | 1.60 | 31.7 | 2.02x | 1.98x | 1.16x | 1.54x | 3 of 3 |
| 3B | code | int8 graph n-gram | 6 | 0.64 | 1.97 | 63% | 1.75 | 34.6 | 1.94x | 1.90x | 1.11x | 1.50x | 3 of 3 |
| 3B | open | int8 graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.7 | 1.91x | 1.73x | 1.00x | n/a | 3 of 3 |
| 3B | open | int8 graph n-gram | 4 | 0.26 | 1.13 | 38% | 1.36 | 26.7 | 1.60x | 1.45x | 0.84x | 1.00x | 2 of 3 |
| 3B | open | int8 graph n-gram | 6 | 0.26 | 1.11 | 38% | 1.45 | 28.6 | 1.48x | 1.34x | 0.78x | 0.93x | 2 of 3 |
| 3B | copy | int8 Stage 10 kernel graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.8 | 2.03x | 1.72x | 1.00x | n/a | 3 of 3 |
| 3B | copy | int8 Stage 10 kernel graph n-gram | 4 | 0.99 | 4.52 | 92% | 2.28 | 45.0 | 3.77x | 3.21x | 1.87x | 2.14x | 3 of 3 |
| 3B | copy | int8 Stage 10 kernel graph n-gram | 6 | 0.98 | 5.94 | 90% | 2.34 | 46.3 | 4.66x | 3.96x | 2.31x | 2.85x | 3 of 3 |
| 3B | code | int8 Stage 10 kernel graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.8 | 1.74x | 1.70x | 1.00x | n/a | 3 of 3 |
| 3B | code | int8 Stage 10 kernel graph n-gram | 4 | 0.64 | 1.89 | 65% | 1.89 | 37.4 | 1.73x | 1.70x | 1.00x | 1.30x | 3 of 3 |
| 3B | code | int8 Stage 10 kernel graph n-gram | 6 | 0.64 | 1.97 | 63% | 1.93 | 38.2 | 1.76x | 1.73x | 1.01x | 1.36x | 3 of 3 |
| 3B | open | int8 Stage 10 kernel graph plain | n/a | n/a | 1.00 | n/a | n/a | 19.7 | 1.91x | 1.73x | 1.00x | n/a | 3 of 3 |
| 3B | open | int8 Stage 10 kernel graph n-gram | 4 | 0.48 | 1.31 | 42% | 1.58 | 31.1 | 1.60x | 1.45x | 0.84x | 1.18x | 1 of 3 |
| 3B | open | int8 Stage 10 kernel graph n-gram | 6 | 0.48 | 1.34 | 41% | 1.60 | 31.5 | 1.62x | 1.46x | 0.85x | 1.20x | 1 of 3 |

**Do int8 and speculation multiply?** All in the graph decoder, against fp16 plain decoding in a graph (with the padded head). `int8 alone` is the gain of int8 plain, `n-gram alone` the gain of the lookup on fp16, `product` their product, `both` what int8 plus the lookup measured with the new kernel and `both (Stage 10 kernel)` with the old one in the same session, `new kernel gain` their ratio, `both / product` how much of the product survived, and `v` the checking cost of each setup. The last two columns are int8 plus the lookup against fp16 plus the lookup in a graph (above one, int8 wins even though both have the lookup) and against fp16 plain decoding in the eager engine.

| target | workload | k | int8 alone | n-gram alone (fp16) | n-gram alone (int8) | product | both | both (Stage 10 kernel) | new kernel gain | both / product | v fp16 | v int8 | v int8 (Stage 10 kernel) | vs fp16 graph + n-gram | vs fp16 eager plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | copy | 4 | 1.58x | 2.31x | 1.50x | 3.65x | 2.36x | 2.01x | 1.17x | 65% | 1.04 | 1.63 | 1.94 | 1.02x | 3.27x |
| 1.5B | copy | 6 | 1.58x | 2.52x | 1.55x | 3.96x | 2.45x | 2.17x | 1.13x | 62% | 1.05 | 1.71 | 1.96 | 0.97x | 3.40x |
| 1.5B | code | 4 | 1.58x | 2.04x | 1.33x | 3.24x | 2.11x | 1.82x | 1.16x | 65% | 1.04 | 1.56 | 1.83 | 1.03x | 2.93x |
| 1.5B | code | 6 | 1.58x | 2.20x | 1.36x | 3.48x | 2.15x | 1.91x | 1.12x | 62% | 1.05 | 1.63 | 1.86 | 0.98x | 2.99x |
| 1.5B | open | 4 | 1.61x | 1.11x | 0.89x | 1.79x | 1.43x | 1.29x | 1.11x | 80% | 1.02 | 1.33 | 1.49 | 1.28x | 1.98x |
| 1.5B | open | 6 | 1.61x | 1.12x | 0.87x | 1.80x | 1.40x | 1.28x | 1.09x | 78% | 1.02 | 1.38 | 1.51 | 1.25x | 1.94x |
| 3B | copy | 4 | 1.72x | 4.19x | 2.21x | 7.19x | 3.80x | 3.21x | 1.18x | 53% | 1.03 | 1.87 | 2.28 | 0.91x | 4.47x |
| 3B | copy | 6 | 1.72x | 5.33x | 2.54x | 9.14x | 4.37x | 3.96x | 1.10x | 48% | 1.04 | 2.08 | 2.34 | 0.82x | 5.14x |
| 3B | code | 4 | 1.70x | 1.88x | 1.16x | 3.21x | 1.98x | 1.70x | 1.17x | 62% | 1.02 | 1.60 | 1.89 | 1.05x | 2.02x |
| 3B | code | 6 | 1.70x | 1.99x | 1.11x | 3.39x | 1.90x | 1.73x | 1.10x | 56% | 1.02 | 1.75 | 1.93 | 0.95x | 1.94x |
| 3B | open | 4 | 1.73x | 1.08x | 0.84x | 1.87x | 1.45x | 1.45x | 1.00x | 77% | 1.00 | 1.36 | 1.58 | 1.34x | 1.60x |
| 3B | open | 6 | 1.73x | 1.08x | 0.78x | 1.86x | 1.34x | 1.46x | 0.92x | 72% | 1.01 | 1.45 | 1.60 | 1.25x | 1.48x |

**Exactness**, per setup against the same weights' own plain greedy output (first prompt of each workload, 32 tokens). At the first difference the table gives the target's own logit gap between the two tokens (a small gap is an fp16 tie). The fp16 eager row is the control: near-ties appear without any graph or int8.

| target | setup | workload | first differing token (0-based) | logit gap |
| --- | --- | --- | ---: | ---: |
| 1.5B | fp16 eager n-gram (control) | copy | none (identical) | n/a |
| 1.5B | fp16 eager n-gram (control) | code | none (identical) | n/a |
| 1.5B | fp16 eager n-gram (control) | open | none (identical) | n/a |
| 1.5B | fp16 graph n-gram | copy | none (identical) | n/a |
| 1.5B | fp16 graph n-gram | code | none (identical) | n/a |
| 1.5B | fp16 graph n-gram | open | none (identical) | n/a |
| 1.5B | int8 graph n-gram | copy | none (identical) | n/a |
| 1.5B | int8 graph n-gram | code | 2.0 | 0.008 |
| 1.5B | int8 graph n-gram | open | none (identical) | n/a |
| 1.5B | int8 graph plain | copy | none (identical) | n/a |
| 1.5B | int8 graph plain | code | none (identical) | n/a |
| 1.5B | int8 graph plain | open | none (identical) | n/a |
| 3B | fp16 eager n-gram (control) | copy | none (identical) | n/a |
| 3B | fp16 eager n-gram (control) | code | none (identical) | n/a |
| 3B | fp16 eager n-gram (control) | open | none (identical) | n/a |
| 3B | fp16 graph n-gram | copy | none (identical) | n/a |
| 3B | fp16 graph n-gram | code | none (identical) | n/a |
| 3B | fp16 graph n-gram | open | none (identical) | n/a |
| 3B | int8 graph n-gram | copy | none (identical) | n/a |
| 3B | int8 graph n-gram | code | none (identical) | n/a |
| 3B | int8 graph n-gram | open | none (identical) | n/a |
| 3B | int8 graph plain | copy | none (identical) | n/a |
| 3B | int8 graph plain | code | none (identical) | n/a |
| 3B | int8 graph plain | open | none (identical) | n/a |

Every difference is a near-tie: yes.

**A control that must fail**: the same decoder with a mask that lets every row see all keys up to the end of its block (so the rollback and the causal mask are both wrong).

| target | workload | output differs from plain greedy |
| --- | --- | --- |
| 1.5B | copy | yes |
| 1.5B | code | yes |
| 1.5B | open | yes |

The control changes the output (so the exactness check can fail): yes.

<a id="stage11-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage11_block_cost.png` | Cost of a checking pass of 5 and 7 tokens relative to a one-token pass in a CUDA graph: fp16, int8 with the Stage 10 kernel and int8 with the row-exact kernels, by model |
| `stage11_layer.png` | Speed of each int8 kernel (the Stage 10 kernel, the two row-exact kernels, the tuned winner) against the fp16 layer on the large layers, timed in turns in a CUDA graph, by rows of x |
| `stage11_multiply.png` | Gain of int8 plus the n-gram lookup with each kernel against the product of the two separate gains, by target and workload |
| `stage11_plain_rate.png` | Decode rate of plain greedy decoding in a graph, fp16 with and without the padded head and int8, by model |
| `stage11_speedup_1p5b.png` | Best speedup over fp16 eager plain decoding by workload and setup, 1.5B target |
| `stage11_speedup_3b.png` | Best speedup over fp16 eager plain decoding by workload and setup, 3B target |

## Stage 12: The Engine

Source: [`results/stage12.json`](results/stage12.json)

<a id="stage12-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
- **Model revision:** Qwen2.5-0.5B 060db649; Qwen2.5-1.5B 8faed761
- **max_pos (RoPE table):** 4096
- **Requests per workload / seed / quick run:** 64 / 0 / False

<a id="stage12-parity"></a>

### Is every mode exact?

Greedy output of the engine in each mode against plain cached decoding (`generate_cached`) of the same weights, 48 new tokens per prompt, prompts of different lengths. A difference counts as an fp16 near-tie only if the logit gap between the two tokens at the first difference is below 0.5; anything else is a bug and stops the notebook. The sampled run is a control that must differ.

| model | mode | prompts | identical | near ties | other | graph mode | spec rounds | drafted | accepted |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0.5B | eager | 12 | 12 | 0 | 0 | none | 0 | 0 | 0 |
| 0.5B | static | 12 | 11 | 1 | 0 | static | 0 | 0 | 0 |
| 0.5B | graph | 12 | 11 | 1 | 0 | graph | 0 | 0 | 0 |
| 0.5B | graph + lookup k=4 | 12 | 12 | 0 | 0 | graph | 165 | 448 | 299 |
| 0.5B | sampled (control) | 12 | 1 | 1 | 10 | n/a | n/a | n/a | n/a |
| 1.5B | eager | 12 | 12 | 0 | 0 | none | 0 | 0 | 0 |
| 1.5B | static | 12 | 12 | 0 | 0 | static | 0 | 0 | 0 |
| 1.5B | graph | 12 | 12 | 0 | 0 | graph | 0 | 0 | 0 |
| 1.5B | graph + lookup k=4 | 12 | 12 | 0 | 0 | graph | 170 | 504 | 256 |
| 1.5B | sampled (control) | 12 | 1 | 1 | 10 | n/a | n/a | n/a | n/a |

<a id="stage12-modes"></a>

### The decode paths under load

The same trace (64 requests of the `mixed` workload at 4.0 requests per second) served by the engine with each decode path. `static` is the control: the bucketed step without capture. Capture happens before the timed run and its cost is shown on its own. Against the eager path, a TPOT ratio below 1 and a tok/s ratio above 1 are faster.

| model | decode path | req/s | tok/s | TTFT p50 (s) | TTFT p99 (s) | TPOT p50 (ms) | TPOT p99 (ms) | goodput (tok/s) | capture (s) | graphs | TPOT vs eager | tok/s vs eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0.5B | eager | 3.06 | 205 | 0.05 | 0.09 | 31.6 | 42.1 | 205 | 0.0 | none | 1.00 | 1.00 |
| 0.5B | static | 3.05 | 204 | 0.05 | 0.10 | 31.8 | 43.3 | 204 | 0.0 | static | 1.01 | 1.00 |
| 0.5B | graph | 3.45 | 231 | 0.03 | 0.08 | 17.6 | 39.1 | 231 | 4.1 | graph | 0.56 | 1.13 |
| 1.5B | eager | 2.81 | 188 | 0.07 | 0.21 | 43.2 | 76.3 | 188 | 0.0 | none | 1.00 | 1.00 |
| 1.5B | static | 2.25 | 151 | 0.10 | 0.31 | 98.7 | 150.5 | 114 | 0.0 | static | 2.29 | 0.80 |
| 1.5B | graph | 2.29 | 154 | 0.09 | 0.26 | 98.6 | 146.9 | 117 | 5.2 | graph | 2.28 | 0.82 |

<a id="stage12-load"></a>

### The engine against Hugging Face

The same 64-request traces at each arrival rate, served open loop by the engine (eager and graph decode) and by Hugging Face `generate` (one request at a time, and static batches of 8). TTFT is time to first token and TPOT the time per output token after the first, as a reader of the stream sees them. A request meets the latency target if its first token comes within 2.0 s and its tokens then come at least every 0.1 s; goodput counts only the tokens of requests that meet it. The Hugging Face baseline is generous: a request counts as finished at its own last token.

| model | rate (req/s) | system | req/s | tok/s | TTFT p50 (s) | TTFT p99 (s) | TPOT p50 (ms) | stall p99 (s) | met target | goodput (tok/s) | errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 1.0 | engine eager | 0.99 | 66 | 0.04 | 0.08 | 28.5 | 0.06 | 100% | 66 | 0 |
| 0.5B | 1.0 | engine graph | 1.05 | 70 | 0.03 | 0.06 | 10.9 | 0.04 | 100% | 70 | 0 |
| 0.5B | 1.0 | HF sequential | 0.45 | 30 | 39.01 | 80.63 | 32.4 | 0.04 | 6% | 2 | 0 |
| 0.5B | 1.0 | HF static batch 8 | 0.90 | 60 | 4.81 | 14.75 | 31.7 | 0.04 | 22% | 18 | 0 |
| 0.5B | 2.0 | engine graph | 2.00 | 134 | 0.03 | 0.06 | 12.3 | 0.05 | 100% | 134 | 0 |
| 0.5B | 2.0 | HF sequential | 0.45 | 30 | 51.86 | 110.09 | 32.6 | 0.04 | 6% | 2 | 0 |
| 0.5B | 2.0 | HF static batch 8 | 1.14 | 77 | 12.66 | 20.89 | 31.3 | 0.04 | 6% | 4 | 0 |
| 0.5B | 2.0 | engine eager | 1.80 | 121 | 0.05 | 0.09 | 29.9 | 0.06 | 100% | 121 | 0 |
| 0.5B | 4.0 | HF sequential | 0.46 | 31 | 58.27 | 124.57 | 32.4 | 0.04 | 6% | 2 | 0 |
| 0.5B | 4.0 | HF static batch 8 | 1.47 | 98 | 14.60 | 27.88 | 31.8 | 0.04 | 11% | 8 | 0 |
| 0.5B | 4.0 | engine eager | 3.06 | 205 | 0.04 | 0.11 | 31.5 | 0.08 | 100% | 205 | 0 |
| 0.5B | 4.0 | engine graph | 3.46 | 231 | 0.03 | 0.08 | 17.5 | 0.06 | 100% | 231 | 0 |
| 0.5B | 8.0 | HF static batch 8 | 1.16 | 78 | 19.87 | 41.10 | 31.4 | 0.04 | 14% | 7 | 0 |
| 0.5B | 8.0 | engine eager | 4.58 | 307 | 0.05 | 0.10 | 34.6 | 0.09 | 100% | 307 | 0 |
| 0.5B | 8.0 | engine graph | 4.56 | 305 | 0.06 | 0.12 | 50.0 | 0.12 | 100% | 305 | 0 |
| 0.5B | 8.0 | HF sequential | 0.46 | 31 | 61.55 | 131.76 | 32.5 | 0.04 | 5% | 1 | 0 |
| 1.5B | 1.0 | engine eager | 0.97 | 65 | 0.06 | 0.15 | 34.6 | 0.09 | 100% | 65 | 0 |
| 1.5B | 1.0 | engine graph | 1.00 | 67 | 0.05 | 0.15 | 24.9 | 0.08 | 100% | 67 | 0 |
| 1.5B | 1.0 | HF sequential | 0.38 | 26 | 51.00 | 106.08 | 38.3 | 0.04 | 6% | 1 | 0 |
| 1.5B | 1.0 | HF static batch 8 | 0.84 | 56 | 5.75 | 14.83 | 36.9 | 0.04 | 16% | 7 | 0 |
| 1.5B | 2.0 | engine graph | 1.71 | 115 | 0.06 | 0.21 | 35.9 | 0.14 | 100% | 115 | 0 |
| 1.5B | 2.0 | HF sequential | 0.39 | 26 | 62.68 | 133.79 | 37.9 | 0.04 | 6% | 1 | 0 |
| 1.5B | 2.0 | HF static batch 8 | 1.05 | 70 | 11.64 | 23.89 | 36.7 | 0.04 | 9% | 5 | 0 |
| 1.5B | 2.0 | engine eager | 1.74 | 117 | 0.06 | 0.16 | 36.8 | 0.12 | 100% | 117 | 0 |
| 1.5B | 4.0 | HF sequential | 0.39 | 26 | 69.37 | 148.46 | 38.0 | 0.04 | 5% | 1 | 0 |
| 1.5B | 4.0 | HF static batch 8 | 1.19 | 80 | 21.21 | 37.78 | 36.5 | 0.04 | 11% | 6 | 0 |
| 1.5B | 4.0 | engine eager | 2.76 | 185 | 0.07 | 0.20 | 44.3 | 0.14 | 100% | 185 | 0 |
| 1.5B | 4.0 | engine graph | 2.29 | 154 | 0.11 | 0.26 | 97.4 | 0.24 | 52% | 117 | 0 |
| 1.5B | 8.0 | HF static batch 8 | 0.93 | 63 | 25.68 | 52.87 | 37.4 | 0.05 | 9% | 3 | 0 |
| 1.5B | 8.0 | engine eager | 3.76 | 252 | 0.11 | 0.39 | 83.1 | 0.26 | 86% | 237 | 0 |
| 1.5B | 8.0 | engine graph | 2.52 | 169 | 0.33 | 2.55 | 145.6 | 0.36 | 19% | 82 | 0 |
| 1.5B | 8.0 | HF sequential | 0.39 | 26 | 72.84 | 156.90 | 38.2 | 0.04 | 3% | 0 | 0 |

<a id="stage12-capacity"></a>

### Capacity under the latency target

The highest tested arrival rate at which at least 90% of the requests met the latency target ("none" means no tested rate did). A finite trace understates capacity, so this is a floor between tested rates, not an exact limit.

| model | system | capacity (req/s) |
| --- | --- | ---: |
| 0.5B | engine eager | 8.0 |
| 0.5B | engine graph | 8.0 |
| 0.5B | HF sequential | none |
| 0.5B | HF static batch 8 | none |
| 1.5B | engine eager | 4.0 |
| 1.5B | engine graph | 2.0 |
| 1.5B | HF sequential | none |
| 1.5B | HF static batch 8 | none |

<a id="stage12-latency"></a>

### Latency mode

One request at a time on the three text workloads of Stage 6, 128 new tokens each, median of 3 repeats per prompt. `accept rate` is accepted drafts over drafted tokens; `tokens/step` is the tokens made per decode step. The outputs of the engine rows are checked against plain decoding (identical, or a near-tie).

| model | workload | system | TPOT (ms) | TTFT (ms) | vs eager engine | accept rate | tokens/step | identical / near tie / runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5B | copy | engine eager | 27.8 | 27 | 1.00 | n/a | 1.00 | 6 / 0 / 6 |
| 0.5B | code | engine eager | 27.8 | 27 | 1.00 | n/a | 1.00 | 6 / 0 / 6 |
| 0.5B | open | engine eager | 27.9 | 27 | 1.00 | n/a | 1.00 | 6 / 0 / 6 |
| 0.5B | copy | engine graph | 8.9 | 27 | 3.12 | n/a | 1.00 | 6 / 0 / 6 |
| 0.5B | code | engine graph | 8.8 | 26 | 3.15 | n/a | 1.00 | 6 / 0 / 6 |
| 0.5B | open | engine graph | 8.5 | 26 | 3.28 | n/a | 1.00 | 6 / 0 / 6 |
| 0.5B | copy | engine graph + lookup k=4 | 3.5 | 27 | 7.97 | 66% | 2.62 | 6 / 0 / 6 |
| 0.5B | code | engine graph + lookup k=4 | 3.2 | 27 | 8.67 | 59% | 2.82 | 6 / 0 / 6 |
| 0.5B | open | engine graph + lookup k=4 | 5.0 | 26 | 5.56 | 42% | 1.72 | 6 / 0 / 6 |
| 0.5B | copy | HF generate | 32.4 | 39 | 0.86 | n/a | n/a | 0 / 0 / 6 |
| 0.5B | code | HF generate | 32.8 | 39 | 0.85 | n/a | n/a | 0 / 0 / 6 |
| 0.5B | open | HF generate | 32.5 | 39 | 0.86 | n/a | n/a | 0 / 0 / 6 |
| 1.5B | copy | engine eager | 33.1 | 36 | 1.00 | n/a | 1.00 | 6 / 0 / 6 |
| 1.5B | code | engine eager | 33.1 | 36 | 1.00 | n/a | 1.00 | 6 / 0 / 6 |
| 1.5B | open | engine eager | 33.2 | 32 | 1.00 | n/a | 1.00 | 6 / 0 / 6 |
| 1.5B | copy | engine graph | 18.9 | 39 | 1.75 | n/a | 1.00 | 6 / 0 / 6 |
| 1.5B | code | engine graph | 18.8 | 38 | 1.76 | n/a | 1.00 | 6 / 0 / 6 |
| 1.5B | open | engine graph | 18.4 | 33 | 1.80 | n/a | 1.00 | 6 / 0 / 6 |
| 1.5B | copy | engine graph + lookup k=4 | 11.5 | 38 | 2.88 | 31% | 1.69 | 6 / 0 / 6 |
| 1.5B | code | engine graph + lookup k=4 | 8.8 | 37 | 3.76 | 45% | 2.21 | 6 / 0 / 6 |
| 1.5B | open | engine graph + lookup k=4 | 17.0 | 32 | 1.95 | 7% | 1.10 | 6 / 0 / 6 |
| 1.5B | copy | HF generate | 38.5 | 47 | 0.86 | n/a | n/a | 0 / 0 / 6 |
| 1.5B | code | HF generate | 38.1 | 46 | 0.87 | n/a | n/a | 0 / 0 / 6 |
| 1.5B | open | HF generate | 38.1 | 45 | 0.87 | n/a | n/a | 0 / 0 / 6 |

<a id="stage12-server"></a>

### The server over HTTP

The engine (graph decode) behind `python -m minivllm.server` in its own process, driven over HTTP by the same client and the same trace as the in-process row (0.5B, 4.0 requests per second). The server was ready 16 s after it was started (model load, one warm-up request, graph capture).

| system | ok | errors | req/s | tok/s | TTFT p50 (s) | TTFT p99 (s) | TPOT p50 (ms) | TPOT p99 (ms) | goodput (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| engine graph over HTTP | 64 | 0 | 3.39 | 227 | 0.06 | 0.13 | 18.7 | 40.6 | 227 |
| engine graph in process | 64 | 0 | 3.46 | 231 | 0.03 | 0.08 | 17.5 | 39.3 | 231 |

A streamed sampled completion produced 25 server-sent events (the last one carries the finish reason and the usage).

<a id="stage12-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage12_goodput_0p5b.png` | Goodput (tokens of requests that met the latency target) against offered load, the engine and Hugging Face, 0.5B |
| `stage12_goodput_1p5b.png` | Goodput (tokens of requests that met the latency target) against offered load, the engine and Hugging Face, 1.5B |
| `stage12_latency_0p5b.png` | Time per token for one request at a time on three workloads: eager, graph, graph with lookup speculation, Hugging Face, 0.5B |
| `stage12_latency_1p5b.png` | Time per token for one request at a time on three workloads: eager, graph, graph with lookup speculation, Hugging Face, 1.5B |
| `stage12_modes_tpot.png` | Median time per output token at the middle arrival rate: eager paged decode, the bucketed static step, and the same step in CUDA graphs, by model |
| `stage12_modes_tpot_p99.png` | 99th percentile time per output token at the middle arrival rate, by decode path and model |
| `stage12_tpot_0p5b.png` | Median time per output token against offered load, the engine and Hugging Face, 0.5B |
| `stage12_tpot_1p5b.png` | Median time per output token against offered load, the engine and Hugging Face, 1.5B |
| `stage12_ttft_p99_0p5b.png` | Time to first token, 99th percentile, against offered load: the engine (eager and graph) and Hugging Face (sequential and static batches), 0.5B |
| `stage12_ttft_p99_1p5b.png` | Time to first token, 99th percentile, against offered load: the engine (eager and graph) and Hugging Face (sequential and static batches), 1.5B |

## Stage 13: The Batched Step

Source: [`results/stage13.json`](results/stage13.json)

<a id="stage13-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
- **Model revision:** Qwen2.5-0.5B 060db649; Qwen2.5-1.5B 8faed761
- **max_pos (RoPE table):** 4096
- **Requests per workload / seed / quick run:** 64 / 0 / False

<a id="stage13-parity"></a>

### Is every variant exact?

Greedy output of each variant against plain cached decoding of the same weights, 48 new tokens per prompt, requests of different lengths. A difference counts as an fp16 near-tie only if the logit gap at the first difference is below 0.5. The sampled run is a control that must differ. The fallback variants are run with a threshold of 2 so a batch of twelve starts on the eager path and ends on the graph path; `eager steps` and `graph steps` show that both ran.

| model | variant | prompts | identical | near ties | other | eager steps | graph steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | V0 expand, eager | 12 | 12 | 0 | 0 | 47 | 0 |
| 0.5B | V1 expand, graph | 12 | 11 | 1 | 0 | 0 | 42 |
| 0.5B | V2 fold, eager | 12 | 12 | 0 | 0 | 47 | 0 |
| 0.5B | V3 fold, graph | 12 | 11 | 1 | 0 | 0 | 42 |
| 0.5B | V4 expand, graph, fine buckets | 12 | 12 | 0 | 0 | 0 | 39 |
| 0.5B | V5 expand, graph, eager fallback | 12 | 12 | 0 | 0 | 45 | 1 |
| 0.5B | V6 all three levers | 12 | 12 | 0 | 0 | 45 | 1 |
| 0.5B | sampled (control) | 12 | 1 | 1 | 10 | n/a | n/a |
| 1.5B | V0 expand, eager | 12 | 12 | 0 | 0 | 47 | 0 |
| 1.5B | V1 expand, graph | 12 | 12 | 0 | 0 | 0 | 42 |
| 1.5B | V2 fold, eager | 12 | 12 | 0 | 0 | 47 | 0 |
| 1.5B | V3 fold, graph | 12 | 12 | 0 | 0 | 0 | 42 |
| 1.5B | V4 expand, graph, fine buckets | 12 | 12 | 0 | 0 | 0 | 39 |
| 1.5B | V5 expand, graph, eager fallback | 12 | 12 | 0 | 0 | 45 | 1 |
| 1.5B | V6 all three levers | 12 | 12 | 0 | 0 | 45 | 1 |
| 1.5B | sampled (control) | 12 | 1 | 1 | 10 | n/a | n/a |

<a id="stage13-bytes"></a>

### Where the bytes go

One layer's attention in a batched decode step, per path. `MB` is the counted traffic (the gather copies the keys and values twice, the expansion reads once and writes once per query head of the group, the attention reads what it is given); `ms` is the measured time per layer on random tensors of the real shapes inside a CUDA graph. `ratio` is the expanded path over the folded one.

| model | batch | blocks | MB expanded | MB folded | bytes ratio | ms expanded | ms folded | time ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 4 | 4 | 2 | 1 | 3.80 | 0.032 | 0.028 | 1.14 |
| 0.5B | 4 | 16 | 10 | 3 | 3.80 | 0.071 | 0.047 | 1.52 |
| 0.5B | 4 | 64 | 40 | 10 | 3.80 | 0.263 | 0.174 | 1.51 |
| 0.5B | 16 | 4 | 10 | 3 | 3.80 | 0.074 | 0.033 | 2.29 |
| 0.5B | 16 | 16 | 40 | 10 | 3.80 | 0.304 | 0.095 | 3.18 |
| 0.5B | 16 | 64 | 159 | 42 | 3.80 | 1.211 | 0.332 | 3.65 |
| 0.5B | 32 | 4 | 20 | 5 | 3.80 | 0.161 | 0.044 | 3.68 |
| 0.5B | 32 | 16 | 80 | 21 | 3.80 | 0.644 | 0.154 | 4.18 |
| 0.5B | 32 | 64 | 319 | 84 | 3.80 | 2.447 | 0.557 | 4.39 |
| 1.5B | 4 | 4 | 4 | 1 | 3.40 | 0.039 | 0.031 | 1.24 |
| 1.5B | 4 | 16 | 18 | 5 | 3.40 | 0.117 | 0.051 | 2.28 |
| 1.5B | 4 | 64 | 71 | 21 | 3.40 | 0.420 | 0.216 | 1.94 |
| 1.5B | 16 | 4 | 18 | 5 | 3.40 | 0.134 | 0.040 | 3.35 |
| 1.5B | 16 | 16 | 71 | 21 | 3.40 | 0.502 | 0.138 | 3.63 |
| 1.5B | 16 | 64 | 285 | 84 | 3.40 | 1.904 | 0.493 | 3.86 |
| 1.5B | 32 | 4 | 36 | 10 | 3.40 | 0.250 | 0.082 | 3.06 |
| 1.5B | 32 | 16 | 143 | 42 | 3.40 | 0.998 | 0.251 | 3.98 |
| 1.5B | 32 | 64 | 570 | 168 | 3.40 | 3.789 | 0.929 | 4.08 |

The two paths' outputs on a ragged batch differ by at most 0.00000 (0.5B, outputs up to 2.2), 0.00000 (1.5B, outputs up to 0.7).

<a id="stage13-profile"></a>

### One step under the profiler

One whole decode step (all layers and the weights) under the profiler, per path, milliseconds per step by kernel family.

| model | path | total (ms) | attention | matmul | copy and index | elementwise |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | expand | 31.8 | 7.5 | 6.4 | 16.1 | 1.8 |
| 0.5B | fold | 21.8 | 2.7 | 7.4 | 8.9 | 2.9 |
| 1.5B | expand | 71.9 | 11.7 | 18.9 | 38.3 | 3.0 |
| 1.5B | fold | 36.9 | 2.5 | 18.5 | 13.2 | 2.6 |

<a id="stage13-levers"></a>

### The levers under load

The Stage 12 traces (64 requests of the `mixed` workload per rate), each variant in its own engine, step log on. V0 expanded attention, eager step; V1 expanded, graph (the Stage 12 baseline); V2 folded, eager; V3 folded, graph; V4 expanded, graph, fine buckets; V5 V1 plus the eager fallback; V6 all three levers. `(repeat)` is V1 run again at the end of the session.

| model | rate (req/s) | variant | req/s | tok/s | TTFT p50 (s) | TPOT p50 (ms) | TPOT p99 (ms) | met target | capture (s) | errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 1.0 | V0 expand, eager | 0.95 | 64 | 0.06 | 42.1 | 48.4 | 100% | 0.0 | 0 |
| 0.5B | 2.0 | V0 expand, eager | 1.67 | 112 | 0.06 | 43.8 | 51.7 | 100% | 0.0 | 0 |
| 0.5B | 4.0 | V0 expand, eager | 2.67 | 179 | 0.07 | 48.5 | 70.9 | 100% | 0.0 | 0 |
| 0.5B | 8.0 | V0 expand, eager | 3.76 | 252 | 0.08 | 59.1 | 84.4 | 100% | 0.0 | 0 |
| 0.5B | 1.0 | V1 expand, graph | 1.05 | 70 | 0.04 | 11.5 | 16.5 | 100% | 6.0 | 0 |
| 0.5B | 2.0 | V1 expand, graph | 1.99 | 133 | 0.04 | 13.5 | 26.9 | 100% | 6.0 | 0 |
| 0.5B | 4.0 | V1 expand, graph | 3.35 | 225 | 0.05 | 19.9 | 44.8 | 100% | 6.0 | 0 |
| 0.5B | 8.0 | V1 expand, graph | 4.18 | 280 | 0.09 | 62.1 | 88.9 | 100% | 6.0 | 0 |
| 0.5B | 1.0 | V2 fold, eager | 0.94 | 63 | 0.06 | 43.8 | 47.7 | 100% | 0.0 | 0 |
| 0.5B | 2.0 | V2 fold, eager | 1.66 | 111 | 0.07 | 45.7 | 53.6 | 100% | 0.0 | 0 |
| 0.5B | 4.0 | V2 fold, eager | 2.63 | 176 | 0.07 | 50.9 | 68.8 | 100% | 0.0 | 0 |
| 0.5B | 8.0 | V2 fold, eager | 3.68 | 246 | 0.08 | 60.5 | 85.7 | 100% | 0.0 | 0 |
| 0.5B | 1.0 | V3 fold, graph | 1.05 | 70 | 0.04 | 10.5 | 14.4 | 100% | 5.9 | 0 |
| 0.5B | 2.0 | V3 fold, graph | 2.01 | 135 | 0.04 | 11.6 | 17.3 | 100% | 5.9 | 0 |
| 0.5B | 4.0 | V3 fold, graph | 3.68 | 247 | 0.04 | 13.5 | 26.1 | 100% | 5.9 | 0 |
| 0.5B | 8.0 | V3 fold, graph | 6.04 | 404 | 0.05 | 21.0 | 48.2 | 100% | 5.9 | 0 |
| 1.5B | 1.0 | V0 expand, eager | 0.93 | 62 | 0.08 | 51.2 | 62.3 | 100% | 0.0 | 0 |
| 1.5B | 2.0 | V0 expand, eager | 1.59 | 107 | 0.09 | 54.7 | 70.3 | 100% | 0.0 | 0 |
| 1.5B | 4.0 | V0 expand, eager | 2.45 | 164 | 0.09 | 61.4 | 98.0 | 100% | 0.0 | 0 |
| 1.5B | 8.0 | V0 expand, eager | 3.20 | 214 | 0.18 | 105.8 | 133.5 | 44% | 0.0 | 0 |
| 1.5B | 1.0 | V1 expand, graph | 0.99 | 66 | 0.07 | 26.5 | 53.5 | 100% | 7.2 | 0 |
| 1.5B | 2.0 | V1 expand, graph | 1.67 | 112 | 0.07 | 39.9 | 94.7 | 100% | 7.2 | 0 |
| 1.5B | 4.0 | V1 expand, graph | 2.16 | 145 | 0.13 | 106.1 | 165.6 | 48% | 7.2 | 0 |
| 1.5B | 8.0 | V1 expand, graph | 2.34 | 156 | 0.46 | 160.4 | 188.5 | 8% | 7.2 | 0 |
| 1.5B | 1.0 | V2 fold, eager | 0.92 | 62 | 0.08 | 52.2 | 62.0 | 100% | 0.0 | 0 |
| 1.5B | 2.0 | V2 fold, eager | 1.59 | 106 | 0.09 | 55.6 | 70.3 | 100% | 0.0 | 0 |
| 1.5B | 4.0 | V2 fold, eager | 2.44 | 163 | 0.10 | 62.2 | 94.5 | 100% | 0.0 | 0 |
| 1.5B | 8.0 | V2 fold, eager | 3.28 | 220 | 0.13 | 85.4 | 118.6 | 88% | 0.0 | 0 |
| 1.5B | 1.0 | V3 fold, graph | 1.01 | 67 | 0.06 | 23.1 | 34.2 | 100% | 6.9 | 0 |
| 1.5B | 2.0 | V3 fold, graph | 1.84 | 123 | 0.06 | 26.9 | 52.7 | 100% | 6.9 | 0 |
| 1.5B | 4.0 | V3 fold, graph | 2.99 | 200 | 0.08 | 38.3 | 77.3 | 100% | 6.9 | 0 |
| 1.5B | 8.0 | V3 fold, graph | 3.87 | 259 | 0.13 | 84.2 | 115.5 | 83% | 6.9 | 0 |
| 1.5B | 1.0 | V4 expand, graph, fine buckets | 1.00 | 67 | 0.06 | 23.6 | 34.5 | 100% | 47.5 | 0 |
| 1.5B | 2.0 | V4 expand, graph, fine buckets | 1.79 | 120 | 0.07 | 30.2 | 62.0 | 100% | 47.5 | 0 |
| 1.5B | 4.0 | V4 expand, graph, fine buckets | 2.75 | 184 | 0.08 | 48.0 | 92.0 | 100% | 47.5 | 0 |
| 1.5B | 8.0 | V4 expand, graph, fine buckets | 3.22 | 216 | 0.30 | 119.2 | 144.2 | 38% | 47.5 | 0 |
| 0.5B | 1.0 | V6 all three levers | 1.05 | 70 | 0.04 | 10.2 | 14.0 | 100% | 39.2 | 0 |
| 0.5B | 2.0 | V6 all three levers | 2.02 | 135 | 0.04 | 11.1 | 15.0 | 100% | 39.2 | 0 |
| 0.5B | 4.0 | V6 all three levers | 3.74 | 251 | 0.04 | 12.3 | 23.2 | 100% | 39.2 | 0 |
| 0.5B | 8.0 | V6 all three levers | 6.34 | 425 | 0.05 | 17.2 | 39.2 | 100% | 39.2 | 0 |
| 0.5B | 1.0 | V1 expand, graph (repeat) | 1.05 | 70 | 0.04 | 11.4 | 16.5 | 100% | 5.9 | 0 |
| 0.5B | 2.0 | V1 expand, graph (repeat) | 1.98 | 133 | 0.04 | 13.6 | 26.7 | 100% | 5.9 | 0 |
| 0.5B | 4.0 | V1 expand, graph (repeat) | 3.35 | 224 | 0.05 | 20.3 | 46.8 | 100% | 5.9 | 0 |
| 0.5B | 8.0 | V1 expand, graph (repeat) | 4.16 | 279 | 0.09 | 64.2 | 91.7 | 100% | 5.9 | 0 |
| 1.5B | 1.0 | V5 expand, graph, eager fallback | 0.99 | 66 | 0.06 | 26.7 | 53.5 | 100% | 7.3 | 0 |
| 1.5B | 2.0 | V5 expand, graph, eager fallback | 1.70 | 114 | 0.07 | 36.6 | 69.2 | 100% | 7.3 | 0 |
| 1.5B | 4.0 | V5 expand, graph, eager fallback | 2.60 | 174 | 0.09 | 58.0 | 96.0 | 100% | 7.3 | 0 |
| 1.5B | 8.0 | V5 expand, graph, eager fallback | 3.30 | 221 | 0.20 | 105.0 | 137.7 | 44% | 7.3 | 0 |
| 1.5B | 1.0 | V6 all three levers | 1.01 | 68 | 0.06 | 22.2 | 29.0 | 100% | 46.0 | 0 |
| 1.5B | 2.0 | V6 all three levers | 1.86 | 125 | 0.06 | 25.3 | 40.0 | 100% | 46.0 | 0 |
| 1.5B | 4.0 | V6 all three levers | 3.14 | 211 | 0.07 | 31.5 | 71.5 | 100% | 46.0 | 0 |
| 1.5B | 8.0 | V6 all three levers | 4.27 | 286 | 0.12 | 72.3 | 108.0 | 98% | 46.0 | 0 |
| 1.5B | 1.0 | V1 expand, graph (repeat) | 0.99 | 66 | 0.06 | 26.5 | 53.4 | 100% | 7.3 | 0 |
| 1.5B | 2.0 | V1 expand, graph (repeat) | 1.67 | 112 | 0.07 | 39.3 | 94.7 | 100% | 7.3 | 0 |
| 1.5B | 4.0 | V1 expand, graph (repeat) | 2.16 | 145 | 0.13 | 106.2 | 165.6 | 45% | 7.3 | 0 |
| 1.5B | 8.0 | V1 expand, graph (repeat) | 2.34 | 157 | 0.38 | 160.6 | 188.7 | 9% | 7.3 | 0 |

<a id="stage13-capacity"></a>

### Capacity under the latency target

The highest tested rate at which at least 90% of requests met the latency target ("none": no tested rate did).

| model | variant | capacity (req/s) |
| --- | --- | ---: |
| 0.5B | V0 expand, eager | 8.0 |
| 0.5B | V1 expand, graph | 8.0 |
| 0.5B | V2 fold, eager | 8.0 |
| 0.5B | V3 fold, graph | 8.0 |
| 1.5B | V0 expand, eager | 4.0 |
| 1.5B | V1 expand, graph | 2.0 |
| 1.5B | V2 fold, eager | 4.0 |
| 1.5B | V3 fold, graph | 4.0 |
| 1.5B | V4 expand, graph, fine buckets | 4.0 |
| 0.5B | V6 all three levers | 8.0 |
| 1.5B | V5 expand, graph, eager fallback | 4.0 |
| 1.5B | V6 all three levers | 8.0 |

<a id="stage13-thresholds"></a>

### The fallback thresholds

The crossover read from the step logs of the four rates pooled: the smallest running batch from which the eager step's median is no slower than the captured step's, for the expanded path (V1 against V0) and the folded path (V3 against V2). The setting is the `graph_max_batch` that follows (none: the captured step is never slower).

| model | path | crossover (running batch) | graph_max_batch |
| --- | --- | ---: | ---: |
| 0.5B | expand | 17 | 16 |
| 0.5B | fold | none | none |
| 1.5B | expand | 9 | 8 |
| 1.5B | fold | 17 | 16 |

<a id="stage13-drift"></a>

### The drift check

The baseline (V1) run twice, at the start and at the end of the session: median time per token and the ratio of the second pass to the first. A gap between variants smaller than the largest deviation is not a finding.

| model | rate (req/s) | first (ms) | repeat (ms) | ratio |
| --- | ---: | ---: | ---: | ---: |
| 0.5B | 1.0 | 11.5 | 11.4 | 0.998 |
| 0.5B | 2.0 | 13.5 | 13.6 | 1.012 |
| 0.5B | 4.0 | 19.9 | 20.3 | 1.016 |
| 0.5B | 8.0 | 62.1 | 64.2 | 1.034 |
| 1.5B | 1.0 | 26.5 | 26.5 | 1.002 |
| 1.5B | 2.0 | 39.9 | 39.3 | 0.985 |
| 1.5B | 4.0 | 106.1 | 106.2 | 1.001 |
| 1.5B | 8.0 | 160.4 | 160.6 | 1.001 |

The largest deviation is 3.4%.

<a id="stage13-closed"></a>

### Closed loop

Closed loop: exactly the given number of requests in flight, a new one the moment one finishes; prompts of 128 tokens, 64 generated each.

| model | variant | in flight | TPOT p50 (ms) | TPOT p99 (ms) | tok/s | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B | V0 expand, eager | 1 | 47.3 | 48.1 | 21 | 0 |
| 1.5B | V0 expand, eager | 4 | 49.2 | 50.5 | 80 | 0 |
| 1.5B | V0 expand, eager | 8 | 51.8 | 54.3 | 148 | 0 |
| 1.5B | V0 expand, eager | 16 | 55.2 | 61.1 | 265 | 0 |
| 1.5B | V0 expand, eager | 32 | 61.8 | 73.2 | 441 | 0 |
| 1.5B | V1 expand, graph | 1 | 19.9 | 20.0 | 49 | 0 |
| 1.5B | V1 expand, graph | 4 | 24.4 | 25.2 | 157 | 0 |
| 1.5B | V1 expand, graph | 8 | 30.2 | 32.4 | 245 | 0 |
| 1.5B | V1 expand, graph | 16 | 42.3 | 47.4 | 338 | 0 |
| 1.5B | V1 expand, graph | 32 | 65.9 | 76.9 | 419 | 0 |
| 1.5B | V3 fold, graph | 1 | 19.5 | 19.5 | 50 | 0 |
| 1.5B | V3 fold, graph | 4 | 22.4 | 23.2 | 170 | 0 |
| 1.5B | V3 fold, graph | 8 | 25.1 | 27.3 | 289 | 0 |
| 1.5B | V3 fold, graph | 16 | 32.4 | 37.5 | 425 | 0 |
| 1.5B | V3 fold, graph | 32 | 46.0 | 57.0 | 563 | 0 |
| 1.5B | V6 all three levers | 1 | 19.7 | 20.0 | 50 | 0 |
| 1.5B | V6 all three levers | 4 | 22.1 | 22.9 | 172 | 0 |
| 1.5B | V6 all three levers | 8 | 24.5 | 26.8 | 296 | 0 |
| 1.5B | V6 all three levers | 16 | 33.4 | 38.5 | 415 | 0 |
| 1.5B | V6 all three levers | 32 | 63.2 | 74.5 | 433 | 0 |

<a id="stage13-http"></a>

### The best variant over HTTP

The best variant of each model (highest capacity, then lowest time per token at the highest rate) served by `python -m minivllm.server` with the flags of its configuration, driven over HTTP by the same client and traces. The vLLM and Stage 12 engine rows are read from `vllm_compare.json`, a different session, for the same trace digests: compare the shape of the curves before the level.

Served: 0.5B V6 all three levers (`--graph --warmup --gqa fold --bucket-scheme fine`, ready after 57 s); 1.5B V6 all three levers (`--graph --warmup --gqa fold --bucket-scheme fine --graph-max-batch 16`, ready after 49 s).

| model | system | source | rate (req/s) | req/s | TTFT p50 (s) | TPOT p50 (ms) | met target | errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | V6 all three levers, over HTTP (this run) | this run | 1.0 | 1.05 | 0.05 | 9.9 | 100% | 0 |
| 0.5B | Stage 12 engine over HTTP | vllm_compare.json | 1.0 | 1.05 | 0.04 | 10.6 | 100% | 0 |
| 0.5B | vLLM | vllm_compare.json | 1.0 | 1.07 | 0.02 | 5.9 | 100% | 0 |
| 0.5B | V6 all three levers, over HTTP (this run) | this run | 2.0 | 2.02 | 0.05 | 11.1 | 100% | 0 |
| 0.5B | Stage 12 engine over HTTP | vllm_compare.json | 2.0 | 1.99 | 0.05 | 12.1 | 100% | 0 |
| 0.5B | vLLM | vllm_compare.json | 2.0 | 2.09 | 0.02 | 5.8 | 100% | 0 |
| 0.5B | V6 all three levers, over HTTP (this run) | this run | 4.0 | 3.71 | 0.05 | 12.5 | 100% | 0 |
| 0.5B | Stage 12 engine over HTTP | vllm_compare.json | 4.0 | 3.41 | 0.06 | 18.4 | 100% | 0 |
| 0.5B | vLLM | vllm_compare.json | 4.0 | 4.01 | 0.02 | 5.8 | 100% | 0 |
| 0.5B | V6 all three levers, over HTTP (this run) | this run | 8.0 | 6.17 | 0.06 | 17.0 | 100% | 0 |
| 0.5B | Stage 12 engine over HTTP | vllm_compare.json | 8.0 | 4.43 | 0.11 | 50.4 | 100% | 0 |
| 0.5B | vLLM | vllm_compare.json | 8.0 | 7.39 | 0.02 | 6.1 | 100% | 0 |
| 1.5B | V6 all three levers, over HTTP (this run) | this run | 1.0 | 1.01 | 0.08 | 21.6 | 100% | 0 |
| 1.5B | Stage 12 engine over HTTP | vllm_compare.json | 1.0 | 1.00 | 0.08 | 24.5 | 100% | 0 |
| 1.5B | vLLM | vllm_compare.json | 1.0 | 1.02 | 0.06 | 16.7 | 100% | 0 |
| 1.5B | V6 all three levers, over HTTP (this run) | this run | 2.0 | 1.86 | 0.08 | 24.8 | 100% | 0 |
| 1.5B | Stage 12 engine over HTTP | vllm_compare.json | 2.0 | 1.70 | 0.11 | 36.4 | 100% | 0 |
| 1.5B | vLLM | vllm_compare.json | 2.0 | 1.94 | 0.05 | 16.5 | 100% | 0 |
| 1.5B | V6 all three levers, over HTTP (this run) | this run | 4.0 | 3.08 | 0.10 | 32.2 | 100% | 0 |
| 1.5B | Stage 12 engine over HTTP | vllm_compare.json | 4.0 | 2.26 | 0.19 | 96.2 | 53% | 0 |
| 1.5B | vLLM | vllm_compare.json | 4.0 | 3.49 | 0.05 | 16.9 | 100% | 0 |
| 1.5B | V6 all three levers, over HTTP (this run) | this run | 8.0 | 4.24 | 0.17 | 66.1 | 98% | 0 |
| 1.5B | Stage 12 engine over HTTP | vllm_compare.json | 8.0 | 2.48 | 0.63 | 143.3 | 19% | 0 |
| 1.5B | vLLM | vllm_compare.json | 8.0 | 5.66 | 0.06 | 18.2 | 100% | 0 |

<a id="stage13-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage13_all_goodput_0p5b.png` | Goodput against offered load, all variants, 0.5B |
| `stage13_all_goodput_1p5b.png` | Goodput against offered load, all variants, 1.5B |
| `stage13_all_tpot_0p5b.png` | Median time per output token against offered load, all variants, 0.5B |
| `stage13_all_tpot_1p5b.png` | Median time per output token against offered load, all variants, 1.5B |
| `stage13_all_ttft_p99_0p5b.png` | Time to first token, 99th percentile, against offered load, all variants, 0.5B |
| `stage13_all_ttft_p99_1p5b.png` | Time to first token, 99th percentile, against offered load, all variants, 1.5B |
| `stage13_bytes_0p5b.png` | One layer's batched attention, the expanded path against the folded one, time against the counted bytes, 0.5B |
| `stage13_bytes_1p5b.png` | One layer's batched attention, the expanded path against the folded one, time against the counted bytes, 1.5B |
| `stage13_closed_tok_s.png` | Output tokens per second against requests in flight (closed loop), 1.5B |
| `stage13_closed_tpot.png` | Median time per output token against requests in flight (closed loop), 1.5B |
| `stage13_cross_expand_0p5b.png` | Decode step time against the running batch, expanded path, graph against eager, 0.5B |
| `stage13_cross_expand_1p5b.png` | Decode step time against the running batch, expanded path, graph against eager, 1.5B |
| `stage13_cross_fold_0p5b.png` | Decode step time against the running batch, folded path, graph against eager, 0.5B |
| `stage13_cross_fold_1p5b.png` | Decode step time against the running batch, folded path, graph against eager, 1.5B |
| `stage13_http_goodput_0p5b.png` | Goodput over HTTP: the best variant, the Stage 12 engine and vLLM, 0.5B |
| `stage13_http_goodput_1p5b.png` | Goodput over HTTP: the best variant, the Stage 12 engine and vLLM, 1.5B |
| `stage13_http_tpot_0p5b.png` | Median time per output token over HTTP: the best variant, the Stage 12 engine and vLLM, 0.5B |
| `stage13_http_tpot_1p5b.png` | Median time per output token over HTTP: the best variant, the Stage 12 engine and vLLM, 1.5B |
| `stage13_levers_goodput_0p5b.png` | Goodput against offered load, the levers one at a time, 0.5B |
| `stage13_levers_goodput_1p5b.png` | Goodput against offered load, the levers one at a time, 1.5B |
| `stage13_levers_steps_0p5b.png` | Decode step time against the running batch at the highest rate, the levers one at a time, 0.5B |
| `stage13_levers_steps_1p5b.png` | Decode step time against the running batch at the highest rate, the levers one at a time, 1.5B |
| `stage13_levers_tpot_0p5b.png` | Median time per output token against offered load, the levers one at a time, 0.5B |
| `stage13_levers_tpot_1p5b.png` | Median time per output token against offered load, the levers one at a time, 1.5B |

## Stage 14: The In-Place Paged Attention Kernel

Source: [`results/stage14.json`](results/stage14.json)

<a id="stage14-setup"></a>

### Setup

- **GPU:** Tesla T4
- **torch / transformers / python:** 2.11.0+cu128 / 5.16.1 / 3.13.15
- **Triton:** 3.6.0
- **Model:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
- **Model revision:** Qwen2.5-0.5B 060db649; Qwen2.5-1.5B 8faed761
- **max_pos (RoPE table):** 4096
- **Requests per workload / seed / quick run:** 64 / 0 / False

<a id="stage14-correct"></a>

### Are the kernels exact?

Each kernel against `paged_attention_reference` (gather, then a softmax in fp32) in fp16, on the head shapes of the model, with a different length in every row, scattered block tables and a pool whose unreachable slots are filled with NaN: `finite` says no NaN reached the output, `same` that the output equals the one from a clean pool.

| model | kernel | max abs difference | largest output | finite with poison | same with poison |
| --- | --- | ---: | ---: | --- | --- |
| 0.5B | simple | 0.00003 | 2.45 | yes | yes |
| 0.5B | grouped | 0.00098 | 2.45 | yes | yes |
| 1.5B | simple | 0.00006 | 2.62 | yes | yes |
| 1.5B | grouped | 0.00098 | 2.62 | yes | yes |

Greedy output of each variant against plain cached decoding of the same weights, 48 new tokens per prompt, requests of different lengths. A difference counts as an fp16 near-tie only if the logit gap at the first difference is below 0.5. The sampled run is a control that must differ. The fallback variants are run with a threshold of 2 so a batch of twelve starts on the eager path and ends on the graph path; `eager steps` and `graph steps` show that both ran.

| model | variant | prompts | identical | near ties | other | eager steps | graph steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | B0 Part 11 graph, the baseline | 12 | 11 | 1 | 0 | 0 | 42 |
| 0.5B | K0 Part 12 best, gather | 12 | 12 | 0 | 0 | 0 | 39 |
| 0.5B | K1 in place, one program per head | 12 | 12 | 0 | 0 | 0 | 39 |
| 0.5B | K2 in place, grouped by key-value head | 12 | 12 | 0 | 0 | 0 | 39 |
| 0.5B | K3 best kernel, powers of two | 12 | 12 | 0 | 0 | 0 | 42 |
| 0.5B | sampled (control) | 12 | 1 | 1 | 10 | n/a | n/a |
| 1.5B | B0 Part 11 graph, the baseline | 12 | 12 | 0 | 0 | 0 | 42 |
| 1.5B | K0 Part 12 best, gather | 12 | 12 | 0 | 0 | 45 | 1 |
| 1.5B | K1 in place, one program per head | 12 | 12 | 0 | 0 | 45 | 1 |
| 1.5B | K2 in place, grouped by key-value head | 12 | 12 | 0 | 0 | 45 | 1 |
| 1.5B | K3 best kernel, powers of two | 12 | 12 | 0 | 0 | 45 | 1 |
| 1.5B | K4 best kernel, no fallback | 12 | 12 | 0 | 0 | 0 | 39 |
| 1.5B | sampled (control) | 12 | 1 | 1 | 10 | n/a | n/a |

<a id="stage14-gate"></a>

### The gate: one layer's attention

One layer's attention, timed inside a CUDA graph on random tensors of the real shapes: the folded gather path of Stage 13 against the two in-place kernels, for a batch `B` of rows that each fill `fill` of a bucket of `NB` blocks. `MB` is the counted traffic (gather path: the gather copies and the attention read; in place: what the rows own) and `GB/s` that traffic over the measured time. `speedup` is the gather path's time over the kernel's: above 1 is a kernel win.

| model | B | NB | fill | MB gather | MB in place | bytes ratio | ms gather | ms per head | ms grouped | speedup per head | speedup grouped | GB/s per head | GB/s grouped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 4 | 4 | 0.25 | 0.66 | 0.03 | 20 | 0.028 | 0.005 | 0.005 | 5.67 | 5.63 | 7 | 7 |
| 0.5B | 4 | 4 | 0.50 | 0.66 | 0.07 | 10 | 0.028 | 0.007 | 0.007 | 4.04 | 3.74 | 10 | 9 |
| 0.5B | 4 | 4 | 1.00 | 0.66 | 0.13 | 5 | 0.028 | 0.011 | 0.012 | 2.61 | 2.26 | 12 | 11 |
| 0.5B | 4 | 16 | 0.25 | 2.62 | 0.13 | 20 | 0.043 | 0.011 | 0.012 | 4.10 | 3.53 | 12 | 11 |
| 0.5B | 4 | 16 | 0.50 | 2.62 | 0.26 | 10 | 0.043 | 0.018 | 0.022 | 2.34 | 1.96 | 14 | 12 |
| 0.5B | 4 | 16 | 1.00 | 2.62 | 0.52 | 5 | 0.043 | 0.034 | 0.041 | 1.27 | 1.03 | 16 | 13 |
| 0.5B | 4 | 64 | 0.25 | 10.49 | 0.52 | 20 | 0.139 | 0.036 | 0.044 | 3.88 | 3.15 | 15 | 12 |
| 0.5B | 4 | 64 | 0.50 | 10.49 | 1.05 | 10 | 0.139 | 0.069 | 0.083 | 2.03 | 1.67 | 15 | 13 |
| 0.5B | 4 | 64 | 1.00 | 10.49 | 2.10 | 5 | 0.150 | 0.134 | 0.164 | 1.12 | 0.91 | 16 | 13 |
| 0.5B | 16 | 4 | 0.25 | 2.62 | 0.13 | 20 | 0.029 | 0.009 | 0.007 | 3.27 | 4.50 | 15 | 20 |
| 0.5B | 16 | 4 | 0.50 | 2.62 | 0.26 | 10 | 0.029 | 0.015 | 0.010 | 1.96 | 2.78 | 18 | 25 |
| 0.5B | 16 | 4 | 1.00 | 2.62 | 0.52 | 5 | 0.029 | 0.027 | 0.018 | 1.08 | 1.59 | 19 | 29 |
| 0.5B | 16 | 16 | 0.25 | 10.49 | 0.52 | 20 | 0.081 | 0.030 | 0.021 | 2.66 | 3.90 | 17 | 25 |
| 0.5B | 16 | 16 | 0.50 | 10.49 | 1.05 | 10 | 0.078 | 0.059 | 0.037 | 1.33 | 2.12 | 18 | 28 |
| 0.5B | 16 | 16 | 1.00 | 10.49 | 2.10 | 5 | 0.081 | 0.106 | 0.070 | 0.76 | 1.15 | 20 | 30 |
| 0.5B | 16 | 64 | 0.25 | 41.94 | 2.10 | 20 | 0.284 | 0.106 | 0.070 | 2.68 | 4.05 | 20 | 30 |
| 0.5B | 16 | 64 | 0.50 | 41.94 | 4.19 | 10 | 0.289 | 0.213 | 0.133 | 1.36 | 2.17 | 20 | 31 |
| 0.5B | 16 | 64 | 1.00 | 41.94 | 8.39 | 5 | 0.282 | 0.402 | 0.265 | 0.70 | 1.07 | 21 | 32 |
| 0.5B | 32 | 4 | 0.25 | 5.24 | 0.26 | 20 | 0.038 | 0.016 | 0.011 | 2.34 | 3.53 | 16 | 24 |
| 0.5B | 32 | 4 | 0.50 | 5.24 | 0.52 | 10 | 0.038 | 0.029 | 0.019 | 1.31 | 1.97 | 18 | 27 |
| 0.5B | 32 | 4 | 1.00 | 5.24 | 1.05 | 5 | 0.038 | 0.054 | 0.034 | 0.70 | 1.11 | 19 | 31 |
| 0.5B | 32 | 16 | 0.25 | 20.97 | 1.05 | 20 | 0.127 | 0.056 | 0.037 | 2.25 | 3.45 | 19 | 29 |
| 0.5B | 32 | 16 | 0.50 | 20.97 | 2.10 | 10 | 0.130 | 0.112 | 0.067 | 1.16 | 1.93 | 19 | 31 |
| 0.5B | 32 | 16 | 1.00 | 20.97 | 4.19 | 5 | 0.131 | 0.208 | 0.127 | 0.63 | 1.03 | 20 | 33 |
| 0.5B | 32 | 64 | 0.25 | 83.89 | 4.19 | 20 | 0.470 | 0.218 | 0.127 | 2.16 | 3.70 | 19 | 33 |
| 0.5B | 32 | 64 | 0.50 | 83.89 | 8.39 | 10 | 0.466 | 0.404 | 0.246 | 1.15 | 1.90 | 21 | 34 |
| 0.5B | 32 | 64 | 1.00 | 83.89 | 16.78 | 5 | 0.467 | 0.804 | 0.485 | 0.58 | 0.96 | 21 | 35 |
| 1.5B | 4 | 4 | 0.25 | 1.31 | 0.07 | 20 | 0.029 | 0.006 | 0.007 | 4.77 | 4.10 | 11 | 9 |
| 1.5B | 4 | 4 | 0.50 | 1.31 | 0.13 | 10 | 0.029 | 0.009 | 0.011 | 3.17 | 2.55 | 14 | 12 |
| 1.5B | 4 | 4 | 1.00 | 1.31 | 0.26 | 5 | 0.029 | 0.015 | 0.020 | 1.89 | 1.45 | 17 | 13 |
| 1.5B | 4 | 16 | 0.25 | 5.24 | 0.26 | 20 | 0.041 | 0.015 | 0.020 | 2.71 | 2.08 | 17 | 13 |
| 1.5B | 4 | 16 | 0.50 | 5.24 | 0.52 | 10 | 0.044 | 0.030 | 0.039 | 1.49 | 1.14 | 18 | 14 |
| 1.5B | 4 | 16 | 1.00 | 5.24 | 1.05 | 5 | 0.042 | 0.054 | 0.072 | 0.78 | 0.58 | 20 | 15 |
| 1.5B | 4 | 64 | 0.25 | 20.97 | 1.05 | 20 | 0.178 | 0.057 | 0.078 | 3.11 | 2.28 | 18 | 13 |
| 1.5B | 4 | 64 | 0.50 | 20.97 | 2.10 | 10 | 0.186 | 0.114 | 0.148 | 1.63 | 1.26 | 18 | 14 |
| 1.5B | 4 | 64 | 1.00 | 20.97 | 4.19 | 5 | 0.171 | 0.213 | 0.286 | 0.80 | 0.60 | 20 | 15 |
| 1.5B | 16 | 4 | 0.25 | 5.24 | 0.26 | 20 | 0.034 | 0.012 | 0.010 | 2.85 | 3.37 | 22 | 26 |
| 1.5B | 16 | 4 | 0.50 | 5.24 | 0.52 | 10 | 0.034 | 0.021 | 0.018 | 1.67 | 1.95 | 26 | 30 |
| 1.5B | 16 | 4 | 1.00 | 5.24 | 1.05 | 5 | 0.034 | 0.038 | 0.032 | 0.89 | 1.05 | 28 | 32 |
| 1.5B | 16 | 16 | 0.25 | 20.97 | 1.05 | 20 | 0.123 | 0.044 | 0.035 | 2.83 | 3.58 | 24 | 30 |
| 1.5B | 16 | 16 | 0.50 | 20.97 | 2.10 | 10 | 0.120 | 0.081 | 0.069 | 1.48 | 1.74 | 26 | 30 |
| 1.5B | 16 | 16 | 1.00 | 20.97 | 4.19 | 5 | 0.119 | 0.157 | 0.124 | 0.76 | 0.96 | 27 | 34 |
| 1.5B | 16 | 64 | 0.25 | 83.89 | 4.19 | 20 | 0.431 | 0.159 | 0.128 | 2.71 | 3.38 | 26 | 33 |
| 1.5B | 16 | 64 | 0.50 | 83.89 | 8.39 | 10 | 0.431 | 0.312 | 0.244 | 1.38 | 1.77 | 27 | 34 |
| 1.5B | 16 | 64 | 1.00 | 83.89 | 16.78 | 5 | 0.429 | 0.584 | 0.479 | 0.74 | 0.90 | 29 | 35 |
| 1.5B | 32 | 4 | 0.25 | 10.49 | 0.52 | 20 | 0.064 | 0.024 | 0.020 | 2.68 | 3.24 | 22 | 27 |
| 1.5B | 32 | 4 | 0.50 | 10.49 | 1.05 | 10 | 0.064 | 0.042 | 0.035 | 1.53 | 1.85 | 25 | 30 |
| 1.5B | 32 | 4 | 1.00 | 10.49 | 2.10 | 5 | 0.065 | 0.080 | 0.063 | 0.81 | 1.03 | 26 | 33 |
| 1.5B | 32 | 16 | 0.25 | 41.94 | 2.10 | 20 | 0.225 | 0.084 | 0.065 | 2.68 | 3.45 | 25 | 32 |
| 1.5B | 32 | 16 | 0.50 | 41.94 | 4.19 | 10 | 0.224 | 0.155 | 0.122 | 1.44 | 1.83 | 27 | 34 |
| 1.5B | 32 | 16 | 1.00 | 41.94 | 8.39 | 5 | 0.225 | 0.306 | 0.234 | 0.73 | 0.96 | 27 | 36 |
| 1.5B | 32 | 64 | 0.25 | 167.77 | 8.39 | 20 | 0.808 | 0.300 | 0.236 | 2.70 | 3.42 | 28 | 35 |
| 1.5B | 32 | 64 | 0.50 | 167.77 | 16.78 | 10 | 0.808 | 0.597 | 0.459 | 1.35 | 1.76 | 28 | 37 |
| 1.5B | 32 | 64 | 1.00 | 167.77 | 33.55 | 5 | 0.808 | 1.207 | 0.907 | 0.67 | 0.89 | 28 | 37 |

<a id="stage14-profile"></a>

### One step under the profiler

One whole decode step (all layers and the weights) under the profiler at a batch of 32 rows of 32 blocks, half filled, per path, milliseconds per step by kernel family.

| model | path | total (ms) | attention | matmul | copy and index | elementwise |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | gather | 14.9 | 1.5 | 6.2 | 5.7 | 1.5 |
| 0.5B | paged_grouped | 19.7 | 7.3 | 7.9 | 1.5 | 3.1 |
| 1.5B | gather | 39.3 | 2.6 | 18.7 | 15.3 | 2.6 |
| 1.5B | paged_grouped | 31.5 | 9.2 | 18.7 | 1.1 | 2.5 |

The costliest kernels of the step, 1.5B, gather:

| kernel | family | ms | calls |
| --- | --- | ---: | ---: |
| turing_fp16_s1688gemm_fp16_256x64_sliced1x2_ldg8_f2f_tn | matmul | 8.65 | 56 |
| void at::native::index_elementwise_kernel<128, 4, at::native | copy and index | 8.44 | 58 |
| turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn | matmul | 7.20 | 29 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_ | copy and index | 5.62 | 84 |
| fmha_cutlassF_f16_aligned_32x128_rf_sm75(PyTorchMemEffAttent | attention | 2.64 | 28 |

The costliest kernels of the step, 1.5B, paged_grouped:

| kernel | family | ms | calls |
| --- | --- | ---: | ---: |
| _paged_grouped_kernel | attention | 9.21 | 28 |
| turing_fp16_s1688gemm_fp16_256x64_sliced1x2_ldg8_f2f_tn | matmul | 8.63 | 56 |
| turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn | matmul | 7.20 | 29 |
| turing_fp16_s1688gemm_fp16_64x64_sliced1x4_ldg8_f2f_tn | matmul | 1.06 | 28 |
| turing_fp16_s1688gemm_fp16_256x64_ldg8_f2f_stages_32x1_tn | matmul | 1.02 | 28 |

<a id="stage14-load"></a>

### The variants under load

The Stage 12 traces (64 requests of the `mixed` workload per rate), each variant in its own engine, step log on. B0 Part 11 graph, the baseline; K0 Part 12 best, gather; K1 in place, one program per head; K2 in place, grouped by key-value head; K3 best kernel, powers of two; K4 best kernel, no fallback. `(repeat)` is B0 run again at the end of the session; `eager` is the share of decode steps that took the eager path.

| model | rate (req/s) | variant | req/s | tok/s | TTFT p50 (s) | TPOT p50 (ms) | TPOT p99 (ms) | met target | eager | capture (s) | errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 1.0 | B0 Part 11 graph, the baseline | 1.05 | 70 | 0.04 | 10.4 | 14.9 | 100% | 0% | 5.0 | 0 |
| 0.5B | 2.0 | B0 Part 11 graph, the baseline | 2.00 | 134 | 0.04 | 12.0 | 21.9 | 100% | 0% | 5.0 | 0 |
| 0.5B | 4.0 | B0 Part 11 graph, the baseline | 3.49 | 234 | 0.04 | 17.2 | 39.4 | 100% | 0% | 5.0 | 0 |
| 0.5B | 8.0 | B0 Part 11 graph, the baseline | 4.68 | 314 | 0.06 | 46.7 | 74.0 | 100% | 0% | 5.0 | 0 |
| 0.5B | 1.0 | K0 Part 12 best, gather | 1.06 | 71 | 0.03 | 9.4 | 13.7 | 100% | 0% | 33.4 | 0 |
| 0.5B | 2.0 | K0 Part 12 best, gather | 2.03 | 136 | 0.04 | 10.1 | 13.2 | 100% | 0% | 33.4 | 0 |
| 0.5B | 4.0 | K0 Part 12 best, gather | 3.78 | 253 | 0.04 | 11.0 | 17.6 | 100% | 0% | 33.4 | 0 |
| 0.5B | 8.0 | K0 Part 12 best, gather | 6.50 | 436 | 0.04 | 14.4 | 33.8 | 100% | 0% | 33.4 | 0 |
| 0.5B | 1.0 | K1 in place, one program per head | 1.06 | 71 | 0.03 | 8.8 | 12.7 | 100% | 0% | 28.5 | 0 |
| 0.5B | 2.0 | K1 in place, one program per head | 2.04 | 137 | 0.04 | 9.3 | 12.5 | 100% | 0% | 28.5 | 0 |
| 0.5B | 4.0 | K1 in place, one program per head | 3.82 | 256 | 0.04 | 10.0 | 16.0 | 100% | 0% | 28.5 | 0 |
| 0.5B | 8.0 | K1 in place, one program per head | 6.67 | 447 | 0.04 | 12.9 | 32.8 | 100% | 0% | 28.5 | 0 |
| 0.5B | 1.0 | K2 in place, grouped by key-value head | 1.05 | 71 | 0.03 | 9.4 | 13.1 | 100% | 0% | 28.6 | 0 |
| 0.5B | 2.0 | K2 in place, grouped by key-value head | 2.03 | 136 | 0.04 | 9.8 | 12.6 | 100% | 0% | 28.6 | 0 |
| 0.5B | 4.0 | K2 in place, grouped by key-value head | 3.78 | 253 | 0.04 | 10.4 | 16.9 | 100% | 0% | 28.6 | 0 |
| 0.5B | 8.0 | K2 in place, grouped by key-value head | 6.57 | 440 | 0.04 | 13.7 | 32.8 | 100% | 0% | 28.6 | 0 |
| 0.5B | 1.0 | K3 best kernel, powers of two | 1.05 | 71 | 0.04 | 9.5 | 13.0 | 100% | 0% | 4.3 | 0 |
| 0.5B | 2.0 | K3 best kernel, powers of two | 2.03 | 136 | 0.04 | 9.9 | 13.0 | 100% | 0% | 4.3 | 0 |
| 0.5B | 4.0 | K3 best kernel, powers of two | 3.77 | 253 | 0.04 | 10.6 | 17.2 | 100% | 0% | 4.3 | 0 |
| 0.5B | 8.0 | K3 best kernel, powers of two | 6.53 | 437 | 0.04 | 13.5 | 33.3 | 100% | 0% | 4.3 | 0 |
| 0.5B | 1.0 | B0 Part 11 graph, the baseline (repeat) | 1.05 | 70 | 0.04 | 10.7 | 15.3 | 100% | 0% | 4.9 | 0 |
| 0.5B | 2.0 | B0 Part 11 graph, the baseline (repeat) | 2.00 | 134 | 0.04 | 12.3 | 23.4 | 100% | 0% | 4.9 | 0 |
| 0.5B | 4.0 | B0 Part 11 graph, the baseline (repeat) | 3.47 | 233 | 0.04 | 17.5 | 38.7 | 100% | 0% | 4.9 | 0 |
| 0.5B | 8.0 | B0 Part 11 graph, the baseline (repeat) | 4.60 | 308 | 0.07 | 49.2 | 83.9 | 100% | 0% | 4.9 | 0 |
| 1.5B | 1.0 | B0 Part 11 graph, the baseline | 1.00 | 67 | 0.06 | 24.5 | 46.5 | 100% | 0% | 6.1 | 0 |
| 1.5B | 2.0 | B0 Part 11 graph, the baseline | 1.73 | 116 | 0.06 | 34.5 | 79.9 | 100% | 0% | 6.1 | 0 |
| 1.5B | 4.0 | B0 Part 11 graph, the baseline | 2.36 | 158 | 0.11 | 92.2 | 137.8 | 55% | 0% | 6.1 | 0 |
| 1.5B | 8.0 | B0 Part 11 graph, the baseline | 2.62 | 176 | 0.35 | 139.3 | 166.2 | 22% | 0% | 6.1 | 0 |
| 1.5B | 1.0 | K0 Part 12 best, gather | 1.01 | 68 | 0.05 | 21.2 | 26.9 | 100% | 0% | 38.3 | 0 |
| 1.5B | 2.0 | K0 Part 12 best, gather | 1.88 | 126 | 0.06 | 22.8 | 35.1 | 100% | 0% | 38.3 | 0 |
| 1.5B | 4.0 | K0 Part 12 best, gather | 3.22 | 215 | 0.06 | 28.6 | 53.4 | 100% | 0% | 38.3 | 0 |
| 1.5B | 8.0 | K0 Part 12 best, gather | 4.50 | 301 | 0.10 | 57.6 | 92.0 | 100% | 22% | 38.3 | 0 |
| 1.5B | 1.0 | K1 in place, one program per head | 1.02 | 68 | 0.05 | 20.4 | 26.2 | 100% | 0% | 35.2 | 0 |
| 1.5B | 2.0 | K1 in place, one program per head | 1.89 | 127 | 0.05 | 22.0 | 32.8 | 100% | 0% | 35.2 | 0 |
| 1.5B | 4.0 | K1 in place, one program per head | 3.27 | 219 | 0.06 | 26.3 | 44.5 | 100% | 0% | 35.2 | 0 |
| 1.5B | 8.0 | K1 in place, one program per head | 4.83 | 324 | 0.09 | 43.6 | 81.9 | 100% | 16% | 35.2 | 0 |
| 1.5B | 1.0 | K2 in place, grouped by key-value head | 1.01 | 67 | 0.05 | 21.6 | 27.6 | 100% | 0% | 33.8 | 0 |
| 1.5B | 2.0 | K2 in place, grouped by key-value head | 1.86 | 125 | 0.06 | 23.2 | 34.2 | 100% | 0% | 33.8 | 0 |
| 1.5B | 4.0 | K2 in place, grouped by key-value head | 3.21 | 215 | 0.06 | 27.3 | 47.7 | 100% | 0% | 33.8 | 0 |
| 1.5B | 8.0 | K2 in place, grouped by key-value head | 4.73 | 317 | 0.09 | 45.6 | 82.0 | 100% | 16% | 33.8 | 0 |
| 1.5B | 1.0 | K3 best kernel, powers of two | 1.01 | 67 | 0.05 | 21.5 | 27.9 | 100% | 0% | 5.2 | 0 |
| 1.5B | 2.0 | K3 best kernel, powers of two | 1.86 | 125 | 0.05 | 23.3 | 34.7 | 100% | 0% | 5.2 | 0 |
| 1.5B | 4.0 | K3 best kernel, powers of two | 3.18 | 213 | 0.06 | 28.0 | 48.3 | 100% | 0% | 5.2 | 0 |
| 1.5B | 8.0 | K3 best kernel, powers of two | 4.62 | 310 | 0.09 | 46.4 | 82.2 | 100% | 19% | 5.2 | 0 |
| 1.5B | 1.0 | K4 best kernel, no fallback | 1.01 | 67 | 0.05 | 21.4 | 27.6 | 100% | 0% | 33.6 | 0 |
| 1.5B | 2.0 | K4 best kernel, no fallback | 1.86 | 125 | 0.05 | 23.1 | 34.0 | 100% | 0% | 33.6 | 0 |
| 1.5B | 4.0 | K4 best kernel, no fallback | 3.20 | 215 | 0.07 | 27.5 | 45.3 | 100% | 0% | 33.6 | 0 |
| 1.5B | 8.0 | K4 best kernel, no fallback | 4.79 | 321 | 0.09 | 38.2 | 77.8 | 100% | 0% | 33.6 | 0 |
| 1.5B | 1.0 | B0 Part 11 graph, the baseline (repeat) | 1.00 | 67 | 0.06 | 24.4 | 46.6 | 100% | 0% | 6.0 | 0 |
| 1.5B | 2.0 | B0 Part 11 graph, the baseline (repeat) | 1.73 | 116 | 0.06 | 34.2 | 74.7 | 100% | 0% | 6.0 | 0 |
| 1.5B | 4.0 | B0 Part 11 graph, the baseline (repeat) | 2.36 | 158 | 0.10 | 93.1 | 138.8 | 55% | 0% | 6.0 | 0 |
| 1.5B | 8.0 | B0 Part 11 graph, the baseline (repeat) | 2.62 | 176 | 0.34 | 139.4 | 166.4 | 22% | 0% | 6.0 | 0 |

<a id="stage14-capacity"></a>

### Capacity under the latency target

The highest tested rate at which at least 90% of requests met the latency target ("none": no tested rate did).

| model | variant | capacity (req/s) |
| --- | --- | ---: |
| 0.5B | B0 Part 11 graph, the baseline | 8.0 |
| 0.5B | K0 Part 12 best, gather | 8.0 |
| 0.5B | K1 in place, one program per head | 8.0 |
| 0.5B | K2 in place, grouped by key-value head | 8.0 |
| 0.5B | K3 best kernel, powers of two | 8.0 |
| 1.5B | B0 Part 11 graph, the baseline | 2.0 |
| 1.5B | K0 Part 12 best, gather | 8.0 |
| 1.5B | K1 in place, one program per head | 8.0 |
| 1.5B | K2 in place, grouped by key-value head | 8.0 |
| 1.5B | K3 best kernel, powers of two | 8.0 |
| 1.5B | K4 best kernel, no fallback | 8.0 |

<a id="stage14-drift"></a>

### The drift check

The baseline (B0) run twice, at the start and at the end of the session: median time per token and the ratio of the second pass to the first. A gap between variants smaller than the largest deviation is not a finding.

| model | rate (req/s) | first (ms) | repeat (ms) | ratio |
| --- | ---: | ---: | ---: | ---: |
| 0.5B | 1.0 | 10.4 | 10.7 | 1.027 |
| 0.5B | 2.0 | 12.0 | 12.3 | 1.020 |
| 0.5B | 4.0 | 17.2 | 17.5 | 1.016 |
| 0.5B | 8.0 | 46.7 | 49.2 | 1.053 |
| 1.5B | 1.0 | 24.5 | 24.4 | 0.998 |
| 1.5B | 2.0 | 34.5 | 34.2 | 0.993 |
| 1.5B | 4.0 | 92.2 | 93.1 | 1.010 |
| 1.5B | 8.0 | 139.3 | 139.4 | 1.001 |

The largest deviation is 5.3%.

<a id="stage14-closed"></a>

### Closed loop

Closed loop: exactly the given number of requests in flight, a new one the moment one finishes; prompts of 128 tokens, 64 generated each.

| model | variant | in flight | TPOT p50 (ms) | TPOT p99 (ms) | tok/s | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B | B0 Part 11 graph, the baseline | 1 | 18.6 | 18.6 | 53 | 0 |
| 1.5B | B0 Part 11 graph, the baseline | 4 | 22.8 | 23.5 | 169 | 0 |
| 1.5B | B0 Part 11 graph, the baseline | 8 | 27.9 | 29.9 | 267 | 0 |
| 1.5B | B0 Part 11 graph, the baseline | 16 | 38.5 | 43.1 | 372 | 0 |
| 1.5B | B0 Part 11 graph, the baseline | 32 | 59.4 | 69.3 | 466 | 0 |
| 1.5B | K0 Part 12 best, gather | 1 | 18.2 | 18.2 | 54 | 0 |
| 1.5B | K0 Part 12 best, gather | 4 | 20.8 | 21.5 | 184 | 0 |
| 1.5B | K0 Part 12 best, gather | 8 | 22.9 | 24.8 | 320 | 0 |
| 1.5B | K0 Part 12 best, gather | 16 | 31.0 | 35.4 | 451 | 0 |
| 1.5B | K0 Part 12 best, gather | 32 | 52.4 | 62.1 | 519 | 0 |
| 1.5B | K1 in place, one program per head | 1 | 17.7 | 17.7 | 55 | 0 |
| 1.5B | K1 in place, one program per head | 4 | 20.1 | 21.1 | 189 | 0 |
| 1.5B | K1 in place, one program per head | 8 | 22.3 | 24.2 | 327 | 0 |
| 1.5B | K1 in place, one program per head | 16 | 30.8 | 35.5 | 451 | 0 |
| 1.5B | K1 in place, one program per head | 32 | 46.8 | 56.6 | 570 | 0 |
| 1.5B | K2 in place, grouped by key-value head | 1 | 18.4 | 18.4 | 53 | 0 |
| 1.5B | K2 in place, grouped by key-value head | 4 | 20.5 | 21.2 | 186 | 0 |
| 1.5B | K2 in place, grouped by key-value head | 8 | 22.2 | 24.1 | 329 | 0 |
| 1.5B | K2 in place, grouped by key-value head | 16 | 30.0 | 34.4 | 464 | 0 |
| 1.5B | K2 in place, grouped by key-value head | 32 | 46.8 | 56.6 | 569 | 0 |

<a id="stage14-http"></a>

### The best variant over HTTP

The best variant of each model (highest capacity, then lowest time per token at the highest rate) served by `python -m minivllm.server` with the flags of its configuration, driven over HTTP by the same client and traces. The other three systems (vLLM, the Stage 12 engine, the Stage 13 best) were measured in other sessions and read from `vllm_compare.json` and `stage13.json` by the notebook, for the same trace digests: compare the shape of the curves before the level.

Served: 0.5B K1 in place, one program per head (`--graph --warmup --gqa fold --bucket-scheme fine --attention paged`, ready after 44 s); 1.5B K4 best kernel, no fallback (`--graph --warmup --gqa fold --bucket-scheme fine --attention paged_grouped`, ready after 52 s).

| model | system | source | rate (req/s) | req/s | TTFT p50 (s) | TPOT p50 (ms) | met target | errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | Part 13: in place, one program per head (K1) | this run | 1.0 | 1.06 | 0.04 | 8.7 | 100% | 0 |
| 0.5B | Part 11: CUDA graphs only | another session | 1.0 | 1.05 | 0.04 | 10.6 | 100% | 0 |
| 0.5B | Part 12: + folded heads, fine buckets, fallback | another session | 1.0 | 1.05 | 0.05 | 9.9 | 100% | 0 |
| 0.5B | vLLM | another session | 1.0 | 1.07 | 0.02 | 5.9 | 100% | 0 |
| 0.5B | Part 13: in place, one program per head (K1) | this run | 2.0 | 2.04 | 0.04 | 9.4 | 100% | 0 |
| 0.5B | Part 11: CUDA graphs only | another session | 2.0 | 1.99 | 0.05 | 12.1 | 100% | 0 |
| 0.5B | Part 12: + folded heads, fine buckets, fallback | another session | 2.0 | 2.02 | 0.05 | 11.1 | 100% | 0 |
| 0.5B | vLLM | another session | 2.0 | 2.09 | 0.02 | 5.8 | 100% | 0 |
| 0.5B | Part 13: in place, one program per head (K1) | this run | 4.0 | 3.80 | 0.04 | 10.3 | 100% | 0 |
| 0.5B | Part 11: CUDA graphs only | another session | 4.0 | 3.41 | 0.06 | 18.4 | 100% | 0 |
| 0.5B | Part 12: + folded heads, fine buckets, fallback | another session | 4.0 | 3.71 | 0.05 | 12.5 | 100% | 0 |
| 0.5B | vLLM | another session | 4.0 | 4.01 | 0.02 | 5.8 | 100% | 0 |
| 0.5B | Part 13: in place, one program per head (K1) | this run | 8.0 | 6.54 | 0.05 | 13.2 | 100% | 0 |
| 0.5B | Part 11: CUDA graphs only | another session | 8.0 | 4.43 | 0.11 | 50.4 | 100% | 0 |
| 0.5B | Part 12: + folded heads, fine buckets, fallback | another session | 8.0 | 6.17 | 0.06 | 17.0 | 100% | 0 |
| 0.5B | vLLM | another session | 8.0 | 7.39 | 0.02 | 6.1 | 100% | 0 |
| 1.5B | Part 13: best kernel, no fallback (K4) | this run | 1.0 | 1.01 | 0.08 | 20.9 | 100% | 0 |
| 1.5B | Part 11: CUDA graphs only | another session | 1.0 | 1.00 | 0.08 | 24.5 | 100% | 0 |
| 1.5B | Part 12: + folded heads, fine buckets, fallback | another session | 1.0 | 1.01 | 0.08 | 21.6 | 100% | 0 |
| 1.5B | vLLM | another session | 1.0 | 1.02 | 0.06 | 16.7 | 100% | 0 |
| 1.5B | Part 13: best kernel, no fallback (K4) | this run | 2.0 | 1.86 | 0.08 | 22.7 | 100% | 0 |
| 1.5B | Part 11: CUDA graphs only | another session | 2.0 | 1.70 | 0.11 | 36.4 | 100% | 0 |
| 1.5B | Part 12: + folded heads, fine buckets, fallback | another session | 2.0 | 1.86 | 0.08 | 24.8 | 100% | 0 |
| 1.5B | vLLM | another session | 2.0 | 1.94 | 0.05 | 16.5 | 100% | 0 |
| 1.5B | Part 13: best kernel, no fallback (K4) | this run | 4.0 | 3.17 | 0.08 | 27.8 | 100% | 0 |
| 1.5B | Part 11: CUDA graphs only | another session | 4.0 | 2.26 | 0.19 | 96.2 | 53% | 0 |
| 1.5B | Part 12: + folded heads, fine buckets, fallback | another session | 4.0 | 3.08 | 0.10 | 32.2 | 100% | 0 |
| 1.5B | vLLM | another session | 4.0 | 3.49 | 0.05 | 16.9 | 100% | 0 |
| 1.5B | Part 13: best kernel, no fallback (K4) | this run | 8.0 | 4.64 | 0.13 | 43.3 | 100% | 0 |
| 1.5B | Part 11: CUDA graphs only | another session | 8.0 | 2.48 | 0.63 | 143.3 | 19% | 0 |
| 1.5B | Part 12: + folded heads, fine buckets, fallback | another session | 8.0 | 4.24 | 0.17 | 66.1 | 98% | 0 |
| 1.5B | vLLM | another session | 8.0 | 5.66 | 0.06 | 18.2 | 100% | 0 |

<a id="stage14-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `stage14_closed_tok_s.png` | Output tokens per second against requests in flight (closed loop), 1.5B |
| `stage14_closed_tpot.png` | Median time per output token against requests in flight (closed loop), 1.5B |
| `stage14_gate_0p5b.png` | One layer's attention at the largest batch, the folded gather path and the two in-place kernels, against blocks per row and the share of the bucket that holds real positions, 0.5B |
| `stage14_gate_1p5b.png` | One layer's attention at the largest batch, the folded gather path and the two in-place kernels, against blocks per row and the share of the bucket that holds real positions, 1.5B |
| `stage14_goodput_0p5b.png` | Goodput against offered load, the baseline, the Stage 13 best and the in-place kernels, 0.5B |
| `stage14_goodput_1p5b.png` | Goodput against offered load, the baseline, the Stage 13 best and the in-place kernels, 1.5B |
| `stage14_http_goodput_0p5b.png` | Goodput over HTTP: this stage's best variant, the Stage 13 best, the Stage 12 engine and vLLM, 0.5B |
| `stage14_http_goodput_1p5b.png` | Goodput over HTTP: this stage's best variant, the Stage 13 best, the Stage 12 engine and vLLM, 1.5B |
| `stage14_http_tpot_0p5b.png` | Median time per output token over HTTP: this stage's best variant, the Stage 13 best, the Stage 12 engine and vLLM, 0.5B |
| `stage14_http_tpot_1p5b.png` | Median time per output token over HTTP: this stage's best variant, the Stage 13 best, the Stage 12 engine and vLLM, 1.5B |
| `stage14_steps_0p5b.png` | Decode step time against the running batch at the highest rate, the baseline, the Stage 13 best and the in-place kernels, 0.5B |
| `stage14_steps_1p5b.png` | Decode step time against the running batch at the highest rate, the baseline, the Stage 13 best and the in-place kernels, 1.5B |
| `stage14_tpot_0p5b.png` | Median time per output token against offered load, the baseline, the Stage 13 best and the in-place kernels, 0.5B |
| `stage14_tpot_1p5b.png` | Median time per output token against offered load, the baseline, the Stage 13 best and the in-place kernels, 1.5B |
| `stage14_ttft_p99_0p5b.png` | Time to first token, 99th percentile, against offered load, the baseline, the Stage 13 best and the in-place kernels, 0.5B |
| `stage14_ttft_p99_1p5b.png` | Time to first token, 99th percentile, against offered load, the baseline, the Stage 13 best and the in-place kernels, 1.5B |

<a id="head-to-head"></a>

## Head to head with vLLM

Source: [`results/vllm_compare.json`](results/vllm_compare.json)

vLLM 0.31.0 ran with the settings `defaults`, CUDA graphs on.

The trace digests are the same as in `stage12.json`.

- vLLM (newest): install succeeded, vllm 0.31.0 with torch 2.13.0+cu132
  - settings `defaults` (0.5B): started
  - settings `defaults` (0.5B): started
  - settings `defaults` (1.5B): started

<a id="head-to-head-load"></a>

### Under load

The same 64-request traces at each arrival rate, in one session. vLLM and this repo's server are separate processes reached over HTTP by the same client; Hugging Face `generate` runs in the client's process (one request at a time, and static batches of 8). The systems run in turns, in a different order at each rate. A request meets the latency target if its first token comes within 2.0 s and its tokens then come at least every 0.1 s; goodput counts only the tokens of requests that meet it.

| model | rate (req/s) | system | req/s | tok/s | TTFT p50 (s) | TTFT p99 (s) | TPOT p50 (ms) | stall p99 (s) | met target | goodput (tok/s) | errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 1.0 | vLLM | 1.07 | 72 | 0.02 | 0.06 | 5.9 | 0.01 | 100% | 72 | 0 |
| 0.5B | 1.0 | engine graph (HTTP) | 1.05 | 70 | 0.04 | 0.07 | 10.6 | 0.04 | 100% | 70 | 0 |
| 0.5B | 1.0 | HF sequential | 0.45 | 30 | 38.97 | 80.59 | 32.3 | 0.04 | 6% | 2 | 0 |
| 0.5B | 1.0 | HF static batch 8 | 0.90 | 61 | 4.34 | 13.73 | 31.2 | 0.04 | 28% | 22 | 0 |
| 0.5B | 2.0 | engine graph (HTTP) | 1.99 | 133 | 0.05 | 0.09 | 12.1 | 0.05 | 100% | 133 | 0 |
| 0.5B | 2.0 | HF sequential | 0.46 | 31 | 51.56 | 109.08 | 32.4 | 0.04 | 6% | 2 | 0 |
| 0.5B | 2.0 | HF static batch 8 | 1.15 | 77 | 12.65 | 20.45 | 31.0 | 0.04 | 6% | 4 | 0 |
| 0.5B | 2.0 | vLLM | 2.09 | 140 | 0.02 | 0.04 | 5.8 | 0.01 | 100% | 140 | 0 |
| 0.5B | 4.0 | HF sequential | 0.46 | 31 | 58.36 | 124.29 | 32.5 | 0.04 | 6% | 2 | 0 |
| 0.5B | 4.0 | HF static batch 8 | 1.31 | 88 | 15.05 | 27.47 | 31.0 | 0.04 | 9% | 7 | 0 |
| 0.5B | 4.0 | vLLM | 4.01 | 269 | 0.02 | 0.04 | 5.8 | 0.01 | 100% | 269 | 0 |
| 0.5B | 4.0 | engine graph (HTTP) | 3.41 | 228 | 0.06 | 0.11 | 18.4 | 0.06 | 100% | 228 | 0 |
| 0.5B | 8.0 | HF static batch 8 | 1.18 | 79 | 19.36 | 40.23 | 30.8 | 0.04 | 14% | 8 | 0 |
| 0.5B | 8.0 | vLLM | 7.39 | 495 | 0.02 | 0.03 | 6.1 | 0.01 | 100% | 495 | 0 |
| 0.5B | 8.0 | engine graph (HTTP) | 4.43 | 297 | 0.11 | 0.20 | 50.4 | 0.12 | 100% | 297 | 0 |
| 0.5B | 8.0 | HF sequential | 0.46 | 31 | 61.46 | 130.99 | 32.2 | 0.04 | 5% | 1 | 0 |
| 1.5B | 1.0 | vLLM | 1.02 | 68 | 0.06 | 0.20 | 16.7 | 0.04 | 100% | 68 | 0 |
| 1.5B | 1.0 | engine graph (HTTP) | 1.00 | 67 | 0.08 | 0.22 | 24.5 | 0.08 | 100% | 67 | 0 |
| 1.5B | 1.0 | HF sequential | 0.38 | 26 | 51.56 | 106.22 | 38.4 | 0.04 | 6% | 1 | 0 |
| 1.5B | 1.0 | HF static batch 8 | 0.83 | 56 | 5.94 | 14.98 | 36.9 | 0.04 | 16% | 7 | 0 |
| 1.5B | 2.0 | engine graph (HTTP) | 1.70 | 114 | 0.11 | 0.25 | 36.4 | 0.14 | 100% | 114 | 0 |
| 1.5B | 2.0 | HF sequential | 0.38 | 26 | 63.86 | 135.40 | 38.5 | 0.04 | 6% | 1 | 0 |
| 1.5B | 2.0 | HF static batch 8 | 1.04 | 69 | 12.11 | 24.68 | 37.3 | 0.04 | 9% | 5 | 0 |
| 1.5B | 2.0 | vLLM | 1.94 | 130 | 0.05 | 0.08 | 16.5 | 0.03 | 100% | 130 | 0 |
| 1.5B | 4.0 | HF sequential | 0.39 | 26 | 69.37 | 149.62 | 38.3 | 0.04 | 5% | 1 | 0 |
| 1.5B | 4.0 | HF static batch 8 | 1.18 | 79 | 21.65 | 38.17 | 37.1 | 0.04 | 11% | 6 | 0 |
| 1.5B | 4.0 | vLLM | 3.49 | 234 | 0.05 | 0.07 | 16.9 | 0.03 | 100% | 234 | 0 |
| 1.5B | 4.0 | engine graph (HTTP) | 2.26 | 151 | 0.19 | 0.37 | 96.2 | 0.22 | 53% | 116 | 0 |
| 1.5B | 8.0 | HF static batch 8 | 0.94 | 63 | 25.40 | 52.75 | 37.4 | 0.05 | 11% | 5 | 0 |
| 1.5B | 8.0 | vLLM | 5.66 | 379 | 0.06 | 0.08 | 18.2 | 0.03 | 100% | 379 | 0 |
| 1.5B | 8.0 | engine graph (HTTP) | 2.48 | 166 | 0.63 | 2.89 | 143.3 | 0.33 | 19% | 80 | 0 |
| 1.5B | 8.0 | HF sequential | 0.38 | 25 | 75.72 | 161.23 | 39.3 | 0.05 | 3% | 0 | 0 |

<a id="head-to-head-capacity"></a>

### Capacity under the latency target

The highest tested arrival rate at which at least 90% of the requests met the latency target ("none" means no tested rate did). A finite trace understates capacity, so this is a floor between tested rates, not an exact limit.

| model | system | capacity (req/s) |
| --- | --- | ---: |
| 0.5B | vLLM | 8.0 |
| 0.5B | engine graph (HTTP) | 8.0 |
| 0.5B | HF sequential | none |
| 0.5B | HF static batch 8 | none |
| 1.5B | vLLM | 8.0 |
| 1.5B | engine graph (HTTP) | 2.0 |
| 1.5B | HF sequential | none |
| 1.5B | HF static batch 8 | none |

<a id="head-to-head-latency"></a>

### Latency mode

One request at a time on the three text workloads of Stage 6, 128 new tokens each, median of 3 repeats per prompt. `accept rate` is accepted drafts over drafted tokens; `tokens/step` is the tokens made per decode step. The outputs of the engine rows are checked against plain decoding (identical, or a near-tie).

| model | workload | system | TPOT (ms) | TTFT (ms) | vs eager engine | accept rate | tokens/step | identical / near tie / runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5B | copy | vLLM | 6.2 | 21 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | code | vLLM | 6.2 | 21 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | open | vLLM | 6.3 | 20 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | copy | engine graph (HTTP) | 9.4 | 43 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | code | engine graph (HTTP) | 9.4 | 43 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | open | engine graph (HTTP) | 9.1 | 42 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | copy | HF sequential | 32.4 | 41 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | code | HF sequential | 32.8 | 41 | n/a | n/a | n/a | n/a / n/a / 6 |
| 0.5B | open | HF sequential | 32.4 | 41 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | copy | vLLM | 17.3 | 30 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | code | vLLM | 19.5 | 29 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | open | vLLM | 20.8 | 29 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | copy | engine graph (HTTP) | 19.2 | 76 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | code | engine graph (HTTP) | 19.3 | 75 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | open | engine graph (HTTP) | 18.8 | 63 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | copy | HF sequential | 39.2 | 51 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | code | HF sequential | 38.9 | 50 | n/a | n/a | n/a | n/a / n/a / 6 |
| 1.5B | open | HF sequential | 38.6 | 49 | n/a | n/a | n/a | n/a / n/a / 6 |

<a id="head-to-head-plots"></a>

### Plots

| File | What it shows |
| --- | --- |
| `vllm_compare_goodput_0p5b.png` | Goodput against offered load: vLLM, the engine's server and Hugging Face, 0.5B |
| `vllm_compare_goodput_1p5b.png` | Goodput against offered load: vLLM, the engine's server and Hugging Face, 1.5B |
| `vllm_compare_latency_0p5b.png` | Time per token for one request at a time on three workloads, vLLM, the engine's server and Hugging Face, 0.5B |
| `vllm_compare_latency_1p5b.png` | Time per token for one request at a time on three workloads, vLLM, the engine's server and Hugging Face, 1.5B |
| `vllm_compare_tpot_0p5b.png` | Median time per output token against offered load: vLLM, the engine's server and Hugging Face, 0.5B |
| `vllm_compare_tpot_1p5b.png` | Median time per output token against offered load: vLLM, the engine's server and Hugging Face, 1.5B |
| `vllm_compare_ttft_p99_0p5b.png` | Time to first token, 99th percentile, against offered load: vLLM, the engine's server and Hugging Face, 0.5B |
| `vllm_compare_ttft_p99_1p5b.png` | Time to first token, 99th percentile, against offered load: vLLM, the engine's server and Hugging Face, 1.5B |
