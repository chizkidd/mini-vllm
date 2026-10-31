# Stages 1 & 2: Baseline & KV Cache

[Overview](../README.md) | [Full report](../REPORT.md) | [Measured numbers](../RESULTS.md) | [How the code works](code.md)

[`01_baseline_and_kv-cache.ipynb`](../01_baseline_and_kv-cache.ipynb) is a guided walkthrough. Each section follows the same shape (Goal, Concept, Code, Result, Takeaway), and you are asked to write down a prediction before every benchmark. The surprising results are the point.

The notebook is a thin driver. Cells that start with `%%writefile` write the `minivllm/` and `bench/` packages, and the notebook then imports them, so the package is the source of truth.

| # | Section | What you learn | Runtime on a Kaggle T4 |
| --- | --- | --- | --- |
| 1 | Setup | Reproducible environment record (`ENV`), `QUICK` flag | seconds |
| 2 | The code | KV cache, model, engine, benchmark helpers | seconds |
| 3 | Load model and parity gate | Prove our forward pass matches Hugging Face before benchmarking | about 1 min |
| 4 | Stage 1: naive baseline | Cost of recomputing everything | about 20 s |
| 5 | Stage 2: KV cache | Prefill vs decode | about 25 s |
| 6 | Roofline ceiling | Why decode is memory-bound | seconds |
| 7 | Crossover analysis | Where does the cache start to win? | about 2 min (about 6 with the probe sweep) |
| 8 | Prefill scaling | TTFT, prefill throughput and decode ITL vs prompt length | about 30 s |
| 9 | `torch.compile` | Closing the gap to the ceiling | about 2 min |
| 10 | Roofline chart | The arithmetic-intensity picture | seconds |
| 11 to 13 | Save results, optional publish, summary | JSON plus figure inventory, what we learned | seconds |

## Results (Stages 1 & 2)

All measured values are in [`RESULTS.md`](../RESULTS.md) (source: [`results/stage1_2.json`](../results/stage1_2.json)). The environment for the run (GPU, library versions, model revision) is listed under [Setup](../RESULTS.md#stage1_2-setup). This section explains what the numbers mean.

**Fixed conditions**

- **Model:** `Qwen/Qwen2.5-0.5B` in fp16, because the T4 has no bf16.
- **Architecture:** 24 layers, 14 query heads, 2 KV heads, head_dim 64.
- **KV cache cost:** 12 KiB per token.
- **Workload:** batch 1, greedy decoding, a random prompt.
- **Timing:** medians over repeated runs after a warm-up, with `cuda.synchronize()` around every timer.

### 1. Is our forward pass correct?

Nothing is benchmarked until it matches Hugging Face. On a real prompt the notebook asserts that the last-position logits agree within fp16 noise, and that the greedy tokens of Hugging Face, `generate_naive` and `generate_cached` are identical over 128 steps. Cached and naive must agree exactly because they compute the same math. Any mismatch means the cache changed the result, which is a bug.

### 2. Is the KV cache faster? Not at first.

Numbers: [Naive vs cached generation](../RESULTS.md#stage1_2-naive-vs-cached).

Cached throughput is flat, because every decode step does the same work. Naive throughput holds for short outputs and then falls as each step recomputes a longer sequence. Below the crossover the cache is slower, because at batch 1 on a 0.5B model both paths are dominated by Python and kernel-launch overhead (24 layers times many small ops), not by FLOPs. Recomputing a short sequence costs about the same as computing one token, so the cache has nothing to save until recomputation gets large.

**2a. Where do they cross?** Three estimates come from the extended sweep: interpolating the speedup on a linear axis, interpolating on a log2 axis, and fitting quadratic naive time against linear cached time (plotted). The sweep has only a few, widely spaced lengths, so these are coarse, and they bracket a range rather than a single point. See [Crossover](../RESULTS.md#stage1_2-crossover) for the values.

![Naive vs cached with fitted crossover](../results/stage1_2_crossover_quadratic_fit.png)

**2b. Does a direct measurement agree?** A dense probe around the predicted crossover measures the speedup directly and interpolates where it crosses one. It is the slowest step in the notebook, so `QUICK = True` skips it, and the crossover table says so when it was skipped. Trust the probe over the sweep-based estimates, which interpolate across a wide gap: in the latest run the probe landed below all three of them. The two can disagree because the speedup curve is shallow near the crossing (it climbs only slowly with output length), so a few percent of measurement noise moves the crossing by tens of tokens. Quote a range, not a point.

![Naive vs cached, zoomed near the crossover](../results/stage1_2_crossover_probe.png)

### 3. Prefill and decode behave differently

Numbers: [Prefill and decode](../RESULTS.md#stage1_2-prefill-decode).

Two different latencies appear in this section, and they are not the same thing:

- **TTFT** (time to first token) is the **prefill**: one forward pass over the whole prompt. It depends on prompt length.
- **ITL** (inter-token latency) is the **decode** step: one single-token forward pass per generated token, reusing the KV cache.

![TTFT and prefill throughput vs prompt length](../results/stage1_2_ttft_and_throughput.png)

**How to read it.** The top panel is time to first token. It is flat across short prompts, so short prefills are launch-bound like decode: a forward pass over hundreds of tokens takes about as long as a forward pass over one token, which means the time is overhead, not math. The bottom panel (same prompt-length axis) is prefill throughput, which therefore climbs almost linearly with prompt length while TTFT stays flat. At the longest prompt TTFT jumps and throughput drops, which is where real compute starts to matter. Processing more tokens per weight read is nearly free at small sizes, which is the case for batching.

![TTFT and decode ITL vs prompt length](../results/stage1_2_ttft_itl_vs_prompt.png)

**Decode does not care about context length here.** On a shared axis, TTFT takes off at the longest prompt while ITL stays a flat line, even though the KV cache grows by a factor of 64 across the sweep. Each decode step is dominated by reading all the weights and launching kernels. The extra KV bytes are a small fraction of what a step already reads, so the step time barely moves. The flat line is the result, not a plotting artifact. It holds up to the longest context measured.

### 4. How close is decode to the hardware limit?

Numbers: [Roofline](../RESULTS.md#stage1_2-roofline).

Batch-1 decode reads every weight plus the KV cache once per token, so the ceiling is `bandwidth / bytes per token`. The roofline table compares eager decode, `torch.compile`d decode and that ceiling, using both the spec bandwidth and a measured copy bandwidth.

`torch.compile(mode="reduce-overhead")` on the single-token decode step roughly doubles decode throughput for a one-time compile cost. Eager and compiled are measured the same way (p50 inter-token latency through `generate_cached`), so the comparison is fair.

On the roofline, decode does about one FLOP per byte, against a ridge point in the hundreds of FLOPs per byte. It is deep in the memory-bound region, so extra compute would not help. Both operating points sit well below the bandwidth roof, so the remaining gap is overhead, not bandwidth.

![Decode roofline on a Tesla T4](../results/stage1_2_roofline_model.png)
![Eager vs compiled decode](../results/stage1_2_compile.png)

### What this sets up (Stages 1 & 2)

Batch-1 decode uses a tiny share of the compute the GPU offers. Serving many sequences at once amortizes each weight read across all of them and raises arithmetic intensity. Stage 3 measures how much that helps, and what padding costs.

### Other plots (Stages 1 & 2)

The plots not embedded above are listed, with captions, under [Plots](../RESULTS.md#stage1_2-plots) in `RESULTS.md`.

## Lessons

- **A KV cache is not a free win at small scale.** Decode costs the same per step whether the step recomputes a short sequence or one token, because both are overhead-bound. The cache only pays off past a crossover that is several hundred new tokens here.
- **Short prefills are launch-bound as well.** TTFT is flat across short prompts, so prefill throughput climbs with prompt length, and only for long prompts does TTFT grow and throughput fall.
- **Decode ITL is independent of context length in this range.** It stays flat across the prompt sweep while the KV cache grows a lot.
- **Eager decode reaches a small share of the bandwidth ceiling.** `torch.compile` roughly doubles that share. CUDA graphs are the natural next test.
- **Decode is memory-bound by construction** (about one FLOP per byte against a ridge in the hundreds). Batching, not faster math, is the lever.
- **Benchmark hygiene matters.** Warm up, synchronize before timing, report medians, and restart the kernel before final runs (stale tensors cause fake OOMs and `torch.compile` recompilation stalls).
- **Run-to-run variance is real.** Repeated full runs on the same T4 move the crossover estimates by a large margin, along with the cached step time and the compile speedup (shared hosts and GPU clocks differ between sessions), and the sweep-based and probe estimates can disagree within one run. Quote ranges, not single points, and prefer the direct measurement.

## Exercises

1. Repeat the crossover sweep with `PROMPT_LEN = 512`. Does the crossover move left or right? (A longer prompt makes every naive step more expensive.)
2. Replace `repeat_interleave` in `model.py` with `enable_gqa=True` in `F.scaled_dot_product_attention` and re-measure ITL.
3. Capture the decode step in a CUDA graph by hand (`torch.cuda.CUDAGraph`) and compare with `mode="reduce-overhead"`.

## How to run it

1. Open `01_baseline_and_kv-cache.ipynb` on Kaggle with Accelerator = GPU T4 and Internet = On.
2. Run top to bottom.
3. Restart the kernel before the final benchmark run. The `torch.compile` section has a one-time warm-up of about a minute.
4. `QUICK` in the setup cell is `False` in the shipped notebook, so the dense probe sweep runs (about 4 extra minutes) and produces the probe figures. Set it to `True` for a faster run that skips them.
5. The publish cell at the end pushes results to GitHub, and code only if this is the newest stage, so rerunning this notebook later publishes results only. It needs a Kaggle secret named `minivllm-github-token` holding a GitHub token, and `PUSH = False` skips it.

