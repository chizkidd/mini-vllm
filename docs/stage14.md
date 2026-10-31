# Stage 14: The In-Place Paged Attention Kernel

[Overview](../README.md) | [Full report](../REPORT.md) | [Measured numbers](../RESULTS.md) | [How the code works](code.md)

> **Status.** Built, tested on a CPU and **measured on a Kaggle T4** (full run, no variant failed): [`13_paged_kernel.ipynb`](../13_paged_kernel.ipynb) saved `results/stage14.json`, and the numbers are in [`RESULTS.md`](../RESULTS.md#stage14). Both kernels compile, are exact and run inside the CUDA graphs. They win clearly when rows fill only part of their bucket, tie or lose at a full bucket, and move their bytes at a small share of the memory bandwidth; in the engine they take a modest share off the time per token at low rates and a large one at the top rate on the larger model. The eager fallback of Stage 13 turned from a help into a cost. The gap to vLLM narrowed again and did not close.

Stage 13 found that a batched decode step was mostly copies and removed the head expansion by folding the query heads. What is left is the **gather**: the blocks of every row are copied, through the block table, into one dense tensor of keys and one of values of a fixed width (the bucket), twice over, and only then does attention run. The cost of the gather is that of the bucket and the padding, not of each row's own length, so a step still gets slower as the load fills the buckets. The best engine of Stage 13 was still about three times vLLM's time per token at the top rate on the larger model, with a curve that rose with the load where vLLM's is flat.

This stage builds exercise 4 of Stage 12: attention that **reads the blocks where they are**. It is the idea of Lab 3's exercise `5.2`, an online softmax over blocks, as a Triton kernel inside the engine.

- [`13_paged_kernel.ipynb`](../13_paged_kernel.ipynb) checks the kernels against a reference on the GPU, gates them at the level of one layer (the folded gather path against each kernel, at several fills of the bucket), profiles one step, runs six variants of the engine on the traces of Stage 12 with the step log on, a closed-loop sweep, and serves the best variant over HTTP beside the vLLM rows. It saves `results/stage14.json`.

## What was built

| Piece | File | What it does |
| --- | --- | --- |
| The reference and the readable version | `minivllm/paged_attention.py` | `paged_attention_reference` (gather, then a softmax in fp32: the oracle) and `paged_attention_blockwise` (the kernels' algorithm in torch, one block at a time, also the CPU stand-in). |
| The kernels | `minivllm/paged_attention.py` | `simple`: one program per (row, query head) walks the row's own blocks through its block table, loads each block of keys and values from the pool, and combines the blocks with an online softmax in fp32. `grouped`: one program per (row, key-value head) loads each block once for all the query heads that share it and does the scores and the weighted sum with `tl.dot`, as flash-decoding kernels do. The number of blocks to visit is read from the device, so both run inside a CUDA graph. |
| The views | `minivllm/kv_cache.py`, `minivllm/paged_graph.py` | `write` (store the new key and value) and `attend` (the in-place attention) on the eager decode view and on the static view of the graph step; `update` is `write` plus the gather, unchanged. |
| The model | `minivllm/model.py` | For a one-token row with a view whose `attention` is not `"gather"`, `forward` writes and attends in place; prefill and the checking pass of speculation still gather. |
| The engine and server | `minivllm/llm_engine.py`, `minivllm/server.py` | `EngineConfig.attention` (`"gather"` default, `"paged"`, `"paged_grouped"`, `"paged_torch"`), validated; `--attention`; `engine_flags` knows it. |
| Measurement | `bench/attention_cost.py` | `attention_bytes(mode="paged", fill=...)` and the kernels in `time_attention_pieces`; the profiler groups the kernels into the attention family. |

```text
gather path   cache blocks --(index copy)--> [B, NB, n_kv, bs, hd] --(permute copy)--> [B, n_kv, NB*bs, hd] --> attention over every column, masked
in place      cache blocks --(read where they are, own blocks only)--> online softmax --> output         no copies, no padding blocks
```

## The decisions, and why

1. **The kernel reads each row's own blocks, and the count comes from the device.** The loop bound is `ceil(length / block_size)`, loaded from a tensor inside the kernel. A captured graph has static shapes but not static loop counts, so the kernel can sit in the graph, a padding row (length one, a table of the NULL block) visits one block, and a bucket wider than any row costs nothing.
2. **Two kernels, because which wins is a measurement.** The per-head kernel is the simplest thing that could work and reads a block once per query head of its group, relying on the cache. The grouped kernel loads once and serves the group from registers, at the price of padding the group to sixteen rows for `tl.dot`. Stage 11 taught that a kernel's layout is where its time goes, so both are kept and the gate chooses.
3. **A gate before the engine.** The first measurement is one layer's attention on random tensors of the real shapes, the folded gather path against each kernel at several fills. If neither wins at any realistic fill, the engine runs are expected to show nothing and the result is a no-go.
4. **The fill is the variable that matters.** With every row filling its bucket the two paths read the same keys and values and the kernel saves only the gather copies; with sparse rows it also skips the padding. The gate varies the fill, and the counted bytes (`attention_bytes(mode="paged", fill=...)`) say what the kernel should move.
5. **Prefill and the checking pass keep gathering.** The kernel is for one query token per row. A prefill uses the implicit causal mask and writes whole blocks, and the checking pass has several query rows with a causal pattern inside the new block (an exercise).
6. **Defaults are Stage 13's.** `attention` defaults to the gather and every earlier test passes unchanged; the in-place modes are opt-in until the results say they should be on.
7. **Tests of the algorithm run on a CPU, in Triton's interpreter.** Both kernels equal the reference over block sizes, ragged lengths, scattered tables, group sizes and head sizes; memory no row owns can be filled with NaN without changing the output; wrong kernels (no rescale, no length mask, a whole last block) are caught; the engine gives the tokens of plain decoding with each mode through the eager and the bucketed paths with every lever of Stage 13. What the interpreter cannot show is whether the kernels compile for a T4 and how fast they run.
8. **The variants test the levers of Stage 13 again.** With no padded gather, fine buckets and the eager fallback should matter less, so there are variants without each of them.

## How it is measured

- **Correctness first**, on the GPU: both kernels against the reference in fp16, on the head shapes of both models, with a different length in every row, scattered tables and unreachable memory poisoned with NaN; then every engine variant against plain decoding with the near-tie rule and a sampled control that must differ, the fallback variants with a threshold of two so a batch starts on the eager path and ends on the graph path.
- **The gate**, then **the variants under load** with the same constants and traces as Stages 12 and 13, one engine alive at a time, the baseline run twice as the drift check, **a closed loop**, and **the best variant over HTTP** beside the vLLM rows (read from `vllm_compare.json`, another session, so the shape of the curves is the safer reading).
- **Each variant runs inside a `try` block** that records the error and carries on, as do the gate rows, the profile and the server run.
- **The notebooks are numbered one below the stages.** The notebook and its plots call the engines "Part 11" (the Stage 12 engine), "Part 12" (the best engine of Stage 13) and "Part 13" (this stage). The rows of the first two and of vLLM are read from `vllm_compare.json` and `stage13.json`, measured in other sessions.

## Predictions (Stage 14)

Written before any GPU run, qualitative on purpose.

1. Both kernels agree with the reference to within fp16 rounding on the GPU, and no NaN reaches the output. Every greedy variant equals plain decoding up to near-ties; the in-place variants are the likeliest to show one.
2. At the level of one layer both kernels beat the folded gather path at low fill and the gap narrows as the fill rises to one, where the gain is only the two gather copies. The counted bytes of the in-place read are a small fraction of the gather path's at low fill; the measured time follows them less closely than the library's did in Stage 13, because a hand-written kernel on a T4 does not reach the library's bandwidth.
3. The grouped kernel beats the per-head one on the larger model, where six query heads share a key-value head; the two are closer on the smaller one.
4. In a profile of a whole step the copy family shrinks to a fraction and the matrix products become the largest family, so the step is faster but not by the margin of the layer.
5. Under load both kernels beat Stage 13's best engine at the top rate on both models, by less on the smaller one, and are close to it at the lowest rate.
6. With no padded gather, powers of two instead of fine buckets, and no fallback, should cost little: those variants are close to the best kernel.
7. The time per token still rises with the load, but less; the capacity of the larger model is no lower than Stage 13's.
8. Over HTTP the best variant narrows the gap to vLLM and does not close it: the weights are read in full by every step.
9. A no-go is possible. If the kernels do not compile for the T4, or are slower than the gather at every fill, that is the result, and the notebook records it.

## Results (Stage 14)

Full run on a Kaggle T4, nothing failed. The numbers are in [`RESULTS.md`](../RESULTS.md#stage14), from `results/stage14.json`.

### 1. Are the kernels exact? Yes.

Both kernels equal the reference to within fp16 rounding on the head shapes of both models (the grouped kernel a little less closely than the per-head one), and filling every slot that no row owns with NaN changes nothing, so neither reads past a row's length. Every greedy variant equals plain decoding on every prompt, except one near-tie in the baseline of the smaller model; the sampled control differs, and the fallback variants took both the eager and the graph path. See [the correctness tables](../RESULTS.md#stage14-correct).

### 2. The gate: the kernels win on sparse rows and not on full ones.

Against the folded gather path of Stage 13, at the level of one layer, both kernels are about two to five times faster when rows fill a quarter of their bucket and about one to three and a half times at half a bucket. At a full bucket the advantage is gone: on the larger model neither kernel beats the gather path by a tenth, and at the largest shape both lose. The prediction that the kernel would still save the two gather copies was wrong: it reads the blocks more slowly than the library reads the gathered tensor. The counted bytes of the in-place read are a twentieth, a tenth and a fifth of the gather path's at those fills, and the measured gain is under half of that ratio everywhere, because the kernels move their bytes at about a tenth of the T4's specified bandwidth while the gather path moves its counted bytes five or six times faster. They are untuned (exercise 1).

Which kernel is faster depends on the batch: the per-head kernel is ahead at a batch of four, where it has more programs to fill the GPU, and the grouped kernel is faster from a batch of sixteen on both models. See [the gate table](../RESULTS.md#stage14-gate).

### 3. One whole step under the profiler: the copies go, the attention rises, and the matrix products are now the largest family.

On the larger model, at a batch of 32 rows of 32 blocks, half filled, the copy family falls to a fraction of a tenth, the attention family grows to several times its size (the kernel does what the copies and the library kernel did together), the matrix products are unchanged and now the largest family, and the step is about a fifth shorter. On the smaller model the same step is **slower** with the kernel, in both full runs of the notebook: the kernel costs more than the copies it removes, and the matrix-product family is also longer in that profile, for a reason not found. This is an extreme step (32 rows with long contexts); under load most steps are small. See [the profile](../RESULTS.md#stage14-profile).

### 4. The variants under load: a modest gain at low rates, a large one at the top rate on the larger model.

The baseline run twice differs by about five percent at worst (smaller model, top rate) and by about one percent or less on the larger model; a gap smaller than that is not a finding, so the smaller model at the top rate is read with care. A first full run of the same notebook (Version 1 on Kaggle, in the history of this repository) gave the same ranking at the top rate of the larger model, but the two kernel variants that keep the fallback differed by about a tenth between sessions where the baseline drift was a percent: the claims check rankings and wide margins, not exact figures.

- **Stage 13's result reproduces.** The baseline to K0 lowers the time per token at the top rate by more than half on both models, and the baseline of the larger model meets the latency target for only a part of the requests at the two highest rates.
- **Both kernels beat K0 at the top rate on the larger model, and the per-head kernel does at every rate on both models**: by a few percent at low rates, about a tenth on the smaller model and about a quarter on the larger one at the top rate. Most of the gain over the original engine had already come from folding.
- **The grouped kernel does not win under load**, against the prediction: the per-head kernel is ahead of it by a few percent in every cell. Most steps carry few rows (about nine in ten of the smaller baseline's steps have four rows or fewer) and at those batches the per-head kernel is ahead at the gate. The rule that picked the grouped kernel as the faster one looked at the largest batch.
- **Fine buckets no longer matter**: powers of two are within a few percent of fine buckets at every rate on both models.
- **The eager fallback now costs.** K4 (no fallback) equals the variants with the fallback up to the middle rates and is about a sixth lower at the top rate on the larger model, the lowest of all variants; the variants with the fallback sent a sixth to a fifth of their steps to the eager path. The threshold was read from the step log of the gather path; with the kernel a graph step is cheap at larger batches, so a step sent to the eager path is now the slower choice.
- **Capacity no longer separates the variants**: every variant but the baseline reaches the highest tested rate on the larger model. The time per token still rises with the load.

See [the load table](../RESULTS.md#stage14-load), [capacity](../RESULTS.md#stage14-capacity) and [the drift check](../RESULTS.md#stage14-drift).

### 5. The closed loop: flatter, not flat.

With one request in flight the variants are alike. At the top concurrency both kernels are fastest and the baseline slowest, with about a tenth more tokens per second than K0 and about a fifth more than the baseline, and the two kernels are tied. The time per token from one to the top concurrency grows by about three times for the baseline and by less for the kernels: the rest of the rise is the matrix products and the other per-row work of a bigger batch. See [the closed loop](../RESULTS.md#stage14-closed).

### 6. How does the best variant compare with vLLM? Closer again, not equal.

Over HTTP the best variant (the grouped kernel without the fallback on the larger model, the per-head kernel on the smaller) answers every request within the target at every rate, and costs little on top of the in-process runs except at the top rate on the larger model. At the top rate it beats the Stage 13 best at both sizes and cuts the Stage 12 engine's time per token to a third or less. On the larger model its time per token went from about eight times vLLM's (the Stage 12 engine) to between three and four times (the Stage 13 best) to between two and three times, and it is still over twice vLLM's at the top rate. The shape is the finding: vLLM's curve is flat, and this engine's still rises with the load (the rows of the other systems are from other sessions, so the shape is the safer reading). See [the HTTP table](../RESULTS.md#stage14-http).

### Predictions against results (Stage 14)

1. *Both kernels agree with the reference on the GPU and every greedy variant is exact up to near-ties.* Held.
2. *At one layer both kernels beat the folded gather path at low fill and the gap narrows towards a full bucket, where the gain is the two copies; the counted bytes are a small fraction; the time follows them less closely.* Half held: the gap narrows, but at a full bucket the kernels tie or lose, so the gain is not even the copies. The bytes and the looser time held.
3. *The grouped kernel beats the per-head kernel on the larger model, and the two are closer on the smaller.* **Wrong** as stated: it depends on the batch (grouped from sixteen, per head at four), and under load the per-head kernel was as good or better on both models.
4. *In a step the copy family shrinks and the matrix products become the largest family.* Held on the larger model; **wrong** on the smaller, where the step got slower.
5. *Under load both kernels beat Stage 13's best at the top rate, by less on the smaller model, and are close at the lowest rate.* Held (the grouped kernel's gain on the smaller model is within the drift).
6. *Powers of two and no fallback are close to the best kernel.* Half held: powers of two are; **no fallback is the best variant at the top rate**, because the fallback now costs.
7. *The time per token still rises, but less, and the capacity is no lower.* Held; the capacity of every variant but the baseline is at the top of the sweep, so it no longer discriminates.
8. *Over HTTP the gap to vLLM narrows and does not close.* Held.
9. *A no-go is possible.* Not triggered: the kernels compile, are exact and win under load. But the gate shows the win only on partly filled rows, and the kernels are far from the bandwidth.

### What this sets up (Stage 14)

The kernels are untuned, at about a tenth of the bandwidth, so tuning (exercises 1 and 2) is the largest measured gain left. The defaults should become the in-place attention without the eager fallback; a batch-dependent choice (per head below sixteen rows, grouped above) is a small exercise the step logs justify. The comparison with vLLM should be rerun with this configuration in the same session (`99_vllm_compare.ipynb`), so it is no longer across sessions. After that the matrix products are the largest family of a step, and the weights are read in full every step: larger batches, speculation and quantized weights are the levers left.

## What this does not do

- **Only one query token per row.** Prefill and the checking pass of speculation still gather.
- **No tuning.** Block sizes, warps and the number of key positions loaded per iteration are fixed by hand (an exercise); Stage 10 and 11 showed how much a tuning run can move a kernel.
- **No split of a long row across programs** (flash-decoding), so one very long row is one program.
- **One model family, one GPU, one workload.** The same Qwen2.5 sizes, the `mixed` workload with random token ids, one T4.
- **The defaults stay as they were.** The results say the in-place attention without the eager fallback is best at the top rate; the defaults change with the next code stage, because the published notebook carries the current modules. The comparison with vLLM crosses sessions.

## Lessons

What the results taught:

- **Run the gate at the batches the load runs.** The rule that chose the grouped kernel looked at the largest batch; the load runs mostly small steps, where the per-head kernel is ahead.
- **Counted bytes are not time.** A kernel that moves a twentieth of the bytes can be only a few times faster, because it moves them at a tenth of the bandwidth. Compute the rate the kernel reaches.
- **A kernel can win a layer and lose a step.** On the smaller model the profiled step got slower with the kernel, though the layer-level gate and the load runs favour it. Measure at all three levels.
- **A threshold read from one path's logs goes stale when the path changes.** The eager fallback was set from the step log of the gather path, and it cost once the kernel made the graph step cheap.
- **Between sessions the top rate moves more than the drift check says.** The baseline passes of a session agree to a percent, while the variants near overload moved by about a tenth between two runs: read rankings and wide margins.

What the build taught:

- **The algorithm is testable without the hardware.** The kernels' online softmax, their masks and their block-table arithmetic were checked in Triton's interpreter against a plain reference, with mutations (no rescale, no length mask, a whole last block) that the comparison catches. What remains for the GPU is whether the kernels compile and how fast they are.
- **Memory a row does not own must not be read.** Filling every unreachable slot with NaN is a cheap test that a masked load really is masked.
- **Make the loop count a device value.** A captured graph cannot have a data-dependent shape, but a kernel can have a data-dependent loop, which is what lets padding cost nothing.
- **Keep the simple kernel next to the clever one.** The per-head kernel is the control for the grouped one.

## Exercises

1. **Tune the kernels.** Load several `block_size` slots of keys per loop iteration and vary the number of warps; read the best setting from a measurement, as in Stage 10.
2. **Split a long row.** Divide the key positions of one row across several programs and combine their partial softmaxes (flash-decoding); find the length where it pays.
3. **Fuse the write.** Write the new token's key and value inside the kernel, so they are written and read in one launch.
4. **Wider groups and other head sizes.** Let the grouped kernel serve more than sixteen query heads per key-value head and a head size that is not a power of two.
5. **The checking pass.** Use the kernel for the several query rows of the checking pass of speculation, with the causal pattern inside the new block.
6. **An int8 cache.** Quantize the KV cache and read it in the kernel (exercise `8.6`); measure at long contexts.
7. **The closed loop against vLLM.** Run `run_closed_http` against vLLM and compare the slopes of time per token with this engine's variants.
8. **Find the next bottleneck.** Profile the best variant at the top rate and name the largest family; is it the matrix products, and what would shrink them?

## How to run it

- **The notebook.** Run [`13_paged_kernel.ipynb`](../13_paged_kernel.ipynb) on a Kaggle T4 (Internet on, the secret `minivllm-github-token` **attached before pressing run**). Set `QUICK = True` and `PUSH = False` first: it runs every variant, the closed loop and the server section on the 0.5B model with fewer requests and two rates, in about ten minutes. The full run takes about an hour; restart the kernel before it.
- **One kernel by hand.**

  ```python
  from minivllm.paged_attention import paged_attention
  out = paged_attention(q, cache.k[layer], cache.v[layer], tables, lens, kernel="grouped")     # q [B, H, hd], tables [B, NB], lens [B]
  ```

- **The engine.** `EngineConfig(graph_decode=True, pad_head=True, gqa="fold", bucket_scheme="fine", attention="paged_grouped")`, or `python -m minivllm.server --graph --warmup --gqa fold --bucket-scheme fine --attention paged_grouped`.
