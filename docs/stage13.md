# Stage 13: The Batched Step

[Overview](../README.md) | [Full report](../REPORT.md) | [Measured numbers](../RESULTS.md) | [How the code works](code.md)

> **Status.** Built, tested on a CPU, and run on a Kaggle T4: [`12_batched_step.ipynb`](../12_batched_step.ipynb) completed with no failed variant, and its results are below. The comparison with vLLM uses the rows of an earlier session (see the decisions); a same-session rerun of [`99_vllm_compare.ipynb`](../99_vllm_compare.ipynb) with the best configuration is the next step.

Stage 12 ended with a plain fact and an unproven explanation. Under load vLLM's time per output token stays nearly flat as the offered load rises, while the engine's rises several times over, and on the larger model the engine's capacity is a quarter of vLLM's. The explanation was a hypothesis, with no per-step profile behind it: a captured step reads every block of its bucket for every row, padding rows included, and `MiniQwen` then copies the keys and values of each grouped-query head up to every query head before attention, so the bytes a step moves grow with the batch times the padded length. A slower step keeps more requests running, which fills a larger bucket, which makes the step slower again.

This stage measures that, and then fixes it with three changes that each aim at one part of the cost. They are run one at a time against the same traces and then together, because the point of an ablation is to say which change did what. It is exercise 5 of Stage 12, built in full.

- [`12_batched_step.ipynb`](../12_batched_step.ipynb) counts the bytes of one layer's attention, times its pieces and profiles a whole step; runs seven variants of the engine on the traces of Stage 12 with the per-step log on; reads the fallback threshold from that log; runs a closed-loop concurrency sweep; and serves the best variant over HTTP next to the vLLM rows of [`99_vllm_compare.ipynb`](../99_vllm_compare.ipynb). It saves `results/stage13.json`.

## What was built

| Piece | File | What it does |
| --- | --- | --- |
| Lever 1, fold | `minivllm/model.py` | `gqa_attention(q, k, v, mask, n_rep, fold)`. With `fold` the query heads that share a key-value head are moved into the query rows (`[B, n_kv, n_rep * T, hd]`), the mask is repeated along the rows, and attention runs against the keys and values as they are in the cache: nothing is copied to `n_rep` heads. The causal case (prefill, no explicit mask) keeps the expansion. `MiniQwen.forward(..., gqa="expand" or "fold")`. |
| Lever 2, fine buckets | `minivllm/paged_graph.py` | `buckets(limit, scheme)` with a `"fine"` scheme: each size at most a quarter above the one before it once past the smallest sizes, so a step pads far less than with powers of two, at the price of more graphs to capture. `PagedGraphs(..., gqa, bucket_scheme)`, and `warmup(..., max_blocks)`, which captures only the block buckets a workload can reach. |
| Lever 3, eager fallback | `minivllm/llm_engine.py` | `EngineConfig.graph_max_batch`: a decode step with more running requests than this takes the eager paged step instead of the captured one. Also `EngineConfig.gqa` and `.bucket_scheme`. All three are validated, all three default to Stage 12's behavior, and every decode entry of the step log now says which path ran. |
| The server flags | `minivllm/server.py` | `--gqa`, `--bucket-scheme`, `--graph-max-batch`, and `engine_flags(**config)`, which turns a measured `EngineConfig` into the flags that serve it, so a configuration measured in process can be served over HTTP exactly as it was measured. |
| Measurement | `bench/attention_cost.py` | `attention_bytes` (the bytes one layer's attention moves, from the shapes), `time_attention_pieces` (the gather, the expansion and both attentions timed on random tensors of the real shapes inside one CUDA graph), `profile_step` (a profiler view of a callable with kernels grouped into families), `step_table` and `step_curves` (the step log reduced to median step time by running batch), and `crossover_batch` and `fallback_setting` (the running batch from which the eager step is no slower than the captured one, and the engine setting that follows). |
| Closed-loop load | `bench/load.py` | `run_closed` and `run_closed_http`: a fixed number of requests in flight, a new one the moment one finishes, in process and over HTTP. |

One layer's attention in a captured decode step does three things to the cache, and the stage counts all three:

```text
gather   the block table picks each row's blocks out of the paged cache and lays them out as [B, n_kv, NB * block_size, hd]
         (keys and values, two copies each: the index, then the permute that joins the block axis to the position axis)
expand   repeat_interleave copies each key-value head to the n_rep query heads that share it           <- lever 1 removes this
attend   the attention kernel reads the keys and values it is given, n_rep times as many after the expansion
```

## The decisions, and why

1. **Folding is a layout change, not an approximation.** Query head `h` uses key-value head `h // n_rep`, so the query tensor `[B, H, T, hd]` is already `[B, n_kv, n_rep, T, hd]` in memory, and viewing it as `[B, n_kv, n_rep * T, hd]` lets one attention call serve the whole group. A test checks that the folded and the expanded attention agree on random tensors (with a different valid length per row, with a staircase mask, for several `n_rep`), that a head-grouping mistake would be caught, and that every engine path produces the tokens of plain decoding with either setting. Prefill keeps the expansion, because it uses the implicit causal mask and an explicit one would cost more than it saves.
2. **Fine buckets are only worth their graphs if the warm-up is restricted.** A finer scheme has several times as many (batch, blocks) pairs, so `warmup` now takes `max_blocks` and captures only the block buckets a workload can reach, instead of every bucket up to the pool. The capture time is reported on its own (it is paid when a server starts, not per request).
3. **The fallback threshold is read from the step logs, not guessed.** `crossover_batch` pools the decode steps of all four rates and, for every running batch size that both logs saw often enough, compares the medians of the captured and the eager step. The answer has three shapes, and the code names them: no crossover (the captured step is never slower: no fallback), a crossover of one (the eager step wins everywhere: no graphs), or a batch size `r` (fall back above `r - 1`). It is read twice, for the expanded path and for the folded one, because after folding there may be nothing left to fall back from.
4. **One engine is alive at a time, and the baseline is run twice.** Seven engines with their graphs would not all fit with the model, and a fixed order of passes lets a slowly drifting GPU clock favor the early ones. So each pass builds its own engine, warms the GPU under load and logs its state, and the baseline (V1) is run again at the end. The difference between its two passes is the drift of the session, and a gap between two variants smaller than that is not a finding.
5. **A closed loop is added next to the open one.** The open-loop load is still the test of a server, but the batch it produces depends on the engine's own speed, which muddies the view of a single step. A closed loop holds the number of requests in flight fixed, so the running batch is known, no queue builds, and the time per token against concurrency is the cost of a batched step per row. The same sweep can be given to any server over HTTP.
6. **The best variant is chosen by a rule written before the run**, not by looking at the plots: the highest capacity at the target, then the lowest median time per token at the highest rate, first passes only. It is then served over HTTP with the flags `engine_flags` makes from the exact configuration that was measured.
7. **The comparison with vLLM crosses sessions, and says so.** The vLLM rows are read from `results/vllm_compare.json`, which another session produced; a GPU is not the same machine from one session to the next, so the level of a gap to those rows is weaker evidence than the shape of a curve. The comparison made inside one session is the in-process one.
8. **A failure of one variant must not cost the hour.** Each variant runs inside a `try` block that records the error in the results and carries on, as do the microbenchmark shapes, the profile and the server run. The later cells use what exists.
9. **The bytes model counts what the code does.** A first back-of-envelope count treated the gather as one copy of the keys and values. The code makes two (the block-table index and the permute that joins the axes), so the model counts both. The model is an estimate (kernels fuse some traffic, and the mask is ignored), which is why the stage also times the pieces and profiles a real step.

## How it is measured

- **Parity before speed.** Seven configurations (the five single-lever variants, and the two that use the fallback with a small threshold so a batch of twelve starts on the eager path and ends on the graph path) are compared with plain cached decoding. A mismatch counts as an fp16 near-tie only if the logit gap at the first difference is small, a sampled control must differ so the comparison can fail, and the fallback variants must show both paths in their step log, or the check did not exercise the fallback.
- **The same traces as Stage 12.** The constants of `11_engine.ipynb` and `99_vllm_compare.ipynb` (workload `mixed`, the same number of requests, rates and seeds), so the trace digests are the same, and the notebook checks them against `vllm_compare.json`.
- **The step log is on.** Every run records, per decode step, the number of running requests, the bucket and which path ran. The notebook saves the median step time by running batch for every pass.
- **The bytes, then the time, then a profile.** Counted bytes of one layer for both paths over a grid of batch and length; the same pieces timed on random tensors inside a CUDA graph; one whole decode step under the profiler, with the kernels grouped into attention, matmul, copies and indexing, and elementwise work.
- **Capacity and goodput** as in Stage 12: the highest tested rate at which at least nine tenths of requests meet the latency target.

## Predictions (Stage 13)

Written before any GPU run, qualitative on purpose (the numbers will come from the run). The variants are V0 (expanded attention, eager step), V1 (expanded, graph: the Stage 12 baseline), V2 (folded, eager), V3 (folded, graph: lever 1), V4 (expanded, graph, fine buckets: lever 2), V5 (V1 plus the fallback: lever 3) and V6 (all three).

1. Every greedy variant equals plain decoding up to near-ties, the folded ones being the likeliest to show one (the folded attention launches a different kernel shape), and the sampled control differs. The fallback variants take both paths.
2. The expansion and the read of its copy are the largest part of the attention's bytes at every shape, so folding removes most of them, and the ratio of the two paths' bytes depends on the number of query heads per key-value head and not on the batch or the length. The timings follow the bytes where the working set is large and flatten at small shapes, where launches and fixed costs dominate even inside a graph. In a profile of a whole step the attention work is a modest share at a small bucket and the largest at a large one, and folding shrinks the second and not the first. The two paths agree numerically up to fp16 rounding.
3. V1 against V0 reproduces Stage 12: on the larger model the graph step has the lower time per token at the lowest rate and the higher one under load.
4. **Lever 1 (V3)** lowers the time per token under load on both models. On the larger model the graph step is no longer slower than the eager one at any rate. On the smaller model the graph keeps its lead at the highest rate, where it lost in Stage 12.
5. **V2 gains little over V0**: the eager step gathers only the columns the longest request needs, so the expansion is a smaller share of its work.
6. **Lever 2 (V4) helps less than lever 1.** Finer buckets cut padding rows and padding columns, but the expansion still multiplies whatever is gathered by `n_rep`.
7. **Lever 3 (V5)** helps the larger model at the highest rates and changes nothing at the lowest, where the running batch never rises above the threshold. The expanded path has a crossover at a middling batch on the larger model. After folding the captured step is cheaper than the eager one at every batch, so the folded path has no crossover on either model, and V6 reduces to folding plus fine buckets.
8. V6 has the lowest time per token under load on the larger model, and its gain over V3 is small. The drift between the two baseline passes is small next to the gap between V1 and V3.
9. In a closed loop at a concurrency of one the variants are within noise of each other, with the graph variants ahead of the eager one. As the concurrency grows, V1 rises fastest, V0 more slowly, V3 and V6 least, and tokens per second rise with the concurrency for every variant, most for the folded ones.
10. Over HTTP every configuration is slower than in process, by the cost of the rest of the stack. The best variant narrows the gap to vLLM that Stage 12 reported (its time per token rises less with the load, its capacity on the larger model is higher) and does not close it: vLLM's time per token stays flat, and the engine's still rises, because a captured step still gathers every block of its bucket for every row and the weights are read in full whatever the bucket.

## Results (Stage 13)

The numbers are in [`RESULTS.md`](../RESULTS.md#stage13-setup), from `results/stage13.json`: one Kaggle T4 session, the 64-request traces of Stage 12 at four rates, the 0.5B and 1.5B models.

### 1. Is every variant exact? Yes, and the fallback was exercised.

All greedy variants equal plain decoding (a few differ only at a near-tie), the sampled control differs, and the fallback variants took both paths. See [parity](../RESULTS.md#stage13-parity).

### 2. Where do the bytes go? Into copies, and folding removes most of them.

The folded path moves under a third of the counted bytes of the expanded one, and the measured time of a layer follows it at large shapes and not at the smallest. The two attentions agree up to fp16 rounding. In a profile of a whole expanded step on the larger model, copying and indexing (the gather, the expansion, the cache writes) is the largest family of kernels, ahead of the matrix products; folding cuts the attention family to under half and the copy family by more than half, and leaves the matrix products where they were. So the hypothesis of Stage 12 was right about the cause. See [bytes](../RESULTS.md#stage13-bytes) and [profile](../RESULTS.md#stage13-profile).

### 3. What does each lever buy? Folding most, and the three together win.

- **Lever 1.** On the larger model folding cuts the graph step's time per token at the top rate to about half and at least doubles the capacity; it also helps the eager step, which the prediction said it would not. On the smaller model it turns the top-rate loss of the graph step into a large win.
- **Lever 2.** Finer buckets alone help, but less than folding.
- **Lever 3.** The fallback alone, with the threshold read from the expanded path's logs, brings the graph step back to about the eager step and no further: it undoes the loss and gains nothing over folding.
- **All three** have the lowest time per token at the top rate at both sizes and, on the larger model, the only variant with nearly every request within the target there.
- The Stage 12 finding reproduces in the same session: the baseline graph step loses to the eager step at the top rate at both sizes and wins at the lowest.

See [the levers](../RESULTS.md#stage13-levers) and [capacity](../RESULTS.md#stage13-capacity).

### 4. The thresholds and the drift check

The two passes of the baseline agree within a few percent, so the gaps above are findings. The expanded path has a crossover at both sizes. The folded path has none on the smaller model and one at a larger batch on the larger model, which the prediction missed. See [thresholds](../RESULTS.md#stage13-thresholds) and [drift](../RESULTS.md#stage13-drift).

### 5. The closed loop: the fallback has a price

With one request in flight the graph variants are alike. At the top concurrency folding alone is the fastest and the baseline the slowest, and all three levers are slower than folding alone: a closed loop holds the batch constant above the threshold, so every step takes the eager path. The threshold was read from open-loop logs, where batches pass through a range of sizes, and it describes that mix, not a general crossing of two costs. See [closed loop](../RESULTS.md#stage13-closed).

### 6. How does the best variant compare with vLLM? Much closer, not equal.

Over HTTP the best variant (all three levers for both sizes) answers every request within the target at every rate. At the top rate it cuts the Stage 12 engine's time per token by more than half at both sizes and completes well over one and a half times as many requests per second on the larger model. Its time per token is still over twice vLLM's at the top rate (the vLLM and Stage 12 rows come from another session, so the shape of the curves is the safer reading), and it still rises with the load where vLLM's does not. See [the HTTP table](../RESULTS.md#stage13-http).

### Predictions against results (Stage 13)

1. *Every variant exact.* Held.
2. *The expansion and the read of its copy are most of the attention bytes; folding removes most; the two paths agree.* Held; and the profile adds that copying is the largest family of a whole step.
3. *V1 against V0 reproduces Stage 12.* Held.
4. *Lever 1 lowers the time per token under load on both models and the graph is no longer slower than eager.* Held.
5. *V2 gains little over V0.* **Wrong**: folding also helped the eager step on the larger model, by a good margin. The eager step's gather is narrower, but its expansion still cost a lot.
6. *Lever 2 helps less than lever 1.* Held.
7. *Lever 3 helps the larger model at high load and nothing at low load; after folding there is no crossover on either model.* Half held: the fallback alone only matches eager, and the folded path has a crossover on the larger model.
8. *All three levers lowest, gain over lever 1 small.* Held for the lowest; the gain over folding alone is not small in open loop, and negative in the closed loop.
9. *Closed loop: V3 and V6 least and close.* **Wrong** at the top concurrency, for the reason in section 5.
10. *Over HTTP the gap to vLLM narrows and does not close.* Held.

### What this sets up (Stage 13)

The profile says where the rest of the step goes: with the expansion gone, the gather and the matrix products are the large families, and the gather still copies every row to a padded width. Next: rerun `99_vllm_compare.ipynb` with the best configuration in the same session (the real size of the gap), make the fallback depend on the length bucket as well as the batch (exercise 3), and build the in-place paged-attention kernel (`12.4`) if the gather is still the largest family afterwards (built and measured in [Stage 14](stage14.md)).

## What this does not do

- **No fused paged-attention kernel.** Folding removes the expansion, but the gather still copies the keys and values of every row to a fixed width. A kernel that reads the blocks in place (exercise 4 of Stage 12) removes that copy and the padding together, and is where the rest of the gap to vLLM is expected to be.
- **No change to the scheduler.** Admission is still first come first served with the reservation rule; chunked prefill, preemption and prefix sharing are the other exercises of Stage 12.
- **One model family, one GPU, one workload.** The same Qwen2.5 sizes, the same `mixed` workload with random token ids, one T4.
- **The defaults stay as they were.** The three levers are off unless asked for. The results say folding should be on and that all three together are best in open loop; the defaults change with the next code stage, because the published notebook carries the current modules. The best configuration is `--graph --warmup --gqa fold --bucket-scheme fine` plus `--graph-max-batch` from the folded path's crossover where there is one.
- **The HTTP comparison with vLLM is across sessions** (see the decisions above).

## Lessons

- **A threshold read from one load describes that load.** The fallback helped in open loop and hurt in a closed loop whose batch never dropped below it. A threshold is a measurement of the mix of batches in the run it came from.
- **A prediction that rests on an assumption about the cost of something should name it.** "Folding will not help the eager step" assumed the eager step's expansion was cheap; the profile shows it was not.
- **The profile before the fix would have been cheaper than the hypothesis.** Stage 12 guessed the cause correctly; the profile in this stage confirms it, and would have said so one stage earlier.

What the build taught:

- **Count what the code does, then test the count.** The first estimate of the gather's bytes was off by a factor of two because the code copies the keys and values twice. The bytes model now has a test against a hand count, and the notebook sets it beside a timing and a profile, because a model of bytes is a hypothesis about time.
- **A change that must not change the answer needs the control that could.** Folding is exact in exact arithmetic, so the test that matters is the one that would fail on a head-grouping mistake: a permuted grouping gives different outputs, and the test checks that it does.
- **Make the threshold an output, not an input.** A fallback batch size picked by hand would be one more number to defend. Reading it from the step logs makes it a measurement, with three honest outcomes, one of which is that there is nothing to fall back to.
- **Put the drift check inside the experiment.** An ablation run in a fixed order on a GPU whose clock moves cannot tell a lever from the order. Running the baseline twice costs one pass and puts a number on how large a difference must be to count.
- **A failure of one arm should not be a failure of the experiment.** Each variant records its error and the run goes on, because the cost of finding a bug an hour in is the hour.

## Exercises

1. **Where the weights stop dominating.** Use `profile_step` at several batch sizes and lengths and find where the attention work overtakes the weight reads in a decode step on each model. Predict the point from `attention_bytes` and the bytes of the weights first.
2. **Fold the checking pass.** The checking pass of speculation runs `T > 1` tokens with an explicit mask. Confirm that the folded attention handles it and measure whether folding helps or hurts there.
3. **A threshold that depends on length.** The fallback uses the batch size alone, but the captured step's cost depends on the block bucket too. Make the decision a function of both, read from `step_table(..., by=("batch_bucket", "block_bucket"))`, and compare it with the single threshold on the traces of the stage.
4. **Attack padding at its source.** Fine buckets pad less after the fact. Admit and order requests so that rows in a step have similar lengths (the scheduler may choose which waiting request to admit), and measure how much padding that removes before and after the buckets.
5. **The price of more graphs.** Fine buckets capture several times as many graphs. Measure the start-up time and the graph memory against the steady-state gain, and try capturing lazily instead of ahead of time.
6. **The closed loop against vLLM.** Run `run_closed_http` against vLLM in the environment of `99_vllm_compare.ipynb` for the same concurrencies and compare the slopes of time per token with this engine's variants.
7. **Predict a different model.** The ratio of the bytes of the two paths depends on `n_rep`. Compute it with `attention_bytes` for a model with a different group size (including one with `n_rep = 1`, where folding must change nothing), predict the speedup of the layer, then measure it with `time_attention_pieces`.
8. **Fold the prefill.** Prefill keeps the expansion because it uses the implicit causal mask. Give it an explicit mask, fold it, and see whether a long prompt gets cheaper or dearer.
9. **How many repeats to see five percent.** Run the baseline several times and report the spread of its median time per token. How many repetitions does it take to resolve a gap of one twentieth between two variants at the highest rate?
10. **Find the next bottleneck.** With the best variant, profile a step at the highest rate again. Name the largest family. Is it the gather, the matmuls or something that is not a kernel at all (the host side of building a step), and what would remove it?

## How to run it

- **The notebook.** Run [`12_batched_step.ipynb`](../12_batched_step.ipynb) on a Kaggle T4 (Internet on, the secret `minivllm-github-token` attached). Set `QUICK = True` first: it runs every variant, the closed loop and the server section on the 0.5B model with fewer requests and two rates, in about ten minutes, and it publishes, so set `PUSH = False` if a smoke run should not push. The full run takes about an hour and twenty minutes; restart the kernel before it.
- **One variant by hand.**

  ```python
  from minivllm.llm_engine import EngineConfig, LLMEngine
  engine = LLMEngine(model, EngineConfig(num_blocks=1024, graph_decode=True, pad_head=True, gqa="fold", bucket_scheme="fine", graph_max_batch=24))
  engine.graphs.warmup(max_blocks=48)       # capture the buckets a workload of up to 48 blocks per request can reach
  ```

- **The same configuration over HTTP.** `python -m minivllm.server --model Qwen/Qwen2.5-1.5B --graph --warmup --gqa fold --bucket-scheme fine --graph-max-batch 24`, or let `engine_flags(**config)` build the flags.
- **A closed-loop sweep.** `bench.load.run_closed(engine, n_concurrent=16, n_requests=32, prompt_len=128, max_new=64)` in process, or `run_closed_http(url, ...)` against any server; `summarize` reduces the results.
- **The step log.** `EngineConfig(record_steps=True)`, then `bench.attention_cost.step_table(engine.step_log, by=("running",))` and `crossover_batch(graph_log, eager_log)`.

