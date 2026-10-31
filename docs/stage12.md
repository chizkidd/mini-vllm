# Stage 12: The Engine, and a Head-to-Head with Hugging Face and vLLM

[Overview](../README.md) | [Full report](../REPORT.md) | [Measured numbers](../RESULTS.md) | [How the code works](code.md)

> **Status.** The engine, its server and the load generator are built and tested on a CPU (a tiny random-weight model, real sockets). [`11_engine.ipynb`](../11_engine.ipynb) and [`99_vllm_compare.ipynb`](../99_vllm_compare.ipynb) have run on a Kaggle T4, and their results are below. The head-to-head with vLLM took two sessions: the first could not create vLLM's environment, and in the second vLLM ran with its default settings.

Stages 1 to 11 were experiments. Each had its own loop: a cache here, a scheduler there, a graph decoder for one sequence, a speculative decoder for one sequence. None of them could serve a request, because a server needs one object that owns the model and its KV memory, accepts requests whenever they arrive, and returns tokens as it makes them. This stage is that object, and then the measurement the whole series was heading for: the same trace served by this engine, by Hugging Face `generate` and by vLLM, timed by the same client on the same clock.

It adds no new lever. Every technique in the engine was measured earlier (continuous batching in Stage 4, the paged cache in Stage 5, lookup speculation in Stages 6 and 10, graph capture in Stage 10), and the engine is where they meet. What is new is what it takes to make them meet: a decode step for a changing set of requests that can live in a CUDA graph, speculation that applies only when it cannot hurt anyone else, sampling per request, and a measurement method that does not hide overload.

- [`11_engine.ipynb`](../11_engine.ipynb) builds and measures the engine: correctness against plain decoding on the GPU, the decode paths under load, the engine against Hugging Face at four arrival rates, latency mode, and the server in its own process. It saves `results/stage12.json`.
- [`99_vllm_compare.ipynb`](../99_vllm_compare.ipynb) runs vLLM, this repo's server and Hugging Face on the same traces in one session. It is an appendix on purpose (numbered 99, outside the stage chain): vLLM brings its own pinned PyTorch and may or may not run on a T4, and an outside package must not be able to stop the numbered notebooks from running or publishing. It publishes only its own results file (`results/vllm_compare.json`), never code.

## What was built

| Piece | File | What it does |
| --- | --- | --- |
| Sampling | `minivllm/sampling.py` | `SamplingParams` (greedy, temperature, top-k, top-p, stop ids, a seed) and a batched sampler. A request owns its own `torch.Generator`, so its tokens do not depend on who else is in the batch. |
| The engine | `minivllm/llm_engine.py` | `LLMEngine`: a waiting queue, first come first served admission on the paged cache, one prefill per admitted request, one decode step for all running requests per iteration, per-request sampling and stop tokens, streamed step outputs, abort, real-clock timestamps. |
| The graph step | `minivllm/paged_graph.py` | A decode step over the paged cache that has fixed shapes: buckets, a fixed-width gather through a static block table, padding rows that write into a reserved block, one packed input buffer, CUDA-graph capture per bucket, and the checking pass of speculation. Eager without CUDA. |
| The server | `minivllm/server.py` | Standard library HTTP: `POST /v1/completions` (one reply or a server-sent-events stream), `GET /health`, `GET /v1/stats`. One engine thread owns the engine. |
| Load generation | `bench/load.py` | Traces from the generators of Stages 4 and 5 with a digest, open-loop drivers for the engine, any streaming server and Hugging Face, and one summary. |
| vLLM | `bench/compare_vllm.py` | vLLM in its own environment, started as a server process with a list of settings tried in turn, every failed attempt recorded. |

```text
client ──HTTP──> handler threads ──inbox──> engine thread ──> LLMEngine.step()
                                                              ├─ admit: first come first served, only if every block the request will ever need is free
                                                              ├─ prefill each admitted request (one pass over its prompt)
                                                              └─ decode ONE step for all running requests
                                                                   │  eager paged view | bucketed static step | the same step as a CUDA graph
                                                                   └─ latency mode: exactly one greedy request running
                                                                        └─ lookup guesses k tokens, one pass of k + 1 tokens checks them,
                                                                           the agreeing prefix plus one token is kept, the rest is rolled back
              <──events (new tokens of each step)── per-request queue ──┘
```

## The decisions, and why

1. **A request is admitted only if the pool has every block it will ever need** (its prompt plus `max_new - 1` tokens). A running request can therefore never run out of memory half way, nothing is ever preempted, and a request's blocks go back to the pool the moment it finishes, whether by length, by a stop token or by an abort. The price is that a request holds memory for tokens it has not generated yet; allocating lazily and preempting on overflow is the obvious next step (exercise 1).
2. **The engine reserves one block that no request owns (the NULL block).** A captured step has a fixed batch size, so a step with three running requests uses the batch-of-four graph with one padding row. A padding row feeds token zero at position zero and writes its key and value into the NULL block, and the block tables of padding rows (and the unused columns of real rows) point at it. The attention mask hides every column past a row's own position, so the contents of the NULL block are never read by a real row. A test fills it with junk and checks that no real row changes, and a negative control shows the check can fail.
3. **Buckets, warmed ahead.** One graph per (batch size, blocks per sequence), both rounded up to the next power of two (and the limit itself). Graphs are captured before the timed run (`warmup`), and the capture cost is reported on its own, because it is paid once when the server starts and not per request. The graphs share one memory pool.
4. **A fixed-width gather.** The eager paged view gathers exactly as many columns as the longest running request needs. A captured step gathers every block of its bucket through a block table that is a static buffer, and the mask hides the rest. This reads more than the eager view does, so the notebook includes a **control**, the same bucketed step run without capture, to separate the cost of the buckets from the benefit of the graph.
5. **All inputs are one buffer.** The token ids, the positions, the block and offset each key is written to, and the block table of a step are one packed integer tensor, overwritten with a single host-to-device copy per step.
6. **Sampling stays outside the graph.** Each request has its own parameters and its own generator, so the graph returns logits and the sampler runs on them. Greedy rows are one `argmax` for the whole batch.
7. **Speculation only when alone.** A batch already shares each read of the weights across its rows, and checking drafts for every row would multiply the work. With exactly one greedy request running, the weights are read for one row, which is where speculation paid in Stages 6 and 10. So the engine speculates in that case and decodes the batch otherwise, switching on its own, and a sampled request is never speculated. The checking pass has a fixed length (`k + 1` tokens, padded with zeros when the lookup found fewer drafts), the rows after the real drafts are ignored, and a draft is cut so the accepted tokens never exceed `max_new` or the reservation. A stop token inside an accepted block ends the request there.
8. **One engine thread.** The engine is not thread-safe, so exactly one thread touches it. Handler threads post commands to an inbox and read events from a queue of their own. A client that disconnects in the middle of a stream has its request aborted, so its blocks return to the pool. An engine failure becomes an error reply for every waiting client and not a hang.

## How it is measured

- **Parity before speed.** Every mode of the engine is compared with plain cached decoding (`generate_cached`) on the GPU before any timing. A mismatch counts as an fp16 near-tie only if the logit gap at the first difference is small, as in earlier stages, and a **control that must fail** (sampled outputs against the greedy reference) shows the comparison can fail.
- **Open loop.** Every request is sent at its arrival time whether or not earlier ones have finished. A closed loop, where a client waits for its reply before sending the next request, cannot overload anything, so it hides exactly the behaviour a server exists to handle.
- **The same trace, proved.** A trace is fixed by the workload (the `mixed` workload of Stages 4 and 5), the number of requests, the rate and the seeds, and its digest is saved with every row. `99_vllm_compare.ipynb` compares its digests with those of `stage12.json`.
- **The same work.** Every request generates exactly its `max_new` tokens: end-of-sequence is suppressed for Hugging Face and ignored for vLLM, and the engine has no stop tokens in these traces.
- **The metrics.** Time to first token (TTFT), time per output token after the first (TPOT, what a reader of the stream experiences), the longest stall between two events of a stream, request and token throughput, **goodput** (the tokens of the requests that met a latency target, per second of the run) and **capacity** (the highest tested rate at which at least nine tenths of requests meet the target).
- **A generous baseline.** Hugging Face's streamer timestamps every step, so a request counts as finished at its own last token and not when its batch returns. Both Hugging Face and the engine are warmed first, and the GPU is warmed under load before each group, with the clock logged.
- **Turns.** The order of the systems rotates from rate to rate, so a drifting clock does not always favour the same one.

## Predictions (Stage 12)

Written before any GPU run, qualitative on purpose (the numbers will come from the run):

1. Every greedy mode of the engine equals plain decoding up to near-ties, and the sampled control differs.
2. The bucketed step without capture is no faster than the eager paged step (it reads more columns for the same launches). The captured step has a lower time per token than both, most on the smaller model, where launches are most of the step (Stage 10), and the gain shrinks on the larger one. Time to first token improves less than time per token, because a prefill is one large pass.
3. Hugging Face run sequentially saturates first, at about one over the time one request takes alone, and above that its queue grows for the whole trace. Static batching serves more than sequential but saturates before the engine, because a batch runs to its longest request and nothing joins it. The engine's capacity under the latency target is the highest of the in-process systems.
4. In latency mode, lookup speculation is fastest on the repetitive workloads and about neutral on open text, and Hugging Face is slowest.
5. The engine behind HTTP in its own process is close to the in-process engine: a little more time to first token, about the same time per token.
6. Against vLLM, on the 0.5B model the two servers are close at low load (a small model's step is mostly launches). On the larger model and under load I expect vLLM to be ahead, because of its fused paged-attention kernel and its scheduler. These are predictions about direction. If this engine matches vLLM on some rate, the likely reason is the narrow workload (greedy decoding, random token ids, no prefix reuse) on a small model on a small GPU, not that the engine is a better design. It is also possible that a recent vLLM does not run on a T4 at all, in which case the comparison says so.

## Results (Stage 12)

The numbers are in [`RESULTS.md`](../RESULTS.md#stage12-setup), generated from `results/stage12.json`: one Kaggle T4 session, 64 requests of the `mixed` workload at four arrival rates, and the 0.5B and 1.5B models. Nothing below repeats a figure; each section links to its table.

### 1. Is every mode exact? Yes, and the control can fail.

Every greedy output of the engine, in all four modes (eager, the uncaptured static control, graph, graph with lookup), equals plain cached decoding or differs only at a near-tie, and the sampled control differs for most prompts, so the comparison can fail ([parity](../RESULTS.md#stage12-parity)). On the GPU the graph modes captured real graphs and fell back to nothing, and the static control captured none.

### 2. What does each decode path buy under load? On the smaller model the graph helps. On the larger one it hurts, and the control says why.

At the middle arrival rate the bucketed step without capture is never faster than the eager paged step. On the smaller model the captured step cuts the median time per token to under two thirds of the eager step's. On the larger model the captured step is more than twice as slow as the eager one, and the uncaptured static control is just as slow, so the loss is in the buckets and not in the capture. The graph lowers the median time to first token by much less than it lowers the time per token on the smaller model, and raises it on the larger one, because a prefill is one large pass and the steps there are slower ([decode paths](../RESULTS.md#stage12-modes)).

What the buckets cost is, as a hypothesis (no per-step profile was saved), the fixed-width gather: a bucketed step reads every block of its bucket for every row, padding rows included, and `model.py` then expands the keys and values of the grouped-query heads before attention, so the bytes moved per step grow with the bucket and not with the batch. Slower steps keep more requests running, which fills a larger bucket, which makes the step slower again. That feedback is plausible for the larger model, where the loss is largest, and it is not measured.

### 3. How does the engine compare with Hugging Face as the load rises? Hugging Face is saturated from the lowest rate.

Hugging Face run one request at a time completes the same number of requests per second at every offered rate, so it is saturated from the lowest rate tested; its median time to first token is over ten seconds at every rate, and in static batches over three. The eager engine's median time to first token stays under a quarter of a second at every rate. At the top rate the engine completes over three times the requests per second of the best Hugging Face setup, at both sizes, and every request of the comparison was answered without errors ([load](../RESULTS.md#stage12-load)).

The captured step's behaviour under load matches section 2. At the lowest rate it has a lower median time per token than the eager step at both sizes, and at the top rate it is slower than the eager step on the smaller model.

### 4. What is the capacity under the latency target? Hugging Face has none in the tested range, and the engine's is at least the range's middle.

No Hugging Face setup meets the latency target at any tested rate. The engine's capacity is at least four requests per second at both sizes, and the top of the tested range on the smaller model (so for that model the range, not the engine, limits the figure). The eager engine's capacity is not below the graph engine's at either size ([capacity](../RESULTS.md#stage12-capacity)).

### 5. What does latency mode buy? A lot, and it is where the graph and the lookup pay.

One request at a time, the graph beats the eager step by over one and a half on every workload, and by more on the smaller model, where launches are more of the step. Lookup speculation beats the graph alone by over one and a half on copy and code at both sizes. It gains less on open text than on copy and code, as the acceptance rate does, which is higher on copy and code at both sizes; on the larger model the lookup is about even with the graph alone on open text, and on the smaller model it still beats the graph alone there by over a third. Hugging Face `generate` is slower per token than the eager engine on every workload and size. Every engine output equals plain decoding or differs at a near-tie ([latency](../RESULTS.md#stage12-latency)).

### 6. What does HTTP cost? Little.

Behind the server, in its own process, the engine keeps within a tenth of the in-process tokens per second and adds under a tenth of a second to the median time to first token ([server](../RESULTS.md#stage12-server)).

### 7. How does it compare with vLLM? *vLLM is ahead under load. One request at a time the engine matches it on the larger model and trails it on the smaller one.*

`99_vllm_compare.ipynb` ran vLLM, this repo's server and Hugging Face on the same traces in one session on the same T4, the systems taking turns in a different order at each rate. The digests of the four traces are equal to the engine notebook's. vLLM ran with its default settings: a recent release started on a Turing GPU with no other attention backend and without the older engine, which the notebook was ready to fall back to. The engine's server uses the graph decode path at both sizes, so on the larger model it carries the loss of section 2; read its rows with that in mind. [The numbers](../RESULTS.md#head-to-head-load):

- **Lightly loaded, a client sees almost the same thing.** At the lowest rate both servers complete the offered rate and meet the latency target at both sizes. vLLM's median time per token is already lower at both sizes.
- **Under load, vLLM's time per token does not move and the engine's does.** Across the four rates vLLM's median time per token stays within a quarter of its lowest value at both sizes, while the engine's rises more than threefold from the lowest rate to the highest. That is what batching is for: the weights are read once for the whole batch, so a bigger batch costs almost nothing more per token. The engine pays for the batch in the gather and in the expansion of the key-value heads (the cause suggested in section 2).
- **That sets the capacity.** vLLM meets the latency target up to the top rate at both sizes. The engine's server does so only on the smaller model, and at the top rate vLLM completes over one and a half times the requests per second of the engine's server at both sizes. Hugging Face meets the target at no tested rate, as in section 4.
- **One request at a time, the answer depends on the size** ([latency mode](../RESULTS.md#head-to-head-latency)). On the smaller model vLLM is over a third faster per token than the engine's server on every workload. On the larger one the two are about even, within a quarter of each other on every workload and neither ahead on all of them across sessions: that step is mostly the read of the weights, which both do once. vLLM's first token comes sooner at both sizes, but the engine runs one large prefill pass while vLLM splits its prefills and keeps a prefix cache (and latency mode sends each prompt three times in a row, which a cache can help), so this row says less about the decode path than the others.

**What this shows, and what it cannot.** The same GPU, the same weights and the same traces: vLLM keeps its step flat as the load rises where this engine's does not, so the loss of section 2 belongs to this engine's batched attention path and not to the hardware or the model. The comparison cannot say which part of vLLM's difference matters most, a fused paged-attention kernel that reads the blocks in place or its scheduler. Exercises 4 and 5 are the experiments that would separate them.

**The first session got no further than the install.** Creating vLLM's environment failed for both releases tried, in a fraction of a second: the system Python of the Kaggle image has no working `ensurepip`, so `python -m venv` fails. The fault was the harness's, not the GPU's and not vLLM's. The install now goes through `uv`, as vLLM's own install page does. That session still showed that results carry across sessions: the engine behind HTTP, in a process of its own, had the capacity it had in process, and Hugging Face completed about the same number of requests per second as in the engine notebook (typically within a few percent, never off by more than a fifth). The second session repeated the whole run, with a smaller share of the GPU for vLLM on the larger model (see the lessons).

### Predictions against results (Stage 12)

1. *Every greedy mode equals plain decoding up to near-ties, and the sampled control differs.* Held.
2. *The bucketed step without capture is no faster than the eager paged step, and the captured step has a lower time per token than both, most on the smaller model, with the gain shrinking on the larger one. Time to first token improves less than time per token.* Mostly held on the smaller model and at the lowest rate. **Wrong on the larger model**, where the gain did not shrink but reversed, and wrong at the top rate on the smaller model. The static control was the right thing to include: it showed the cause is the buckets. Time to first token improved less than time per token on the smaller model, as predicted, and got worse on the larger one.
3. *Hugging Face run sequentially saturates first; static batching saturates before the engine; the engine's capacity is the highest.* Held, and the saturation is stronger than predicted: Hugging Face is past its capacity at the lowest rate tested, so the sweep did not find where it saturates. The engine's capacity on the smaller model is the top of the range, so the sweep did not find its limit either.
4. *Lookup speculation is fastest on repetitive workloads and about neutral on open text; Hugging Face is slowest.* Held, except that on the smaller model the lookup still helps on open text.
5. *The engine behind HTTP is close to the in-process engine.* Held.
6. *Against vLLM.* Mostly held. *Close at low load on the smaller model* held only in what a client sees: both servers complete the offered rate and meet the target, but vLLM's time per token is already lower. *vLLM ahead on the larger model and under load* held, by a wide margin in capacity. *A fused kernel and a scheduler as the reason* is consistent with the flat line of vLLM's step but was not isolated. *A recent vLLM may not run on a T4 at all* did not happen: it ran with its default settings. One thing the prediction missed: alone on the larger model the engine is about as fast per token as vLLM.

### What this sets up (Stage 12)

The two modes are different regimes. One request at a time is where the engine's techniques pay most (a captured step, a lookup that is checked in one pass), and a batch of changing requests is where the bucketed fixed-width step costs more than it saves once the model or the batch is large. The levers that follow from the findings, each an exercise below: avoid expanding the KV heads in decode attention, use buckets finer than powers of two, fall back to the eager step above a batch size, and log the per-step batch and bucket (the engine does with `record_steps=True`; the notebook did not turn it on) to see the feedback loop instead of guessing at it. The comparison with vLLM (section 7) says how much these levers have to close: under load, vLLM's time per token stays flat as the engine's rises several times over, and the engine's capacity on the larger model is a quarter of vLLM's.

## What this does not do

- **No preemption, no prefix sharing, no chunked prefill.** A long prompt is one pass, and it stalls the decode steps of the running requests while it runs (the budget for prefills per step bounds how many, not how long).
- **The batched graph step loses to the eager step under load on the larger model** (section 2); the engine keeps both paths and defaults to the eager one.
- **The paged attention is a gather in PyTorch**, not a fused kernel that reads the blocks in place. This is the largest difference from vLLM's attention path.
- **Speculation is greedy-only and for one request at a time.** The guesser is the n-gram lookup of Stage 6. A draft model in the engine, or speculation for a small batch, are exercises.
- **One model family, one GPU.** The engine runs the Qwen2 architecture of the earlier stages on one device, fp16 (the int8 weights of Stages 8 to 11 run unchanged but are not part of this comparison).
- **The vLLM comparison covers one narrow workload and one untuned setup.** The load traces use random token ids, so nothing shares a prefix and vLLM's prefix cache has nothing to do there; chunked prefill, its other features and anything outside the workload are not exercised, and the engine is not charged for lacking them. vLLM ran with its defaults and a memory share chosen so that vLLM, the engine's server and the Hugging Face model fit on one GPU together, which is not how either would be deployed. The share was smaller on the larger model, and its cost to vLLM is the smaller key-value cache it kept, which stayed larger than the engine's pool.

## Lessons

- **An engine is a set of promises, and the tests are the promises.** Tokens do not depend on who else is in the batch. A request never runs out of memory half way. A finished request's blocks return at once. A stop token in the middle of an accepted block ends the request there. A padding row never touches a real request's memory. A client that goes away frees its blocks. None of these is visible in a benchmark and all of them are where an engine breaks, so each is a CPU test with a tiny model, run on both the eager and the bucketed decode path.
- **Composing needed design, not just glue.** A CUDA graph wants static shapes and a batch of changing requests has none, so the engine rounds up to buckets and masks what the rounding added. Speculation wants one sequence and the engine serves many, so it applies speculation only when it is alone. Neither idea is in the earlier stages, and both come from what the earlier stages measured.
- **Cheap controls separate causes.** The bucketed step without capture costs one flag and said whether a result belonged to the graph or to the buckets: on the larger model both lost by the same amount, so the buckets were the problem, and without the control the finding would have read as "CUDA graphs hurt", which is false in latency mode.
- **A technique that wins at batch one can lose in a batch.** The captured step is the best thing in latency mode and the worst under load on the larger model. Each mode needs its own measurement, and an average over both would have hidden both results.
- **Open-loop load shows feedback.** A closed loop that waits for each reply cannot make the engine fall behind, so it could not show that a slower step keeps more requests running, which makes the step slower. Only requests that arrive on their own clock can.
- **Measure through the same door.** vLLM and this repo's server run as separate processes behind the same HTTP interface and are driven by the same client, so the difference between them is the systems and not the harness. The in-process engine is measured separately and the cost of the HTTP layer is its own row.
- **Check the prerequisite where it is cheapest to check.** The first vLLM attempt died at its first command, yet the notebook only reached it after it had built the traces and loaded both models, and the run went on for the rest of the session without it. A prerequisite that can fail (an environment, a download, a server that must start on this GPU) should be tried first, on the smaller model, and its failure printed before the long measurement starts. The notebook now has a cell that brings vLLM up on the smaller model before the long loop.
- **A failure of the harness is not a result about the system.** "vLLM did not run" is true and says nothing about vLLM: the environment step failed, in a Python that `venv` cannot use. The record keeps the two apart (the install attempt, then each start attempt with its reason), so the headline table leaves the vLLM row out until there is something to report instead of blaming the GPU.
- **Budget the memory of everything that shares the GPU.** The second session first failed on the larger model, in Hugging Face's first warm-up, because vLLM's reserved share, the engine server's pools and the Hugging Face model together filled the whole GPU. vLLM's share is a hard reservation and the engine's pools grow with the buckets it captures, so the share for the larger model was lowered (it is in the config of the results) and the run repeated from the top.
- **A flat line is a result.** vLLM's time per token barely moves from the lowest rate to the highest, which is the whole point of batching, and the engine's rises several times over the same range. Nothing else in the series showed as plainly what a batched step has to do, and it needed an outside engine on the same trace to see it.
- **An outside package should not control your pipeline.** The vLLM comparison lives in an appendix notebook outside the stage chain, in its own environment, and publishes only its own results file, so a version of vLLM that does not support the GPU costs one column of one table and nothing else.

## Exercises

1. **Preemption.** Allocate blocks lazily and, when the pool runs dry, preempt the newest request (free its blocks, put it back at the head of the queue, recompute its prompt on re-admission). Measure what it buys at high load and what it costs in recomputation.
2. **Chunked prefill.** Split a long prompt into chunks and interleave them with decode steps, to bound the stall a prefill causes for the requests already running. Look at the stall percentile before and after.
3. **Prefix caching.** Share the blocks of identical prompt prefixes between requests (copy on write at the first block that differs). It needs reference counts in the block allocator. Measure it on a trace where requests share a system prompt.
4. **A paged-attention kernel.** The gather copies the keys and values of every running request each step. A kernel that reads the blocks in place removes the copy; the bucketed step makes the interface to it simple.
5. **Fix the batched graph step.** Three levers, one at a time, against the same traces: fold the query heads of each key-value head into the query length so the keys and values are not expanded (a decode step of one token has the same mask for every query head), use buckets finer than powers of two, and fall back to the eager paged step above a batch size. Turn on `record_steps` and save the log in the results, so the batch and the bucket of every step are visible.
6. **Sampling in the graph.** Move the greedy rows' `argmax` into the graph and measure the step.
7. **Speculation for two or three requests.** Allow lookup speculation while up to `m` requests run and find the `m` where it stops paying.
8. **int8 in the engine.** `quantize_model` returns a model the engine can run unchanged. Measure the decode paths with it and compare with Stages 9 to 11.
9. **vLLM's own speculation.** Run vLLM with its n-gram speculation in latency mode and compare it with the engine's lookup.
10. **Another workload.** Replace `mixed` with a workload with a long shared prefix, or with very long outputs, and see which result moves.

## How to run it

- **Notebooks.** Run [`11_engine.ipynb`](../11_engine.ipynb) on a Kaggle T4 (Internet on), then [`99_vllm_compare.ipynb`](../99_vllm_compare.ipynb) (it installs vLLM into its own environment with `uv`, so it takes longer to start, and a cell before the long loop brings vLLM up on the smaller model so that a failure shows early). Set `QUICK = True` for a short pass. Three processes share the GPU in that notebook (vLLM, the engine's server and the Hugging Face model), so `GPU_SHARE` sets how much vLLM reserves per model size; on a different GPU, check that the three fit.
- **The server.** `python -m minivllm.server --model Qwen/Qwen2.5-0.5B --graph --warmup --speculate-k 4 --port 8000`, then `curl -N localhost:8000/v1/completions -d '{"prompt": "Hello", "max_tokens": 32, "stream": true}'`.
- **The engine in code.**

  ```python
  from minivllm.llm_engine import EngineConfig, LLMEngine
  from minivllm.sampling import SamplingParams
  engine = LLMEngine(model, EngineConfig(num_blocks=1024, graph_decode=True, speculate_k=4))
  outputs = engine.generate([prompt_ids], SamplingParams(max_new=64))      # or add_request(...) and step() yourself
  ```

- **A trace against any system.** `bench.load.make_trace("mixed", n=64, rate=4.0)` builds a trace; `run_engine`, `run_http` and `run_hf` serve it and return results that `summarize` reduces.
