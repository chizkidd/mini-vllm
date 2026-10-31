# What I learned about inference engineering

[Overview](../README.md) | [Full report](../REPORT.md) | [Measured numbers](../RESULTS.md) | [Headline results](../README.md#headline-results) | [How the code works](code.md)

This repo built a small LLM inference engine in eleven measured stages on a single Tesla T4, and then composed the pieces into one engine behind a server ([`stage12.md`](stage12.md)). This page is about the lessons of the stages; what the engine run added is in the section on it below. This page is the short version of what that taught me, written to be read in a few minutes and to be a base for conversation. It has no numbers of its own: every figure is in the [headline table](../README.md#headline-results) or in [`RESULTS.md`](../RESULTS.md), from the results files, and each section points at the stage that measured it.

## The one idea

Generating a token is limited by two things that are not arithmetic: the bytes the GPU must read for it (every weight, plus the KV cache) and the overhead around each step (kernel launches, Python, scheduling). A decode step at batch one does about one operation per byte it reads, hundreds of times below the point where the GPU would run out of arithmetic, so more arithmetic per byte is free until that point. And an eager step reaches only a small share of the bandwidth ceiling, so a large part of the gap is overhead and not memory. Every technique here does one of three things: reads fewer bytes per token, shares each read across more tokens, or removes overhead. The way to know which one a change is doing is to compare the step against its ceiling (Stages 1 to 3).

## The levers, and where each one stops

**KV cache (Stage 2).** Do not recompute the keys and values of tokens already seen. It is slower than recomputing for short outputs, because it adds work and launches per step, and wins only past a crossover of several hundred new tokens. Every optimization has a break-even point and the first job is to find it.

**Batching (Stage 3).** A decode step is memory-bound, so extra rows in it are almost free: throughput grows nearly in step with the batch while each request keeps its own rate. What it costs: padding to the longest prompt, slots that keep computing after their request finished, short requests waiting for long ones, and KV memory per sequence.

**Continuous batching (Stage 4).** Schedule per iteration, not per batch: a finished sequence leaves at once and a waiting one joins. Goodput roughly doubles at the same concurrency and the load the server can sustain rises. What it costs is that a prefill now lands in the middle of other requests' decode steps, which is why a budget for prefills per iteration exists.

**Paged KV cache (Stage 5).** Allocate the cache in blocks and give each sequence a block table, so memory is held for the tokens a request needs and not for the longest one the server might see. More sequences fit in the same memory and goodput rises while memory is the limit. It costs a gather through the table on every step. This repo reserves blocks for a whole request at admission, which avoids preemption and leaves it (and sharing blocks between requests) as the obvious next step.

**Speculative decoding (Stages 6 and 7).** A pass over `k + 1` tokens costs about what a pass over one token costs when the step is memory-bound, so guess several tokens, check them all in one pass, keep the longest prefix the model agrees with plus one more, and roll back the rest. The output is the model's own greedy output (up to fp16 ties), and a rollback is only moving a length. With a free guesser the speedup is the tokens per round divided by the cost of the checking pass in plain steps, which is why that one number, `v`, decides everything later. The n-gram lookup is free and works when the text repeats. A draft model guesses better but costs a step of its own, and run eagerly a small model's step costs nearly as much as the big one's because launches and not weights set the cost.

**Weight-only quantization (Stages 8 and 9).** Storing weights in one byte halves the bytes read, which only pays if a kernel actually reads them as bytes. Every way of using an int8 weight that PyTorch offered was slower than fp16 on this GPU (an eager dequantize writes and re-reads a full fp16 copy), and only a fused kernel turned that around. Accuracy needs its own checks and controls: per-channel scales keep perplexity within a fraction of a percent, and a per-tensor scale and a wrong scale show that the check can fail.

**CUDA graphs (Stages 7 and 10).** At a small model a step is mostly launches, and capturing the whole step as a graph removes them: the biggest relative gain is at the smallest model and it fades as weights dominate. A graph needs static shapes, which forced real design choices: a static KV cache, a mask computed from positions inside the graph, token buffers the graph reads, and a rollback that is only not advancing a length.

**When levers do not compose (Stages 10 and 11).** Int8 should multiply with speculation, since one cuts bytes per read and the other cuts reads per token. It does not here. The kernel that reads each int8 weight once for the several rows of a checking pass is limited by what it computes for each extra row, so an int8 checking pass costs close to two plain steps where an fp16 one costs one, and the gains combine only partly. A rewrite of that kernel narrowed the gap and did not close it. On a GPU with more arithmetic per byte it might; on this one it is a measured limit.

## What composing them taught (Stage 12)

**The gap to a naive server is mostly scheduling.** Hugging Face `generate` is not slow per token: alone, it is within a small factor of the engine. Under load it is saturated from the lowest rate tested, because it serves one request or one batch at a time and everything else waits. Continuous batching on a paged cache is what removes the wait, and it matters far more than any kernel in this repo.

**A technique that wins at batch one can lose in a batch.** The captured decode step is the best thing in latency mode and the worst under load on the larger model, and a control (the same step without the capture) showed the loss was the fixed-width buckets and not the graph. Static shapes are paid for in bytes moved, and the bill grows with the batch.

**Open-loop load finds feedback that batch-one timing and closed-loop clients cannot.** A slower step keeps more requests running, which fills a bigger bucket, which makes the step slower. A client that waits for each reply never lets that start.

**Speculation belongs to the case where a request runs alone.** With one request the weights are read for one row and a lookup checked in one pass is a large win on repetitive text; with a batch the weight read is already shared. The engine switches on its own.

**The remaining gap to vLLM is the batched step, not the hardware.** On the same GPU, weights and traces, vLLM's time per token is nearly the same at the lowest load and the highest, while this engine's rises several times over. One request at a time the two are close, and on the larger model the engine is about as fast; the difference only appears when a batch has to be served, which is where a fused paged-attention kernel and a lean scheduler pay. The comparison cannot say which of the two matters more.

**Test the promises, not the speed.** An engine is a set of invariants (tokens independent of the batch, blocks returned at once, padding never touching real memory, a stop token inside an accepted block ending the request). Each is a test with a tiny model, long before any benchmark.

## What fixing the batched step taught (Stage 13)

- **Count the bytes, then profile, then change.** The cost of a batched decode step was mostly copies: the gather of every row to a padded width and the expansion of the key-value heads to every query head. Folding the query heads into the query rows removes the expansion without changing the arithmetic, and it roughly halves the time per token under load on the larger model.
- **Controls and ablations earn their keep.** Running each lever alone, together, and the baseline twice showed that folding was the main win, that finer buckets help less, that the fallback alone only undoes the loss, and how large a difference has to be to count.
- **A threshold describes the load it was read from.** The eager fallback helped in an open loop and cost time in a closed loop whose batch never fell below it.
- **The gap narrows before it closes.** The best variant over HTTP is much closer to vLLM, and its time per token still rises with the load; what is left is the gather itself.

## What reading the blocks in place taught (Stage 14)

- **Removing the copies is not the same as being fast.** The in-place kernels win clearly when rows fill part of their bucket and tie or lose when they fill all of it, because they read the blocks at a small share of the bandwidth that the library's copies reach. The counted bytes said twenty times less; the clock said a few times.
- **Gate at the batches the load runs.** The gate picked the grouped kernel by the largest batch; the load is mostly small steps, where the per-head kernel is ahead.
- **A lever's threshold goes stale when its neighbours change.** With the kernel a graph step is cheap at large batches, so the eager fallback of Stage 13 went from a help to a cost.
- **A kernel can win a layer and lose a step.** On the smaller model the profiled step got slower with the kernel while the load runs favoured it; measure at the layer, the step and the load.
- **The gap to vLLM narrowed again and the shape did not change.** The best variant over HTTP is between two and three times vLLM's time per token at the top rate, and its curve still rises where vLLM's is flat; what is left is an untuned kernel and the weights read in full every step.

## Surprises and negative results

- The KV cache loses to recomputing at short lengths.
- An eager draft model costs almost as much as the model it drafts for.
- Int8 weights made decoding slower in every way PyTorch offered, until a kernel read them as bytes.
- Int8 and speculation do not multiply, and a second kernel did not fix it.
- My offline check of a compiled kernel was right about spills, wrong about the sign of one change, and blind to shared memory a Triton kernel allocates dynamically.
- A padded output projection that one layer's timing suggested would save an eighth of the step saved a few percent when the whole step was measured.
- The batched CUDA-graph decode step lost to the eager one under load on the larger model, while the same graph won by a wide margin one request at a time.
- The first timing after a heavy phase on a power-capped GPU runs at a different clock from the rest, so a whole table of rates was off until the clocks were logged.

## How I measured

- **Prediction first.** Each notebook writes down what it expects before the benchmark, and the write-up says which predictions failed.
- **Parity before speed.** A stage must produce the previous stage's tokens (up to fp16 near-ties, whose logit gap is read) before any number counts.
- **Controls that must fail.** A wrong scale, a per-tensor scale, a mask that lets rows see later keys, a speculative loop that skips its rollback. A check that cannot fail proves nothing.
- **Medians, spreads, warm-up.** Timings are medians with the spread recorded; compared candidates take turns so a drifting clock hits them alike; the GPU's clock, temperature and power cap are logged, and a number whose rounds disagree is flagged.
- **A gate before a kernel.** A go or no-go on one layer, timed where the kernel will run, before any end-to-end claim.
- **Results are stored from the notebooks.** Figures live in `results/*.json` and `RESULTS.md`.

## What this repo does not do

Everything was measured on one Tesla T4, a GPU whose arithmetic is scarce relative to its bandwidth under a power cap, so conclusions about int8 and speculation are about that regime. The eleven stages are separate experiments; [Stage 12](stage12.md) composes them into one engine with a server and measures it against Hugging Face on the same traces. The same traces served by vLLM show how far that engine is from a production one: close at low load and alone on the larger model, but far behind under load. Still missing: prefix caching, chunked prefill, preemption, a fused paged-attention kernel (the gather is done in PyTorch, and the batched captured step loses to the eager one under load on the larger model), quantized KV and multiple GPUs.

## Questions I can now answer

1. **Why is decoding memory-bound, and how would you check?** One token per step means about one operation per byte of weights read; compare the step time with bytes read over measured bandwidth (Stages 1 to 3).
2. **Why does batching raise throughput, and what limits the batch?** The weight read is shared, so extra rows are nearly free until compute or KV memory binds (Stage 3).
3. **Static against continuous batching?** Per-batch scheduling idles finished slots and blocks short requests; per-iteration scheduling does neither, at the cost of prefills landing in the middle of decodes (Stage 4).
4. **What does paging solve, and what does it cost?** Fragmentation and over-reservation of KV memory; it costs a gather per step, and a design decision about reserving or preempting (Stage 5).
5. **How does speculative decoding keep the output exact?** The target checks every guess in one pass and keeps only what it would have produced itself, then rolls back; the output equals greedy decoding up to fp16 ties (Stage 6).
6. **What decides its speedup?** Tokens accepted per round over the cost of the checking pass in plain steps, minus the guesser's own cost (Stages 6 and 7).
7. **Why can a draft model make decoding slower?** If the draft's step costs most of the target's, the guesses are not paid for; at small sizes launches, not weights, set that cost (Stage 7).
8. **Why is weight-only int8 not automatically faster?** The bytes only help if the kernel reads them as bytes, and the conversion work must hide behind the memory time (Stages 8 and 9).
9. **When do CUDA graphs help, and what must be static?** When launch overhead is a large share of the step; shapes, buffers and the mask must be fixed, so the cache is static and a rollback is moving a length (Stage 10).
10. **Why might int8 and speculation not multiply?** The multi-row checking pass does more arithmetic per weight, which can make it cost more than a plain step even when the weights are read once (Stages 10 and 11).
11. **How do you benchmark honestly on a shared, power-capped GPU?** Warm up, take turns, log the clock, record spreads, and repeat the old baseline in the same session (Stages 10 and 11).
12. **How do you know a speedup is not a bug?** Parity with the previous stage, exactness against plain decoding, and controls that are supposed to fail.
13. **How would you build the engine that ties these together, and how would you know it is correct?** Reserve each request's blocks at admission so nothing is preempted, keep a block that no request owns for padding rows, round the batch and the block count up to buckets so a graph can serve a changing set of requests, speculate only when a request is alone, sample outside the graph with a generator per request, and test every promise (tokens independent of the batch, blocks returned at once, padding never touching real memory) on a tiny model before measuring anything (Stage 12).
14. **Why can a CUDA graph make a server slower?** A graph needs static shapes, so a batch of changing requests is rounded up to buckets and the gather reads every block of the bucket. That cost grows with the batch and the model while the launch savings do not, so the graph wins one request at a time and can lose in a batch; a control that runs the same step without the capture tells the two apart (Stage 12).
15. **Why measure a server open loop?** A client that waits for each reply before sending the next request cannot overload anything, so it hides the behaviour a server exists to handle; arrivals have to follow their own clock (Stage 12).
16. **What would change on an A100 or H100?** More arithmetic per byte moves the point where extra rows stop being free, so quantization with speculation has more room; the roofline from Stage 3 is the tool to check it with.
