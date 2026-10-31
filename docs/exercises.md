# Exercises

[Overview](../README.md) | [Full report](../REPORT.md) | [Measured numbers](../RESULTS.md) | [How the code works](code.md)

Each stage ends with exercises. This page collects all of them in stage order, so they can be planned together; the same list is in the Exercises part of [`REPORT.md`](../REPORT.md). Numbers are per stage: "Stage 12, exercise 5" is the fifth exercise of the Stage 12 write-up, and Stages 1 and 2, which share one write-up, are called `1_2` (as in `stage1_2.md`), so `1_2.3` is its third exercise. Some questions come back in a larger form in a later stage (speculative sampling in Stages 7, 10 and 11, for instance), and the measurements an exercise mentions are in [`RESULTS.md`](../RESULTS.md). 

## [Stages 1 & 2: Baseline & KV Cache](stage1_2.md)

1. Repeat the crossover sweep with `PROMPT_LEN = 512`. Does the crossover move left or right? (A longer prompt makes every naive step more expensive.)
2. Replace `repeat_interleave` in `model.py` with `enable_gqa=True` in `F.scaled_dot_product_attention` and re-measure ITL.
3. Capture the decode step in a CUDA graph by hand (`torch.cuda.CUDAGraph`) and compare with `mode="reduce-overhead"`.

## [Stage 3: Static Batching](stage3.md)

1. Sort the `mixed` requests by prompt length before batching and see how much padding disappears.
2. Group requests by expected output length and see how much idle-slot waste disappears.
3. Replace `repeat_interleave` with `enable_gqa=True` and re-run the batch-size sweep, since that copy grows with the batch.
4. Prefill long prompts in chunks and see what happens to TTFT.
5. Extend the batch-size sweep beyond 64 to find where the step stops being free.

## [Stage 4: Continuous Batching](stage4.md)

1. Prefill a long prompt in chunks across several iterations instead of all at once. What happens to the tail gap between tokens, and to the first-token latency of that request?
2. Increase the number of requests and the arrival rate. Does the capacity gap between the two policies grow? When does the drain of a short trace stop mattering?
3. Admit waiting requests shortest-prompt-first instead of first-come-first-served. What does it do to the median and the tail of the time to first token?
4. Combine a prefill budget with chunking. Which of the two moves the tail gap more?
5. Sweep the slot count at a fixed offered load. Where does adding slots stop helping, and what does it cost in reserved cache?

## [Stage 5: Paged KV Cache](stage5.md)

1. Allocate lazily: reserve only the blocks a sequence has filled and add one when it needs one. When the pool runs dry, preempt the newest sequence by freeing its blocks and prefilling it again later. How much more memory does this save, and what happens to the tail latency of preempted requests?
2. Time a decode step that gathers through the block table against one that reads a pre-gathered dense tensor, to see how much of the paged step is the copy. Then write the attention so it reads the blocks in place and compare.
3. Share a prefix: give two requests with the same prompt the same prompt blocks, with a reference count, and copy a block only when one sequence writes into it. How much memory does a batch with a long shared system prompt save?
4. Sweep the block size at a tight memory budget, with short requests and again with long ones. Where does the best block size move?
5. Admit the first waiting request that fits instead of strict first come, first served. What does it do to throughput, and to the tail latency of the large request that keeps being passed over?

## [Stage 6: N-gram Speculation](stage6.md)

1. Use `Qwen/Qwen2.5-0.5B` to draft for larger models of the same family and measure how cheap the draft has to be for speculation to win. This is now the next stage, [`06_draft_model_speculation.ipynb`](../06_draft_model_speculation.ipynb).
2. Run speculative decoding inside the continuous or paged scheduler. Each sequence accepts a different number of tokens per pass, so the batch is ragged. What has to change in the decode view and the mask, and how does the measured gain compare with the projection?
3. Speculative sampling (accepting a draft token with probability `min(1, p_target / p_draft)`) needs the draft's probabilities, which an n-gram lookup does not have, so it moved to the next stage.
4. Propose the continuations of the two most recent matches and check both in one pass with a tree-shaped mask. When does it beat a single, longer draft?
5. Sweep the n-gram lengths of the lookup. Does requiring a longer match raise the acceptance, and at what cost in how often a draft is found?

## [Stage 7: Draft-Model Speculation](stage7.md)

1. Implement speculative sampling. Greedy checking accepts a draft token only if it equals the target's argmax. For sampling at a temperature above zero, accept a draft token with probability `min(1, p_target / p_draft)` and, on rejection, resample from the normalised residual `max(0, p_target - p_draft)`. The proposer must now return the draft's probabilities as well. Why does this keep the output distribution exactly the target's, and can you confirm it by sampling many short continuations?
2. Run draft-model speculation inside the continuous or paged scheduler: one draft cache per running sequence and a ragged accept length per sequence per round. What has to change in the draft's decode view, and how does the measured gain compare with the projection?
3. Build a hybrid proposer that tries the n-gram match first and calls the draft model only when there is no match. When does it beat both proposers, and how often is each side used per workload?
4. Use the first few layers of the target as the draft (self-speculation: no second model, shared weights and cache). How does the acceptance depend on how many layers are kept, and where does the product of acceptance and cost peak?
5. Choose `k` per round from the running acceptance estimate and the measured `c` and `v`, by maximising the predicted speedup. Does it beat the best fixed `k` on a mixed workload?
6. Make the draft cheaper (quantize its weights, share its embedding table) and measure `c` again. Which part of the draft step moves, given where its time goes? Stage 8 tried the quantization half on this GPU, with the ways PyTorch offers, and `c` went up (see its section 6); sharing the embedding table is untried.
7. Capture the first guess of each round in a CUDA graph too. Its input has a variable length (the bonus token, and sometimes the last accepted guess as well), so try one graph per length, or always feed the same number of tokens and mask the extra. How much of the gap between the micro-benchmark cost ratio and the cost per guess seen in the run does it close?
8. Time the plain baseline of each workload several times and take the median, then rerun the 3B target. How much of the 3B `copy` speedup was the slow baseline?

## [Stage 8: Weight-Only Int8 Quantization](stage8.md)

1. Use one scale per group of 128 input weights instead of one per row. How much does the error per row fall, and what does it cost in stored bytes and in the matmul?
2. Pack two 4-bit weights per byte with group scales (int4). The bytes halve again; what happens to perplexity, and to the speed of a way that has to unpack them?
3. The gate points here. Write the int8-to-fp16 conversion inside the matmul yourself, as a Triton kernel, or try an existing fused weight-only int8 implementation, and compare it with the three ways of the gate. How close to the copy bandwidth does it get, and does the 3B decode step then come near half of its fp16 time? First find out why the three ways lose: profile the eager conversion apart from the matmul, and print the code `torch.compile` generates for the dequantize to see whether the conversion was fused.
4. Quantize the activations too (W8A8), so the matmul can use the integer path of the tensor cores. When does the extra rounding show up in perplexity, and what does the speed gain depend on?
5. Qwen2.5-7B in fp16 does not fit on a T4 and in int8 it does. Load it layer by layer, quantizing on the GPU as you go, and measure its decode step against its floor.
6. Quantize the KV cache. At long contexts the cache, not the weights, is what a step reads. Quantize keys and values to int8 with a scale per head and token, and measure the error and the step time at a long context.
7. Find the weight channels or activations with the largest values and test whether keeping them in fp16 improves the one-scale-per-matrix control more than the per-channel one. Also locate the positions where the largest logit difference of the per-channel model occurs.
8. Combine Stage 7 and this stage under a real load: does a quantized target move the best draft length, or the winner between the n-gram lookup and the draft model?

## [Stage 9: A Fused Int8 Kernel, and the Quantized Model as Its Own Draft](stage9.md)

1. The `k/v` projections have few output rows, so a one-token kernel gives the T4's multiprocessors little to do. Split the loop over `K` across programs and reduce the partial sums, and see whether the narrow layers stop being launch-bound.
2. Capture the entire int8 decode step in a CUDA graph, not only the draft's single-token step, and compare it with eager mode for int8 and fp16. How much of the step is launch cost?
3. Pack two 4-bit weights per byte with a scale per group of 128 (int4). The kernel unpacks in registers; what happens to perplexity, to the bandwidth the kernel reaches, and to the acceptance of a 4-bit self-draft?
4. Let the int8 self-draft propose several candidates per position and check them in one pass with a tree mask. Does the higher acceptance still pay once the checking pass is wider?
5. Quantize the activations too (W8A8) and use the integer path of the tensor cores. When does the extra rounding show in perplexity and in acceptance?
6. Qwen2.5-7B in fp16 does not fit on a T4 and in int8 it does. Load it layer by layer and measure its decode step against its floor with this kernel.
7. Quantize the KV cache. At long contexts the cache, not the weights, is what a step reads. Quantize keys and values to int8 and measure the step time at a long context.
8. Profile the kernel and find whether the conversion, the reduction or the memory system limits it, then change the tile shape or the loop to move it.

## [Stage 10: A Decode Loop That Lives in CUDA Graphs, and Do Int8 and Speculation Multiply](stage10.md)

1. Replace the three-dimensional accumulator of the multi-row kernel by one two-dimensional accumulator per row in an unrolled loop, each updated from the same int8 tile. Does the time per layer stop growing with the rows?
2. The winners sit at the largest tile of the grid and the smallest warp count. Extend the grid (a wider tile, a longer step along `K`, other warp counts) and see where the optimum is.
3. Profile the multi-row kernel (Nsight Compute or `triton.testing`) and find whether the conversion, the FMAs, the layout conversions or the memory system limits it at eight rows.
4. `tl.dot` needs sixteen rows, and the Stage 9 tile is flat in the rows at a tenth to a fifth of the fp16 layer's speed. Use a transposed formulation (the weights as the sixteen-row operand, the few activation rows as the narrow one) and see whether the tensor cores beat the FMA units.
5. Make `DraftModel` feed the tokens the target accepted as one captured fixed-length pass, so a round with a 0.5B draft is graph launches end to end. What does the cost ratio become, and does the 0.5B draft beat the lookup on open text?
6. Capture checking passes for several lengths and run the smallest that fits the drafts found, instead of padding to `k + 1`. When does it pay, given that every extra graph costs memory?
7. Replace the greedy acceptance with speculative sampling: accept a draft with probability `min(1, p / q)` and resample on rejection. The output distribution stays the target's. What changes in the rollback?
8. The multi-row kernel stops at eight rows. Where is the crossover with the dequantize path, and with a kernel tuned for the batch sizes of a server?
9. The `k/v` projections have few output rows. Split the loop over `K` across programs and reduce the partial sums; does the pass stop being launch-bound inside the graph?
10. Capture the prompt pass for a few bucketed lengths. How much of the time to the first token is launches?
11. Check several candidate continuations per position in one pass with a tree mask. With a fixed-shape graph, what is the mask, and what does a wider block cost on int8?

## [Stage 11: A Better Multi-Row Int8 Kernel, and Does It Make Int8 and Speculation Multiply](stage11.md)

1. The grids carry the conversion of `x` as always on. Add it as a dimension of the search for the tiles that need the fewest instructions, and see which layers pick the fp16 `x` (the run says `mlp gate/up` and `mlp down` should).
2. `q`, `k` and `v` read the same input, and so do `gate` and `up`. Convert `x` once per input, or have the norm write fp32 directly, and measure the whole pass and not the layer.
3. Profile the winners with Nsight Compute: the machine code was only counted offline. Find whether the conversion, the multiply-adds, the cache reads of `x` or the memory system limits them at five rows.
4. Load the next weight tile into registers while the current one is multiplied, so more bytes are in flight at the low occupancy these kernels have.
5. The time of the winner against the rows has a knee. Move it: change what is unrolled, split the rows across two programs, or process a pair of rows per weight read, and measure where the knee goes.
6. `tl.dot` needs sixteen rows, so a checking pass pads. Use a transposed formulation (the weights as the sixteen-row operand, the few activation rows as the narrow one) and see whether the tensor cores beat the FMA units.
7. The narrow layers still choose `gemvm`. Split the loop over `K` across programs and reduce the partial sums, or relax the fit rule for them, and see whether the new kernels take them.
8. Tune in turns: confirm the top few candidates of every case with `interleaved_times` before installing a winner, and see how much `v` falls.
9. Run the sweeps at draft lengths of one to three. The cost model says what a two-token or three-token pass is worth against the tokens it yields.
10. Make `DraftModel` feed the tokens the target accepted as one captured fixed-length pass, so a round with a 0.5B draft is graph launches end to end.
11. Replace the greedy acceptance with speculative sampling: accept a draft with probability `min(1, p / q)` and resample on rejection. What changes in the rollback?

## [Stage 12: The Engine, and a Head-to-Head with Hugging Face and vLLM](stage12.md)

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

## [Stage 13: The Batched Step](stage13.md)

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

## [Stage 14: The In-Place Paged Attention Kernel](stage14.md)

1. **Tune the kernels.** Load several `block_size` slots of keys per loop iteration and vary the number of warps; read the best setting from a measurement, as in Stage 10.
2. **Split a long row.** Divide the key positions of one row across several programs and combine their partial softmaxes (flash-decoding); find the length where it pays.
3. **Fuse the write.** Write the new token's key and value inside the kernel, so they are written and read in one launch.
4. **Wider groups and other head sizes.** Let the grouped kernel serve more than sixteen query heads per key-value head and a head size that is not a power of two.
5. **The checking pass.** Use the kernel for the several query rows of the checking pass of speculation, with the causal pattern inside the new block.
6. **An int8 cache.** Quantize the KV cache and read it in the kernel (exercise `8.6`); measure at long contexts.
7. **The closed loop against vLLM.** Run `run_closed_http` against vLLM and compare the slopes of time per token with this engine's variants.
8. **Find the next bottleneck.** Profile the best variant at the top rate and name the largest family; is it the matrix products, and what would shrink them?
