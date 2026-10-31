"""
Where a batched decode step spends its time, and how to read it (Stage 13).

Stage 12 left one plain fact: under load vLLM's time per token stays flat while this engine's rises, because a captured step does
work for every padded row and every padded column, and the grouped-query attention it runs copies the whole gathered cache six times over
before reading it. This module has the tools the Stage 13 notebook uses to see that and to check the fixes against it.

  attention_bytes(...)            the bytes one layer's cache gather, key/value expansion and attention read move, from the shapes alone.
  time_attention_pieces(...)      the same pieces timed on random tensors of the real shapes: the gather, the expansion, the attention
                                  over the expanded copy, the attention over the unexpanded cache (folded), and the two whole paths.
  profile_step(fn)                a torch.profiler view of any callable, with kernels grouped into families (attention, matmul, copies).
  decode_steps / step_table       the engine's step log (EngineConfig.record_steps) reduced to the decode steps and their median time
                                  by running batch, or by (batch bucket, block bucket).
  step_curves                     several variants' step tables in one list, ready for bench.plots.line_groups.
  crossover_batch / fallback_setting
                                  the running batch from which the eager step is not slower than the captured one, and the engine
                                  setting that follows from it (EngineConfig.graph_max_batch).

Everything here runs on a CPU too (the timings are then of the CPU, which tests the code and says nothing about a GPU).
"""
import statistics
from collections import defaultdict

import torch

from bench.metrics import time_in_graph


# ---- the bytes ----------------------------------------------------------------------------------------------------------

def attention_bytes(B, NB, block_size, n_heads, n_kv, head_dim, dtype_bytes=2, mode="expand", fill=1.0):
    """
    Bytes moved by ONE layer's attention for a batched decode step of B rows whose gather is NB blocks wide, counting each pass over the
    data as one read plus one write. `kv` is the size of the gathered keys (or values) of the whole batch.

      gather   the keys and values copied out of the paged cache twice: once by the block-table index, once by the permute that puts the
               sequence axis together. 2 tensors x 2 copies x (read kv + write kv) = 8 kv.
      expand   `repeat_interleave` of the key and value heads up to the query heads: read kv, write n_rep x kv, for each of the two. Zero when
               mode is "fold" or there is one query head per key head.
      attend   the attention kernel reads the keys and values it is given: n_rep x kv each after the expansion, kv each when folded.

      Stage 14: mode "paged" reads the blocks where they are: no gather, no expansion, and the attention reads only the `fill` share of the bucket
      that holds real positions (1.0: every row fills its bucket; padding blocks and the unused tail of a bucket cost nothing).

    Returns the three, their sum as `total`, and `kv`. Estimates: the real kernels fuse some of this and the mask is ignored.
    """
    assert mode in ("expand", "fold", "paged") and n_heads % n_kv == 0 and 0 < fill <= 1
    n_rep = n_heads // n_kv
    kv = B * NB * block_size * n_kv * head_dim * dtype_bytes
    expanded = mode == "expand" and n_rep > 1
    if mode == "paged":
        attend = round(2 * kv * fill)
        return dict(gather=0, expand=0, attend=attend, total=attend, kv=kv)
    gather = 8 * kv
    expand = 2 * (1 + n_rep) * kv if expanded else 0
    attend = 2 * (n_rep if expanded else 1) * kv
    return dict(gather=gather, expand=expand, attend=attend, total=gather + expand + attend, kv=kv)


# ---- the same pieces, timed -----------------------------------------------------------------------------------------------

def time_attention_pieces(B, NB, block_size, n_heads, n_kv, head_dim, device=None, dtype=None, reps=10, iters=15, seed=0, paged=None, fill=0.5):
    """
    Time the pieces of one layer's batched attention on random tensors of the real shapes, each called `reps` times inside one CUDA graph
    (bench.metrics.time_in_graph), so launch gaps are out and what is left is what the kernels take. Median milliseconds per call:

      gather            the keys and values out of a paged cache by a block table, as the captured step does it
      expand            repeat_interleave of both up to the query heads
      attend_expanded   the attention over the expanded copy
      attend_folded     the attention over the unexpanded cache with the query heads folded into the rows
      path_expand       gather + expand + attend_expanded, the engine's default
      path_fold         gather + attend_folded
      paged_<kernel>    Stage 14: the in-place kernels of minivllm/paged_attention.py (no gather) for each kernel named in `paged` (default: "simple" and
                        "grouped" on a CUDA device with Triton, none otherwise), over rows whose length is the `fill` share of the bucket

    Returns {name: median_ms, ..., "spread": the largest interquartile spread of the six, "mode": "graph" or "eager", "device": ...}.
    """
    from minivllm.model import gqa_attention
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    assert n_heads % n_kv == 0
    n_rep, L, blocks = n_heads // n_kv, NB * block_size, B * NB
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    cache_k, cache_v = rand(blocks, n_kv, block_size, head_dim), rand(blocks, n_kv, block_size, head_dim)
    table = torch.randperm(blocks, generator=g).reshape(B, NB).to(device)             # every row owns its blocks, as real requests do
    q = rand(B, n_heads, 1, head_dim)
    mask = (torch.arange(L, device=device)[None, :] <= L // 2)[:, None, None, :].expand(B, 1, 1, L).contiguous()     # as the captured step builds it

    def gather():
        out = []
        for t in (cache_k, cache_v):
            out.append(t[table].permute(0, 2, 1, 3, 4).reshape(B, n_kv, L, head_dim))
        return out

    k, v = gather()
    kx, vx = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)

    def expand():
        return k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)

    def attend_expanded():
        return torch.nn.functional.scaled_dot_product_attention(q, kx, vx, attn_mask=mask)

    def attend_folded():
        return gqa_attention(q, k, v, mask, n_rep, fold=True)

    def path_expand():
        a, b = gather()
        return gqa_attention(q, a, b, mask, n_rep, fold=False)

    def path_fold():
        a, b = gather()
        return gqa_attention(q, a, b, mask, n_rep, fold=True)

    lens = torch.full((B,), max(1, int(fill * L)), device=device)
    if paged is None:
        from minivllm.paged_attention import HAVE_TRITON
        paged = ("simple", "grouped") if HAVE_TRITON and device == "cuda" else ()

    def paged_path(kernel):
        from minivllm.paged_attention import paged_attention
        return lambda: paged_attention(q[:, :, 0], cache_k, cache_v, table, lens, kernel=kernel)

    pieces = dict(gather=gather, expand=expand, attend_expanded=attend_expanded, attend_folded=attend_folded, path_expand=path_expand, path_fold=path_fold, **{f"paged_{kernel}": paged_path(kernel) for kernel in paged})
    out, spreads, modes = {}, [], set()
    with torch.no_grad():
        for name, fn in pieces.items():
            st = time_in_graph(fn, reps=reps, iters=iters)
            out[name] = st["median_ms"]
            spreads.append(st["spread"])
            modes.add(st["mode"])
    out.update(spread=max(spreads), mode=modes.pop() if len(modes) == 1 else "mixed", device=device)
    return out


# ---- a profile of one step -------------------------------------------------------------------------------------------------

FAMILIES = ("attention", "matmul", "copy and index", "elementwise", "other")

_ATTENTION = ("attention", "flash", "fmha", "sdpa", "softmax", "scaled_dot", "paged")
_MATMUL = ("gemm", "gemv", "cutlass", "cublas", "xmma", "ampere_", "turing_", "volta_", "sm75", "sm80", "aten::mm", "aten::addmm", "aten::bmm",
           "aten::linear", "aten::matmul")
_COPY = ("copy", "memcpy", "index", "gather", "scatter", "cat", "repeat", "contiguous", "clone", "permute", "transpose", "expand", "reshape", "to_copy")
_ELEMENTWISE = ("elementwise", "vectorized", "add", "mul", "div", "silu", "sigmoid", "rsqrt", "pow", "mean", "sum", "neg", "cos", "sin",
                "where", "mask", "softmax", "reduce", "layer_norm", "rms")


def kernel_family(name):
    """Which family a kernel or operator name belongs to, by substring: attention, matmul, copy and index, elementwise, other."""
    low = name.lower()
    for family, words in (("attention", _ATTENTION), ("matmul", _MATMUL), ("copy and index", _COPY), ("elementwise", _ELEMENTWISE)):
        if any(w in low for w in words):
            return family
    return "other"


def _device_us(event):
    for attr in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
        value = getattr(event, attr, None)
        if value:
            return float(value)
    return 0.0


def profile_step(fn, steps=5, warmup=3, top=12):
    """
    Run fn() `steps` times under torch.profiler and report the time of one call. On a GPU these are the CUDA kernels' own times; on a CPU the
    operators' self times. Returns {"device": "cuda"|"cpu", "total_ms", "by_family": {family: ms}, "top": [{name, family, ms, calls}, ...]}
    with the `top` costliest kernels (ms and calls are per call of fn).
    """
    from torch.profiler import ProfilerActivity, profile
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if cuda:
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])) as prof:
            for _ in range(steps):
                fn()
            if cuda:
                torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        if cuda:
            if not str(getattr(e, "device_type", "")).endswith("CUDA"):
                continue                                              # the CPU-side launch events would count the kernels twice
            us = _device_us(e)
        else:
            us = float(e.self_cpu_time_total)
        if us > 0:
            rows.append(dict(name=e.key, family=kernel_family(e.key), ms=us / 1000 / steps, calls=e.count / steps))
    rows.sort(key=lambda r: -r["ms"])
    by_family = {f: sum(r["ms"] for r in rows if r["family"] == f) for f in FAMILIES}
    return dict(device="cuda" if cuda else "cpu", total_ms=sum(r["ms"] for r in rows), by_family=by_family, top=rows[:top])


# ---- the engine's step log ---------------------------------------------------------------------------------------------

def decode_steps(step_log, path=None):
    """The decode entries of an engine step log that did not pay for a graph capture, optionally only those that took `path` ("graph" or "eager")."""
    return [s for s in step_log if s["kind"] == "decode" and not s.get("capture_s") and (path is None or s.get("path") == path)]


def step_table(step_log, by=("running",), min_steps=1):
    """
    Median decode step time by one key or several: "running" (requests in the step), "batch_bucket" and "block_bucket" (the captured shape,
    present on graph steps only). One row per key value with at least `min_steps` steps:
    {key: value, ..., n, median_ms, p25_ms, p75_ms}, sorted by the key.
    """
    keys = (by,) if isinstance(by, str) else tuple(by)
    groups = defaultdict(list)
    for s in decode_steps(step_log):
        key = tuple(s.get(k) for k in keys)
        if None not in key:
            groups[key].append(1000 * s["dur_s"])
    rows = []
    for key in sorted(groups):
        ms = groups[key]
        if len(ms) < min_steps:
            continue
        q = statistics.quantiles(ms, n=4) if len(ms) >= 2 else [ms[0], ms[0], ms[0]]
        rows.append(dict(zip(keys, key), n=len(ms), median_ms=statistics.median(ms), p25_ms=q[0], p75_ms=q[2]))
    return rows


def step_curves(logs, by="running", min_steps=3):
    """{variant: step log} to one list of step_table rows with a `variant` column, for bench.plots.line_groups(df, by, "median_ms", "variant", ...)."""
    return [dict(variant=name, **row) for name, log in logs.items() for row in step_table(log, by, min_steps)]


def crossover_batch(graph_log, eager_log, min_steps=5):
    """
    The smallest running batch r such that the eager step's median time is no longer than the captured step's at r and at every larger batch
    both logs have (at least `min_steps` steps each), or None if the eager step is slower than the captured one at the largest common batch
    (or the logs share no batch). From r up, the engine should take the eager path: that is EngineConfig.graph_max_batch = r - 1.
    """
    g = {r["running"]: r["median_ms"] for r in step_table(graph_log, "running", min_steps)}
    e = {r["running"]: r["median_ms"] for r in step_table(eager_log, "running", min_steps)}
    cross = None
    for b in sorted(set(g) & set(e), reverse=True):
        if e[b] > g[b]:
            break
        cross = b
    return cross


def fallback_setting(cross, max_seqs):
    """
    EngineConfig fields that follow from a crossover: {} when there is none (the captured step is never slower: no fallback), `graph_max_batch`
    = cross - 1 when the eager step wins from `cross` up, and graph_decode off when it wins from a batch of one.
    """
    if cross is None:
        return {}
    if cross <= 1:
        return dict(graph_decode=False)
    assert cross <= max_seqs, "a crossover cannot be above the largest batch the engine runs"
    return dict(graph_max_batch=cross - 1)
