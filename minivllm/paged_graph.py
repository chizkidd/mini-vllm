"""
Captured decode steps over the paged KV cache (the engine's CUDA-graph path).

Stage 10 put one sequence's decode step in a CUDA graph by giving it a static cache: every shape and every address the same on every
step. A server decodes a changing set of requests, each with its own length and its own blocks, so the same trick needs two more ideas.

  Buckets.   A graph has fixed shapes, so the engine keeps one graph per (batch size, blocks per sequence) pair it needs, rounded up to
             a bucket (1, 2, 4, 8, ... up to the limit). A step with three running requests uses the batch-4 graph with one padding row;
             a step whose longest request holds 5 blocks uses the 8-block graph. Graphs are captured the first time a bucket is needed
             (or ahead of time with `warmup`), and share one memory pool.
  A fixed-width gather. The eager paged view gathers exactly as many columns as the longest request needs. A captured step gathers every
             block of its bucket, through a block table that is a static buffer, and the attention mask (computed from the positions
             inside the graph) hides the columns past each request's own length. Columns of blocks a request does not own point at the
             engine's reserved NULL block, whose contents the mask never lets anyone read.
  Padding rows. A row that is not a request feeds token 0 at position 0 and writes its key and value into the NULL block, so a padding
             row can never touch a real request's memory.

Everything a step reads is one packed integer buffer, `meta`, that the engine overwrites with ONE host-to-device copy per step: the token
ids, their positions, the block and offset each token's key is written to, and the block table. Sampling stays outside the graph (the
graph returns logits), because every request has its own parameters and its own generator.

A step can be longer than one token (T > 1, one row): that is the checking pass of speculative decoding on the paged cache, which writes
T slots, attends with an explicit mask and rolls back by moving a length (see StaticForward for the same idea on the static cache).

Without a CUDA device, or if capture fails, a step runs the same pass eagerly on the same buffers and says why in `graph_error`, so the
logic (buckets, padding, the NULL block) is tested on a CPU and the graph is a pure speed-up.
"""
import time

import torch


class StaticPagedView:
    """
    The view of a PagedCache for a captured step: B rows of T tokens, NB blocks per row, every index a static device tensor.
    update writes the T keys of each row at (wblk, woff) and returns the first NB blocks of every row, gathered, as [B, n_kv, NB * block_size, hd].
    """
    length = 0                               # positions are passed to forward explicitly

    def __init__(self, cache, table, wblk, woff, attention="gather"):
        self.c, self.table, self.wblk, self.woff = cache, table, wblk, woff
        self.attention = attention                    # Stage 14: the "paged" modes read the blocks in place for one-token rows (see attend)
        self.end = table.shape[1] * cache.block_size

    def write(self, layer, k, v):
        c = self.c
        # two index tensors around a slice put the index dims first: the value has to be [B, T, n_kv, hd]
        c.k[layer, self.wblk, :, self.woff] = k.transpose(1, 2)
        c.v[layer, self.wblk, :, self.woff] = v.transpose(1, 2)

    def attend(self, layer, q, positions):
        """In-place attention of one token per row over each row's own blocks, through the static block table: no gather, and padding blocks are never read."""
        from .paged_attention import attend_in_place
        return attend_in_place(self.c.k[layer], self.c.v[layer], self.table, q, positions, self.attention)

    def update(self, layer, k, v):
        c = self.c
        self.write(layer, k, v)
        B, nb = self.table.shape
        out = []
        for cache_t in (c.k, c.v):
            g = cache_t[layer][self.table]                                    # [B, nb, n_kv, block_size, hd]
            out.append(g.permute(0, 2, 1, 3, 4).reshape(B, c.n_kv_heads, nb * c.block_size, c.head_dim))
        return out[0], out[1]

    def advance(self, T):
        pass


def row_width(T, NB):
    """Integers in one row of a step's `meta` buffer: ids, positions, write blocks and write offsets (T each), then the block table."""
    return 4 * T + NB


def pad_row(T, NB, null_block):
    """A row that is not a request: token 0 at position 0, writing into the NULL block, reading only the NULL block."""
    return [0] * T + [0] * T + [null_block] * T + [0] * T + [null_block] * NB


def seq_row(cache, seq_id, null_block, tokens, start, NB, max_pos):
    """
    The row of a request that feeds `tokens` at positions start .. start + T - 1 (T = 1 for a decode step). Slots past the blocks the
    request owns write into the NULL block (they only exist in a checking pass padded to a fixed length, and their rows are ignored),
    and positions are clamped to the RoPE table for the same reason.
    """
    T, bs, table = len(tokens), cache.block_size, cache.tables[seq_id]
    pos = [start + i for i in range(T)]
    inside = [p < len(table) * bs for p in pos]
    wblk = [table[p // bs] if ok else null_block for p, ok in zip(pos, inside)]
    woff = [p % bs if ok else 0 for p, ok in zip(pos, inside)]
    tbl = list(table[:NB])
    return list(tokens) + [min(p, max_pos - 1) for p in pos] + wblk + woff + tbl + [null_block] * (NB - len(tbl))


class PagedStep:
    """
    One captured pass over the paged cache for a fixed (B, T, NB). `load(rows)` overwrites the static buffers, `replay()` runs the pass and
    returns the logits ([B, vocab] for the last token of each row, or [B, T, vocab] with all_logits). The logits are the graph's output
    tensor: read them before the next replay.
    """
    def __init__(self, model, cache, null_block, B, T, NB, all_logits=False, capture=True, pad_head=False, pool=None, gqa="expand", attention="gather"):
        self.model, self.cache, self.null_block = model, cache, null_block
        self.B, self.T, self.NB, self.all_logits, self.pad_head, self.gqa, self.attention = B, T, NB, all_logits, pad_head, gqa, attention
        self.device = model.device
        self.meta = torch.tensor([pad_row(T, NB, null_block)] * B, dtype=torch.long, device=self.device)
        self.ids, self.pos = self.meta[:, :T], self.meta[:, T:2 * T]
        self.wblk, self.woff, self.table = self.meta[:, 2 * T:3 * T], self.meta[:, 3 * T:4 * T], self.meta[:, 4 * T:]
        self._cols = torch.arange(NB * cache.block_size, device=self.device)
        self.logits, self._graph, self.graph_error, self.mode, self.capture_s = None, None, None, "static", 0.0
        if capture:
            self._try_capture(pool)

    def _forward(self):
        view = StaticPagedView(self.cache, self.table, self.wblk, self.woff, self.attention if self.T == 1 else "gather")
        mask = (self._cols[None, None, :] <= self.pos[:, :, None])[:, None]               # [B, 1, T, NB * block_size]
        return self.model.forward(self.ids, view, positions=self.pos, attn_mask=mask, all_logits=self.all_logits, pad_head=self.pad_head, gqa=self.gqa)

    @torch.no_grad()
    def _try_capture(self, pool):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self.graph_error = "CUDA graphs need a CUDA device"
            return
        t0 = time.perf_counter()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):                      # warm-up on a side stream, as CUDA graph capture requires (writes only to the NULL block)
                    self._forward()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.logits = self._forward()
            self._graph, self.mode = graph, "graph"
            torch.cuda.synchronize()
        except Exception as e:                          # capture can fail for reasons that depend on the torch build
            self._graph, self.graph_error = None, f"{type(e).__name__}: {e}"
        self.capture_s = time.perf_counter() - t0

    @torch.no_grad()
    def load(self, rows):
        """Overwrite the static buffers with `rows` (B lists of row_width(T, NB) ints), in one host-to-device copy."""
        assert len(rows) == self.B, "a step needs exactly B rows (pad with pad_row)"
        self.meta.copy_(torch.tensor(rows, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def replay(self):
        if self._graph is not None:
            self._graph.replay()
        else:
            self.logits = self._forward()
        return self.logits


BUCKET_SCHEMES = ("pow2", "fine")


def buckets(limit, scheme="pow2"):
    """
    The sizes a step is rounded up to, ending at `limit`.
      pow2:  1, 2, 4, 8, ... A step is padded by up to double, in each of the two dimensions.
      fine:  each size at most a quarter above the one before it, once past the smallest sizes (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 27,
             33, ...). A step is padded by about a fifth at most, at the price of more graphs to capture.
    """
    assert scheme in BUCKET_SCHEMES, f"bucket scheme must be one of {BUCKET_SCHEMES}, got {scheme!r}"
    out, b = [], 1
    while b < limit:
        out.append(b)
        b = b * 2 if scheme == "pow2" else max(b + 1, b + b // 4)
    return out + [limit]


def bucket_for(n, sizes):
    """The smallest bucket that holds n."""
    return next(s for s in sizes if s >= n)


class PagedGraphs:
    """
    The steps an engine needs, one per (B, T, NB) bucket, created the first time they are asked for and sharing one graph memory pool.
    `capture=False` keeps the static buffers and the fixed-width gather but runs eagerly (the ablation that separates the effect of the
    graph from the effect of the buckets).
    """
    def __init__(self, model, cache, null_block, max_batch, max_blocks, capture=True, pad_head=False, gqa="expand", bucket_scheme="pow2", attention="gather"):
        self.model, self.cache, self.null_block, self.capture, self.pad_head = model, cache, null_block, capture, pad_head
        self.gqa, self.bucket_scheme, self.attention = gqa, bucket_scheme, attention
        self.batch_sizes, self.block_sizes = buckets(max_batch, bucket_scheme), buckets(max_blocks, bucket_scheme)
        self.steps = {}                                     # (B, T, NB, all_logits) -> PagedStep
        self.pool = torch.cuda.graph_pool_handle() if capture and model.device.type == "cuda" and torch.cuda.is_available() else None

    def step(self, B, NB, T=1, all_logits=False):
        key = (B, T, NB, all_logits)
        if key not in self.steps:
            self.steps[key] = PagedStep(self.model, self.cache, self.null_block, B, T, NB, all_logits=all_logits, capture=self.capture,
                                        pad_head=self.pad_head, pool=self.pool, gqa=self.gqa, attention=self.attention)
        return self.steps[key]

    def warmup(self, batch_sizes=None, block_sizes=None, verify_k=0, max_blocks=None):
        """Capture the one-token step of every listed bucket now, so no capture happens while requests are waiting. verify_k > 0 also
        captures the checking pass of that draft length (one row, k + 1 tokens) for every listed block bucket. max_blocks keeps only the block
        buckets a request of that many blocks can reach (up to the bucket that holds it), so a finer scheme does not capture the unreachable."""
        blocks = self.block_sizes if block_sizes is None else block_sizes
        if max_blocks is not None:
            top = bucket_for(max_blocks, self.block_sizes)
            blocks = [b for b in blocks if b <= top]
        for B in (self.batch_sizes if batch_sizes is None else batch_sizes):
            for NB in blocks:
                self.step(B, NB)
        if verify_k > 0:
            for NB in blocks:
                self.step(1, NB, T=verify_k + 1, all_logits=True)

    @property
    def mode(self):
        """"graph" if every step is a graph, "static" if any of them runs eagerly (no steps yet: "static")."""
        return "graph" if self.steps and all(s.mode == "graph" for s in self.steps.values()) else "static"

    @property
    def graph_error(self):
        return next((s.graph_error for s in self.steps.values() if s.graph_error), None)

    @property
    def capture_s(self):
        return sum(s.capture_s for s in self.steps.values())
