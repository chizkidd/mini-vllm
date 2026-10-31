"""
LLMEngine: the pieces of Stages 1 to 10 behind one interface (a request in, tokens out, on a real clock).

The stages were separate experiments, each with its own loop. A server needs one object that owns the model and its KV memory, takes
requests whenever they arrive, and produces tokens as it goes. This engine is that object:

  requests     add_request(prompt_ids, SamplingParams) puts a request in a waiting queue; abort(id) removes it.
  scheduling   continuous batching on a paged KV cache (Stages 4 and 5), first come first served. A request is admitted when the pool has
               blocks for everything it will store (prompt + max_new - 1 tokens), so a running request can never run out of memory half way
               and nothing is ever preempted; its blocks go back the moment it finishes, whether by length or by a stop token.
  each step    one call to step() admits what fits and prefills it (one request at a time, the prompt as one pass), then runs ONE decode
               step for every running request together, each at its own position and reading its own blocks through its block table.
  sampling     per request (minivllm/sampling.py): greedy, or temperature / top-k / top-p from the request's own seeded generator.
  output       step() returns the tokens produced in that step for every request that produced any, so a caller can stream them.
  clock        every timestamp comes from `clock` (time.perf_counter by default), taken after the sampled tokens are on the host, which
               waits for the GPU; a test can pass a fake clock. TTFT, inter-token latency and end-to-end time are on each RequestOutput.

Greedy requests produce exactly the tokens of `generate_cached` (up to fp16 near-ties) whatever else is running; the tests check that
under block pressure and with requests arriving mid-run.

  graph decode  EngineConfig(graph_decode=True) runs the decode step as a captured CUDA graph per (batch size, blocks per sequence)
                bucket (minivllm/paged_graph.py); without CUDA it runs the same bucketed step eagerly, so the logic is tested on a CPU.
  speculation   EngineConfig(speculate_k=k) turns on lookup speculation (Stage 6) in latency mode: while exactly ONE request is running
                and it is greedy, a step guesses up to k tokens by n-gram lookup in its own context, checks them in one pass of k + 1
                tokens over the paged cache and keeps the longest prefix the model agrees with plus one more, so the output is the same
                as plain greedy decoding and a rollback is only not advancing the request's length. With two or more requests running
                (or a sampled one) the engine decodes the batch as usual: speculation is a latency tool, and a batch already shares
                each read of the weights across its rows.

The engine is not thread-safe: one thread calls add_request, abort and step (the server in minivllm/server.py wraps it in a queue).
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from .kv_cache import PagedCache
from .model import GQA_MODES
from .paged_attention import ATTENTION_MODES
from .paged_graph import BUCKET_SCHEMES, PagedGraphs, bucket_for, pad_row, seq_row
from .sampling import SamplingParams, finish_reason, sample_rows
from .speculative import PromptLookup

NULL = "__null__"          # the sequence id that owns the engine's one reserved block (see LLMEngine.__init__)


@dataclass
class EngineConfig:
    """
    num_blocks:            blocks of the KV pool that requests can use (one more is reserved by the engine).
    block_size:            tokens per block.
    max_seqs:              the most requests decoded together.
    max_prefills_per_step: prompts prefilled per step (None: every waiting request that fits), so a burst of arrivals cannot stall the
                           decode steps of the requests already running for long.
    seed:                  derives the seed of a request that does not set its own.
    record_steps:          keep a log of every prefill and decode step (for the benchmarks).
    graph_decode:          run decode steps through the bucketed static step of minivllm/paged_graph.py instead of the eager paged view.
    graph_capture:         with graph_decode, capture each bucket as a CUDA graph (False: the same static step run eagerly, to separate
                           the effect of the buckets from the effect of the graph).
    pad_head:              compute a one-row fp16 output projection on two rows (MiniQwen.head), which a captured step needs on a small model.
    speculate_k:           draft up to this many tokens per round in latency mode (0: no speculation). The checking pass is the static step
                           of minivllm/paged_graph.py with k + 1 tokens, captured when graph_decode is on.
    speculate_ngram_max:   longest n-gram the lookup tries to match (the default proposer is PromptLookup; pass `proposer` to LLMEngine to use another).
    gqa:                   "expand" (the engine of Stage 12) copies the keys and values to as many heads as the queries have in every decode step;
                           "fold" moves the queries that share a key-value head into the query rows and copies nothing (MiniQwen.forward).
    bucket_scheme:         "pow2" rounds a decode step up to the next power of two in batch and in blocks; "fine" to the next size at most a quarter
                           larger (minivllm/paged_graph.py), which pads less and captures more graphs.
    graph_max_batch:       with graph_decode, a decode step with more running requests than this runs the eager paged step instead of the bucketed
                           one (None: never). The step log says which ran.
    attention:             "gather" (the default) copies each row's blocks into dense tensors before attention; "paged" and "paged_grouped" read them
                           where they are with a Triton kernel (minivllm/paged_attention.py, Stage 14: no gather, padding blocks never read, a row costs
                           what its own length costs; one program per query head, or per key-value head); "paged_torch" is the same algorithm in torch,
                           a CPU stand-in that cannot be captured. Applies to one-token decode steps; prefill and the checking pass of speculation gather.
    """
    num_blocks: int = 256
    block_size: int = 16
    max_seqs: int = 32
    max_prefills_per_step: Optional[int] = None
    seed: int = 0
    record_steps: bool = False
    graph_decode: bool = False
    graph_capture: bool = True
    pad_head: bool = False
    speculate_k: int = 0
    speculate_ngram_max: int = 3
    gqa: str = "expand"
    bucket_scheme: str = "pow2"
    graph_max_batch: Optional[int] = None
    attention: str = "gather"

    def __post_init__(self):
        if self.gqa not in GQA_MODES:
            raise ValueError(f"gqa must be one of {GQA_MODES}, got {self.gqa!r}")
        if self.attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {self.attention!r}")
        if self.bucket_scheme not in BUCKET_SCHEMES:
            raise ValueError(f"bucket_scheme must be one of {BUCKET_SCHEMES}, got {self.bucket_scheme!r}")
        if self.graph_max_batch is not None and not 1 <= self.graph_max_batch < self.max_seqs:
            raise ValueError(f"graph_max_batch must be None or between 1 and max_seqs - 1 ({self.max_seqs - 1}), got {self.graph_max_batch!r}")


@dataclass
class RequestOutput:
    request_id: int
    prompt_len: int
    tokens: List[int]
    finish_reason: Optional[str]
    arrival_s: float
    admit_s: Optional[float]
    first_token_s: Optional[float]
    finish_s: float
    token_times: List[float]

    @property
    def ttft_s(self):
        return None if self.first_token_s is None else self.first_token_s - self.arrival_s

    @property
    def e2e_s(self):
        return self.finish_s - self.arrival_s

    @property
    def itl_s(self):
        """The gaps between consecutive tokens (empty for a one-token output)."""
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


@dataclass
class StepOutput:
    request_id: int
    new_tokens: List[int]
    finished: bool
    finish_reason: Optional[str]


@dataclass
class _Seq:
    id: int
    prompt_ids: List[int]
    params: SamplingParams
    need: int
    arrival_s: float
    tokens: List[int] = field(default_factory=list)
    token_times: List[float] = field(default_factory=list)
    admit_s: Optional[float] = None
    first_token_s: Optional[float] = None
    generator: Optional[torch.Generator] = None


class LLMEngine:
    def __init__(self, model, config=None, clock: Optional[Callable[[], float]] = None, proposer=None):
        self.model, self.cfg = model, config or EngineConfig()
        self.clock = clock or time.perf_counter
        cfg = self.cfg
        # one block more than requests may use, owned by NULL: padding rows of a captured step write their keys and values there
        self.cache = PagedCache(model.n_layers, cfg.num_blocks + 1, cfg.block_size, model.n_kv, model.hd, model.device, model.dtype)
        self.cache.add(NULL, cfg.block_size)
        self.null_block = self.cache.tables[NULL][0]
        self.graphs = (PagedGraphs(model, self.cache, self.null_block, cfg.max_seqs, cfg.num_blocks, capture=cfg.graph_decode and cfg.graph_capture,
                                   pad_head=cfg.pad_head, gqa=cfg.gqa, bucket_scheme=cfg.bucket_scheme, attention=cfg.attention) if cfg.graph_decode or cfg.speculate_k > 0 else None)
        self.proposer = proposer or PromptLookup(ngram_max=cfg.speculate_ngram_max)
        self.waiting, self.running = deque(), {}           # running: request id -> _Seq, in admission order
        self.finished = {}                                 # request id -> RequestOutput, until popped
        self._next_id = 0
        self.step_log, self.counters = [], dict(steps=0, prefills=0, decode_steps=0, tokens=0, spec_rounds=0, spec_drafted=0, spec_accepted=0)

    # ---- requests ---------------------------------------------------------------------------------------------------

    def add_request(self, prompt_ids, params=None, request_id=None, arrival_s=None):
        """Queue a request and return its id. Raises ValueError for a request that could never run (empty, or longer than the pool or the RoPE table)."""
        params = params or SamplingParams()
        prompt_ids = [int(t) for t in prompt_ids]
        if not prompt_ids:
            raise ValueError("empty prompt")
        need = len(prompt_ids) + params.max_new - 1
        if self.cache.blocks_for(need) > self.cfg.num_blocks:
            raise ValueError(f"request needs {self.cache.blocks_for(need)} blocks and the pool has {self.cfg.num_blocks}")
        if need + 1 > self.model.cos.shape[0]:
            raise ValueError("request is longer than the RoPE table")
        rid = self._next_id if request_id is None else request_id
        if rid in self.finished or rid in self.running or any(s.id == rid for s in self.waiting):
            raise ValueError(f"request id {rid} is already in use")
        self._next_id = max(self._next_id, rid + 1) if isinstance(rid, int) else self._next_id
        self.waiting.append(_Seq(id=rid, prompt_ids=prompt_ids, params=params, need=need, arrival_s=self.clock() if arrival_s is None else arrival_s))
        return rid

    def abort(self, request_id):
        """Drop a request, waiting or running (its blocks are freed). Returns False if there is no such unfinished request."""
        for s in list(self.waiting):
            if s.id == request_id:
                self.waiting.remove(s)
                self._record(s, "abort", self.clock())
                return True
        if request_id in self.running:
            self._finish(self.running[request_id], "abort", self.clock())
            return True
        return False

    def has_unfinished(self):
        return bool(self.waiting or self.running)

    def pop_output(self, request_id):
        return self.finished.pop(request_id)

    # ---- one iteration ----------------------------------------------------------------------------------------------

    def step(self):
        """Admit and prefill what fits, then one decode step for every running request. Returns a StepOutput per request that produced tokens."""
        produced = {}                                      # request id -> tokens made this step
        admitted = 0
        while (self.waiting and len(self.running) < self.cfg.max_seqs
               and (self.cfg.max_prefills_per_step is None or admitted < self.cfg.max_prefills_per_step)
               and self.cache.can_admit(self.waiting[0].need)):       # first come, first served: the head of the queue waits for memory
            seq = self.waiting.popleft()
            produced.setdefault(seq.id, []).append(self._prefill(seq))
            admitted += 1
        if self.running:
            for rid, tok in self._latency_step() or self._decode():
                produced.setdefault(rid, []).append(tok)
        self.counters["steps"] += 1
        outs = []
        for rid, toks in produced.items():
            done = self.finished.get(rid)
            outs.append(StepOutput(request_id=rid, new_tokens=toks, finished=done is not None, finish_reason=None if done is None else done.finish_reason))
        return outs

    def generate(self, prompts, params=None):
        """Run a list of prompts to completion and return their RequestOutputs in order (a convenience over add_request and step)."""
        plist = params if isinstance(params, (list, tuple)) else [params] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_unfinished():
            self.step()
        return [self.pop_output(i) for i in ids]

    # ---- internals --------------------------------------------------------------------------------------------------

    def _generator(self, seq):
        if seq.generator is None:
            seq.generator = torch.Generator(device=self.model.device)
            seed = seq.params.seed if seq.params.seed is not None else self.cfg.seed * 1_000_003 + (seq.id if isinstance(seq.id, int) else hash(seq.id) % 1_000_003)
            seq.generator.manual_seed(int(seed))
        return seq.generator

    def _prefill(self, seq):
        t0 = self.clock()
        self.cache.add(seq.id, seq.need)
        ids = torch.tensor([seq.prompt_ids], device=self.model.device)
        logits = self.model.forward(ids, self.cache.prefill_view(seq.id))                       # [1, vocab]: the last position
        tok = sample_rows(logits, [seq.params], [self._generator(seq) if not seq.params.greedy else None])[0]
        now = self.clock()
        seq.admit_s = t0
        self.counters["prefills"] += 1
        if self.cfg.record_steps:
            self.step_log.append(dict(kind="prefill", start_s=t0, dur_s=now - t0, prompt_len=len(seq.prompt_ids)))
        self._append(seq, tok, now, running=True)
        return tok

    def _decode(self):
        seqs = list(self.running.values())
        t0 = self.clock()
        lens = [self.cache.lens[s.id] for s in seqs]
        info = {}
        cfg = self.cfg
        static = cfg.graph_decode and (cfg.graph_max_batch is None or len(seqs) <= cfg.graph_max_batch)     # lever 3: big batches take the eager step
        info["path"] = "graph" if static else "eager"
        logits = self._decode_static(seqs, lens, info) if static else self._decode_eager(seqs, lens)
        toks = sample_rows(logits[:len(seqs)], [s.params for s in seqs], [None if s.params.greedy else self._generator(s) for s in seqs])
        now = self.clock()
        self.counters["decode_steps"] += 1
        if self.cfg.record_steps:
            self.step_log.append(dict(kind="decode", start_s=t0, dur_s=now - t0, running=len(seqs), waiting=len(self.waiting), **info))
        out = []
        for seq, tok in zip(seqs, toks):
            self.cache.lens[seq.id] += 1
            self._append(seq, tok, now)
            out.append((seq.id, tok))
        return out

    def _decode_eager(self, seqs, lens):
        """One token for every running request through the paged view, which gathers only as many columns as the longest one needs."""
        view = self.cache.decode_view([s.id for s in seqs], self.cfg.attention)
        positions = torch.tensor(lens, device=self.model.device)[:, None]                       # [n, 1]
        mask = (torch.arange(view.end, device=self.model.device)[None, :] <= positions)[:, None, None, :]
        last = torch.tensor([s.tokens[-1] for s in seqs], device=self.model.device)[:, None]
        return self.model.forward(last, view, positions=positions, attn_mask=mask, pad_head=self.cfg.pad_head, gqa=self.cfg.gqa)    # [n, vocab]

    def _decode_static(self, seqs, lens, info):
        """The same step through the bucketed static pass: rows padded to a batch bucket, blocks to a block bucket, one copy of the inputs."""
        g, cache, n = self.graphs, self.cache, len(seqs)
        assert all(l < cache.cap[s.id] for s, l in zip(seqs, lens)), "KV reservation overflow"
        B = bucket_for(n, g.batch_sizes)
        NB = bucket_for(max(cache.blocks_for(l + 1) for l in lens), g.block_sizes)
        captured = g.capture_s
        step = g.step(B, NB)
        rope = self.model.cos.shape[0]
        rows = [seq_row(cache, s.id, self.null_block, [s.tokens[-1]], l, NB, rope) for s, l in zip(seqs, lens)]
        step.load(rows + [pad_row(1, NB, self.null_block)] * (B - n))
        info.update(batch_bucket=B, block_bucket=NB, capture_s=g.capture_s - captured)
        return step.replay()                                                                    # [B, vocab]; the rows past n are padding

    def _latency_step(self):
        """
        One round of lookup speculation when exactly one greedy request is running and the lookup found something to check; returns the
        (request id, token) pairs it produced, or None when the caller should run an ordinary decode step instead.
        """
        if self.cfg.speculate_k <= 0 or len(self.running) != 1:
            return None
        seq = next(iter(self.running.values()))
        k = min(self.cfg.speculate_k, seq.params.max_new - len(seq.tokens) - 1)      # the drafts, the checked token and the bonus must fit in max_new
        if not seq.params.greedy or k <= 0:
            return None
        drafts = [int(d) for d in self.proposer(seq.prompt_ids + seq.tokens, k)][:k]
        if not drafts:
            return None
        g, cache, kmax = self.graphs, self.cache, self.cfg.speculate_k
        t0 = self.clock()
        start = cache.lens[seq.id]
        assert start + len(drafts) < cache.cap[seq.id], "KV reservation overflow"
        NB = bucket_for(min(cache.blocks_for(start + len(drafts) + 1), self.cfg.num_blocks), g.block_sizes)
        captured = g.capture_s
        step = g.step(1, NB, T=kmax + 1, all_logits=True)
        block = [seq.tokens[-1]] + drafts + [0] * (kmax - len(drafts))                # zeros pad a short draft to the captured length; their rows are ignored
        step.load([seq_row(cache, seq.id, self.null_block, block, start, NB, self.model.cos.shape[0])])
        choices = step.replay()[0].argmax(-1).tolist()                               # the model's own greedy choice after each fed token
        a = 0
        while a < len(drafts) and choices[a] == drafts[a]:
            a += 1
        now = self.clock()
        self.counters["spec_rounds"] += 1
        self.counters["spec_drafted"] += len(drafts)
        self.counters["spec_accepted"] += a
        if self.cfg.record_steps:
            self.step_log.append(dict(kind="verify", start_s=t0, dur_s=now - t0, running=1, waiting=len(self.waiting), drafted=len(drafts), accepted=a,
                                      block_bucket=NB, capture_s=g.capture_s - captured))
        cache.lens[seq.id] += a + 1                  # the rollback: the fed tokens after the a accepted drafts are simply not counted
        out = []
        for tok in choices[:a + 1]:
            self._append(seq, tok, now)
            out.append((seq.id, tok))
            if seq.id not in self.running:           # a stop token inside the accepted block ends the request there
                break
        return out

    def _append(self, seq, tok, now, running=False):
        seq.tokens.append(tok)
        seq.token_times.append(now)
        self.counters["tokens"] += 1
        if seq.first_token_s is None:
            seq.first_token_s = now
        reason = finish_reason(tok, len(seq.tokens), seq.params)
        if running:
            self.running[seq.id] = seq
        if reason is not None:
            self._finish(seq, reason, now)

    def _finish(self, seq, reason, now):
        self.cache.free(seq.id)
        self.running.pop(seq.id, None)
        self._record(seq, reason, now)

    def _record(self, seq, reason, now):
        self.finished[seq.id] = RequestOutput(request_id=seq.id, prompt_len=len(seq.prompt_ids), tokens=list(seq.tokens), finish_reason=reason,
                                              arrival_s=seq.arrival_s, admit_s=seq.admit_s, first_token_s=seq.first_token_s, finish_s=now,
                                              token_times=list(seq.token_times))
