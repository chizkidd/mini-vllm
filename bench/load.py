"""
Load generation for comparing serving systems on the same trace (the engine, Hugging Face `generate`, vLLM).

A trace is a list of `minivllm.scheduler.Request` (prompt token ids, tokens to generate, arrival time in seconds from the run start),
built by the same `make_requests` and `with_arrivals` as the load sweeps of Stages 4 and 5, so a trace is fixed by (workload, n, rate,
seeds) and `trace_digest` proves two runs used the same one. Load is OPEN LOOP: a request is sent at its arrival time whether or not the
system has finished the earlier ones, which is what makes a slow system's queue grow the way it would under real traffic. (A closed loop,
where a client waits for its reply before sending the next request, hides overload.)

Every system is driven to the same contract and produces `RequestResult`s on the same clock:

  run_engine(engine, trace)           the LLMEngine in process, one thread: add the requests that have arrived, step, repeat.
  run_http(url, trace)                any server with a streaming /v1/completions (the engine's own server, vLLM's): one thread per
                                      request sleeps until its arrival, sends it, and timestamps every server-sent event.
  run_closed(engine, n, ...)          the same two, CLOSED loop (Stage 13): n requests in flight, a new one the moment one finishes.
  run_closed_http(url, n, ...)        It measures what a system does at a fixed concurrency (time per token against batch) with no queue
                                      growing, which is the question when the cost of a batched step is what is being studied.
  run_hf(model, trace, max_batch)     Hugging Face `generate`: requests that have arrived are served first come first served, up to
                                      `max_batch` at once (1 is sequential serving, more is static batching: left padded, running until
                                      the longest request in the batch has its tokens). A streamer timestamps every step. Generous to
                                      Hugging Face: a request counts as finished at its own last token, not when its batch returns.

Every request generates exactly its `max_new` tokens (no stopping on the end-of-sequence token), so all systems do the same work.

`summarize` reduces results to time to first token (TTFT), time per output token (TPOT, what a reader of the stream experiences), response
time, the longest stall between two events of a stream, request and token throughput, and goodput: tokens of the requests that met a
latency target, per second of the run.
"""
import hashlib
import http.client
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from bench.metrics import pct
from bench.workload import WORKLOADS, make_requests, with_arrivals


# ---- traces -----------------------------------------------------------------------------------------------------------

def make_trace(workload="mixed", n=48, rate=2.0, seed=0, arrival_seed=1, vocab=(1000, 20000)):
    """
    n requests of a named workload (bench.workload.WORKLOADS, or a dict of its keys) with Poisson arrivals at `rate` requests per second
    (rate None or 0: all arrive at time 0). Random token ids stand in for text, which every system accepts as token ids.
    """
    spec = WORKLOADS[workload] if isinstance(workload, str) else workload
    reqs = make_requests(n, seed=seed, vocab=vocab, **spec)
    return with_arrivals(reqs, rate, seed=arrival_seed) if rate else reqs


def trace_digest(trace):
    """A short hash of everything that defines a trace, to record that two runs (two notebooks, two machines) used the same one."""
    h = hashlib.sha1()
    for r in trace:
        h.update(json.dumps([r.id, round(r.arrival_s, 6), r.max_new, list(r.prompt_ids)]).encode())
    return h.hexdigest()[:12]


def trace_info(trace):
    lens = [len(r.prompt_ids) for r in trace]
    return dict(n=len(trace), digest=trace_digest(trace), prompt_tokens=sum(lens), output_tokens=sum(r.max_new for r in trace),
                mean_prompt=sum(lens) / len(lens), max_prompt=max(lens), arrival_span_s=max(r.arrival_s for r in trace) - min(r.arrival_s for r in trace))


# ---- results ----------------------------------------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Times are seconds from the run start. `chunks` is [(time, tokens in this event)]: the tokens of one engine step arrive together."""
    id: int
    arrival_s: float
    prompt_len: int
    max_new: int
    n_tokens: int = 0
    first_token_s: Optional[float] = None
    finish_s: Optional[float] = None
    chunks: List[Tuple[float, int]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self):
        return self.error is None and self.finish_s is not None and self.first_token_s is not None

    @property
    def ttft_s(self):
        return self.first_token_s - self.arrival_s

    @property
    def e2e_s(self):
        return self.finish_s - self.arrival_s

    @property
    def tpot_s(self):
        """Seconds per output token after the first, as a reader of the stream sees them (None for a one-token output)."""
        return (self.finish_s - self.first_token_s) / (self.n_tokens - 1) if self.n_tokens > 1 else None

    @property
    def stalls_s(self):
        """The gaps between consecutive events of the stream."""
        return [b[0] - a[0] for a, b in zip(self.chunks, self.chunks[1:])]


def summarize(results, slo=None):
    """
    The numbers of one run. `slo` is a dict with `ttft_s` and/or `tpot_s` limits: a request meets it when it is within every limit given,
    and goodput counts only the tokens of requests that do (None: goodput is None). All rates are per second of the run's span, which runs
    from the first arrival to the last finish.
    """
    ok = [r for r in results if r.ok]
    out = dict(n_requests=len(results), n_ok=len(ok), n_errors=len(results) - len(ok))
    if not ok:
        return out
    span = max(r.finish_s for r in ok) - min(r.arrival_s for r in results)
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    stalls = [g for r in ok for g in r.stalls_s]
    ttft, e2e = [r.ttft_s for r in ok], [r.e2e_s for r in ok]
    nan = float("nan")
    out.update(
        span_s=span,
        req_s=len(ok) / span,
        out_tok_s=sum(r.n_tokens for r in ok) / span,
        ttft_p50_s=pct(ttft, 50), ttft_p99_s=pct(ttft, 99), ttft_mean_s=sum(ttft) / len(ttft),
        tpot_p50_s=pct(tpot, 50) if tpot else nan, tpot_p99_s=pct(tpot, 99) if tpot else nan,
        e2e_p50_s=pct(e2e, 50), e2e_p99_s=pct(e2e, 99), e2e_mean_s=sum(e2e) / len(e2e),
        stall_p99_s=pct(stalls, 99) if stalls else nan, stall_max_s=max(stalls) if stalls else nan,
    )
    if slo:
        def meets(r):
            return (r.ttft_s <= slo.get("ttft_s", float("inf"))) and (r.tpot_s is None or r.tpot_s <= slo.get("tpot_s", float("inf")))
        good = [r for r in ok if meets(r)]
        out.update(slo=dict(slo), slo_attainment=len(good) / len(results), goodput_tok_s=sum(r.n_tokens for r in good) / span)
    else:
        out.update(slo=None, slo_attainment=None, goodput_tok_s=None)
    return out


def _group(times):
    """[(t, count)] from a list of per-token times: tokens that share a timestamp arrived in one event."""
    chunks = []
    for t in times:
        if chunks and chunks[-1][0] == t:
            chunks[-1][1] += 1
        else:
            chunks.append([t, 1])
    return [(t, n) for t, n in chunks]


# ---- the engine, in process ---------------------------------------------------------------------------------------------

def run_engine(engine, trace, sleep=time.sleep):
    """
    Drive an LLMEngine open loop on its own clock: before each step, add every request whose arrival time has come; when nothing is
    running, sleep until the next arrival. Arrival times are the scheduled ones (not the moment the loop noticed them), so a long step
    that delays admission counts against the engine's TTFT, as it would for a real client.
    """
    from minivllm.sampling import SamplingParams
    clock = engine.clock
    pending = deque(sorted(trace, key=lambda r: r.arrival_s))
    t0 = clock()
    while pending or engine.has_unfinished():
        now = clock() - t0
        while pending and pending[0].arrival_s <= now:
            r = pending.popleft()
            engine.add_request(r.prompt_ids, SamplingParams(max_new=r.max_new), request_id=r.id, arrival_s=t0 + r.arrival_s)
        if engine.has_unfinished():
            engine.step()
        elif pending:
            sleep(max(0.0, pending[0].arrival_s - (clock() - t0)))
    return _engine_results(engine, trace, {r.id: r.arrival_s for r in trace}, t0)


def _engine_results(engine, trace, arrivals, t0):
    """Take every request's output from the engine and turn it into a RequestResult; `arrivals` is {id: seconds from t0 the client sent it}."""
    results = []
    for r in trace:
        o = engine.pop_output(r.id)
        times = [t - t0 for t in o.token_times]
        results.append(RequestResult(id=r.id, arrival_s=arrivals[r.id], prompt_len=len(r.prompt_ids), max_new=r.max_new, n_tokens=len(o.tokens),
                                     first_token_s=times[0] if times else None, finish_s=o.finish_s - t0, chunks=_group(times),
                                     error=None if o.finish_reason in ("length", "stop") else f"finished with {o.finish_reason}"))
    return results


def closed_trace(n_requests, prompt_len=128, max_new=64, seed=0, vocab=(1000, 20000)):
    """n_requests requests of one prompt length and one output length, all with arrival time 0 (a closed loop sends them as slots free up)."""
    return make_requests(n_requests, seed=seed, vocab=vocab, prompt_lens=(prompt_len,), short_out=(max_new, max_new), long_out=(max_new, max_new), p_long=0.0)


def run_closed(engine, n_concurrent, n_requests, prompt_len=128, max_new=64, seed=0, vocab=(1000, 20000), trace=None):
    """
    Drive an LLMEngine CLOSED loop: keep `n_concurrent` requests in flight, and send the next the moment one finishes, until `n_requests` have
    been served. The time a request is sent is its arrival, so its TTFT includes any wait for a free slot in the engine (none when
    n_concurrent <= EngineConfig.max_seqs and the cache has room). `trace` replaces the generated requests (their arrival times are ignored).
    """
    from minivllm.sampling import SamplingParams
    todo = deque(trace if trace is not None else closed_trace(n_requests, prompt_len, max_new, seed, vocab))
    requests = list(todo)
    clock, sent, finished = engine.clock, {}, 0
    t0 = clock()
    while todo or engine.has_unfinished():
        while todo and len(sent) - finished < n_concurrent:
            r = todo.popleft()
            sent[r.id] = clock() - t0
            engine.add_request(r.prompt_ids, SamplingParams(max_new=r.max_new), request_id=r.id, arrival_s=t0 + sent[r.id])
        finished += sum(out.finished for out in engine.step())
    return _engine_results(engine, requests, sent, t0)


# ---- a server over HTTP -------------------------------------------------------------------------------------------------

def _stream_request(host, port, path, body, timeout, clock):
    """POST a streaming completion; returns ([(time, tokens in the event)], completion_tokens from the usage event or None)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read(300).decode(errors='replace')}")
        chunks, usage = [], None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            now = clock()
            obj = json.loads(payload)
            if obj.get("error"):
                raise RuntimeError(str(obj["error"]))
            if obj.get("usage"):
                usage = obj["usage"].get("completion_tokens", usage)
            for ch in obj.get("choices") or []:
                n = len(ch["tokens"]) if "tokens" in ch else (1 if ch.get("text") else 0)       # this repo's server lists tokens; others send a text delta
                if n:
                    chunks.append((now, n))
        return chunks, usage
    finally:
        conn.close()


def run_http(url, trace, model="model", timeout=600.0, sleep=time.sleep, clock=time.perf_counter, extra=None):
    """
    Open-loop load over HTTP against a streaming /v1/completions (this repo's server, or vLLM's OpenAI-compatible server). One thread per
    request sleeps until its arrival time, sends token ids with temperature 0 and ignore_eos, and timestamps every event. The token count
    is the server's own `usage` when it sends one, else the number of events that carried text.
    """
    u = urlparse(url)
    t0 = clock()
    results = {r.id: RequestResult(id=r.id, arrival_s=r.arrival_s, prompt_len=len(r.prompt_ids), max_new=r.max_new) for r in trace}

    def worker(r):
        sleep(max(0.0, t0 + r.arrival_s - clock()))
        _send(u, r, results[r.id], model, timeout, clock, t0, extra)

    threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in trace]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [results[r.id] for r in trace]


def _send(u, r, res, model, timeout, clock, t0, extra):
    """Send request `r` now and fill in `res` from the stream (or its error); times are seconds from `t0`."""
    body = dict(model=model, prompt=list(r.prompt_ids), max_tokens=r.max_new, temperature=0, stream=True, ignore_eos=True,
                stream_options={"include_usage": True}, **(extra or {}))
    try:
        chunks, usage = _stream_request(u.hostname, u.port, "/v1/completions", body, timeout, clock)
        chunks = [(t - t0, n) for t, n in chunks]
        res.chunks, res.n_tokens = chunks, usage if usage is not None else sum(n for _, n in chunks)
        res.first_token_s = chunks[0][0] if chunks else None
        res.finish_s = clock() - t0 if not chunks else chunks[-1][0]
        if not chunks:
            res.error = "no tokens"
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"


def run_closed_http(url, n_concurrent, n_requests, prompt_len=128, max_new=64, seed=0, vocab=(1000, 20000), trace=None, model="model",
                    timeout=600.0, clock=time.perf_counter, extra=None):
    """
    Closed-loop load over HTTP: `n_concurrent` client threads, each taking the next request of the trace as soon as its previous one has
    finished, until `n_requests` have been served. Same requests (the same token ids) as `run_closed`, so a server and the in-process engine
    can be given the same sweep. A request's arrival is the moment its thread sent it.
    """
    u = urlparse(url)
    todo = deque(trace if trace is not None else closed_trace(n_requests, prompt_len, max_new, seed, vocab))
    requests, lock, results = list(todo), threading.Lock(), {}
    t0 = clock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                r = todo.popleft()
            res = results[r.id] = RequestResult(id=r.id, arrival_s=clock() - t0, prompt_len=len(r.prompt_ids), max_new=r.max_new)
            _send(u, r, res, model, timeout, clock, t0, extra)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(n_concurrent, len(requests)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [results[r.id] for r in requests]


# ---- Hugging Face generate -------------------------------------------------------------------------------------------------

class _StepStamps:
    """A `streamer` for generate(): records the clock at every step. The first put() is the prompt, the rest are one new token per row."""
    def __init__(self, clock):
        self.clock, self.times = clock, []

    def put(self, value):
        self.times.append(self.clock())

    def end(self):
        pass


def run_hf(model, trace, max_batch=1, pad_id=0, clock=time.perf_counter, sleep=time.sleep, sync=None):
    """
    Serve a trace with `model.generate` (a transformers causal LM), first come first served, up to `max_batch` requests at a time, as one
    worker would: take the requests that have arrived, left-pad them to the longest prompt, generate until the longest `max_new` of the batch
    is done (every request gets exactly its tokens: end-of-sequence is suppressed), then look at the queue again. `sync` (e.g.
    torch.cuda.synchronize) is called before each timestamp so asynchronous GPU work is counted.
    """
    import torch
    pending = deque(sorted(trace, key=lambda r: r.arrival_s))
    device = next(model.parameters()).device
    results, t0 = {}, clock()
    while pending:
        now = clock() - t0
        if pending[0].arrival_s > now:
            sleep(pending[0].arrival_s - now)
            continue
        batch = []
        while pending and pending[0].arrival_s <= clock() - t0 and len(batch) < max_batch:
            batch.append(pending.popleft())
        longest, steps = max(len(r.prompt_ids) for r in batch), max(r.max_new for r in batch)
        ids = torch.tensor([[pad_id] * (longest - len(r.prompt_ids)) + list(r.prompt_ids) for r in batch], device=device)
        mask = torch.tensor([[0] * (longest - len(r.prompt_ids)) + [1] * len(r.prompt_ids) for r in batch], device=device)
        def stamp():
            if sync:
                sync()
            return clock()

        stamps = _StepStamps(stamp)
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps, min_new_tokens=steps, do_sample=False, pad_token_id=pad_id,
                           streamer=stamps)
        times = [t - t0 for t in stamps.times[1:]]            # one per generated step; the first put() was the prompt
        for r in batch:
            mine = times[:r.max_new]
            results[r.id] = RequestResult(id=r.id, arrival_s=r.arrival_s, prompt_len=len(r.prompt_ids), max_new=r.max_new, n_tokens=len(mine),
                                          first_token_s=mine[0] if mine else None, finish_s=mine[-1] if mine else None, chunks=_group(mine),
                                          error=None if len(mine) == r.max_new else f"only {len(mine)} of {r.max_new} tokens")
    return [results[r.id] for r in trace]


# ---- helpers for the notebooks ---------------------------------------------------------------------------------------------

def latency_slo(solo_ttft_s, solo_tpot_s, ttft_factor=5.0, tpot_factor=3.0):
    """A latency target relative to what one request gets on an idle system: TTFT within `ttft_factor` times it, TPOT within `tpot_factor` times it."""
    return dict(ttft_s=ttft_factor * solo_ttft_s, tpot_s=tpot_factor * solo_tpot_s)


def run_row(label, results, trace, slo=None, **extra):
    """One row of a comparison table: the label, the trace, and the summary of a run."""
    return dict(system=label, trace=trace_info(trace)["digest"], **extra, **summarize(results, slo))
