"""
A small HTTP server in front of the LLMEngine (standard library only).

    python -m minivllm.server --model Qwen/Qwen2.5-0.5B --graph --speculate-k 4 --port 8000

    GET  /health            {"status": "ok", "running": n, "waiting": m}
    GET  /v1/stats          the engine's counters
    POST /v1/completions    {"prompt": "text" | [ids], "prompt_token_ids": [ids], "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
                             "stop_token_ids": [ids], "seed": null, "stream": false}

A non-streaming call returns one JSON object when the request finishes. With "stream": true the reply is a server-sent-events stream:
one `data: {...}` event per engine step that produced tokens for the request, then a final event with the finish reason, then
`data: [DONE]`. Text needs a tokenizer (anything with `encode(str) -> ids` and `decode(ids) -> str`); without one, only token ids go in
and come out.

The engine is not thread-safe, so exactly one thread, the engine thread, ever touches it. HTTP handler threads put commands in an inbox
queue (add a request, abort one) and read events from a queue of their own per request; the engine thread drains the inbox, runs a step
and hands each request's new tokens to its queue. A client that disconnects in the middle of a stream has its request aborted, so its
blocks go back to the pool.
"""
import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .llm_engine import EngineConfig, LLMEngine
from .sampling import SamplingParams


class _Handle:
    """What an HTTP handler thread holds for one request: a queue of ("tokens", ids) / ("done", RequestOutput) / ("error", message, status) events."""
    def __init__(self, rid):
        self.rid, self.events = rid, queue.Queue()


class EngineThread:
    """Owns an LLMEngine on a background thread. submit() and abort() are safe to call from any thread."""
    def __init__(self, engine, idle_s=0.02):
        self.engine, self.idle_s = engine, idle_s
        self.inbox, self.handles = queue.Queue(), {}
        self._lock, self._next_id = threading.Lock(), 0
        self._stop, self.error = threading.Event(), None
        self.thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.inbox.put(None)
        self.thread.join(timeout=5.0)

    def submit(self, prompt_ids, params):
        """Queue a request; returns a handle whose events the caller reads. The arrival time is taken here, so queueing delay counts in TTFT."""
        with self._lock:
            rid, self._next_id = self._next_id, self._next_id + 1
        h = _Handle(rid)
        if self.error is not None:
            h.events.put(("error", f"engine stopped: {self.error}", 500))
        else:
            self.inbox.put(("add", h, prompt_ids, params, self.engine.clock()))
        return h

    def abort(self, rid):
        self.inbox.put(("abort", rid))

    def snapshot(self):
        e = self.engine
        return dict(running=len(e.running), waiting=len(e.waiting), counters=dict(e.counters))

    # ---- the engine thread ------------------------------------------------------------------------------------------

    def _run(self):
        e = self.engine
        try:
            while not self._stop.is_set():
                busy = e.has_unfinished()
                try:
                    cmd = self.inbox.get(timeout=0.0 if busy else self.idle_s)
                except queue.Empty:
                    cmd = None
                while cmd is not None:
                    self._command(cmd)
                    try:
                        cmd = self.inbox.get_nowait()
                    except queue.Empty:
                        cmd = None
                if self._stop.is_set():
                    break
                if e.has_unfinished():
                    for o in e.step():
                        h = self.handles.get(o.request_id)
                        if h is None:
                            continue
                        h.events.put(("tokens", o.new_tokens))
                        if o.finished:
                            h.events.put(("done", e.pop_output(o.request_id)))
                            del self.handles[o.request_id]
        except Exception as exc:                 # a bug in the engine must not leave clients waiting forever
            self.error = f"{type(exc).__name__}: {exc}"
            for h in self.handles.values():
                h.events.put(("error", f"engine failed: {self.error}", 500))
            self.handles.clear()
            while True:
                try:
                    cmd = self.inbox.get_nowait()
                except queue.Empty:
                    break
                if cmd is not None and cmd[0] == "add":
                    cmd[1].events.put(("error", f"engine failed: {self.error}", 500))

    def _command(self, cmd):
        if cmd is None:
            return
        e = self.engine
        if cmd[0] == "add":
            _, h, prompt_ids, params, arrival = cmd
            try:
                e.add_request(prompt_ids, params, request_id=h.rid, arrival_s=arrival)
            except ValueError as err:
                h.events.put(("error", str(err), 400))
                return
            self.handles[h.rid] = h
        elif cmd[0] == "abort":
            h = self.handles.pop(cmd[1], None)
            if h is not None and e.abort(cmd[1]):
                h.events.put(("done", e.pop_output(cmd[1])))


class _TextStreamer:
    """Turns a growing list of token ids into text deltas, holding back a trailing partial character."""
    def __init__(self, tokenizer):
        self.tok, self.ids, self.sent = tokenizer, [], ""

    def push(self, ids, final=False):
        self.ids += ids
        text = self.tok.decode(self.ids)
        if not final and text.endswith("�"):
            return ""
        delta, self.sent = text[len(self.sent):], text
        return delta


def _params_from(body):
    """SamplingParams from a request body; raises ValueError for anything malformed."""
    try:
        return SamplingParams(max_new=int(body.get("max_tokens", 64)), temperature=float(body.get("temperature", 0.0)),
                              top_p=float(body.get("top_p", 1.0)), top_k=int(body.get("top_k", 0)),
                              stop_ids=[int(t) for t in body.get("stop_token_ids", [])],
                              seed=None if body.get("seed") is None else int(body["seed"]))
    except (AssertionError, TypeError, ValueError) as err:
        raise ValueError(f"bad sampling parameters: {err}")


class MiniServer:
    """The HTTP server: `MiniServer(engine, tokenizer).start()` serves on a background thread; port 0 picks a free port (see .port)."""
    def __init__(self, engine, tokenizer=None, host="127.0.0.1", port=8000):
        self.engine_thread, self.tokenizer = EngineThread(engine), tokenizer
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self.httpd.daemon_threads = True
        self.host, self.port = host, self.httpd.server_address[1]
        self._http_thread = threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True)

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        self.engine_thread.start()
        self._http_thread.start()
        return self

    def serve_forever(self):
        self.engine_thread.start()
        try:
            self.httpd.serve_forever()
        finally:
            self.engine_thread.stop()

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine_thread.stop()


def _make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        # ---- helpers ----
        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code, message):
            self._json(code, {"error": {"message": message}})

        # ---- routes ----
        def do_GET(self):
            if self.path == "/health":
                self._json(200, dict(status="ok" if server.engine_thread.error is None else "failed", **{k: v for k, v in server.engine_thread.snapshot().items() if k != "counters"}))
            elif self.path == "/v1/stats":
                self._json(200, server.engine_thread.snapshot())
            else:
                self._error(404, f"no such path: {self.path}")

        def do_POST(self):
            if self.path != "/v1/completions":
                self._error(404, f"no such path: {self.path}")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
                ids = self._prompt_ids(body)
                params = _params_from(body)
            except (ValueError, TypeError, json.JSONDecodeError) as err:
                self._error(400, str(err))
                return
            handle = server.engine_thread.submit(ids, params)
            first = handle.events.get()
            if first[0] == "error":
                self._error(first[2], first[1])
                return
            if body.get("stream"):
                self._stream(handle, first, len(ids))
            else:
                self._complete(handle, first, len(ids))

        def _prompt_ids(self, body):
            ids = body.get("prompt_token_ids", body.get("prompt") if isinstance(body.get("prompt"), list) else None)    # OpenAI style: a prompt may be a list of token ids
            if ids is not None:
                if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
                    raise ValueError("prompt_token_ids must be a list of integers")
                return ids
            if isinstance(body.get("prompt"), str):
                if server.tokenizer is None:
                    raise ValueError("this server has no tokenizer: send prompt_token_ids")
                return list(server.tokenizer.encode(body["prompt"]))
            raise ValueError("send either prompt (text) or prompt_token_ids")

        def _events(self, handle, first):
            """The events of a request, starting with the one already read."""
            ev = first
            while True:
                yield ev
                if ev[0] in ("done", "error"):
                    return
                ev = handle.events.get()

        def _choice(self, tokens, text, reason):
            c = dict(index=0, tokens=tokens, finish_reason=reason)
            if text is not None:
                c["text"] = text
            return c

        def _complete(self, handle, first, prompt_len):
            out = None
            for ev in self._events(handle, first):
                if ev[0] == "error":
                    self._error(ev[2], ev[1])
                    return
                if ev[0] == "done":
                    out = ev[1]
            tok = server.tokenizer
            self._json(200, dict(id=f"cmpl-{handle.rid}", object="text_completion",
                                 choices=[self._choice(out.tokens, None if tok is None else tok.decode(out.tokens), out.finish_reason)],
                                 usage=dict(prompt_tokens=prompt_len, completion_tokens=len(out.tokens)),
                                 timing=dict(ttft_s=out.ttft_s, e2e_s=out.e2e_s)))

        def _stream(self, handle, first, prompt_len):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            streamer = None if server.tokenizer is None else _TextStreamer(server.tokenizer)

            def send(obj):
                self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
                self.wfile.flush()

            try:
                for ev in self._events(handle, first):
                    if ev[0] == "tokens":
                        text = None if streamer is None else streamer.push(ev[1])
                        send(dict(id=f"cmpl-{handle.rid}", object="text_completion.chunk", choices=[self._choice(ev[1], text, None)]))
                    elif ev[0] == "error":
                        send(dict(error=dict(message=ev[1])))
                    else:
                        out = ev[1]
                        tail = None if streamer is None else streamer.push([], final=True)
                        send(dict(id=f"cmpl-{handle.rid}", object="text_completion.chunk", choices=[self._choice([], tail, out.finish_reason)],
                                  usage=dict(prompt_tokens=prompt_len, completion_tokens=len(out.tokens)), timing=dict(ttft_s=out.ttft_s, e2e_s=out.e2e_s)))
                send(b"[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                server.engine_thread.abort(handle.rid)            # the client went away: free its blocks

    return Handler


def make_parser():
    ap = argparse.ArgumentParser(description="Serve a Qwen2 model with the minivllm engine")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--int8", action="store_true", help="weight-only int8 (Stage 8)")
    ap.add_argument("--num-blocks", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=2048, help="longest prompt + output a request may have (sizes the RoPE table and the warm-up)")
    ap.add_argument("--graph", action="store_true", help="decode steps as CUDA graphs per batch/length bucket")
    ap.add_argument("--speculate-k", type=int, default=0, help="lookup speculation when one greedy request runs (0: off)")
    ap.add_argument("--warmup", action="store_true", help="capture every graph bucket up to max-model-len before serving, as vLLM does at start-up")
    ap.add_argument("--gqa", default="expand", choices=["expand", "fold"], help="decode attention: copy the keys and values to every query head, or fold the query heads (Stage 13)")
    ap.add_argument("--bucket-scheme", default="pow2", choices=["pow2", "fine"], help="graph buckets: powers of two, or sizes at most a quarter apart (Stage 13)")
    ap.add_argument("--graph-max-batch", type=int, default=None, help="with --graph, decode steps with more running requests than this run eagerly (Stage 13)")
    ap.add_argument("--attention", default="gather", choices=["gather", "paged", "paged_grouped", "paged_torch"], help="decode attention: gather the blocks into dense tensors, or read them in place with a Triton kernel (Stage 14); paged_torch is the same algorithm in torch, a CPU stand-in")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def engine_flags(warmup=True, **config):
    """
    The command-line flags that make `python -m minivllm.server` build the engine these EngineConfig fields describe, so a benchmark can run
    over HTTP exactly the configuration it measured in process. Only the fields the flags can express are accepted (anything else raises),
    and `pad_head` must equal `graph_decode`, which is how the server sets it.
    """
    allowed = {"graph_decode", "pad_head", "gqa", "bucket_scheme", "graph_max_batch", "speculate_k", "attention"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"the server has no flag for {sorted(unknown)}")
    graph = config.get("graph_decode", False)
    if config.get("pad_head", graph) != graph:
        raise ValueError("the server pads the head exactly when it runs graphs: pad_head must equal graph_decode")
    flags = ["--graph"] + (["--warmup"] if warmup else []) if graph else []
    flags += ["--gqa", config.get("gqa", "expand"), "--bucket-scheme", config.get("bucket_scheme", "pow2")]
    if config.get("attention", "gather") != "gather":
        flags += ["--attention", config["attention"]]
    if config.get("graph_max_batch") is not None:
        flags += ["--graph-max-batch", str(config["graph_max_batch"])]
    if config.get("speculate_k"):
        flags += ["--speculate-k", str(config["speculate_k"])]
    return flags


def main(argv=None):
    args = make_parser().parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model import MiniQwen
    from .quant import quantize_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
    model = MiniQwen(hf, max_pos=args.max_model_len)
    if args.int8:
        model = quantize_model(model)
    engine = LLMEngine(model, EngineConfig(num_blocks=args.num_blocks, block_size=args.block_size, max_seqs=args.max_seqs, graph_decode=args.graph,
                                           speculate_k=args.speculate_k, pad_head=args.graph, gqa=args.gqa, bucket_scheme=args.bucket_scheme,
                                           graph_max_batch=args.graph_max_batch, attention=args.attention))
    engine.generate([[1, 2, 3, 4]], SamplingParams(max_new=4))              # first use of the GPU kernels, before the first client
    if args.warmup and engine.graphs is not None:
        engine.graphs.warmup(max_blocks=engine.cache.blocks_for(args.max_model_len), verify_k=args.speculate_k)
    srv = MiniServer(engine, AutoTokenizer.from_pretrained(args.model), args.host, args.port)
    print(f"serving {args.model} on {srv.url} ({device}, graph={args.graph}, speculate_k={args.speculate_k}, gqa={args.gqa}, "
          f"buckets={args.bucket_scheme}, graph_max_batch={args.graph_max_batch}, attention={args.attention})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
