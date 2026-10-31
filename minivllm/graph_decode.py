import torch

from .draft import StaticForward, StaticKVCache, _tick


class GraphDecoder:
    """
    Decoding that lives in CUDA graphs (Stage 10): one sequence, a StaticKVCache, and one captured forward pass for every block length
    it will need, so a decode step and a speculative checking pass are each ONE graph launch and no kernel launches happen from
    Python in between. That is what lets a hand-written kernel pay: a Triton kernel launched eagerly costs about as much CPU time as it
    saves (Stage 9), and inside a graph it costs nothing.

      plain(ids, max_new):                       greedy decoding, one captured one-token pass per token
      speculative(ids, max_new, k, proposer):    the same loop as minivllm.speculative.generate_speculative, but the checking pass is a
                                                 captured pass of FIXED length k + 1: the block is [last token] + drafts, padded with
                                                 zeros when the proposer gave fewer than k, and the rows after the real drafts are
                                                 ignored. A round with no draft at all has nothing to check, so it is the ordinary
                                                 one-token pass instead of a block. The output equals plain greedy decoding of the
                                                 same weights up to fp16 near-ties (Stage 6), for any proposer.

    The prompt is prefilled eagerly through the same static cache (a prompt is a big block, so a captured pass would not help). The
    rollback after a rejected draft is only moving the length: see StaticForward. `ks` lists the draft lengths to capture a checking
    pass for; max_len must cover the prompt, max_new and k + 1. causal=False is the negative control (see StaticForward). pad_head (Stage 11) computes a one-row fp16
    output projection on two rows (MiniQwen.head), so an fp16 model's captured one-token pass is not slowed by cuBLAS's one-row kernel.
    """
    def __init__(self, model, max_len, ks=(), capture=True, causal=True, pad_head=False):
        self.model, self.max_len, self.capture, self.causal, self.pad_head = model, max_len, capture, causal, pad_head
        self.device = model.device
        self.cache = StaticKVCache(model.n_layers, model.n_kv, max_len, model.hd, self.device, model.dtype)
        self.step = StaticForward(model, self.cache, 1, all_logits=False, capture=capture, causal=causal, pad_head=pad_head)
        self.verify = {}
        for k in ks:
            self.add_block(k)
        self._cols = torch.arange(max_len, device=self.device)

    def add_block(self, k):
        """Capture the checking pass for drafts of length k (a pass of k + 1 tokens). Do this before timing anything."""
        if k not in self.verify:
            self.verify[k] = StaticForward(self.model, self.cache, k + 1, all_logits=True, capture=self.capture, causal=self.causal, pad_head=self.pad_head)
        return self.verify[k]

    @property
    def mode(self):
        """"graph" if every captured pass is a graph, "static" if any of them fell back to running eagerly."""
        return "graph" if self.step.mode == "graph" and all(v.mode == "graph" for v in self.verify.values()) else "static"

    @property
    def graph_error(self):
        return next((f.graph_error for f in [self.step, *self.verify.values()] if f.graph_error), None)

    @torch.no_grad()
    def _prefill(self, ids):
        """Feed the whole prompt through the static cache (eagerly) and return the logits after its last token, [1, vocab]."""
        n = len(ids)
        assert n >= 1 and n <= self.max_len, "the prompt does not fit in the cache"
        positions = torch.arange(n, device=self.device)
        self.cache.set_write(positions)
        mask = (self._cols[None, :] <= positions[:, None])[None, None]
        logits = self.model.forward(torch.tensor([list(ids)], device=self.device), self.cache, positions=positions[None, :], attn_mask=mask)
        self.cache.length = n
        return logits

    @torch.no_grad()
    def plain(self, ids, max_new):
        """Greedy decoding. Returns (the max_new generated tokens, {"ttft": seconds, "itl": [seconds per decoded token]})."""
        n = len(ids)
        assert n + max_new <= self.max_len, "the cache does not have room for the prompt and the generated tokens"
        t0 = _tick()
        tok = self._prefill(ids).argmax(-1, keepdim=True)                       # [1, 1]
        t1 = _tick()
        out = [tok.clone()]
        step = self.step
        step.ids.copy_(tok)
        step.pos.fill_(n)
        for _ in range(max_new - 1):
            nxt = step.replay().argmax(-1, keepdim=True)
            out.append(nxt.clone())
            step.ids.copy_(nxt)
            step.pos.add_(1)
        tokens = torch.cat(out, dim=1)[0].tolist()                              # one sync for the whole generation
        t2 = _tick()
        return tokens, {"ttft": t1 - t0, "itl": [(t2 - t1) / max(1, max_new - 1)] * (max_new - 1)}

    @torch.no_grad()
    def speculative(self, ids, max_new, k, proposer):
        """
        Speculative greedy decoding with drafts of up to k tokens from `proposer(context, k)` (PromptLookup, DraftModel, ...).
        Returns (the max_new generated tokens, {"ttft": seconds, "steps": [one dict per round]}) in the format of generate_speculative:
        each round records how many tokens were drafted and accepted and the seconds of the checking pass (`step_s`).
        """
        n = len(ids)
        assert k >= 1 and n + max_new + k <= self.max_len, "the cache does not have room for the prompt, the generated tokens and a block"
        fwd = self.add_block(k)
        T = k + 1
        t0 = _tick()
        context = list(ids) + [int(self._prefill(ids).argmax(-1))]
        ttft = _tick() - t0
        n_cache, steps = n, []                                                    # keys of the prompt are cached; context[-1] is not fed yet
        while len(context) - n < max_new:
            drafts = [int(d) for d in proposer(context, k)][:k]
            t1 = _tick()
            if drafts:
                block = [context[-1]] + drafts + [0] * (k - len(drafts))
                choices = fwd.run(block, n_cache)[0].argmax(-1).tolist()         # the model's own greedy choice after every fed token
            else:                                                                 # nothing to check: an ordinary one-token step
                choices = [int(self.step.run([context[-1]], n_cache).argmax(-1))]
            step_s = _tick() - t1
            a = 0
            while a < len(drafts) and drafts[a] == choices[a]:
                a += 1
            context += choices[:a + 1]                                            # the a accepted drafts, then one token of its own
            n_cache += a + 1
            steps.append({"drafted": len(drafts), "accepted": a, "step_s": step_s})
        return context[n:n + max_new], {"ttft": ttft, "steps": steps}
