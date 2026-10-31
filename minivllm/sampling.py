"""
Sampling for the engine: greedy, temperature, top-k and top-p (nucleus), per request.

Everything before this module was greedy decoding, which is what makes a speculative or batched run comparable to plain decoding
token for token. A server also has to sample, and the one property worth guarding is that a request's tokens do not depend on who else
is in the batch: each request owns a `torch.Generator` seeded from its own seed, so the same request with the same seed produces the
same tokens alone or in a crowd (up to the model's own numerical noise).

The order is the usual one: scale the logits by 1 / temperature, drop everything outside the top-k, drop everything outside the
smallest set of tokens whose probability reaches top-p, renormalize and draw. Temperature 0 is greedy (an argmax, no randomness).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SamplingParams:
    """
    max_new:     the most tokens to generate (the first one comes from the prompt's last position).
    temperature: 0 means greedy decoding; above 0 the logits are divided by it before sampling.
    top_p:       keep the smallest set of tokens whose probability reaches this (1.0 keeps everything).
    top_k:       keep the k most likely tokens (0 keeps everything; ties at the k-th logit are all kept).
    stop_ids:    token ids that end the request; the stop token is part of the output.
    seed:        seed of the request's own generator; None lets the engine derive one from its seed and the request id.
    """
    max_new: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop_ids: Tuple[int, ...] = ()
    seed: Optional[int] = None

    def __post_init__(self):
        assert self.max_new >= 1, "max_new must be at least 1"
        assert self.temperature >= 0.0, "temperature cannot be negative"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k >= 0, "top_k cannot be negative"
        self.stop_ids = tuple(int(t) for t in self.stop_ids)

    @property
    def greedy(self):
        return self.temperature == 0.0


def filter_logits(logits, top_k=0, top_p=1.0):
    """Copy of a 1-D logits vector with every token outside the top-k and the top-p set at -inf."""
    out = logits.clone()
    if top_k and top_k < out.numel():
        kth = torch.topk(out, top_k).values[-1]
        out[out < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = torch.sort(out, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        before = torch.cumsum(probs, dim=-1) - probs               # probability mass ahead of each token: the first token always stays
        out[order[before >= top_p]] = float("-inf")
    return out


def sample_token(logits, params, generator=None):
    """One token id from a 1-D logits vector under `params` (greedy if the temperature is 0)."""
    if params.greedy:
        return int(logits.argmax(-1))
    x = filter_logits(logits.float() / params.temperature, params.top_k, params.top_p)
    return int(torch.multinomial(torch.softmax(x, dim=-1), 1, generator=generator))


def sample_rows(logits, params_list, generators=None):
    """
    One token id per row of a [n, vocab] logits matrix, each row under its own parameters and generator. Greedy rows are one argmax for
    the whole batch (a single wait for the GPU); only the rows that sample are handled one by one.
    """
    out = logits.argmax(-1).tolist()
    for i, p in enumerate(params_list):
        if not p.greedy:
            out[i] = sample_token(logits[i], p, None if generators is None else generators[i])
    return out


def finish_reason(token, n_generated, params):
    """'stop' if the token is a stop id, 'length' if max_new tokens have been generated, else None."""
    if token in params.stop_ids:
        return "stop"
    if n_generated >= params.max_new:
        return "length"
    return None
