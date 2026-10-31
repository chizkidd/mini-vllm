import torch
import torch.nn.functional as F

from .quant import QWeight, qlinear

def linear(x, w, b=None):
    """F.linear, or the int8 weight-only product (Stage 8) when w is a QWeight. The fp16 path is F.linear itself."""
    if isinstance(w, QWeight):
        return qlinear(x, w, b)
    return F.linear(x, w, b)

def rms_norm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

GQA_MODES = ("expand", "fold")


def gqa_attention(q, k, v, attn_mask, n_rep, fold=False):
    """
    Attention with grouped-query heads. q: [B, n_heads, T, hd]; k, v: [B, n_kv, L, hd], where query head h uses key-value head h // n_rep.
    attn_mask: bool [B, 1, T, L] (True = may attend), or None, which means causal when T > 1 and nothing masked when T == 1.

    fold=False (the model as it was through Stage 12): repeat k and v n_rep times along the head axis, so attention sees as many key-value
        heads as query heads. The copy is n_rep times the size of the keys and values, written and then read again, every layer and every step.
    fold=True: leave k and v alone and move the n_rep query heads that share a key-value head into the query rows. q becomes
        [B, n_kv, n_rep * T, hd] (row r * T + t is query head r of the group at position t), the mask is repeated along its row axis to match,
        and the output is viewed back. The same arithmetic with no copy. A causal call without a mask (a prefill) cannot be folded, because
        the causal shortcut treats row i as position i, so it falls back to expanding.
    """
    B, H, T, hd = q.shape
    causal = attn_mask is None and T > 1
    if n_rep == 1:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
    if fold and not causal:
        rows = q.reshape(B, k.shape[1], n_rep * T, hd)
        mask = None if attn_mask is None else attn_mask.repeat(1, 1, n_rep, 1)
        return F.scaled_dot_product_attention(rows, k, v, attn_mask=mask).reshape(B, H, T, hd)
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)


class MiniQwen:
    def __init__(self, hf_model, max_pos=2048):
        cfg = hf_model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv
        self.eps = cfg.rms_norm_eps

        m = hf_model.model
        self.embed = m.embed_tokens.weight
        self.norm_w = m.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.device, self.dtype = self.embed.device, self.embed.dtype
        self.layers = []
        for l in m.layers:
            a, mlp = l.self_attn, l.mlp
            self.layers.append(dict(
                ln1=l.input_layernorm.weight, ln2=l.post_attention_layernorm.weight,
                q_w=a.q_proj.weight, q_b=a.q_proj.bias,
                k_w=a.k_proj.weight, k_b=a.k_proj.bias,
                v_w=a.v_proj.weight, v_b=a.v_proj.bias,
                o_w=a.o_proj.weight,
                gate=mlp.gate_proj.weight, up=mlp.up_proj.weight, down=mlp.down_proj.weight,
            ))

        theta = getattr(cfg, "rope_theta", None)
        if theta is None:
            theta = cfg.rope_parameters["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=self.device).float() / self.hd))
        t = torch.arange(max_pos, device=self.device).float()
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.cos, self.sin = emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def head(self, x, pad=False):
        """The output projection. With `pad` and exactly one row of x, an fp16 head is computed on two copies of the row and the first is kept.
        Under CUDA-graph capture cuBLAS picks a slow kernel for a one-row fp16 product (Stage 10 measured the head at about twice its
        time with two rows, and the head is a large share of a small model's step), so a graph decoder pads and an fp16 baseline is not
        handicapped against an int8 model, whose kernel has no such cliff. The padded product is a different cuBLAS kernel and so can differ
        from the one-row result in the last fp16 bit."""
        if pad and not isinstance(self.lm_head, QWeight) and x.numel() == x.shape[-1]:
            flat = x.reshape(1, -1)
            return linear(torch.cat([flat, flat]), self.lm_head)[:1].reshape(*x.shape[:-1], -1)
        return linear(x, self.lm_head)

    @torch.no_grad()
    def forward(self, ids, cache=None, positions=None, attn_mask=None, all_logits=False, pad_head=False, gqa="expand"):
        """
        ids: [B, T]. Returns logits for the LAST position only: [B, vocab].
        Batching of sequences with different lengths (Stage 3) adds two optional inputs:
          positions: [B, T] RoPE position of every token (default: consecutive from the cache length).
          attn_mask: bool [B, 1, T, L], True = this query may attend to this key, where L is the
                     number of keys after the cache update (default: causal, no padding).
        Speculative decoding (Stage 6) adds all_logits: return the logits of every position, [B, T, vocab], so
        one pass can check several drafted tokens. Feeding T > 1 tokens on top of a non-empty cache (the checking
        pass) needs an explicit attn_mask, because the default causal mask assumes an empty cache.
        Stage 13 adds gqa: "expand" (the default) copies the keys and values to as many heads as the queries have, "fold" moves the
        queries that share a key-value head into the query rows instead and copies nothing (see gqa_attention).
        """
        assert gqa in GQA_MODES, f"gqa must be one of {GQA_MODES}, got {gqa!r}"
        B, T = ids.shape
        pos0 = cache.length if cache is not None else 0
        assert not (cache is not None and T > 1 and pos0 > 0 and attn_mask is None), \
            "several tokens on a non-empty cache need an explicit attn_mask"
        if positions is None:
            assert pos0 + T <= self.cos.shape[0], "sequence longer than RoPE table"
            cos = self.cos[pos0:pos0 + T][None, None]
            sin = self.sin[pos0:pos0 + T][None, None]
        else:
            cos = self.cos[positions][:, None]   # [B, 1, T, head_dim]
            sin = self.sin[positions][:, None]

        x = F.embedding(ids, self.embed)
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L["ln1"], self.eps)
            q = linear(h, L["q_w"], L["q_b"]).view(B, T, self.n_heads, self.hd).transpose(1, 2)
            k = linear(h, L["k_w"], L["k_b"]).view(B, T, self.n_kv, self.hd).transpose(1, 2)
            v = linear(h, L["v_w"], L["v_b"]).view(B, T, self.n_kv, self.hd).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if cache is not None and T == 1 and getattr(cache, "attention", "gather") != "gather":
                cache.write(i, k, v)                                        # Stage 14: read the blocks where they are, no gather
                a = cache.attend(i, q, positions)
            else:
                if cache is not None:
                    k, v = cache.update(i, k, v)
                a = gqa_attention(q, k, v, attn_mask, self.n_rep, fold=(gqa == "fold"))
            a = a.transpose(1, 2).reshape(B, T, -1)
            x = x + linear(a, L["o_w"])
            h = rms_norm(x, L["ln2"], self.eps)
            x = x + linear(F.silu(linear(h, L["gate"])) * linear(h, L["up"]), L["down"])
        if cache is not None:
            cache.advance(T)
        x = rms_norm(x if all_logits else x[:, -1], self.norm_w, self.eps)
        return self.head(x, pad_head)
