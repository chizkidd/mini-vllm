"""
Attention that reads the paged KV cache where it is (Stage 14, exercise 12.4 of Stage 12).

A decode step over the paged cache has so far GATHERED: the blocks of every row are copied, through the block table, into one dense tensor of
keys and one of values of a fixed width (the bucket), and ordinary attention then runs on the copies (`StaticPagedView.update`). Stage 13
showed that those copies, not the attention arithmetic, were most of the step, and folding the query heads removed the expansion but not the
gather. The gather copies every row to the width of the longest, padding included, so its cost grows with the batch times the bucket.

The kernels here need no gather. For one query token per row (a decode step) they walk the row's OWN blocks through its block table, load each
block of keys and values straight from the pool and combine the blocks with an online softmax (a running maximum, a running denominator and a
running weighted sum, rescaled when the maximum grows), so nothing is copied, padding blocks are never touched, and a row costs what its own
length costs. The number of blocks to visit is read from the device (`lens`), so the kernels run inside a CUDA graph.

  paged_attention_reference   gather, then softmax in fp32: the oracle the tests compare with (any device).
  paged_attention_blockwise   the algorithm of the kernels in plain torch, one block at a time: readable, and the CPU stand-in.
  paged_attention             the entry point: kernel="simple" (one program per row and query head), "grouped" (one program per row and
                              key-value head, every block loaded once for all the query heads that share it, the scores and the weighted
                              sum on the tensor cores as tl.dot as flash-decoding kernels do), or "torch" (blockwise).

Shapes: q [B, H, hd] (one token per row), k_pool and v_pool [num_blocks, n_kv, block_size, hd] (one layer of PagedCache.k / .v), tables
[B, NB] block ids (any value past a row's last block, such as the engine's NULL block, is never read), lens [B] the number of valid positions
of each row INCLUDING the token whose key was just written (at least 1). Returns [B, H, hd].

Triton is optional. Without it HAVE_TRITON is False and the "simple" and "grouped" kernels raise; "torch" always works. With
TRITON_INTERPRET=1 (set before the first import) the kernels run on the CPU in Triton's interpreter, which is how the tests check them.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                       # no Triton on this machine
    triton = tl = None
    HAVE_TRITON = False

KERNELS = ("simple", "grouped", "torch")
KERNEL_OF = {"paged": "simple", "paged_grouped": "grouped", "paged_torch": "torch"}      # EngineConfig.attention -> kernel
ATTENTION_MODES = ("gather", *KERNEL_OF)


def attend_in_place(pool_k, pool_v, table, q, positions, mode):
    """The decode attention of a paged view in an in-place mode: q [B, H, 1, hd], positions [B, 1] (the position just written), table [B, NB]. Returns [B, H, 1, hd]."""
    lens = positions[:, 0] + 1
    return paged_attention(q[:, :, 0], pool_k, pool_v, table, lens, kernel=KERNEL_OF[mode])[:, :, None]


def _check(q, k_pool, v_pool, tables, lens):
    B, H, hd = q.shape
    n_kv = k_pool.shape[1]
    assert H % n_kv == 0, "query heads must be a multiple of key-value heads"
    assert k_pool.shape == v_pool.shape and k_pool.shape[3] == hd, "pools must be [num_blocks, n_kv, block_size, hd] with the query's head size"
    assert tables.shape[0] == B and lens.shape == (B,), "one block table and one length per row"
    return B, H, hd, n_kv, H // n_kv, k_pool.shape[2]


# ---- the oracle and the readable version -------------------------------------------------------------------------------------

def paged_attention_reference(q, k_pool, v_pool, tables, lens):
    """Gather every row's blocks, mask the positions past its length, softmax in fp32. What the kernels must equal."""
    B, H, hd, n_kv, n_rep, bs = _check(q, k_pool, v_pool, tables, lens)
    NB = tables.shape[1]
    k = k_pool[tables].permute(0, 2, 1, 3, 4).reshape(B, n_kv, NB * bs, hd).float()
    v = v_pool[tables].permute(0, 2, 1, 3, 4).reshape(B, n_kv, NB * bs, hd).float()
    k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
    s = torch.einsum("bhd,bhld->bhl", q.float(), k) / hd ** 0.5
    valid = torch.arange(NB * bs, device=q.device)[None, None, :] < lens[:, None, None]
    w = torch.softmax(s.masked_fill(~valid, float("-inf")), dim=-1)
    return torch.einsum("bhl,bhld->bhd", w, v).to(q.dtype)


def paged_attention_blockwise(q, k_pool, v_pool, tables, lens):
    """The kernels' algorithm in torch: for each row visit only its own blocks, keep a running maximum, denominator and weighted sum."""
    B, H, hd, n_kv, n_rep, bs = _check(q, k_pool, v_pool, tables, lens)
    out = torch.empty_like(q)
    lens_l = lens.tolist()
    for b in range(B):
        n = lens_l[b]
        qb = q[b].float().reshape(n_kv, n_rep, hd)
        m = torch.full((n_kv, n_rep), float("-inf"), device=q.device)
        denom = torch.zeros(n_kv, n_rep, device=q.device)
        acc = torch.zeros(n_kv, n_rep, hd, device=q.device)
        for j in range(-(-n // bs)):
            blk = int(tables[b, j])
            cnt = min(bs, n - j * bs)
            k, v = k_pool[blk, :, :cnt].float(), v_pool[blk, :, :cnt].float()               # [n_kv, cnt, hd], read in place
            s = torch.einsum("grd,gcd->grc", qb, k) / hd ** 0.5
            new_m = torch.maximum(m, s.max(-1).values)
            scale = torch.exp(m - new_m)
            p = torch.exp(s - new_m[..., None])
            denom = denom * scale + p.sum(-1)
            acc = acc * scale[..., None] + torch.einsum("grc,gcd->grd", p, v)
            m = new_m
        out[b] = (acc / denom[..., None]).reshape(H, hd).to(q.dtype)
    return out


# ---- the kernels ----------------------------------------------------------------------------------------------------------------

if HAVE_TRITON:
    @triton.jit
    def _paged_simple_kernel(q_ptr, k_ptr, v_ptr, tbl_ptr, len_ptr, o_ptr, scale,
                             stride_qb, stride_qh, stride_kb, stride_kh, stride_ks, stride_tb, stride_ob, stride_oh,
                             N_REP: tl.constexpr, BS: tl.constexpr, BSP: tl.constexpr, HD: tl.constexpr):
        # one program per (row, query head): walk the row's own blocks, online softmax in fp32
        b = tl.program_id(0)
        h = tl.program_id(1)
        kvh = h // N_REP
        n = tl.load(len_ptr + b)
        d = tl.arange(0, HD)
        slot = tl.arange(0, BSP)
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d).to(tl.float32) * scale
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((HD,), dtype=tl.float32)
        for j in range(0, (n + BS - 1) // BS):
            blk = tl.load(tbl_ptr + b * stride_tb + j).to(tl.int64)
            valid = (slot < BS) & (j * BS + slot < n)
            off = blk * stride_kb + kvh * stride_kh + slot[:, None] * stride_ks + d[None, :]
            k = tl.load(k_ptr + off, mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1)
            s = tl.where(valid, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            v = tl.load(v_ptr + off, mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new
        tl.store(o_ptr + b * stride_ob + h * stride_oh + d, (acc / l_i).to(o_ptr.dtype.element_ty))

    @triton.jit
    def _paged_grouped_kernel(q_ptr, k_ptr, v_ptr, tbl_ptr, len_ptr, o_ptr, scale,
                              stride_qb, stride_qh, stride_kb, stride_kh, stride_ks, stride_tb, stride_ob, stride_oh,
                              N_REP: tl.constexpr, RP: tl.constexpr, BS: tl.constexpr, BSP: tl.constexpr, HD: tl.constexpr):
        # one program per (row, key-value head): every block is loaded once and serves all the query heads of the group (padded to RP rows
        # for tl.dot); scores and weighted sums on the tensor cores, online softmax per query head
        b = tl.program_id(0)
        g = tl.program_id(1)
        n = tl.load(len_ptr + b)
        d = tl.arange(0, HD)
        r = tl.arange(0, RP)
        slot = tl.arange(0, BSP)
        rmask = r < N_REP
        q = tl.load(q_ptr + b * stride_qb + (g * N_REP + r)[:, None] * stride_qh + d[None, :], mask=rmask[:, None], other=0.0)
        m_i = tl.zeros((RP,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((RP,), dtype=tl.float32)
        acc = tl.zeros((RP, HD), dtype=tl.float32)
        for j in range(0, (n + BS - 1) // BS):
            blk = tl.load(tbl_ptr + b * stride_tb + j).to(tl.int64)
            valid = (slot < BS) & (j * BS + slot < n)
            off = blk * stride_kb + g * stride_kh + slot[:, None] * stride_ks + d[None, :]
            k = tl.load(k_ptr + off, mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(valid[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + off, mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        o = (acc / l_i[:, None]).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + b * stride_ob + (g * N_REP + r)[:, None] * stride_oh + d[None, :], o, mask=rmask[:, None])


def _pow2(n, floor=1):
    p = floor
    while p < n:
        p *= 2
    return p


def paged_attention(q, k_pool, v_pool, tables, lens, kernel="simple"):
    """Attention of one token per row over the paged cache, read in place. See the module text for shapes and for the kernels."""
    assert kernel in KERNELS, f"kernel must be one of {KERNELS}, got {kernel!r}"
    B, H, hd, n_kv, n_rep, bs = _check(q, k_pool, v_pool, tables, lens)
    if kernel == "torch":
        return paged_attention_blockwise(q, k_pool, v_pool, tables, lens)
    assert HAVE_TRITON, "this kernel needs Triton (use kernel='torch')"
    assert hd & (hd - 1) == 0 and hd >= 16, "the head size must be a power of two, at least 16"
    out = torch.empty_like(q)
    scale = hd ** -0.5
    strides = (q.stride(0), q.stride(1), k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), tables.stride(0), out.stride(0), out.stride(1))
    if kernel == "simple":
        _paged_simple_kernel[(B, H)](q, k_pool, v_pool, tables, lens, out, scale, *strides, N_REP=n_rep, BS=bs, BSP=_pow2(bs), HD=hd)
    else:
        assert n_rep <= 16, "the grouped kernel serves up to 16 query heads per key-value head"
        _paged_grouped_kernel[(B, n_kv)](q, k_pool, v_pool, tables, lens, out, scale, *strides, N_REP=n_rep, RP=_pow2(n_rep, 16), BS=bs,
                                         BSP=_pow2(bs, 16), HD=hd)
    return out
