"""
A fused weight-only int8 matmul in Triton (Stage 9): y = (x @ q.T) * scale, with q an int8 [N, K] matrix and one scale per
output row. The weights are read from memory as bytes, converted to the activation dtype in registers and multiplied there,
so the only traffic is one byte per weight. Stage 8 found that every way PyTorch offers to multiply by an int8 weight
(an eager dequantize, the same under torch.compile, torch._weight_int8pack_mm) is slower than the fp16 layer on a T4,
because none of them reads the weights as bytes at the rate fp16 reads its own. This kernel is the fix being tested.

Three kernels and a fallback, because a decode step, a checking block and a prompt ask for different things:
  gemv1:  one token (M == 1). A tile of BLOCK_N weight rows by BLOCK_K weights is multiplied by the matching slice of x on the
          FMA units and the products are kept in a [BLOCK_N, BLOCK_K] accumulator, reduced once at the end.
  gemvm:  up to 8 rows of x (Stage 10: a checking block of k + 1 tokens, a small decode batch). The same streaming loop as gemv1,
          but each weight tile is multiplied by every row of x while it is in registers, with a [MAX_M, BLOCK_N, BLOCK_K]
          accumulator, so the weights are still read once and a block of tokens costs about what one token costs as long as the FMA
          units have room (they do for a handful of rows). No tensor cores and no shared-memory staging, which is what made the dot
          kernel slow on a T4 in Stage 9.
  gemvr, gemvs (Stage 11): the same job as gemvm, written so that nothing goes through shared memory. gemvm keeps one 3-D accumulator,
          and the layout changes Triton needs around it (a broadcast of x, a reduction over an axis) become shared-memory stores, loads
          and barriers inside the loop, which Stage 10 found to be where its time went. These two have one accumulator per row, named
          and unrolled for exactly ROWS rows (1 to 8), and the x row is loaded as a pointer tile whose row stride is zero, so every
          thread reads x straight from the cache. gemvr keeps a [BLOCK_N, BLOCK_K] tile per row and reduces it once at the end (the
          fewest instructions per weight, the most registers); gemvs sums over K inside the loop into a [BLOCK_N] vector per row (more
          instructions, far fewer registers). Which one wins for a shape is a measurement, so both are candidates of the tuning run.
  dot:    M tokens through the tensor cores (Stage 9). The x rows are tiled by BLOCK_M (16 for up to 16 rows, as tl.dot needs, and
          64 beyond), the int8 tile is converted to the activation dtype and a tensor-core dot accumulates in fp32.
  dequant: not a kernel: the dequantize path of Stage 8 (a fp16 copy of the matrix, then cuBLAS), the fallback for blocks too big
          for gemvm, such as a prompt.
Both accumulate in fp32 and apply the scale to the fp32 result, so unlike the eager dequantize there is no fp16 overflow to
guard against (no input prescale) and the scale never touches the weights. Block sizes come from an explicit table filled by a
tuning run (bench.metrics.tune_int8_kernel) and not from Triton's autotuner, which times candidate launches and so cannot run
inside CUDA-graph capture; an untuned shape falls back to a heuristic.

Triton is optional. Without it HAVE_TRITON is False and int8_matmul raises; the rest of the package never imports this module
unless a QWeight has path "triton". With TRITON_INTERPRET=1 (set before the first import) the kernels run on the CPU in Triton's
interpreter, which is how the tests check them without a GPU.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                       # no Triton on this machine
    triton = tl = None
    HAVE_TRITON = False


if HAVE_TRITON:
    @triton.jit
    def _gemv1_kernel(x_ptr, q_ptr, s_ptr, y_ptr, N, K, stride_qn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K
            w = tl.load(q_ptr + offs_n[:, None] * stride_qn + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0)
            x = tl.load(x_ptr + k, mask=k_mask, other=0.0)
            acc += w.to(tl.float32) * x.to(tl.float32)[None, :]
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        y = tl.sum(acc, axis=1) * s
        tl.store(y_ptr + offs_n, y.to(y_ptr.dtype.element_ty), mask=n_mask)

    @triton.jit
    def _gemvm_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_qn, stride_ym,
                      MAX_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, MAX_M)
        n_mask = offs_n < N
        m_mask = offs_m < M
        acc = tl.zeros((MAX_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K
            w = tl.load(q_ptr + offs_n[:, None] * stride_qn + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
            x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            acc += w[None, :, :] * x[:, None, :]
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        y = tl.sum(acc, axis=2) * s[None, :]
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], y.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _load_w(q_ptr, offs_n, k, N, K, stride_qn, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
        """A [BLOCK_N, BLOCK_K] tile of int8 weights as fp32. When N and K are multiples of the block sizes the mask is dropped, which
        lets the compiler issue wide loads (a masked load is read a byte at a time)."""
        ptrs = q_ptr + offs_n[:, None] * stride_qn + k[None, :]
        if EVEN_K:
            if EVEN_N:
                w = tl.load(ptrs)
            else:
                w = tl.load(ptrs, mask=(offs_n < N)[:, None], other=0)
        else:
            w = tl.load(ptrs, mask=(offs_n < N)[:, None] & (k < K)[None, :], other=0)
        return w.to(tl.float32)

    @triton.jit
    def _load_x(x_ptr, row: tl.constexpr, stride_xm, k, zero_n, K, EVEN_K: tl.constexpr):
        """Row `row` of x, as a [BLOCK_N, BLOCK_K] tile whose rows are all the same (the zero in the address). Every thread then reads
        the x values it needs from the cache itself, instead of one copy being broadcast across threads through shared memory."""
        ptrs = x_ptr + row * stride_xm + k[None, :] + zero_n
        if EVEN_K:
            x = tl.load(ptrs)
        else:
            x = tl.load(ptrs, mask=(k < K)[None, :], other=0.0)
        return x.to(tl.float32)

    @triton.jit
    def _gemvr_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_qn, stride_ym,
                      ROWS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        zero_n = offs_n[:, None] * 0
        acc0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 1:
            acc1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 2:
            acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 3:
            acc3 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 4:
            acc4 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 5:
            acc5 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 6:
            acc6 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        if ROWS > 7:
            acc7 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            w = _load_w(q_ptr, offs_n, k, N, K, stride_qn, EVEN_K, EVEN_N)
            acc0 += w * _load_x(x_ptr, 0, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 1:
                acc1 += w * _load_x(x_ptr, 1, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 2:
                acc2 += w * _load_x(x_ptr, 2, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 3:
                acc3 += w * _load_x(x_ptr, 3, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 4:
                acc4 += w * _load_x(x_ptr, 4, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 5:
                acc5 += w * _load_x(x_ptr, 5, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 6:
                acc6 += w * _load_x(x_ptr, 6, stride_xm, k, zero_n, K, EVEN_K)
            if ROWS > 7:
                acc7 += w * _load_x(x_ptr, 7, stride_xm, k, zero_n, K, EVEN_K)
        s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        tl.store(y_ptr + 0 * stride_ym + offs_n, (tl.sum(acc0, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 0))
        if ROWS > 1:
            tl.store(y_ptr + 1 * stride_ym + offs_n, (tl.sum(acc1, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 1))
        if ROWS > 2:
            tl.store(y_ptr + 2 * stride_ym + offs_n, (tl.sum(acc2, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 2))
        if ROWS > 3:
            tl.store(y_ptr + 3 * stride_ym + offs_n, (tl.sum(acc3, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 3))
        if ROWS > 4:
            tl.store(y_ptr + 4 * stride_ym + offs_n, (tl.sum(acc4, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 4))
        if ROWS > 5:
            tl.store(y_ptr + 5 * stride_ym + offs_n, (tl.sum(acc5, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 5))
        if ROWS > 6:
            tl.store(y_ptr + 6 * stride_ym + offs_n, (tl.sum(acc6, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 6))
        if ROWS > 7:
            tl.store(y_ptr + 7 * stride_ym + offs_n, (tl.sum(acc7, axis=1) * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 7))

    @triton.jit
    def _gemvs_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_qn, stride_ym,
                      ROWS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        zero_n = offs_n[:, None] * 0
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 1:
            acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 2:
            acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 3:
            acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 4:
            acc4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 5:
            acc5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 6:
            acc6 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if ROWS > 7:
            acc7 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            w = _load_w(q_ptr, offs_n, k, N, K, stride_qn, EVEN_K, EVEN_N)
            acc0 += tl.sum(w * _load_x(x_ptr, 0, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 1:
                acc1 += tl.sum(w * _load_x(x_ptr, 1, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 2:
                acc2 += tl.sum(w * _load_x(x_ptr, 2, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 3:
                acc3 += tl.sum(w * _load_x(x_ptr, 3, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 4:
                acc4 += tl.sum(w * _load_x(x_ptr, 4, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 5:
                acc5 += tl.sum(w * _load_x(x_ptr, 5, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 6:
                acc6 += tl.sum(w * _load_x(x_ptr, 6, stride_xm, k, zero_n, K, EVEN_K), axis=1)
            if ROWS > 7:
                acc7 += tl.sum(w * _load_x(x_ptr, 7, stride_xm, k, zero_n, K, EVEN_K), axis=1)
        s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        tl.store(y_ptr + 0 * stride_ym + offs_n, (acc0 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 0))
        if ROWS > 1:
            tl.store(y_ptr + 1 * stride_ym + offs_n, (acc1 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 1))
        if ROWS > 2:
            tl.store(y_ptr + 2 * stride_ym + offs_n, (acc2 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 2))
        if ROWS > 3:
            tl.store(y_ptr + 3 * stride_ym + offs_n, (acc3 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 3))
        if ROWS > 4:
            tl.store(y_ptr + 4 * stride_ym + offs_n, (acc4 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 4))
        if ROWS > 5:
            tl.store(y_ptr + 5 * stride_ym + offs_n, (acc5 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 5))
        if ROWS > 6:
            tl.store(y_ptr + 6 * stride_ym + offs_n, (acc6 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 6))
        if ROWS > 7:
            tl.store(y_ptr + 7 * stride_ym + offs_n, (acc7 * s).to(y_ptr.dtype.element_ty), mask=(offs_n < N) & (M > 7))

    @triton.jit
    def _dot_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_qn, stride_ym,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K
            x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(q_ptr + offs_n[None, :] * stride_qn + k[:, None], mask=n_mask[None, :] & k_mask[:, None], other=0)
            acc += tl.dot(x, w.to(x.dtype))
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        y = acc * s[None, :]
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], y.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


VARIANTS = ("gemv1", "gemvm", "gemvr", "gemvs", "dot", "dequant")
ROW_VARIANTS = ("gemvr", "gemvs")      # the Stage 11 multi-row kernels, compiled for an exact number of rows
NUM_SMS = 40            # the T4's multiprocessors; only used to size the default tile so that every SM gets work
MAX_GEMVM_ROWS = 8      # the most rows the multi-row kernels take; more go to the dot kernel or the dequantize path

_CONFIGS = {}           # (N, K, m_bucket) -> config, filled by a tuning run
_LAST = {"kernel": None}    # the compiled kernel of the latest launch, which kernel_info reads

# The candidates a tuning run times (bench.metrics.tune_int8_kernel). Many of them share a compiled kernel across shapes, since
# block sizes are compile-time constants and nothing else about a shape changes the code, so tuning many shapes compiles few kernels.
GEMV1_GRID = [{"variant": "gemv1", "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw} for bn in (2, 4, 8, 16) for bk in (128, 256, 512) for nw in (2, 4, 8)]
GEMVM_GRID = [{"variant": "gemvm", "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw} for bn in (1, 2, 4) for bk in (64, 128, 256) for nw in (2, 4, 8)]
# Stage 11. x_fp32 says that the host converts x to fp32 once before the launch (one small kernel per call), which the compiled loop
# needs: with an fp16 x every use of a value converts it again, and offline (compiled for sm_75, no GPU) that was a fifth to two
# fifths more instructions per weight in the loop. The grids are crossed with the fit rule below, per row count.
GEMVR_GRID = [{"variant": "gemvr", "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "x_fp32": True}
              for bn in (1, 2, 4, 8, 16, 32) for bk in (64, 128, 256) for nw in (1, 2, 4)]
GEMVS_GRID = [{"variant": "gemvs", "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "x_fp32": True}
              for bn in (16, 32, 64, 128) for bk in (16, 32, 64) for nw in (1, 2, 4)]
DOT_GRID = [{"variant": "dot", "BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "num_stages": 2}
            for bn in (16, 32, 64, 128) for bk in (64, 128, 256) for nw in (4, 8)]
DEQUANT = {"variant": "dequant"}


def tile_per_thread(cfg):
    """The weights of one tile that each thread of the program holds at once: BLOCK_N * BLOCK_K / threads."""
    return cfg["BLOCK_N"] * cfg["BLOCK_K"] / (32 * cfg.get("num_warps", 4))


def fits(cfg, rows):
    """
    Whether a gemvr or gemvs configuration is worth timing for `rows` rows. The rule comes from compiling both kernels for the T4
    (sm_75, offline, without a GPU: Triton ships its own ptxas) for every tile in the grids at 2, 3, 5 and 8 rows and reading the
    registers, local memory and instructions of the compiled loop. gemvr holds a [BLOCK_N, BLOCK_K] accumulator per row, so its
    registers grow with rows * tile_per_thread: from about 128 of that product it spills accumulators to local memory, and a tile of
    fewer than 8 per thread counted several times the instructions per weight of the larger tiles at 2 to 5 rows. gemvs keeps only a [BLOCK_N] vector per row, but a tile
    of fewer than 16 per thread makes Triton reduce across threads through shared memory with barriers (the cost of gemvm) and a tile
    of more than 64 (or more than 32 from 6 rows) spills.
    """
    t = tile_per_thread(cfg)
    if cfg["variant"] == "gemvr":
        return t >= 8 and rows * t < 128
    return 16 <= t <= 64 and (rows <= 5 or t <= 32)


def row_grid(variant, rows):
    """The configurations of a multi-row kernel's grid that fit `rows` rows (a fresh list of fresh dicts)."""
    grid = GEMVR_GRID if variant == "gemvr" else GEMVS_GRID
    return [dict(c) for c in grid if fits(c, rows)]


def m_bucket(M):
    """The row counts that share a tuned configuration. Stage 10 shared one for 3 to 4 and one for 5 to 8 rows; the multi-row kernels are
    compiled for an exact number of rows and the best tile changes with it, so each of 1 to 8 (a checking block of k + 1 tokens) has its
    own, then 9 to 16, and a prompt."""
    return M if M <= MAX_GEMVM_ROWS else 16 if M <= 16 else 64


def max_rows(M):
    """The compile-time row count of the gemvm kernel that holds M rows: 2, 4 or 8."""
    return 2 if M <= 2 else 4 if M <= 4 else 8


def default_config(N, K, M):
    """A configuration for a shape that was not tuned. For one token the tile is sized so the grid has at least a couple of programs
    per SM; for 2 to 8 rows the gemvs kernel, sized the same way (it is the one the offline compile shows to be safe at every row
    count: no spills and no barriers); beyond that the dequantize path."""
    if M == 1:
        bn = 16
        while bn > 1 and -(-N // bn) < 2 * NUM_SMS:
            bn //= 2
        return {"variant": "gemv1", "BLOCK_N": bn, "BLOCK_K": 256, "num_warps": 4}
    if M <= MAX_GEMVM_ROWS:
        bn = 64
        while bn > 16 and -(-N // bn) < 2 * NUM_SMS:
            bn //= 2
        return {"variant": "gemvs", "BLOCK_N": bn, "BLOCK_K": 32, "num_warps": 2 if bn == 64 else 1, "x_fp32": True}
    return dict(DEQUANT)


def set_configs(table):
    """Install a tuned table {(N, K, m_bucket): config}; it replaces whatever was there."""
    _CONFIGS.clear()
    _CONFIGS.update(table)


def get_configs():
    return dict(_CONFIGS)


def config_for(N, K, M):
    return _CONFIGS.get((N, K, m_bucket(M))) or default_config(N, K, M)


def int8_matmul(x, q, scale, config=None):
    """
    x [..., K] (fp16 on a GPU; fp32 or fp16 in the interpreter), q int8 [N, K], scale [N] (or one value) -> [..., N] in x's dtype.
    `config` overrides the table, for tuning and tests: {"variant": "gemv1" or "gemvm", "BLOCK_N", "BLOCK_K", "num_warps"},
    {"variant": "dot", "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"} or {"variant": "dequant"}.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not available on this machine, so the 'triton' path cannot run")
    N, K = q.shape
    assert x.shape[-1] == K and q.dtype == torch.int8 and q.is_contiguous()
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    cfg = config or config_for(N, K, M)
    if cfg["variant"] == "dequant":
        from .quant import PRE
        import torch.nn.functional as F
        return (F.linear(x * PRE, q.to(x.dtype)) * (scale.to(x.dtype) * (1.0 / PRE))).reshape(*x.shape[:-1], N)
    s = scale.reshape(-1)
    if s.numel() == 1:
        s = s.expand(N).contiguous()
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if cfg["variant"] == "gemv1":
        assert M == 1, "the gemv1 kernel handles one token; use gemvm, dot or dequant for more"
        grid = (triton.cdiv(N, cfg["BLOCK_N"]),)
        _LAST["kernel"] = _gemv1_kernel[grid](x2, q, s, y, N, K, q.stride(0), BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], num_warps=cfg.get("num_warps", 4))
    elif cfg["variant"] in ("gemvr", "gemvs"):
        assert 1 <= M <= MAX_GEMVM_ROWS, f"the {cfg['variant']} kernel handles at most {MAX_GEMVM_ROWS} rows"
        bn, bk = cfg["BLOCK_N"], cfg["BLOCK_K"]
        xk = x2.float() if (cfg.get("x_fp32") and x2.dtype != torch.float32) else x2     # one conversion here instead of one per use in the loop
        kernel = _gemvr_kernel if cfg["variant"] == "gemvr" else _gemvs_kernel
        _LAST["kernel"] = kernel[(triton.cdiv(N, bn),)](xk, q, s, y, M, N, K, xk.stride(0), q.stride(0), y.stride(0), ROWS=M, BLOCK_N=bn,
                                                        BLOCK_K=bk, EVEN_K=K % bk == 0, EVEN_N=N % bn == 0, num_warps=cfg.get("num_warps", 4))
    elif cfg["variant"] == "gemvm":
        assert 1 <= M <= MAX_GEMVM_ROWS, f"the gemvm kernel handles at most {MAX_GEMVM_ROWS} rows"
        grid = (triton.cdiv(N, cfg["BLOCK_N"]),)
        _LAST["kernel"] = _gemvm_kernel[grid](x2, q, s, y, M, N, K, x2.stride(0), q.stride(0), y.stride(0), MAX_M=max_rows(M), BLOCK_N=cfg["BLOCK_N"],
                                              BLOCK_K=cfg["BLOCK_K"], num_warps=cfg.get("num_warps", 4))
    else:
        assert cfg["variant"] == "dot", cfg
        grid = (triton.cdiv(N, cfg["BLOCK_N"]), triton.cdiv(M, cfg["BLOCK_M"]))
        _LAST["kernel"] = _dot_kernel[grid](x2, q, s, y, M, N, K, x2.stride(0), q.stride(0), y.stride(0), BLOCK_M=cfg["BLOCK_M"],
                                            BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], num_warps=cfg.get("num_warps", 4),
                                            num_stages=cfg.get("num_stages", 2))
    return y.reshape(*x.shape[:-1], N)


def kernel_info(cfg, rows, N=512, K=512, device="cpu", dtype=torch.float32):
    """
    What the compiler made of a configuration: registers per thread, registers spilled to local memory and static shared memory, read
    from the compiled kernel after a launch on random data of the given shape (the even-K and even-N fast paths depend on the shape).
    {} where the numbers are not available: in Triton's CPU interpreter, or in a Triton version that does not expose them. Stage 11 uses it
    on the GPU to check the offline compile (registers, spills) behind `fits`.
    """
    if not HAVE_TRITON:
        return {}
    x = torch.randn(rows, K, device=device, dtype=dtype)
    q = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8)
    _LAST["kernel"] = None
    try:
        int8_matmul(x, q, torch.rand(N, device=device, dtype=torch.float32) * 0.01, cfg)
    except Exception:
        return {}
    k = _LAST["kernel"]
    out = {name: getattr(k, attr, None) for name, attr in (("regs", "n_regs"), ("spills", "n_spills"))}
    out["shared"] = getattr(getattr(k, "metadata", None), "shared", None)
    return {a: b for a, b in out.items() if b is not None}
