"""Integer-only Llama-2 forward pass — the specification our Solidity must match.

Derived from the algorithm in Karpathy's llama2.c (MIT). Every operation here is
chosen to map onto a single EVM opcode sequence, and no floating point is used
anywhere in the inference path.

Numeric design (ours):
  * Activations are Q32 fixed point held in Python ints, mirroring int256 on chain.
    Q32 leaves 224 bits of headroom in a 256-bit register, so dot-product
    accumulation cannot overflow for any realistic dimension.
  * Weights are int16 with a PER-ROW power-of-two scale. llama2.c is float; a
    per-tensor scale (one exponent for a whole matrix) wastes precision when rows
    differ in magnitude, which they do. Per-row costs one byte per row and buys
    back several bits of mantissa on the small-magnitude rows.
  * exp() is evaluated as 2^(x*log2e) through a bit-product of precomputed
    2^(2^-i) constants, each derived by an exact integer square-root chain, so the
    table itself contains no floating point.
  * Integer sqrt is Newton's method, floor-exact.
"""

F = 32
ONE = 1 << F
LOG2E_Q32 = 6196328018          # floor(log2(e) * 2^32)
EPS_Q32 = 42950                 # 1e-5 in Q32, llama2.c's rmsnorm epsilon
EXP_STEPS = 32


def isqrt(a: int) -> int:
    """Floor integer square root (Newton). Exact for all non-negative a."""
    if a < 0:
        raise ValueError("isqrt of negative")
    if a < 2:
        return a
    x = 1 << ((a.bit_length() + 1) // 2)   # power-of-two seed >= sqrt(a)
    while True:
        y = (x + a // x) >> 1
        if y >= x:
            return x
        x = y


def _exp2_table():
    """EXP2[i] = 2^(-2^-(i+1)) in Q64, built by repeated integer sqrt.

    NEGATIVE exponents: for x <= 0 we evaluate 2^-(int+frac), so the fractional
    factors must be 2^(-1/2), 2^(-1/4), ... each in (0.5, 1]. Seeding with
    isqrt(1<<127) gives sqrt(2^-1) = 2^-0.5 in Q64, and squaring-root from there
    walks down the halving exponents without ever leaving integer arithmetic.
    """
    table = []
    c = isqrt(1 << 127)            # 2^-0.5 in Q64
    for _ in range(EXP_STEPS):
        table.append(c)
        c = isqrt(c << 64)         # sqrt of previous -> next halved exponent
    return table


EXP2 = _exp2_table()


def exp_q32(x: int) -> int:
    """exp(x) for Q32 x <= 0, result Q32 in [0, 2^32].

    exp(x) = 2^(x * log2 e). Split the exponent into integer and fractional
    parts: the integer part is a shift, the fraction is a product of the
    precomputed 2^(2^-i) constants selected by its binary expansion.
    """
    if x > 0:
        raise ValueError("exp_q32 expects x <= 0")
    if x == 0:
        return ONE
    e = (-x * LOG2E_Q32) >> F      # |x| * log2(e), Q32
    int_part = e >> F
    frac = e - (int_part << F)     # Q32 fraction in [0, 1)
    if int_part >= 64:
        return 0                   # underflows Q32 entirely
    acc = 1 << 64                  # 1.0 in Q64
    for i in range(EXP_STEPS):
        if (frac >> (F - 1 - i)) & 1:
            acc = (acc * EXP2[i]) >> 64
    return (acc >> int_part) >> 32  # Q64 -> Q32, applying the integer shift


def rmsnorm(x, g, gshift, dim):
    """out_i = x_i * g_i / sqrt(mean(x^2) + eps), all Q32."""
    ss = 0
    for v in x:
        ss += (v * v) >> F         # keep the sum in Q32
    ss = ss // dim + EPS_Q32
    rms = isqrt(ss << F)           # sqrt in Q32
    out = []
    for i in range(dim):
        num = x[i] * g[i]          # Q(32+gshift)
        out.append(((num << F) // rms) >> gshift)
    return out


def matmul(rows, cols, x, w, shifts, base):
    """out_i = sum_j w[base + i*cols + j] * x_j, requantised by this ROW's shift."""
    out = []
    for i in range(rows):
        acc = 0
        off = base + i * cols
        for j in range(cols):
            acc += w[off + j] * x[j]
        out.append(acc >> shifts[i])
    return out


def rope(vec, n_heads, hs, cos_row, sin_row):
    """Rotate consecutive pairs within each head; tables are Q30."""
    for h in range(n_heads):
        for p in range(hs // 2):
            c, s = cos_row[p], sin_row[p]
            i = h * hs + 2 * p
            v0, v1 = vec[i], vec[i + 1]
            vec[i] = (v0 * c - v1 * s) >> 30
            vec[i + 1] = (v0 * s + v1 * c) >> 30


def softmax_weighted_sum(scores, values, hs, steps):
    """Softmax over `scores` then a weighted sum of value vectors, all integer."""
    mx = max(scores[:steps])
    exps = [exp_q32(scores[t] - mx) for t in range(steps)]
    total = sum(exps)
    out = []
    for j in range(hs):
        acc = 0
        for t in range(steps):
            acc += exps[t] * values[t][j]
        out.append(acc // total)
    return out


def silu(z: int) -> int:
    """z * sigmoid(z) in Q32, using exp_q32 on the correct branch."""
    if z >= 0:
        e = exp_q32(-z)
        sig = (ONE << F) // (ONE + e)
    else:
        e = exp_q32(z)
        sig = (e << F) // (ONE + e)
    return (z * sig) >> F


def argmax(logits):
    """First maximum wins, matching a left-to-right scan with strict >."""
    best, bi = logits[0], 0
    for i in range(1, len(logits)):
        if logits[i] > best:
            best, bi = logits[i], i
    return bi
