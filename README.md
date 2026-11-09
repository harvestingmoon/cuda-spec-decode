# DPACE megakernel

**Unofficial implementation, based on the [D-PACE](https://github.com/Lucas-TY/D-PACE) GitHub repo
(`specforge/core/dflash.py::_dpace_weight`, MIT, sgl-project, commit `f36bad6e`).**
Not affiliated with the authors; the official loss is the Triton/torch path in SpecForge above.

A single CUDA kernel fuses the whole D-PACE forward softmax, target gather, cross entropy, the
prefix-survival / continuation-value scan and the weighted reduction  into one pass over the
logits, plus a fused backward. No softmax tensor is materialized, no V-sized intermediate is saved.

## Files

| file | what |
|---|---|
| `dpac_cupy.py` | the megakernel (forward + backward), cupy + torch entry points |
| `dpace.py` | reference transcription of `_dpace_weight`, used as the correctness oracle |
| `test_dpac_cupy.py` | 36 checks against that oracle, incl. specforge's own test formula |

## Usage

```python
from dpac_cupy import dpace_loss          # torch, autograd-aware

# logits   [N, B, V]  float32, CUDA      (N = batch of sequences, B = block <= 32)
# targets  [N, B]     int64
# mask     [N, B]     bool/float, None = all positions scored
loss = dpace_loss(logits, targets, alpha=0.5, position_mask=mask, loss_type="dpace")
loss.backward()
```

Cupy-only / eval:

```python
from dpac_cupy import fused_dpace_loss
loss = fused_dpace_loss(logits_cp, targets_cp, alpha=0.5, position_mask=mask_cp)
```

## Exactly what it computes

Same objective as SpecForge, including the details the naive port gets wrong:

```
q_j      = exp(-CE_j)                      # draft prob on the target token
smooth_j = (1 - alpha) * q_j + alpha        if mask_j > 0 else 1      # no-op in prefix
P_j      = prod_{i<=j} smooth_i             # prefix survival
S_j      = sum_{k>=j} P_k * mask_k          # == P_j * continuation value
w_j      = S_j (or P_j, or S_j/P_j) * mask_j
loss     = sum_j w_j * CE_j / N             # divided by batch size, NOT by the weight sum
```

Gradients: the weights are detached, so `dL/dlogits[n,j,:] = (w_j / N) * (softmax - onehot)`  the
whole scan drops out of the backward, which is why it needs no saved softmax and no reductions.

## Caveats
 
`alpha=0` is degenerate: the prefix product underflows and the weights collapse (measured loss
`0.006` vs `12.0` at alpha=0.5 on the same batch). Use the paper default 0.5. This matches
SpecForge — it is not a kernel artifact.

Because the loss divides by batch size rather than by the weight sum, its scale and the gradient
magnitude both move with draft confidence. Don't compare loss values against a weight-normalized
implementation (e.g. an older revision of `dpace.py`).

## Verified

`python test_dpac_cupy.py` 36/36 pass (RTX 3060 6 GB, WSL2, cupy 14.0.1, torch 2.11+cu130):

- forward matches SpecForge's `_naive_dpace_weight` oracle to <= 2e-3 rel: mask density 0/0.05/0.2/0.5,
  B = 1/4/8/16/32, V = 7 … 151936, alpha = 0/0.1/0.5/0.9/1, all three loss types, all-negative logits
- backward matches torch autograd on that formula to 7e-9 … 2e-7 rel; masked rows get exactly zero grad
- independent finite-difference spot check: 1.3e-3
- all-masked → 0.0, bool masks accepted, bad alpha / loss_type / out-of-range targets raise
- forward is bitwise repeatable (no atomics anywhere)

SpecForge's own `tests/test_utils/test_dflash_losses.py` also passes (9/9) against the upstream
checkout, which is what the oracle above is copied from.

## Speed (fwd + bwd, median of 30, RTX 3060)

| shape | kernel | vectorized torch | plain `F.cross_entropy` |
|---|---|---|---|
| N=4 B=8 V=32000 | 0.91 ms | 2.00 ms | 0.30 ms |
| N=4 B=8 V=151936 | 2.13 ms | 3.06 ms | 0.60 ms |
| N=32 B=8 V=32000 | 1.24 ms | 1.10 ms | 0.95 ms |
| N=8 B=16 V=32000 | 1.26 ms | 4.61 ms | 0.48 ms |

Honest read: the win over the reference is mostly at large B and large V. At N≈32 the ~10-line
vectorized torch version is already at parity, and the kernel is always ~2-3x a bare
`F.cross_entropy` because it reads the logits twice (max pass, then exp-sum) and writes a full
`[N,B,V]` grad. The obvious next step would be a single online softmax pass and splitting the grid
over `(n, j)` instead of one block per sequence — neither is implemented.

## Limits

- fp32 only (asserts; no fp16/bf16 path), `B <= 32`, contiguous inputs required
- one thread block per sequence → for small N most SMs idle
- assumes the default CUDA stream (cupy and torch both sit on it); a custom `torch.cuda.stream(...)`
  would need an explicit sync
- the grad returned by `backward` is a DLPack view into cupy's allocator, so torch's peak-memory
  accounting doesn't count it

## Attribution

D-PACE and SpecForge: <https://github.com/Lucas-TY/D-PACE> (MIT, © 2025 sgl-project).
This kernel is an independent reimplementation of that loss for a different model/draft setup; the
math follows `_dpace_weight`, the CUDA is not derived from SpecForge's Triton kernels.