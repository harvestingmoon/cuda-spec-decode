# cuda-spec-decode

Unofficial CUDA/cupy implementations of speculative-decoding drafters and their fused training
objectives, built from [D-PACE / SpecForge](https://github.com/Lucas-TY/D-PACE)
(`specforge/core/`, MIT, sgl-project, commit `f36bad6e`) and
[speculators](https://github.com/vllm-project/speculators) (vllm-project). Not affiliated with
either; the reference implementations are upstream.

Each megakernel fuses a whole training loss — softmax, per-position loss and the position-weighting
reduction — into one pass over the logits, plus a fused backward. No softmax tensor is
materialized, no `[., V]` intermediate is saved.

## Roadmap

Target: a megakernel per drafter, with the objective each method is trained with, and weight
loading wired up to the matching verifier.

| Method | Description | Optimization |
|---|---|---|
| EAGLE | Feature-based autoregressive drafting | LK loss |
| MTP | Multi-token prediction heads | — |
| DFlash | Block-parallel drafting | D-PACE |

Status:

| | drafter | loss kernel |
|---|---|---|
| DFlash | done (`models/dflash/model.py`) | done — DFlash objective + D-PACE |
| EAGLE | planned | planned — LK loss |
| MTP | planned | not decided |
| D-PACE | model-agnostic | done (`losses/dpace.py`) |

Weights: each `models/<name>/` is meant to pair with a `load_verifier_weights()`-style hook that
reuses the verifier's `embed_tokens` and frozen `lm_head` (full- or reduced-vocab `t2d`/`d2t`
mapping), so a drafter checkpoint never re-ships the head. Only the DFlash drafter exists so far;
the hook pattern is taken from `speculators` and is not implemented here yet.

## Layout

```
losses/                     # model-agnostic loss optimizations -- no model imports
  dpace.py                  #   D-PACE megakernel (forward + backward), cupy + torch entry points
  dpace_ref.py              #   torch transcription of _dpace_weight (correctness oracle)
  tests/test_dpace.py       #   38 checks vs specforge's own test formula
models/
  dflash/
    model.py                #   DFlash block drafter (slot queries, mask, feature fusion, draft)
    loss.py                 #   its fused training objective (forward + backward)
    tests/test_model.py     #   21 checks: shape/mask/draft/batch + the loss hook
    tests/test_loss.py      #   43 checks vs speculators' loss_function / dflash_loss_decay
```

**Split rule.** `losses/` holds only what can train *any* drafter that emits `[N, B, V]` logits —
today that is D-PACE. A drafter and anything tied to its conventions live in `models/<name>/`:
DFlash's exponential slot decay is part of the DFlash recipe, so its objective ships with the model.
A model may import `losses/`, never the reverse. See `models/README.md`.

The objective is a hook, so the same drafter trains under either loss:

```python
from models.dflash import DFlashSpeculator, dflash_loss
from losses.dpace import dpace_loss

model = DFlashSpeculator(d_target=4096, d_model=256, target_vocab_size=128256)
loss = model.compute_loss(feats, tokens)                       # plain CE
loss = model.compute_loss(feats, tokens, loss_fn=dflash_loss)  # DFlash objective
loss = model.compute_loss(feats, tokens, loss_fn=lambda lg, tg: dpace_loss(lg, tg, 0.5))
```

## Usage — D-PACE

```python
from losses.dpace import dpace_loss       # torch, autograd-aware

# logits   [N, B, V]  float32, CUDA      (N = batch of sequences, B = block <= 32)
# targets  [N, B]     int64
# mask     [N, B]     bool/float, None = all positions scored
loss = dpace_loss(logits, targets, alpha=0.5, position_mask=mask, loss_type="dpace")
loss.backward()
```

Cupy-only / eval:

```python
from losses.dpace import fused_dpace_loss
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

`python losses/tests/test_dpace.py` 38/38 pass (RTX 3060 6 GB, WSL2, cupy 14.0.1, torch 2.11+cu130):

- forward matches SpecForge's `_naive_dpace_weight` oracle to <= 2e-3 rel: mask density 0/0.05/0.2/0.5,
  B = 1/4/8/16/32, V = 7 … 151936, alpha = 0/0.1/0.5/0.9/1, all three loss types, all-negative logits
- backward matches torch autograd on that formula to 7e-9 … 2e-7 rel; masked rows get exactly zero grad
- independent finite-difference spot check: 1.3e-3
- all-masked → 0.0, bool masks accepted, bad alpha / loss_type / out-of-range targets raise
- forward is bitwise repeatable (no atomics anywhere)

SpecForge's own `tests/test_utils/test_dflash_losses.py` also passes (9/9) against the upstream
checkout, which is what the oracle above is copied from.

Run both suites from the repo root; they import through the `losses` package.

## Speed — D-PACE (fwd + bwd, median of 30, RTX 3060)

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

## DFlash drafter + objective

The DFlash baseline from [speculators](https://github.com/vllm-project/speculators): per-position
KL from the draft to the **target logits**, masked, weighted by an exponential decay over the slot
index, summed over positions and divided by the valid-token count.

```python
from models.dflash import dflash_loss, DFlashSpeculator

# logits, target_logits  [N, B, V]  float32, CUDA   (B = positions/anchors*block_size)
# loss_mask              [N, B]     bool/float, None = all positions
loss = dflash_loss(logits, target_logits, loss_mask=mask, gamma=4.0,
                   sample_from_anchor=False, kind="kl", block_size=None)
loss.backward()
```

`kind="ce"` switches to hard-label cross entropy (`argmax(target_logits)`); `block_size` sets how
the decay index cycles when one sequence holds several anchors. Cupy-only: `fused_dflash_loss`.

## Exactly what it computes

```
elementwise_j = KL(softmax(draft_j) || softmax(target_j))     # or CE vs argmax(target_j)
decay_j       = exp(-(pos_j - 1)/gamma) * (pos_j != 0)        # sample_from_anchor=False
                exp(-pos_j / gamma)                           # sample_from_anchor=True
loss          = mean_n( sum_j mask_j * decay_j * elementwise_j / (sum_j mask_j + 1e-5) )
```

Same masking, decay, `+1e-5` denominator and batch reduction as `loss_function` upstream. KL is
computed in two passes with an algebraically rewritten form, `(B - C)/A - log A + log D`, so it
needs no third pass over the rows:

```
A = sum exp(xt - mp)      B = sum exp(xt - mp)(xt - mp)
C = sum exp(xt - mp)(xl - mq)          D = sum exp(xl - mq)
```

Backward is `dKL/dlogits = softmax(draft) - softmax(target)` (or `- onehot` for CE), scaled by
`mask * decay / (count + 1e-5) / N`. Forward saves only `mq, mp, logzq, logzp, argmax` per row.

## Verified — `python models/dflash/tests/test_loss.py`

Oracle is speculators' own `loss_function` + `dflash_loss_decay` + `eager.kl_div_loss` / `eager.ce_loss`,
copied verbatim. 43/43 pass:

- forward equals the oracle to 6 decimals: `kl` and `ce`, mask density 1/0.7/0.5/0, B = 1/4/8/12/16/32,
  V = 7 … 151936, N up to 64, gamma = 1/4/10/100, `sample_from_anchor` both ways, `block_size` < B
- backward matches torch autograd to 3e-7 … 6e-7 rel (all cases incl. masks and block_size)
- finite-difference spot check: 3.9e-5 (kl) / 1.1e-4 (ce)
- masked and anchor rows get exactly zero grad; slot 0 has zero decay under `sample_from_anchor=False`
- all-masked → 0.0, per-sequence loss independent of the rest of the batch, forward bitwise repeatable
- bad `gamma` / `kind` / `block_size` / shapes raise

## Speed — DFlash (median of 30, RTX 3060)

| shape | kernel fwd | kernel fwd+bwd | eager torch fwd | eager torch fwd+bwd |
|---|---|---|---|---|
| N=4 B=8 V=32000 | 0.20 ms | 0.65 ms | 0.32 ms | 0.61 ms |
| N=4 B=8 V=151936 | 0.41 ms | 0.73 ms | 1.26 ms | 1.78 ms |
| N=32 B=8 V=32000 | 0.43 ms | 0.76 ms | 1.87 ms | 2.75 ms |
| N=8 B=16 V=32000 | 0.23 ms | 0.79 ms | 0.95 ms | 1.42 ms |
| N=128 B=8 V=32000 | 1.65 ms | 3.34 ms | 7.46 ms | 10.94 ms |

1.6-4.4x forward, 1.4-3.3x forward+backward. Unlike D-PACE this kernel is one block per position
row, so it scales with `N*B` and does not idle on small batches.

## Limits

- fp32 only (asserts; no fp16/bf16 path), contiguous inputs required; D-PACE additionally caps `B <= 32`
- D-PACE: one thread block per sequence → for small N most SMs idle
- both: assumes the default CUDA stream (cupy and torch both sit on it); a custom
  `torch.cuda.stream(...)` would need an explicit sync
- the grad returned by `backward` is a DLPack view into cupy's allocator, so torch's peak-memory
  accounting doesn't count it
- DFlash: two full passes over two rows (max, then the weighted sums); an online single-pass variant
  is not implemented

## Attribution

D-PACE and SpecForge: <https://github.com/Lucas-TY/D-PACE> (MIT, © 2025 sgl-project).
Speculators / DFlash loss: <https://github.com/vllm-project/speculators> (vllm-project).
These kernels are independent reimplementations of those losses; the math follows the upstream
Python (`_dpace_weight`, `loss_function` + `dflash_loss_decay`), the CUDA is not derived from their
Triton kernels.