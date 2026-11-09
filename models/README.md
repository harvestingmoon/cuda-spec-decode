# models/

Drafter implementations live here — one folder per method, self-contained, no shared
base class until there is a second one to share.

Nothing is in here yet. What belongs:

```
models/
  medusa/     # Medusa heads: tree attention + per-head cross entropy
  dflash/     # DFlash block drafter: target-feature fusion, block queries,
              #   causal-context + bidirectional-block attention mask
  eagle/      # EAGLE autoregressive drafter
```

Rules of the split:

- `losses/` owns *how the drafter is trained* (the fused kernels). A loss must work for any
  drafter that emits `[N, B, V]` logits, so nothing in `losses/` may import a model.
- `models/` owns *what the drafter is* — attention masks, block queries, feature fusion,
  inference-time drafting. It calls into `losses/`, never the reverse.
- a model folder may carry its own reference loss as a baseline, but the shipped one should
  come from `losses/`.

D-PACE is a loss, not a model: it lives in `losses/` and applies to any drafter.
