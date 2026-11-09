# models/

Drafter implementations live here — one folder per method, self-contained: the model, its
inference path, its training objective when that objective is specific to it, and its tests.

```
models/
  dflash/
    model.py          # DFlash block drafter: target-feature fusion, slot queries,
                      #   causal-context + bidirectional-block attention mask, draft()
    loss.py           # its fused training objective (KL/CE to target logits + exp decay)
    tests/test_model.py
    tests/test_loss.py
```

Planned, same shape:

```
  eagle/      # EAGLE feature-based autoregressive drafter
  mtp/        # multi-token prediction heads
```

Rules of the split:

- **`losses/` = model-agnostic.** Anything there must train *any* drafter that emits
  `[N, B, V]` logits, so it may never import a model. Today that is D-PACE only.
- **`models/<name>/` = one drafter** plus whatever is specific to it. A model may import
  `losses/`; the reverse is forbidden. So `models/dflash/loss.py` (an objective tied to the
  DFlash slot/decay convention) lives with its model, while `losses/dpace.py` can train
  DFlash, EAGLE or MTP unchanged.
- the loss hook makes the swap real, not aspirational:

  ```python
  from models.dflash import DFlashSpeculator, dflash_loss
  from losses.dpace import dpace_loss

  model = DFlashSpeculator(d_target=4096, d_model=256, target_vocab_size=128256)
  loss = model.compute_loss(feats, tokens)                       # plain CE
  loss = model.compute_loss(feats, tokens, loss_fn=dflash_loss)  # DFlash objective
  loss = model.compute_loss(feats, tokens, loss_fn=lambda lg, tg: dpace_loss(lg, tg, 0.5))
  ```
