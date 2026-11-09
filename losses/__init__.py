"""Model-agnostic loss optimizations (unofficial reimplementations).

A loss here trains *any* drafter that emits ``[N, B, V]`` logits, so nothing in this
package imports a model. D-PACE lives here because it applies to any drafter; the
DFlash objective is specific to the DFlash drafter and ships with it in
``models/dflash/loss.py``.

    from losses import dpace_loss, fused_dpace_loss
"""
from losses.dpace import dpace_loss, fused_dpace_loss

__all__ = ["dpace_loss", "fused_dpace_loss"]
