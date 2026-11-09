"""Reference D-PACE loss (torch), transcribed from specforge/core/dflash.py::_dpace_weight."""
import torch
import torch.nn.functional as F

_DPACE_TYPES = ("dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only")


def d_pace_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    position_mask: torch.Tensor | None = None,
    loss_type: str = "dpace",
) -> torch.Tensor:
    """D-PACE loss, matching specforge/core/dflash.py::_dpace_weight.

    logits:        [bsz, positions, vocab]
    targets:       [bsz, positions] token ids
    position_mask: [bsz, positions] float/bool; 0 = not scored (anchor, padding, loss_mask=0)
    Returns: sum(w * ce) / bsz  -- normalized by batch size, NOT by the weight sum.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dpace_alpha must be in [0, 1], got {alpha}")
    if loss_type not in _DPACE_TYPES:
        raise ValueError(f"unknown loss_type {loss_type!r}")

    bsz, positions, vocab = logits.shape
    mask = (torch.ones_like(targets, dtype=logits.dtype) if position_mask is None
            else (position_mask > 0).to(logits.dtype))

    neg_log_q = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    ).view_as(targets)

    with torch.no_grad():
        prob = torch.exp(-neg_log_q)
        smooth = (1.0 - alpha) * prob + alpha
        smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))  # no-op inside prefix products
        prefix = torch.cumprod(smooth, dim=-1)                           # P_j

        if loss_type == "dpace-cumulative-confidence-only":
            weight = prefix
        else:
            suffix = torch.flip(  # sum_{k>=j} P_k * mask_k
                torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1]
            )
            weight = (suffix if loss_type == "dpace"
                      else suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny))
        weight = weight * mask

    return (neg_log_q * weight).sum() / float(bsz)
