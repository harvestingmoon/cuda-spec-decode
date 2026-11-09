"""Fused speculative-decoding losses (unofficial reimplementations).

Each module is self-contained: the CUDA source, the cupy-only entry point and the
torch autograd entry point live together, and the matching torch reference sits in
``*_ref.py`` next to it.

    from losses import dpace_loss, fused_dpace_loss, dflash_loss, fused_dflash_loss
"""
from losses.dpace import dpace_loss, fused_dpace_loss
from losses.dflash import dflash_loss, fused_dflash_loss

__all__ = ["dpace_loss", "fused_dpace_loss", "dflash_loss", "fused_dflash_loss"]
