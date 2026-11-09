"""Hot-swappable drafter/loss pipeline.

Pick a drafter and an objective by name; everything else (model build, batch, step,
optimizer) is shared. New methods register one entry in ``METHODS``.

    python pipeline.py --list
    python pipeline.py --check                      # one step per (method, loss)
    python pipeline.py --method dflash --loss dpace --steps 50
    PY=/path/to/env/bin/python scripts/run.sh --method dflash --loss ce

Losses are model-agnostic where possible: D-PACE and plain CE train any drafter,
the DFlash objective needs the verifier's logits and is bound with
``dflash_objective(...)``.
"""
import argparse
import sys

import torch
import torch.nn.functional as F

from losses.dpace import dpace_loss
from models.dflash import DFlashSpeculator
from models.dflash.loss import dflash_objective


# --------------------------------------------------------------------------- #
# methods: name -> builder + the objectives it can be trained with
# --------------------------------------------------------------------------- #
def _build_dflash(a):
    """DFlash drafter sharing a frozen verifier lm_head, as at inference time."""
    head = torch.nn.Linear(a.d_target, a.vocab, bias=False)
    for p in head.parameters():
        p.requires_grad = False
    return DFlashSpeculator(
        d_target=a.d_target,
        d_model=a.d_model,
        target_vocab_size=a.vocab,
        block_size=a.block,
        num_layers=a.layers,
        num_heads=a.heads,
        target_lm_head=head,
    )


METHODS = {
    "dflash": {"build": _build_dflash, "losses": ("ce", "dflash", "dpace")},
}

# losses that need real CUDA kernels
_GPU_ONLY_LOSSES = {"dflash", "dpace"}


# --------------------------------------------------------------------------- #
# batch + objective selection
# --------------------------------------------------------------------------- #
def make_batch(a, gen):
    """Synthetic stand-in for a verifier rollout batch."""
    return {
        "feats": torch.randn(a.batch, a.seq, a.d_target, generator=gen, device=a.device),
        "tokens": torch.randint(0, a.vocab, (a.batch, a.block), generator=gen, device=a.device),
        "target_logits": torch.randn(a.batch, a.block, a.vocab, generator=gen, device=a.device),
        # slot 0 of each block is the anchor and is never trained
        "mask": torch.ones(a.batch, a.block, device=a.device).index_fill_(1, torch.tensor([0], device=a.device), 0.0),
    }


def select_loss(name, batch, a):
    """Return fn(logits, tokens) -> scalar, or None for the model's own CE."""
    if name == "ce":
        return None
    if name == "dflash":
        return dflash_objective(
            batch["target_logits"], loss_mask=batch["mask"], gamma=a.gamma,
            sample_from_anchor=a.sample_from_anchor,
        )
    if name == "dpace":
        return lambda logits, tokens: dpace_loss(logits, tokens, a.alpha,
                                                 position_mask=batch["mask"])
    raise ValueError(f"unknown loss {name!r}, expected one of {sorted(_ALL_LOSSES)}")


_ALL_LOSSES = {n for m in METHODS.values() for n in m["losses"]}


def step(model, batch, loss_name, a):
    logits = model(batch["feats"])
    lfn = select_loss(loss_name, batch, a)
    if lfn is None:
        loss = F.cross_entropy(logits.reshape(-1, a.vocab), batch["tokens"].reshape(-1))
    else:
        loss = lfn(logits, batch["tokens"])
    return logits, loss


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def run(args):
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    model = METHODS[args.method]["build"](args).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    batch = make_batch(args, gen)

    print(f"method={args.method} loss={args.loss} device={args.device} "
          f"model_params={sum(p.numel() for p in params):,}")
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        _, loss = step(model, batch, args.loss, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if i % max(1, args.steps // 10) == 0 or i == args.steps - 1:
            print(f"  step {i:4d}  loss {float(loss.detach()):.6f}")
    print("ok")


def check(args):
    """One step per (method, loss) pair -- proves the swap is wired, not just importable."""
    fails = []
    for method, spec in METHODS.items():
        for loss_name in spec["losses"]:
            if loss_name in _GPU_ONLY_LOSSES and not torch.cuda.is_available():
                print(f"SKIP {method}/{loss_name}: needs CUDA")
                continue
            torch.manual_seed(args.seed)
            gen = torch.Generator(device=args.device).manual_seed(args.seed)
            a = argparse.Namespace(**vars(args))
            a.method, a.loss = method, loss_name
            model = spec["build"](a).to(a.device)
            batch = make_batch(a, gen)
            _, loss = step(model, batch, loss_name, a)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            ok = (torch.isfinite(loss).item() and grads
                  and all(torch.isfinite(g).all().item() for g in grads))
            print(f"{'PASS' if ok else 'FAIL':4s} {method}/{loss_name}  loss={float(loss.detach()):.6f}")
            if not ok:
                fails.append(f"{method}/{loss_name}")
    print("\nALL PASS" if not fails else f"\nFAILURES: {fails}")
    return 1 if fails else 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Hot-swappable spec-decode drafter/loss pipeline")
    p.add_argument("--method", default="dflash", choices=sorted(METHODS))
    p.add_argument("--loss", default="dpace", choices=sorted(_ALL_LOSSES))
    p.add_argument("--list", action="store_true", help="show methods and their objectives")
    p.add_argument("--check", action="store_true", help="run one step per (method, loss)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    # shapes
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=16, help="context length fed to the drafter")
    p.add_argument("--block", type=int, default=4, help="draft block size")
    p.add_argument("--d-target", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--vocab", type=int, default=512)
    # objective knobs
    p.add_argument("--gamma", type=float, default=4.0, help="DFlash decay")
    p.add_argument("--alpha", type=float, default=0.5, help="D-PACE smoothing")
    p.add_argument("--sample-from-anchor", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for m, spec in METHODS.items():
            print(f"{m}: {', '.join(spec['losses'])}")
        return 0
    if args.check:
        return check(args)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
