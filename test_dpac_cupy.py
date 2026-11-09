"""Cross-checks dpac_cupy against specforge's D-PACE spec.

Oracle is copied verbatim from D-PACE/tests/test_utils/test_dflash_losses.py
(_naive_dpace_weight + the `sum(w*ce)/bsz` reduction), which that repo's own
suite pins against specforge/core/dflash.py::_dpace_weight.
"""
import os, sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp
from dpac_cupy import fused_dpace_loss, dpace_loss
from dpace import d_pace_loss

fails = []
def chk(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL':4s} {name} {extra}")
    if not cond: fails.append(name)


# ---------------- specforge oracle ----------------
def _naive_dpace_weight(prob, binary_mask, alpha, loss_type):
    smooth = (1.0 - alpha) * prob + alpha
    smooth = torch.where(binary_mask > 0, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)
    if loss_type == "dpace-cumulative-confidence-only":
        return prefix
    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1), dims=[-1]
    )
    if loss_type == "dpace":
        return suffix
    if loss_type == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(loss_type)


def oracle_loss(logits, targets, alpha=0.5, position_mask=None, loss_type="dpace"):
    bsz, _, vocab = logits.shape
    mask = (torch.ones_like(targets, dtype=logits.dtype) if position_mask is None
            else (position_mask > 0).to(logits.dtype))
    neg_log_q = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1),
                                reduction="none").view_as(targets)
    with torch.no_grad():
        weight = _naive_dpace_weight(torch.exp(-neg_log_q), mask, alpha, loss_type)
    return (neg_log_q * weight * mask).sum() / float(bsz)


TYPES = ("dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only")
rng = np.random.default_rng(0)

def run_case(name, N, B, V, alpha=0.5, mask_prob=None, loss_type="dpace", shift=0.0):
    lo = (rng.normal(0, 1, (N, B, V)) + shift).astype(np.float32)
    tg = rng.integers(0, V, (N, B))
    mk = None if mask_prob is None else (rng.random((N, B)) < mask_prob).astype(np.float32)
    lt, tt = torch.from_numpy(lo).cuda(), torch.from_numpy(tg).cuda()
    mt = None if mk is None else torch.from_numpy(mk).cuda()
    want = float(oracle_loss(lt, tt, alpha, mt, loss_type))
    got = float(fused_dpace_loss(cp.asarray(lo), cp.asarray(tg, cp.int64),
                                 alpha, None if mk is None else cp.asarray(mk), loss_type))
    ref = float(d_pace_loss(lt, tt, alpha, mt, loss_type))   # our torch reference
    ok = (np.isfinite(got) and abs(got - want) < 2e-3 * max(1, abs(want))
          and abs(ref - want) < 2e-3 * max(1, abs(want)))
    chk(name, ok, f"kernel={got:.6f} ref={ref:.6f} spec={want:.6f}")


print("-- forward vs specforge --")
run_case("no mask            N8 B4 V1024", 8, 4, 1024)
run_case("mask p=0.5         N8 B8 V1024", 8, 8, 1024, mask_prob=0.5)
run_case("mask p=0.2         N8 B8 V1024", 8, 8, 1024, mask_prob=0.2)
run_case("mask p=0.05 (sparse)N8 B16 V1024", 8, 16, 1024, mask_prob=0.05)
run_case("B=1 masked", 8, 1, 1024, mask_prob=0.5)
run_case("B=32 masked", 8, 32, 1024, mask_prob=0.5)
run_case("V=7 masked", 6, 8, 7, mask_prob=0.5)
run_case("V=1000 masked", 6, 8, 1000, mask_prob=0.5)
run_case("V=32000 masked", 4, 8, 32000, mask_prob=0.5)
run_case("V=151936 masked", 2, 8, 151936, mask_prob=0.5)
run_case("all-negative logits masked", 8, 4, 1024, mask_prob=0.5, shift=-20.0)
for a in (0.0, 0.1, 0.5, 0.9, 1.0):
    run_case(f"alpha={a} masked", 4, 8, 2048, alpha=a, mask_prob=0.5)
for t in TYPES:
    run_case(f"loss_type={t}", 4, 8, 1024, mask_prob=0.5, loss_type=t)

# all-invalid row / all-invalid sequence must be 0 and finite
lo = rng.normal(0, 1, (4, 8, 512)).astype(np.float32); tg = rng.integers(0, 512, (4, 8))
mk = np.zeros((4, 8), np.float32)
g = float(fused_dpace_loss(cp.asarray(lo), cp.asarray(tg, cp.int64), 0.5, cp.asarray(mk)))
chk("all-masked -> 0", g == 0.0, f"got={g}")
mk2 = np.ones((4, 8), np.float32); mk2[0] = 0.0
lt, tt = torch.from_numpy(lo).cuda(), torch.from_numpy(tg).cuda()
g = float(fused_dpace_loss(cp.asarray(lo), cp.asarray(tg, cp.int64), 0.5, cp.asarray(mk2)))
w = float(oracle_loss(lt, tt, 0.5, torch.from_numpy(mk2).cuda()))
chk("one fully-masked sequence", abs(g - w) < 2e-3, f"got={g:.6f} spec={w:.6f}")

# bool mask accepted
lbt = torch.from_numpy(lo).cuda(); mbool = torch.from_numpy(mk2.astype(bool)).cuda()
chk("bool mask accepted",
    abs(float(dpace_loss(lbt, tt, 0.5, mbool))
        - float(fused_dpace_loss(cp.asarray(lo), cp.asarray(tg, cp.int64), 0.5, cp.asarray(mk2)))) < 1e-5)

print("\n-- backward vs torch autograd on the spec formula --")
for name, (N, B, V) in [("N4 B8 V1024 masked", (4, 8, 1024)), ("N8 B16 V4096 masked", (8, 16, 4096)),
                        ("N2 B8 V32000 masked", (2, 8, 32000)), ("V=7 masked", (3, 8, 7))]:
    lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tg = rng.integers(0, V, (N, B))
    mk = (rng.random((N, B)) < 0.5).astype(np.float32); mk[0, 0] = 0.0
    lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    tt = torch.from_numpy(tg).cuda(); mt = torch.from_numpy(mk).cuda()
    oracle_loss(lt, tt, 0.5, mt).backward(); g_ref = lt.grad.clone()
    lt2 = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    dpace_loss(lt2, tt, 0.5, mt).backward()
    rel = (lt2.grad - g_ref).abs().max().item() / max(1e-9, g_ref.abs().max().item())
    chk(f"bwd {name}", rel < 2e-3, f"maxrel={rel:.2e}")

for t in TYPES:
    N, B, V = 3, 8, 512
    lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tg = rng.integers(0, V, (N, B))
    mk = (rng.random((N, B)) < 0.6).astype(np.float32)
    lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    tt = torch.from_numpy(tg).cuda(); mt = torch.from_numpy(mk).cuda()
    oracle_loss(lt, tt, 0.5, mt, t).backward(); g_ref = lt.grad.clone()
    lt2 = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    dpace_loss(lt2, tt, 0.5, mt, t).backward()
    rel = (lt2.grad - g_ref).abs().max().item() / max(1e-9, g_ref.abs().max().item())
    chk(f"bwd loss_type={t}", rel < 2e-3, f"maxrel={rel:.2e}")

# masked positions must get exactly zero grad
N, B, V = 2, 8, 256
lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tg = rng.integers(0, V, (N, B))
mk = np.ones((N, B), np.float32); mk[:, 3] = 0.0
lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
tt = torch.from_numpy(tg).cuda(); mt = torch.from_numpy(mk).cuda()
dpace_loss(lt, tt, 0.5, mt).backward()
chk("masked row grad == 0", lt.grad[:, 3].abs().max().item() == 0.0)
chk("unmasked row grad != 0", lt.grad[:, 2].abs().max().item() > 0)

print("\n-- finite-difference spot check --")
N, B, V = 2, 4, 64
lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tg = rng.integers(0, V, (N, B))
mk = (rng.random((N, B)) < 0.6).astype(np.float32)
lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
tt = torch.from_numpy(tg).cuda(); mt = torch.from_numpy(mk).cuda()
dpace_loss(lt, tt, 0.5, mt).backward(); ga = lt.grad.clone()
eps, maxerr = 1e-3, 0.0
for _ in range(12):
    n, b, v = rng.integers(0, N), rng.integers(0, B), rng.integers(0, V)
    lp = torch.from_numpy(lo.copy()).cuda(); lp[n, b, v] += eps
    lm = torch.from_numpy(lo.copy()).cuda(); lm[n, b, v] -= eps
    fd = (float(dpace_loss(lp, tt, 0.5, mt)) - float(dpace_loss(lm, tt, 0.5, mt))) / (2 * eps)
    maxerr = max(maxerr, abs(fd - float(ga[n, b, v])))
chk("bwd finite-difference (12 probes)", maxerr < 3e-3, f"maxerr={maxerr:.2e}")

print("\n-- misc --")
lo = rng.normal(0, 1, (16, 8, 8192)).astype(np.float32); tg = rng.integers(0, 8192, (16, 8))
lc, tc = cp.asarray(lo), cp.asarray(tg, cp.int64)
a = float(fused_dpace_loss(lc, tc))
chk("forward bitwise repeatable", all(float(fused_dpace_loss(lc, tc)) == a for _ in range(10)))
try:
    d_pace_loss(torch.from_numpy(lo[:1]).cuda(), torch.from_numpy(tg[:1]).cuda(), 1.5)
    chk("alpha>1 rejected", False)
except ValueError as e:
    chk("alpha>1 rejected", True, f"({e})")
try:
    d_pace_loss(torch.from_numpy(lo[:1]).cuda(), torch.from_numpy(tg[:1]).cuda(), 0.5, None, "topk_mask")
    chk("bad loss_type rejected", False)
except ValueError as e:
    chk("bad loss_type rejected", True, f"({e})")
try:
    bad = tg[:1].copy(); bad[0, 0] = 999999
    fused_dpace_loss(cp.asarray(lo[:1]), cp.asarray(bad, cp.int64))
    chk("OOB target rejected", False)
except ValueError as e:
    chk("OOB target rejected", True, f"({e})")
try:
    fused_dpace_loss(cp.asarray(lo[:2]), cp.asarray(tg[:2], cp.int64))
    l2 = dpace_loss(torch.from_numpy(lo[:2]).cuda(), torch.from_numpy(tg[:2]).cuda())
    chk("no-mask defaults to all-ones", abs(float(l2) - float(fused_dpace_loss(cp.asarray(lo[:2]), cp.asarray(tg[:2], cp.int64)))) < 1e-5)
except Exception as e:
    chk("no-mask defaults to all-ones", False, str(e))
with torch.no_grad():
    l = dpace_loss(torch.from_numpy(lo[:2]).cuda(), torch.from_numpy(tg[:2]).cuda())
chk("works under torch.no_grad", bool(torch.isfinite(l)))

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)
