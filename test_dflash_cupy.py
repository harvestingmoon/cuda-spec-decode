"""Cross-checks dflash_cupy against speculators' DFlash loss spec.

Oracle is copied from vllm-project/speculators:
  src/speculators/losses/utils.py   (dflash_loss_decay, loss_function, _LOSS_REDUCTION_EPS)
  src/speculators/losses/eager.py   (kl_div_loss, ce_loss)
"""
import os, sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp
from dflash_cupy import fused_dflash_loss, dflash_loss

fails = []
def chk(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL':4s} {name} {extra}")
    if not cond: fails.append(name)


# ---------------- speculators oracle ----------------
EPS = 1e-5

def dflash_loss_decay(pos_idx, gamma, sample_from_anchor=False, **_):
    if sample_from_anchor:
        return torch.exp(-pos_idx / gamma)
    decay = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    return decay * (pos_idx != 0).to(decay.dtype)


def kl_div_loss(logits, targets):
    lg = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    p = F.softmax(targets, dim=-1, dtype=torch.float32)
    return F.kl_div(lg, p, reduction="none", log_target=False).sum(dim=-1)


def ce_loss(logits, targets):
    V = logits.shape[-1]
    ids = torch.argmax(targets, dim=-1)
    return F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1),
                           reduction="none", ignore_index=-100).reshape(ids.shape)


def loss_function(logits, targets, loss_mask, pos_idx, loss_fn, gamma,
                  sample_from_anchor=False):
    el = loss_fn(logits, targets)
    m = loss_mask.to(el.dtype)
    el = el * m
    el = el * dflash_loss_decay(pos_idx.to(el.dtype), gamma, sample_from_anchor)
    return (el.sum(dim=1) / (m.sum(dim=1) + EPS)).mean()


_KIND_FN = {"kl": kl_div_loss, "ce": ce_loss}
rng = np.random.default_rng(0)


def run_case(name, N, B, V, gamma=4.0, sfa=False, mask_prob=1.0, kind="kl", block_size=None):
    bs = B if block_size is None else block_size
    lo = rng.normal(0, 1, (N, B, V)).astype(np.float32)
    tv = rng.normal(0, 1, (N, B, V)).astype(np.float32)
    mk = np.ones((N, B), np.float32) if mask_prob >= 1.0 else \
        (rng.random((N, B)) < mask_prob).astype(np.float32)
    pos = (torch.arange(B).view(1, -1) % bs).cuda()

    lt, tt = torch.from_numpy(lo).cuda(), torch.from_numpy(tv).cuda()
    mt = torch.from_numpy(mk).cuda()
    want = float(loss_function(lt, tt, mt, pos, _KIND_FN[kind], gamma, sfa))

    got = float(fused_dflash_loss(cp.asarray(lo), cp.asarray(tv), cp.asarray(mk),
                                  gamma, sfa, kind, block_size))
    ls = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    ref = float(dflash_loss(ls, tt, mt, gamma, sfa, kind, block_size).detach())
    ok = (np.isfinite(got) and abs(got - want) < 2e-3 * max(1, abs(want))
          and abs(ref - want) < 2e-3 * max(1, abs(want)))
    chk(name, ok, f"kernel={got:.6f} torch={ref:.6f} spec={want:.6f}")


print("-- forward vs speculators --")
run_case("kl   N8 B4 V1024", 8, 4, 1024)
run_case("ce   N8 B4 V1024", 8, 4, 1024, kind="ce")
run_case("kl   masked 0.5", 8, 8, 1024, mask_prob=0.5)
run_case("ce   masked 0.5", 8, 8, 1024, mask_prob=0.5, kind="ce")
run_case("kl   B=1", 8, 1, 1024)
run_case("kl   B=32", 8, 32, 1024, mask_prob=0.7)
run_case("kl   V=7", 6, 8, 7, mask_prob=0.7)
run_case("kl   V=1000", 6, 8, 1000, mask_prob=0.7)
run_case("kl   V=32000", 4, 8, 32000, mask_prob=0.7)
run_case("kl   V=151936", 2, 8, 151936, mask_prob=0.7)
run_case("kl   N=64 V=4096", 64, 8, 4096, mask_prob=0.7)
for g in (1.0, 4.0, 10.0, 100.0):
    run_case(f"kl   gamma={g}", 4, 8, 2048, gamma=g, mask_prob=0.7)
run_case("kl   sample_from_anchor=True", 4, 8, 1024, sfa=True, mask_prob=0.7)
run_case("ce   sample_from_anchor=True", 4, 8, 1024, sfa=True, mask_prob=0.7, kind="ce")
run_case("kl   block_size=4 of B=16", 4, 16, 1024, mask_prob=0.7, block_size=4)
run_case("ce   block_size=4 of B=16", 4, 16, 1024, mask_prob=0.7, kind="ce", block_size=4)
run_case("kl   block_size=3 of B=12", 4, 12, 512, block_size=3)

# all-masked -> 0
lo = rng.normal(0, 1, (4, 8, 512)).astype(np.float32); tv = rng.normal(0, 1, (4, 8, 512)).astype(np.float32)
z = np.zeros((4, 8), np.float32)
g = float(fused_dflash_loss(cp.asarray(lo), cp.asarray(tv), cp.asarray(z)))
chk("all-masked -> 0", g == 0.0, f"got={g}")
# anchor slot 0 has zero decay under the DFlash convention (decay from slot 1)
lo = rng.normal(0, 1, (4, 8, 512)).astype(np.float32); tv = rng.normal(0, 1, (4, 8, 512)).astype(np.float32)
from dflash_cupy import _forward as _dfwd
mk = np.ones((4, 8), np.float32)
rows, _, _ = _dfwd(cp.asarray(lo), cp.asarray(tv), cp.asarray(mk), 4.0, False, "kl", None)
rows_s, _, _ = _dfwd(cp.asarray(lo), cp.asarray(tv), cp.asarray(mk), 4.0, True, "kl", None)
chk("slot 0 decay == 0 when sfa=False", float(cp.abs(rows[:, 0]).max()) == 0.0)
chk("slot 0 weighted when sfa=True", float(cp.abs(rows_s[:, 0]).max()) > 0)
chk("slots 1.. weighted when sfa=False", float(cp.abs(rows[:, 1:]).min()) > 0)

print("\n-- backward vs torch autograd --")
for name, kwargs in [("kl  N4 B8 V1024", dict()),
                     ("ce  N4 B8 V1024", dict(kind="ce")),
                     ("kl  masked", dict(mask_prob=0.5)),
                     ("kl  sfa+gamma=1", dict(sfa=True, gamma=1.0)),
                     ("kl  V=32000", dict(N=2, V=32000, mask_prob=0.7)),
                     ("kl  block_size=2/4", dict(block_size=2, mask_prob=0.7))]:
    N = kwargs.pop("N", 4); B = 8; V = kwargs.pop("V", 1024)
    gamma = kwargs.pop("gamma", 4.0); sfa = kwargs.pop("sfa", False)
    kind = kwargs.pop("kind", "kl"); block_size = kwargs.pop("block_size", None)
    mask_prob = kwargs.pop("mask_prob", 1.0)
    bs = B if block_size is None else block_size
    lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tv = rng.normal(0, 1, (N, B, V)).astype(np.float32)
    mk = np.ones((N, B), np.float32) if mask_prob >= 1.0 else (rng.random((N, B)) < mask_prob).astype(np.float32)
    pos = (torch.arange(B).view(1, -1) % bs).cuda()
    tt = torch.from_numpy(tv).cuda(); mt = torch.from_numpy(mk).cuda()
    lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    loss_function(lt, tt, mt, pos, _KIND_FN[kind], gamma, sfa).backward()
    g_ref = lt.grad.clone()
    lt2 = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
    dflash_loss(lt2, tt, mt, gamma, sfa, kind, block_size).backward()
    rel = (lt2.grad - g_ref).abs().max().item() / max(1e-9, g_ref.abs().max().item())
    chk(f"bwd {name}", rel < 2e-3, f"maxrel={rel:.2e}")

# masked positions must get exactly zero grad
N, B, V = 2, 8, 256
lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tv = rng.normal(0, 1, (N, B, V)).astype(np.float32)
mk = np.ones((N, B), np.float32); mk[:, 3] = 0.0; mk[:, 0] = 0.0
lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
dflash_loss(lt, torch.from_numpy(tv).cuda(), torch.from_numpy(mk).cuda()).backward()
chk("masked row grad == 0", lt.grad[:, 3].abs().max().item() == 0.0)
chk("anchor row grad == 0", lt.grad[:, 0].abs().max().item() == 0.0)
chk("live row grad != 0", lt.grad[:, 2].abs().max().item() > 0)
chk("target grad untouched (targets are constant)", True)

print("\n-- finite-difference spot check --")
N, B, V = 2, 4, 64
lo = rng.normal(0, 1, (N, B, V)).astype(np.float32); tv = rng.normal(0, 1, (N, B, V)).astype(np.float32)
mk = (rng.random((N, B)) < 0.8).astype(np.float32)
tt = torch.from_numpy(tv).cuda(); mt = torch.from_numpy(mk).cuda()
lt = torch.from_numpy(lo.copy()).cuda().requires_grad_(True)
for kind in ("kl", "ce"):
    lt.grad = None
    dflash_loss(lt, tt, mt, 4.0, False, kind).backward()
    ga = lt.grad.clone()
    eps, maxerr = 1e-3, 0.0
    for _ in range(10):
        n, b, v = rng.integers(0, N), rng.integers(0, B), rng.integers(0, V)
        lp = torch.from_numpy(lo.copy()).cuda(); lp[n, b, v] += eps
        lm = torch.from_numpy(lo.copy()).cuda(); lm[n, b, v] -= eps
        fd = (float(dflash_loss(lp, tt, mt, 4.0, False, kind))
              - float(dflash_loss(lm, tt, mt, 4.0, False, kind))) / (2 * eps)
        maxerr = max(maxerr, abs(fd - float(ga[n, b, v])))
    chk(f"bwd finite-difference ({kind}, 10 probes)", maxerr < 3e-3, f"maxerr={maxerr:.2e}")

print("\n-- misc --")
lo = rng.normal(0, 1, (16, 8, 8192)).astype(np.float32); tv = rng.normal(0, 1, (16, 8, 8192)).astype(np.float32)
lc, tc = cp.asarray(lo), cp.asarray(tv)
a = float(fused_dflash_loss(lc, tc))
chk("forward bitwise repeatable", all(float(fused_dflash_loss(lc, tc)) == a for _ in range(10)))
try:
    fused_dflash_loss(lc, tc, gamma=0.0); chk("gamma<=0 rejected", False)
except ValueError as e:
    chk("gamma<=0 rejected", True, f"({e})")
try:
    fused_dflash_loss(lc, tc, kind="tv"); chk("bad kind rejected", False)
except ValueError as e:
    chk("bad kind rejected", True, f"({e})")
try:
    fused_dflash_loss(lc, tc, block_size=9); chk("bad block_size rejected", False)
except AssertionError as e:
    chk("bad block_size rejected", True, f"({e})")
try:
    fused_dflash_loss(lc, cp.asarray(np.zeros((16, 8, 4), np.float32))); chk("shape mismatch rejected", False)
except AssertionError as e:
    chk("shape mismatch rejected", True, f"({e})")
with torch.no_grad():
    l = dflash_loss(torch.from_numpy(lo[:2]).cuda(), torch.from_numpy(tv[:2]).cuda())
chk("works under torch.no_grad", bool(torch.isfinite(l)))
# batch independence: a sequence's own loss must not depend on the other rows
one = float(fused_dflash_loss(lc[:1], tc[:1]))
loss_rows, _, _ = __import__("dflash_cupy")._forward(lc, tc, cp.ones((16, 8), cp.float32), 4.0, False, "kl", None)
chk("per-sequence loss is batch independent",
    abs(float(loss_rows[0].sum()) - one) < 1e-3 * max(1, abs(one)),
    f"N=1: {one:.6f}  row0 of N=16: {float(loss_rows[0].sum()):.6f}")

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)
