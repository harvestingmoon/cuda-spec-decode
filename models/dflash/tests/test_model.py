"""Sanity + integration tests for the DFlash drafter.

The first five checks are the original single-script suite from
``spec_decode/dflash.py``; the rest cover the loss hook, i.e. that the same
drafter trains under the fused DFlash objective and under D-PACE.
"""
import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from models.dflash.model import DFlashSpeculator
from models.dflash.loss import dflash_loss
from losses.dpace import dpace_loss

fails = []
def chk(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL':4s} {name} {extra}")
    if not cond: fails.append(name)


torch.manual_seed(42)
B, S_ctx, d_target, d_model, V, block_size, num_layers = 2, 16, 64, 32, 256, 4, 2

frozen_lm_head = nn.Linear(d_target, V, bias=False)
for p in frozen_lm_head.parameters():
    p.requires_grad = False

model = DFlashSpeculator(
    d_target=d_target, d_model=d_model, target_vocab_size=V,
    block_size=block_size, num_layers=num_layers, target_lm_head=frozen_lm_head,
)
feats = torch.randn(B, S_ctx, d_target)

# 1. shape & sanity
logits = model(feats)
chk("forward shape", logits.shape == (B, block_size, V), str(tuple(logits.shape)))
chk("forward finite", bool(torch.isfinite(logits).all()))

# 2. loss + backward, frozen head untouched
model.train()
gt = torch.randint(0, V, (B, block_size))
loss = model.compute_loss(feats, gt)
chk("loss scalar & finite", loss.dim() == 0 and bool(torch.isfinite(loss)), f"{float(loss.detach()):.4f}")
loss.backward()
chk("proj_in grad", model.proj_in.weight.grad is not None)
chk("slot_queries grad", model.slot_queries.grad is not None)
chk("proj_out grad", model.proj_out.weight.grad is not None)
chk("frozen lm_head untouched", frozen_lm_head.weight.grad is None)
model.zero_grad(set_to_none=True)

# 3. drafting
model.eval()
tok, lg = model.draft(feats)
chk("draft tokens shape/dtype", tok.shape == (B, block_size) and tok.dtype == torch.int64)
chk("draft logits shape", lg.shape == (B, block_size, V))
chk("draft == argmax(forward)", bool((tok == model(feats).argmax(-1)).all()))

# 4. mask isolation
m = model._build_dflash_mask(S_ctx=4, S_block=2, device=torch.device("cpu"))
chk("context cannot see block", bool((m[0, 0, :4, 4:] == float("-inf")).all()))
chk("block sees context", bool((m[0, 0, 4:, :4] == 0.0).all()))
chk("block is bidirectional", bool((m[0, 0, 4:, 4:] == 0.0).all()))
iu = torch.triu_indices(4, 4, offset=1)
chk("context is causal", bool((m[0, 0, :4, :4][iu[0], iu[1]] == float("-inf")).all()))

# 5. batch invariance
with torch.no_grad():
    single = model(feats[0:1])
    batched = model(torch.cat([feats[0:1], feats[1:2]], dim=0))
chk("batch invariance", (single[0] - batched[0]).abs().max().item() < 1e-4)

# 6. loss hook: fused DFlash objective on GPU
gpu = DFlashSpeculator(
    d_target=d_target, d_model=d_model, target_vocab_size=V,
    block_size=block_size, num_layers=num_layers, target_lm_head=frozen_lm_head,
).cuda().float()
gf = feats.cuda(); ggt = gt.cuda()

# target logits = the frozen target head applied to random target features, as in training
with torch.no_grad():
    target_logits = torch.randn(B, block_size, V, device="cuda")
    ce_ref = F.cross_entropy(gpu(gf).reshape(-1, V), ggt.reshape(-1))
    fused = dflash_loss(gpu(gf), target_logits, None, 4.0, False, "kl")
chk("fused dflash loss runs on model logits", bool(torch.isfinite(fused)), f"loss={float(fused.detach()):.4f} ce={float(ce_ref.detach()):.4f}")
chk("fused loss != plain CE (different objective)", abs(float(fused) - float(ce_ref)) > 1e-6)

out = gpu.compute_loss(gf, ggt, loss_fn=lambda lg_, tg_: dflash_loss(lg_, target_logits, None, 4.0, False, "kl"))
out.backward()
chk("dflash obj: grads reach proj_in", gpu.proj_in.weight.grad is not None
    and bool(torch.isfinite(gpu.proj_in.weight.grad).all()))
gpu.zero_grad(set_to_none=True)

# 7. loss hook: D-PACE objective on the same drafter
dp = gpu.compute_loss(gf, ggt, loss_fn=lambda lg_, tg_: dpace_loss(lg_, tg_, 0.5))
chk("dpace obj: finite", bool(torch.isfinite(dp)), f"loss={float(dp.detach()):.4f}")
dp.backward()
chk("dpace obj: grads reach slot_queries", gpu.slot_queries.grad is not None
    and bool(torch.isfinite(gpu.slot_queries.grad).all()))
gpu.zero_grad(set_to_none=True)

# 8. the two objectives disagree on the same logits (sanity: the hook really swaps)
with torch.no_grad():
    l_ce = gpu.compute_loss(gf, ggt)
    l_dp = gpu.compute_loss(gf, ggt, loss_fn=lambda lg_, tg_: dpace_loss(lg_, tg_, 0.5))
chk("CE vs D-PACE differ", abs(float(l_ce) - float(l_dp)) > 1e-6,
    f"ce={float(l_ce.detach()):.4f} dpace={float(l_dp.detach()):.4f}")

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)
