import cupy as cp
import numpy as np

# -----------------------------------------------------------------------------
# CUDA: fused DFlash loss (+ backward)
#
# Matches vllm-project/speculators:
#   src/speculators/losses/utils.py :: dflash_loss_decay + loss_function
#   src/speculators/losses/eager.py :: kl_div_loss / ce_loss
#
#   elementwise_t = loss_fn(logits_t, target_logits_t)          # KL or CE over V
#   weight_t      = mask_t * exp(-(pos_t - 1).clamp(0) / gamma) * (pos_t != 0)
#   loss          = mean_n( sum_t weight_nt * elementwise_nt / (sum_t mask_nt + 1e-5) )
#
# One block per position row. Two passes over each row pair (max, then the
# weighted sums), no [., V] intermediate saved.
# -----------------------------------------------------------------------------
dflash_cuda_source = r'''
#define VLEN 32
#define NCOMP 6          // 0=mq 1=mp 2=A(sum p) 3=B(sum p*(xt-mp)) 4=C(sum p*(xl-mq)) 5=D(sum q)

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__
void dflash_loss_fwd(
    const float* __restrict__ logits,         // [R, V] draft logits
    const float* __restrict__ target_logits,  // [R, V] target logits
    const float* __restrict__ mask,           // [N, B] loss mask
    const float* __restrict__ denom,          // [N]    mask.sum(1) + 1e-5
    float* __restrict__ out_loss,             // [R]    weight * elementwise loss
    float* __restrict__ out_scale,            // [R]    weight / N   (dL/dloss scale)
    float* __restrict__ out_mq,               // [R]    max(draft logits)
    float* __restrict__ out_mp,               // [R]    max(target logits)
    float* __restrict__ out_logzq,            // [R]    log(sum exp draft)
    float* __restrict__ out_logzp,            // [R]    log(sum exp target)
    long long* __restrict__ out_tgt,           // [R]    argmax(target logits)
    const int N,
    const int B,                               // positions per sequence (row grouping)
    const int BS,                              // block_size: pos index cycles over BS
    const int V,
    const float gamma,
    const int sample_from_anchor,
    const int kind                            // 0 = KL, 1 = CE (argmax of target)
) {
    int r = blockIdx.x;                       // one block per (n, j) row
    if (r >= N * B) return;

    int tid = threadIdx.x;
    int bsize = blockDim.x;
    const float* xl = logits + (size_t)r * V;
    const float* xt = target_logits + (size_t)r * V;

    __shared__ float s[NCOMP][VLEN];
    __shared__ int s_argmax;

    // ---- pass 1: row maxima (draft and target together)
    float mq = -1e30f, mp = -1e30f;
    for (int v = tid; v < V; v += bsize) {
        float a = xl[v], b = xt[v];
        mq = fmaxf(mq, a);
        mp = fmaxf(mp, b);
    }
    mq = warp_max(mq); mp = warp_max(mp);
    if ((tid & 31) == 0) { s[0][tid >> 5] = mq; s[1][tid >> 5] = mp; }
    __syncthreads();
    if (tid == 0) {
        float a = -1e30f, b = -1e30f;
        for (int w = 0; w < (bsize >> 5); ++w) { a = fmaxf(a, s[0][w]); b = fmaxf(b, s[1][w]); }
        s[0][0] = a; s[1][0] = b;
        s_argmax = V;
    }
    __syncthreads();
    mq = s[0][0]; mp = s[1][0];
    __syncthreads();                          // s reused below

    // ---- pass 2: normalizers and the p-weighted terms
    // A = sum exp(xt-mp), B = sum exp(xt-mp)*(xt-mp), C = sum exp(xt-mp)*(xl-mq), D = sum exp(xl-mq)
    float A = 0.f, Bs = 0.f, Cs = 0.f, D = 0.f;
    for (int v = tid; v < V; v += bsize) {
        float a = xl[v], b = xt[v];
        float ph = expf(b - mp);
        A += ph; Bs += ph * (b - mp); Cs += ph * (a - mq); D += expf(a - mq);
        if (b == mp) atomicMin(&s_argmax, v);   // argmax(target) = first max index
    }
    A = warp_sum(A); Bs = warp_sum(Bs); Cs = warp_sum(Cs); D = warp_sum(D);
    if ((tid & 31) == 0) {
        int w = tid >> 5;
        s[2][w] = A; s[3][w] = Bs; s[4][w] = Cs; s[5][w] = D;
    }
    __syncthreads();

    if (tid == 0) {
        float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
        for (int w = 0; w < (bsize >> 5); ++w) {
            a += s[2][w]; b += s[3][w]; c += s[4][w]; d += s[5][w];
        }
        float logzq = mq + logf(d);
        float logzp = mp + logf(a);
        float elementwise;
        if (kind == 0) {
            // KL(p||q) = (B - C)/A - log A + log D
            elementwise = (b - c) / fmaxf(a, 1e-38f) - logf(fmaxf(a, 1e-38f)) + logf(fmaxf(d, 1e-38f));
        } else {
            int tgt = (s_argmax < V) ? s_argmax : 0;
            elementwise = logzq - xl[tgt];
        }

        int n = r / B;
        int pos = r % BS;
        float decay = sample_from_anchor ? expf(-(float)pos / gamma)
                                         : ((pos == 0) ? 0.0f : expf(-(float)(pos - 1) / gamma));
        float w = mask[r] * decay / denom[n];

        out_loss[r]  = w * elementwise;
        out_scale[r] = w / (float)N;
        out_mq[r] = mq; out_mp[r] = mp; out_logzq[r] = logzq; out_logzp[r] = logzp;
        out_tgt[r] = (long long)((s_argmax < V) ? s_argmax : 0);
    }
}

extern "C" __global__
void dflash_loss_bwd(
    const float* __restrict__ logits,         // [R, V]
    const float* __restrict__ target_logits,  // [R, V]
    const float* __restrict__ scale,          // [R]
    const float* __restrict__ mq,             // [R]
    const float* __restrict__ mp,             // [R]
    const float* __restrict__ logzq,          // [R]
    const float* __restrict__ logzp,          // [R]
    const long long* __restrict__ target_id,  // [R] argmax(target), CE only
    const float* __restrict__ dloss,          // upstream grad (scalar)
    float* __restrict__ dlogits,              // [R, V]
    const int R,
    const int V,
    const int kind
) {
    int r = blockIdx.x;
    if (r >= R) return;
    float s = scale[r] * dloss[0];
    float lzq = logzq[r], mqv = mq[r], mpv = mp[r], lzp = logzp[r];
    long long tgt = target_id[r];

    const float* src = logits + (size_t)r * V;
    const float* xtr = target_logits + (size_t)r * V;
    float* dst = dlogits + (size_t)r * V;

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float q = expf(src[v] - lzq);                       // softmax(draft)
        float dterm;
        if (kind == 0) {
            float p = expf(xtr[v] - lzp);                   // softmax(target)
            dterm = q - p;                                  // dKL/dlogit
        } else {
            dterm = q - (v == tgt ? 1.0f : 0.0f);           // dCE/dlogit
        }
        dst[v] = s * dterm;
    }
    (void)mpv;
}
'''

_KINDS = {"kl": 0, "ce": 1}

dflash_module = cp.RawModule(code=dflash_cuda_source, options=('--use_fast_math',))
dflash_fwd = dflash_module.get_function('dflash_loss_fwd')
dflash_bwd = dflash_module.get_function('dflash_loss_bwd')

_BLOCK_DIM = 256


def _check(logits, target_logits, mask, gamma, kind, block_size):
    assert logits.ndim == 3 and logits.dtype == cp.float32, "logits must be float32 [N, B, V]"
    assert target_logits.shape == logits.shape and target_logits.dtype == cp.float32, \
        "target_logits must be float32 [N, B, V]"
    assert mask.shape == logits.shape[:2] and mask.dtype == cp.float32, \
        "loss_mask must be float32 [N, B]"
    N, B, V = logits.shape
    assert block_size is None or 1 <= block_size <= B, "block_size must be in [1, B]"
    assert logits.flags.c_contiguous and target_logits.flags.c_contiguous \
        and mask.flags.c_contiguous, "inputs must be contiguous"
    if not gamma > 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if kind not in _KINDS:
        raise ValueError(f"unknown loss kind {kind!r}, expected one of {sorted(_KINDS)}")


def _forward(logits, target_logits, mask, gamma, sample_from_anchor, kind, block_size):
    """Returns (loss_rows[N,B], scale[N,B], (mq, mp, lzq, lzp, tgt))."""
    _check(logits, target_logits, mask, gamma, kind, block_size)
    N, B, V = logits.shape
    R = N * B
    bs = B if block_size is None else block_size
    denom = (mask.sum(axis=1, dtype=cp.float32) + 1e-5).astype(cp.float32)

    loss_rows = cp.empty((N, B), cp.float32)
    scale = cp.empty((N, B), cp.float32)
    mq = cp.empty(R, cp.float32); mp = cp.empty(R, cp.float32)
    lzq = cp.empty(R, cp.float32); lzp = cp.empty(R, cp.float32)
    tgt = cp.empty(R, cp.int64)

    dflash_fwd((R,), (_BLOCK_DIM,),
               (logits, target_logits, mask, denom, loss_rows, scale,
                mq, mp, lzq, lzp, tgt, N, B, bs, V, np.float32(gamma),
                int(bool(sample_from_anchor)), _KINDS[kind]))
    return loss_rows, scale, (mq, mp, lzq, lzp, tgt)


def fused_dflash_loss(logits_cp: cp.ndarray, target_logits_cp: cp.ndarray,
                      loss_mask=None, gamma: float = 4.0, sample_from_anchor: bool = False,
                      kind: str = "kl", block_size: int | None = None) -> cp.ndarray:
    """Forward-only DFlash loss (eval). logits/target_logits [N,B,V] float32."""
    mask = (cp.ones(logits_cp.shape[:2], cp.float32) if loss_mask is None
            else loss_mask.astype(cp.float32, copy=False))
    loss_rows, _, _ = _forward(logits_cp, target_logits_cp, mask, gamma,
                               sample_from_anchor, kind, block_size)
    N = logits_cp.shape[0]
    return loss_rows.sum() / N          # sum over positions, mean over batch


def dflash_objective(target_logits, **kwargs):
    """Bind the target logits so the objective fits the (logits, targets) hook.

    ``DFlashSpeculator.compute_loss(..., loss_fn=...)`` calls ``loss_fn(logits, targets)``,
    but this objective needs the verifier's logits rather than token ids.
    """
    return lambda logits, _targets: dflash_loss(logits, target_logits, **kwargs)


def _make_torch_fn():
    import torch  # lazy: cupy-only users never import torch

    class DFlashLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, target_logits, loss_mask, gamma, sample_from_anchor,
                    kind, block_size):
            assert logits.is_cuda and logits.dtype == torch.float32, "logits must be float32 cuda"
            assert target_logits.shape == logits.shape
            logits = logits.contiguous()
            target_logits = target_logits.contiguous()
            N, B, V = logits.shape
            bs = B if block_size is None else block_size
            mask = (torch.ones((N, B), dtype=torch.float32, device=logits.device)
                    if loss_mask is None else loss_mask.to(torch.float32).contiguous())
            loss_rows, scale, (mq, mp, lzq, lzp, tgt) = _forward(
                cp.from_dlpack(logits.detach()), cp.from_dlpack(target_logits.detach()),
                cp.from_dlpack(mask), gamma, sample_from_anchor, kind, block_size)

            ctx.save_for_backward(logits.detach(), target_logits.detach())
            ctx.saved_extra = (scale, mq, mp, lzq, lzp, tgt, kind, N, B, V, bs)
            return torch.from_dlpack(loss_rows).sum() / float(N)

        @staticmethod
        def backward(ctx, dloss):
            logits, target_logits = ctx.saved_tensors
            scale, mq, mp, lzq, lzp, tgt, kind, N, B, V, _bs = ctx.saved_extra
            dlogits = cp.empty((N * B, V), cp.float32)
            dflash_bwd((N * B,), (_BLOCK_DIM,),
                       (cp.from_dlpack(logits), cp.from_dlpack(target_logits),
                        scale, mq, mp, lzq, lzp, tgt,
                        cp.from_dlpack(dloss.detach().contiguous()), dlogits,
                        N * B, V, _KINDS[kind]))
            return torch.from_dlpack(dlogits).view(N, B, V), None, None, None, None, None, None

    return DFlashLoss


_dflash_torch = None


def dflash_loss(logits, target_logits, loss_mask=None, gamma: float = 4.0,
                sample_from_anchor: bool = False, kind: str = "kl",
                block_size: int | None = None):
    """DFlash loss for torch tensors, with autograd.

    logits / target_logits: [N,B,V] float32 cuda. loss_mask: [N,B] bool/float or None.
    kind="kl" (speculators default) or "ce" (hard argmax(target) labels).
    """
    global _dflash_torch
    if _dflash_torch is None:
        _dflash_torch = _make_torch_fn()
    return _dflash_torch.apply(logits, target_logits, loss_mask, gamma,
                               sample_from_anchor, kind, block_size)
