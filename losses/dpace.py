import cupy as cp
import numpy as np

# -----------------------------------------------------------------------------
# CUDA: Fused Softmax + Target Gather + Cross Entropy + D-PACE Scan (+ backward)
# Loss matches specforge/core/dflash.py::_dpace_weight:
#   smooth_j = where(mask_j, (1-a)*q_j + a, 1)     (invalid -> no-op in the prefix product)
#   prefix_j = prod_{i<=j} smooth_i
#   suffix_j = sum_{k>=j} prefix_k * mask_k        (== prefix_j * continuation_value_j)
#   loss     = sum_j (w_j * ce_j * mask_j) / N     (divide by batch size, not by weight sum)
# -----------------------------------------------------------------------------
dpace_cuda_source = r'''
#define FLT_TINY 1.1754944e-38f

extern "C" __global__
void dpace_megakernel(
    const float* __restrict__ logits,      // [N, B, V]
    const long long* __restrict__ targets, // [N, B]
    const float* __restrict__ mask,        // [N, B] 0/1 position mask
    float* __restrict__ out_loss,          // [N]     sum_j w_j*ce_j  (un-normalized)
    float* __restrict__ out_lse,           // [N, B]  log-sum-exp per row (kept for backward)
    float* __restrict__ out_scale,         // [N, B]  w_j / N  (dL/dCE per row)
    const int N,                           // batch size
    const int B,                           // block size (e.g., 8, 16, <= 32)
    const int V,                           // vocab size
    const float alpha,                     // smoothing floor (e.g. 0.5f)
    const int loss_type                    // 0=dpace, 1=prefix only, 2=continuation only
) {
    int n = blockIdx.x;                    // 1 thread block per sequence
    if (n >= N) return;

    int tid = threadIdx.x;
    int bsize = blockDim.x;

    extern __shared__ float s_mem[];
    float* s_q    = s_mem;                 // [B] exp(-ce)
    float* s_ce   = s_mem + B;             // [B] cross entropy
    float* s_mask = s_mem + 2 * B;         // [B] position mask

    __shared__ float s_red[32];            // per-warp partials (deterministic reduce)

    for (int j = 0; j < B; ++j) {
        const float* row = logits + ((size_t)n * B + j) * V;
        long long tgt = targets[n * B + j];

        // 1a. row max
        float tmax = -1e30f;
        for (int v = tid; v < V; v += bsize) tmax = fmaxf(tmax, row[v]);
        for (int o = 16; o > 0; o >>= 1)
            tmax = fmaxf(tmax, __shfl_down_sync(0xffffffff, tmax, o));
        if ((tid & 31) == 0) s_red[tid >> 5] = tmax;
        __syncthreads();
        if (tid == 0) {
            float m = -1e30f;
            for (int w = 0; w < (bsize >> 5); ++w) m = fmaxf(m, s_red[w]);
            s_red[0] = m;
        }
        __syncthreads();
        float row_max = s_red[0];
        __syncthreads();                   // s_red reused below

        // 1b. sum of exp(x - max)
        float tsum = 0.0f;
        for (int v = tid; v < V; v += bsize) tsum += expf(row[v] - row_max);
        for (int o = 16; o > 0; o >>= 1)
            tsum += __shfl_down_sync(0xffffffff, tsum, o);
        if ((tid & 31) == 0) s_red[tid >> 5] = tsum;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < (bsize >> 5); ++w) s += s_red[w];
            s_red[0] = s;
        }
        __syncthreads();

        // 1c. thread 0 extracts target prob & CE
        if (tid == 0) {
            float lse = row_max + logf(s_red[0]);
            out_lse[n * B + j] = lse;
            float ce = (tgt >= 0 && tgt < V) ? (lse - row[tgt]) : 0.0f;
            s_ce[j]   = ce;
            s_q[j]    = expf(-ce);
            s_mask[j] = mask[n * B + j];
        }
        __syncthreads();
    }

    // 2. D-PACE weights, loss and dL/dCE scale (thread 0; B is tiny)
    if (tid == 0) {
        float smooth[32], P[32], S[32], w[32];

        for (int j = 0; j < B; ++j) {
            float qt = (1.0f - alpha) * s_q[j] + alpha;
            smooth[j] = (s_mask[j] > 0.0f) ? qt : 1.0f;   // invalid positions are a no-op
        }

        float run = 1.0f;                                  // prefix_j = prod_{i<=j} smooth_i
        for (int j = 0; j < B; ++j) { run *= smooth[j]; P[j] = run; }

        S[B - 1] = P[B - 1] * s_mask[B - 1];               // suffix_j = sum_{k>=j} P_k * mask_k
        for (int j = B - 2; j >= 0; --j)
            S[j] = S[j + 1] + P[j] * s_mask[j];

        float wl = 0.0f;
        for (int j = 0; j < B; ++j) {
            float v = (loss_type == 1) ? P[j]
                    : (loss_type == 2) ? S[j] / fmaxf(P[j], FLT_TINY)
                    :                    S[j];
            w[j] = v * s_mask[j];
            wl += w[j] * s_ce[j];
        }

        out_loss[n] = wl;

        float inv_n = 1.0f / (float)N;                      // weights are detached -> grad via CE only
        for (int j = 0; j < B; ++j) out_scale[n * B + j] = w[j] * inv_n;
    }
}

extern "C" __global__
void dpace_backward(
    const float* __restrict__ logits,      // [N, B, V]
    const long long* __restrict__ targets, // [N, B]
    const float* __restrict__ lse,         // [N, B]
    const float* __restrict__ scale,       // [N, B]
    const float* __restrict__ dloss,       // upstream grad (scalar: loss is a mean over N)
    float* __restrict__ dlogits,           // [N, B, V]
    const int N,
    const int B,
    const int V
) {
    int row_id = blockIdx.x;               // 1 block per (n, j) row
    if (row_id >= N * B) return;

    float l = lse[row_id];
    float s = scale[row_id] * dloss[0];
    long long tgt = targets[row_id];

    const float* src = logits + (size_t)row_id * V;
    float* dst = dlogits + (size_t)row_id * V;

    // dCE/dlogit = softmax - onehot; masked positions have scale == 0
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        dst[v] = s * (expf(src[v] - l) - (v == tgt ? 1.0f : 0.0f));
}
'''

_LOSS_TYPES = {"dpace": 0, "dpace-cumulative-confidence-only": 1, "dpace-continuation-value-only": 2}

dpace_module = cp.RawModule(code=dpace_cuda_source, options=('--use_fast_math',))
dpace_kernel = dpace_module.get_function('dpace_megakernel')
dpace_backward_kernel = dpace_module.get_function('dpace_backward')

_BLOCK_DIM = 256


def _check(logits_cp, targets_cp, mask_cp, alpha, loss_type):
    assert logits_cp.ndim == 3 and logits_cp.dtype == cp.float32, "logits must be float32 [N, B, V]"
    assert targets_cp.shape == logits_cp.shape[:2] and targets_cp.dtype == cp.int64, \
        "targets must be int64 [N, B]"
    assert mask_cp.shape == logits_cp.shape[:2] and mask_cp.dtype == cp.float32, \
        "position mask must be float32 [N, B]"
    N, B, V = logits_cp.shape
    assert 1 <= B <= 32, "kernel configured for block_size <= 32"
    assert logits_cp.flags.c_contiguous and targets_cp.flags.c_contiguous \
        and mask_cp.flags.c_contiguous, "inputs must be contiguous"
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dpace_alpha must be in [0, 1], got {alpha}")
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"unknown loss_type {loss_type!r}")
    # trust boundary: a stray token id silently reads out of bounds
    if bool((targets_cp < 0).any()) or bool((targets_cp >= V).any()):
        raise ValueError(f"targets out of range [0, {V})")


def _forward(logits_cp, targets_cp, mask_cp, alpha, loss_type):
    """Returns cupy (loss[N], lse[N,B], scale[N,B])."""
    _check(logits_cp, targets_cp, mask_cp, alpha, loss_type)
    N, B, V = logits_cp.shape
    loss = cp.empty((N,), cp.float32)
    lse = cp.empty((N, B), cp.float32)
    scale = cp.empty((N, B), cp.float32)
    dpace_kernel((N,), (_BLOCK_DIM,),
                 (logits_cp, targets_cp, mask_cp, loss, lse, scale,
                  N, B, V, np.float32(alpha), _LOSS_TYPES[loss_type]),
                 shared_mem=3 * B * 4)
    return loss, lse, scale


def fused_dpace_loss(logits_cp: cp.ndarray, targets_cp: cp.ndarray, alpha: float = 0.5,
                     position_mask=None, loss_type: str = "dpace") -> cp.ndarray:
    """Forward-only D-PACE loss (eval). logits [N,B,V] float32, targets [N,B] int64."""
    mask = (cp.ones(logits_cp.shape[:2], cp.float32) if position_mask is None
            else position_mask.astype(cp.float32, copy=False))
    return _forward(logits_cp, targets_cp, mask, alpha, loss_type)[0].mean()


def _make_torch_fn():
    import torch  # lazy: cupy-only users never import torch

    class DPACELoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, targets, alpha, position_mask, loss_type):
            assert logits.is_cuda and logits.dtype == torch.float32, "logits must be float32 cuda"
            assert targets.dtype == torch.int64, "targets must be int64"
            logits = logits.contiguous()
            targets = targets.contiguous()
            mask = (torch.ones(logits.shape[:2], dtype=torch.float32, device=logits.device)
                    if position_mask is None else position_mask.to(torch.float32).contiguous())
            loss_n, lse, scale = _forward(cp.from_dlpack(logits.detach()),
                                          cp.from_dlpack(targets.detach()),
                                          cp.from_dlpack(mask), alpha, loss_type)
            ctx.save_for_backward(logits.detach(), targets)
            ctx.lse, ctx.scale = lse, scale
            return torch.from_dlpack(loss_n).mean()

        @staticmethod
        def backward(ctx, dloss):
            logits, targets = ctx.saved_tensors
            N, B, V = logits.shape
            dlogits = cp.empty((N, B, V), cp.float32)
            dpace_backward_kernel((N * B,), (_BLOCK_DIM,),
                                  (cp.from_dlpack(logits), cp.from_dlpack(targets),
                                   ctx.lse, ctx.scale, cp.from_dlpack(dloss.detach().contiguous()),
                                   dlogits, N, B, V))
            return torch.from_dlpack(dlogits), None, None, None, None

    return DPACELoss


_dpace_torch = None


def dpace_loss(logits, targets, alpha: float = 0.5, position_mask=None, loss_type: str = "dpace"):
    """D-PACE loss for torch tensors, with autograd.

    logits [N,B,V] float32 cuda, targets [N,B] int64, position_mask [N,B] bool/float or None.
    Same numbers as specforge/core/dflash.py: sum(w*ce)/N.
    """
    global _dpace_torch
    if _dpace_torch is None:
        _dpace_torch = _make_torch_fn()
    return _dpace_torch.apply(logits, targets, alpha, position_mask, loss_type)
