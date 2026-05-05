import math
import torch
import triton
import triton.language as tl


@triton.jit
def _stage1_logits_lse_kernel(
    q_nope_ptr,          # [T, H, 512] bf16
    q_pe_ptr,            # [T, H, 64] bf16
    ckv_ptr,             # [P, 64, 512] bf16
    kpe_ptr,             # [P, 64, 64] bf16
    sparse_idx_ptr,      # [T, 2048] int32
    logits_ptr,          # [T, H, 2048] fp32
    lse_ptr,             # [T, H] fp32, base-2
    sm_scale_ptr,        # scalar fp32
    num_tokens,
    stride_qt, stride_qh, stride_qd,
    stride_qpet, stride_qpeh, stride_qped,
    stride_ckvp, stride_ckvs, stride_ckvd,
    stride_kpep, stride_kpes, stride_kped,
    stride_sit, stride_sik,
    stride_lt, stride_lh, stride_lk,
    stride_lset, stride_lseh,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PE: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % 16
    tok = pid // 16
    if tok >= num_tokens:
        return

    offs_k = tl.arange(0, BLOCK_K)
    idx = tl.load(sparse_idx_ptr + tok * stride_sit + offs_k * stride_sik)
    valid = idx >= 0

    safe_idx = tl.where(valid, idx, 0)
    page = safe_idx // 64
    off = safe_idx & 63

    qn_base = q_nope_ptr + tok * stride_qt + head * stride_qh
    qp_base = q_pe_ptr + tok * stride_qpet + head * stride_qpeh

    logits = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for d0 in range(0, 512, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        qv = tl.load(qn_base + offs_d * stride_qd).to(tl.float32)
        kv_ptrs = ckv_ptr + page[:, None] * stride_ckvp + off[:, None] * stride_ckvs + offs_d[None, :] * stride_ckvd
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        logits += tl.sum(kv * qv[None, :], axis=1)

    for d0 in range(0, 64, BLOCK_PE):
        offs_d = d0 + tl.arange(0, BLOCK_PE)
        qv = tl.load(qp_base + offs_d * stride_qped).to(tl.float32)
        kv_ptrs = kpe_ptr + page[:, None] * stride_kpep + off[:, None] * stride_kpes + offs_d[None, :] * stride_kped
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        logits += tl.sum(kv * qv[None, :], axis=1)

    sm_scale = tl.load(sm_scale_ptr)
    logits = logits * sm_scale
    neg_inf = -float("inf")
    logits = tl.where(valid, logits, neg_inf)

    tl.store(logits_ptr + tok * stride_lt + head * stride_lh + offs_k * stride_lk, logits)

    has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
    m = tl.max(logits, axis=0)
    log2e = 1.4426950408889634
    exp2v = tl.math.exp2((logits - m) * log2e)
    s = tl.sum(exp2v, axis=0)
    lse = m * log2e + tl.math.log2(s)
    lse = tl.where(has_valid, lse, neg_inf)
    tl.store(lse_ptr + tok * stride_lset + head * stride_lseh, lse)


@triton.jit
def _stage2_out_kernel(
    logits_ptr,          # [T, H, 2048] fp32
    lse_ptr,             # [T, H] fp32 base-2
    ckv_ptr,             # [P, 64, 512] bf16
    sparse_idx_ptr,      # [T, 2048] int32
    out_ptr,             # [T, H, 512] bf16
    num_tokens,
    stride_lt, stride_lh, stride_lk,
    stride_lset, stride_lseh,
    stride_ckvp, stride_ckvs, stride_ckvd,
    stride_sit, stride_sik,
    stride_ot, stride_oh, stride_od,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_dblocks = 512 // BLOCK_D
    dblock = pid % num_dblocks
    x = pid // num_dblocks
    head = x % 16
    tok = x // 16
    if tok >= num_tokens:
        return

    offs_d = dblock * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    lse = tl.load(lse_ptr + tok * stride_lset + head * stride_lseh)
    has_valid = lse != -float("inf")

    for k0 in range(0, 2048, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        idx = tl.load(sparse_idx_ptr + tok * stride_sit + offs_k * stride_sik)
        valid = (idx >= 0) & has_valid

        safe_idx = tl.where(valid, idx, 0)
        page = safe_idx // 64
        off = safe_idx & 63

        logits = tl.load(logits_ptr + tok * stride_lt + head * stride_lh + offs_k * stride_lk)
        w = tl.math.exp2(logits * 1.4426950408889634 - lse)
        w = tl.where(valid, w, 0.0)

        kv_ptrs = ckv_ptr + page[:, None] * stride_ckvp + off[:, None] * stride_ckvs + offs_d[None, :] * stride_ckvd
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(kv * w[:, None], axis=0)

    tl.store(out_ptr + tok * stride_ot + head * stride_oh + offs_d * stride_od, acc.to(tl.bfloat16))


def _get_arg(name, idx, args, kwargs):
    if idx < len(args):
        return args[idx]
    if name in kwargs:
        return kwargs[name]
    raise TypeError(f"Missing required argument: {name}")


def _to_cuda_tensor(x, dtype, device):
    if torch.is_tensor(x):
        t = x
        if t.device.type == "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this Triton kernel, but CUDA is not available.")
            t = t.cuda(device=device)
        elif t.device.type != "cuda":
            raise RuntimeError(f"Unsupported tensor device: {t.device}")
        elif t.device != device:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t.contiguous()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton kernel, but CUDA is not available.")
    return torch.tensor(x, device=device, dtype=dtype).contiguous()


def run(*args, **kwargs):
    q_nope = _get_arg("q_nope", 0, args, kwargs)
    q_pe = _get_arg("q_pe", 1, args, kwargs)
    ckv_cache = _get_arg("ckv_cache", 2, args, kwargs)
    kpe_cache = _get_arg("kpe_cache", 3, args, kwargs)
    sparse_indices = _get_arg("sparse_indices", 4, args, kwargs)
    sm_scale = _get_arg("sm_scale", 5, args, kwargs)

    tensor_inputs = [x for x in (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices) if torch.is_tensor(x)]
    orig_device = tensor_inputs[0].device if tensor_inputs else torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton kernel, but CUDA is not available.")

    target_device = None
    for x in tensor_inputs:
        if x.device.type == "cuda":
            target_device = x.device
            break
    if target_device is None:
        target_device = torch.device("cuda")

    q_nope_gpu = _to_cuda_tensor(q_nope, torch.bfloat16, target_device)
    q_pe_gpu = _to_cuda_tensor(q_pe, torch.bfloat16, target_device)
    ckv_cache_gpu = _to_cuda_tensor(ckv_cache, torch.bfloat16, target_device)
    kpe_cache_gpu = _to_cuda_tensor(kpe_cache, torch.bfloat16, target_device)
    sparse_indices_gpu = _to_cuda_tensor(sparse_indices, torch.int32, target_device)
    sm_scale_gpu = _to_cuda_tensor(sm_scale, torch.float32, target_device)

    if q_nope_gpu.ndim != 3:
        raise ValueError(f"q_nope must be 3D, got shape {tuple(q_nope_gpu.shape)}")
    if q_pe_gpu.ndim != 3:
        raise ValueError(f"q_pe must be 3D, got shape {tuple(q_pe_gpu.shape)}")
    if ckv_cache_gpu.ndim != 3:
        raise ValueError(f"ckv_cache must be 3D, got shape {tuple(ckv_cache_gpu.shape)}")
    if kpe_cache_gpu.ndim != 3:
        raise ValueError(f"kpe_cache must be 3D, got shape {tuple(kpe_cache_gpu.shape)}")
    if sparse_indices_gpu.ndim != 2:
        raise ValueError(f"sparse_indices must be 2D, got shape {tuple(sparse_indices_gpu.shape)}")

    num_tokens, num_qo_heads, head_dim_ckv = q_nope_gpu.shape
    t2, h2, head_dim_kpe = q_pe_gpu.shape
    num_pages, page_size, ckv_dim = ckv_cache_gpu.shape
    p2, ps2, kpe_dim = kpe_cache_gpu.shape
    t3, topk = sparse_indices_gpu.shape

    if num_qo_heads != 16:
        raise ValueError(f"Expected num_qo_heads == 16, got {num_qo_heads}")
    if head_dim_ckv != 512:
        raise ValueError(f"Expected head_dim_ckv == 512, got {head_dim_ckv}")
    if head_dim_kpe != 64:
        raise ValueError(f"Expected head_dim_kpe == 64, got {head_dim_kpe}")
    if page_size != 64:
        raise ValueError(f"Expected page_size == 64, got {page_size}")
    if topk != 2048:
        raise ValueError(f"Expected topk == 2048, got {topk}")
    if t2 != num_tokens or h2 != num_qo_heads:
        raise ValueError("q_pe shape mismatch with q_nope")
    if t3 != num_tokens:
        raise ValueError("sparse_indices.shape[0] must equal num_tokens")
    if ckv_dim != 512:
        raise ValueError(f"Expected ckv_cache.shape[-1] == 512, got {ckv_dim}")
    if p2 != num_pages or ps2 != page_size or kpe_dim != 64:
        raise ValueError("kpe_cache shape mismatch")
    if ckv_cache_gpu.shape[1] != 64:
        raise ValueError("ckv_cache.shape[1] must equal page_size=64")

    logits = torch.empty((num_tokens, num_qo_heads, topk), device=target_device, dtype=torch.float32)
    lse = torch.empty((num_tokens, num_qo_heads), device=target_device, dtype=torch.float32)
    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), device=target_device, dtype=torch.bfloat16)

    grid1 = (num_tokens * num_qo_heads,)
    _stage1_logits_lse_kernel[grid1](
        q_nope_gpu,
        q_pe_gpu,
        ckv_cache_gpu,
        kpe_cache_gpu,
        sparse_indices_gpu,
        logits,
        lse,
        sm_scale_gpu,
        num_tokens,
        q_nope_gpu.stride(0), q_nope_gpu.stride(1), q_nope_gpu.stride(2),
        q_pe_gpu.stride(0), q_pe_gpu.stride(1), q_pe_gpu.stride(2),
        ckv_cache_gpu.stride(0), ckv_cache_gpu.stride(1), ckv_cache_gpu.stride(2),
        kpe_cache_gpu.stride(0), kpe_cache_gpu.stride(1), kpe_cache_gpu.stride(2),
        sparse_indices_gpu.stride(0), sparse_indices_gpu.stride(1),
        logits.stride(0), logits.stride(1), logits.stride(2),
        lse.stride(0), lse.stride(1),
        BLOCK_K=2048,
        BLOCK_D=64,
        BLOCK_PE=32,
        num_warps=8,
        num_stages=3,
    )

    grid2 = (num_tokens * num_qo_heads * (head_dim_ckv // 64),)
    _stage2_out_kernel[grid2](
        logits,
        lse,
        ckv_cache_gpu,
        sparse_indices_gpu,
        output,
        num_tokens,
        logits.stride(0), logits.stride(1), logits.stride(2),
        lse.stride(0), lse.stride(1),
        ckv_cache_gpu.stride(0), ckv_cache_gpu.stride(1), ckv_cache_gpu.stride(2),
        sparse_indices_gpu.stride(0), sparse_indices_gpu.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_K=128,
        BLOCK_D=64,
        num_warps=4,
        num_stages=3,
    )

    if orig_device.type == "cuda":
        return output.to(orig_device), lse.to(orig_device)
    return output.cpu(), lse.cpu()