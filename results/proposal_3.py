import math
import torch
import triton
import triton.language as tl


@triton.jit
def _stage1_lse_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    max_logits_ptr,
    sum_exp2_ptr,
    lse_ptr,
    num_tokens,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_qpet,
    stride_qpeh,
    stride_qped,
    stride_ckv_p,
    stride_ckv_s,
    stride_ckv_d,
    stride_kpe_p,
    stride_kpe_s,
    stride_kpe_d,
    stride_sparse_t,
    stride_sparse_k,
    stride_m_t,
    stride_m_h,
    stride_s_t,
    stride_s_h,
    stride_lse_t,
    stride_lse_h,
    BLOCK_K: tl.constexpr,
    BLOCK_DCKV: tl.constexpr,
    BLOCK_DKPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= num_tokens:
        return

    log2e = 1.4426950408889634

    m_i = -float("inf")
    l_i = 0.0

    for k0 in range(0, 2048, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        idx_ptrs = sparse_indices_ptr + pid_t * stride_sparse_t + offs_k * stride_sparse_k
        idx = tl.load(idx_ptrs, mask=offs_k < 2048, other=-1).to(tl.int32)
        valid = idx >= 0

        page = idx // 64
        slot = idx % 64

        logits = tl.zeros([BLOCK_K], dtype=tl.float32)

        for d0 in range(0, 512, BLOCK_DCKV):
            offs_d = d0 + tl.arange(0, BLOCK_DCKV)
            q_ptrs = q_nope_ptr + pid_t * stride_qt + pid_h * stride_qh + offs_d * stride_qd
            q = tl.load(q_ptrs, mask=offs_d < 512, other=0.0).to(tl.float32)

            k_ptrs = (
                ckv_cache_ptr
                + page[:, None] * stride_ckv_p
                + slot[:, None] * stride_ckv_s
                + offs_d[None, :] * stride_ckv_d
            )
            k = tl.load(
                k_ptrs,
                mask=valid[:, None] & (offs_d[None, :] < 512),
                other=0.0,
            ).to(tl.float32)
            logits += tl.sum(k * q[None, :], axis=1)

        offs_dpe = tl.arange(0, BLOCK_DKPE)
        qpe_ptrs = q_pe_ptr + pid_t * stride_qpet + pid_h * stride_qpeh + offs_dpe * stride_qped
        qpe = tl.load(qpe_ptrs, mask=offs_dpe < 64, other=0.0).to(tl.float32)
        kpe_ptrs = (
            kpe_cache_ptr
            + page[:, None] * stride_kpe_p
            + slot[:, None] * stride_kpe_s
            + offs_dpe[None, :] * stride_kpe_d
        )
        kpe = tl.load(
            kpe_ptrs,
            mask=valid[:, None] & (offs_dpe[None, :] < 64),
            other=0.0,
        ).to(tl.float32)
        logits += tl.sum(kpe * qpe[None, :], axis=1)

        logits = logits * sm_scale
        logits = tl.where(valid, logits, -float("inf"))

        m_blk = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp2((m_i - m_new) * log2e)
        p = tl.exp2((logits - m_new) * log2e)
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    lse_val = tl.where(l_i > 0.0, m_i * log2e + tl.log2(l_i), -float("inf"))

    tl.store(max_logits_ptr + pid_t * stride_m_t + pid_h * stride_m_h, m_i)
    tl.store(sum_exp2_ptr + pid_t * stride_s_t + pid_h * stride_s_h, l_i)
    tl.store(lse_ptr + pid_t * stride_lse_t + pid_h * stride_lse_h, lse_val)


@triton.jit
def _stage2_out_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    max_logits_ptr,
    sum_exp2_ptr,
    output_ptr,
    num_tokens,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_qpet,
    stride_qpeh,
    stride_qped,
    stride_ckv_p,
    stride_ckv_s,
    stride_ckv_d,
    stride_kpe_p,
    stride_kpe_s,
    stride_kpe_d,
    stride_sparse_t,
    stride_sparse_k,
    stride_m_t,
    stride_m_h,
    stride_s_t,
    stride_s_h,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    BLOCK_K: tl.constexpr,
    BLOCK_DCKV: tl.constexpr,
    BLOCK_DKPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)

    if pid_t >= num_tokens:
        return

    log2e = 1.4426950408889634

    offs_v = pid_v * BLOCK_DV + tl.arange(0, BLOCK_DV)
    mask_v = offs_v < 512

    m_i = tl.load(max_logits_ptr + pid_t * stride_m_t + pid_h * stride_m_h)
    l_i = tl.load(sum_exp2_ptr + pid_t * stride_s_t + pid_h * stride_s_h)

    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if l_i > 0.0:
        for k0 in range(0, 2048, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            idx_ptrs = sparse_indices_ptr + pid_t * stride_sparse_t + offs_k * stride_sparse_k
            idx = tl.load(idx_ptrs, mask=offs_k < 2048, other=-1).to(tl.int32)
            valid = idx >= 0

            page = idx // 64
            slot = idx % 64

            logits = tl.zeros([BLOCK_K], dtype=tl.float32)

            for d0 in range(0, 512, BLOCK_DCKV):
                offs_d = d0 + tl.arange(0, BLOCK_DCKV)
                q_ptrs = q_nope_ptr + pid_t * stride_qt + pid_h * stride_qh + offs_d * stride_qd
                q = tl.load(q_ptrs, mask=offs_d < 512, other=0.0).to(tl.float32)

                k_ptrs = (
                    ckv_cache_ptr
                    + page[:, None] * stride_ckv_p
                    + slot[:, None] * stride_ckv_s
                    + offs_d[None, :] * stride_ckv_d
                )
                k = tl.load(
                    k_ptrs,
                    mask=valid[:, None] & (offs_d[None, :] < 512),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(k * q[None, :], axis=1)

            offs_dpe = tl.arange(0, BLOCK_DKPE)
            qpe_ptrs = q_pe_ptr + pid_t * stride_qpet + pid_h * stride_qpeh + offs_dpe * stride_qped
            qpe = tl.load(qpe_ptrs, mask=offs_dpe < 64, other=0.0).to(tl.float32)
            kpe_ptrs = (
                kpe_cache_ptr
                + page[:, None] * stride_kpe_p
                + slot[:, None] * stride_kpe_s
                + offs_dpe[None, :] * stride_kpe_d
            )
            kpe = tl.load(
                kpe_ptrs,
                mask=valid[:, None] & (offs_dpe[None, :] < 64),
                other=0.0,
            ).to(tl.float32)
            logits += tl.sum(kpe * qpe[None, :], axis=1)

            logits = logits * sm_scale
            w = tl.exp2((logits - m_i) * log2e) / l_i
            w = tl.where(valid, w, 0.0)

            v_ptrs = (
                ckv_cache_ptr
                + page[:, None] * stride_ckv_p
                + slot[:, None] * stride_ckv_s
                + offs_v[None, :] * stride_ckv_d
            )
            v = tl.load(
                v_ptrs,
                mask=valid[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w[:, None] * v, axis=0)

    out_ptrs = output_ptr + pid_t * stride_out_t + pid_h * stride_out_h + offs_v * stride_out_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_v)


def _move_tensor_to_cuda(x, name):
    if not torch.is_tensor(x):
        return x, None
    orig_device = x.device
    if x.is_cuda:
        return x, orig_device
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available but tensor '{name}' requires GPU execution.")
    return x.cuda(), orig_device


def _restore_tensor_to_device(x, device):
    if device is None:
        return x
    if x.device == device:
        return x
    return x.to(device)


def run(*args, **kwargs):
    names = ["q_nope", "q_pe", "ckv_cache", "kpe_cache", "sparse_indices", "sm_scale"]
    vals = {}
    for i, name in enumerate(names):
        if i < len(args):
            vals[name] = args[i]
        elif name in kwargs:
            vals[name] = kwargs[name]
        else:
            raise TypeError(f"Missing required argument: {name}")

    q_nope = vals["q_nope"]
    q_pe = vals["q_pe"]
    ckv_cache = vals["ckv_cache"]
    kpe_cache = vals["kpe_cache"]
    sparse_indices = vals["sparse_indices"]
    sm_scale = vals["sm_scale"]

    orig_devices = {}

    q_nope, orig_devices["q_nope"] = _move_tensor_to_cuda(q_nope, "q_nope")
    q_pe, orig_devices["q_pe"] = _move_tensor_to_cuda(q_pe, "q_pe")
    ckv_cache, orig_devices["ckv_cache"] = _move_tensor_to_cuda(ckv_cache, "ckv_cache")
    kpe_cache, orig_devices["kpe_cache"] = _move_tensor_to_cuda(kpe_cache, "kpe_cache")
    sparse_indices, orig_devices["sparse_indices"] = _move_tensor_to_cuda(sparse_indices, "sparse_indices")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton execution.")

    if torch.is_tensor(sm_scale):
        sm_scale, orig_devices["sm_scale"] = _move_tensor_to_cuda(sm_scale, "sm_scale")
        sm_scale_val = float(sm_scale.to(torch.float32).item())
    else:
        sm_scale_val = float(sm_scale)
        orig_devices["sm_scale"] = None

    if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda):
        raise RuntimeError("All tensor inputs must be CUDA tensors after device management.")

    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    sparse_indices = sparse_indices.contiguous()

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, ckv_dim = ckv_cache.shape
    kpe_pages, kpe_page_size, kpe_dim = kpe_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert ckv_dim == 512
    assert page_size == 64
    assert topk == 2048
    assert sparse_indices.shape[0] == num_tokens
    assert kpe_pages == num_pages
    assert kpe_page_size == 64
    assert kpe_dim == 64

    output = torch.empty((num_tokens, 16, 512), device=q_nope.device, dtype=torch.bfloat16)
    lse = torch.empty((num_tokens, 16), device=q_nope.device, dtype=torch.float32)
    max_logits = torch.empty((num_tokens, 16), device=q_nope.device, dtype=torch.float32)
    sum_exp2 = torch.empty((num_tokens, 16), device=q_nope.device, dtype=torch.float32)

    grid1 = (num_tokens, 16)
    _stage1_lse_kernel[grid1](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        max_logits,
        sum_exp2,
        lse,
        num_tokens,
        sm_scale_val,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        ckv_cache.stride(0),
        ckv_cache.stride(1),
        ckv_cache.stride(2),
        kpe_cache.stride(0),
        kpe_cache.stride(1),
        kpe_cache.stride(2),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        max_logits.stride(0),
        max_logits.stride(1),
        sum_exp2.stride(0),
        sum_exp2.stride(1),
        lse.stride(0),
        lse.stride(1),
        BLOCK_K=128,
        BLOCK_DCKV=64,
        BLOCK_DKPE=64,
        num_warps=8,
        num_stages=2,
    )

    grid2 = (num_tokens, 16, 8)
    _stage2_out_kernel[grid2](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        max_logits,
        sum_exp2,
        output,
        num_tokens,
        sm_scale_val,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        ckv_cache.stride(0),
        ckv_cache.stride(1),
        ckv_cache.stride(2),
        kpe_cache.stride(0),
        kpe_cache.stride(1),
        kpe_cache.stride(2),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        max_logits.stride(0),
        max_logits.stride(1),
        sum_exp2.stride(0),
        sum_exp2.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_K=128,
        BLOCK_DCKV=64,
        BLOCK_DKPE=64,
        BLOCK_DV=64,
        num_warps=8,
        num_stages=2,
    )

    out_device = orig_devices["q_nope"]
    output = _restore_tensor_to_device(output, out_device)
    lse = _restore_tensor_to_device(lse, out_device)
    return output, lse