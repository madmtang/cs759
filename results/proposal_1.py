import math
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_stage1_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    max_logits_ptr,
    sum_exp2_ptr,
    lse_ptr,
    num_tokens,
    num_pages,
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
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    token_idx = pid0
    head_idx = pid1

    if token_idx >= num_tokens or head_idx >= 16:
        return

    offs_dckv = tl.arange(0, BLOCK_DCKV)
    offs_dkpe = tl.arange(0, BLOCK_DKPE)

    qn_ptrs = q_nope_ptr + token_idx * stride_qt + head_idx * stride_qh + offs_dckv * stride_qd
    qp_ptrs = q_pe_ptr + token_idx * stride_qpet + head_idx * stride_qpeh + offs_dkpe * stride_qped

    qn = tl.load(qn_ptrs, mask=offs_dckv < 512, other=0.0).to(tl.float32)
    qp = tl.load(qp_ptrs, mask=offs_dkpe < 64, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0

    for k_start in range(0, 2048, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        idx_ptrs = sparse_indices_ptr + token_idx * stride_sparse_t + offs_k * stride_sparse_k
        idx = tl.load(idx_ptrs, mask=offs_k < 2048, other=-1).to(tl.int32)
        valid = idx >= 0

        page_idx = idx // 64
        page_off = idx % 64

        logits = tl.zeros([BLOCK_K], dtype=tl.float32)

        for d_start in range(0, 512, BLOCK_DCKV):
            d_offs = d_start + tl.arange(0, BLOCK_DCKV)
            qv = tl.load(
                q_nope_ptr + token_idx * stride_qt + head_idx * stride_qh + d_offs * stride_qd,
                mask=d_offs < 512,
                other=0.0,
            ).to(tl.float32)

            kv_ptrs = (
                ckv_cache_ptr
                + page_idx[:, None] * stride_ckv_p
                + page_off[:, None] * stride_ckv_s
                + d_offs[None, :] * stride_ckv_d
            )
            kv = tl.load(
                kv_ptrs,
                mask=valid[:, None] & (d_offs[None, :] < 512),
                other=0.0,
            ).to(tl.float32)
            logits += tl.sum(kv * qv[None, :], axis=1)

        for d_start in range(0, 64, BLOCK_DKPE):
            d_offs = d_start + tl.arange(0, BLOCK_DKPE)
            qv = tl.load(
                q_pe_ptr + token_idx * stride_qpet + head_idx * stride_qpeh + d_offs * stride_qped,
                mask=d_offs < 64,
                other=0.0,
            ).to(tl.float32)

            kv_ptrs = (
                kpe_cache_ptr
                + page_idx[:, None] * stride_kpe_p
                + page_off[:, None] * stride_kpe_s
                + d_offs[None, :] * stride_kpe_d
            )
            kv = tl.load(
                kv_ptrs,
                mask=valid[:, None] & (d_offs[None, :] < 64),
                other=0.0,
            ).to(tl.float32)
            logits += tl.sum(kv * qv[None, :], axis=1)

        logits = logits * sm_scale
        logits = tl.where(valid, logits, -float("inf"))

        m_blk = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_blk)

        exp_old = tl.exp2(m_i * 1.4426950408889634 - m_new * 1.4426950408889634) * l_i
        exp_blk = tl.sum(tl.exp2(logits * 1.4426950408889634 - m_new * 1.4426950408889634), axis=0)
        l_i = exp_old + exp_blk
        m_i = m_new

    has_valid = l_i > 0.0
    lse_val = tl.where(has_valid, m_i * 1.4426950408889634 + tl.log2(l_i), -float("inf"))

    tl.store(max_logits_ptr + token_idx * stride_m_t + head_idx * stride_m_h, m_i)
    tl.store(sum_exp2_ptr + token_idx * stride_s_t + head_idx * stride_s_h, l_i)
    tl.store(lse_ptr + token_idx * stride_lse_t + head_idx * stride_lse_h, lse_val)


@triton.jit
def _sparse_attn_stage2_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    max_logits_ptr,
    sum_exp2_ptr,
    output_ptr,
    num_tokens,
    num_pages,
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
    BLOCK_DV: tl.constexpr,
    BLOCK_DKPE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    token_idx = pid0
    head_idx = pid1
    dv_block = pid2

    if token_idx >= num_tokens or head_idx >= 16:
        return

    dv_start = dv_block * BLOCK_DV
    offs_dv = dv_start + tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < 512

    m_i = tl.load(max_logits_ptr + token_idx * stride_m_t + head_idx * stride_m_h)
    l_i = tl.load(sum_exp2_ptr + token_idx * stride_s_t + head_idx * stride_s_h)

    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if l_i > 0.0:
        for k_start in range(0, 2048, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            idx_ptrs = sparse_indices_ptr + token_idx * stride_sparse_t + offs_k * stride_sparse_k
            idx = tl.load(idx_ptrs, mask=offs_k < 2048, other=-1).to(tl.int32)
            valid = idx >= 0

            page_idx = idx // 64
            page_off = idx % 64

            logits = tl.zeros([BLOCK_K], dtype=tl.float32)

            for d_start in range(0, 512, BLOCK_DV):
                d_offs = d_start + tl.arange(0, BLOCK_DV)
                qv = tl.load(
                    q_nope_ptr + token_idx * stride_qt + head_idx * stride_qh + d_offs * stride_qd,
                    mask=d_offs < 512,
                    other=0.0,
                ).to(tl.float32)

                kv_ptrs = (
                    ckv_cache_ptr
                    + page_idx[:, None] * stride_ckv_p
                    + page_off[:, None] * stride_ckv_s
                    + d_offs[None, :] * stride_ckv_d
                )
                kv = tl.load(
                    kv_ptrs,
                    mask=valid[:, None] & (d_offs[None, :] < 512),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(kv * qv[None, :], axis=1)

            for d_start in range(0, 64, BLOCK_DKPE):
                d_offs = d_start + tl.arange(0, BLOCK_DKPE)
                qv = tl.load(
                    q_pe_ptr + token_idx * stride_qpet + head_idx * stride_qpeh + d_offs * stride_qped,
                    mask=d_offs < 64,
                    other=0.0,
                ).to(tl.float32)

                kv_ptrs = (
                    kpe_cache_ptr
                    + page_idx[:, None] * stride_kpe_p
                    + page_off[:, None] * stride_kpe_s
                    + d_offs[None, :] * stride_kpe_d
                )
                kv = tl.load(
                    kv_ptrs,
                    mask=valid[:, None] & (d_offs[None, :] < 64),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(kv * qv[None, :], axis=1)

            logits = logits * sm_scale
            logits = tl.where(valid, logits, -float("inf"))
            p = tl.exp2(logits * 1.4426950408889634 - m_i * 1.4426950408889634) / l_i
            p = tl.where(valid, p, 0.0)

            v_ptrs = (
                ckv_cache_ptr
                + page_idx[:, None] * stride_ckv_p
                + page_off[:, None] * stride_ckv_s
                + offs_dv[None, :] * stride_ckv_d
            )
            v = tl.load(
                v_ptrs,
                mask=valid[:, None] & dv_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(p[:, None] * v, axis=0)

    out_ptrs = output_ptr + token_idx * stride_out_t + head_idx * stride_out_h + offs_dv * stride_out_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=dv_mask)


def _move_to_cuda_if_needed(x):
    if not torch.is_tensor(x):
        return x, None
    orig_device = x.device
    if x.is_cuda:
        return x, orig_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU execution is required for Triton kernels.")
    return x.cuda(), orig_device


def _restore_to_device(x, device):
    if device is None:
        return x
    if x.device == device:
        return x
    return x.to(device)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    orig_devices = {}

    q_nope, orig_devices["q_nope"] = _move_to_cuda_if_needed(q_nope)
    q_pe, orig_devices["q_pe"] = _move_to_cuda_if_needed(q_pe)
    ckv_cache, orig_devices["ckv_cache"] = _move_to_cuda_if_needed(ckv_cache)
    kpe_cache, orig_devices["kpe_cache"] = _move_to_cuda_if_needed(kpe_cache)
    sparse_indices, orig_devices["sparse_indices"] = _move_to_cuda_if_needed(sparse_indices)

    if not torch.is_tensor(sm_scale):
        sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=q_nope.device)
    else:
        sm_scale, orig_devices["sm_scale"] = _move_to_cuda_if_needed(sm_scale)
        sm_scale = sm_scale.to(torch.float32)

    if not q_nope.is_cuda or not q_pe.is_cuda or not ckv_cache.is_cuda or not kpe_cache.is_cuda or not sparse_indices.is_cuda:
        raise RuntimeError("All tensors must be on CUDA for Triton execution.")

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert sparse_indices.shape[0] == num_tokens
    assert ckv_cache.shape[1] == page_size

    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    sparse_indices = sparse_indices.contiguous()

    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), device=q_nope.device, dtype=torch.bfloat16)
    lse = torch.empty((num_tokens, num_qo_heads), device=q_nope.device, dtype=torch.float32)
    max_logits = torch.empty((num_tokens, num_qo_heads), device=q_nope.device, dtype=torch.float32)
    sum_exp2 = torch.empty((num_tokens, num_qo_heads), device=q_nope.device, dtype=torch.float32)

    grid1 = (num_tokens, num_qo_heads)
    _sparse_attn_stage1_kernel[grid1](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        max_logits,
        sum_exp2,
        lse,
        num_tokens,
        num_pages,
        float(sm_scale.item()),
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
        BLOCK_K=32,
        BLOCK_DCKV=32,
        BLOCK_DKPE=32,
        num_warps=8,
        num_stages=2,
    )

    grid2 = (num_tokens, num_qo_heads, 8)
    _sparse_attn_stage2_kernel[grid2](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        max_logits,
        sum_exp2,
        output,
        num_tokens,
        num_pages,
        float(sm_scale.item()),
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
        BLOCK_K=32,
        BLOCK_DV=64,
        BLOCK_DKPE=32,
        num_warps=8,
        num_stages=2,
    )

    out_device = orig_devices["q_nope"]
    output = _restore_to_device(output, out_device)
    lse = _restore_to_device(lse, out_device)
    return output, lse