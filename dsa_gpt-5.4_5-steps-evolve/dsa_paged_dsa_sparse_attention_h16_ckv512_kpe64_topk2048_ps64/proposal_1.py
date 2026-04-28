import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_sparse_attention_stage1_kernel(
    q_nope_ptr,          # [T, H, DCKV]
    q_pe_ptr,            # [T, H, DKPE]
    ckv_cache_ptr,       # [P, PS, DCKV]
    kpe_cache_ptr,       # [P, PS, DKPE]
    sparse_indices_ptr,  # [T, TOPK]
    max_logits_ptr,      # [T, H]
    sum_exp_ptr,         # [T, H]
    T,
    NUM_PAGES,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_s, stride_ckv_d,
    stride_kpe_p, stride_kpe_s, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_m_t, stride_m_h,
    stride_s_t, stride_s_h,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    DCKV: tl.constexpr,
    DKPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= T:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < 16

    offs_dckv = tl.arange(0, DCKV)
    offs_dkpe = tl.arange(0, DKPE)

    qn_ptrs = q_nope_ptr + pid_t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_dckv[None, :] * stride_qn_d
    qp_ptrs = q_pe_ptr + pid_t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_dkpe[None, :] * stride_qp_d

    qn = tl.load(qn_ptrs, mask=mask_h[:, None], other=0.0).to(tl.float32)
    qp = tl.load(qp_ptrs, mask=mask_h[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)

    for k0 in range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        idx_ptrs = sparse_indices_ptr + pid_t * stride_idx_t + offs_k * stride_idx_k
        idx = tl.load(idx_ptrs, mask=offs_k < TOPK, other=-1).to(tl.int32)

        valid = idx >= 0
        page = idx // PAGE_SIZE
        slot = idx % PAGE_SIZE
        valid = valid & (page >= 0) & (page < NUM_PAGES)

        logits = tl.full([BLOCK_H, BLOCK_K], -float("inf"), tl.float32)

        for dk0 in range(0, DCKV, 64):
            d_offs = dk0 + tl.arange(0, 64)
            k_ptrs = (
                ckv_cache_ptr
                + page[None, :] * stride_ckv_p
                + slot[None, :] * stride_ckv_s
                + d_offs[:, None] * stride_ckv_d
            )
            k = tl.load(k_ptrs, mask=valid[None, :] & (d_offs[:, None] < DCKV), other=0.0).to(tl.float32)
            q = tl.load(
                q_nope_ptr + pid_t * stride_qn_t + offs_h[:, None] * stride_qn_h + d_offs[None, :] * stride_qn_d,
                mask=mask_h[:, None] & (d_offs[None, :] < DCKV),
                other=0.0,
            ).to(tl.float32)
            logits += tl.dot(q, tl.trans(k))

        for dp0 in range(0, DKPE, 32):
            p_offs = dp0 + tl.arange(0, 32)
            kp_ptrs = (
                kpe_cache_ptr
                + page[None, :] * stride_kpe_p
                + slot[None, :] * stride_kpe_s
                + p_offs[:, None] * stride_kpe_d
            )
            kp = tl.load(kp_ptrs, mask=valid[None, :] & (p_offs[:, None] < DKPE), other=0.0).to(tl.float32)
            q = tl.load(
                q_pe_ptr + pid_t * stride_qp_t + offs_h[:, None] * stride_qp_h + p_offs[None, :] * stride_qp_d,
                mask=mask_h[:, None] & (p_offs[None, :] < DKPE),
                other=0.0,
            ).to(tl.float32)
            logits += tl.dot(q, tl.trans(kp))

        logits = logits * sm_scale
        logits = tl.where(valid[None, :], logits, -float("inf"))

        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((logits - m_new[:, None]) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out_m_ptrs = max_logits_ptr + pid_t * stride_m_t + offs_h * stride_m_h
    out_s_ptrs = sum_exp_ptr + pid_t * stride_s_t + offs_h * stride_s_h
    tl.store(out_m_ptrs, m_i, mask=mask_h)
    tl.store(out_s_ptrs, l_i, mask=mask_h)


@triton.jit
def _dsa_sparse_attention_stage2_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    max_logits_ptr,
    sum_exp_ptr,
    output_ptr,
    T,
    NUM_PAGES,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_s, stride_ckv_d,
    stride_kpe_p, stride_kpe_s, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_m_t, stride_m_h,
    stride_s_t, stride_s_h,
    stride_o_t, stride_o_h, stride_o_d,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    DCKV: tl.constexpr,
    DKPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_hd = tl.program_id(1)

    if pid_t >= T:
        return

    num_h_blocks = 16 // BLOCK_H
    pid_h = pid_hd % num_h_blocks
    pid_d = pid_hd // num_h_blocks

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_h = offs_h < 16
    mask_d = offs_d < DCKV

    m_i = tl.load(max_logits_ptr + pid_t * stride_m_t + offs_h * stride_m_h, mask=mask_h, other=-float("inf"))
    l_i = tl.load(sum_exp_ptr + pid_t * stride_s_t + offs_h * stride_s_h, mask=mask_h, other=1.0)

    acc = tl.zeros([BLOCK_H, BLOCK_D], tl.float32)

    for k0 in range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        idx_ptrs = sparse_indices_ptr + pid_t * stride_idx_t + offs_k * stride_idx_k
        idx = tl.load(idx_ptrs, mask=offs_k < TOPK, other=-1).to(tl.int32)

        valid = idx >= 0
        page = idx // PAGE_SIZE
        slot = idx % PAGE_SIZE
        valid = valid & (page >= 0) & (page < NUM_PAGES)

        logits = tl.full([BLOCK_H, BLOCK_K], -float("inf"), tl.float32)

        for dk0 in range(0, DCKV, 64):
            d_offs = dk0 + tl.arange(0, 64)
            k_ptrs = (
                ckv_cache_ptr
                + page[None, :] * stride_ckv_p
                + slot[None, :] * stride_ckv_s
                + d_offs[:, None] * stride_ckv_d
            )
            k = tl.load(k_ptrs, mask=valid[None, :] & (d_offs[:, None] < DCKV), other=0.0).to(tl.float32)
            q = tl.load(
                q_nope_ptr + pid_t * stride_qn_t + offs_h[:, None] * stride_qn_h + d_offs[None, :] * stride_qn_d,
                mask=mask_h[:, None] & (d_offs[None, :] < DCKV),
                other=0.0,
            ).to(tl.float32)
            logits += tl.dot(q, tl.trans(k))

        for dp0 in range(0, DKPE, 32):
            p_offs = dp0 + tl.arange(0, 32)
            kp_ptrs = (
                kpe_cache_ptr
                + page[None, :] * stride_kpe_p
                + slot[None, :] * stride_kpe_s
                + p_offs[:, None] * stride_kpe_d
            )
            kp = tl.load(kp_ptrs, mask=valid[None, :] & (p_offs[:, None] < DKPE), other=0.0).to(tl.float32)
            q = tl.load(
                q_pe_ptr + pid_t * stride_qp_t + offs_h[:, None] * stride_qp_h + p_offs[None, :] * stride_qp_d,
                mask=mask_h[:, None] & (p_offs[None, :] < DKPE),
                other=0.0,
            ).to(tl.float32)
            logits += tl.dot(q, tl.trans(kp))

        logits = logits * sm_scale
        logits = tl.where(valid[None, :], logits, -float("inf"))
        p = tl.exp2((logits - m_i[:, None]) * 1.4426950408889634) / l_i[:, None]
        p = tl.where(valid[None, :], p, 0.0)

        v_ptrs = (
            ckv_cache_ptr
            + page[:, None] * stride_ckv_p
            + slot[:, None] * stride_ckv_s
            + offs_d[None, :] * stride_ckv_d
        )
        v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(p, v)

    out_ptrs = output_ptr + pid_t * stride_o_t + offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_h[:, None] & mask_d[None, :])


def _move_to_cuda(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            return x, x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but GPU execution is required")
        return x.cuda(), x.device
    return x, None


def _restore_tensor_device(x, device):
    if device is None or not torch.is_tensor(x):
        return x
    if x.device == device:
        return x
    return x.to(device)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    orig_devices = {}
    q_nope, orig_devices["q_nope"] = _move_to_cuda(q_nope)
    q_pe, orig_devices["q_pe"] = _move_to_cuda(q_pe)
    ckv_cache, orig_devices["ckv_cache"] = _move_to_cuda(ckv_cache)
    kpe_cache, orig_devices["kpe_cache"] = _move_to_cuda(kpe_cache)
    sparse_indices, orig_devices["sparse_indices"] = _move_to_cuda(sparse_indices)

    if not torch.is_tensor(sm_scale):
        sm_scale_t = torch.tensor(sm_scale, dtype=torch.float32, device=q_nope.device)
        sm_scale_orig_device = None
    else:
        sm_scale_t, sm_scale_orig_device = _move_to_cuda(sm_scale)
        sm_scale_t = sm_scale_t.to(torch.float32)

    if q_nope.dtype != torch.bfloat16:
        q_nope = q_nope.to(torch.bfloat16)
    if q_pe.dtype != torch.bfloat16:
        q_pe = q_pe.to(torch.bfloat16)
    if ckv_cache.dtype != torch.bfloat16:
        ckv_cache = ckv_cache.to(torch.bfloat16)
    if kpe_cache.dtype != torch.bfloat16:
        kpe_cache = kpe_cache.to(torch.bfloat16)
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)

    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    sparse_indices = sparse_indices.contiguous()

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

    device = q_nope.device
    max_logits = torch.empty((num_tokens, num_qo_heads), device=device, dtype=torch.float32)
    sum_exp = torch.empty((num_tokens, num_qo_heads), device=device, dtype=torch.float32)
    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), device=device, dtype=torch.bfloat16)

    BLOCK_H = 4
    BLOCK_K = 32
    BLOCK_D = 128

    grid1 = (num_tokens, triton.cdiv(num_qo_heads, BLOCK_H))
    _dsa_sparse_attention_stage1_kernel[grid1](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        max_logits, sum_exp,
        num_tokens, num_pages, float(sm_scale_t.item()),
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        max_logits.stride(0), max_logits.stride(1),
        sum_exp.stride(0), sum_exp.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        PAGE_SIZE=64,
        TOPK=2048,
        DCKV=512,
        DKPE=64,
        num_warps=8,
        num_stages=3,
    )

    grid2 = (num_tokens, (num_qo_heads // BLOCK_H) * triton.cdiv(head_dim_ckv, BLOCK_D))
    _dsa_sparse_attention_stage2_kernel[grid2](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        max_logits, sum_exp, output,
        num_tokens, num_pages, float(sm_scale_t.item()),
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        max_logits.stride(0), max_logits.stride(1),
        sum_exp.stride(0), sum_exp.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=64,
        TOPK=2048,
        DCKV=512,
        DKPE=64,
        num_warps=8,
        num_stages=3,
    )

    lse = max_logits + torch.log2(sum_exp)

    output = _restore_tensor_device(output, orig_devices["q_nope"])
    lse = _restore_tensor_device(lse, orig_devices["q_nope"])
    return output, lse