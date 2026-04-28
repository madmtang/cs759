import math
import torch
import triton
import triton.language as tl


@triton.jit
def _stage1_lse_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_cache_ptr, kpe_cache_ptr,
    sparse_indices_ptr,
    lse_ptr,
    num_tokens, num_pages, sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_s, stride_ckv_d,
    stride_kpe_p, stride_kpe_s, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_lse_t, stride_lse_h,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_hb = tl.program_id(1)

    offs_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = offs_h < 16
    token_mask = pid_t < num_tokens
    qmask = token_mask & hmask

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)

    qn_ptrs_base = q_nope_ptr + pid_t * stride_qn_t + offs_h[:, None] * stride_qn_h
    qp_ptrs_base = q_pe_ptr + pid_t * stride_qp_t + offs_h[:, None] * stride_qp_h

    for k0 in range(0, 2048, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = token_mask & (offs_k < 2048)
        idx = tl.load(
            sparse_indices_ptr + pid_t * stride_idx_t + offs_k * stride_idx_k,
            mask=kmask,
            other=-1,
        ).to(tl.int32)

        valid = idx >= 0
        page = idx // PAGE_SIZE
        slot = idx % PAGE_SIZE
        valid = valid & (page < num_pages)

        logits = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)

        for d0 in range(0, 512, 64):
            offs_d = d0 + tl.arange(0, 64)

            q = tl.load(
                qn_ptrs_base + offs_d[None, :] * stride_qn_d,
                mask=qmask[:, None],
                other=0.0,
            ).to(tl.float32)

            k = tl.load(
                ckv_cache_ptr
                + page[None, :] * stride_ckv_p
                + slot[None, :] * stride_ckv_s
                + offs_d[:, None] * stride_ckv_d,
                mask=valid[None, :],
                other=0.0,
            ).to(tl.float32)

            logits += tl.dot(q, k)

        for d0 in range(0, 64, 32):
            offs_d = d0 + tl.arange(0, 32)

            q = tl.load(
                qp_ptrs_base + offs_d[None, :] * stride_qp_d,
                mask=qmask[:, None],
                other=0.0,
            ).to(tl.float32)

            k = tl.load(
                kpe_cache_ptr
                + page[None, :] * stride_kpe_p
                + slot[None, :] * stride_kpe_s
                + offs_d[:, None] * stride_kpe_d,
                mask=valid[None, :],
                other=0.0,
            ).to(tl.float32)

            logits += tl.dot(q, k)

        logits = logits * sm_scale
        logits = tl.where(valid[None, :], logits, -float("inf"))

        m_blk = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    lse_e = m_i + tl.log(l_i)
    lse_2 = lse_e * 1.4426950408889634

    tl.store(
        lse_ptr + pid_t * stride_lse_t + offs_h * stride_lse_h,
        lse_2,
        mask=qmask,
    )


@triton.jit
def _stage2_out_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_cache_ptr, kpe_cache_ptr,
    sparse_indices_ptr,
    lse_ptr,
    output_ptr,
    num_tokens, num_pages, sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_s, stride_ckv_d,
    stride_kpe_p, stride_kpe_s, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_lse_t, stride_lse_h,
    stride_o_t, stride_o_h, stride_o_d,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid = tl.program_id(1)

    num_h_blocks = tl.cdiv(16, BLOCK_H)
    pid_hb = pid % num_h_blocks
    pid_db = pid // num_h_blocks

    offs_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_dv = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)

    hmask = offs_h < 16
    dmask = offs_dv < 512
    token_mask = pid_t < num_tokens
    qmask = token_mask & hmask

    lse2 = tl.load(
        lse_ptr + pid_t * stride_lse_t + offs_h * stride_lse_h,
        mask=qmask,
        other=-float("inf"),
    )
    lse_e = lse2 * 0.6931471805599453

    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    qn_ptrs_base = q_nope_ptr + pid_t * stride_qn_t + offs_h[:, None] * stride_qn_h
    qp_ptrs_base = q_pe_ptr + pid_t * stride_qp_t + offs_h[:, None] * stride_qp_h

    for k0 in range(0, 2048, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = token_mask & (offs_k < 2048)
        idx = tl.load(
            sparse_indices_ptr + pid_t * stride_idx_t + offs_k * stride_idx_k,
            mask=kmask,
            other=-1,
        ).to(tl.int32)

        valid = idx >= 0
        page = idx // PAGE_SIZE
        slot = idx % PAGE_SIZE
        valid = valid & (page < num_pages)

        logits = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)

        for d0 in range(0, 512, 64):
            offs_d = d0 + tl.arange(0, 64)

            q = tl.load(
                qn_ptrs_base + offs_d[None, :] * stride_qn_d,
                mask=qmask[:, None],
                other=0.0,
            ).to(tl.float32)

            k = tl.load(
                ckv_cache_ptr
                + page[None, :] * stride_ckv_p
                + slot[None, :] * stride_ckv_s
                + offs_d[:, None] * stride_ckv_d,
                mask=valid[None, :],
                other=0.0,
            ).to(tl.float32)

            logits += tl.dot(q, k)

        for d0 in range(0, 64, 32):
            offs_d = d0 + tl.arange(0, 32)

            q = tl.load(
                qp_ptrs_base + offs_d[None, :] * stride_qp_d,
                mask=qmask[:, None],
                other=0.0,
            ).to(tl.float32)

            k = tl.load(
                kpe_cache_ptr
                + page[None, :] * stride_kpe_p
                + slot[None, :] * stride_kpe_s
                + offs_d[:, None] * stride_kpe_d,
                mask=valid[None, :],
                other=0.0,
            ).to(tl.float32)

            logits += tl.dot(q, k)

        logits = logits * sm_scale
        w = tl.exp(logits - lse_e[:, None])
        w = tl.where(valid[None, :], w, 0.0)

        v = tl.load(
            ckv_cache_ptr
            + page[:, None] * stride_ckv_p
            + slot[:, None] * stride_ckv_s
            + offs_dv[None, :] * stride_ckv_d,
            mask=valid[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(w, v)

    tl.store(
        output_ptr
        + pid_t * stride_o_t
        + offs_h[:, None] * stride_o_h
        + offs_dv[None, :] * stride_o_d,
        acc.to(tl.bfloat16),
        mask=qmask[:, None] & dmask[None, :],
    )


def _to_cuda_tensor(x, dtype=None):
    if torch.is_tensor(x):
        orig_device = x.device
        if x.is_cuda:
            t = x
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but GPU execution is required")
            t = x.cuda()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t, orig_device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but GPU execution is required")
        return torch.tensor(x, device="cuda", dtype=dtype), None


def _restore_tensor(x, device):
    if device is None:
        return x
    if x.device == device:
        return x
    return x.to(device)


def run(*args, **kwargs):
    names = ["q_nope", "q_pe", "ckv_cache", "kpe_cache", "sparse_indices", "sm_scale"]
    vals = {}
    for i, n in enumerate(names):
        if i < len(args):
            vals[n] = args[i]
        elif n in kwargs:
            vals[n] = kwargs[n]
        else:
            raise TypeError(f"Missing required argument: {n}")

    q_nope, out_device = _to_cuda_tensor(vals["q_nope"], torch.bfloat16)
    q_pe, _ = _to_cuda_tensor(vals["q_pe"], torch.bfloat16)
    ckv_cache, _ = _to_cuda_tensor(vals["ckv_cache"], torch.bfloat16)
    kpe_cache, _ = _to_cuda_tensor(vals["kpe_cache"], torch.bfloat16)
    sparse_indices, _ = _to_cuda_tensor(vals["sparse_indices"], torch.int32)
    sm_scale_t, _ = _to_cuda_tensor(vals["sm_scale"], torch.float32)

    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    sparse_indices = sparse_indices.contiguous()
    sm_scale = float(sm_scale_t.item())

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, ckv_dim2 = ckv_cache.shape
    kpe_dim2 = kpe_cache.shape[-1]
    topk = sparse_indices.shape[-1]

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
    if ckv_dim2 != 512:
        raise ValueError(f"Expected ckv_cache.shape[2] == 512, got {ckv_dim2}")
    if kpe_dim2 != 64:
        raise ValueError(f"Expected kpe_cache.shape[2] == 64, got {kpe_dim2}")
    if sparse_indices.shape[0] != num_tokens:
        raise ValueError("sparse_indices.shape[0] must equal num_tokens")
    if ckv_cache.shape[1] != page_size:
        raise ValueError("ckv_cache.shape[1] must equal page_size")

    device = q_nope.device
    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), device=device, dtype=torch.bfloat16)
    lse = torch.empty((num_tokens, num_qo_heads), device=device, dtype=torch.float32)

    BLOCK_H = 4
    BLOCK_K = 64
    BLOCK_D = 128

    grid1 = (num_tokens, triton.cdiv(16, BLOCK_H))
    _stage1_lse_kernel[grid1](
        q_nope, q_pe,
        ckv_cache, kpe_cache,
        sparse_indices,
        lse,
        num_tokens, num_pages, sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        lse.stride(0), lse.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        PAGE_SIZE=64,
        num_warps=8,
        num_stages=3,
    )

    grid2 = (num_tokens, triton.cdiv(16, BLOCK_H) * triton.cdiv(512, BLOCK_D))
    _stage2_out_kernel[grid2](
        q_nope, q_pe,
        ckv_cache, kpe_cache,
        sparse_indices,
        lse,
        output,
        num_tokens, num_pages, sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        lse.stride(0), lse.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=64,
        num_warps=8,
        num_stages=3,
    )

    valid_counts = (sparse_indices != -1).sum(dim=-1)
    if torch.any(valid_counts == 0):
        empty = valid_counts == 0
        output[empty] = 0
        lse[empty] = -float("inf")

    output = _restore_tensor(output, out_device)
    lse = _restore_tensor(lse, out_device)
    return output, lse