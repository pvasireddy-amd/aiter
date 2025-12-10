# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _static_per_tensor_quant_fp8_i8_kernel(
    qx_ptr,
    x_in_ptr,
    scale_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


@triton.jit
def _dynamic_per_tensor_quant_fp8_i8_kernel(
    x_in_ptr,
    scale_out_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / DTYPE_MAX, sem="relaxed")


@triton.jit
def _dynamic_per_token_quant_fp8_i8_kernel(
    qx_ptr,
    scale_out_ptr,
    x_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
    global_scale=None,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32
    global_scale: optional precomputed global scale for quantization
    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    
    # Apply global scale if provided
    if global_scale is not None:
        x = x / global_scale
    
    # Calculate local scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _mxfp4_find_global_max_kernel(
    x_ptr,
    global_max_ptr,
    stride_x_m_in,
    stride_x_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Pass 1: Find the global maximum absolute value across all blocks.
    Each block:
    1. Loads its tile of data (automatically uses shared memory in Triton)
    2. Computes the maximum absolute value within the tile
    3. Performs ONE atomic write to contribute to the global maximum
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Cast strides to int64
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    
    # Load entire block data into registers/shared memory
    x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    
    x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0, cache_modifier=".cg").to(tl.float32)
    
    # Find the maximum absolute value across the entire block
    # This is done in shared memory/registers - no global memory access
    local_max = tl.max(tl.abs(x))
    
    # Only ONE atomic write per block to global memory
    # Scale by 1/6 for mxfp4 range
    tl.atomic_max(global_max_ptr, local_max / 6.0, sem="relaxed")


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    global_scale_ptr=None,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    
    # Load global scale if provided
    global_scale = None
    if global_scale_ptr is not None:
        global_scale = tl.load(global_scale_ptr)
    
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE, global_scale
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (
                bs_offs_n < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
            )[None, :]
            tl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=bs_mask,
            )


def mxfp4_quant_two_pass(
    x: "torch.Tensor",
    block_size_m: int = 128,
    block_size_n: int = 128,
    mxfp4_quant_block_size: int = 32,
    num_iter: int = 1,
    num_stages: int = 4,
    use_global_scale: bool = True,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Two-pass approach for mxfp4 quantization with optional global scaling.
    
    Pass 1: Find global maximum across all blocks (if use_global_scale is True)
    Pass 2: Quantize using the global scale (or without if use_global_scale is False)
    
    Args:
        x: Input tensor of shape (M, N)
        block_size_m: Block size in M dimension
        block_size_n: Block size in N dimension
        mxfp4_quant_block_size: Quantization block size (typically 32)
        num_iter: Number of iterations per block
        num_stages: Number of pipeline stages
        use_global_scale: Whether to use global scaling (two-pass) or local scaling
        
    Returns:
        x_fp4: Quantized output of shape (M, N//2)
        bs: Block scales of shape (M, N//mxfp4_quant_block_size)
        global_scale: The computed global scale (or zeros if not used)
    """
    import torch
    
    M, N = x.shape
    assert N % 2 == 0, "N must be even for mxfp4 quantization"
    
    # Allocate output tensors
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    bs = torch.empty(
        (M, triton.cdiv(N, mxfp4_quant_block_size)),
        dtype=torch.uint8,
        device=x.device,
    )
    
    global_scale = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    if use_global_scale:
        # Pass 1: Find global maximum
        grid_pass1 = (
            triton.cdiv(M, block_size_m),
            triton.cdiv(N, block_size_n),
        )
        
        _mxfp4_find_global_max_kernel[grid_pass1](
            x_ptr=x,
            global_max_ptr=global_scale,
            stride_x_m_in=x.stride(0),
            stride_x_n_in=x.stride(1),
            M=M,
            N=N,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
        )
    
    # Pass 2: Quantize with or without global scale
    grid_pass2 = (
        triton.cdiv(M, block_size_m),
        triton.cdiv(triton.cdiv(N, block_size_n), num_iter),
    )
    
    _dynamic_mxfp4_quant_kernel[grid_pass2](
        x_ptr=x,
        x_fp4_ptr=x_fp4,
        bs_ptr=bs,
        stride_x_m_in=x.stride(0),
        stride_x_n_in=x.stride(1),
        stride_x_fp4_m_in=x_fp4.stride(0),
        stride_x_fp4_n_in=x_fp4.stride(1),
        stride_bs_m_in=bs.stride(0),
        stride_bs_n_in=bs.stride(1),
        M=M,
        N=triton.cdiv(N, block_size_n),
        global_scale_ptr=global_scale if use_global_scale else None,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        NUM_ITER=num_iter,
        NUM_STAGES=num_stages,
        MXFP4_QUANT_BLOCK_SIZE=mxfp4_quant_block_size,
        SCALING_MODE=1,  # Not used, but required by the kernel signature
    )
    
    return x_fp4, bs, global_scale
