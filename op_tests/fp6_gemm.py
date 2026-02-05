import torch
import triton
import triton.language as tl
import tcast
import numpy as np
import argparse

# Import the CPU implementation for comparison
from fp6_cpu import ref_out

@triton.jit
def triton_fp6_pack_32bit_kernel(
    fp6_packed_ptr, output_ptr,
    n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr
):
    """Pack 5 FP6 values into 32 bits (Mode 2) with automatic padding"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Each group processes 5 FP6 values
    base_idx = pid * 5
    
    # Initialize packed value as scalar with explicit type
    packed_32bit = tl.zeros((), dtype=tl.uint32)
    
    # Process each of the 5 FP6 values (already converted to 6-bit format)
    for i in tl.static_range(5):
        idx = base_idx + i
        # Load FP6 value if within bounds, otherwise use 0 (padding)
        if idx < n_elements:
            fp6_val = tl.load(fp6_packed_ptr + idx)
        else:
            fp6_val = tl.zeros((), dtype=tl.uint8)
            
        # Add to packed value at correct position
        shift = i * 6
        packed_32bit = packed_32bit | (fp6_val.to(tl.uint32) << shift)
    
    # Store as uint32
    tl.store(output_ptr + pid, packed_32bit.to(tl.uint32))

@triton.jit
def triton_fp6_pack_96bit_kernel(
    fp6_packed_ptr, output_ptr,
    n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr
):
    """Pack 16 FP6 values into 96 bits (Mode 3) with automatic padding"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Each group processes 16 FP6 values
    base_idx = pid * 16
    
    # Initialize 3x32-bit words
    word0 = tl.zeros((), dtype=tl.uint32)
    word1 = tl.zeros((), dtype=tl.uint32)
    word2 = tl.zeros((), dtype=tl.uint32)
    
    # Process each of the 16 FP6 values
    for i in tl.static_range(16):
        idx = base_idx + i
        # Load FP6 value if within bounds, otherwise use 0 (padding)
        if idx < n_elements:
            fp6_val = tl.load(fp6_packed_ptr + idx).to(tl.uint32)
        else:
            fp6_val = tl.zeros((), dtype=tl.uint32)
        
        # Calculate bit position within 96-bit group
        bit_offset = i * 6
        word_idx = bit_offset // 32
        bit_in_word = bit_offset % 32
        
        # Handle values that may span word boundaries
        if word_idx == 0:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word0
                word0 = word0 | (fp6_val << bit_in_word)
            else:
                # Spans word0 and word1
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word0 = word0 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word1 = word1 | (fp6_val >> bits_in_first)
        elif word_idx == 1:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word1
                word1 = word1 | (fp6_val << bit_in_word)
            else:
                # Spans word1 and word2
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word1 = word1 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word2 = word2 | (fp6_val >> bits_in_first)
        else:
            # word_idx == 2, fits entirely in word2
            word2 = word2 | (fp6_val << bit_in_word)
    
    # Store 3 words
    out_idx = pid * 3
    tl.store(output_ptr + out_idx, word0)
    tl.store(output_ptr + out_idx + 1, word1)
    tl.store(output_ptr + out_idx + 2, word2)

@triton.jit
def triton_fp6_pack_192bit_kernel(
    fp6_packed_ptr, output_ptr,
    n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr
):
    """Pack 32 FP6 values into 192 bits (Mode 4) with automatic padding"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Each group processes 32 FP6 values
    base_idx = pid * 32
    
    # Initialize 6x32-bit words
    word0 = tl.zeros((), dtype=tl.uint32)
    word1 = tl.zeros((), dtype=tl.uint32)
    word2 = tl.zeros((), dtype=tl.uint32)
    word3 = tl.zeros((), dtype=tl.uint32)
    word4 = tl.zeros((), dtype=tl.uint32)
    word5 = tl.zeros((), dtype=tl.uint32)
    
    # Process each of the 32 FP6 values
    for i in tl.static_range(32):
        idx = base_idx + i
        # Load FP6 value if within bounds, otherwise use 0 (padding)
        if idx < n_elements:
            fp6_val = tl.load(fp6_packed_ptr + idx).to(tl.uint32)
        else:
            fp6_val = tl.zeros((), dtype=tl.uint32)
        
        # Calculate bit position within 192-bit group
        bit_offset = i * 6
        word_idx = bit_offset // 32
        bit_in_word = bit_offset % 32
        
        # Handle values that may span word boundaries
        if word_idx == 0:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word0
                word0 = word0 | (fp6_val << bit_in_word)
            else:
                # Spans word0 and word1
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word0 = word0 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word1 = word1 | (fp6_val >> bits_in_first)
        elif word_idx == 1:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word1
                word1 = word1 | (fp6_val << bit_in_word)
            else:
                # Spans word1 and word2
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word1 = word1 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word2 = word2 | (fp6_val >> bits_in_first)
        elif word_idx == 2:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word2
                word2 = word2 | (fp6_val << bit_in_word)
            else:
                # Spans word2 and word3
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word2 = word2 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word3 = word3 | (fp6_val >> bits_in_first)
        elif word_idx == 3:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word3
                word3 = word3 | (fp6_val << bit_in_word)
            else:
                # Spans word3 and word4
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word3 = word3 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word4 = word4 | (fp6_val >> bits_in_first)
        elif word_idx == 4:
            if bit_in_word + 6 <= 32:
                # Fits entirely in word4
                word4 = word4 | (fp6_val << bit_in_word)
            else:
                # Spans word4 and word5
                bits_in_first = 32 - bit_in_word
                bits_in_second = 6 - bits_in_first
                word4 = word4 | ((fp6_val & ((1 << bits_in_first) - 1)) << bit_in_word)
                word5 = word5 | (fp6_val >> bits_in_first)
        else:
            # word_idx == 5, fits entirely in word5
            word5 = word5 | (fp6_val << bit_in_word)
    
    # Store 6 words
    out_idx = pid * 6
    tl.store(output_ptr + out_idx, word0)
    tl.store(output_ptr + out_idx + 1, word1)
    tl.store(output_ptr + out_idx + 2, word2)
    tl.store(output_ptr + out_idx + 3, word3)
    tl.store(output_ptr + out_idx + 4, word4)
    tl.store(output_ptr + out_idx + 5, word5)

@triton.jit
def triton_dequant_and_pack_kernel(
    fp6_tensor_ptr, scale_tensor_ptr, output_ptr, scaled_vals_ptr,
    N, n_scales, group_size,
    BLOCK_SIZE: tl.constexpr,
    packing_mode: tl.constexpr
):
    """
    Dequantize FP6 fake tensor and pack into 8-bit format for mode 0.
    
    Args:
        fp6_tensor_ptr: Pointer to FP6 tensor values
        scale_tensor_ptr: Pointer to scale tensor (int32 exponents)
        output_ptr: Pointer to output tensor (packed FP6 in uint8)
        scaled_vals_ptr: Pointer to store scaled values for verification
        N: Total number of elements
        n_scales: Number of scale values
        group_size: Number of elements per scale group
        BLOCK_SIZE: Block size for kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load FP6 values (already in bfloat16 format from tcast)
    vals = tl.load(fp6_tensor_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)

    # Compute which scale group each element belongs to
    scale_indices = offsets // group_size
    scale_mask = scale_indices < n_scales
    
    # Load scale values (int32 exponents)
    scale_vals = tl.load(scale_tensor_ptr + scale_indices, mask=mask & scale_mask, other=0).to(tl.int32)

    # Compute scaling factor: 2^(127 - S)
    scale_factors = tl.exp2((127 - scale_vals).to(tl.float32)).to(tl.bfloat16)

    # Apply scale
    scaled_vals = vals * scale_factors
    
    # Store scaled values for verification
    tl.store(scaled_vals_ptr + offsets, scaled_vals, mask=mask)
  
    # Convert to int16 for bit manipulation
    bf16_int = scaled_vals.to(tl.int16, bitcast=True)
    
    # Extract components from BF16: [sign(1)][exponent(8)][mantissa(7)]
    sign = (bf16_int >> 15) & 0x1
    exponent = (bf16_int >> 7) & 0xFF
    mantissa = bf16_int & 0x7F
    
    # Convert to FP6 format
    fp6_exponent = tl.where((exponent < 127), 0, exponent - 126)
    #fp6_mantissa = mantissa >> 4  # Take top 3 bits of mantissa
    fp6_mantissa =  tl.where((exponent == 124), 1, 
                    tl.where((exponent == 125), 2 | (mantissa >> 6), 
                    tl.where((exponent == 126), 4 | (mantissa >> 5), mantissa >> 4)))  # Handle subnormals

    # Pack into FP6 format: [00][sign(1)][exp(2)][mantissa(3)]
    fp6_packed = (sign << 5) | (fp6_exponent << 3) | fp6_mantissa
    
    # For both modes, store the FP6 values as uint8
    # Mode 1 will require a separate kernel to pack into 24-bit groups
    tl.store(output_ptr + offsets, fp6_packed.to(tl.uint8), mask=mask)

@triton.jit
def triton_fp6_pack_24bit_kernel(
    fp6_packed_ptr, output_ptr,
    n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr
):
    """Pack 4 FP6 values into 24 bits (Mode 1) with automatic padding"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Each group processes 4 FP6 values
    base_idx = pid * 4
    
    # Initialize packed value as scalar with explicit type
    packed_24bit = tl.zeros((), dtype=tl.uint32)
    
    # Process each of the 4 FP6 values (already converted to 6-bit format)
    for i in tl.static_range(4):
        idx = base_idx + i
        # Load FP6 value if within bounds, otherwise use 0 (padding)
        if idx < n_elements:
            fp6_val = tl.load(fp6_packed_ptr + idx)
        else:
            fp6_val = tl.zeros((), dtype=tl.uint8)
            
        # Add to packed value at correct position
        shift = i * 6
        packed_24bit = packed_24bit | (fp6_val.to(tl.uint32) << shift)
    
    # Write 3 bytes - convert to proper type first
    packed_uint = packed_24bit.to(tl.uint32)
    byte0 = (packed_uint & 0xFF).to(tl.uint8)
    byte1 = ((packed_uint >> 8) & 0xFF).to(tl.uint8)
    byte2 = ((packed_uint >> 16) & 0xFF).to(tl.uint8)
    
    out_idx = pid * 3
    tl.store(output_ptr + out_idx, byte0)
    tl.store(output_ptr + out_idx + 1, byte1)
    tl.store(output_ptr + out_idx + 2, byte2)

def triton_fused_pack_kernel(tcast_tensor, packing_mode):
    """
    Triton implementation of FP6 packing for modes 0, 1, 2, 3, and 4.
    
    Args:
        tcast_tensor: Tensorcast quantized tensor containing .tensor and .scaledata.scale
        packing_mode: 0 (8-bit), 1 (24-bit), 2 (32-bit), 3 (96-bit), or 4 (192-bit) packing
        
    Returns:
        packed_tensor: Packed representation
        scaled_vals: The scaled values before FP6 conversion 
    """
    if packing_mode > 4:
        raise ValueError(f"Invalid packing mode: {packing_mode}. Supported modes are 0, 1, 2, 3, and 4.")

    fp6_tensor = tcast_tensor.tensor
    scale_tensor = tcast_tensor.scaledata.scale
    n_elements = fp6_tensor.numel()
    n_scales = scale_tensor.numel()
    
    # Determine group size (elements per scale)
    group_size = n_elements // n_scales if n_scales > 0 else n_elements
    
    # Allocate output tensor for packed values (uint8)
    device = fp6_tensor.device
    fp6_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
    
    # Allocate tensor for scaled values if requested
    scaled_vals_output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
    
    # Launch dequantization and packing kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    triton_dequant_and_pack_kernel[grid](
        fp6_tensor, scale_tensor, fp6_output, scaled_vals_output,
        n_elements, n_scales, group_size,
        BLOCK_SIZE=BLOCK_SIZE, packing_mode=packing_mode
    )

    scaled_vals_reshaped = scaled_vals_output.reshape(fp6_tensor.shape)
    
    if packing_mode == 0:
        # Mode 0: Return FP6 values as-is
        packed_tensor = fp6_output.reshape(fp6_tensor.shape)
        return packed_tensor, scaled_vals_reshaped
    
    elif packing_mode == 1:
        # Mode 1: 24-bit packing (4 FP6 values -> 3 bytes)
        # Calculate groups using ceiling division - kernel handles padding automatically
        n_groups = triton.cdiv(n_elements, 4)
        n_bytes = n_groups * 3
        
        # Allocate output for 24-bit packed values
        packed_output = torch.empty(n_bytes, dtype=torch.uint8, device=device)
        
        # Launch 24-bit packing kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_pack_24bit_kernel[grid](
            fp6_output, packed_output,
            n_elements, n_groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape to (-1, 3) like CPU implementation
        return packed_output.reshape(-1, 3), scaled_vals_reshaped

    elif packing_mode == 2:
        # Mode 2: 32-bit packing (5 FP6 values -> 32 bits)
        # Calculate groups using ceiling division - kernel handles padding automatically
        n_groups = triton.cdiv(n_elements, 5)
        
        # Allocate output for 32-bit packed values
        packed_output = torch.empty(n_groups, dtype=torch.uint32, device=device)
        
        # Launch 32-bit packing kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_pack_32bit_kernel[grid](
            fp6_output, packed_output,
            n_elements, n_groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return packed_output, scaled_vals_reshaped
    
    elif packing_mode == 3:
        # Mode 3: 96-bit packing (16 FP6 values -> 96 bits)
        # Calculate groups using ceiling division - kernel handles padding automatically
        n_groups = triton.cdiv(n_elements, 16)
        
        # Allocate output for 96-bit packed values (3x32-bit words per group)
        packed_output = torch.empty(n_groups * 3, dtype=torch.uint32, device=device)
        
        # Launch 96-bit packing kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_pack_96bit_kernel[grid](
            fp6_output, packed_output,
            n_elements, n_groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape to (-1, 3) to match CPU implementation
        return packed_output.reshape(-1, 3), scaled_vals_reshaped
    
    elif packing_mode == 4:
        # Mode 4: 192-bit packing (32 FP6 values -> 192 bits)
        # Calculate groups using ceiling division - kernel handles padding automatically
        n_groups = triton.cdiv(n_elements, 32)
        
        # Allocate output for 192-bit packed values (6x32-bit words per group)
        packed_output = torch.empty(n_groups * 6, dtype=torch.uint32, device=device)
        
        # Launch 192-bit packing kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_pack_192bit_kernel[grid](
            fp6_output, packed_output,
            n_elements, n_groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape to (-1, 6) to match CPU implementation
        return packed_output.reshape(-1, 6), scaled_vals_reshaped
    
    else:
        raise ValueError(f"Invalid packing mode: {packing_mode}")

@triton.jit
def triton_fp6_unpack_8bit_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack FP6 E2M3 format to FP8 E4M3 and BF16 values (Mode 0)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load packed FP6 values (stored as uint8)
    fp6_vals = tl.load(packed_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    
    # Extract FP6 components: [00][S][EE][MMM]
    sign = ((fp6_vals >> 5) & 0x1).to(tl.int32)        # Extract sign bit
    fp6_exponent = ((fp6_vals >> 3) & 0x3).to(tl.int32)   # Extract 2-bit exponent  
    fp6_mantissa = (fp6_vals & 0x7).to(tl.int32)        # Extract 3-bit mantissa

    # 1. Convert into FP8 E4M3 format
    # Add 6 to exponent for E4M3 bias
    fp8_exponent = fp6_exponent + 6  # Now 4 bits (range 6-9)
    # Pack into FP8 E4M3: [S][EEEE][MMM]
    fp8_extended = (sign << 7) | (fp8_exponent << 3) | fp6_mantissa
    
    # Store FP8 values
    tl.store(fp8_ptr + offsets, fp8_extended.to(tl.uint8), mask=mask)

    # 2. Convert into BF16 to verify the extracted tensor
    # New exponent = FP6 exponent + 126 (to account for BF16 bias of 127 - FP6 bias of 1)
    bf16_exponent = fp6_exponent + 126
    
    # Extend mantissa from 3 bits to 7 bits by adding 0s to the right
    bf16_mantissa = fp6_mantissa << 4  # Shift left by 4 to add 4 zero bits
    
    # Pack into BF16: [S][EEEEEEEE][MMMMMMM]
    bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
    
    # Convert the integer representation to bfloat16
    # Store as bfloat16 by interpreting the bit pattern
    bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
    tl.store(bf16_ptr + offsets, bf16_value, mask=mask)
    
@triton.jit
def triton_fp6_unpack_24bit_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format to FP8 E4M3 and BF16 values (Mode 1)"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Load 3 bytes for this group
    byte_idx = pid * 3
    byte0 = tl.load(packed_ptr + byte_idx).to(tl.uint32)
    byte1 = tl.load(packed_ptr + byte_idx + 1).to(tl.uint32)
    byte2 = tl.load(packed_ptr + byte_idx + 2).to(tl.uint32)
    
    # Reconstruct 24-bit value
    packed_24bit = byte0 | (byte1 << 8) | (byte2 << 16)
    
    # Extract 4 FP6 values
    base_idx = pid * 4
    for i in tl.static_range(4):
        idx = base_idx + i
        if idx < n_elements:
            # Extract 6-bit FP6 value
            shift = i * 6
            fp6_val = (packed_24bit >> shift) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = ((fp6_val >> 5) & 0x1).to(tl.int32)
            fp6_exponent = ((fp6_val >> 3) & 0x3).to(tl.int32)
            fp6_mantissa = (fp6_val & 0x7).to(tl.int32)
            
            # 1. Convert to FP8 format
            fp8_exponent = tl.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0, 
                            tl.where((fp6_exponent == 0) & (fp6_mantissa == 1), 4,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 1) & (fp6_mantissa < 4), 5,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 3), 6, fp6_exponent + 6))))

            fp8_mantissa = tl.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0, 
                            tl.where((fp6_exponent == 0) & (fp6_mantissa == 1), 0,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 1) & (fp6_mantissa < 4), (fp6_mantissa & 1) << 2,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 3), (fp6_mantissa & 3) << 1, fp6_mantissa))))
            
            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp8_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))

            # 2. Convert to BF8 E4M3 format
            bf16_exponent = tl.where((fp8_exponent == 0), 0, fp8_exponent + 120)  # Add 6 for E4M3 bias
            bf16_mantissa = fp8_mantissa << 4  # Take top 3 bits of mantissa
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)
            

@triton.jit
def triton_fp6_unpack_32bit_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 32-bit packed format to FP8 E4M3 and BF16 values (Mode 2)"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Load 32-bit packed value
    packed_32bit = tl.load(packed_ptr + pid).to(tl.uint32)
    
    # Extract 5 FP6 values
    base_idx = pid * 5
    for i in tl.static_range(5):
        idx = base_idx + i
        if idx < n_elements:
            # Extract 6-bit FP6 value
            shift = i * 6
            fp6_val = (packed_32bit >> shift) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = ((fp6_val >> 5) & 0x1).to(tl.int32)
            fp6_exponent = ((fp6_val >> 3) & 0x3).to(tl.int32)
            fp6_mantissa = (fp6_val & 0x7).to(tl.int32)
            
            # 1. Convert to FP8 E4M3 format
            fp8_exponent = fp6_exponent + 6  # Add 6 for E4M3 bias
            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp6_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))
            
            # 2. Convert to BF16 format
            bf16_exponent = fp6_exponent + 126  # BF16 bias adjustment
            bf16_mantissa = fp6_mantissa << 4   # Extend mantissa to 7 bits
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)

@triton.jit
def triton_fp6_unpack_96bit_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 96-bit packed format to FP8 E4M3 and BF16 values (Mode 3)"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Load 3x32-bit words for this group
    word_idx = pid * 3
    word0 = tl.load(packed_ptr + word_idx).to(tl.uint32)
    word1 = tl.load(packed_ptr + word_idx + 1).to(tl.uint32)
    word2 = tl.load(packed_ptr + word_idx + 2).to(tl.uint32)
    
    # Extract 16 FP6 values
    base_idx = pid * 16
    for i in tl.static_range(16):
        idx = base_idx + i
        if idx < n_elements:
            # Calculate bit position within 96-bit group
            bit_offset = i * 6
            word_idx = bit_offset // 32
            bit_in_word = bit_offset % 32
            
            # Extract 6-bit FP6 value (may span word boundaries)
            if word_idx == 0:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word0
                    fp6_val = (word0 >> bit_in_word) & 0x3F
                else:
                    # Spans word0 and word1
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word0 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word1 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            elif word_idx == 1:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word1
                    fp6_val = (word1 >> bit_in_word) & 0x3F
                else:
                    # Spans word1 and word2
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word1 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word2 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            else:
                # word_idx == 2, fits entirely in word2
                fp6_val = (word2 >> bit_in_word) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = ((fp6_val >> 5) & 0x1).to(tl.int32)
            fp6_exponent = ((fp6_val >> 3) & 0x3).to(tl.int32)
            fp6_mantissa = (fp6_val & 0x7).to(tl.int32)
            
            # 1. Convert to FP8 E4M3 format
            fp8_exponent = fp6_exponent + 6  # Add 6 for E4M3 bias
            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp6_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))
            
            # 2. Convert to BF16 format
            bf16_exponent = fp6_exponent + 126  # BF16 bias adjustment
            bf16_mantissa = fp6_mantissa << 4   # Extend mantissa to 7 bits
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)

@triton.jit
def triton_fp6_unpack_192bit_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 192-bit packed format to FP8 E4M3 and BF16 values (Mode 4)"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Load 6x32-bit words for this group
    word_idx = pid * 6
    word0 = tl.load(packed_ptr + word_idx).to(tl.uint32)
    word1 = tl.load(packed_ptr + word_idx + 1).to(tl.uint32)
    word2 = tl.load(packed_ptr + word_idx + 2).to(tl.uint32)
    word3 = tl.load(packed_ptr + word_idx + 3).to(tl.uint32)
    word4 = tl.load(packed_ptr + word_idx + 4).to(tl.uint32)
    word5 = tl.load(packed_ptr + word_idx + 5).to(tl.uint32)
    
    # Extract 32 FP6 values
    base_idx = pid * 32
    for i in tl.static_range(32):
        idx = base_idx + i
        if idx < n_elements:
            # Calculate bit position within 192-bit group
            bit_offset = i * 6
            word_idx = bit_offset // 32
            bit_in_word = bit_offset % 32
            
            # Extract 6-bit FP6 value (may span word boundaries)
            if word_idx == 0:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word0
                    fp6_val = (word0 >> bit_in_word) & 0x3F
                else:
                    # Spans word0 and word1
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word0 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word1 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            elif word_idx == 1:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word1
                    fp6_val = (word1 >> bit_in_word) & 0x3F
                else:
                    # Spans word1 and word2
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word1 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word2 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            elif word_idx == 2:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word2
                    fp6_val = (word2 >> bit_in_word) & 0x3F
                else:
                    # Spans word2 and word3
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word2 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word3 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            elif word_idx == 3:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word3
                    fp6_val = (word3 >> bit_in_word) & 0x3F
                else:
                    # Spans word3 and word4
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word3 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word4 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            elif word_idx == 4:
                if bit_in_word + 6 <= 32:
                    # Fits entirely in word4
                    fp6_val = (word4 >> bit_in_word) & 0x3F
                else:
                    # Spans word4 and word5
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    first_part = (word4 >> bit_in_word) & ((1 << bits_in_first) - 1)
                    second_part = word5 & ((1 << bits_in_second) - 1)
                    fp6_val = first_part | (second_part << bits_in_first)
            else:
                # word_idx == 5, fits entirely in word5
                fp6_val = (word5 >> bit_in_word) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = ((fp6_val >> 5) & 0x1).to(tl.int32)
            fp6_exponent = ((fp6_val >> 3) & 0x3).to(tl.int32)
            fp6_mantissa = (fp6_val & 0x7).to(tl.int32)
            
            # 1. Convert to FP8 E4M3 format
            fp8_exponent = fp6_exponent + 6  # Add 6 for E4M3 bias
            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp6_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))
            
            # 2. Convert to BF16 format
            bf16_exponent = fp6_exponent + 126  # BF16 bias adjustment
            bf16_mantissa = fp6_mantissa << 4   # Extend mantissa to 7 bits
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)

def triton_fused_unpack_gemm_kernel(packed_tensor, packing_mode, original_shape=None):
    """
    Unpack FP6 values from packed representation for use with FP8 GEMM kernels.
    
    Args:
        packed_tensor: Packed tensor from triton_fused_pack_kernel
        packing_mode: 0 (8-bit), 1 (24-bit), 2 (32-bit), or 3 (96-bit) packing
        original_shape: Original shape for trimming padded values (for modes that use padding)
        
    Returns:
        tuple: (fp8_tensor, bf16_tensor)
            fp8_tensor: Tensor ready for FP8 GEMM (E4M3 format)
            bf16_tensor: BF16 representation for verification
    """
    if packing_mode > 4:
        raise ValueError(f"Invalid packing mode: {packing_mode}. Supported modes are 0, 1, 2, and 3.")
        
    device = packed_tensor.device
    
    if device.type != 'cuda':
        raise ValueError("Triton implementation requires CUDA device")
    
    if packing_mode == 0:
        n_elements = packed_tensor.numel()
        
        # Allocate output tensors
        fp8_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
        bf16_output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        triton_fp6_unpack_8bit_kernel[grid](
            packed_tensor.flatten(), fp8_output, bf16_output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape to original shape
        return fp8_output.reshape(packed_tensor.shape), bf16_output.reshape(packed_tensor.shape)
        
    elif packing_mode == 1:
        # Mode 1: 24-bit packing (3 bytes contain 4 FP6 values)
        packed_bytes = packed_tensor.flatten()
        n_groups = packed_bytes.numel() // 3
        n_elements = n_groups * 4
        
        # Allocate output tensors
        fp8_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
        bf16_output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        
        # Launch kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        triton_fp6_unpack_24bit_kernel[grid](
            packed_bytes, fp8_output, bf16_output,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Trim to original size if shape provided
        fp8_output = fp8_output.view(torch.float8_e4m3fn)
        if original_shape:
            original_size = np.prod(original_shape)
            fp8_output = fp8_output[:original_size]
            bf16_output = bf16_output[:original_size]
            return fp8_output.reshape(original_shape), bf16_output.reshape(original_shape)
        return fp8_output, bf16_output
        
    elif packing_mode == 2:
        # Mode 2: 32-bit packing (5 FP6 values per 32-bit word)
        n_groups = packed_tensor.numel()
        n_elements = n_groups * 5
        
        # Allocate output tensors
        fp8_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
        bf16_output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        
        # Launch kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_unpack_32bit_kernel[grid](
            packed_tensor.flatten(), fp8_output, bf16_output,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Trim to original size if shape provided
        if original_shape:
            original_size = np.prod(original_shape)
            fp8_output = fp8_output[:original_size]
            bf16_output = bf16_output[:original_size]
            return fp8_output.reshape(original_shape), bf16_output.reshape(original_shape)
        
        return fp8_output, bf16_output
    
    elif packing_mode == 3:
        # Mode 3: 96-bit packing (16 FP6 values in 3x32-bit words)
        packed_words = packed_tensor.flatten()
        n_groups = packed_words.numel() // 3
        n_elements = n_groups * 16
        
        # Allocate output tensors
        fp8_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
        bf16_output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        
        # Launch kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_unpack_96bit_kernel[grid](
            packed_words, fp8_output, bf16_output,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Trim to original size if shape provided
        if original_shape:
            original_size = np.prod(original_shape)
            fp8_output = fp8_output[:original_size]
            bf16_output = bf16_output[:original_size]
            return fp8_output.reshape(original_shape), bf16_output.reshape(original_shape)
        
        return fp8_output, bf16_output
    
    elif packing_mode == 4:
        # Mode 4: 192-bit packing (32 FP6 values in 6x32-bit words)
        packed_words = packed_tensor.flatten()
        n_groups = packed_words.numel() // 6
        n_elements = n_groups * 32
        
        # Allocate output tensors
        fp8_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
        bf16_output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        
        # Launch kernel
        grid = (n_groups,)
        BLOCK_SIZE = 1
        
        triton_fp6_unpack_192bit_kernel[grid](
            packed_words, fp8_output, bf16_output,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Trim to original size if shape provided
        if original_shape:
            original_size = np.prod(original_shape)
            fp8_output = fp8_output[:original_size]
            bf16_output = bf16_output[:original_size]
            return fp8_output.reshape(original_shape), bf16_output.reshape(original_shape)
        
        return fp8_output, bf16_output
    
    else:
        raise ValueError(f"Invalid packing mode: {packing_mode}")

def truncate_to_fp6_precision(bf16_tensor):
    """Truncate BF16 values to FP6 precision (3-bit mantissa)"""
    # Convert to int representation
    bf16_int = bf16_tensor.view(torch.int16)
    
    # Extract components
    sign = (bf16_int >> 15) & 0x1
    exponent = (bf16_int >> 7) & 0xFF
    mantissa = bf16_int & 0x7F
    
    # Truncate mantissa to 3 bits (keep top 3 bits)
    fp6_mantissa = mantissa >> 4
    
    # Extend back to 7 bits (same as unpacking process)
    truncated_mantissa = fp6_mantissa << 4
    
    # Reconstruct BF16 with truncated precision
    truncated_int = (sign << 15) | (exponent << 7) | truncated_mantissa
    return truncated_int.view(torch.bfloat16)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Triton FP6 packing implementation')
    parser.add_argument('--M', type=int, default=128, help='Matrix dimension M')
    parser.add_argument('--K', type=int, default=128, help='Matrix dimension K')
    parser.add_argument('--N', type=int, default=128, help='Matrix dimension N')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("CUDA not available, cannot run Triton implementation")
        return
    else:
        # Inputs in BF16
        activations = torch.randn(args.M, args.K, dtype=torch.bfloat16, device=device)
        weights = torch.randn(args.K, args.N, dtype=torch.bfloat16, device=device)

        tcast_fp6_act = ref_out(activations, device=device, cast_mode=tcast.mxfp6e2)
        tcast_fp6_wt = ref_out(weights, device=device, cast_mode=tcast.mxfp6e2)
        
        # Test all packing modes
        #for mode in [0, 1, 2, 3, 4]:
        for mode in [1]:
            print(f"\n{'='*60}")
            print(f"Testing Triton packing mode {mode}")
            print(f"{'='*60}")
            
            # Pack activations
            print("\nPacking activations...")
            act_packed, act_scaled = triton_fused_pack_kernel(tcast_fp6_act, mode)
            print(f"  Packed shape: {act_packed.shape}")
            print(f"  Packed dtype: {act_packed.dtype}")

            # Pack weights
            print("\nPacking weights...")
            wt_packed, wt_scaled = triton_fused_pack_kernel(tcast_fp6_wt, mode)
            print(f"  Packed shape: {wt_packed.shape}")
            print(f"  Packed dtype: {wt_packed.dtype}")
            
            # Test unpacking
            print(f"\nTesting Triton unpacking for mode {mode}...")
            # For modes with padding, pass the original shape
            if mode > 0:
                act_fp8, act_bf16 = triton_fused_unpack_gemm_kernel(act_packed, mode, tcast_fp6_act.tensor.shape)
                wt_fp8, wt_bf16 = triton_fused_unpack_gemm_kernel(wt_packed, mode, tcast_fp6_wt.tensor.shape)
            else:
                act_fp8, act_bf16 = triton_fused_unpack_gemm_kernel(act_packed, mode)
                wt_fp8, wt_bf16 = triton_fused_unpack_gemm_kernel(wt_packed, mode)
            
            from aiter import gemm_a8w8_CK
            _act_scale = (tcast_fp6_act.scaledata.scale).to(torch.uint8)
            _wt_scale = (tcast_fp6_wt.scaledata.scale).to(torch.uint8)
            _act_scale = _act_scale.view(torch.float8_e8m0fnu)
            _wt_scale = _wt_scale.view(torch.float8_e8m0fnu)
            _results8 = gemm_a8w8_CK(act_fp8, wt_fp8, _act_scale, _wt_scale)
            _result16 = tcast_fp6_act @ tcast_fp6_wt
            _error = torch.norm(_result16 - _results8.to(_result16.dtype)) / torch.norm(_result16)
            print(f"  GEMM result error (FP8 vs FP6 matmul): {_error.item():.6f}")

            print(f"  Unpacked activations FP8 shape: {act_fp8.shape}, dtype: {act_fp8.dtype}")
            print(f"  Unpacked activations BF16 shape: {act_bf16.shape}, dtype: {act_bf16.dtype}")
            print(f"  Unpacked weights FP8 shape: {wt_fp8.shape}, dtype: {wt_fp8.dtype}")
            print(f"  Unpacked weights BF16 shape: {wt_bf16.shape}, dtype: {wt_bf16.dtype}")

            diff = torch.abs(act_scaled - act_bf16.reshape(act_scaled.shape))
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")

            # Find positions where there are differences  
            nonzero_diff_mask = (diff != 0)
            nonzero_count = nonzero_diff_mask.sum().item()

            print(f"\nAnalyzing {nonzero_count} positions with differences:")

            if nonzero_count > 0:
                # Flatten for easier analysis
                act_scaled_flat = act_scaled.flatten()
                act_bf16_flat = act_bf16.reshape(act_scaled.shape).flatten()
                diff_flat = diff.flatten()
                nonzero_flat_mask = (diff_flat != 0)
                
                # Extract exponents for ALL positions
                scaled_int = act_scaled_flat.view(torch.int16)
                recon_int = act_bf16_flat.view(torch.int16)
                
                scaled_exponents = (scaled_int >> 7) & 0xFF
                recon_exponents = (recon_int >> 7) & 0xFF
                
                # Focus only on positions with differences
                diff_scaled_exp = scaled_exponents[nonzero_flat_mask]
                diff_recon_exp = recon_exponents[nonzero_flat_mask]
                
                # Check how many are the expected problematic mappings (0/124/125 → 126)
                expected_mapping = ((diff_scaled_exp == 0) | (diff_scaled_exp == 124) | (diff_scaled_exp == 125)) & (diff_recon_exp == 126)
                expected_count = expected_mapping.sum().item()
                
                print(f"  Expected mappings (0/124/125 → 126): {expected_count}")
                print(f"  Unexpected differences: {nonzero_count - expected_count}")
                
                if expected_count == nonzero_count:
                    print("  ✓ ALL differences are due to exponent mapping!")
                else:
                    print("  ✗ There are OTHER bugs beyond exponent mapping!")
                    
                    # Show examples of unexpected differences
                    unexpected_mask = ~expected_mapping
                    if unexpected_mask.any():
                        print("\n  Examples of unexpected differences:")
                        indices_with_diff = torch.where(nonzero_flat_mask)[0]
                        unexpected_indices = indices_with_diff[unexpected_mask][:5]
                        
                        for i, idx in enumerate(unexpected_indices):
                            idx_item = idx.item()
                            orig_val = act_scaled_flat[idx_item].item()
                            recon_val = act_bf16_flat[idx_item].item()
                            orig_exp = scaled_exponents[idx_item].item()
                            recon_exp = recon_exponents[idx_item].item()
                            diff_val = diff_flat[idx_item].item()
                            
                            print(f"    {i+1}: exp {orig_exp} → {recon_exp}, "
                                f"val {orig_val:.6f} → {recon_val:.6f}, "
                                f"diff {diff_val:.6f}")

            act_scaled_truncated = truncate_to_fp6_precision(act_scaled)
            act_bf16_reshaped = act_bf16.reshape(act_scaled.shape)
        
if __name__ == "__main__":
    main()
