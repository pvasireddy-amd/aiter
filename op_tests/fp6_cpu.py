import torch
import tcast
import numpy as np
import argparse

# Tensorcast based output
def ref_out(x, device="cpu", cast_mode=tcast.mxfp6e2):
    """Run tcast quantization on specified device"""
    x_device = x.to(device)
    c = tcast.cast(x_device, cast_mode)
    return c

def expand_block_scales(tensor_shape, block_scales, block_size=32):
    """
    Expand per-block scales to full per-element scales for a tensor.
    
    Args:
        tensor_shape (tuple): Shape of the tensor (e.g., (M, K))
        block_scales (torch.Tensor): 1D tensor of block multipliers
        block_size (int): Number of elements per block
        
    Returns:
        torch.Tensor: Full tensor of per-element scales, same shape as tensor
    """
    # Total number of elements
    total_elements = 1
    for dim in tensor_shape:
        total_elements *= dim
    
    # Create index for each element in flattened tensor
    element_indices = torch.arange(total_elements, device=block_scales.device)
    
    # Map each element to its block
    block_indices = element_indices // block_size
    block_indices = torch.clamp(block_indices, 0, block_scales.numel() - 1)
    
    # Assign block scale to each element
    scales_flat = block_scales.flatten()[block_indices]
    
    # Reshape to original tensor shape
    return scales_flat.view(tensor_shape)

def extract_fp6_components(bf16_tensor):
    """Extract sign, exponent, and mantissa from BF16 tensor for FP6 conversion"""
    # Convert to int representation
    bf16_int = bf16_tensor.view(torch.int16)
    
    # BF16 format: [sign(1)][exponent(8)][mantissa(7)]
    sign = (bf16_int >> 15) & 0x1
    exponent = (bf16_int >> 7) & 0xFF
    mantissa = bf16_int & 0x7F
    
    return sign, exponent, mantissa

def fp6_to_packed(scaled_bf16, packing_mode):
    """Convert scaled BF16 values to FP6 packed format"""
    sign, exponent, mantissa = extract_fp6_components(scaled_bf16)
    # Convert BF16 exponent to FP6 exponent
    # BF16 bias is 127, FP6 E2M3 bias is 1
    # Special case: BF16 exponent 0 maps to FP6 exponent 0 (zero/subnormal values)
    # Otherwise: FP6 exponent = BF16 exponent - 126
    fp6_exponent = torch.where((exponent == 0) | (exponent == 124) | (exponent == 125), 0, exponent - 126)
    
    # Check FP6 exponent is in valid range [0, 3] (2 bits) - fail if not
    invalid_exps = (fp6_exponent < 0) | (fp6_exponent > 3)
    if invalid_exps.any():
        invalid_count = invalid_exps.sum().item()
        min_exp = fp6_exponent.min().item()
        max_exp = fp6_exponent.max().item()
        raise ValueError(
            f"FP6 exponent out of valid range [0, 3]. "
            f"Found {invalid_count} invalid values. "
            f"Exponent range: [{min_exp}, {max_exp}]"
        )
    
    # Truncate mantissa from 7 bits to 3 bits (take top 3 bits)
    fp6_mantissa = mantissa >> 4  # Shift right by 4 to get top 3 bits
    # Get individual 6-bit FP6 values  
    fp6_packed_6bit = (sign << 5) | (fp6_exponent << 3) | fp6_mantissa
    
    if packing_mode == 0:
        # Pack into 6 bits: [sign(1)][exp(2)][mantissa(3)]
        return fp6_packed_6bit
    elif packing_mode == 1:
        # Flatten and pad to multiple of 4
        flat_fp6 = fp6_packed_6bit.flatten()
        n_elements = flat_fp6.numel()
        pad_size = (4 - (n_elements % 4)) % 4
        if pad_size > 0:
            flat_fp6 = torch.cat([flat_fp6, torch.zeros(pad_size, dtype=flat_fp6.dtype, device=flat_fp6.device)])
        
        # Reshape into groups of 4
        n_groups = flat_fp6.numel() // 4
        grouped_fp6 = flat_fp6.reshape(n_groups, 4)
        
        # Pack each group of 4 into 24-bit values
        packed_24bit = torch.zeros(n_groups, dtype=torch.int32, device=flat_fp6.device)
        # Extract only the FP6 bits (ignore MSB 00's)
        fp6_only = grouped_fp6 & 0x3F  # Mask to get only bits 5-0

        for i in range(4):
            packed_24bit |= fp6_only[:, i].to(torch.int32) << (i * 6)
        
        # Convert 24-bit values to 3 bytes each  
        n_bytes = n_groups * 3
        packed_bytes = torch.zeros(n_bytes, dtype=torch.uint8, device=flat_fp6.device)
        
        for i in range(n_groups):
            base_idx = i * 3
            val = packed_24bit[i]
            packed_bytes[base_idx] = (val & 0xFF).to(torch.uint8)           # Byte 0
            packed_bytes[base_idx + 1] = ((val >> 8) & 0xFF).to(torch.uint8)  # Byte 1  
            packed_bytes[base_idx + 2] = ((val >> 16) & 0xFF).to(torch.uint8) # Byte 2
        
        return packed_bytes.reshape(-1, 3)
    elif packing_mode == 2:
        # Mode 2: 32-bit packing (5 FP6 values + 2 padding bits)
        flat_fp6 = fp6_packed_6bit.flatten()
        n_elements = flat_fp6.numel()
        pad_size = (5 - (n_elements % 5)) % 5
        if pad_size > 0:
            flat_fp6 = torch.cat([flat_fp6, torch.zeros(pad_size, dtype=flat_fp6.dtype, device=flat_fp6.device)])
        
        # Reshape into groups of 5
        n_groups = flat_fp6.numel() // 5
        grouped_fp6 = flat_fp6.reshape(n_groups, 5)
        
        # Pack each group of 5 into 32-bit values (30 bits + 2 padding)
        packed_32bit = torch.zeros(n_groups, dtype=torch.int64, device=flat_fp6.device)
        
        for i in range(5):
            packed_32bit |= grouped_fp6[:, i].to(torch.int64) << (i * 6)
        
        return packed_32bit.to(torch.uint32)
    elif packing_mode == 3:
        # Mode 3: 96-bit packing (16 FP6 values)
        flat_fp6 = fp6_packed_6bit.flatten()
        n_elements = flat_fp6.numel()
        pad_size = (16 - (n_elements % 16)) % 16
        if pad_size > 0:
            flat_fp6 = torch.cat([flat_fp6, torch.zeros(pad_size, dtype=flat_fp6.dtype, device=flat_fp6.device)])
        
        # Reshape into groups of 16
        n_groups = flat_fp6.numel() // 16
        grouped_fp6 = flat_fp6.reshape(n_groups, 16)
        
        # Pack each group of 16 into 96 bits (3x32-bit values)
        packed_96bit = torch.zeros(n_groups, 3, dtype=torch.int64, device=flat_fp6.device)
        
        # Pack 16 6-bit values into 96 bits sequentially
        # We'll build a 96-bit value and then split it into 3 32-bit words
        for g in range(n_groups):
            bit_offset = 0
            for i in range(16):
                val = grouped_fp6[g, i].to(torch.int64)
                word_idx = bit_offset // 32
                bit_in_word = bit_offset % 32
                
                # Handle values that span across word boundaries
                if bit_in_word + 6 <= 32:
                    # Fits entirely in current word
                    packed_96bit[g, word_idx] |= val << bit_in_word
                else:
                    # Spans two words
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    
                    # Lower bits go in current word
                    packed_96bit[g, word_idx] |= (val & ((1 << bits_in_first) - 1)) << bit_in_word
                    # Upper bits go in next word
                    packed_96bit[g, word_idx + 1] |= (val >> bits_in_first)
                
                bit_offset += 6
        
        return packed_96bit.to(torch.uint32).reshape(-1, 3)
    elif packing_mode == 4:
        # Mode 4: 192-bit packing (32 FP6 values)
        flat_fp6 = fp6_packed_6bit.flatten()
        n_elements = flat_fp6.numel()
        pad_size = (32 - (n_elements % 32)) % 32
        if pad_size > 0:
            flat_fp6 = torch.cat([flat_fp6, torch.zeros(pad_size, dtype=flat_fp6.dtype, device=flat_fp6.device)])
        
        # Reshape into groups of 32
        n_groups = flat_fp6.numel() // 32
        grouped_fp6 = flat_fp6.reshape(n_groups, 32)
        
        # Pack each group of 32 into 192 bits (6x32-bit values)
        packed_192bit = torch.zeros(n_groups, 6, dtype=torch.int64, device=flat_fp6.device)
        
        # Pack 32 6-bit values into 192 bits sequentially
        for g in range(n_groups):
            bit_offset = 0
            for i in range(32):
                val = grouped_fp6[g, i].to(torch.int64)
                word_idx = bit_offset // 32
                bit_in_word = bit_offset % 32
                
                # Handle values that span across word boundaries
                if bit_in_word + 6 <= 32:
                    # Fits entirely in current word
                    packed_192bit[g, word_idx] |= val << bit_in_word
                else:
                    # Spans two words
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    
                    # Lower bits go in current word
                    packed_192bit[g, word_idx] |= (val & ((1 << bits_in_first) - 1)) << bit_in_word
                    # Upper bits go in next word
                    packed_192bit[g, word_idx + 1] |= (val >> bits_in_first)
                
                bit_offset += 6
        
        return packed_192bit.to(torch.uint32).reshape(-1, 6)

    
def cpu_fused_pack_kernel(tcast_tensor, packing_mode):
    """
    Pack FP6 values according to specified mode.
    
    Args:
        tcast_tensor: Tensorcast quantized tensor containing .tensor and .scaledata.scale
        packing_mode: 0 (8-bit), 1 (24-bit), or 2 (32-bit) packing
        
    Returns:
        tuple: (packed_tensor, fp6_scaled_tensor, original_shape)
            packed_tensor: Packed representation
            fp6_scaled_tensor: Scaled tensor for verification
            original_shape: Original tensor shape for unpacking
    """
    # Get tensors out of the fake tensor
    fp6_tensor = tcast_tensor.tensor
    scale_tensor = tcast_tensor.scaledata.scale

    device = "cpu"

    group_size = 32
    # For a group size of 32, multiply the fp6_tensor by the corresponding scale
    # 1 scale from scale_tensor for every 32 values in the fp6_tensor
    
    # Compute scale factors
    scales_f = torch.ldexp(torch.ones_like(scale_tensor, dtype=torch.bfloat16), 127 - scale_tensor)
    scales_full = expand_block_scales(fp6_tensor.shape, scales_f, block_size=group_size).to(device)
    fp6_scaled_tensor = (fp6_tensor * scales_full)
    
    # Check range: should be in +/- 7.5
    min_val = fp6_scaled_tensor.min().item()
    max_val = fp6_scaled_tensor.max().item()
    # print(min_val)
    # print(max_val)
    if min_val < -7.5 or max_val > 7.5:
        raise ValueError(
            f"Scaled tensor has values outside FP6 range [-7.5, 7.5]\n"
            f"  Range: [{min_val:.4f}, {max_val:.4f}]"
        )
    
    # Store original shape
    original_shape = fp6_tensor.shape
    
    # Flatten for processing
    flat_tensor = fp6_tensor.flatten()
    n_elements = flat_tensor.numel()

    # Packing modes
    if packing_mode == 0:
        # Mode 0: 8 bit packing
        # For each FP6 value, we represent it as 8-bit value by padding zeros to MSB
        fp6_packed = fp6_to_packed(fp6_scaled_tensor, 0)
        
        return fp6_packed, fp6_scaled_tensor, original_shape
    elif packing_mode in [1, 2, 3, 4]:
        fp6_packed = fp6_to_packed(fp6_scaled_tensor, packing_mode)
        return fp6_packed, fp6_scaled_tensor, original_shape
    else:
        raise ValueError(f"Invalid packing mode: {packing_mode}. Supported modes are 0, 1, 2, 3, and 4.")

def cpu_fused_unpack_gemm_kernel(packed_tensor, packing_mode, original_shape):
    """
    Unpack FP6 values from packed representation for use with FP8 GEMM kernels.
    
    Args:
        packed_tensor: Packed tensor from cpu_fused_pack_kernel
        packing_mode: 0 (8-bit), 1 (24-bit), or 2 (32-bit) packing
        original_shape: Original tensor shape to trim padded values
        
    Returns:
        tuple: (fp8_tensor, bf16_tensor)
            fp8_tensor: Tensor ready for FP8 GEMM (E4M3 format)
            bf16_tensor: BF16 representation for verification
    """
    device = packed_tensor.device
    
    packed_bytes = packed_tensor.flatten()

    if packing_mode == 0:
        # Mode 0: 8-bit packing
        # Values are in FP6 E2M3 format but stored in 8bit layout
        # Extract FP6 components: [00][S][EE][MMM]
        sign = (packed_bytes >> 5) & 0x1        # Extract sign bit
        fp6_exponent = (packed_bytes >> 3) & 0x3  # Extract 2-bit exponent  
        fp6_mantissa = packed_bytes & 0x7       # Extract 3-bit mantissa

        # Use helper functions for conversion
        fp8_tensor = convert_fp6_to_fp8_e4m3(sign, fp6_exponent, fp6_mantissa)
        bf16_tensor = convert_fp6_to_bf16(sign, fp6_exponent, fp6_mantissa)
        
        return fp8_tensor, bf16_tensor
        
    elif packing_mode == 1:
        # Mode 1: 24-bit packing (3 bytes contain 4 FP6 values)
        num_groups = packed_bytes.numel() // 3
        
        # Reconstruct 24-bit values from 3 bytes
        packed_24bit = torch.zeros(num_groups, dtype=torch.int32, device=device)
        packed_24bit |= packed_bytes[0::3].to(torch.int32)
        packed_24bit |= packed_bytes[1::3].to(torch.int32) << 8
        packed_24bit |= packed_bytes[2::3].to(torch.int32) << 16
        
        # Total number of FP6 values (4 per group)
        n_fp6_values = num_groups * 4
        
        # Initialize arrays for all values
        sign_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_exponent_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_mantissa_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        
        # Extract 4 FP6 values from each 24-bit group
        for i in range(4):
            # Extract 6-bit FP6 value
            shift = i * 6
            fp6_val = (packed_24bit >> shift) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = (fp6_val >> 5) & 0x1
            fp6_exponent = (fp6_val >> 3) & 0x3
            fp6_mantissa = fp6_val & 0x7
            
            # Store in arrays with proper indexing
            sign_all[i::4] = sign
            fp6_exponent_all[i::4] = fp6_exponent
            fp6_mantissa_all[i::4] = fp6_mantissa
        
        # Use helper functions for conversion
        fp8_tensor = convert_fp6_to_fp8_e4m3(sign_all, fp6_exponent_all, fp6_mantissa_all)
        bf16_tensor = convert_fp6_to_bf16(sign_all, fp6_exponent_all, fp6_mantissa_all)
        
        # Trim to original size
        original_size = np.prod(original_shape)
        fp8_tensor = fp8_tensor[:original_size]
        bf16_tensor = bf16_tensor[:original_size]
        
        return fp8_tensor, bf16_tensor
        
    elif packing_mode == 2:
        # Mode 2: 32-bit packing (5 FP6 values + 2 padding bits)
        # Get 32-bit packed values and convert to int32 for bitwise operations
        packed_32bit = packed_tensor.flatten().to(torch.int32)
        num_groups = packed_32bit.numel()
        
        # Total number of FP6 values (5 per group)
        n_fp6_values = num_groups * 5
        
        # Initialize arrays for all values
        sign_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_exponent_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_mantissa_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        
        # Extract 5 FP6 values from each 32-bit group
        for i in range(5):
            # Extract 6-bit FP6 value
            shift = i * 6
            fp6_val = (packed_32bit >> shift) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = (fp6_val >> 5) & 0x1
            fp6_exponent = (fp6_val >> 3) & 0x3
            fp6_mantissa = fp6_val & 0x7
            
            # Store in arrays with proper indexing
            sign_all[i::5] = sign
            fp6_exponent_all[i::5] = fp6_exponent
            fp6_mantissa_all[i::5] = fp6_mantissa
        
        # Use helper functions for conversion
        fp8_tensor = convert_fp6_to_fp8_e4m3(sign_all, fp6_exponent_all, fp6_mantissa_all)
        bf16_tensor = convert_fp6_to_bf16(sign_all, fp6_exponent_all, fp6_mantissa_all)
        
        # Trim to original size
        original_size = np.prod(original_shape)
        fp8_tensor = fp8_tensor[:original_size]
        bf16_tensor = bf16_tensor[:original_size]
        
        return fp8_tensor, bf16_tensor
        
    elif packing_mode == 3:
        # Mode 3: 96-bit packing (16 FP6 values in 3 32-bit words)
        # Reshape to process 3 words at a time, convert to int32 for bitwise operations
        packed_words = packed_tensor.flatten().view(-1, 3).to(torch.int32)
        num_groups = packed_words.shape[0]
        
        # Total number of FP6 values (16 per group)
        n_fp6_values = num_groups * 16
        
        # Initialize arrays for all values
        sign_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_exponent_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_mantissa_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        
        # Extract 16 FP6 values from each 96-bit group
        for g in range(num_groups):
            bit_offset = 0
            for i in range(16):
                word_idx = bit_offset // 32
                bit_in_word = bit_offset % 32
                
                # Extract 6-bit FP6 value (may span word boundaries)
                if bit_in_word + 6 <= 32:
                    # Fits entirely in current word
                    fp6_val = (packed_words[g, word_idx] >> bit_in_word) & 0x3F
                else:
                    # Spans two words
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    
                    # Extract bits from first word
                    first_part = (packed_words[g, word_idx] >> bit_in_word) & ((1 << bits_in_first) - 1)
                    # Extract bits from second word
                    second_part = packed_words[g, word_idx + 1] & ((1 << bits_in_second) - 1)
                    # Combine
                    fp6_val = first_part | (second_part << bits_in_first)
                
                # Extract FP6 components: [S][EE][MMM]
                sign = (fp6_val >> 5) & 0x1
                fp6_exponent = (fp6_val >> 3) & 0x3
                fp6_mantissa = fp6_val & 0x7
                
                # Store in arrays
                idx = g * 16 + i
                sign_all[idx] = sign
                fp6_exponent_all[idx] = fp6_exponent
                fp6_mantissa_all[idx] = fp6_mantissa
                
                bit_offset += 6
        
        # Use helper functions for conversion
        fp8_tensor = convert_fp6_to_fp8_e4m3(sign_all, fp6_exponent_all, fp6_mantissa_all)
        bf16_tensor = convert_fp6_to_bf16(sign_all, fp6_exponent_all, fp6_mantissa_all)
        
        # Trim to original size
        original_size = np.prod(original_shape)
        fp8_tensor = fp8_tensor[:original_size]
        bf16_tensor = bf16_tensor[:original_size]
        
        return fp8_tensor, bf16_tensor
    
    elif packing_mode == 4:
        # Mode 4: 192-bit packing (32 FP6 values in 6 32-bit words)
        # Reshape to process 6 words at a time, convert to int32 for bitwise operations
        packed_words = packed_tensor.flatten().view(-1, 6).to(torch.int32)
        num_groups = packed_words.shape[0]
        
        # Total number of FP6 values (32 per group)
        n_fp6_values = num_groups * 32
        
        # Initialize arrays for all values
        sign_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_exponent_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        fp6_mantissa_all = torch.zeros(n_fp6_values, dtype=torch.int32, device=device)
        
        # Extract 32 FP6 values from each 192-bit group
        for g in range(num_groups):
            bit_offset = 0
            for i in range(32):
                word_idx = bit_offset // 32
                bit_in_word = bit_offset % 32
                
                # Extract 6-bit FP6 value (may span word boundaries)
                if bit_in_word + 6 <= 32:
                    # Fits entirely in current word
                    fp6_val = (packed_words[g, word_idx] >> bit_in_word) & 0x3F
                else:
                    # Spans two words
                    bits_in_first = 32 - bit_in_word
                    bits_in_second = 6 - bits_in_first
                    
                    # Extract bits from first word
                    first_part = (packed_words[g, word_idx] >> bit_in_word) & ((1 << bits_in_first) - 1)
                    # Extract bits from second word
                    second_part = packed_words[g, word_idx + 1] & ((1 << bits_in_second) - 1)
                    # Combine
                    fp6_val = first_part | (second_part << bits_in_first)
                
                # Extract FP6 components: [S][EE][MMM]
                sign = (fp6_val >> 5) & 0x1
                fp6_exponent = (fp6_val >> 3) & 0x3
                fp6_mantissa = fp6_val & 0x7
                
                # Store in arrays
                idx = g * 32 + i
                sign_all[idx] = sign
                fp6_exponent_all[idx] = fp6_exponent
                fp6_mantissa_all[idx] = fp6_mantissa
                
                bit_offset += 6
        
        # Use helper functions for conversion
        fp8_tensor = convert_fp6_to_fp8_e4m3(sign_all, fp6_exponent_all, fp6_mantissa_all)
        bf16_tensor = convert_fp6_to_bf16(sign_all, fp6_exponent_all, fp6_mantissa_all)
        
        # Trim to original size
        original_size = np.prod(original_shape)
        fp8_tensor = fp8_tensor[:original_size]
        bf16_tensor = bf16_tensor[:original_size]
        
        return fp8_tensor, bf16_tensor
    
    else:
        raise ValueError(f"Invalid packing mode: {packing_mode}. Supported modes are 0, 1, 2, 3, and 4.")
def convert_fp6_to_fp8_e4m3(sign, fp6_exponent, fp6_mantissa):
    """
    Convert FP6 components to FP8 E4M3 format.
    
    Args:
        sign: Sign bit(s)
        fp6_exponent: 2-bit FP6 exponent(s)
        fp6_mantissa: 3-bit FP6 mantissa(s)
        
    Returns:
        fp8_tensor: FP8 E4M3 format tensor
    """
    # Add 6 to exponent for E4M3 bias
    fp8_exponent = fp6_exponent + 6  # Now 4 bits (range 6-9)
    # Pack into FP8 E4M3: [S][EEEE][MMM]
    fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp6_mantissa
    return fp8_packed.to(torch.uint8)

def convert_fp6_to_bf16(sign, fp6_exponent, fp6_mantissa):
    """
    Convert FP6 components to BF16 format.
    
    Args:
        sign: Sign bit(s)
        fp6_exponent: 2-bit FP6 exponent(s)
        fp6_mantissa: 3-bit FP6 mantissa(s)
        
    Returns:
        bf16_tensor: BF16 format tensor
    """
    # New exponent = FP6 exponent + 126 (to account for BF16 bias of 127 - FP6 bias of 1)
    bf16_exponent = fp6_exponent + 126
    
    # Extend mantissa from 3 bits to 7 bits by adding 0s to the right
    bf16_mantissa = fp6_mantissa << 4  # Shift left by 4 to add 4 zero bits
    
    # Pack into BF16: [S][EEEEEEEE][MMMMMMM]
    bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
    return bf16_packed.to(torch.int16).view(torch.bfloat16)

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
    parser = argparse.ArgumentParser(description='CPU FP6 packing implementation')
    parser.add_argument('--M', type=int, default=128, help='Matrix dimension M')
    parser.add_argument('--K', type=int, default=128, help='Matrix dimension K')
    parser.add_argument('--N', type=int, default=128, help='Matrix dimension N')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    args = parser.parse_args()
    
    # Inputs in BF16
    activations = torch.randn(args.M, args.K, dtype=torch.bfloat16, device=args.device)
    weights = torch.randn(args.K, args.N, dtype=torch.bfloat16, device=args.device)
        
    # Fake quantized (tensorcast based) outputs
    tcast_fp6_act = ref_out(activations, device=args.device, cast_mode=tcast.mxfp6e2)
    tcast_fp6_wt = ref_out(weights, device=args.device, cast_mode=tcast.mxfp6e2)

    # Testing all packing modes
    for mode in [0, 1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Testing packing mode {mode}")
        print(f"{'='*60}")
        
        # Call to cpu fused pack kernel for activations
        print("\nPacking activations...")
        act_packed, act_scaled, act_original_shape = cpu_fused_pack_kernel(tcast_fp6_act, mode)
        print(f"  Packed shape: {act_packed.shape}")
        print(f"  Packed dtype: {act_packed.dtype}")
        
        # Call to cpu fused pack kernel for weights
        print("\nPacking weights...")
        wt_packed, wt_scaled, wt_original_shape = cpu_fused_pack_kernel(tcast_fp6_wt, mode)
        print(f"  Packed shape: {wt_packed.shape}")
        print(f"  Packed dtype: {wt_packed.dtype}")
        
        # Verify range of original values
        act_min = tcast_fp6_act.tensor.min().item()
        act_max = tcast_fp6_act.tensor.max().item()
        wt_min = tcast_fp6_wt.tensor.min().item()
        wt_max = tcast_fp6_wt.tensor.max().item()
        
        # print(f"\nFP6 value ranges:")
        # print(f"  Activations: [{act_min:.4f}, {act_max:.4f}]")
        # print(f"  Weights: [{wt_min:.4f}, {wt_max:.4f}]")
        
        # Test unpacking
        print(f"\nTesting unpacking for mode {mode}...")
        act_fp8, act_bf16 = cpu_fused_unpack_gemm_kernel(act_packed, mode, act_original_shape)
        wt_fp8, wt_bf16 = cpu_fused_unpack_gemm_kernel(wt_packed, mode, wt_original_shape)
        
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
