import torch
import triton
import triton.language as tl
import time
import numpy as np

@triton.jit
def triton_fp6_unpack_nested_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format using NESTED CONDITIONALS (original approach)"""
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
            
            # 1. Convert to FP8 format using NESTED CONDITIONALS
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

            # 2. Convert to BF16 format
            bf16_exponent = tl.where((fp8_exponent == 0), 0, fp8_exponent + 120)
            bf16_mantissa = fp8_mantissa << 4
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)


@triton.jit
def triton_fp6_unpack_reduced_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format using REDUCED CONDITIONALS (intermediate approach)"""
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
            
            # 1. Convert to FP8 format using REDUCED CONDITIONALS (only 2 levels deep)
            # First handle the subnormal case (fp6_exponent == 0)
            subnormal_exp = tl.where(fp6_mantissa == 0, 0,
                                    tl.where(fp6_mantissa == 1, 4,
                                    tl.where(fp6_mantissa < 4, 5, 6)))
            
            subnormal_man = tl.where(fp6_mantissa < 2, 0,
                                    tl.where(fp6_mantissa < 4, (fp6_mantissa & 1) << 2,
                                    (fp6_mantissa & 3) << 1))
            
            # Then select between subnormal and normal cases
            fp8_exponent = tl.where(fp6_exponent == 0, subnormal_exp, fp6_exponent + 6)
            fp8_mantissa = tl.where(fp6_exponent == 0, subnormal_man, fp6_mantissa)
            
            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp8_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))

            # 2. Convert to BF16 format
            bf16_exponent = tl.where((fp8_exponent == 0), 0, fp8_exponent + 120)
            bf16_mantissa = fp8_mantissa << 4
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)


@triton.jit
def triton_fp6_unpack_lut_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    exp_lut_ptr, man_lut_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format using LOOKUP TABLES (optimized approach)"""
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
            
            # 1. Convert to FP8 format using LOOKUP TABLES
            fp8_exponent = tl.where(fp6_exponent == 0, 
                                   tl.load(exp_lut_ptr + fp6_mantissa), 
                                   fp6_exponent + 6)
            fp8_mantissa = tl.where(fp6_exponent == 0,
                                   tl.load(man_lut_ptr + fp6_mantissa),
                                   fp6_mantissa)

            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp8_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))

            # 2. Convert to BF16 format
            bf16_exponent = tl.where((fp8_exponent == 0), 0, fp8_exponent + 120)
            bf16_mantissa = fp8_mantissa << 4
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)


@triton.jit
def triton_fp6_unpack_packed_lut_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    combined_lut_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format using PACKED LOOKUP TABLE (single 16-bit LUT)"""
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
            
            # 1. Convert to FP8 format using PACKED LOOKUP TABLE
            # Single lookup for both exponent and mantissa
            combined_val = tl.load(combined_lut_ptr + fp6_mantissa).to(tl.int32)
            lut_exponent = combined_val & 0xFF
            lut_mantissa = (combined_val >> 8) & 0xFF
            
            fp8_exponent = tl.where(fp6_exponent == 0, lut_exponent, fp6_exponent + 6)
            fp8_mantissa = tl.where(fp6_exponent == 0, lut_mantissa, fp6_mantissa)

            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp8_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))

            # 2. Convert to BF16 format
            bf16_exponent = tl.where((fp8_exponent == 0), 0, fp8_exponent + 120)
            bf16_mantissa = fp8_mantissa << 4
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)


@triton.jit
def triton_fp6_unpack_tl_tensor_kernel(
    packed_ptr, fp8_ptr, bf16_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format using BIT MANIPULATION (in-kernel lookup tables)"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Pack lookup tables into single integers
    # For exponent LUT: [0, 4, 5, 5, 6, 6, 6, 6]
    # We only need mantissa 0-3 for subnormal case, so: [0, 4, 5, 5]
    # Packed as 4-bit values: 0x5540 (little-endian: 0, 4, 5, 5)
    exp_lut_packed: tl.constexpr = 0x5540
    
    # For mantissa LUT: [0, 0, 0, 4, 0, 2, 4, 6]
    # We only need mantissa 0-3 for subnormal case, so: [0, 0, 0, 4]
    # But we also need mantissa 4-7: [0, 2, 4, 6]
    # Packed as 4-bit values: 0x4000 for 0-3, 0x6420 for 4-7
    man_lut_packed_low: tl.constexpr = 0x4000   # [0, 0, 0, 4]
    man_lut_packed_high: tl.constexpr = 0x6420  # [0, 2, 4, 6]
    
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
            
            # 1. Convert to FP8 format using BIT MANIPULATION
            # For subnormal case (fp6_exponent == 0), extract from packed LUT
            # Extract 4-bit value from packed LUT based on mantissa
            subnormal_exp = tl.where(fp6_mantissa < 4,
                                    (exp_lut_packed >> (fp6_mantissa * 4)) & 0xF,
                                    6)  # mantissa >= 4 always maps to exp=6
            
            # For mantissa LUT, handle both low (0-3) and high (4-7) cases
            subnormal_man = tl.where(fp6_mantissa < 4,
                                    (man_lut_packed_low >> (fp6_mantissa * 4)) & 0xF,
                                    (man_lut_packed_high >> ((fp6_mantissa - 4) * 4)) & 0xF)
            
            # Select between subnormal and normal cases
            fp8_exponent = tl.where(fp6_exponent == 0, subnormal_exp, fp6_exponent + 6)
            fp8_mantissa = tl.where(fp6_exponent == 0, subnormal_man, fp6_mantissa)

            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp8_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))

            # 2. Convert to BF16 format
            bf16_exponent = tl.where((fp8_exponent == 0), 0, fp8_exponent + 120)
            bf16_mantissa = fp8_mantissa << 4
            bf16_packed = (sign << 15) | (bf16_exponent << 7) | bf16_mantissa
            bf16_value = bf16_packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
            tl.store(bf16_ptr + idx, bf16_value)


def benchmark_fp6_unpacking(packed_tensor, n_warmup=50, n_runs=200):
    """Benchmark all five approaches: nested, reduced, lookup table, packed lookup table, and bit manipulation"""
    device = packed_tensor.device
    
    # Setup for unpacking
    packed_bytes = packed_tensor.flatten()
    n_groups = packed_bytes.numel() // 3
    n_elements = n_groups * 4
    
    # Create separate lookup tables
    exp_lut = torch.tensor([0, 4, 5, 5, 6, 6, 6, 6], dtype=torch.int8, device=device)
    man_lut = torch.tensor([0, 0, 0, 4, 0, 2, 4, 6], dtype=torch.int8, device=device)
    
    # Create packed lookup table (16-bit: [mantissa(8)][exponent(8)])
    combined_lut = torch.tensor([
        (0 << 8) | 0,  # mantissa=0, exp=0
        (0 << 8) | 4,  # mantissa=0, exp=4  
        (0 << 8) | 5,  # mantissa=0, exp=5
        (4 << 8) | 5,  # mantissa=4, exp=5
        (0 << 8) | 6,  # mantissa=0, exp=6
        (2 << 8) | 6,  # mantissa=2, exp=6
        (4 << 8) | 6,  # mantissa=4, exp=6
        (6 << 8) | 6,  # mantissa=6, exp=6
    ], dtype=torch.int16, device=device)
    
    # Allocate output tensors
    fp8_output_nested = torch.empty(n_elements, dtype=torch.uint8, device=device)
    bf16_output_nested = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
    
    fp8_output_reduced = torch.empty(n_elements, dtype=torch.uint8, device=device)
    bf16_output_reduced = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
    
    fp8_output_lut = torch.empty(n_elements, dtype=torch.uint8, device=device)
    bf16_output_lut = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
    
    fp8_output_packed_lut = torch.empty(n_elements, dtype=torch.uint8, device=device)
    bf16_output_packed_lut = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
    
    fp8_output_tl_tensor = torch.empty(n_elements, dtype=torch.uint8, device=device)
    bf16_output_tl_tensor = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
    
    # Setup kernel launch parameters
    grid = (n_groups,)
    BLOCK_SIZE = 1
    
    # Warmup for nested conditionals
    print("Warming up nested conditionals kernel...")
    for _ in range(n_warmup):
        triton_fp6_unpack_nested_kernel[grid](
            packed_bytes, fp8_output_nested, bf16_output_nested,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    
    # Benchmark nested conditionals
    print("Benchmarking nested conditionals...")
    start_time = time.time()
    for _ in range(n_runs):
        triton_fp6_unpack_nested_kernel[grid](
            packed_bytes, fp8_output_nested, bf16_output_nested,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    nested_time = (time.time() - start_time) / n_runs * 1000  # Convert to ms
    
    # Warmup for reduced conditionals
    print("Warming up reduced conditionals kernel...")
    for _ in range(n_warmup):
        triton_fp6_unpack_reduced_kernel[grid](
            packed_bytes, fp8_output_reduced, bf16_output_reduced,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    
    # Benchmark reduced conditionals
    print("Benchmarking reduced conditionals...")
    start_time = time.time()
    for _ in range(n_runs):
        triton_fp6_unpack_reduced_kernel[grid](
            packed_bytes, fp8_output_reduced, bf16_output_reduced,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    reduced_time = (time.time() - start_time) / n_runs * 1000  # Convert to ms
    
    # Warmup for lookup tables
    print("Warming up lookup table kernel...")
    for _ in range(n_warmup):
        triton_fp6_unpack_lut_kernel[grid](
            packed_bytes, fp8_output_lut, bf16_output_lut,
            exp_lut, man_lut,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    
    # Benchmark lookup tables
    print("Benchmarking lookup tables...")
    start_time = time.time()
    for _ in range(n_runs):
        triton_fp6_unpack_lut_kernel[grid](
            packed_bytes, fp8_output_lut, bf16_output_lut,
            exp_lut, man_lut,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    lut_time = (time.time() - start_time) / n_runs * 1000  # Convert to ms
    
    # Warmup for packed lookup table
    print("Warming up packed lookup table kernel...")
    for _ in range(n_warmup):
        triton_fp6_unpack_packed_lut_kernel[grid](
            packed_bytes, fp8_output_packed_lut, bf16_output_packed_lut,
            combined_lut,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    
    # Benchmark packed lookup table
    print("Benchmarking packed lookup table...")
    start_time = time.time()
    for _ in range(n_runs):
        triton_fp6_unpack_packed_lut_kernel[grid](
            packed_bytes, fp8_output_packed_lut, bf16_output_packed_lut,
            combined_lut,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    packed_lut_time = (time.time() - start_time) / n_runs * 1000  # Convert to ms
    
    # Warmup for bit manipulation
    print("Warming up bit manipulation kernel...")
    for _ in range(n_warmup):
        triton_fp6_unpack_tl_tensor_kernel[grid](
            packed_bytes, fp8_output_tl_tensor, bf16_output_tl_tensor,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    
    # Benchmark bit manipulation
    print("Benchmarking bit manipulation...")
    start_time = time.time()
    for _ in range(n_runs):
        triton_fp6_unpack_tl_tensor_kernel[grid](
            packed_bytes, fp8_output_tl_tensor, bf16_output_tl_tensor,
            n_groups, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    tl_tensor_time = (time.time() - start_time) / n_runs * 1000  # Convert to ms
    
    # Verify outputs match
    fp8_match_reduced = torch.allclose(fp8_output_nested, fp8_output_reduced)
    bf16_match_reduced = torch.allclose(bf16_output_nested, bf16_output_reduced)
    fp8_match_lut = torch.allclose(fp8_output_nested, fp8_output_lut)
    bf16_match_lut = torch.allclose(bf16_output_nested, bf16_output_lut)
    fp8_match_packed_lut = torch.allclose(fp8_output_nested, fp8_output_packed_lut)
    bf16_match_packed_lut = torch.allclose(bf16_output_nested, bf16_output_packed_lut)
    fp8_match_tl_tensor = torch.allclose(fp8_output_nested, fp8_output_tl_tensor)
    bf16_match_tl_tensor = torch.allclose(bf16_output_nested, bf16_output_tl_tensor)
    
    return (nested_time, reduced_time, lut_time, packed_lut_time, tl_tensor_time,
            fp8_match_reduced, bf16_match_reduced, fp8_match_lut, bf16_match_lut,
            fp8_match_packed_lut, bf16_match_packed_lut, fp8_match_tl_tensor, bf16_match_tl_tensor)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("CUDA not available, cannot run Triton benchmarks")
        return
    
    print("FP6 to FP8 Conversion Timing Comparison")
    print("=" * 60)
    print("Comparing nested conditionals vs lookup tables")
    print()
    
    # Test different tensor sizes
    sizes = [
        (128, 128),    # Small
        (512, 512),    # Medium
        (1024, 1024),  # Large
        (2048, 2048),  # Extra large
        (4096, 4096)
    ]
    
    results = []
    
    for M, K in sizes:
        print(f"\nTesting size: {M}x{K}")
        print("-" * 40)
        
        # Create dummy packed data (simulating FP6 packed in 24-bit format)
        n_elements = M * K
        n_groups = (n_elements + 3) // 4  # 4 FP6 values per group
        n_bytes = n_groups * 3  # 3 bytes per group
        
        # Create random packed data
        packed_tensor = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device=device)
        packed_tensor = packed_tensor.reshape(-1, 3)
        
        # Run benchmarks
        (nested_time, reduced_time, lut_time, packed_lut_time, tl_tensor_time,
         fp8_match_reduced, bf16_match_reduced, fp8_match_lut, bf16_match_lut,
         fp8_match_packed_lut, bf16_match_packed_lut, fp8_match_tl_tensor, bf16_match_tl_tensor) = benchmark_fp6_unpacking(packed_tensor)
        
        # Calculate speedups
        speedup_reduced = nested_time / reduced_time
        speedup_lut = nested_time / lut_time
        speedup_packed_lut = nested_time / packed_lut_time
        speedup_tl_tensor = nested_time / tl_tensor_time
        
        # Store results
        results.append({
            'size': f"{M}x{K}",
            'elements': n_elements,
            'nested_time': nested_time,
            'reduced_time': reduced_time,
            'lut_time': lut_time,
            'packed_lut_time': packed_lut_time,
            'tl_tensor_time': tl_tensor_time,
            'speedup_reduced': speedup_reduced,
            'speedup_lut': speedup_lut,
            'speedup_packed_lut': speedup_packed_lut,
            'speedup_tl_tensor': speedup_tl_tensor,
            'fp8_match_reduced': fp8_match_reduced,
            'bf16_match_reduced': bf16_match_reduced,
            'fp8_match_lut': fp8_match_lut,
            'bf16_match_lut': bf16_match_lut,
            'fp8_match_packed_lut': fp8_match_packed_lut,
            'bf16_match_packed_lut': bf16_match_packed_lut,
            'fp8_match_tl_tensor': fp8_match_tl_tensor,
            'bf16_match_tl_tensor': bf16_match_tl_tensor
        })
        
        print(f"Nested conditionals (4 levels):  {nested_time:.3f} ms")
        print(f"Reduced conditionals (2 levels): {reduced_time:.3f} ms (speedup: {speedup_reduced:.2f}x)")
        print(f"Lookup tables (2 lookups):       {lut_time:.3f} ms (speedup: {speedup_lut:.2f}x)")
        print(f"Packed lookup table (1 lookup):  {packed_lut_time:.3f} ms (speedup: {speedup_packed_lut:.2f}x)")
        print(f"Bit manipulation (in-kernel):    {tl_tensor_time:.3f} ms (speedup: {speedup_tl_tensor:.2f}x)")
        print(f"Results match: Reduced={fp8_match_reduced}/{bf16_match_reduced}, LUT={fp8_match_lut}/{bf16_match_lut}, Packed LUT={fp8_match_packed_lut}/{bf16_match_packed_lut}, Bit manip={fp8_match_tl_tensor}/{bf16_match_tl_tensor}")
    
    # Summary table
    print("\n" + "=" * 150)
    print("SUMMARY")
    print("=" * 150)
    print(f"{'Size':>10} | {'Elements':>10} | {'Nested (ms)':>12} | {'Reduced (ms)':>13} | {'LUT (ms)':>10} | {'Packed LUT':>12} | {'Bit manip':>12} | {'Speed (R)':>10} | {'Speed (L)':>10} | {'Speed (P)':>10} | {'Speed (B)':>10}")
    print("-" * 150)
    
    for result in results:
        print(f"{result['size']:>10} | {result['elements']:>10,} | "
              f"{result['nested_time']:>12.3f} | {result['reduced_time']:>13.3f} | "
              f"{result['lut_time']:>10.3f} | {result['packed_lut_time']:>12.3f} | "
              f"{result['tl_tensor_time']:>12.3f} | "
              f"{result['speedup_reduced']:>10.2f}x | "
              f"{result['speedup_lut']:>10.2f}x | "
              f"{result['speedup_packed_lut']:>10.2f}x | "
              f"{result['speedup_tl_tensor']:>10.2f}x")
    
    # Average speedups
    avg_speedup_reduced = np.mean([r['speedup_reduced'] for r in results])
    avg_speedup_lut = np.mean([r['speedup_lut'] for r in results])
    avg_speedup_packed_lut = np.mean([r['speedup_packed_lut'] for r in results])
    avg_speedup_tl_tensor = np.mean([r['speedup_tl_tensor'] for r in results])
    print("-" * 150)
    print(f"{'Average speedup:':>68} | {'':>12} | {'':>12} | {avg_speedup_reduced:>10.2f}x | {avg_speedup_lut:>10.2f}x | {avg_speedup_packed_lut:>10.2f}x | {avg_speedup_tl_tensor:>10.2f}x")
    
if __name__ == "__main__":
    main()
