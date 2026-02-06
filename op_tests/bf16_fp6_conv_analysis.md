# BF16 to FP6 Sub-normal Value Conversion Analysis

## Value Mappings
BF16 format: [s|e|m] → FP6 format: [s|e|m]

| Value  | BF16 Format      | FP6 Format   |
|--------|------------------|--------------|
| 0      | 0\|0\|0          | 0\|00\|000   |
| 0.125  | 0\|124\|000      | 0\|00\|001   |
| 0.25   | 0\|125\|000      | 0\|00\|010   |
| 0.375  | 0\|125\|100      | 0\|00\|011   |
| 0.5    | 0\|126\|000      | 0\|00\|100   |
| 0.625  | 0\|126\|010      | 0\|00\|101   |
| 0.75   | 0\|126\|100      | 0\|00\|110   |
| 0.875  | 0\|126\|110      | 0\|00\|111   |

## Conversion Algorithm

### Step 1:
For BF16 exponents 124, 125, 126, calculate offset from 124:

| BF16 Exponent | Offset (exp - 124) |
|---------------|-------------------|
| 124           | 0                 |
| 125           | 1                 |
| 126           | 2                 |

### Step 2: 
Take power of 2 of the offset: (just shift op)

| BF16 Exponent | Base Value (2^offset) |
|---------------|-----------------------|
| 124           | 1                     |
| 125           | 2                     |
| 126           | 4                     |

### Step 3: 
1. Take 3 MSB bits of BF16 mantissa
2. Shift right by 1
3. Add the base value from Step 2

| Value  | BF16 Mantissa | Top 3 bits | Shifted | Base | Result | FP6 Mantissa |
|--------|---------------|------------|---------|------|--------|--------------|
| 0.125  | 000           | 000        | 000     | 1    | 001    | 001 ✓        |
| 0.25   | 000           | 000        | 000     | 2    | 010    | 010 ✓        |
| 0.375  | 100           | 100        | 010     | 2    | 011    | 011 ✓        |
| 0.5    | 000           | 000        | 000     | 4    | 100    | 100 ✓        |
| 0.625  | 010           | 010        | 001     | 4    | 101    | 101 ✓        |
| 0.75   | 100           | 100        | 010     | 4    | 110    | 110 ✓        |
| 0.875  | 110           | 110        | 011     | 4    | 111    | 111 ✓        |
