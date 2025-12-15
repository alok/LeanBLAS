# Complex BLAS Support in LeanBLAS

## Overview

LeanBLAS provides comprehensive support for complex number operations through the BLAS (Basic Linear Algebra Subprograms) interface. This implementation supports double-precision complex numbers (`ComplexFloat64`) with full FFI integration to optimized BLAS libraries.

## Architecture

### Complex Number Representation

Complex numbers in LeanBLAS use the `ComplexFloat` type:

```lean
structure ComplexFloat where
  x : Float  -- Real part
  y : Float  -- Imaginary part
```

### Array Types

- `ComplexFloat64Array`: A wrapper around `ByteArray` ensuring proper 16-byte alignment for complex numbers
- Memory layout: Interleaved format `[re₀, im₀, re₁, im₁, ...]` compatible with BLAS

### Type Classes

1. **LevelOneData**: Core BLAS Level 1 operations
   - `get`: Extract complex numbers from arrays
   - `dot`: Conjugate dot product (zdotc)
   - `nrm2`: Euclidean norm
   - `asum`: Sum of absolute values
   - `iamax`: Index of maximum absolute value
   - `swap`, `copy`, `axpy`, `scal`: Vector operations

2. **LevelOneDataExt**: Extended operations
   - `const`: Create constant vectors
   - `sum`: Sum all elements
   - `axpby`: Generalized AXPY (Y := αX + βY)
   - `imaxRe/imaxIm`: Find max real/imaginary parts
   - `iminRe/iminIm`: Find min real/imaginary parts
   - Element-wise operations: `mul`, `div`, `inv`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`

## FFI Bindings

### Naming Convention

Complex BLAS functions follow standard naming:
- `z` prefix: double-precision complex
- `c` prefix: single-precision complex (not yet implemented)
- Examples: `zdotc`, `zdotu`, `zgemm`

### Key Functions

```lean
-- Conjugate dot product
@[extern "leanblas_cblas_zdotc"]
opaque zdotc : USize → ComplexFloat64Array → USize → USize → 
               ComplexFloat64Array → USize → USize → ComplexFloat

-- Unconjugated dot product
@[extern "leanblas_cblas_zdotu"]
opaque zdotu : USize → ComplexFloat64Array → USize → USize → 
               ComplexFloat64Array → USize → USize → ComplexFloat

-- 2-norm (returns real value)
@[extern "leanblas_cblas_dznrm2"]
opaque dznrm2 : USize → ComplexFloat64Array → USize → USize → Float

-- Complex scaling
@[extern "leanblas_cblas_zscal"]
opaque zscal : USize → ComplexFloat → ComplexFloat64Array → USize → USize → ComplexFloat64Array
```

## Usage Examples

### Creating Complex Arrays

```lean
-- Using the #c64 macro
let x := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩, ⟨5.0, 6.0⟩]

-- Manual construction
let arr := ComplexFloatArray.ofArray #[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
let x := ComplexFloatArray.toComplexFloat64Array arr
```

### Basic Operations

```lean
-- Dot product
let dot := zdotc 3 x 0 1 y 0 1  -- Conjugate dot product
let dot_u := zdotu 3 x 0 1 y 0 1  -- Unconjugated dot product

-- Norm
let norm := dznrm2 3 x 0 1

-- Scaling
let scaled := zscal 3 ⟨2.0, -1.0⟩ x 0 1
```

### Extended Operations

```lean
open LevelOneDataExt

-- Sum all elements
let sum := sum (Array := ComplexFloat64Array) 3 x 0 1

-- Element-wise operations
let sqrt_x := sqrt (Array := ComplexFloat64Array) 3 x 0 1
let exp_x := exp (Array := ComplexFloat64Array) 3 x 0 1
```

### Stride and Offset

All operations support stride and offset parameters for working with sub-vectors:

```lean
-- Use every other element starting from index 1
let dot_strided := zdotc 2 x 1 2 y 1 2
```

## Implementation Details

### Memory Safety

- All array accesses check bounds
- FFI functions handle stride and offset calculations
- ByteArray ensures proper alignment

### Performance Considerations

- Zero-copy FFI interface
- Direct calls to optimized BLAS libraries
- Minimal overhead from Lean wrapper

### Complex Arithmetic

The implementation provides full complex arithmetic:
- Basic operations: `+`, `-`, `*`, `/`
- Functions: `conj`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`
- Special values: `ComplexFloat.zero`, `ComplexFloat.one`, `ComplexFloat.I`

## Testing

Comprehensive test suites are provided:
- `LeanBLASTest/ComplexLevel1Comprehensive.lean`: Comprehensive complex Level 1 tests
- `LeanBLASTest/ComplexLevel2Comprehensive.lean`: Comprehensive complex Level 2 tests
- `LeanBLASTest/ComplexNumericalValidation.lean`: NumPy-derived numerical validation vectors
- `LeanBLASTest/ComplexValidation.lean`: Complex arithmetic, strides, and extended ops
- `LeanBLASTest/ComplexEdgeCases.lean`: Special values, branch cuts, overflow/underflow

## Future Work

1. Single-precision complex support (`ComplexFloat32`)
2. Level 2 and Level 3 BLAS operations for complex matrices
3. Optimized element-wise operations using SIMD
4. Formal verification of complex arithmetic properties
