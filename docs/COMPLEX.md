# Complex Number Support in LeanBLAS

## Overview

LeanBLAS provides comprehensive support for complex number operations through the `ComplexFloat` type and `ComplexFloat64Array` array type. This support includes all Level 1, 2, and 3 BLAS operations with proper handling of complex arithmetic, conjugation, and Hermitian operations.

## Complex Number Types

### ComplexFloat

The basic complex number type with Float64 real and imaginary parts:

```lean
structure ComplexFloat where
  re : Float
  im : Float
deriving BEq, Inhabited, Repr
```

### ComplexFloat64Array  

An array type for storing complex numbers in interleaved format:

```lean
structure ComplexFloat64Array where
  data : ByteArray
  h : data.size % 16 = 0
```

Complex numbers are stored as consecutive [real, imaginary] pairs in memory, compatible with standard BLAS libraries.

## Creating Complex Arrays

### Manual Construction

```lean
-- Create a complex array with specific values
let arr := ComplexFloat64Array.mk (ByteArray.mk #[
  0, 0, 0, 0, 0, 0, 240, 63,  -- 1.0 (real part)
  0, 0, 0, 0, 0, 0, 0, 64,    -- 2.0 (imaginary part)
  -- This represents the complex number 1.0 + 2.0i
]) (by decide)
```

### Using Helper Functions

```lean
-- Create complex numbers
let z1 : ComplexFloat := ⟨1.0, 2.0⟩    -- 1 + 2i
let z2 : ComplexFloat := ⟨3.0, -1.0⟩   -- 3 - i

-- Complex conjugate
let z_conj := z1.conj  -- 1 - 2i
```

## Complex BLAS Operations

### Level 1 (Vector-Vector)

#### Complex Dot Products

```lean
-- zdotu: Dot product without conjugation
-- result = ∑ xᵢ * yᵢ
let dot_u := zdotu N x offX incX y offY incY

-- zdotc: Dot product with conjugation  
-- result = ∑ conj(xᵢ) * yᵢ
let dot_c := zdotc N x offX incX y offY incY
```

#### Complex Vector Norm

```lean
-- dznrm2: Returns real-valued 2-norm of complex vector
-- result = √(∑ |xᵢ|²) where |xᵢ|² = re² + im²
let norm : Float := dznrm2 N x offX incX
```

#### Other Level 1 Operations

- `zscal`: Scale by complex scalar
- `zaxpy`: Complex y := α*x + y  
- `zcopy`: Copy complex vector
- `zswap`: Swap complex vectors
- `dzasum`: Sum of magnitudes (returns real)
- `izamax`: Index of max magnitude element

### Level 2 (Matrix-Vector)

#### General Matrix-Vector Multiplication

```lean
-- zgemv: y := α * op(A) * x + β * y
-- where op(A) can be A, A^T, or A^H (conjugate transpose)
let result := zgemv order transA M N alpha A offA lda x offX incX beta y offY incY
```

#### Hermitian Matrix-Vector Multiplication

```lean
-- zhemv: y := α * A * x + β * y
-- where A is Hermitian (A = A^H)
let result := zhemv order uplo N alpha A offA lda x offX incX beta y offY incY
```

#### Rank Updates

```lean
-- zgerc: A := α * x * conj(y)^T + A (conjugated rank-1 update)
let A_new := zgerc order M N alpha x offX incX y offY incY A offA lda

-- zgeru: A := α * x * y^T + A (unconjugated rank-1 update)  
let A_new := zgeru order M N alpha x offX incX y offY incY A offA lda

-- zher: A := α * x * conj(x)^T + A (Hermitian rank-1 update, α must be real)
let A_new := zher order uplo N alpha x offX incX A offA lda

-- zher2: A := α * x * conj(y)^T + conj(α) * y * conj(x)^T + A
let A_new := zher2 order uplo N alpha x offX incX y offY incY A offA lda
```

### Level 3 (Matrix-Matrix)

#### General Matrix Multiplication

```lean
-- zgemm: C := α * op(A) * op(B) + β * C
let C_new := zgemm order transA transB M N K alpha A offA lda B offB ldb beta C offC ldc
```

#### Hermitian Matrix Multiplication  

```lean
-- zhemm: C := α * A * B + β * C (or α * B * A + β * C)
-- where A is Hermitian
let C_new := hemm order side uplo M N alpha A offA lda B offB ldb beta C offC ldc
```

#### Hermitian Rank-k Updates

```lean
-- zherk: C := α * A * A^H + β * C
-- where α and β must be real, and C is Hermitian
let C_new := herk order uplo trans N K alpha A offA lda beta C offC ldc

-- zher2k: C := α * A * B^H + conj(α) * B * A^H + β * C  
-- where β must be real, and C is Hermitian
let C_new := her2k order uplo trans N K alpha A offA lda B offB ldb beta C offC ldc
```

## Important Differences from Real BLAS

### Conjugation Behavior

1. **Dot Products**: 
   - `zdotu`: No conjugation
   - `zdotc`: Conjugates first vector

2. **Transpose Operations**:
   - `Transpose.Trans`: Regular transpose
   - `Transpose.ConjTrans`: Conjugate transpose (Hermitian transpose)

3. **Hermitian Operations**:
   - Only store upper or lower triangle
   - Require real scalars for certain parameters
   - `zherk` and `zher2k` require real `alpha` and `beta`
   - `zher` requires real `alpha`

### Memory Layout

Complex numbers are stored in interleaved format:
```
[re₀, im₀, re₁, im₁, re₂, im₂, ...]
```

Each complex number occupies 16 bytes (two Float64 values).

## Performance Considerations

1. **Cache Efficiency**: Interleaved storage keeps real and imaginary parts together
2. **SIMD Operations**: Layout is compatible with vectorized complex arithmetic
3. **Memory Alignment**: ComplexFloat64Array ensures 16-byte alignment
4. **Conjugate Operations**: May have different performance characteristics than non-conjugate variants

## Example: Solving Complex Linear Systems

```lean
-- Solve A * x = b where A is triangular
def solve_triangular_complex (A : ComplexFloat64Array) (b : ComplexFloat64Array) : IO ComplexFloat64Array := do
  let x := zcopy b.size b 0 1  -- Copy b to x
  -- Solve using triangular solve
  let solution := ztrsv Order.RowMajor UpLo.Upper Transpose.NoTrans false 
                       b.size A 0 b.size x 0 1
  return solution
```

## Testing Complex Operations

See `test/ComplexBLASTest.lean` for comprehensive examples of using complex BLAS operations.

## FFI Implementation Details

The C wrapper functions in `c/levelone.c`, `c/leveltwo.c`, and `c/levelthree.c` handle:
- Unpacking Lean ComplexFloat objects to C99 `double complex`
- Proper memory management with `ensure_exclusive_byte_array`
- Conversion between Lean and CBLAS enumerations
- Handling of stride and offset calculations

## Future Enhancements

1. **Single Precision**: Support for ComplexFloat32
2. **Mixed Precision**: Operations mixing Float64 and ComplexFloat64
3. **Optimizations**: Complex-specific optimizations for small matrices
4. **Convenience Functions**: Higher-level wrappers for common patterns