import LeanBLAS.FFI.FloatArray

set_option autoImplicit false

namespace BLAS.CBLAS

/-! # CBLAS Level 1 FFI Bindings for Float32

Low-level FFI bindings to CBLAS Level 1 (vector-vector) operations for Float32.
Functions use `s` prefix for single precision. Direct calls to optimized BLAS kernels.

Note: Lean's `Float` type is 64-bit, so we convert to/from 32-bit at the FFI boundary.
This enables GPU-efficient computation while maintaining Lean's numeric consistency.
-/

/-- Dot product: result = X·Y (single precision) -/
@[extern "leanblas_cblas_sdot"]
opaque sdot (N : USize) (X : @& Float32Array) (offX incX : USize) (Y : @& Float32Array) (offY incY : USize) : Float

/-- Euclidean norm: result = ||X||₂ (single precision) -/
@[extern "leanblas_cblas_snrm2"]
opaque snrm2 (N : USize) (X : @& Float32Array) (offX incX : USize) : Float

/-- Sum of absolute values: result = Σ|X[i]| (single precision) -/
@[extern "leanblas_cblas_sasum"]
opaque sasum (N : USize) (X : @& Float32Array) (offX incX : USize) : Float

/-- Index of max absolute value (single precision) -/
@[extern "leanblas_cblas_isamax"]
opaque isamax (N : USize) (X : @& Float32Array) (offX incX : USize) : USize

/-- Swap vectors: X ↔ Y (single precision) -/
@[extern "leanblas_cblas_sswap"]
opaque sswap (N : USize) (X : Float32Array) (offX incX : USize) (Y : Float32Array) (offY incY : USize) : Float32Array × Float32Array

/-- Copy vector: Y := X (single precision) -/
@[extern "leanblas_cblas_scopy"]
opaque scopy (N : USize) (X : @& Float32Array) (offX incX : USize) (Y : Float32Array) (offY incY : USize) : Float32Array

/-- Scaled addition: Y := αX + Y (single precision) -/
@[extern "leanblas_cblas_saxpy"]
opaque saxpy (N : USize) (a : Float) (X : @& Float32Array) (offX incX : USize) (Y : Float32Array) (offY incY : USize) : Float32Array

/-- Construct Givens rotation: returns (r, z, c, s) (single precision) -/
@[extern "leanblas_cblas_srotg"]
opaque srotg (a : Float) (b : Float) : (Float × Float × Float × Float)

/-- Apply Givens rotation: (X,Y) := G(c,s)·(X,Y) (single precision) -/
@[extern "leanblas_cblas_srot"]
opaque srot (N : USize) (X : Float32Array) (offX incX : USize) (Y : Float32Array) (offY incY : USize) (c s : Float) : Float32Array × Float32Array

/-- Scale vector: X := αX (single precision) -/
@[extern "leanblas_cblas_sscal"]
opaque sscal (N : USize) (a : Float) (X : Float32Array) (offX incX : USize) : Float32Array

/-- Construct modified Givens rotation (single precision) -/
@[extern "leanblas_cblas_srotmg"]
opaque srotmg (d1 : Float) (d2 : Float) (b1 : Float) (b2 : Float) : (Float × Float × Float × Float × Float)

/-! ## Extended Operations (non-standard BLAS) -/

/-- Create constant vector: result[i] = α (single precision) -/
@[extern "leanblas_cblas_sconst"]
opaque sconst (N : USize) (alpha : Float) : Float32Array

/-- Sum of elements: result = ΣX[i] (single precision) -/
@[extern "leanblas_cblas_ssum"]
opaque ssum (N : USize) (X : @& Float32Array) (offX : USize) (incX : USize) : Float

/-- Scaled addition with two scalars: result = αX + βY (single precision) -/
@[extern "leanblas_cblas_saxpby"]
opaque saxpby (N : USize) (alpha : Float) (X : Float32Array) (offX : USize) (incX : USize)
                          (beta : Float)  (Y : Float32Array) (offY : USize) (incY : USize) : Float32Array

/-- Scale and add constant: result = αX + β (single precision) -/
@[extern "leanblas_cblas_sscaladd"]
opaque sscaladd (N : USize) (alpha : Float) (X : Float32Array) (offX : USize) (incX : USize) (beta : Float) : Float32Array

/-- Index of max real value (single precision) -/
@[extern "leanblas_cblas_simax_re"]
opaque simaxRe (N : USize) (X : @&Float32Array) (offX : USize) (incX : USize) : USize

/-- Index of min real value (single precision) -/
@[extern "leanblas_cblas_simin_re"]
opaque siminRe (N : USize) (X : @&Float32Array) (offX : USize) (incX : USize) : USize

/-- Element-wise multiply: result[i] = X[i]·Y[i] (single precision) -/
@[extern "leanblas_cblas_smul"]
opaque smul (N : USize) (X : Float32Array) (offX : USize) (incX : USize) (Y : Float32Array) (offY : USize) (incY : USize) : Float32Array

/-- Element-wise divide: result[i] = X[i]/Y[i] (single precision) -/
@[extern "leanblas_cblas_sdiv"]
opaque sdiv (N : USize) (X : Float32Array) (offX : USize) (incX : USize) (Y : Float32Array) (offY : USize) (incY : USize) : Float32Array

/-- Element-wise inverse: result[i] = 1/X[i] (single precision) -/
@[extern "leanblas_cblas_sinv"]
opaque sinv (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

/-- Element-wise absolute value: result[i] = |X[i]| (single precision) -/
@[extern "leanblas_cblas_sabs"]
opaque sabs (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

/-- Element-wise square root: result[i] = √X[i] (single precision) -/
@[extern "leanblas_cblas_ssqrt"]
opaque ssqrt (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

/-- Element-wise exponential: result[i] = eˣ⁽ⁱ⁾ (single precision) -/
@[extern "leanblas_cblas_sexp"]
opaque sexp (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

/-- Element-wise natural log: result[i] = ln(X[i]) (single precision) -/
@[extern "leanblas_cblas_slog"]
opaque slog (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

/-- Element-wise sine: result[i] = sin(X[i]) (single precision) -/
@[extern "leanblas_cblas_ssin"]
opaque ssin (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

/-- Element-wise cosine: result[i] = cos(X[i]) (single precision) -/
@[extern "leanblas_cblas_scos"]
opaque scos (N : USize) (X : Float32Array) (offX : USize) (incX : USize) : Float32Array

end BLAS.CBLAS
