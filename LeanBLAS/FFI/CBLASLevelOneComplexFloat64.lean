import LeanBLAS.FFI.FloatArray

set_option autoImplicit false

namespace BLAS.CBLAS

/-! # CBLAS Level 1 FFI Bindings for ComplexFloat64

Low-level FFI bindings to CBLAS Level 1 (vector-vector) operations for ComplexFloat64.
Functions use `z` prefix for double-precision complex. Complex numbers stored as
interleaved [re, im] pairs.
-/

/-- Conjugate dot product: result = Σ conj(X[i])·Y[i] -/
@[extern "leanblas_cblas_zdotc"]
opaque zdotc (N : USize) (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize) : ComplexFloat

/-- Unconjugated dot product: result = Σ X[i]·Y[i] -/
@[extern "leanblas_cblas_zdotu"]
opaque zdotu (N : USize) (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize) : ComplexFloat

/-- Euclidean norm: result = ||X||₂ = √(Σ|X[i]|²) -/
@[extern "leanblas_cblas_dznrm2"]
opaque dznrm2 (N : USize) (X : @& ComplexFloat64Array) (offX incX : USize) : Float

/-- Sum of absolute values: result = Σ(|Re(X[i])| + |Im(X[i])|) -/
@[extern "leanblas_cblas_dzasum"]
opaque dzasum (N : USize) (X : @& ComplexFloat64Array) (offX incX : USize) : Float

/-- Index of max absolute value (using |Re| + |Im|) -/
@[extern "leanblas_cblas_izamax"]
opaque izamax (N : USize) (X : @& ComplexFloat64Array) (offX incX : USize) : USize

/-- Swap vectors: X ↔ Y -/
@[extern "leanblas_cblas_zswap"]
opaque zswap (N : USize) (X : ComplexFloat64Array) (offX incX : USize)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array × ComplexFloat64Array

/-- Copy vector: Y := X -/
@[extern "leanblas_cblas_zcopy"]
opaque zcopy (N : USize) (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array

/-- Scaled addition: Y := αX + Y -/
@[extern "leanblas_cblas_zaxpy"]
opaque zaxpy (N : USize) (alpha : ComplexFloat) (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array

/-- Scale by complex: X := αX -/
@[extern "leanblas_cblas_zscal"]
opaque zscal (N : USize) (alpha : ComplexFloat) (X : ComplexFloat64Array) (offX incX : USize) : ComplexFloat64Array

/-- Scale by real: X := αX (α ∈ ℝ) -/
@[extern "leanblas_cblas_zdscal"]
opaque zdscal (N : USize) (alpha : Float) (X : ComplexFloat64Array) (offX incX : USize) : ComplexFloat64Array

end BLAS.CBLAS
