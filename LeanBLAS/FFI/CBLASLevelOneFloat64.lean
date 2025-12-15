import LeanBLAS.FFI.FloatArray

set_option autoImplicit false

namespace BLAS.CBLAS

/-! # CBLAS Level 1 FFI Bindings for Float64

Low-level FFI bindings to CBLAS Level 1 (vector-vector) operations for Float64.
Functions use `d` prefix for double precision. Direct calls to optimized BLAS kernels.
-/

/-- Dot product: result = X·Y -/
@[extern "leanblas_cblas_ddot"]
opaque ddot (N : USize) (X : @& Float64Array) (offX incX : USize) (Y : @& Float64Array) (offY incY : USize) : Float

/-- Euclidean norm: result = ||X||₂ -/
@[extern "leanblas_cblas_dnrm2"]
opaque dnrm2 (N : USize) (X : @& Float64Array) (offX incX : USize) : Float

/-- Sum of absolute values: result = Σ|X[i]| -/
@[extern "leanblas_cblas_dasum"]
opaque dasum (N : USize) (X : @& Float64Array) (offX incX : USize) : Float

/-- Index of max absolute value -/
@[extern "leanblas_cblas_idamax"]
opaque idamax (N : USize) (X : @& Float64Array) (offX incX : USize) : USize

/-- Swap vectors: X ↔ Y -/
@[extern "leanblas_cblas_dswap"]
opaque dswap (N : USize) (X : Float64Array) (offX incX : USize) (Y : Float64Array) (offY incY : USize) : Float64Array × Float64Array

/-- Copy vector: Y := X -/
@[extern "leanblas_cblas_dcopy"]
opaque dcopy (N : USize) (X : @& Float64Array) (offX incX : USize) (Y : Float64Array) (offY incY : USize) : Float64Array

/-- Scaled addition: Y := αX + Y -/
@[extern "leanblas_cblas_daxpy"]
opaque daxpy (N : USize) (a : Float) (X : @& Float64Array) (offX incX : USize) (Y : Float64Array) (offY incY : USize) : Float64Array

/-- Construct Givens rotation: returns (r, z, c, s) -/
@[extern "leanblas_cblas_drotg"]
opaque drotg (a : Float) (b : Float) : (Float × Float × Float × Float)

/-- Construct modified Givens rotation -/
@[extern "leanblas_cblas_drotmg"]
opaque drotmg (d1 : Float) (d2 : Float) (b1 : Float) (b2 : Float) : (Float × Float × Float × Float × Float)

/-- Apply Givens rotation: (X,Y) := G(c,s)·(X,Y) -/
@[extern "leanblas_cblas_drot"]
opaque drot (N : USize) (X : Float64Array) (offX incX : USize) (Y : Float64Array) (offY incY : USize) (c s : Float) : Float64Array × Float64Array

/-- Scale vector: X := αX -/
@[extern "leanblas_cblas_dscal"]
opaque dscal (N : USize) (a : Float) (X : Float64Array) (offX incX : USize) : Float64Array

/-! ## Extended Operations (non-standard BLAS) -/

/-- Create constant vector: result[i] = α -/
@[extern "leanblas_cblas_dconst"]
opaque dconst (N : USize) (alpha : Float) : Float64Array

/-- Sum of elements: result = ΣX[i] -/
@[extern "leanblas_cblas_dsum"]
opaque dsum (N : USize) (X : @&Float64Array) (offX : USize) (incX : USize) : Float

/-- Scaled addition with two scalars: result = αX + βY -/
@[extern "leanblas_cblas_daxpby"]
opaque daxpby (N : USize) (alpha : Float) (X : Float64Array) (offX : USize) (incX : USize)
                          (beta : Float)  (Y : Float64Array) (offY : USize) (incY : USize) : Float64Array

/-- Scale and add constant: result = αX + β -/
@[extern "leanblas_cblas_dscaladd"]
opaque dscaladd (N : USize) (alpha : Float) (X : Float64Array) (offX : USize) (incX : USize) (beta : Float) : Float64Array

/-- Index of max real value -/
@[extern "leanblas_cblas_dimax_re"]
opaque dimaxRe (N : USize) (X : @&Float64Array) (offX : USize) (incX : USize) : USize

/-- Index of min real value -/
@[extern "leanblas_cblas_dimin_re"]
opaque diminRe (N : USize) (X : @&Float64Array) (offX : USize) (incX : USize) : USize

/-- Element-wise multiply: result[i] = X[i]·Y[i] -/
@[extern "leanblas_cblas_dmul"]
opaque dmul (N : USize) (X : Float64Array) (offX : USize) (incX : USize) (Y : Float64Array) (offY : USize) (incY : USize) : Float64Array

/-- Element-wise divide: result[i] = X[i]/Y[i] -/
@[extern "leanblas_cblas_ddiv"]
opaque ddiv (N : USize) (X : Float64Array) (offX : USize) (incX : USize) (Y : Float64Array) (offY : USize) (incY : USize) : Float64Array

/-- Element-wise inverse: result[i] = 1/X[i] -/
@[extern "leanblas_cblas_dinv"]
opaque dinv (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array

/-- Element-wise absolute value: result[i] = |X[i]| -/
@[extern "leanblas_cblas_dabs"]
opaque dabs (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array

/-- Element-wise square root: result[i] = √X[i] -/
@[extern "leanblas_cblas_dsqrt"]
opaque dsqrt (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array

/-- Element-wise exponential: result[i] = eˣ⁽ⁱ⁾ -/
@[extern "leanblas_cblas_dexp"]
opaque dexp (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array

/-- Element-wise natural log: result[i] = ln(X[i]) -/
@[extern "leanblas_cblas_dlog"]
opaque dlog (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array

/-- Element-wise sine: result[i] = sin(X[i]) -/
@[extern "leanblas_cblas_dsin"]
opaque dsin (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array

/-- Element-wise cosine: result[i] = cos(X[i]) -/
@[extern "leanblas_cblas_dcos"]
opaque dcos (N : USize) (X : Float64Array) (offX : USize) (incX : USize) : Float64Array
