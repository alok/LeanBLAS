import LeanBLAS.FFI.FloatArray

namespace BLAS.CBLAS

/-! # CBLAS Level 2 FFI Bindings for ComplexFloat64

Low-level FFI bindings to CBLAS Level 2 (matrix-vector) operations for ComplexFloat64.
Functions use `z` prefix for double-precision complex. Supports Hermitian operations.
-/

/-- General matrix-vector: Y := αAX + βY -/
@[extern "leanblas_cblas_zgemv"]
opaque zgemv (order : Order) (transA : Transpose) (M N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : @& ComplexFloat64Array) (offX incX : USize) (beta : ComplexFloat)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array

/-- Hermitian matrix-vector: Y := αAX + βY (A = Aᴴ) -/
@[extern "leanblas_cblas_zhemv"]
opaque zhemv (order : Order) (uplo : UpLo) (N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : @& ComplexFloat64Array) (offX incX : USize) (beta : ComplexFloat)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array

/-- Triangular matrix-vector: X := op(A)X -/
@[extern "leanblas_cblas_ztrmv"]
opaque ztrmv (order : Order) (uplo : UpLo) (transA : Transpose) (diag : Diag)
             (N : USize) (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : ComplexFloat64Array) (offX incX : USize) : ComplexFloat64Array

/-- Triangular solve: X := op(A)⁻¹X -/
@[extern "leanblas_cblas_ztrsv"]
opaque ztrsv (order : Order) (uplo : UpLo) (transA : Transpose) (diag : Diag)
             (N : USize) (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : ComplexFloat64Array) (offX incX : USize) : ComplexFloat64Array

/-- Rank-1 update with conjugate: A := αXȲᵀ + A -/
@[extern "leanblas_cblas_zgerc"]
opaque zgerc (order : Order) (M N : USize) (alpha : ComplexFloat)
             (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize)
             (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

/-- Rank-1 update without conjugate: A := αXYᵀ + A -/
@[extern "leanblas_cblas_zgeru"]
opaque zgeru (order : Order) (M N : USize) (alpha : ComplexFloat)
             (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize)
             (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

/-- Hermitian rank-1 update: A := αXX̄ᵀ + A (α ∈ ℝ) -/
@[extern "leanblas_cblas_zher"]
opaque zher (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
            (X : @& ComplexFloat64Array) (offX incX : USize)
            (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

/-- Hermitian rank-2 update: A := αXȲᵀ + ᾱYX̄ᵀ + A -/
@[extern "leanblas_cblas_zher2"]
opaque zher2 (order : Order) (uplo : UpLo) (N : USize) (alpha : ComplexFloat)
             (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize)
             (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

end BLAS.CBLAS
