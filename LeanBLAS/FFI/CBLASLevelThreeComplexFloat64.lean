import LeanBLAS.FFI.FloatArray

namespace BLAS.CBLAS

/-! # CBLAS Level 3 FFI Bindings for ComplexFloat64

Low-level FFI bindings to CBLAS Level 3 (matrix-matrix) operations for ComplexFloat64.
Functions use `z` prefix for double-precision complex. O(n³) complexity.
Hermitian operations use conjugate transpose (A = Aᴴ).
-/

/-- General matrix multiply: C := αop(A)op(B) + βC -/
@[extern "leanblas_cblas_zgemm"]
opaque zgemm (order : Order) (transA transB : Transpose) (M N K : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Symmetric matrix multiply: C := αAB + βC or C := αBA + βC (A symmetric) -/
@[extern "leanblas_cblas_zsymm"]
opaque zsymm (order : Order) (side : Side) (uplo : UpLo) (M N : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Hermitian matrix multiply: C := αAB + βC or C := αBA + βC (A = Aᴴ) -/
@[extern "leanblas_cblas_zhemm"]
opaque zhemm (order : Order) (side : Side) (uplo : UpLo) (M N : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Symmetric rank-k update: C := αAAᵀ + βC or C := αAᵀA + βC -/
@[extern "leanblas_cblas_zsyrk"]
opaque zsyrk (order : Order) (uplo : UpLo) (transA : Transpose) (N K : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Hermitian rank-k update: C := αAAᴴ + βC or C := αAᴴA + βC (α,β ∈ ℝ) -/
@[extern "leanblas_cblas_zherk"]
opaque zherk (order : Order) (uplo : UpLo) (transA : Transpose) (N K : USize)
             (alpha : Float)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (beta : Float)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Symmetric rank-2k update: C := αABᵀ + αBAᵀ + βC -/
@[extern "leanblas_cblas_zsyr2k"]
opaque zsyr2k (order : Order) (uplo : UpLo) (transA : Transpose) (N K : USize)
              (alpha : ComplexFloat)
              (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
              (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
              (beta : ComplexFloat)
              (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Hermitian rank-2k update: C := αABᴴ + ᾱBAᴴ + βC (β ∈ ℝ) -/
@[extern "leanblas_cblas_zher2k"]
opaque zher2k (order : Order) (uplo : UpLo) (transA : Transpose) (N K : USize)
              (alpha : ComplexFloat)
              (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
              (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
              (beta : Float)
              (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- Triangular matrix multiply: B := αop(A)B or B := αBop(A) -/
@[extern "leanblas_cblas_ztrmm"]
opaque ztrmm (order : Order) (side : Side) (uplo : UpLo) (transA : Transpose) (diag : Diag)
             (M N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : ComplexFloat64Array) (offB : USize) (ldb : USize) : ComplexFloat64Array

/-- Triangular solve: solve op(A)X = αB or Xop(A) = αB for X -/
@[extern "leanblas_cblas_ztrsm"]
opaque ztrsm (order : Order) (side : Side) (uplo : UpLo) (transA : Transpose) (diag : Diag)
             (M N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : ComplexFloat64Array) (offB : USize) (ldb : USize) : ComplexFloat64Array

end BLAS.CBLAS
