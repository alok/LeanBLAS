import LeanBLAS.FFI.FloatArray
import LeanBLAS.Spec.LevelTwo

set_option autoImplicit false

namespace BLAS.CBLAS

/-! ## Level 3 BLAS Float64 FFI Declarations -/

-- General matrix-matrix multiplication
-- C := alpha*A*B + beta*C
@[extern "leanblas_cblas_dgemm"]
opaque dgemm (order : Order) (transA : Transpose) (transB : Transpose)
    (M : USize) (N : USize) (K : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (B : @& Float64Array) (offB : USize) (ldb : USize) (beta : Float)
    (C : Float64Array) (offC : USize) (ldc : USize) : Float64Array

-- Symmetric matrix-matrix multiplication
-- C := alpha*A*B + beta*C  or  C := alpha*B*A + beta*C
@[extern "leanblas_cblas_dsymm"]
opaque dsymm (order : Order) (side : Side) (uplo : UpLo)
    (M : USize) (N : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (B : @& Float64Array) (offB : USize) (ldb : USize) (beta : Float)
    (C : Float64Array) (offC : USize) (ldc : USize) : Float64Array

/-- Symmetric rank-k update: C := α*A*Aᵀ + β*C or C := α*Aᵀ*A + β*C -/
@[extern "leanblas_cblas_dsyrk"]
opaque dsyrk (order : Order) (uplo : UpLo) (transA : Transpose)
    (N : USize) (K : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize) (beta : Float)
    (C : Float64Array) (offC : USize) (ldc : USize) : Float64Array

/-- Symmetric rank-2k update: C := α*A*Bᵀ + α*B*Aᵀ + β*C -/
@[extern "leanblas_cblas_dsyr2k"]
opaque dsyr2k (order : Order) (uplo : UpLo) (transA : Transpose)
    (N : USize) (K : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (B : @& Float64Array) (offB : USize) (ldb : USize) (beta : Float)
    (C : Float64Array) (offC : USize) (ldc : USize) : Float64Array

/-- Triangular matrix-matrix multiply: B := α*op(A)*B or B := α*B*op(A).
    - side: Left (A*B) or Right (B*A)
    - diag: Unit (diagonal assumed 1) or NonUnit (use actual diagonal values) -/
@[extern "leanblas_cblas_dtrmm"]
opaque dtrmm (order : Order) (side : Side) (uplo : UpLo)
    (transA : Transpose) (diag : Diag)
    (M : USize) (N : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (B : Float64Array) (offB : USize) (ldb : USize) : Float64Array

/-- Triangular solve: solve op(A)*X = α*B or X*op(A) = α*B for X.
    Solution overwrites B. -/
@[extern "leanblas_cblas_dtrsm"]
opaque dtrsm (order : Order) (side : Side) (uplo : UpLo)
    (transA : Transpose) (diag : Diag)
    (M : USize) (N : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (B : Float64Array) (offB : USize) (ldb : USize) : Float64Array
