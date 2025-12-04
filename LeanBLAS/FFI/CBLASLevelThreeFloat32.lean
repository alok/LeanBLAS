import LeanBLAS.FFI.FloatArray
import LeanBLAS.Spec.LevelTwo

namespace BLAS.CBLAS

/-! # CBLAS Level 3 FFI Bindings for Float32

Low-level FFI bindings to CBLAS Level 3 (matrix-matrix) operations for Float32.
Functions use `s` prefix for single precision. O(n³) complexity.

Float32 (single precision) is essential for GPU-efficient computation since:
- Most GPUs have 2x higher throughput for Float32 vs Float64
- Neural network training typically uses single precision
- Memory bandwidth is halved compared to Float64
-/

/-- General matrix-matrix multiplication: C := α*A*B + β*C -/
@[extern "leanblas_cblas_sgemm"]
opaque sgemm (order : Order) (transA : Transpose) (transB : Transpose)
    (M : USize) (N : USize) (K : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (B : @& Float32Array) (offB : USize) (ldb : USize) (beta : Float)
    (C : Float32Array) (offC : USize) (ldc : USize) : Float32Array

/-- Symmetric matrix-matrix multiplication: C := α*A*B + β*C or C := α*B*A + β*C -/
@[extern "leanblas_cblas_ssymm"]
opaque ssymm (order : Order) (side : Side) (uplo : UpLo)
    (M : USize) (N : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (B : @& Float32Array) (offB : USize) (ldb : USize) (beta : Float)
    (C : Float32Array) (offC : USize) (ldc : USize) : Float32Array

/-- Symmetric rank-k update: C := α*A*Aᵀ + β*C or C := α*Aᵀ*A + β*C -/
@[extern "leanblas_cblas_ssyrk"]
opaque ssyrk (order : Order) (uplo : UpLo) (transA : Transpose)
    (N : USize) (K : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize) (beta : Float)
    (C : Float32Array) (offC : USize) (ldc : USize) : Float32Array

/-- Symmetric rank-2k update: C := α*A*Bᵀ + α*B*Aᵀ + β*C -/
@[extern "leanblas_cblas_ssyr2k"]
opaque ssyr2k (order : Order) (uplo : UpLo) (transA : Transpose)
    (N : USize) (K : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (B : @& Float32Array) (offB : USize) (ldb : USize) (beta : Float)
    (C : Float32Array) (offC : USize) (ldc : USize) : Float32Array

/-- Triangular matrix-matrix multiply: B := α*op(A)*B or B := α*B*op(A).
    - side: Left (A*B) or Right (B*A)
    - diag: Unit (diagonal assumed 1) or NonUnit (use actual diagonal values) -/
@[extern "leanblas_cblas_strmm"]
opaque strmm (order : Order) (side : Side) (uplo : UpLo)
    (transA : Transpose) (diag : Diag)
    (M : USize) (N : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (B : Float32Array) (offB : USize) (ldb : USize) : Float32Array

/-- Triangular solve: solve op(A)*X = α*B or X*op(A) = α*B for X.
    Solution overwrites B. -/
@[extern "leanblas_cblas_strsm"]
opaque strsm (order : Order) (side : Side) (uplo : UpLo)
    (transA : Transpose) (diag : Diag)
    (M : USize) (N : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (B : Float32Array) (offB : USize) (ldb : USize) : Float32Array

end BLAS.CBLAS
