import LeanBLAS.FFI.FloatArray

namespace BLAS.CBLAS

/-! # CBLAS Level 2 FFI Bindings for Float32

Low-level FFI bindings to CBLAS Level 2 (matrix-vector) operations for Float32.
Functions use `s` prefix for single precision. O(n²) complexity.

Float32 (single precision) is essential for GPU-efficient computation since:
- Most GPUs have 2x higher throughput for Float32 vs Float64
- Neural network training typically uses single precision
- Memory bandwidth is halved compared to Float64
-/

/-- General matrix-vector: Y := αAX + βY -/
@[extern "leanblas_cblas_sgemv"]
opaque sgemv (order : Order) (transA : Transpose) (M : USize) (N : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (X : @& Float32Array) (offX incX : USize) (beta : Float)
    (Y : Float32Array) (offY incY : USize) : Float32Array

/-- Band matrix-vector multiply -/
@[extern "leanblas_cblas_sgbmv"]
opaque sbmv (order : Order) (transA : Transpose) (N : USize) (M : USize) (KL KU : USize) (alpha : Float)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (X : @& Float32Array) (offX incX : USize) (beta : Float)
    (Y : Float32Array) (offY incY : USize) : Float32Array

/-- Triangular matrix-vector multiply: X := op(A)X where A is triangular.
    - order: Row or column major storage
    - uplo: Upper or Lower triangular
    - transA: No transpose, transpose, or conjugate transpose
    - diag: Unit (diagonal assumed 1) or NonUnit (use actual diagonal values)
    - N: Order of matrix A -/
@[extern "leanblas_cblas_strmv"]
opaque strmv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (X : Float32Array) (offX incX : USize) : Float32Array

/-- Triangular banded matrix-vector multiply: X := op(A)X where A is triangular banded.
    - K: Number of super-diagonals (upper) or sub-diagonals (lower) -/
@[extern "leanblas_cblas_stbmv"]
opaque stbmv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N K : USize)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (X : Float32Array) (offX incX : USize) : Float32Array

/-- Triangular packed matrix-vector multiply: X := op(A)X where A is triangular in packed format.
    Packed format stores only the triangular part in a 1D array. -/
@[extern "leanblas_cblas_stpmv"]
opaque stpmv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float32Array) (offA : USize)
    (X : Float32Array) (offX incX : USize) : Float32Array

/-- Triangular solve: solve op(A)X = B for X, result stored in X.
    A is triangular, X contains B on entry and solution on exit. -/
@[extern "leanblas_cblas_strsv"]
opaque strsv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (X : Float32Array) (offX incX : USize) : Float32Array

/-- Triangular banded solve: solve op(A)X = B for X where A is triangular banded. -/
@[extern "leanblas_cblas_stbsv"]
opaque stbsv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N K : USize)
    (A : @& Float32Array) (offA : USize) (lda : USize)
    (X : Float32Array) (offX incX : USize) : Float32Array

/-- Triangular packed solve: solve op(A)X = B for X where A is triangular in packed format. -/
@[extern "leanblas_cblas_stpsv"]
opaque stpsv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float32Array) (offA : USize)
    (X : Float32Array) (offX incX : USize) : Float32Array

/-- General rank-1 update: A := αXYᵀ + A -/
@[extern "leanblas_cblas_sger"]
opaque sger (order : Order) (M : USize) (N : USize) (alpha : Float)
    (X : @& Float32Array) (offX incX : USize)
    (Y : @& Float32Array) (offY incY : USize)
    (A : Float32Array) (offA : USize) (lda : USize) : Float32Array

/-- Symmetric rank-1 update: A := αXXᵀ + A -/
@[extern "leanblas_cblas_ssyr"]
opaque ssyr (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
    (X : @& Float32Array) (offX incX : USize)
    (A : Float32Array) (offA : USize) (lda : USize) : Float32Array

/-- Symmetric rank-2 update: A := αXYᵀ + αYXᵀ + A -/
@[extern "leanblas_cblas_ssyr2"]
opaque ssyr2 (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
    (X : @& Float32Array) (offX incX : USize)
    (Y : @& Float32Array) (offY incY : USize)
    (A : Float32Array) (offA : USize) (lda : USize) : Float32Array

/-! ## Extended Operations -/

/-- Convert packed to dense matrix -/
@[extern "leanblas_cblas_spacked_to_dense"]
opaque spackedToDense (N : USize) (uplo : UpLo)
  (orderAp : Order) (Ap : @& Float32Array)
  (orderA : Order) (A : Float32Array) (offA : USize) (lds : USize) : Float32Array

/-- Convert dense to packed matrix -/
@[extern "leanblas_cblas_sdense_to_packed"]
opaque sdenseToPacked (N : USize) (uplo : UpLo)
  (orderA : Order) (A : @& Float32Array) (offA : USize) (lda : USize)
  (orderAp : Order) (Ap : Float32Array) : Float32Array

/-- General rank-1 update on packed triangular: Ap := αXYᵀ + Ap -/
@[extern "leanblas_cblas_sgpr"]
opaque sgpr (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
    (X : @& Float32Array) (offX incX : USize)
    (Y : @& Float32Array) (offY incY : USize)
    (Ap : Float32Array) (offA : USize) : Float32Array

end BLAS.CBLAS
