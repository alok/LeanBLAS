import LeanBLAS.FFI.FloatArray
import LeanBLAS.Spec.LevelTwo

set_option autoImplicit false

namespace BLAS.CBLAS

/-! # CBLAS Level 2 FFI Bindings for Float64

Low-level FFI bindings to CBLAS Level 2 (matrix-vector) operations for Float64.
Functions use `d` prefix for double precision. O(n²) complexity.
-/

/-- General matrix-vector: Y := αAX + βY -/
@[extern "leanblas_cblas_dgemv"]
opaque dgemv (order : Order) (transA : Transpose) (M : USize) (N : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (X : @& Float64Array) (offX incX : USize) (beta : Float)
    (Y : Float64Array) (offY incY : USize) : Float64Array

/-- Band matrix-vector multiply -/
@[extern "leanblas_cblas_dgbmv"]
opaque dbmv (order : Order) (transA : Transpose) (N : USize) (M : USize) (KL KU : USize) (alpha : Float)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (X : @& Float64Array) (offX incX : USize) (beta : Float)
    (Y : Float64Array) (offY incY : USize) : Float64Array

/-- Triangular matrix-vector multiply: X := op(A)X where A is triangular.
    - order: Row or column major storage
    - uplo: Upper or Lower triangular
    - transA: No transpose, transpose, or conjugate transpose
    - diag: Unit (diagonal assumed 1) or NonUnit (use actual diagonal values)
    - N: Order of matrix A -/
@[extern "leanblas_cblas_dtrmv"]
opaque dtrmv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (X : Float64Array) (offX incX : USize) : Float64Array

/-- Triangular banded matrix-vector multiply: X := op(A)X where A is triangular banded.
    - K: Number of super-diagonals (upper) or sub-diagonals (lower) -/
@[extern "leanblas_cblas_dtbmv"]
opaque dtbmv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N K : USize)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (X : Float64Array) (offX incX : USize) : Float64Array

/-- Triangular packed matrix-vector multiply: X := op(A)X where A is triangular in packed format.
    Packed format stores only the triangular part in a 1D array. -/
@[extern "leanblas_cblas_dtpmv"]
opaque dtpmv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float64Array) (offA : USize)
    (X : Float64Array) (offX incX : USize) : Float64Array

/-- Triangular solve: solve op(A)X = B for X, result stored in X.
    A is triangular, X contains B on entry and solution on exit. -/
@[extern "leanblas_cblas_dtrsv"]
opaque dtrsv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (X : Float64Array) (offX incX : USize) : Float64Array

/-- Triangular banded solve: solve op(A)X = B for X where A is triangular banded. -/
@[extern "leanblas_cblas_dtbsv"]
opaque dtbsv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N K : USize)
    (A : @& Float64Array) (offA : USize) (lda : USize)
    (X : Float64Array) (offX incX : USize) : Float64Array

/-- Triangular packed solve: solve op(A)X = B for X where A is triangular in packed format. -/
@[extern "leanblas_cblas_dtpsv"]
opaque dtpsv (order : Order) (uplo : UpLo)
    (transA : Transpose) (diag : Diag) (N : USize)
    (A : @& Float64Array) (offA : USize)
    (X : Float64Array) (offX incX : USize) : Float64Array

/-- General rank-1 update: A := αXYᵀ + A -/
@[extern "leanblas_cblas_dger"]
opaque dger (order : Order) (M : USize) (N : USize) (alpha : Float)
    (X : @& Float64Array) (offX incX : USize)
    (Y : @& Float64Array) (offY incY : USize)
    (A : Float64Array) (offA : USize) (lda : USize) : Float64Array

/-- Symmetric rank-1 update: A := αXXᵀ + A -/
@[extern "leanblas_cblas_dsyr"]
opaque dsyr (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
    (X : @& Float64Array) (offX incX : USize)
    (A : Float64Array) (offA : USize) (lda : USize) : Float64Array

/-- Symmetric rank-2 update: A := αXYᵀ + αYXᵀ + A -/
@[extern "leanblas_cblas_dsyr2"]
opaque dsyr2 (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
    (X : @& Float64Array) (offX incX : USize)
    (Y : @& Float64Array) (offY incY : USize)
    (A : Float64Array) (offA : USize) (lda : USize) : Float64Array

/-! ## Extended Operations -/

/-- Convert packed to dense matrix -/
@[extern "leanblas_cblas_dpacked_to_dense"]
opaque dpackedToDense (N : USize) (uplo : UpLo)
  (orderAp : Order) (Ap : @& Float64Array)
  (orderA : Order) (A : Float64Array) (offA : USize) (lds : USize) : Float64Array

/-- Convert dense to packed matrix -/
@[extern "leanblas_cblas_ddense_to_packed"]
opaque ddenseToPacked (N : USize) (uplo : UpLo)
  (orderA : Order) (A : @& Float64Array) (offA : USize) (lda : USize)
  (orderAp : Order) (Ap : Float64Array) : Float64Array

/-- General rank-1 update on packed triangular: Ap := αXYᵀ + Ap -/
@[extern "leanblas_cblas_dgpr"]
opaque dgpr (order : Order) (uplo : UpLo) (N : USize) (alpha : Float)
    (X : @& Float64Array) (offX incX : USize)
    (Y : @& Float64Array) (offY incY : USize)
    (Ap : Float64Array) (offA : USize) : Float64Array
