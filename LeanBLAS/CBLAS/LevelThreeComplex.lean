import LeanBLAS.FFI.CBLASLevelThreeComplexFloat64
import LeanBLAS.Spec.LevelThree

namespace BLAS.CBLAS

open Sorry

set_option linter.unusedVariables false

/-! # CBLAS Level 3 Complex Implementation

This module provides the CBLAS implementation of Level 3 BLAS operations
for ComplexFloat64Array types. These are matrix-matrix operations on complex numbers.

## Complex Matrix Operations

Complex Level 3 operations have specific behaviors:
- General operations work with full complex arithmetic
- Hermitian operations (e.g., `zhemm`, `zherk`) use conjugate transpose symmetry
- Some operations like `zherk` and `zher2k` require real scalars for beta

## Implementation Notes

The FFI bindings handle complex matrix storage:
- Elements stored as interleaved [real, imaginary] pairs
- Leading dimension calculations account for complex element size
- Hermitian matrices store only upper or lower triangle
-/

-- Level 3 Complex BLAS wrappers that provide the Nat-based interface
private def zgemm' (order : Order) (transA transB : Transpose) (M N K : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
          (beta : ComplexFloat)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if order = Order.RowMajor then
                (if transA = Transpose.NoTrans then K else M)
              else (if transA = Transpose.NoTrans then M else K)) else lda
  let ldb' := if ldb = 0 then (if order = Order.RowMajor then
                (if transB = Transpose.NoTrans then N else K)
              else (if transB = Transpose.NoTrans then K else N)) else ldb
  let ldc' := if ldc = 0 then (if order = Order.RowMajor then N else M) else ldc
  CBLAS.zgemm order transA transB M.toUSize N.toUSize K.toUSize alpha
              A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize
              beta C offC.toUSize ldc'.toUSize

private def zsymm' (order : Order) (side : Side) (uplo : UpLo) (M N : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
          (beta : ComplexFloat)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if side = Side.Left then M else N) else lda
  let ldb' := if ldb = 0 then (if order = Order.RowMajor then N else M) else ldb
  let ldc' := if ldc = 0 then (if order = Order.RowMajor then N else M) else ldc
  CBLAS.zsymm order side uplo M.toUSize N.toUSize alpha
              A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize
              beta C offC.toUSize ldc'.toUSize

private def zhemm' (order : Order) (side : Side) (uplo : UpLo) (M N : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
          (beta : ComplexFloat)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if side = Side.Left then M else N) else lda
  let ldb' := if ldb = 0 then (if order = Order.RowMajor then N else M) else ldb
  let ldc' := if ldc = 0 then (if order = Order.RowMajor then N else M) else ldc
  CBLAS.zhemm order side uplo M.toUSize N.toUSize alpha
              A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize
              beta C offC.toUSize ldc'.toUSize

private def zsyrk' (order : Order) (uplo : UpLo) (transA : Transpose) (N K : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (beta : ComplexFloat)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if order = Order.RowMajor then
                (if transA = Transpose.NoTrans then K else N)
              else (if transA = Transpose.NoTrans then N else K)) else lda
  let ldc' := if ldc = 0 then N else ldc
  CBLAS.zsyrk order uplo transA N.toUSize K.toUSize alpha
              A offA.toUSize lda'.toUSize beta C offC.toUSize ldc'.toUSize

private def zherk' (order : Order) (uplo : UpLo) (transA : Transpose) (N K : Nat)
          (alpha : Float)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (beta : Float)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if order = Order.RowMajor then
                (if transA = Transpose.NoTrans then K else N)
              else (if transA = Transpose.NoTrans then N else K)) else lda
  let ldc' := if ldc = 0 then N else ldc
  CBLAS.zherk order uplo transA N.toUSize K.toUSize alpha
              A offA.toUSize lda'.toUSize beta C offC.toUSize ldc'.toUSize

private def zsyr2k' (order : Order) (uplo : UpLo) (transA : Transpose) (N K : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
          (beta : ComplexFloat)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if order = Order.RowMajor then
                (if transA = Transpose.NoTrans then K else N)
              else (if transA = Transpose.NoTrans then N else K)) else lda
  let ldb' := lda'  -- Usually same as A
  let ldc' := if ldc = 0 then N else ldc
  CBLAS.zsyr2k order uplo transA N.toUSize K.toUSize alpha
               A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize
               beta C offC.toUSize ldc'.toUSize

private def zher2k' (order : Order) (uplo : UpLo) (transA : Transpose) (N K : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
          (beta : Float)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if order = Order.RowMajor then
                (if transA = Transpose.NoTrans then K else N)
              else (if transA = Transpose.NoTrans then N else K)) else lda
  let ldb' := lda'  -- Usually same as A
  let ldc' := if ldc = 0 then N else ldc
  CBLAS.zher2k order uplo transA N.toUSize K.toUSize alpha
               A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize
               beta C offC.toUSize ldc'.toUSize

private def ztrmm' (order : Order) (side : Side) (uplo : UpLo) (transA : Transpose) (diag : Diag)
          (M N : Nat) (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if side = Side.Left then M else N) else lda
  let ldb' := if ldb = 0 then (if order = Order.RowMajor then N else M) else ldb
  CBLAS.ztrmm order side uplo transA diag M.toUSize N.toUSize alpha
              A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize

private def ztrsm' (order : Order) (side : Side) (uplo : UpLo) (transA : Transpose) (diag : Diag)
          (M N : Nat) (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0) : ComplexFloat64Array :=
  let lda' := if lda = 0 then (if side = Side.Left then M else N) else lda
  let ldb' := if ldb = 0 then (if order = Order.RowMajor then N else M) else ldb
  CBLAS.ztrsm order side uplo transA diag M.toUSize N.toUSize alpha
              A offA.toUSize lda'.toUSize B offB.toUSize ldb'.toUSize

/-- CBLAS implementation of Level 3 BLAS operations for ComplexFloat64Array.

This instance provides optimized complex matrix-matrix operations through FFI
bindings to CBLAS libraries. -/
instance : LevelThreeData ComplexFloat64Array Float ComplexFloat where
  gemm order transA transB M N K alpha A offA lda B offB ldb beta C offC ldc :=
    zgemm' order transA transB M N K alpha A offA lda B offB ldb beta C offC ldc
  symm order side uplo M N alpha A offA lda B offB ldb beta C offC ldc :=
    zsymm' order side uplo M N alpha A offA lda B offB ldb beta C offC ldc
  syrk order uplo transA N K alpha A offA lda beta C offC ldc :=
    zsyrk' order uplo transA N K alpha A offA lda beta C offC ldc
  syr2k order uplo transA N K alpha A offA lda B offB ldb beta C offC ldc :=
    zsyr2k' order uplo transA N K alpha A offA lda B offB ldb beta C offC ldc
  trmm order side uplo transA diag M N alpha A offA lda B offB ldb :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    ztrmm' order side uplo transA diag' M N alpha A offA lda B offB ldb
  trsm order side uplo transA diag M N alpha A offA lda B offB ldb :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    ztrsm' order side uplo transA diag' M N alpha A offA lda B offB ldb

-- Complex-specific Level 3 operations not in the standard interface
/-- Hermitian matrix-matrix multiplication (complex-specific).
    C := alpha * A * B + beta * C or C := alpha * B * A + beta * C
    where A is Hermitian -/
def hemm (order : Order) (side : Side) (uplo : UpLo) (M N : Nat)
         (alpha : ComplexFloat)
         (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
         (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
         (beta : ComplexFloat)
         (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  zhemm' order side uplo M N alpha A offA lda B offB ldb beta C offC ldc

/-- Hermitian rank-k update (complex-specific).
    C := alpha * A * A^H + beta * C
    where alpha and beta must be real -/
def herk (order : Order) (uplo : UpLo) (transA : Transpose) (N K : Nat)
         (alpha : Float)
         (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
         (beta : Float)
         (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  zherk' order uplo transA N K alpha A offA lda beta C offC ldc

/-- Hermitian rank-2k update (complex-specific).
    C := alpha * A * B^H + conj(alpha) * B * A^H + beta * C
    where beta must be real -/
def her2k (order : Order) (uplo : UpLo) (transA : Transpose) (N K : Nat)
          (alpha : ComplexFloat)
          (A : ComplexFloat64Array) (offA : Nat := 0) (lda : Nat := 0)
          (B : ComplexFloat64Array) (offB : Nat := 0) (ldb : Nat := 0)
          (beta : Float)
          (C : ComplexFloat64Array) (offC : Nat := 0) (ldc : Nat := 0) : ComplexFloat64Array :=
  zher2k' order uplo transA N K alpha A offA lda B offB ldb beta C offC ldc

end BLAS.CBLAS