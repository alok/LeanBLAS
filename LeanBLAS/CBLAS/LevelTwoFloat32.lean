import LeanBLAS.FFI.CBLASLevelTwoFloat32
import LeanBLAS.Spec.LevelTwo

/-!
# CBLAS Level 2 Implementation for Float32

This module provides the CBLAS (C interface to BLAS) implementation of Level 2 BLAS operations
for Float32Array types. Level 2 operations are matrix-vector operations with O(n²) complexity.

## Overview

Float32 (single precision) operations are essential for GPU-efficient computation since:
- Most GPUs have 2x higher throughput for Float32 vs Float64
- Neural network training typically uses single precision
- Memory bandwidth is halved compared to Float64

## Implementation Details

All operations use FFI bindings to optimized BLAS libraries (s* prefix functions).
Lean's Float type is 64-bit, so we convert to 32-bit at the FFI boundary.
-/

namespace BLAS.CBLAS

/-- CBLAS implementation of Level 2 BLAS operations for Float32Array.

This instance provides efficient matrix-vector operations through FFI calls
to optimized BLAS libraries using single precision. -/
instance : LevelTwoData Float32Array Float Float where

  gemv order trans M N a A offA ldaA X offX incX b Y offY incY :=
    sgemv order trans M.toUSize N.toUSize a
      A offA.toUSize ldaA.toUSize X offX.toUSize incX.toUSize b Y offY.toUSize incY.toUSize

  bmv order trans M N KL KU a A offA ldaA X offX incX b Y offY incY :=
    sbmv order trans M.toUSize N.toUSize KL.toUSize KU.toUSize a
      A offA.toUSize ldaA.toUSize X offX.toUSize incX.toUSize b Y offY.toUSize incY.toUSize

  trmv order uplo trans diag N A offA lda X offX incX :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    strmv order uplo trans diag' N.toUSize A offA.toUSize lda.toUSize X offX.toUSize incX.toUSize

  tbmv order uplo trans diag N K A offA lda X offX incX :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    stbmv order uplo trans diag' N.toUSize K.toUSize A offA.toUSize lda.toUSize X offX.toUSize incX.toUSize

  tpmv order uplo trans diag N A offA X offX incX :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    stpmv order uplo trans diag' N.toUSize A offA.toUSize X offX.toUSize incX.toUSize

  trsv order uplo trans diag N A offA lda X offX incX :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    strsv order uplo trans diag' N.toUSize A offA.toUSize lda.toUSize X offX.toUSize incX.toUSize

  tbsv order uplo trans diag N K A offA lda X offX incX :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    stbsv order uplo trans diag' N.toUSize K.toUSize A offA.toUSize lda.toUSize X offX.toUSize incX.toUSize

  tpsv order uplo trans diag N A offA X offX incX :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    stpsv order uplo trans diag' N.toUSize A offA.toUSize X offX.toUSize incX.toUSize

  ger order M N a X offX incX Y offY incY A offA lda :=
    sger order M.toUSize N.toUSize a X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize A offA.toUSize lda.toUSize

  her order uplo N alpha X offX incX A offA lda :=
    ssyr order uplo N.toUSize alpha X offX.toUSize incX.toUSize A offA.toUSize lda.toUSize

  her2 order uplo N alpha X offX incX Y offY incY A offA lda :=
    ssyr2 order uplo N.toUSize alpha X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize A offA.toUSize lda.toUSize

end BLAS.CBLAS
