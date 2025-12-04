import LeanBLAS.FFI.CBLASLevelThreeFloat32
import LeanBLAS.Spec.LevelThree

/-!
# CBLAS Level 3 Implementation for Float32

This module provides the CBLAS (C interface to BLAS) implementation of Level 3 BLAS operations
for Float32Array types. Level 3 operations are matrix-matrix operations with O(n³) complexity.

## Overview

Float32 (single precision) Level 3 operations are essential for GPU-efficient computation since:
- Most GPUs have 2x higher throughput for Float32 vs Float64
- Neural network training typically uses single precision
- Memory bandwidth is halved compared to Float64

## Performance Characteristics

Level 3 operations achieve the highest arithmetic intensity (operations per memory access)
making them ideal for:
- Cache optimization
- Parallel execution
- Vectorization
- GPU acceleration

The single-precision variants are particularly efficient for:
- Deep learning workloads
- Graphics applications
- Real-time simulations
-/

namespace BLAS.CBLAS

/-- CBLAS implementation of Level 3 BLAS operations for Float32Array.

This instance provides high-performance matrix-matrix operations through FFI
bindings to optimized BLAS libraries using single precision. -/
instance : LevelThreeData Float32Array Float Float where
  gemm order transA transB M N K_dim alpha A offA lda B offB ldb beta C offC ldc :=
    sgemm order transA transB M.toUSize N.toUSize K_dim.toUSize alpha A offA.toUSize lda.toUSize B offB.toUSize ldb.toUSize beta C offC.toUSize ldc.toUSize

  symm order side uplo M N alpha A offA lda B offB ldb beta C offC ldc :=
    ssymm order side uplo M.toUSize N.toUSize alpha A offA.toUSize lda.toUSize B offB.toUSize ldb.toUSize beta C offC.toUSize ldc.toUSize

  syrk order uplo transA N K_dim alpha A offA lda beta C offC ldc :=
    ssyrk order uplo transA N.toUSize K_dim.toUSize alpha A offA.toUSize lda.toUSize beta C offC.toUSize ldc.toUSize

  syr2k order uplo transA N K_dim alpha A offA lda B offB ldb beta C offC ldc :=
    ssyr2k order uplo transA N.toUSize K_dim.toUSize alpha A offA.toUSize lda.toUSize B offB.toUSize ldb.toUSize beta C offC.toUSize ldc.toUSize

  trmm order side uplo transA diag M N alpha A offA lda B offB ldb :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    strmm order side uplo transA diag' M.toUSize N.toUSize alpha A offA.toUSize lda.toUSize B offB.toUSize ldb.toUSize

  trsm order side uplo transA diag M N alpha A offA lda B offB ldb :=
    let diag' := if diag then Diag.Unit else Diag.NonUnit
    strsm order side uplo transA diag' M.toUSize N.toUSize alpha A offA.toUSize lda.toUSize B offB.toUSize ldb.toUSize

end BLAS.CBLAS
