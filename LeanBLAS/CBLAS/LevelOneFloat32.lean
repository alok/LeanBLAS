import LeanBLAS.FFI.CBLASLevelOneFloat32
import LeanBLAS.Spec.LevelOne

/-!
# CBLAS Level 1 Implementation for Float32

This module provides the CBLAS implementation of Level 1 BLAS operations
for Float32Array types. Level 1 operations are vector-vector operations with O(n) complexity.

## Overview

The implementation uses FFI bindings to call optimized BLAS libraries for actual computation.
Float32 (single precision) is essential for GPU-efficient computation since:
- Most GPUs have 2x higher throughput for Float32 vs Float64
- Neural network training typically uses single precision
- Memory bandwidth is halved compared to Float64

## Implementation Details

Lean's `Float` type is 64-bit, so we convert to/from 32-bit at the FFI boundary.
This maintains Lean's numeric consistency while enabling efficient GPU computation.
-/

namespace BLAS.CBLAS

open Sorry

/-- CBLAS implementation of Level 1 BLAS operations for Float32Array.

This instance provides efficient implementations of vector operations by calling
optimized BLAS libraries through FFI. All index parameters (offsets and strides)
are converted from Nat to USize for C compatibility. -/
instance : LevelOneData Float32Array Float Float where
  size x := x.size
  get x i := Float32Array.get x i
  dot N X offX incX Y offY incY := sdot N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  nrm2 N X offX incX := snrm2 N.toUSize X offX.toUSize incX.toUSize
  asum N X offX incX := sasum N.toUSize X offX.toUSize incX.toUSize
  iamax N X offX incX := isamax N.toUSize X offX.toUSize incX.toUSize |>.toNat
  swap N X offX incX Y offY incY := sswap N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  copy N X offX incX Y offY incY := scopy N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  axpy N a X offX incX Y offY incY := saxpy N.toUSize a X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  rotg a b := srotg a b
  rotmg d1 d2 b1 b2 := srotmg d1 d2 b1 b2
  rot N X offX incX Y offY incY c s := srot N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize c s
  scal N a X offX incX := sscal N.toUSize a X offX.toUSize incX.toUSize

set_option linter.unusedVariables false in
instance : LevelOneDataExt Float32Array Float Float where
  const N a := sconst N.toUSize a
  sum N X offX incX := ssum N.toUSize X offX.toUSize incX.toUSize
  axpby N a X offX incX b Y offY incY := saxpby N.toUSize a X offX.toUSize incX.toUSize b Y offY.toUSize incY.toUSize
  scaladd N a X offX incX b := sscaladd N.toUSize a X offX.toUSize incX.toUSize b

  imaxRe N X offX incX h := (simaxRe N.toUSize X offX.toUSize incX.toUSize).toNat
  imaxIm N X offX incX h := offX
  iminRe N X offX incX h := (siminRe N.toUSize X offX.toUSize incX.toUSize).toNat
  iminIm N X offX incX h := offX

  mul N X offX incX Y offY incY := smul N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  div N X offX incX Y offY incY := sdiv N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  inv N X offX incX := sinv N.toUSize X offX.toUSize incX.toUSize
  abs N X offX incX := sabs N.toUSize X offX.toUSize incX.toUSize
  sqrt N X offX incX := ssqrt N.toUSize X offX.toUSize incX.toUSize
  exp N X offX incX := sexp N.toUSize X offX.toUSize incX.toUSize
  log N X offX incX := slog N.toUSize X offX.toUSize incX.toUSize
  sin N X offX incX := ssin N.toUSize X offX.toUSize incX.toUSize
  cos N X offX incX := scos N.toUSize X offX.toUSize incX.toUSize

end BLAS.CBLAS
