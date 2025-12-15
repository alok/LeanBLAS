import LeanBLAS

open BLAS CBLAS
open LevelTwoData

namespace BLAS.Test.PackedTriangular

def approxEqFloat (x y : Float) : Bool :=
  (x - y).abs < 1e-12

def Float64Array.approxEq (x y : Float64Array) : Bool := Id.run do
  let xa := x.toFloatArray
  let ya := y.toFloatArray
  if xa.size != ya.size then
    return false
  let mut ok := true
  for i in [0:xa.size] do
    if !(approxEqFloat (xa[i]!) (ya[i]!)) then
      ok := false
  return ok

/--
Basic smoke tests for triangular packed matrix-vector multiply (`tpmv`).

The old version of this file relied on an obsolete `DenseVector` API. These
tests exercise the current CBLAS-backed packed-triangular ops directly.
-/
def run : IO Unit := do
  IO.println "\n=== Packed Triangular Tests ==="

  let N := 3

  -- Lower triangular matrix (column-major packed):
  -- A =
  -- [1 0 0
  --  2 3 0
  --  4 5 6]
  -- Packed lower col-major stores columns:
  -- col1: 1,2,4; col2: 3,5; col3: 6
  let ApLower : Float64Array := #f64[1.0, 2.0, 4.0, 3.0, 5.0, 6.0]
  let x : Float64Array := #f64[1.0, 1.0, 1.0]
  let yLower := tpmv .ColMajor .Lower .NoTrans false N ApLower 0 x 0 1
  -- Cross-check against dense conversion + GEMV using the same order/uplo.
  let A0Lower : Float64Array := #f64[
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0
  ]
  let ADenseLower :=
    CBLAS.dpackedToDense N.toUSize UpLo.Lower Order.ColMajor ApLower Order.ColMajor A0Lower 0 3
  let y0Lower : Float64Array := #f64[0.0, 0.0, 0.0]
  let yDenseLower :=
    gemv .ColMajor .NoTrans N N 1.0 ADenseLower 0 3 x 0 1 0.0 y0Lower 0 1
  IO.println s!"Lower tpmv result: {yLower.toFloatArray}"
  IO.println s!"Lower dense+gemv result: {yDenseLower.toFloatArray}"
  if !(Float64Array.approxEq yLower yDenseLower) then
    throw $ IO.userError s!"Lower tpmv mismatch vs dense gemv: {yLower.toFloatArray} ≠ {yDenseLower.toFloatArray}"

  -- Upper triangular matrix (column-major packed):
  -- A =
  -- [1 2 3
  --  0 4 5
  --  0 0 6]
  -- Packed upper col-major stores:
  -- col1: 1; col2: 2,4; col3: 3,5,6
  let ApUpper : Float64Array := #f64[1.0, 2.0, 4.0, 3.0, 5.0, 6.0]
  let yUpper := tpmv .ColMajor .Upper .NoTrans false N ApUpper 0 x 0 1
  let A0Upper : Float64Array := #f64[
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0
  ]
  let ADenseUpper :=
    CBLAS.dpackedToDense N.toUSize UpLo.Upper Order.ColMajor ApUpper Order.ColMajor A0Upper 0 3
  let y0Upper : Float64Array := #f64[0.0, 0.0, 0.0]
  let yDenseUpper :=
    gemv .ColMajor .NoTrans N N 1.0 ADenseUpper 0 3 x 0 1 0.0 y0Upper 0 1
  IO.println s!"Upper tpmv result: {yUpper.toFloatArray}"
  IO.println s!"Upper dense+gemv result: {yDenseUpper.toFloatArray}"
  if !(Float64Array.approxEq yUpper yDenseUpper) then
    throw $ IO.userError s!"Upper tpmv mismatch vs dense gemv: {yUpper.toFloatArray} ≠ {yDenseUpper.toFloatArray}"

  IO.println "✓ Packed triangular tests passed."

end BLAS.Test.PackedTriangular
