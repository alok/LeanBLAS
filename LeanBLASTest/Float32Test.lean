import LeanBLAS

open BLAS CBLAS

/-- Simple test for Float32 BLAS operations -/
def main : IO Unit := do
  IO.println "Testing Float32 BLAS operations..."

  -- Test sconst: create a constant vector
  let n : Nat := 5
  let arr := sconst n.toUSize 3.0
  IO.println s!"Created Float32Array with {arr.size} elements"

  -- Test sdot (need two arrays)
  let arr2 := sconst n.toUSize 2.0
  let dotResult := sdot n.toUSize arr 0 1 arr2 0 1
  IO.println s!"Dot product of [3,3,3,3,3] · [2,2,2,2,2] = {dotResult} (expected: 30.0)"

  -- Test snrm2
  let onesArr := sconst n.toUSize 1.0
  let norm := snrm2 n.toUSize onesArr 0 1
  IO.println s!"Euclidean norm of [1,1,1,1,1] = {norm} (expected: ~2.236)"

  -- Test sasum
  let sumAbs := sasum n.toUSize arr 0 1
  IO.println s!"Sum of absolute values of [3,3,3,3,3] = {sumAbs} (expected: 15.0)"

  -- Test ssum
  let sum := ssum n.toUSize arr 0 1
  IO.println s!"Sum of [3,3,3,3,3] = {sum} (expected: 15.0)"

  -- Test sscal
  let scaledArr := sscal n.toUSize 2.0 (sconst n.toUSize 5.0) 0 1
  let scaledSum := ssum n.toUSize scaledArr 0 1
  IO.println s!"Sum after scaling [5,5,5,5,5] by 2.0 = {scaledSum} (expected: 50.0)"

  -- Check all tests passed
  let dotOk := (dotResult - 30.0).abs < 0.01
  let normOk := (norm - 2.236).abs < 0.01
  let sumAbsOk := (sumAbs - 15.0).abs < 0.01
  let sumOk := (sum - 15.0).abs < 0.01
  let scaledOk := (scaledSum - 50.0).abs < 0.01

  if dotOk && normOk && sumAbsOk && sumOk && scaledOk then
    IO.println "\n✓ All Float32 tests passed!"
  else
    IO.println "\n✗ Some tests failed"
    if !dotOk then IO.println "  - Dot product failed"
    if !normOk then IO.println "  - Norm failed"
    if !sumAbsOk then IO.println "  - Asum failed"
    if !sumOk then IO.println "  - Sum failed"
    if !scaledOk then IO.println "  - Scal failed"
