import LeanBLAS
import LeanBLAS.CBLAS.LevelOneComplex
import LeanBLAS.FFI.FloatArray

/-!
# Numerical Validation Tests for Complex Level 1 BLAS Operations

This module provides comprehensive numerical validation tests for complex Level 1 BLAS
operations, comparing results against expected values with appropriate tolerances.
-/

open BLAS CBLAS

namespace BLAS.Test.ComplexCorrectness.Level1

/-- Helper for complex number approximate equality -/
def complexApproxEq (x y : ComplexFloat) (ε : Float := 1e-10) : Bool :=
  Float.abs (x.x - y.x) < ε && Float.abs (x.y - y.y) < ε

/-- Test zdotu (unconjugated dot product) -/
def test_zdotu : IO Bool := do
  IO.println "\n=== Testing zdotu (unconjugated dot product) ==="
  
  -- Test 1: Basic dot product
  let x1_arr := ComplexFloatArray.ofArray #[⟨1.0, 0.0⟩, ⟨2.0, 1.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  let y1_arr := ComplexFloatArray.ofArray #[⟨3.0, 4.0⟩, ⟨1.0, -2.0⟩]
  let y1 := _root_.ComplexFloatArray.toComplexFloat64Array y1_arr
  -- Expected: (1+0i)*(3+4i) + (2+1i)*(1-2i) = 3+4i + (2-4i+1i-2i²) = 3+4i + 4-3i = 7+1i
  let result1 := unconjugated_dot 2 x1 0 1 y1 0 1
  let expected1 : ComplexFloat := { x := 7.0, y := 1.0 }
  let test1_ok := complexApproxEq result1 expected1
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"
  
  return test1_ok

/-- Test zdotc (conjugated dot product) -/
def test_zdotc : IO Bool := do
  IO.println "\n=== Testing zdotc (conjugated dot product) ==="
  
  -- Test 1: Basic conjugate dot product
  let x1_arr := ComplexFloatArray.ofArray #[⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  let y1_arr := ComplexFloatArray.ofArray #[⟨1.0, 0.0⟩, ⟨0.0, -1.0⟩]
  let y1 := _root_.ComplexFloatArray.toComplexFloat64Array y1_arr
  -- Expected: conj(1+0i)*(1+0i) + conj(0+1i)*(0-1i) = 1*1 + (0-1i)*(0-1i) = 1 + 1 = 2
  let result1 := dot 2 x1 0 1 y1 0 1
  let expected1 : ComplexFloat := { x := 2.0, y := 0.0 }
  let test1_ok := complexApproxEq result1 expected1
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"
  
  return test1_ok

/-- Test dznrm2 (2-norm of complex vector) -/
def test_dznrm2 : IO Bool := do
  IO.println "\n=== Testing dznrm2 (2-norm) ==="
  
  -- Test 1: Basic norm
  let x1_arr := ComplexFloatArray.ofArray #[⟨3.0, 4.0⟩, ⟨0.0, 0.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  -- Expected: sqrt(|3+4i|² + |0|²) = sqrt(25 + 0) = 5
  let result1 := nrm2 2 x1 0 1
  let expected1 := 5.0
  let test1_ok := Float.abs (result1 - expected1) < 1e-10
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"
  
  return test1_ok

/-- Test dzasum (sum of absolute values) -/
def test_dzasum : IO Bool := do
  IO.println "\n=== Testing dzasum (sum of absolute values) ==="
  
  -- Test 1: Basic sum
  let x1_arr := ComplexFloatArray.ofArray #[⟨3.0, 4.0⟩, ⟨-1.0, 0.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  -- Expected: |3| + |4| + |-1| + |0| = 3 + 4 + 1 + 0 = 8
  let result1 := asum 2 x1 0 1
  let expected1 := 8.0
  let test1_ok := Float.abs (result1 - expected1) < 1e-10
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"
  
  return test1_ok

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Complex Level 1 BLAS Numerical Validation Tests ==="
  IO.println "Testing complex BLAS operations against expected numerical results"
  
  let mut all_passed := true
  
  -- Run all tests
  let tests : List (String × IO Bool) := [
    ("zdotu", test_zdotu),
    ("zdotc", test_zdotc),
    ("dznrm2", test_dznrm2),
    ("dzasum", test_dzasum)
  ]
  
  for (name, test) in tests do
    let passed ← test
    if !passed then
      all_passed := false
      IO.println s!"\n❌ {name} tests FAILED!"
    else
      IO.println s!"\n✅ {name} tests PASSED!"
  
  if all_passed then
    IO.println "\n🎉 All Complex Level 1 numerical validation tests PASSED!"
  else
    IO.println "\n❌ Some Complex Level 1 tests FAILED!"
    throw $ IO.userError "Complex Level 1 validation failed"

end BLAS.Test.ComplexCorrectness.Level1

def main : IO Unit := BLAS.Test.ComplexCorrectness.Level1.main