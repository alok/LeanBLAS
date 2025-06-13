import LeanBLAS
import LeanBLAS.FFI.FloatArray
import LeanBLAS.FFI.CBLASLevelOneComplexFloat64

/-!
# Simple Numerical Validation Tests for Complex Level 1 BLAS Operations

This module provides simple numerical validation tests for complex Level 1 BLAS
operations, comparing results against expected values with appropriate tolerances.
-/

open BLAS CBLAS

/-- Helper for complex number approximate equality -/
def complexApproxEq (x y : ComplexFloat) (ε : Float := 1e-10) : Bool :=
  Float.abs (x.x - y.x) < ε && Float.abs (x.y - y.y) < ε

/-- Test zdotu (unconjugated dot product) -/
def test_zdotu : IO Unit := do
  IO.println "\n=== Testing zdotu (unconjugated dot product) ==="
  
  -- Test 1: Basic dot product
  IO.println "Creating arrays..."
  let x1_arr := ComplexFloatArray.ofArray #[⟨1.0, 0.0⟩, ⟨2.0, 1.0⟩]
  IO.println "Converting x1..."
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  let y1_arr := ComplexFloatArray.ofArray #[⟨3.0, 4.0⟩, ⟨1.0, -2.0⟩]
  IO.println "Converting y1..."
  let y1 := _root_.ComplexFloatArray.toComplexFloat64Array y1_arr
  -- Expected: (1+0i)*(3+4i) + (2+1i)*(1-2i) = 3+4i + (2-4i+1i-2i²) = 3+4i + 4-3i = 7+1i
  IO.println "Calling zdotu..."
  let result1 := zdotu 2 x1 0 1 y1 0 1
  let expected1 : ComplexFloat := { x := 7.0, y := 1.0 }
  let test1_ok := complexApproxEq result1 expected1
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"

/-- Test zdotc (conjugated dot product) -/
def test_zdotc : IO Unit := do
  IO.println "\n=== Testing zdotc (conjugated dot product) ==="
  
  -- Test 1: Basic conjugate dot product
  let x1_arr := ComplexFloatArray.ofArray #[⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  let y1_arr := ComplexFloatArray.ofArray #[⟨1.0, 0.0⟩, ⟨0.0, -1.0⟩]
  let y1 := _root_.ComplexFloatArray.toComplexFloat64Array y1_arr
  -- Expected: conj(1+0i)*(1+0i) + conj(0+1i)*(0-1i) = 1*1 + (0-1i)*(0-1i) = 1 + 1 = 2
  let result1 := zdotc 2 x1 0 1 y1 0 1
  let expected1 : ComplexFloat := { x := 2.0, y := 0.0 }
  let test1_ok := complexApproxEq result1 expected1
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"

/-- Test dznrm2 (2-norm of complex vector) -/
def test_dznrm2 : IO Unit := do
  IO.println "\n=== Testing dznrm2 (2-norm) ==="
  
  -- Test 1: Basic norm
  let x1_arr := ComplexFloatArray.ofArray #[⟨3.0, 4.0⟩, ⟨0.0, 0.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  -- Expected: sqrt(|3+4i|² + |0|²) = sqrt(25 + 0) = 5
  let result1 := dznrm2 2 x1 0 1
  let expected1 := 5.0
  let test1_ok := Float.abs (result1 - expected1) < 1e-10
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"

/-- Test dzasum (sum of absolute values) -/
def test_dzasum : IO Unit := do
  IO.println "\n=== Testing dzasum (sum of absolute values) ==="
  
  -- Test 1: Basic sum
  let x1_arr := ComplexFloatArray.ofArray #[⟨3.0, 4.0⟩, ⟨-1.0, 0.0⟩]
  let x1 := _root_.ComplexFloatArray.toComplexFloat64Array x1_arr
  -- Expected: |3| + |4| + |-1| + |0| = 3 + 4 + 1 + 0 = 8
  let result1 := dzasum 2 x1 0 1
  let expected1 := 8.0
  let test1_ok := Float.abs (result1 - expected1) < 1e-10
  IO.println s!"  Test 1: {result1} vs expected {expected1} - {if test1_ok then "✓" else "✗"}"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Simple Complex Level 1 BLAS Numerical Validation Tests ==="
  IO.println "Testing complex BLAS operations against expected numerical results"
  
  -- Run tests directly using FFI functions
  test_zdotu
  test_zdotc
  test_dznrm2
  test_dzasum
  
  IO.println "\n✅ All tests completed!"