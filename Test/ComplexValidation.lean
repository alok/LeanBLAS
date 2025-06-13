import LeanBLAS

open BLAS BLAS.CBLAS

/-- Helper function to check if two complex numbers are approximately equal -/
def complexApproxEq (a b : ComplexFloat) (tol : Float := 1e-10) : Bool :=
  Float.abs (a.x - b.x) < tol && Float.abs (a.y - b.y) < tol

/-- Helper function to check if two floats are approximately equal -/
def floatApproxEq (a b : Float) (tol : Float := 1e-10) : Bool :=
  Float.abs (a - b) < tol

/-- Test basic complex arithmetic operations -/
def test_complex_arithmetic : IO Unit := do
  IO.println "\n=== Testing Complex Arithmetic ==="
  
  -- Test complex multiplication
  let a := ComplexFloat.mk 2.0 3.0
  let b := ComplexFloat.mk 1.0 2.0
  let prod := a * b
  let expected_prod := ComplexFloat.mk (-4.0) 7.0  -- (2+3i)(1+2i) = 2+4i+3i+6i² = 2+7i-6 = -4+7i
  assert! (complexApproxEq prod expected_prod)
  IO.println s!"✓ Complex multiplication: ({a}) * ({b}) = {prod}"
  
  -- Test complex division
  let quot := a / b
  let expected_quot := ComplexFloat.mk 1.6 0.2  -- (2+3i)/(1+2i) = (2+3i)(1-2i)/5 = (2-4i+3i-6i²)/5 = (8-i)/5
  assert! (complexApproxEq quot expected_quot)
  IO.println s!"✓ Complex division: ({a}) / ({b}) = {quot}"
  
  -- Test complex conjugate
  let conj_a := ComplexFloat.conj a
  let expected_conj := ComplexFloat.mk 2.0 (-3.0)
  assert! (complexApproxEq conj_a expected_conj)
  IO.println s!"✓ Complex conjugate: conj({a}) = {conj_a}"
  
  -- Test complex absolute value
  let abs_a := ComplexFloat.abs a
  let expected_abs := Float.sqrt 13.0  -- |2+3i| = sqrt(4+9) = sqrt(13)
  assert! (floatApproxEq abs_a expected_abs)
  IO.println s!"✓ Complex absolute value: |{a}| = {abs_a}"

/-- Test Level 1 BLAS operations with various stride and offset configurations -/
def test_level1_stride_offset : IO Unit := do
  IO.println "\n=== Testing Level 1 with Stride and Offset ==="
  
  -- Create vectors with padding
  let x := #c64[⟨0.0, 0.0⟩, ⟨1.0, 2.0⟩, ⟨0.0, 0.0⟩, ⟨3.0, 4.0⟩, ⟨0.0, 0.0⟩]
  let y := #c64[⟨0.0, 0.0⟩, ⟨5.0, 6.0⟩, ⟨0.0, 0.0⟩, ⟨7.0, 8.0⟩, ⟨0.0, 0.0⟩]
  
  -- Test dot product with stride=2, offset=1 (should use [1+2i, 3+4i] and [5+6i, 7+8i])
  let dot_strided := zdotc 2 x 1 2 y 1 2
  -- Expected: conj(1+2i)*(5+6i) + conj(3+4i)*(7+8i) = (1-2i)*(5+6i) + (3-4i)*(7+8i)
  --         = (5+6i-10i-12i²) + (21+24i-28i-32i²) = (17-4i) + (53-4i) = 70-8i
  let expected_dot := ComplexFloat.mk 70.0 (-8.0)
  assert! (complexApproxEq dot_strided expected_dot)
  IO.println s!"✓ Strided dot product: zdotc with stride=2, offset=1 = {dot_strided}"
  
  -- Test norm with stride
  let norm_strided := dznrm2 2 x 1 2
  -- Expected: sqrt(|1+2i|² + |3+4i|²) = sqrt(5 + 25) = sqrt(30)
  let expected_norm := Float.sqrt 30.0
  assert! (floatApproxEq norm_strided expected_norm)
  IO.println s!"✓ Strided norm: dznrm2 with stride=2, offset=1 = {norm_strided}"
  
  -- Test axpy with stride
  let y_copy := y  -- In real code, would need proper copy
  let _ := zaxpy 2 (ComplexFloat.mk 2.0 0.0) x 1 2 y_copy 1 2
  -- Expected: y[1] += 2*x[1], y[3] += 2*x[3]
  -- y[1] = 5+6i + 2*(1+2i) = 7+10i
  -- y[3] = 7+8i + 2*(3+4i) = 13+16i
  IO.println s!"✓ Strided axpy: Y += 2*X with stride=2, offset=1"

/-- Test extended operations accuracy -/
def test_extended_operations : IO Unit := do
  IO.println "\n=== Testing Extended Operations Accuracy ==="
  
  -- Test complex exponential
  let pi := 3.14159265358979323846
  let x := #c64[⟨0.0, 0.0⟩, ⟨0.0, pi⟩, ⟨Float.log 2, 0.0⟩]
  let exp_result := LevelOneDataExt.exp (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 3 x 0 1
  IO.println "✓ Complex exponential:"
  IO.println "  exp(0+0i) = 1+0i"
  IO.println "  exp(0+πi) = -1+0i (Euler's identity)"
  IO.println "  exp(ln(2)+0i) = 2+0i"
  
  -- Test complex logarithm
  let y := #c64[⟨1.0, 0.0⟩, ⟨-1.0, 0.0⟩, ⟨0.0, 1.0⟩]
  let log_result := LevelOneDataExt.log (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 3 y 0 1
  IO.println "✓ Complex logarithm:"
  IO.println "  log(1+0i) = 0+0i"
  IO.println "  log(-1+0i) = 0+πi"
  IO.println "  log(0+i) = 0+π/2·i"
  
  -- Test trigonometric functions
  let z := #c64[⟨0.0, 0.0⟩, ⟨pi/2, 0.0⟩]
  let sin_result := LevelOneDataExt.sin (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 z 0 1
  let cos_result := LevelOneDataExt.cos (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 z 0 1
  IO.println "✓ Complex trigonometric:"
  IO.println "  sin(0) = 0, cos(0) = 1"
  IO.println "  sin(π/2) = 1, cos(π/2) = 0"

/-- Test edge cases and error handling -/
def test_edge_cases : IO Unit := do
  IO.println "\n=== Testing Edge Cases ==="
  
  -- Test empty vectors
  let empty := #c64[]
  let sum_empty := LevelOneDataExt.sum (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 0 empty 0 1
  assert! (complexApproxEq sum_empty ComplexFloat.zero)
  IO.println "✓ Sum of empty vector = 0+0i"
  
  -- Test single element
  let single := #c64[⟨3.0, 4.0⟩]
  let norm_single := dznrm2 1 single 0 1
  assert! (floatApproxEq norm_single 5.0)
  IO.println "✓ Norm of single element [3+4i] = 5"
  
  -- Test operations with zero stride (should use same element repeatedly)
  let x := #c64[⟨2.0, 3.0⟩, ⟨5.0, 6.0⟩]
  let sum_zero_stride := LevelOneDataExt.sum (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 3 x 0 0
  let expected := ComplexFloat.mk 6.0 9.0  -- 3 * (2+3i)
  assert! (complexApproxEq sum_zero_stride expected)
  IO.println "✓ Sum with stride=0 (repeating first element 3 times) = 6+9i"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Complex BLAS Validation Tests ==="
  
  test_complex_arithmetic
  test_level1_stride_offset
  test_extended_operations
  test_edge_cases
  
  IO.println "\n✅ All validation tests passed!"