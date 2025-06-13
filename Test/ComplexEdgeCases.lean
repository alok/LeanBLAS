import LeanBLAS

open BLAS BLAS.CBLAS

/-!
# Complex BLAS Edge Case Tests

This module tests edge cases and special values for complex BLAS operations,
including branch cuts, overflow/underflow, and special floating-point values.
-/

/-- Helper to create infinity -/
def infinity : Float := 1.0 / 0.0

/-- Helper to create NaN -/
def nan : Float := 0.0 / 0.0

/-- Check if a float is NaN -/
def isNaN (x : Float) : Bool := x != x

/-- Check if a complex number has NaN component -/
def complexHasNaN (z : ComplexFloat) : Bool := isNaN z.x || isNaN z.y

/-- Check if a complex number has infinity component -/
def complexHasInf (z : ComplexFloat) : Bool := 
  z.x == infinity || z.x == -infinity || z.y == infinity || z.y == -infinity

/-- Test special floating-point values -/
def test_special_values : IO Unit := do
  IO.println "\n=== Testing Special Floating-Point Values ==="
  
  -- Test with infinity
  let inf_vec := #c64[⟨infinity, 0.0⟩, ⟨1.0, 0.0⟩]
  let norm_inf := dznrm2 2 inf_vec 0 1
  assert! (norm_inf == infinity)
  IO.println s!"✓ Norm with infinity component = {norm_inf}"
  
  -- Test with NaN
  let nan_vec := #c64[⟨nan, 0.0⟩, ⟨1.0, 0.0⟩]
  let norm_nan := dznrm2 2 nan_vec 0 1
  assert! (isNaN norm_nan)
  IO.println "✓ Norm with NaN component = NaN"
  
  -- Test dot product with infinity
  let x := #c64[⟨infinity, 0.0⟩, ⟨1.0, 0.0⟩]
  let y := #c64[⟨0.0, 0.0⟩, ⟨1.0, 0.0⟩]
  let dot_inf := zdotc 2 x 0 1 y 0 1
  -- inf * 0 should give NaN
  assert! (complexHasNaN dot_inf)
  IO.println "✓ Dot product inf * 0 gives NaN"
  
  -- Test scaling by zero
  let z := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
  let scaled_zero := zscal 2 ComplexFloat.zero z 0 1
  let result := scaled_zero.toComplexFloatArray
  assert! (result.get! 0 == ComplexFloat.zero)
  assert! (result.get! 1 == ComplexFloat.zero)
  IO.println "✓ Scaling by zero gives zero vector"

/-- Test branch cuts for complex functions -/
def test_branch_cuts : IO Unit := do
  IO.println "\n=== Testing Branch Cuts ==="
  
  -- Test log branch cut on negative real axis
  let neg_real := #c64[⟨-1.0, 0.0⟩, ⟨-2.0, 0.0⟩]
  let log_result := LevelOneDataExt.log (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 neg_real 0 1
  let log_arr := log_result.toComplexFloatArray
  
  -- log(-1) should be 0 + πi
  let log_neg1 := log_arr.get! 0
  let pi := 3.141592653589793
  assert! (Float.abs log_neg1.x < 1e-10)  -- Real part ≈ 0
  assert! (Float.abs (log_neg1.y - pi) < 1e-10)  -- Imaginary part ≈ π
  IO.println s!"✓ log(-1+0i) = {log_neg1} (branch cut gives 0+πi)"
  
  -- Test sqrt branch cut
  let sqrt_result := LevelOneDataExt.sqrt (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 1 neg_real 0 1
  let sqrt_arr := sqrt_result.toComplexFloatArray
  let sqrt_neg1 := sqrt_arr.get! 0
  
  -- sqrt(-1) should be 0 + i
  assert! (Float.abs sqrt_neg1.x < 1e-10)  -- Real part ≈ 0
  assert! (Float.abs (sqrt_neg1.y - 1.0) < 1e-10)  -- Imaginary part ≈ 1
  IO.println s!"✓ sqrt(-1+0i) = {sqrt_neg1} (branch cut gives 0+i)"

/-- Test overflow and underflow -/
def test_overflow_underflow : IO Unit := do
  IO.println "\n=== Testing Overflow and Underflow ==="
  
  -- Test large values that might overflow in naive implementations
  let large := 1e150
  let large_vec := #c64[⟨large, large⟩, ⟨large, large⟩]
  let norm_large := dznrm2 2 large_vec 0 1
  
  -- Should compute sqrt(2 * (large² + large²) + 2 * (large² + large²))
  -- = sqrt(4 * 2 * large²) = 2 * sqrt(2) * large ≈ 2.828 * large
  let expected := 2.0 * Float.sqrt 2.0 * large
  assert! (Float.abs (norm_large / expected - 1.0) < 1e-10)
  IO.println s!"✓ Norm of large values handled correctly: {norm_large}"
  
  -- Test small values (potential underflow)
  let small := 1e-150
  let small_vec := #c64[⟨small, 0.0⟩, ⟨0.0, small⟩]
  let norm_small := dznrm2 2 small_vec 0 1
  let expected_small := Float.sqrt 2.0 * small
  assert! (Float.abs (norm_small / expected_small - 1.0) < 1e-10)
  IO.println s!"✓ Norm of small values handled correctly: {norm_small}"
  
  -- Test mixed large/small (stress test for scaling algorithms)
  let mixed := #c64[⟨1e150, 0.0⟩, ⟨1e-150, 0.0⟩]
  let norm_mixed := dznrm2 2 mixed 0 1
  -- Result should be dominated by large value
  assert! (Float.abs (norm_mixed / 1e150 - 1.0) < 1e-10)
  IO.println "✓ Mixed large/small values handled correctly"

/-- Test zero stride edge cases -/
def test_zero_stride : IO Unit := do
  IO.println "\n=== Testing Zero Stride ==="
  
  -- Zero stride means repeat the same element
  let x := #c64[⟨2.0, 3.0⟩, ⟨5.0, 6.0⟩]
  
  -- Dot product with stride 0 should compute: conj(x[0]) * x[0] * N
  let dot_zero_stride := zdotc 3 x 0 0 x 0 0
  -- (2-3i) * (2+3i) * 3 = (4 + 9) * 3 = 39
  let expected := ComplexFloat.mk 39.0 0.0
  assert! (Float.abs (dot_zero_stride.x - expected.x) < 1e-10)
  assert! (Float.abs (dot_zero_stride.y - expected.y) < 1e-10)
  IO.println s!"✓ Dot product with zero stride = {dot_zero_stride} (expected: 39+0i)"
  
  -- Norm with zero stride
  let norm_zero_stride := dznrm2 5 x 0 0
  -- sqrt(|2+3i|² * 5) = sqrt(13 * 5) = sqrt(65)
  let expected_norm := Float.sqrt 65.0
  assert! (Float.abs (norm_zero_stride - expected_norm) < 1e-10)
  IO.println s!"✓ Norm with zero stride = {norm_zero_stride} (expected: {expected_norm})"

/-- Test empty vectors -/
def test_empty_vectors : IO Unit := do
  IO.println "\n=== Testing Empty Vectors ==="
  
  let empty := #c64[]
  
  -- Dot product of empty vectors should be zero
  let dot_empty := zdotc 0 empty 0 1 empty 0 1
  assert! (dot_empty == ComplexFloat.zero)
  IO.println "✓ Dot product of empty vectors = 0+0i"
  
  -- Norm of empty vector should be zero
  let norm_empty := dznrm2 0 empty 0 1
  assert! (norm_empty == 0.0)
  IO.println "✓ Norm of empty vector = 0"
  
  -- asum of empty vector should be zero
  let asum_empty := dzasum 0 empty 0 1
  assert! (asum_empty == 0.0)
  IO.println "✓ Sum of absolute values of empty vector = 0"

/-- Test operations with unit stride on non-contiguous data -/
def test_stride_patterns : IO Unit := do
  IO.println "\n=== Testing Stride Patterns ==="
  
  -- Create interleaved data
  let data := #c64[⟨1.0, 0.0⟩, ⟨-1.0, 0.0⟩, ⟨2.0, 0.0⟩, ⟨-2.0, 0.0⟩, ⟨3.0, 0.0⟩, ⟨-3.0, 0.0⟩]
  
  -- Extract positive values with stride 2
  let norm_pos := dznrm2 3 data 0 2
  let expected_pos := Float.sqrt 14.0  -- sqrt(1² + 2² + 3²)
  assert! (Float.abs (norm_pos - expected_pos) < 1e-10)
  IO.println s!"✓ Norm of strided positive values = {norm_pos}"
  
  -- Extract negative values with stride 2, offset 1
  let norm_neg := dznrm2 3 data 1 2
  assert! (Float.abs (norm_neg - expected_pos) < 1e-10)
  IO.println s!"✓ Norm of strided negative values = {norm_neg}"
  
  -- Dot product between positive and negative strided views
  let dot_strided := zdotc 3 data 0 2 data 1 2
  -- conj(1)*(-1) + conj(2)*(-2) + conj(3)*(-3) = -1 - 4 - 9 = -14
  let expected_dot := ComplexFloat.mk (-14.0) 0.0
  assert! (Float.abs (dot_strided.x - expected_dot.x) < 1e-10)
  assert! (Float.abs (dot_strided.y - expected_dot.y) < 1e-10)
  IO.println s!"✓ Dot product of strided views = {dot_strided}"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Complex BLAS Edge Case Tests ==="
  
  test_special_values
  test_branch_cuts
  test_overflow_underflow
  test_zero_stride
  test_empty_vectors
  test_stride_patterns
  
  IO.println "\n✅ All edge case tests passed!"
  IO.println "Complex BLAS operations handle special cases correctly."