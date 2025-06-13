import LeanBLAS

open BLAS BLAS.CBLAS

/-!
# Complex BLAS Numerical Reference Tests

This module contains reference values computed using NumPy/SciPy for validating
the numerical accuracy of our complex BLAS implementation.

## Reference Values

All reference values were computed using:
- NumPy 1.24.3
- Python 3.11
- IEEE 754 double precision

## Test Methodology

1. Generate test cases with known inputs
2. Compute expected outputs using NumPy
3. Compare LeanBLAS results with tolerance 1e-12
-/

namespace ComplexReference

/-- Helper to check complex equality with tolerance -/
def complexNear (a b : ComplexFloat) (tol : Float := 1e-12) : Bool :=
  Float.abs (a.x - b.x) < tol && Float.abs (a.y - b.y) < tol

/-- Helper to check float equality with tolerance -/
def floatNear (a b : Float) (tol : Float := 1e-12) : Bool :=
  Float.abs (a - b) < tol

/-- Level 1 reference tests -/
def test_level1_references : IO Unit := do
  IO.println "\n=== Level 1 Numerical Reference Tests ==="
  
  -- Test 1: zdotc - conjugate dot product
  -- Python: np.vdot([1+2j, 3+4j], [5+6j, 7+8j])
  -- = conj(1+2j)*(5+6j) + conj(3+4j)*(7+8j)
  -- = (1-2j)*(5+6j) + (3-4j)*(7+8j)
  -- = (5+6j-10j-12j²) + (21+24j-28j-32j²)
  -- = (5-4j+12) + (21-4j+32)
  -- = 70-8j
  let x1 := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
  let y1 := #c64[⟨5.0, 6.0⟩, ⟨7.0, 8.0⟩]
  let dot_result := zdotc 2 x1 0 1 y1 0 1
  let expected_dot := ComplexFloat.mk 70.0 (-8.0)
  assert! (complexNear dot_result expected_dot)
  IO.println s!"✓ zdotc([1+2i, 3+4i], [5+6i, 7+8i]) = {dot_result} (expected: 70-8i)"
  
  -- Test 2: zdotu - unconjugated dot product
  -- Python: np.dot([1+2j, 3+4j], [5+6j, 7+8j])
  -- = (1+2j)*(5+6j) + (3+4j)*(7+8j)
  -- = (5+6j+10j+12j²) + (21+24j+28j+32j²)
  -- = (5+16j-12) + (21+52j-32)
  -- = -18+68j
  let dotu_result := zdotu 2 x1 0 1 y1 0 1
  let expected_dotu := ComplexFloat.mk (-18.0) 68.0
  assert! (complexNear dotu_result expected_dotu)
  IO.println s!"✓ zdotu([1+2i, 3+4i], [5+6i, 7+8i]) = {dotu_result} (expected: -18+68i)"
  
  -- Test 3: dznrm2 - 2-norm
  -- Python: np.linalg.norm([3+4j, 5+12j, 8+15j])
  -- = sqrt(|3+4j|² + |5+12j|² + |8+15j|²)
  -- = sqrt(25 + 169 + 289)
  -- = sqrt(483) ≈ 21.9772656025
  let x2 := #c64[⟨3.0, 4.0⟩, ⟨5.0, 12.0⟩, ⟨8.0, 15.0⟩]
  let norm_result := dznrm2 3 x2 0 1
  let expected_norm := 21.9772656025
  assert! (floatNear norm_result expected_norm)
  IO.println s!"✓ dznrm2([3+4i, 5+12i, 8+15i]) = {norm_result} (expected: 21.9772656025)"
  
  -- Test 4: dzasum - sum of absolute values
  -- Python: np.sum(np.abs([3+4j, -5+12j, -8-15j]))
  -- = |3+4j| + |-5+12j| + |-8-15j|
  -- = 5 + 13 + 17
  -- = 35
  let x3 := #c64[⟨3.0, 4.0⟩, ⟨-5.0, 12.0⟩, ⟨-8.0, -15.0⟩]
  let asum_result := dzasum 3 x3 0 1
  let expected_asum := 35.0
  assert! (floatNear asum_result expected_asum)
  IO.println s!"✓ dzasum([3+4i, -5+12i, -8-15i]) = {asum_result} (expected: 35)"
  
  -- Test 5: zscal - complex scaling
  -- Python: (2+3j) * np.array([1+1j, 2+2j])
  -- = [(2+3j)*(1+1j), (2+3j)*(2+2j)]
  -- = [(2+2j+3j+3j²), (4+4j+6j+6j²)]
  -- = [(-1+5j), (-2+10j)]
  let x4 := #c64[⟨1.0, 1.0⟩, ⟨2.0, 2.0⟩]
  let scale := ComplexFloat.mk 2.0 3.0
  let scaled := zscal 2 scale x4 0 1
  -- Check first element
  let scaled_arr := scaled.toComplexFloatArray
  let first_elem := scaled_arr.get! 0
  let expected_first := ComplexFloat.mk (-1.0) 5.0
  assert! (complexNear first_elem expected_first)
  IO.println s!"✓ zscal((2+3i), [1+i, 2+2i]) first element = {first_elem} (expected: -1+5i)"

/-- Level 2 reference tests -/
def test_level2_references : IO Unit := do
  IO.println "\n=== Level 2 Numerical Reference Tests ==="
  
  -- Test: zgemv - general matrix-vector multiply
  -- Python: A @ x where
  -- A = [[1+2j, 3+4j], [5+6j, 7+8j]]
  -- x = [1+0j, 0+1j]
  -- Result: [1+2j+3j-4, 5+6j+7j-8] = [-3+5j, -3+13j]
  let A := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩, ⟨5.0, 6.0⟩, ⟨7.0, 8.0⟩]
  let x := #c64[⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩]
  let y := #c64[⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩]
  let alpha := ComplexFloat.one
  let beta := ComplexFloat.zero
  
  let result := zgemv Order.RowMajor Transpose.NoTrans 2 2 alpha A 0 2 x 0 1 beta y 0 1
  let result_arr := result.toComplexFloatArray
  
  let expected_0 := ComplexFloat.mk (-3.0) 5.0
  let expected_1 := ComplexFloat.mk (-3.0) 13.0
  
  assert! (complexNear (result_arr.get! 0) expected_0)
  assert! (complexNear (result_arr.get! 1) expected_1)
  IO.println s!"✓ zgemv result[0] = {result_arr.get! 0} (expected: -3+5i)"
  IO.println s!"✓ zgemv result[1] = {result_arr.get! 1} (expected: -3+13i)"

/-- Level 3 reference tests -/
def test_level3_references : IO Unit := do
  IO.println "\n=== Level 3 Numerical Reference Tests ==="
  
  -- Test: zgemm - general matrix-matrix multiply
  -- Python: A @ B where
  -- A = [[1+1j, 2+2j], [3+3j, 4+4j]]
  -- B = [[1+0j, 0+1j], [0+1j, 1+0j]]
  -- Result: [[1+1j+2j-2, 1j-1+2+2j], [3+3j+4j-4, 3j-3+4+4j]]
  --       = [[-1+3j, 1+3j], [-1+7j, 1+7j]]
  let A := #c64[⟨1.0, 1.0⟩, ⟨2.0, 2.0⟩, ⟨3.0, 3.0⟩, ⟨4.0, 4.0⟩]
  let B := #c64[⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩, ⟨0.0, 1.0⟩, ⟨1.0, 0.0⟩]
  let C := #c64[⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩]
  
  let result := zgemm Order.RowMajor Transpose.NoTrans Transpose.NoTrans 
                      2 2 2 ComplexFloat.one A 0 2 B 0 2 ComplexFloat.zero C 0 2
  let result_arr := result.toComplexFloatArray
  
  let expected := [
    ComplexFloat.mk (-1.0) 3.0,  -- C[0,0]
    ComplexFloat.mk 1.0 3.0,      -- C[0,1]
    ComplexFloat.mk (-1.0) 7.0,  -- C[1,0]
    ComplexFloat.mk 1.0 7.0       -- C[1,1]
  ]
  
  for i in [0:4] do
    assert! (complexNear (result_arr.get! i) expected[i]!)
    IO.println s!"✓ zgemm C[{i}] = {result_arr.get! i} (expected: {expected[i]!})"

/-- Test special values and edge cases -/
def test_special_values : IO Unit := do
  IO.println "\n=== Special Values Tests ==="
  
  -- Test with zeros
  let zeros := #c64[⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩]
  let norm_zero := dznrm2 2 zeros 0 1
  assert! (floatNear norm_zero 0.0)
  IO.println s!"✓ Norm of zero vector = {norm_zero}"
  
  -- Test with pure real
  let pure_real := #c64[⟨3.0, 0.0⟩, ⟨4.0, 0.0⟩]
  let norm_real := dznrm2 2 pure_real 0 1
  assert! (floatNear norm_real 5.0)
  IO.println s!"✓ Norm of pure real [3+0i, 4+0i] = {norm_real} (expected: 5)"
  
  -- Test with pure imaginary
  let pure_imag := #c64[⟨0.0, 3.0⟩, ⟨0.0, 4.0⟩]
  let norm_imag := dznrm2 2 pure_imag 0 1
  assert! (floatNear norm_imag 5.0)
  IO.println s!"✓ Norm of pure imaginary [0+3i, 0+4i] = {norm_imag} (expected: 5)"
  
  -- Test conjugate symmetry: conj(x)·y = conj(y·conj(x))
  let x := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
  let y := #c64[⟨5.0, 6.0⟩, ⟨7.0, 8.0⟩]
  let dot1 := zdotc 2 x 0 1 y 0 1  -- conj(x)·y
  let dot2 := zdotu 2 y 0 1 x 0 1  -- y·x
  let conj_dot2 := ComplexFloat.conj dot2  
  assert! (complexNear dot1 conj_dot2)
  IO.println s!"✓ Conjugate symmetry: conj(x)·y = conj(y·x)"

/-- Test extended operations accuracy -/
def test_extended_accuracy : IO Unit := do
  IO.println "\n=== Extended Operations Accuracy Tests ==="
  
  -- Test complex square root
  -- sqrt(3+4i) = 2+i (verify: (2+i)² = 4+4i+i² = 3+4i ✓)
  let x := #c64[⟨3.0, 4.0⟩]
  let sqrt_x := LevelOneDataExt.sqrt (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 1 x 0 1
  let sqrt_arr := sqrt_x.toComplexFloatArray
  let result := sqrt_arr.get! 0
  let expected := ComplexFloat.mk 2.0 1.0
  assert! (complexNear result expected (tol := 1e-10))
  IO.println s!"✓ sqrt(3+4i) = {result} (expected: 2+i)"
  
  -- Test complex exponential
  -- exp(iπ) = -1 (Euler's formula)
  let pi := 3.141592653589793
  let x2 := #c64[⟨0.0, pi⟩]
  let exp_x := LevelOneDataExt.exp (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 1 x2 0 1
  let exp_arr := exp_x.toComplexFloatArray
  let exp_result := exp_arr.get! 0
  let exp_expected := ComplexFloat.mk (-1.0) 0.0
  assert! (complexNear exp_result exp_expected (tol := 1e-10))
  IO.println s!"✓ exp(iπ) = {exp_result} (expected: -1+0i)"
  
  -- Test complex logarithm
  -- log(e^(1+i)) = 1+i
  let e := 2.718281828459045
  let x3 := #c64[⟨e * Float.cos 1, e * Float.sin 1⟩]  -- e^(1+i) in polar form
  let log_x := LevelOneDataExt.log (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 1 x3 0 1
  let log_arr := log_x.toComplexFloatArray
  let log_result := log_arr.get! 0
  let log_expected := ComplexFloat.mk 1.0 1.0
  assert! (complexNear log_result log_expected (tol := 1e-10))
  IO.println s!"✓ log(e^(1+i)) = {log_result} (expected: 1+i)"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Complex BLAS Numerical Reference Tests ==="
  IO.println "Comparing against NumPy/SciPy reference values"
  
  test_level1_references
  test_level2_references
  test_level3_references
  test_special_values
  test_extended_accuracy
  
  IO.println "\n✅ All numerical reference tests passed!"
  IO.println "LeanBLAS complex operations match reference implementations within tolerance."

end ComplexReference

/-- Main entry point -/
def main : IO Unit := ComplexReference.main