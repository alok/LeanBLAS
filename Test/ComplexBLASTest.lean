import LeanBLAS

open BLAS BLAS.CBLAS

/-- Test complex Level 1 operations -/
def test_complex_level1 : IO Unit := do
  IO.println "\n=== Complex Level 1 Tests ==="
  
  -- Test zdotu (dot product without conjugation)
  IO.println "Testing zdotu (dot product):"
  let x1 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 240, 63,  -- 1.0 (real)
    0, 0, 0, 0, 0, 0, 0, 0,     -- 0.0 (imag)
    0, 0, 0, 0, 0, 0, 0, 64,    -- 2.0 (real)
    0, 0, 0, 0, 0, 0, 240, 63   -- 1.0 (imag)
  ]) (by decide)
  
  let y1 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 8, 64,    -- 3.0 (real)
    0, 0, 0, 0, 0, 0, 16, 64,   -- 4.0 (imag)
    0, 0, 0, 0, 0, 0, 240, 63,  -- 1.0 (real)
    0, 0, 0, 0, 0, 0, 0, 192    -- -2.0 (imag)
  ]) (by decide)
  
  -- zdotu([1+0i, 2+1i], [3+4i, 1-2i]) = (1+0i)*(3+4i) + (2+1i)*(1-2i)
  --                                    = 3+4i + (2-4i+1i-2i²) 
  --                                    = 3+4i + (2-3i+2)
  --                                    = 3+4i + 4-3i = 7+1i
  let dot_result := CBLAS.zdotu 2 x1 0 1 y1 0 1
  IO.println s!"  Expected: 7+1i"
  IO.println s!"  Got: {dot_result}"
  
  -- Test zscal (scale by complex number)
  IO.println "\nTesting zscal (complex scaling):"
  let scale_factor : ComplexFloat := ⟨2.0, -1.0⟩  -- 2-i
  let x_copy := x1  -- Would need proper copy in real test
  let scaled := CBLAS.zscal 2 scale_factor x_copy 0 1
  IO.println s!"  Scaling [1+0i, 2+1i] by {scale_factor}"
  IO.println "  Expected: [2-i, 5+0i]"
  
  -- Test znrm2 (2-norm)
  IO.println "\nTesting znrm2 (2-norm):"
  let norm := CBLAS.dznrm2 2 x1 0 1
  IO.println s!"  ||[1+0i, 2+1i]||₂ = sqrt(1² + 2² + 1²) = sqrt(6) ≈ {norm}"

/-- Test complex Level 2 operations -/
def test_complex_level2 : IO Unit := do
  IO.println "\n=== Complex Level 2 Tests ==="
  
  IO.println "Testing zhemv (Hermitian matrix-vector multiply):"
  -- Create a 2x2 Hermitian matrix A = [[2, 1-i], [1+i, 3]]
  -- Stored as full matrix for simplicity
  let A := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 0, 64,     -- 2.0 (real)
    0, 0, 0, 0, 0, 0, 0, 0,      -- 0.0 (imag)
    0, 0, 0, 0, 0, 0, 240, 63,   -- 1.0 (real)
    0, 0, 0, 0, 0, 0, 240, 191,  -- -1.0 (imag)
    0, 0, 0, 0, 0, 0, 240, 63,   -- 1.0 (real)  
    0, 0, 0, 0, 0, 0, 240, 63,   -- 1.0 (imag)
    0, 0, 0, 0, 0, 0, 8, 64,     -- 3.0 (real)
    0, 0, 0, 0, 0, 0, 0, 0       -- 0.0 (imag)
  ]) (by decide)
  
  let x := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 240, 63,   -- 1.0 (real)
    0, 0, 0, 0, 0, 0, 240, 63,   -- 1.0 (imag)
    0, 0, 0, 0, 0, 0, 0, 64,     -- 2.0 (real)
    0, 0, 0, 0, 0, 0, 0, 0       -- 0.0 (imag)
  ]) (by decide)
  
  let y := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 0, 0,      -- 0.0 (real)
    0, 0, 0, 0, 0, 0, 0, 0,      -- 0.0 (imag)
    0, 0, 0, 0, 0, 0, 0, 0,      -- 0.0 (real)
    0, 0, 0, 0, 0, 0, 0, 0       -- 0.0 (imag)
  ]) (by decide)
  
  -- A * x = [[2, 1-i], [1+i, 3]] * [1+i, 2] 
  --       = [2(1+i) + (1-i)2, (1+i)(1+i) + 3*2]
  --       = [2+2i + 2-2i, 1+i+i+i² + 6]
  --       = [4+0i, 1+2i-1+6] = [4+0i, 6+2i]
  let result := hemv Order.RowMajor UpLo.Upper 2 ⟨1.0, 0.0⟩ A 0 2 x 0 1 ⟨0.0, 0.0⟩ y 0 1
  IO.println "  A * [1+i, 2+0i] where A is Hermitian [[2, 1-i], [1+i, 3]]"
  IO.println "  Expected: [4+0i, 6+2i]"
  
  IO.println "\nTesting zgerc (rank-1 update with conjugation):"
  -- A = A + x * conj(y)^T
  let A2 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0,      -- 1+0i
    0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0,        -- 2+0i
    0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 0, 0,        -- 3+0i
    0, 0, 0, 0, 0, 0, 16, 64, 0, 0, 0, 0, 0, 0, 0, 0        -- 4+0i
  ]) (by decide)
  
  let x2 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 240, 63,   -- 1+i
    0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0         -- 2+0i
  ]) (by decide)
  
  let y2 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 240, 63,     -- 2+i
    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 240, 191   -- 1-i
  ]) (by decide)
  
  let updated := gerc Order.RowMajor 2 2 ⟨1.0, 0.0⟩ x2 0 1 y2 0 1 A2 0 2
  IO.println "  A + [1+i, 2] * conj([2+i, 1-i])^T"
  IO.println "  Expected: A + [[3+i, 2-2i], [4+2i, 2]]"

/-- Test complex Level 3 operations -/
def test_complex_level3 : IO Unit := do
  IO.println "\n=== Complex Level 3 Tests ==="
  
  IO.println "Testing zgemm (general matrix multiply):"
  -- A = [[1+i, 2], [3, 4+i]], B = [[5, 6+i], [7+i, 8]]
  let A := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 240, 63,   -- 1+i
    0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0,        -- 2+0i
    0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 0, 0,        -- 3+0i
    0, 0, 0, 0, 0, 0, 16, 64, 0, 0, 0, 0, 0, 0, 240, 63     -- 4+i
  ]) (by decide)
  
  let B := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 20, 64, 0, 0, 0, 0, 0, 0, 0, 0,       -- 5+0i
    0, 0, 0, 0, 0, 0, 24, 64, 0, 0, 0, 0, 0, 0, 240, 63,    -- 6+i
    0, 0, 0, 0, 0, 0, 28, 64, 0, 0, 0, 0, 0, 0, 240, 63,    -- 7+i
    0, 0, 0, 0, 0, 0, 32, 64, 0, 0, 0, 0, 0, 0, 0, 0        -- 8+0i
  ]) (by decide)
  
  let C := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]) (by decide)
  
  -- C = A * B
  let result := CBLAS.zgemm Order.RowMajor Transpose.NoTrans Transpose.NoTrans 
                2 2 2 ⟨1.0, 0.0⟩ A 0 2 B 0 2 ⟨0.0, 0.0⟩ C 0 2
  IO.println "  C = A * B where:"
  IO.println "  A = [[1+i, 2], [3, 4+i]], B = [[5, 6+i], [7+i, 8]]"
  IO.println "  Computing C₁₁ = (1+i)*5 + 2*(7+i) = 5+5i + 14+2i = 19+7i"
  
  IO.println "\nTesting zherk (Hermitian rank-k update):"
  -- C = alpha * A * A^H + beta * C
  let A2 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64,     -- 1+2i
    0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 240, 191,    -- 3-i
    0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 240, 63,     -- 2+i
    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64      -- 1+2i
  ]) (by decide)
  
  let C2 := ComplexFloat64Array.mk (ByteArray.mk #[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]) (by decide)
  
  -- Note: herk requires real alpha and beta
  let herk_result := herk Order.RowMajor UpLo.Upper Transpose.NoTrans 
                     2 2 1.0 A2 0 2 0.0 C2 0 2
  IO.println "  C = A * A^H (Hermitian product)"
  IO.println "  Result is Hermitian (only upper triangle computed)"

/-- Run all complex BLAS tests -/
def main : IO Unit := do
  IO.println "=== Complex BLAS Test Suite ==="
  IO.println "Testing complex number support in LeanBLAS"
  
  test_complex_level1
  test_complex_level2
  test_complex_level3
  
  IO.println "\n=== All complex tests completed ==="
  IO.println "Note: Visual verification needed for actual results"
  IO.println "In production, would compare with expected values programmatically"