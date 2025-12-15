import LeanBLAS

open BLAS BLAS.CBLAS

/-- Test the convenient constructors for ComplexFloat64Array -/
def main : IO Unit := do
  IO.println "=== Testing ComplexFloat64Array Constructors ==="
  
  -- Test zeros
  IO.println "\n1. Testing zeros constructor:"
  let zeros := ComplexFloat64Array.zeros 5
  IO.println s!"  zeros(5) = {zeros}"
  
  -- Test ones
  IO.println "\n2. Testing ones constructor:"
  let ones := ComplexFloat64Array.ones 3
  IO.println s!"  ones(3) = {ones}"
  
  -- Test const
  IO.println "\n3. Testing const constructor:"
  let const_val := ComplexFloat.mk 2.0 3.0
  let const_arr := ComplexFloat64Array.const 4 const_val
  IO.println s!"  const(4, 2+3i) = {const_arr}"
  
  -- Test ofList
  IO.println "\n4. Testing ofList constructor:"
  let list_arr := ComplexFloat64Array.ofList [⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩, ⟨-1.0, 0.0⟩]
  IO.println s!"  ofList([1, i, -1]) = {list_arr}"
  
  -- Test ofRealImag
  IO.println "\n5. Testing ofRealImag constructor:"
  let reals := #[1.0, 2.0, 3.0]
  let imags := #[4.0, 5.0, 6.0]
  let ri_arr := ComplexFloat64Array.ofRealImag reals imags
  IO.println s!"  ofRealImag([1,2,3], [4,5,6]) = {ri_arr}"
  
  -- Test ofReal
  IO.println "\n6. Testing ofReal constructor:"
  let real_arr := ComplexFloat64Array.ofReal #[1.0, -2.0, 3.0]
  IO.println s!"  ofReal([1,-2,3]) = {real_arr}"
  
  -- Test ofImag
  IO.println "\n7. Testing ofImag constructor:"
  let imag_arr := ComplexFloat64Array.ofImag #[1.0, -2.0, 3.0]
  IO.println s!"  ofImag([1,-2,3]) = {imag_arr}"
  
  -- Test range
  IO.println "\n8. Testing range constructor:"
  let range_arr := ComplexFloat64Array.range ⟨0.0, 0.0⟩ ⟨2.0, 4.0⟩ 5
  IO.println s!"  range(0+0i, 2+4i, 5) = {range_arr}"
  
  -- Test random
  IO.println "\n9. Testing random constructor:"
  let rand_arr := ComplexFloat64Array.random 5 42
  IO.println s!"  random(5, seed=42) = {rand_arr}"
  
  -- Test eye (identity matrix)
  IO.println "\n10. Testing eye constructor:"
  let eye3 := ComplexFloat64Array.eye 3
  IO.println s!"  eye(3) = {eye3}"
  
  -- Test diag
  IO.println "\n11. Testing diag constructor:"
  let diag_vec := ComplexFloat64Array.ofList [⟨1.0, 0.0⟩, ⟨2.0, 0.0⟩, ⟨3.0, 0.0⟩]
  let diag_mat := ComplexFloat64Array.diag diag_vec
  IO.println s!"  diag([1,2,3]) = {diag_mat}"
  
  -- Test extraction of real and imaginary parts
  IO.println "\n12. Testing real/imag extraction:"
  let complex := ComplexFloat64Array.ofList [⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
  let real_parts := complex.realParts
  let imag_parts := complex.imagParts
  IO.println s!"  Complex array: {complex}"
  IO.println s!"  Real parts: {real_parts.toFloatArray}"
  IO.println s!"  Imag parts: {imag_parts.toFloatArray}"
  
  -- Demonstrate usage in BLAS operations
  IO.println "\n13. Using constructors with BLAS operations:"
  let x := ComplexFloat64Array.ones 3
  let y := ComplexFloat64Array.const 3 ⟨2.0, 1.0⟩
  let dot := zdotc 3 x 0 1 y 0 1
  IO.println s!"  dot(ones(3), const(3, 2+i)) = {dot}"
  IO.println s!"  Expected: conj(1) * (2+i) * 3 = 1 * (2+i) * 3 = 6+3i"
  
  IO.println "\n✅ All constructor tests completed!"
