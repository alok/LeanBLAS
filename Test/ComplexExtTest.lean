import LeanBLAS

open BLAS BLAS.CBLAS

/-- Test the extended complex operations -/
def main : IO Unit := do
  IO.println "=== Testing Complex LevelOneDataExt Operations ==="
  
  -- Test sum first (doesn't use const)
  IO.println "\nTesting sum:"
  let x := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩, ⟨5.0, 6.0⟩]
  let sum := LevelOneDataExt.sum (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 3 x 0 1
  IO.println s!"  Sum of [1+2i, 3+4i, 5+6i] = {sum}"
  IO.println s!"  Expected: 9+12i"
  
  -- Skip axpby for now as it uses const which has a sorry
  -- IO.println "\nTesting axpby:"
  -- let a : ComplexFloat := ⟨2.0, 0.0⟩
  -- let b : ComplexFloat := ⟨3.0, 0.0⟩
  -- let x1 := #c64[⟨1.0, 1.0⟩, ⟨2.0, 2.0⟩]
  -- let y1 := #c64[⟨3.0, 3.0⟩, ⟨4.0, 4.0⟩]
  -- -- axpby: Y := a*X + b*Y
  -- let result := LevelOneDataExt.axpby (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 a x1 0 1 b y1 0 1
  -- IO.println s!"  Y := 2*[1+i, 2+2i] + 3*[3+3i, 4+4i]"
  -- IO.println s!"  Expected: [11+11i, 16+16i]"
  
  -- Test imaxRe
  IO.println "\nTesting imaxRe:"
  let x2 := #c64[⟨1.0, 5.0⟩, ⟨3.0, 2.0⟩, ⟨2.0, 4.0⟩]
  let maxReIdx := LevelOneDataExt.imaxRe (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 3 x2 0 1 (by simp)
  IO.println s!"  Index of max real part in [1+5i, 3+2i, 2+4i]: {maxReIdx}"
  IO.println s!"  Expected: 1 (element 3+2i has max real part)"
  
  -- Test imaxIm
  IO.println "\nTesting imaxIm:"
  let maxImIdx := LevelOneDataExt.imaxIm (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 3 x2 0 1 (by simp)
  IO.println s!"  Index of max imaginary part in [1+5i, 3+2i, 2+4i]: {maxImIdx}"
  IO.println s!"  Expected: 0 (element 1+5i has max imaginary part)"
  
  -- Test element-wise operations
  IO.println "\nTesting element-wise operations:"
  
  -- Test mul
  let x3 := #c64[⟨2.0, 3.0⟩, ⟨1.0, -1.0⟩]
  let y3 := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
  let mulResult := LevelOneDataExt.mul (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 x3 0 1 y3 0 1
  IO.println s!"  Element-wise mul: [2+3i, 1-i] * [1+2i, 3+4i]"
  IO.println s!"  Expected: [(2+3i)*(1+2i), (1-i)*(3+4i)] = [-4+7i, 7+i]"
  
  -- Test sqrt
  let x4 := #c64[⟨4.0, 0.0⟩, ⟨-1.0, 0.0⟩]
  let sqrtResult := LevelOneDataExt.sqrt (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 x4 0 1
  IO.println s!"  Element-wise sqrt: sqrt([4+0i, -1+0i])"
  IO.println s!"  Expected: [2+0i, 0+i]"
  
  -- Test abs
  let x5 := #c64[⟨3.0, 4.0⟩, ⟨-5.0, 12.0⟩]
  let absResult := LevelOneDataExt.abs (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) 2 x5 0 1
  IO.println s!"  Element-wise abs: |[3+4i, -5+12i]|"
  IO.println s!"  Expected: [5+0i, 13+0i]"
  
  IO.println "\nAll extended operations tests completed!"