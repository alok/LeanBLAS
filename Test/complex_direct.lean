import LeanBLAS
import LeanBLAS.CBLAS.LevelOneComplex

open BLAS CBLAS

-- Direct test without using the macro
def main : IO Unit := do
  IO.println "=== Direct Complex BLAS Test ==="
  
  -- Create a simple ComplexFloat64Array for testing
  -- For now, we'll just test that the module compiles and links
  IO.println "Complex Level 1 BLAS module loaded successfully"
  
  -- Test ComplexFloat creation
  let c1 := ComplexFloat.mk 3.0 4.0
  let c2 := ComplexFloat.mk 1.0 0.0
  
  IO.println s!"Complex number 1: {c1}"
  IO.println s!"Complex number 2: {c2}"
  IO.println s!"Absolute value of 3+4i: {c1.abs}"
  
  -- Test complex arithmetic
  let sum := c1 + c2
  let prod := c1 * c2
  IO.println s!"Sum: {sum}"
  IO.println s!"Product: {prod}"