import LeanBLAS

open BLAS

def main : IO Unit := do
  -- Create using the #c64 syntax which should work
  let arr := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]
  IO.println "Successfully created ComplexFloat64Array using #c64 syntax"
  
  -- Try to convert back
  let complexArr := arr.toComplexFloatArray
  IO.println s!"Converted back, size: {complexArr.size}"