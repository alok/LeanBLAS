import LeanBLAS
import LeanBLAS.ComplexArray

open BLAS

def main : IO Unit := do
  -- Test creating ComplexFloatArray directly
  let arr1 := ComplexFloatArray.ofArray #[⟨1.0, 2.0⟩]
  IO.println s!"Created ComplexFloatArray with size: {arr1.size}"
  
  -- Test if it's a valid object
  IO.println s!"ComplexFloatArray data size: {arr1.data.size}"
  
  -- Now test the macro
  IO.println "\nTesting #c64 macro..."
  let arr2 := #c64[⟨1.0, 2.0⟩]
  IO.println "Created ComplexFloat64Array using #c64 macro - no crash means it worked!"