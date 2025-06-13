import LeanBLAS
import LeanBLAS.ComplexArray

open BLAS

def main : IO Unit := do
  -- Create a simple ComplexFloatArray
  let arr := ComplexFloatArray.ofArray #[⟨1.0, 2.0⟩]
  IO.println s!"Created ComplexFloatArray of size {arr.size}"
  
  -- Try to convert to ComplexFloat64Array
  try
    let c64arr := ComplexFloatArray.toComplexFloat64Array arr
    IO.println "Successfully converted to ComplexFloat64Array"
  catch e =>
    IO.println s!"Error: {e}"