import LeanBLAS
import LeanBLAS.CBLAS.LevelOneComplex
import LeanBLAS.FFI.FloatArray

open BLAS CBLAS

def main : IO Unit := do
  -- Create complex arrays directly
  let x_data := ComplexFloatArray.ofArray #[ComplexFloat.mk 1.0 0.0, ComplexFloat.mk 0.0 1.0]
  let x := x_data.toComplexFloat64Array
  
  -- Simple norm test
  let norm := dznrm2 2 x 0 1
  IO.println s!"Norm: {norm}"
  
  -- Simple dot product test  
  let y_data := ComplexFloatArray.ofArray #[ComplexFloat.mk 1.0 0.0, ComplexFloat.mk 0.0 (-1.0)]
  let y := y_data.toComplexFloat64Array
  
  let dot_result := zdotc 2 x 0 1 y 0 1
  IO.println s!"Dot product: {dot_result}"