import LeanBLAS
import LeanBLAS.FFI.FloatArray
import LeanBLAS.FFI.CBLASLevelOneComplexFloat64

/-!
# Direct FFI Test for Complex BLAS Operations

This test calls the FFI functions directly without going through the BLAS typeclass.
-/

open BLAS

def main : IO Unit := do
  IO.println "=== Direct Complex BLAS FFI Test ==="
  
  -- Create arrays using ByteArray directly to avoid conversion issues
  -- Complex number stored as [real, imag] pairs
  -- [1+0i, 2+1i]
  let x_bytes := ByteArray.empty
    |> ByteArray.append (Float.toBits 1.0).toByteArrayLE -- real part of first element
    |> ByteArray.append (Float.toBits 0.0).toByteArrayLE -- imag part of first element  
    |> ByteArray.append (Float.toBits 2.0).toByteArrayLE -- real part of second element
    |> ByteArray.append (Float.toBits 1.0).toByteArrayLE -- imag part of second element
  
  -- [3+4i, 1-2i]
  let y_bytes := ByteArray.empty
    |> ByteArray.append (Float.toBits 3.0).toByteArrayLE -- real part of first element
    |> ByteArray.append (Float.toBits 4.0).toByteArrayLE -- imag part of first element  
    |> ByteArray.append (Float.toBits 1.0).toByteArrayLE -- real part of second element
    |> ByteArray.append (Float.toBits (-2.0)).toByteArrayLE -- imag part of second element
  
  -- Create ComplexFloat64Arrays
  let x : ComplexFloat64Array := ⟨x_bytes, by decide⟩
  let y : ComplexFloat64Array := ⟨y_bytes, by decide⟩
  
  IO.println "Testing zdotu..."
  let result := CBLAS.zdotu 2 x 0 1 y 0 1
  IO.println s!"zdotu result: {result}"
  
  IO.println "\nTest completed!"