import LeanBLAS
import LeanBLAS.CBLAS.LevelTwoComplex
import LeanBLAS.CBLAS.LevelThreeComplex

/-!
# Complex BLAS Examples

This module demonstrates how to use complex number support in LeanBLAS
through practical examples.
-/

open BLAS CBLAS
open LevelOneData LevelTwoData LevelThreeData

namespace ComplexBLASExamples

/-- Example 1: Basic complex vector operations -/
def example_complex_vectors : IO Unit := do
  IO.println "=== Example 1: Complex Vector Operations ==="
  
  -- Create complex vectors using the #c64 notation
  let x := #c64[⟨1.0, 2.0⟩, ⟨3.0, -1.0⟩, ⟨0.0, 4.0⟩]
  let y := #c64[⟨2.0, 0.0⟩, ⟨1.0, 1.0⟩, ⟨-1.0, 2.0⟩]
  
  -- Compute dot product (conjugate)
  let dot_result := dot 3 x 0 1 y 0 1
  IO.println s!"Conjugate dot product: {dot_result}"
  
  -- Compute 2-norm
  let norm := nrm2 3 x 0 1
  IO.println s!"2-norm of x: {norm}"
  
  -- Scale vector by complex scalar
  let alpha : ComplexFloat := ⟨2.0, -1.0⟩
  let x_scaled := scal 3 alpha x 0 1
  IO.println s!"Scaled vector: x * (2-i)"
  
  -- Compute y = alpha*x + y
  let y_new := axpy 3 alpha x 0 1 y 0 1
  IO.println s!"After axpy: y = (2-i)*x + y"

/-- Example 2: Hermitian matrix operations -/
def example_hermitian_matrix : IO Unit := do
  IO.println "\n=== Example 2: Hermitian Matrix Operations ==="
  
  -- Create a 3x3 Hermitian matrix (stored as upper triangular)
  -- H = [2+0i,   1+i,   3-2i]
  --     [1-i,    4+0i,  0+i ]
  --     [3+2i,   0-i,   5+0i]
  let H_data := #c64[⟨2.0, 0.0⟩, ⟨1.0, 1.0⟩, ⟨3.0, -2.0⟩,
                     ⟨0.0, 0.0⟩, ⟨4.0, 0.0⟩, ⟨0.0, 1.0⟩,
                     ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨5.0, 0.0⟩]
  
  -- Create a vector
  let v := #c64[⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩, ⟨1.0, -1.0⟩]
  
  -- Result vector
  let result := #c64[⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩]
  
  -- Compute y = H * v using Hermitian matrix-vector multiply
  let y := hemv Order.RowMajor UpLo.Upper 3 
                ComplexFloat.one H_data 0 3 
                v 0 1 
                ComplexFloat.zero result 0 1
  
  IO.println "Hermitian matrix-vector product computed"
  
  -- Hermitian rank-1 update: H = H + alpha * v * v^H
  let alpha : ComplexFloat := ⟨0.5, 0.0⟩  -- Must use real alpha for her
  let H_updated := her Order.RowMajor UpLo.Upper 3 alpha v 0 1 H_data 0 3
  
  IO.println "Hermitian rank-1 update performed"

/-- Example 3: Complex matrix multiplication -/
def example_complex_gemm : IO Unit := do
  IO.println "\n=== Example 3: Complex Matrix Multiplication ==="
  
  -- Create two 2x2 complex matrices
  let A := #c64[⟨1.0, 1.0⟩, ⟨2.0, 0.0⟩,
                ⟨0.0, 1.0⟩, ⟨3.0, -1.0⟩]
  
  let B := #c64[⟨2.0, 0.0⟩, ⟨1.0, 2.0⟩,
                ⟨1.0, -1.0⟩, ⟨0.0, 1.0⟩]
  
  -- Result matrix
  let C := #c64[⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩,
                ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩]
  
  -- Compute C = A * B
  let C_result := gemm Order.RowMajor Transpose.NoTrans Transpose.NoTrans
                       2 2 2
                       ComplexFloat.one A 0 2
                       B 0 2
                       ComplexFloat.zero C 0 2
  
  IO.println "Complex matrix multiplication: C = A * B"
  
  -- Compute C = A^H * B (conjugate transpose)
  let C_conj := gemm Order.RowMajor Transpose.ConjTrans Transpose.NoTrans
                     2 2 2
                     ComplexFloat.one A 0 2
                     B 0 2
                     ComplexFloat.zero C 0 2
  
  IO.println "With conjugate transpose: C = A^H * B"

/-- Example 4: Solving triangular systems -/
def example_triangular_solve : IO Unit := do
  IO.println "\n=== Example 4: Triangular System Solve ==="
  
  -- Upper triangular matrix
  let U := #c64[⟨2.0, 0.0⟩, ⟨1.0, 1.0⟩, ⟨3.0, -1.0⟩,
                ⟨0.0, 0.0⟩, ⟨3.0, 0.0⟩, ⟨2.0, 2.0⟩,
                ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨4.0, 0.0⟩]
  
  -- Right-hand side
  let b := #c64[⟨5.0, 3.0⟩, ⟨6.0, -2.0⟩, ⟨8.0, 0.0⟩]
  
  -- Solve U * x = b
  let x := trsv Order.RowMajor UpLo.Upper Transpose.NoTrans false
                3 U 0 3 b 0 1
  
  IO.println "Solved upper triangular system U * x = b"
  
  -- For verification, compute U * x
  let check := trmv Order.RowMajor UpLo.Upper Transpose.NoTrans false
                    3 U 0 3 x 0 1
  
  IO.println "Verification: computed U * x"

/-- Example 5: Working with complex numbers in scientific computing -/
def example_fft_like : IO Unit := do
  IO.println "\n=== Example 5: FFT-like Computation ==="
  
  -- Create twiddle factors for a simple DFT
  let omega := (Float.acos (-1.0)) * 2.0 / 4.0  -- For 4-point DFT
  let twiddles := #c64[⟨1.0, 0.0⟩,
                       ⟨Float.cos omega, -Float.sin omega⟩,
                       ⟨Float.cos (2*omega), -Float.sin (2*omega)⟩,
                       ⟨Float.cos (3*omega), -Float.sin (3*omega)⟩]
  
  -- Input signal
  let signal := #c64[⟨1.0, 0.0⟩, ⟨2.0, 0.0⟩, ⟨3.0, 0.0⟩, ⟨4.0, 0.0⟩]
  
  IO.println "Created twiddle factors and signal for DFT-like computation"
  
  -- This demonstrates how complex BLAS can be used in signal processing
  -- Real DFT would use matrix multiplication with Vandermonde matrix

/-- Main function to run all examples -/
def main : IO Unit := do
  IO.println "=== LeanBLAS Complex Number Examples ==="
  IO.println "Demonstrating complex BLAS operations\n"
  
  example_complex_vectors
  example_hermitian_matrix
  example_complex_gemm
  example_triangular_solve
  example_fft_like
  
  IO.println "\n=== All examples completed successfully ==="

end ComplexBLASExamples

-- Run the examples
def main : IO Unit := ComplexBLASExamples.main