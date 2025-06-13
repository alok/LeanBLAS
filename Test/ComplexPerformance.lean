import LeanBLAS

open BLAS BLAS.CBLAS

/-!
# Complex BLAS Performance Benchmarks

This module benchmarks complex BLAS operations against real operations
to measure overhead and identify optimization opportunities.
-/

/-- Measure execution time of an IO action in nanoseconds -/
def timeIO (action : IO α) : IO (α × Nat) := do
  let start ← IO.monoNanosNow
  let result ← action
  let end_time ← IO.monoNanosNow
  return (result, end_time - start)

/-- Convert nanoseconds to milliseconds -/
def nanosToMillis (nanos : Nat) : Float :=
  nanos.toFloat / 1000000.0

/-- Run a benchmark multiple times and report average -/
def benchmark (name : String) (iterations : Nat) (action : IO Unit) : IO Unit := do
  -- Warmup
  for _ in [0:10] do
    action
  
  -- Measure
  let mut totalTime : Nat := 0
  for _ in [0:iterations] do
    let (_, time) ← timeIO action
    totalTime := totalTime + time
  
  let avgTime := nanosToMillis (totalTime / iterations)
  IO.println s!"{name}: {avgTime} ms (avg over {iterations} iterations)"

/-- Benchmark Level 1 operations -/
def benchmark_level1 : IO Unit := do
  IO.println "\n=== Level 1 Performance Benchmarks ==="
  
  let n := 10000
  
  -- Create test vectors
  let realData := Array.replicate n 1.5
  let real_x := FloatArray.mk realData
  let real_y := FloatArray.mk realData
  let real_x64 := real_x.toFloat64Array
  let real_y64 := real_y.toFloat64Array
  
  let complexData := Array.replicate (n * 2) 1.5
  let complex_floats := FloatArray.mk complexData
  let complex_x := ComplexFloatArray.toComplexFloat64Array { data := complex_floats }
  let complex_y := ComplexFloatArray.toComplexFloat64Array { data := complex_floats }
  
  -- Benchmark dot product
  benchmark "Real dot product (ddot)" 100 do
    let _ := ddot n.toUSize real_x64 0 1 real_y64 0 1
    pure ()
  
  benchmark "Complex dot product (zdotc)" 100 do
    let _ := zdotc n.toUSize complex_x 0 1 complex_y 0 1
    pure ()
  
  -- Benchmark norm
  benchmark "Real norm (dnrm2)" 100 do
    let _ := dnrm2 n.toUSize real_x64 0 1
    pure ()
  
  benchmark "Complex norm (dznrm2)" 100 do
    let _ := dznrm2 n.toUSize complex_x 0 1
    pure ()
  
  -- Benchmark axpy
  benchmark "Real axpy (daxpy)" 100 do
    let _ := daxpy n.toUSize 2.0 real_x64 0 1 real_y64 0 1
    pure ()
  
  benchmark "Complex axpy (zaxpy)" 100 do
    let _ := zaxpy n.toUSize ⟨2.0, 0.0⟩ complex_x 0 1 complex_y 0 1
    pure ()
  
  -- Benchmark scaling
  benchmark "Real scaling (dscal)" 100 do
    let _ := dscal n.toUSize 2.0 real_x64 0 1
    pure ()
  
  benchmark "Complex scaling (zscal)" 100 do
    let _ := zscal n.toUSize ⟨2.0, 1.0⟩ complex_x 0 1
    pure ()

/-- Benchmark Level 2 operations -/
def benchmark_level2 : IO Unit := do
  IO.println "\n=== Level 2 Performance Benchmarks ==="
  
  let m := 500
  let n := 500
  
  -- Create test matrices and vectors
  let realMatData := Array.replicate (m * n) 1.5
  let realMat := FloatArray.mk realMatData
  let real_A := realMat.toFloat64Array
  
  let realVecData := Array.replicate n 1.5
  let realVec := FloatArray.mk realVecData
  let real_x := realVec.toFloat64Array
  
  let realVecData2 := Array.replicate m 0.0
  let realVec2 := FloatArray.mk realVecData2
  let real_y := realVec2.toFloat64Array
  
  let complexMatData := Array.replicate (m * n * 2) 1.5
  let complexMat := FloatArray.mk complexMatData
  let complex_A := ComplexFloatArray.toComplexFloat64Array { data := complexMat }
  
  let complexVecData := Array.replicate (n * 2) 1.5
  let complexVec := FloatArray.mk complexVecData
  let complex_x := ComplexFloatArray.toComplexFloat64Array { data := complexVec }
  
  let complexVecData2 := Array.replicate (m * 2) 0.0
  let complexVec2 := FloatArray.mk complexVecData2
  let complex_y := ComplexFloatArray.toComplexFloat64Array { data := complexVec2 }
  
  -- Benchmark matrix-vector multiply
  benchmark "Real gemv (dgemv)" 20 do
    let _ := dgemv Order.RowMajor Transpose.NoTrans m.toUSize n.toUSize 
                   1.0 real_A 0 n.toUSize real_x 0 1 0.0 real_y 0 1
    pure ()
  
  benchmark "Complex gemv (zgemv)" 20 do
    let _ := zgemv Order.RowMajor Transpose.NoTrans m.toUSize n.toUSize 
                   ComplexFloat.one complex_A 0 n.toUSize complex_x 0 1 
                   ComplexFloat.zero complex_y 0 1
    pure ()

/-- Benchmark Level 3 operations -/
def benchmark_level3 : IO Unit := do
  IO.println "\n=== Level 3 Performance Benchmarks ==="
  
  let n := 200  -- Smaller size for Level 3 due to O(n³) complexity
  
  -- Create test matrices
  let realMatData := Array.replicate (n * n) 1.5
  let realMat := FloatArray.mk realMatData
  let real_A := realMat.toFloat64Array
  let real_B := realMat.toFloat64Array
  
  let realMatData2 := Array.replicate (n * n) 0.0
  let realMat2 := FloatArray.mk realMatData2
  let real_C := realMat2.toFloat64Array
  
  let complexMatData := Array.replicate (n * n * 2) 1.5
  let complexMat := FloatArray.mk complexMatData
  let complex_A := ComplexFloatArray.toComplexFloat64Array { data := complexMat }
  let complex_B := ComplexFloatArray.toComplexFloat64Array { data := complexMat }
  
  let complexMatData2 := Array.replicate (n * n * 2) 0.0
  let complexMat2 := FloatArray.mk complexMatData2
  let complex_C := ComplexFloatArray.toComplexFloat64Array { data := complexMat2 }
  
  -- Benchmark matrix-matrix multiply
  benchmark "Real gemm (dgemm)" 5 do
    let _ := dgemm Order.RowMajor Transpose.NoTrans Transpose.NoTrans
                   n.toUSize n.toUSize n.toUSize
                   1.0 real_A 0 n.toUSize real_B 0 n.toUSize
                   0.0 real_C 0 n.toUSize
    pure ()
  
  benchmark "Complex gemm (zgemm)" 5 do
    let _ := zgemm Order.RowMajor Transpose.NoTrans Transpose.NoTrans
                   n.toUSize n.toUSize n.toUSize
                   ComplexFloat.one complex_A 0 n.toUSize complex_B 0 n.toUSize
                   ComplexFloat.zero complex_C 0 n.toUSize
    pure ()

/-- Benchmark extended operations -/
def benchmark_extended : IO Unit := do
  IO.println "\n=== Extended Operations Performance ==="
  
  let n := 10000
  
  -- Create test vector
  let complexData := Array.replicate (n * 2) 1.5
  let complex_floats := FloatArray.mk complexData
  let complex_x := ComplexFloatArray.toComplexFloat64Array { data := complex_floats }
  
  -- Benchmark element-wise operations
  benchmark "Complex sqrt" 50 do
    let _ := LevelOneDataExt.sqrt (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) n complex_x 0 1
    pure ()
  
  benchmark "Complex exp" 50 do
    let _ := LevelOneDataExt.exp (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) n complex_x 0 1
    pure ()
  
  benchmark "Complex log" 50 do
    let _ := LevelOneDataExt.log (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) n complex_x 0 1
    pure ()
  
  benchmark "Complex sin" 50 do
    let _ := LevelOneDataExt.sin (Array := ComplexFloat64Array) (R := Float) (K := ComplexFloat) n complex_x 0 1
    pure ()

/-- Analyze overhead ratios -/
def analyze_overhead : IO Unit := do
  IO.println "\n=== Performance Analysis ==="
  IO.println "Expected overhead for complex vs real operations:"
  IO.println "- Dot product: ~4x (4 real muls + 2 adds vs 1 mul)"
  IO.println "- Norm: ~2x (2 real muls + 1 add + sqrt vs 1 mul + sqrt)"
  IO.println "- AXPY: ~4x (4 real muls + 4 adds vs 1 mul + 1 add)"
  IO.println "- GEMM: ~4x (complex mul is 4 real muls + 2 adds)"
  IO.println ""
  IO.println "Note: Actual overhead may be less due to:"
  IO.println "- SIMD vectorization of complex operations"
  IO.println "- Cache effects from contiguous data access"
  IO.println "- Optimized BLAS implementations"

/-- Main benchmark runner -/
def main : IO Unit := do
  IO.println "=== Complex BLAS Performance Benchmarks ==="
  IO.println "Measuring complex vs real operation overhead"
  
  benchmark_level1
  benchmark_level2
  benchmark_level3
  benchmark_extended
  analyze_overhead
  
  IO.println "\n✅ Performance benchmarks completed!"
  IO.println "Use these results to identify optimization opportunities."