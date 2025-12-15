# LeanBLAS

BLAS (Basic Linear Algebra Subprograms) bindings for Lean 4 with mathematical formalization and comprehensive testing.

## Overview

LeanBLAS provides type-safe BLAS operations with a focus on mathematical correctness. Unlike traditional BLAS libraries, LeanBLAS:

- Formalizes the mathematics of linear algebra operations
- Provides property-based testing that goes beyond typical numerical validation
- Includes formal mathematical specifications alongside efficient implementations
- Features the most comprehensive BLAS testing framework available

## Features

- **Complete BLAS Coverage**: Level 1, 2, and 3 operations with type-safe interfaces
- **Full Complex Number Support**: 
  - ComplexFloat64 arrays with efficient interleaved storage
  - All standard complex BLAS operations (z-prefixed)
  - Complex-specific operations (hemv, hemm, herk, her2k)
  - Hermitian and symmetric matrix operations
  - Conjugate transpose support
- **FFI Bindings**: Efficient integration with OpenBLAS/system BLAS
- **Mathematical Formalization**: Specifications using Lean's type system
- **World-Class Testing**:
  - Property-based testing with automatic edge case discovery
  - Formal correctness proofs
  - Performance benchmarking with GFLOPS measurement
  - Numerical stability verification
- **Platform Support**: macOS and Linux (Windows not supported)

## Installation

### Prerequisites

**Ubuntu/Debian:**

```bash
sudo apt-get install libopenblas-dev
```

**macOS:**

```bash
brew install openblas
```

Windows is not supported.

### Build

```bash
git clone https://github.com/lecopivo/LeanBLAS
cd LeanBLAS
lake build
```

## Project Setup

### Using lakefile.lean

```lean
import Lake
open Lake DSL System

def linkArgs :=
  if System.Platform.isWindows then
    panic! "Windows is not supported!"
  else if System.Platform.isOSX then
    #["-L/opt/homebrew/opt/openblas/lib", "-L/usr/local/opt/openblas/lib", "-lblas"]
  else -- assuming Linux
    #["-L/usr/lib/x86_64-linux-gnu/", "-lblas", "-lm"]

require leanblas from git "https://github.com/lecopivo/LeanBLAS" @ "main"
```

### Using lakefile.toml

```toml
[[require]]
name = "leanblas"
git = "https://github.com/lecopivo/LeanBLAS"
rev = "main"

[moreLinkArgs]
linux = ["-L/usr/lib/x86_64-linux-gnu/", "-lblas", "-lm"]
macos = ["-L/opt/homebrew/opt/openblas/lib", "-L/usr/local/opt/openblas/lib", "-lblas"]
```

## Quick Start

### Creating Arrays

LeanBLAS uses `Float64Array` for real numbers and `ComplexFloat64Array` for complex numbers:

```lean
import LeanBLAS

-- Create real arrays using the #f64[...] syntax
def x := #f64[1.0, 2.0, 3.0, 4.0]
def y := #f64[5.0, 6.0, 7.0, 8.0]

-- Create complex arrays using the #c64[...] syntax
def cx := #c64[⟨1.0, 2.0⟩, ⟨3.0, -1.0⟩]  -- [1+2i, 3-i]
def cy := #c64[⟨2.0, 0.0⟩, ⟨0.0, 1.0⟩]   -- [2+0i, 0+i]
```

### Level 1 Examples (Vector Operations)

```lean
import LeanBLAS
import LeanBLAS.CBLAS.LevelOne

-- Dot product
def test_dot : IO Unit := do
  let x := #f64[1.0, 2.0, 3.0]
  let y := #f64[4.0, 5.0, 6.0]
  let result := BLAS.CBLAS.ddot 3 x 0 1 y 0 1
  IO.println s!"Dot product: {result}"  -- Expected: 32.0

-- Euclidean norm
def test_norm : IO Unit := do
  let x := #f64[3.0, 4.0]
  let norm := BLAS.CBLAS.dnrm2 2 x 0 1
  IO.println s!"Norm: {norm}"  -- Expected: 5.0

-- Scale vector
def test_scale : IO Unit := do
  let x := #f64[1.0, 2.0, 3.0]
  let scaled := BLAS.CBLAS.dscal 3 2.0 x 0 1
  IO.println s!"Scaled: {scaled.toFloatArray}"  -- Expected: [2.0, 4.0, 6.0]
```

### Level 2 Examples (Matrix-Vector Operations)

```lean
import LeanBLAS
import LeanBLAS.CBLAS.LevelTwo

-- Matrix-vector multiplication (GEMV)
def test_gemv : IO Unit := do
  -- A = [1 2; 3 4] (2x2 matrix in row-major order)
  let A := #f64[1.0, 2.0, 3.0, 4.0]
  let x := #f64[5.0, 6.0]
  let y := #f64[0.0, 0.0]
  
  -- y = 1.0 * A * x + 0.0 * y
  let result := BLAS.CBLAS.dgemv 
    BLAS.Order.RowMajor BLAS.Transpose.NoTrans
    2 2 1.0 A 0 2 x 0 1 0.0 y 0 1
  
  IO.println s!"Result: {result.toFloatArray}"  -- Expected: [17.0, 39.0]
```

### Level 3 Examples (Matrix-Matrix Operations)

```lean
import LeanBLAS
import LeanBLAS.CBLAS.LevelThree

-- Matrix multiplication (GEMM)
def test_gemm : IO Unit := do
  -- A = [1 2; 3 4], B = [5 6; 7 8] (2x2 matrices)
  let A := #f64[1.0, 2.0, 3.0, 4.0]
  let B := #f64[5.0, 6.0, 7.0, 8.0]
  let C := #f64[0.0, 0.0, 0.0, 0.0]
  
  -- C = 1.0 * A * B + 0.0 * C
  let result := BLAS.CBLAS.dgemm 
    BLAS.Order.RowMajor 
    BLAS.Transpose.NoTrans BLAS.Transpose.NoTrans
    2 2 2 1.0 A 0 2 B 0 2 0.0 C 0 2
  
  IO.println s!"Result: {result.toFloatArray}"  -- Expected: [19.0, 22.0, 43.0, 50.0]
```

### Complex Number Examples

```lean
import LeanBLAS
import LeanBLAS.CBLAS.LevelOneComplex

-- Complex dot product
def test_complex_dot : IO Unit := do
  let x := #c64[⟨1.0, 2.0⟩, ⟨3.0, -1.0⟩]  -- [1+2i, 3-i]
  let y := #c64[⟨2.0, 0.0⟩, ⟨1.0, 1.0⟩]   -- [2+0i, 1+i]
  
  -- Conjugate dot product: conj(x) · y
  let dot_c := BLAS.CBLAS.dot 2 x 0 1 y 0 1
  IO.println s!"Conjugate dot: {dot_c}"  -- (1-2i)*(2) + (3+i)*(1+i) = 2-4i + 2+4i = 4
  
  -- 2-norm of complex vector
  let norm := BLAS.CBLAS.nrm2 2 x 0 1
  IO.println s!"2-norm: {norm}"  -- sqrt(|1+2i|² + |3-i|²) = sqrt(5 + 10) = sqrt(15)

-- Complex matrix operations
import LeanBLAS.CBLAS.LevelThreeComplex

def test_complex_gemm : IO Unit := do
  -- A = [1+i, 2; 0+i, 3-i], B = [1, i; 2i, 1]
  let A := #c64[⟨1.0, 1.0⟩, ⟨2.0, 0.0⟩, ⟨0.0, 1.0⟩, ⟨3.0, -1.0⟩]
  let B := #c64[⟨1.0, 0.0⟩, ⟨0.0, 1.0⟩, ⟨0.0, 2.0⟩, ⟨1.0, 0.0⟩]
  let C := #c64[⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩, ⟨0.0, 0.0⟩]
  
  -- C = A * B
  let result := BLAS.CBLAS.gemm Order.RowMajor 
    Transpose.NoTrans Transpose.NoTrans
    2 2 2 ComplexFloat.one A 0 2 B 0 2 ComplexFloat.zero C 0 2
  
  IO.println "Complex matrix multiplication completed"
```

## Testing Framework

LeanBLAS features the most comprehensive BLAS testing suite available:

### Quick Test

```bash
lake exe ComprehensiveTests  # Runs the quick (essential) suite by default
```

### Comprehensive Testing

```bash
lake exe ComprehensiveTests  # Run all tests with unified reporting
lake exe PropertyTests       # Property-based testing with random inputs
lake exe EdgeCaseTests       # Boundary conditions and numerical edge cases
lake exe CorrectnessTests    # Mathematical correctness verification
lake exe Level3Tests         # Level 3 BLAS operations testing
```

### Performance Analysis

```bash
lake exe BenchmarkTests      # Full performance analysis with scaling
lake exe BenchmarksQuickTest # Quick performance sanity check
lake exe Level3Benchmarks    # Matrix multiplication benchmarks
lake exe Gallery             # Showcase of all benchmarks
```

### Additional Testing Tools

- Python validation scripts: `test_level3.py`, `cross_check_numpy.py`
- Local CI script: `run_ci_local.sh`
- Level 3 test runner: `run_level3_tests.sh`

## Available Operations

### Level 1 (Vector-Vector)

- `dot`, `ddot`, `sdot` - Dot products
- `nrm2` - Euclidean norm
- `asum` - Sum of absolute values
- `axpy` - y := a*x + y
- `copy` - Copy vector
- `scal` - Scale vector
- `swap` - Swap vectors
- `rotg`, `rot` - Givens rotations

### Level 2 (Matrix-Vector)

- `gemv` - General matrix-vector multiplication
- `symv` - Symmetric matrix-vector multiplication
- `trmv` - Triangular matrix-vector multiplication
- `ger` - Rank-1 update
- `syr` - Symmetric rank-1 update
- `trsv` - Triangular solve

### Level 3 (Matrix-Matrix)

- `gemm` - General matrix-matrix multiplication
- `symm` - Symmetric matrix-matrix multiplication
- `trmm` - Triangular matrix-matrix multiplication
- `syrk` - Symmetric rank-k update
- `syr2k` - Symmetric rank-2k update
- `trsm` - Triangular solve with multiple right-hand sides

### Complex Number Operations

LeanBLAS provides full support for complex arithmetic:

#### Complex Level 1

- `zdotu`, `zdotc` - Complex dot products (unconjugated/conjugated)
- `dznrm2` - Complex vector 2-norm (returns real)
- `zscal` - Scale by complex scalar
- `zaxpy` - Complex y := a*x + y
- `zcopy`, `zswap` - Complex vector operations

#### Complex Level 2

- `zgemv` - General complex matrix-vector multiplication
- `zhemv` - Hermitian matrix-vector multiplication
- `ztrmv`, `ztrsv` - Triangular operations
- `zgerc`, `zgeru` - Rank-1 updates (conjugated/unconjugated)
- `zher`, `zher2` - Hermitian rank updates

#### Complex Level 3

- `zgemm` - General complex matrix multiplication
- `zhemm` - Hermitian matrix multiplication
- `ztrmm`, `ztrsm` - Triangular matrix operations
- `zherk`, `zher2k` - Hermitian rank-k updates
- `zsyrk`, `zsyr2k` - Symmetric rank-k updates

## Mathematical Formalization

LeanBLAS goes beyond traditional BLAS implementations by providing:

### Type-Safe Specifications

```lean
class LevelOneData (Array : Type*) (R K : Type*) where
  dot (N : Nat) (X : Array) (offX incX : Nat) (Y : Array) (offY incY : Nat) : K
  nrm2 (N : Nat) (X : Array) (offX incX : Nat) : R
  -- ... more operations
```

### Formal Properties

```lean
class LawfulBLAS (Array : Type*) (R K : Type*) [RCLike R] [RCLike K] [BLAS Array R K] : Prop
```

The `LawfulBLAS` class ensures operations satisfy mathematical laws like:

- Dot product commutativity: `dot(x,y) = dot(y,x)`
- Cauchy-Schwarz inequality: `|dot(x,y)| ≤ ||x|| * ||y||`
- Triangle inequality: `||x+y|| ≤ ||x|| + ||y||`

### API Design Philosophy

LeanBLAS uses offset and stride parameters for flexibility:

- `off`: Starting index in the array
- `inc`: Stride between elements (can be negative)

This allows working with:

- Subvectors without copying
- Non-contiguous data layouts
- Reverse iteration (negative stride)

## Project Structure

```text
LeanBLAS/
├── LeanBLAS/          # Main library code
│   ├── BLAS.lean      # Core typeclasses
│   ├── Spec/          # Mathematical specifications
│   ├── CBLAS/         # FFI implementations
│   └── FFI/           # Low-level bindings
├── LeanBLASTest/      # Comprehensive test suite
├── c/                 # C wrapper code
└── *.py               # Python validation scripts
```

## Documentation

- **In-Code Documentation**: Mathematical specifications in source files
- **`DOCUMENTATION.md`**: Documentation standards and guidelines
- **`docs/COMPLEX.md`**: Comprehensive guide to complex number support
- **`STATUS.md`**: Detailed implementation status and test results
- **`AGENT.md`**: Development workflow and build commands

## Current Status

- ✅ Complete Level 1, 2, and 3 BLAS specifications
- ✅ FFI bindings to system BLAS
- ✅ Comprehensive testing framework
- ✅ Mathematical formalization
- ✅ Full complex number support (Level 1, 2, and 3)
- ⚠️ Some proofs use `sorry` (work in progress)

## Contributing

LeanBLAS welcomes contributions! Key areas:

- Completing mathematical proofs
- Adding more numerical stability tests
- Performance optimizations
- Extended precision support

## License

Apache 2.0
