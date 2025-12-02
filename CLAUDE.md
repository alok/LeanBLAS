# LeanBLAS Development Guidelines

## Overview
LeanBLAS is a Lean 4 binding to BLAS (Basic Linear Algebra Subprograms) for high-performance numerical computing.

## Build Commands
- Build library: `lake build LeanBLAS`
- Build FFI: `lake build libleanblasc`
- Run tests: `lake build ComprehensiveTests && .lake/build/bin/ComprehensiveTests`

## Local Dependency Notes

### Cross-Package FFI Linking
When LeanBLAS is used as a local path dependency (e.g., in SciLean), Lake has issues resolving `moreLinkObjs` targets across package boundaries. The solution:

1. **In LeanBLAS**: The `lean_lib LeanBLAS` does NOT include `moreLinkObjs := #[libleanblasc]`
2. **In dependent packages**: Executables must explicitly link the FFI library:
   ```lean
   lean_exe MyExecutable where
     root := `MyExecutable
     moreLinkArgs := #["-L" ++ (leanblasPath / ".lake/build/lib").toString, "-lleanblasc"]
   ```

This workaround is necessary because Lake's target type resolution fails with "type mismatch in target 'libleanblasc': expected 'filepath', got unknown" when accessing targets across local path dependencies.

### Prerequisites
- OpenBLAS or system BLAS
- On macOS: `brew install openblas`
- On Linux: `apt install libopenblas-dev`

## Code Style
- FFI functions prefixed with `d` for double precision (e.g., `ddot`, `dgemm`)
- Complex variants use `z` prefix (e.g., `zdot`, `zgemm`)
- Level 1/2/3 BLAS organized in separate modules

## Testing
Tests use property-based testing and numerical validation against known results.
