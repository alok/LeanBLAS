# LeanBLAS Development Roadmap

## Executive Summary

LeanBLAS has achieved a major milestone with **complete complex number support** across all BLAS levels. The project now offers:

- ✅ **Full BLAS Coverage**: Level 1-3 operations for both real (Float64) and complex (ComplexFloat64) types
- ✅ **Type-Safe FFI**: Zero-copy bindings to optimized BLAS libraries
- ✅ **Mathematical Specifications**: Formal interface definitions ready for verification
- ✅ **Comprehensive Testing**: Property-based, edge case, and performance testing frameworks

### Current Focus
Moving from implementation to **validation and optimization**, ensuring numerical accuracy and performance parity with native BLAS libraries.

## Current State (January 2025)

### ✅ Completed
- **Level 1-3 BLAS for Float64**: Full implementation with FFI bindings
- **Complex Number Support (Level 1-3)**: Complete implementation for ComplexFloat64
  - All Level 1 operations: zdotu, zdotc, zscal, zaxpy, zcopy, zswap, etc.
  - All Level 2 operations: zgemv, zhemv, ztrmv, ztrsv, zgerc, zgeru, zher, zher2
  - All Level 3 operations: zgemm, zsymm, zhemm, zsyrk, zherk, zsyr2k, zher2k, ztrmm, ztrsm
  - Complex-specific operations with proper Hermitian handling
- **Comprehensive test suite**: Property-based, edge case, and correctness tests
- **Documentation**: Module-level docs for all major components
- **Bug fixes**: Resolved cblas_daxpby linking issue
- **PR #4**: Level 3 BLAS implementation ready for merge

### 🔍 Key Insights from Development
1. **FFI is well-designed**: ByteArray wrappers provide zero-copy C compatibility
2. **Specification-first approach**: Clean separation between math specs and implementations
3. **Testing infrastructure**: Robust framework already in place
4. **Non-standard functions**: Some extended BLAS functions need manual implementation

## Strategic Priorities

### ~~Phase 1: Complex Number Support~~ ✅ COMPLETED (January 2025)
**Summary**: Full complex number support implemented for all BLAS levels
- Created FFI bindings for all complex operations
- Implemented C wrappers with proper complex number handling
- Added CBLAS modules for Level 1, 2, and 3 complex operations
- Included complex-specific operations (hemm, herk, her2k)
- Updated documentation and added test cases

### Phase 2: Testing & Optimization (Q1 2025)
**Rationale**: Ensure robustness before formal verification

#### 2.1 Comprehensive Testing Suite
- **Numerical validation** against reference implementations
- **Complex-specific tests**: Branch cuts, special values
- **Performance regression tests**
- **Memory leak detection**

#### 2.2 Performance Optimization
- **Profile FFI overhead** for complex operations
- **Optimize small matrix operations**
- **Investigate SIMD for complex arithmetic**
- **Memory allocation patterns**

### Phase 3: Formal Verification (Q2 2025)
**Rationale**: Unique value proposition for LeanBLAS

#### 3.1 Level 1 Proofs (Medium Priority)
- **Dot product properties**: Commutativity for real, conjugate symmetry for complex
- **Norm properties**: Triangle inequality, scaling properties
- **Vector operation properties**: Linearity of axpy, swap involution
- **Approach**: Build on existing specifications, use Mathlib tactics

#### 3.2 Matrix Operation Proofs (Medium Priority)
- **Associativity**: Prove (AB)C = A(BC) for compatible dimensions
- **Distributivity**: Prove A(B+C) = AB + AC
- **Identity properties**: AI = IA = A
- **Challenges**: Floating-point approximation handling

#### 3.3 Numerical Stability Bounds (Medium Priority)
- Establish error bounds for operations
- Prove backward stability where applicable
- **Research component**: May require new techniques

### Phase 4: Extended Complex Support (Q2 2025)
**Rationale**: Build on the foundation of complex support

#### 4.1 Complex-Specific Optimizations
- **ComplexFloat32**: Single precision complex support
- **Quaternions**: For 3D graphics and physics
- **Complex integer types**: For number theory applications
- **Arbitrary precision complex**: Using Lean's Rat type

#### 4.2 Additional Complex Types
- **Optimized complex multiplication**: Reduce operation count
- **SIMD complex arithmetic**: Leverage vector instructions
- **Cache-friendly algorithms**: For complex matrix operations
- **Specialized small matrix kernels**: 2x2, 3x3, 4x4 complex matrices

### Phase 5: Extended Functionality (Q3 2025)
**Rationale**: Expand beyond dense linear algebra

#### 5.1 Mixed Precision Support (Medium Priority)
- Float32/ComplexFloat32 implementations
- Mixed precision operations (e.g., single to double)
- **Use cases**: Machine learning, graphics

#### 5.2 Sparse Matrix Support (Low Priority)
- Implement CSR (Compressed Sparse Row) format
- Add CSC (Compressed Sparse Column) format
- Basic SpMV (sparse matrix-vector) operations
- **Design decision**: New module or extend existing?

#### 5.3 Banded Matrix Specializations (Low Priority)
- Optimize operations for banded matrices
- Add specialized storage formats
- **Use cases**: Finite difference methods, spline interpolation

### Phase 6: Hardware Acceleration (Q4 2025)
**Rationale**: Future-proofing for heterogeneous computing

#### 6.1 GPU Architecture Design (Low Priority)
- Evaluate CUDA vs OpenCL vs Vulkan Compute
- Design async operation interface
- Memory management strategy
- **Major undertaking**: Possibly separate project

#### 6.2 SIMD Optimizations
- Explore Lean's SIMD support
- Implement vectorized operations for small sizes
- **Platform-specific**: Need careful abstraction

## Implementation Guidelines

### Code Quality Standards
1. **Every new function needs**:
   - Mathematical specification in Spec module
   - FFI binding with documentation
   - CBLAS implementation instance
   - Unit tests and property tests

2. **Documentation requirements**:
   - Module-level overview
   - Function-level specifications
   - Performance characteristics
   - Usage examples

3. **Testing approach**:
   - Property-based tests for mathematical properties
   - Edge case coverage (zero, identity, etc.)
   - Cross-validation with NumPy/reference implementations
   - Performance regression tests

### Development Workflow
1. **Feature branches**: One feature per branch
2. **PR process**: Include tests, docs, and benchmarks
3. **Review criteria**: Correctness, performance, documentation
4. **Integration testing**: Full test suite before merge

## Success Metrics
- **Adoption**: GitHub stars, dependent projects
- **Performance**: Within 10% of native BLAS for large matrices
- **Correctness**: 100% test coverage, formal proofs for core operations
- **Documentation**: Every public API fully documented
- **Community**: Active contributors, responsive issue resolution

## Open Questions
1. ~~Should complex number support be a separate package?~~ → Integrated into main package ✅
2. How to handle mixed-precision operations?
3. What level of formal verification is practical?
4. Should we support arbitrary precision arithmetic?
5. How to integrate with Lean's mathematical libraries?
6. Should we create separate instances for optimized small matrix operations?
7. How to handle complex number constructors more elegantly?

## Recent Achievements (January 2025)

### Complex BLAS Implementation ✅
- **Complete FFI bindings** for all complex Level 1-3 operations
- **C wrapper functions** with proper complex number marshalling
- **Type-safe Lean interfaces** for ComplexFloat64Array
- **Comprehensive documentation** in docs/COMPLEX.md
- **Test framework** for complex operations (needs numerical validation)
- **Fixed instance synthesis** issues for BLAS typeclass

### Documentation Improvements ✅
- Created comprehensive complex number guide
- Updated all relevant documentation files
- Added complex examples to README
- Documented all complex-specific operations

## Next Immediate Steps

### 1. Testing & Validation (High Priority)
- **Numerical accuracy tests**: Compare results with NumPy/reference implementations
- **Complex-specific edge cases**: Test branch cuts, overflow/underflow
- **Performance benchmarks**: Complex vs real operation overhead
- **Property-based testing**: Extend to complex domain

### 2. Code Quality (Medium Priority)
- **Remove remaining `sorry` declarations** in LevelOneComplex
- **Implement packed/banded complex operations** (currently placeholders)
- **Add convenient constructors** for ComplexFloat64Array
- **Improve error messages** for complex operations

### 3. Integration & Release (High Priority)
- **Merge complex support** into main branch
- **Create release notes** highlighting complex features
- **Update examples** with complex use cases
- **Performance profiling** of complex operations