# LeanBLAS + SciLean: The Verified ML Stack

## Executive Summary

**MAJOR UPDATE**: Recent commits to SciLean add **tinygrad-style lazy tensor compilation** and **CUDA JIT backend**. The stack is now nearly complete for neural network training.

**New Additions (Dec 2024):**
- `SciLean/Compiler/LazyTensor.lean` (946 lines) - Lazy evaluation, UOp IR, kernel fusion
- `SciLean/Compiler/CUDABackend.lean` (539 lines) - CUDA JIT compilation from Lean

**Remaining work**: Implement the CUDA FFI C layer, add Float32 to LeanBLAS, complete proofs.

---

## Actual Current Architecture (Updated Dec 2024)

```
Layer 6: User Code                              ← Examples, MNIST demo
Layer 5: High-Level Ops (Dense, Conv, etc.)     ← SciLean.Modules.ML (partial)
Layer 4: Autodiff + Tensor                      ← SciLean.AD + DataArrayN (WORKING)
Layer 3: Lazy Compiler                          ← SciLean.Compiler.LazyTensor (NEW!)
Layer 2: Backend Abstraction                    ← TensorBackend (CPU + Metal + CUDA)
Layer 1: Device Primitives                      ← LEANBLAS (COMPLETE)
Layer 0: Hardware                               ← OpenBLAS, Metal, CUDA (FFI stubs)
```

**LeanBLAS (Layer 1):**
- Complete BLAS Levels 1/2/3 for Float64 and ComplexFloat64
- Clean architecture: Spec → CBLAS → FFI → C wrappers → OpenBLAS
- Used by SciLean as the CPU compute backend

**SciLean (Layers 2-6):**
- `DataArrayN`: Shape-safe tensors via dependent types (`Float^[m,n,k]`)
- `TensorBackend`: Abstract interface for CPU/GPU backends
- `HasRevFDeriv`: Reverse-mode autodiff with mathematical specification
- `FunTrans` tactic: Automatic derivative rule composition
- Metal GPU backend: GEMM, GEMV, element-wise ops on Apple Silicon
- ML modules: Dense layers, activations, optimizers (partial)

**NEW - SciLean.Compiler (tinygrad-style):**
- `Sint`: Symbolic integers for dynamic shapes
- `LazyNode`: Lazy computation graph (ADD, MUL, REDUCE, RESHAPE, etc.)
- `UOp`: Micro-operations IR for code generation
- Pattern matching optimization rules (x+0=x, x*1=x, etc.)
- Gradient rules for all ops (reverse-mode AD)
- `CUDABackend`: JIT compilation to PTX via NVRTC

---

## How SciLean Uses LeanBLAS

```lean
-- SciLean/Data/DataArray/Float.lean
import LeanBLAS
open BLAS CBLAS

instance : LevelOneData (DataArray Float) Float Float where
  dot N X offX incX Y offY incY := ddot N.toUSize ...
  scal N a X offX incX := dscal N.toUSize ...

instance : LevelThreeData (DataArray Float) Float Float where
  gemm order transA transB M N K alpha ... := dgemm order ...
```

SciLean wraps LeanBLAS functions with type-safe Lean interfaces, handling `Nat → USize` conversions.

---

## Comparison: Framework Stacks

| Framework | Primitives | IR/Compiler | Autodiff | GPU Kernels |
|-----------|-----------|-------------|----------|-------------|
| **tinygrad** | ~25 ops | UOp graph | Tape-based | Custom codegen |
| **JAX** | XLA primitives | jaxpr → StableHLO | Composable | XLA compiler |
| **PyTorch 2.0** | ATen ops | TorchInductor | Autograd | **Triton** |
| **Flux.jl** | Julia dispatch | Julia AST | Zygote.jl | CUDA.jl |
| **SciLean** | LeanBLAS | FunTrans tactic | HasRevFDeriv | Metal (hand-written) |

**Key Insight**: Modern frameworks increasingly use **Triton** or similar DSL compilers for GPU kernels, rather than hand-writing CUDA.

---

## Triton vs SciLean's New Compiler

[Triton](https://openai.com/index/triton/) is OpenAI's Python DSL for GPU kernels. SciLean now has a similar architecture:

| Feature | Triton | SciLean.Compiler |
|---------|--------|------------------|
| DSL | Python-like | Pure Lean |
| IR | Triton-IR → LLVM | LazyNode → UOp → CUDA C |
| Optimization | Pattern matching | Pattern matching |
| Shapes | Runtime | Symbolic (Sint) + static |
| Gradients | Manual | Built-in rules |
| Verification | None | **Provable** |

**SciLean's Architecture (from LazyTensor.lean):**
```lean
-- LazyNode: computation graph
inductive LazyNode where
  | buffer : Nat → Array Sint → DType → LazyNode
  | unary : UnaryOp → LazyNode → LazyNode
  | binary : BinaryOp → LazyNode → LazyNode → LazyNode
  | reduce : ReduceOp → Array Nat → LazyNode → LazyNode
  | movement : MovementOp → LazyNode → LazyNode

-- UOp: micro-operations for codegen
inductive UOp where
  | load : UOp → UOp
  | store : UOp → UOp → UOp
  | add : UOp → UOp → UOp
  | mulacc : UOp → UOp → UOp → UOp  -- fused multiply-add
  | range : Sint → Sint → AxisType → UOp  -- GPU threads/blocks
  ...
```

**What's Working:**
- Lazy evaluation with Thunk
- Pattern matching optimization (x+0=x, x*1=x, etc.)
- Gradient rules for all basic ops
- Topological sort for reverse-mode AD
- CUDA C code generation from UOp

**What's Missing:**
- CUDA FFI C implementation (`scilean_cuda_*` functions)
- Memory coalescing optimization
- Tensor core support (WMMA)
- Multi-GPU / distributed

---

## Strategic Vision: What Lean Uniquely Offers

### Already Working (SciLean)

1. **Shape-Safe Tensors**: `Float^[m,n,k]` with compile-time dimension checking
2. **Autodiff Specification**: `HasRevFDeriv` typeclass with mathematical proofs
3. **Backend Abstraction**: `TensorBackend` for CPU/GPU dispatch

### The Frontier

1. **Verified Autodiff**: SciLean has the specs, but proofs have `sorry`s
2. **Kernel Compiler**: No automatic GPU codegen from Lean specs
3. **Fusion/Optimization**: FunTrans works, but no kernel-level fusion

---

## Recommended Growth Path (Updated Dec 2024)

The kernel compiler now exists! Priorities shift to completing the FFI and verification:

### Phase 1: Complete CUDA FFI (Immediate Priority)
**The C layer is missing - kernels can't actually run**

SciLean's `CUDABackend.lean` declares these FFI functions:
```lean
@[extern "scilean_cuda_available"] opaque cudaAvailable : Unit → Bool
@[extern "scilean_cuda_init"] opaque cudaInit : Unit → IO Nat
@[extern "scilean_cuda_alloc"] opaque cudaAlloc : USize → IO DevicePtr
@[extern "scilean_cuda_compile"] opaque cudaCompile : @& String → IO CUDAModule
@[extern "scilean_cuda_launch"] opaque cudaLaunch : CUDAModule → ...
```

**Need to implement**: `C/cuda_backend.c` or `C/cuda_backend.cu` with NVRTC

### Phase 2: Add Float32 to LeanBLAS
**GPU efficiency requires single precision**

- [x] BLAS Levels 1/2/3 for Float64
- [x] Complex number support (ComplexFloat64)
- [ ] **Float32 variants** (s* prefix: sdot, sgemm, sscal, etc.)
- [ ] ComplexFloat32 (c* prefix)

This unblocks GPU training since Float32 is ~2x faster on most GPUs.

### Phase 3: Wire LazyTensor → LeanBLAS
**Connect the compiler to the primitives**

Currently `CPUBackend` in LazyTensor.lean is a stub:
```lean
instance : TensorBackend CPUBackend where
  execute _backend _kernel := do
    -- Would dispatch to BLAS operations
    pure ()
```

Wire it to LeanBLAS's GEMM, DOT, etc.

### Phase 4: Verify the Gradient Rules
**Prove autodiff is correct**

LazyTensor.lean has gradient rules like:
```lean
-- Multiplication: d/dx (x * y) = y, d/dy (x * y) = x
{ name := "mul"
  matchNode := fun n => match n with | .binary .mul _ _ => true | _ => false
  gradFn := fun ctx ret => ... }
```

Prove these match mathematical definitions.

### Phase 5: Optimize Codegen
**Match Triton's performance**

- Memory coalescing
- Shared memory usage
- Tensor cores (WMMA ops)
- Kernel fusion heuristics

---

## Actual Ecosystem (Updated Dec 2024)

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Applications                             │
│              (MNIST demo, physics sims, examples)                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 SciLean.Modules.ML                               │
│            Dense, Conv (partial), Activations                    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│           SciLean.AD + SciLean.Data.DataArrayN                   │
│    HasRevFDeriv, FunTrans, Float^[m,n,k] dependent tensors       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│            SciLean.Compiler.LazyTensor (NEW!)                    │
│      Sint, LazyNode, UOp IR, Pattern Matching, Gradient Rules    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  SciLean.TensorBackend                           │
│         CPU (LeanBLAS) | Metal | CUDA (FFI stubs)                │
└─────────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
┌───┴───┐             ┌───────┴───────┐         ┌───────┴───────┐
│LeanBLAS│             │ Metal Shaders │         │  CUDA JIT     │
│ Float64│             │ (hand-written)│         │ (needs FFI C) │
│+Complex│             │ GEMM, GEMV    │         │ via NVRTC     │
└───┬───┘             └───────────────┘         └───────────────┘
    │
┌───┴────┐
│OpenBLAS│
└────────┘
```

---

## Competitive Analysis (Revised)

| Framework | Tensors | Autodiff | GPU Kernels | Verification |
|-----------|---------|----------|-------------|--------------|
| PyTorch 2.0 | Dynamic | Autograd | Triton | None |
| JAX | Static (jit) | Composable | XLA | None |
| tinygrad | Lazy | Tape | Custom codegen | None |
| **SciLean** | Static (types) | HasRevFDeriv | Metal (hand) | **Partial** |

**SciLean's Current Edge:**
- Compile-time shape checking via dependent types
- Mathematical specification of AD (can be proved correct)
- Same language for spec and implementation

**SciLean's Gap:**
- No kernel compiler (vs Triton, XLA)
- Incomplete proofs (sorries)
- Limited GPU coverage (Metal only, hand-written)

---

## The Triton Opportunity

PyTorch 2.0 shows the path: **TorchInductor + Triton** handles most custom kernels.

For Lean/SciLean:

| Approach | Effort | Risk | Reward |
|----------|--------|------|--------|
| **Lean → Triton Python** | Medium | Low | Inherit Triton's optimizations |
| **Lean → MLIR** | High | Medium | Full control, multi-target |
| **Lean → Metal/SPIR-V** | High | High | Pure Lean, maximum verification |

**Recommended**: Start with Lean → Triton for quick wins, then explore MLIR for production.

---

## Resource Estimate (Revised)

| Phase | Effort | Impact | Notes |
|-------|--------|--------|-------|
| LeanBLAS Float32 | 1-2 months | High | Needed for GPU efficiency |
| Complete SciLean proofs | 3-6 months | Medium | Fills sorries |
| Lean → Triton prototype | 3-6 months | **Very High** | Unlocks GPU perf |
| CUDA/cuBLAS backend | 2-3 months | High | Alternative to Triton |
| Full kernel compiler | 12+ months | Transformative | The "LeanTriton" vision |

**To train real models**: ~6-12 months focused work on GPU story.

---

## Key Files Reference

**LeanBLAS:**
- `LeanBLAS/CBLAS/LevelThree.lean` - GEMM interface
- `LeanBLAS/FFI/` - C bindings
- `LeanBLAS/ComplexFloat.lean` - Complex number type

**SciLean:**
- `~/scilean/SciLean/Data/DataArray/` - Tensor implementation
- `~/scilean/SciLean/AD/` - Autodiff framework
- `~/scilean/SciLean/Data/TensorBackend.lean` - Backend abstraction
- `~/scilean/Metal/metal_backend.mm` - GPU kernels

---

## Conclusion (Updated Dec 2024)

**The stack is nearly complete!** Recent work added tinygrad-style compilation:

| Component | Status |
|-----------|--------|
| Shape-safe tensors | ✅ Working |
| Reverse-mode autodiff | ✅ Working |
| CPU backend (LeanBLAS) | ✅ Working |
| Metal GPU | ✅ Working |
| Lazy evaluation | ✅ NEW |
| UOp IR | ✅ NEW |
| CUDA codegen | ✅ NEW |
| CUDA FFI C layer | ❌ **Missing** |
| Float32 BLAS | ❌ **Missing** |
| Proofs complete | ❌ Partial |

**The immediate gap**: CUDA FFI C implementation (`scilean_cuda_*` functions)

**Next steps:**
1. **Implement CUDA FFI** - ~500 lines of C with NVRTC
2. **Add Float32 to LeanBLAS** - Enable efficient GPU training
3. **Wire CPUBackend → LeanBLAS** - Connect LazyTensor to BLAS primitives
4. **Complete proofs** - Verify gradient rules are mathematically correct

**The vision is achievable**: A formally verified deep learning framework where shapes are type-checked, gradients are provably correct, and GPU kernels match their Lean specifications.

**Key new files:**
- `~/scilean/SciLean/Compiler/LazyTensor.lean` (946 lines)
- `~/scilean/SciLean/Compiler/CUDABackend.lean` (539 lines)
