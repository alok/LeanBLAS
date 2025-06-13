# Complex FFI Workaround

## Issue

The `#c64[...]` macro currently causes a segfault due to an FFI issue with the `ComplexFloatArray.toComplexFloat64Array` function. The C function expects a Lean constructor but receives something else, causing:

```
INTERNAL PANIC: leanblas_complex_float_array_to_byte_array: not a constructor
```

## Workaround

Until this is fixed, create `ComplexFloat64Array` directly using ByteArray literals:

```lean
-- Instead of this (causes segfault):
let x := #c64[⟨1.0, 2.0⟩, ⟨3.0, 4.0⟩]

-- Use this:
let x := ComplexFloat64Array.mk (ByteArray.mk #[
  -- First complex number: 1.0 + 2.0i
  0, 0, 0, 0, 0, 0, 240, 63,  -- 1.0 (real part)
  0, 0, 0, 0, 0, 0, 0, 64,    -- 2.0 (imaginary part)
  -- Second complex number: 3.0 + 4.0i  
  0, 0, 0, 0, 0, 0, 8, 64,    -- 3.0 (real part)
  0, 0, 0, 0, 0, 0, 16, 64    -- 4.0 (imaginary part)
]) (by decide)
```

## Byte Representation Reference

Common float values in little-endian byte representation:

- 0.0: `0, 0, 0, 0, 0, 0, 0, 0`
- 1.0: `0, 0, 0, 0, 0, 0, 240, 63`
- 2.0: `0, 0, 0, 0, 0, 0, 0, 64`
- 3.0: `0, 0, 0, 0, 0, 0, 8, 64`
- 4.0: `0, 0, 0, 0, 0, 0, 16, 64`
- -1.0: `0, 0, 0, 0, 0, 0, 240, 191`
- -2.0: `0, 0, 0, 0, 0, 0, 0, 192`

## Alternative Approach

For testing, you can also use the high-level BLAS operations which work correctly:

```lean
import LeanBLAS
import LeanBLAS.CBLAS.LevelOneComplex

def test : IO Unit := do
  -- Create arrays using direct ByteArray (workaround)
  let x := ComplexFloat64Array.mk (ByteArray.mk #[...]) (by decide)
  let y := ComplexFloat64Array.mk (ByteArray.mk #[...]) (by decide)
  
  -- Use BLAS operations normally
  let dot_result := CBLAS.zdotc 2 x 0 1 y 0 1
  IO.println s!"Dot product: {dot_result}"
```

## Root Cause

The issue appears to be that Lean 4's FFI is not properly boxing the ComplexFloatArray structure when passing it to the extern C function. The C code expects a constructor object but receives something else.

## TODO

1. Investigate proper FFI handling of Lean structures
2. Update the C code to handle the actual object format
3. Or update the Lean declaration to properly box the structure
4. Add regression tests once fixed