import LeanBLASTest.PackedTriangular

def main : IO Unit :=
  BLAS.Test.PackedTriangular.run

-- NOTE: This file is built as an executable via Lake (`lake exe PackedTriangularTests`).
-- Avoid top-level `#eval` so FFI symbols are only needed at link/runtime.
