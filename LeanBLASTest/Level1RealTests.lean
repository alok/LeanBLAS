import LeanBLASTest.Level1Real

def main : IO Unit :=
  BLAS.Test.Level1Real.main

-- NOTE: This file is built as an executable via Lake (`lake exe Level1RealTests`).
-- A top-level `#eval main` would run during compilation, before the BLAS FFI
-- library is linked into the Lean process, causing missing-extern errors.
-- Use the Lake executable to run this test suite instead.
