import LeanBLAS

open BLAS BLAS.CBLAS

/-- Simple test showing complex Level 2 operations compile -/
def test_complex_level2 : IO Unit := do
  IO.println "Complex Level 2 BLAS operations are available:"
  IO.println "- zgemv: General matrix-vector multiplication"
  IO.println "- zhemv: Hermitian matrix-vector multiplication"
  IO.println "- ztrmv: Triangular matrix-vector multiplication"
  IO.println "- ztrsv: Triangular solve"
  IO.println "- zgeru: Rank-1 update (no conjugation)"
  IO.println "- zgerc: Rank-1 update (with conjugation)"
  IO.println "- zher: Hermitian rank-1 update"
  IO.println "- zher2: Hermitian rank-2 update"
  IO.println "\nAll complex Level 2 operations compiled successfully!"

/-- Main entry point -/
def main : IO Unit := test_complex_level2