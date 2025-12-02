import LeanBLAS.ComplexFloat

/-!
# Test Utilities

Shared helper functions for numerical testing of BLAS operations.
-/

namespace BLAS.Test

/-- Default tolerance for floating point comparisons -/
def DEFAULT_TOL : Float := 1e-10

/-- Check if two floats are approximately equal -/
def floatApproxEq (x y : Float) (ε : Float := DEFAULT_TOL) : Bool :=
  Float.abs (x - y) < ε

/-- Check if two complex numbers are approximately equal -/
def complexApproxEq (x y : ComplexFloat) (ε : Float := DEFAULT_TOL) : Bool :=
  Float.abs (x.re - y.re) < ε && Float.abs (x.im - y.im) < ε

/-- Check if a float is NaN -/
def isNaN (x : Float) : Bool := x != x

/-- Check if a complex number has NaN component -/
def complexHasNaN (z : ComplexFloat) : Bool := isNaN z.re || isNaN z.im

/-- Helper to create positive infinity -/
def infinity : Float := 1.0 / 0.0

/-- Check if a complex number has infinity component -/
def complexHasInf (z : ComplexFloat) : Bool :=
  z.re == infinity || z.re == -infinity || z.im == infinity || z.im == -infinity

/-- Assert with message -/
def assertWithMsg (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"✓ {msg}"
  else
    IO.println s!"✗ FAILED: {msg}"
    throw $ IO.userError msg

/-- Run a test and report result -/
def runTest (name : String) (test : IO Bool) : IO Bool := do
  let passed ← test
  if passed then
    IO.println s!"✅ {name} PASSED"
  else
    IO.println s!"❌ {name} FAILED"
  return passed

end BLAS.Test
