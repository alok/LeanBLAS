import Lake

open Lake DSL System Lean Elab

def linkArgs := -- (#[] : Array String)
  if System.Platform.isWindows then
    #[]
  else if System.Platform.isOSX then
    #["-L/opt/homebrew/opt/openblas/lib", "-lblas"]
  else -- assuming linux
    #["-L/usr/lib/x86_64-linux-gnu/", "-lblas"]
def inclArgs :=
  if System.Platform.isWindows then
    #[]
  else if System.Platform.isOSX then
    #["-I/opt/homebrew/opt/openblas/include"]
  else -- assuming linux
    #[]

package leanblas {
  moreLinkArgs := linkArgs
  preferReleaseBuild := true
}

-- Use latest mathlib
require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "master"

----------------------------------------------------------------------------------------------------
-- Build Lean ↔ BLAS bindings ---------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
-- Build the C FFI library (for internal use and executables)
target libleanblasc pkg : FilePath := do
  pkg.afterBuildCacheAsync do
    let mut oFiles : Array (Job FilePath) := #[]
    for file in (← (pkg.dir / "c").readDir) do
      if file.path.extension == some "c" then
        let oFile := pkg.buildDir / "c" / (file.fileName.stripSuffix ".c" ++ ".o")
        let srcJob ← inputTextFile file.path
        let weakArgs := #["-I", (← getLeanIncludeDir).toString]
        oFiles := oFiles.push (← buildO oFile srcJob weakArgs (#["-DNDEBUG", "-O3", "-fPIC"] ++ inclArgs) "gcc" getLeanTrace)
    let name := nameToStaticLib "leanblasc"
    buildStaticLib (pkg.sharedLibDir / name) oFiles

----------------------------------------------------------------------------------------------------

-- Note: moreLinkObjs removed - dependents must link libleanblasc explicitly
-- This is required for cross-package local path dependencies
@[default_target]
lean_lib LeanBLAS where
  roots := #[`LeanBLAS]

@[test_driver]
lean_exe ComprehensiveTests where
  root := `Test.TestRunner

lean_exe CBLASLevelOneTest where
  root := `Test.cblas_level_one
  supportInterpreter := true

lean_exe TriangularTest where
  root := `Test.packed_triangular

lean_exe PropertyTests where
  root := `Test.Property

lean_exe EdgeCaseTests where
  root := `Test.EdgeCases

lean_exe BenchmarkTests where
  root := `Test.Benchmarks

lean_exe CorrectnessTests where
  root := `Test.Correctness

lean_exe Level3Tests where
  root := `Test.Level3

lean_exe BenchmarksFixedTest where
  root := `Test.BenchmarksFixed

-- Small utility executable used internally for measuring timings while
-- developing the benchmark suite.  Not part of the public API.
lean_exe TimeTest where
  root := `TimeTest

-- Public showcase executable
lean_exe Gallery where
  root := `Gallery

lean_exe ComplexValidation where
  root := `Test.ComplexValidation
  supportInterpreter := true

lean_exe ComplexNumericalReference where
  root := `Test.ComplexNumericalReference
  supportInterpreter := true

lean_exe ComplexEdgeCases where
  root := `Test.ComplexEdgeCases
  supportInterpreter := true

lean_exe ComplexPerformance where
  root := `Test.ComplexPerformance
  supportInterpreter := true

lean_exe ComplexArrayTest where
  root := `Test.ComplexArrayTest
  supportInterpreter := true

lean_exe ComplexCorrectnessLevel2 where
  root := `Test.ComplexCorrectnessLevel2
  supportInterpreter := true

lean_exe ComplexNumericalValidation where
  root := `Test.ComplexNumericalValidation
  supportInterpreter := true

lean_exe Level3Benchmarks where
  root := `Test.BenchmarksLevel3

lean_exe ComplexLevel1Comprehensive where
  root := `Test.ComplexLevel1Comprehensive
  supportInterpreter := true

lean_exe ComplexLevel2Comprehensive where
  root := `Test.ComplexLevel2Comprehensive
  supportInterpreter := true

lean_exe ComplexExamples where
  root := `examples.ComplexExamples
  supportInterpreter := true