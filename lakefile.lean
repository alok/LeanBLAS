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
  -- On macOS, Mathlib's shared library currently fails to link due to an upstream
  -- duplicate-symbol issue with lld. Disabling precompilation avoids requiring
  -- Mathlib:shared during development.
  precompileModules := if System.Platform.isOSX then false else true
  -- moreLinkObjs := #[libleanblasc]  -- disabled: causes cross-package target resolution issues

-- Test modules live under the `LeanBLASTest` namespace. We keep them in a
-- separate library to avoid collisions with dependency packages that also ship
-- `Test.*` modules (e.g. `plausible`).
lean_lib LeanBLASTest where
  roots := #[`LeanBLASTest]
  precompileModules := if System.Platform.isOSX then false else true

@[test_driver]
lean_exe ComprehensiveTests where
  root := `LeanBLASTest.TestRunner
  moreLinkObjs := #[libleanblasc]

lean_exe Level1RealTests where
  root := `LeanBLASTest.Level1RealTests
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe PackedTriangularTests where
  root := `LeanBLASTest.PackedTriangularTests
  moreLinkObjs := #[libleanblasc]

lean_exe PropertyTests where
  root := `LeanBLASTest.PropertyTests
  moreLinkObjs := #[libleanblasc]

lean_exe EdgeCaseTests where
  root := `LeanBLASTest.EdgeCaseTests
  moreLinkObjs := #[libleanblasc]

lean_exe BenchmarkTests where
  root := `LeanBLASTest.BenchmarkTests
  moreLinkObjs := #[libleanblasc]

lean_exe CorrectnessTests where
  root := `LeanBLASTest.CorrectnessTests
  moreLinkObjs := #[libleanblasc]

lean_exe Level3Tests where
  root := `LeanBLASTest.Level3Tests
  moreLinkObjs := #[libleanblasc]

lean_exe BenchmarksQuickTest where
  root := `LeanBLASTest.BenchmarksQuick
  moreLinkObjs := #[libleanblasc]

-- Small utility executable used internally for measuring timings while
-- developing the benchmark suite.  Not part of the public API.
lean_exe TimeTest where
  root := `TimeTest
  moreLinkObjs := #[libleanblasc]

-- Public showcase executable
lean_exe Gallery where
  root := `Gallery

lean_exe ComplexValidation where
  root := `LeanBLASTest.ComplexValidationTests
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexNumericalReference where
  root := `LeanBLASTest.ComplexNumericalReference
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexEdgeCases where
  root := `LeanBLASTest.ComplexEdgeCasesTests
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexArrayTest where
  root := `LeanBLASTest.ComplexArrayTest
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexCorrectnessLevel2 where
  root := `LeanBLASTest.ComplexCorrectnessLevel2
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexNumericalValidation where
  root := `LeanBLASTest.ComplexNumericalValidationTests
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe Level3Benchmarks where
  root := `LeanBLASTest.BenchmarksLevel3
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexLevel1Comprehensive where
  root := `LeanBLASTest.ComplexLevel1ComprehensiveTests
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexLevel2Comprehensive where
  root := `LeanBLASTest.ComplexLevel2ComprehensiveTests
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe ComplexExamples where
  root := `examples.ComplexExamples
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]

lean_exe Float32Test where
  root := `LeanBLASTest.Float32Test
  supportInterpreter := true
  moreLinkObjs := #[libleanblasc]
