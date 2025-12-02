import LeanBLAS.FFI.FloatArray
import LeanBLAS.ComplexFloat

/-!
# Convenient Constructors and Utilities for Complex Arrays

This module provides ergonomic constructors and utility functions for working
with ComplexFloat64Array, making it easier to create and manipulate complex
arrays in LeanBLAS.
-/

namespace BLAS

/-- Create a ComplexFloat64Array filled with zeros -/
def ComplexFloat64Array.zeros (n : Nat) : ComplexFloat64Array :=
  let arr := ComplexFloatArray.ofArray (Array.replicate n ComplexFloat.zero)
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a ComplexFloat64Array filled with ones -/
def ComplexFloat64Array.ones (n : Nat) : ComplexFloat64Array :=
  let arr := ComplexFloatArray.ofArray (Array.replicate n ComplexFloat.one)
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a ComplexFloat64Array filled with a constant value -/
def ComplexFloat64Array.const (n : Nat) (value : ComplexFloat) : ComplexFloat64Array :=
  let arr := ComplexFloatArray.ofArray (Array.replicate n value)
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a ComplexFloat64Array from a list of complex numbers -/
def ComplexFloat64Array.ofList (xs : List ComplexFloat) : ComplexFloat64Array :=
  let arr := ComplexFloatArray.ofArray xs.toArray
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a ComplexFloat64Array from separate real and imaginary arrays -/
def ComplexFloat64Array.ofRealImag (reals : Array Float) (imags : Array Float) : ComplexFloat64Array :=
  let n := min reals.size imags.size
  let complexArray := Array.range n |>.map fun idx =>
    ComplexFloat.mk (reals[idx]!) (imags[idx]!)
  let arr := ComplexFloatArray.ofArray complexArray
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a ComplexFloat64Array from a real array (imaginary parts are zero) -/
def ComplexFloat64Array.ofReal (reals : Array Float) : ComplexFloat64Array :=
  let complexArray := reals.map fun r => ComplexFloat.mk r 0.0
  let arr := ComplexFloatArray.ofArray complexArray
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a ComplexFloat64Array from an imaginary array (real parts are zero) -/
def ComplexFloat64Array.ofImag (imags : Array Float) : ComplexFloat64Array :=
  let complexArray := imags.map fun i => ComplexFloat.mk 0.0 i
  let arr := ComplexFloatArray.ofArray complexArray
  ComplexFloatArray.toComplexFloat64Array arr

/-- Create a range of complex numbers with given start, stop, and step -/
def ComplexFloat64Array.range (start stop : ComplexFloat) (n : Nat) : ComplexFloat64Array :=
  if n ≤ 1 then
    ComplexFloat64Array.ofList [start]
  else
    let step_r := (stop.x - start.x) / (n - 1).toFloat
    let step_i := (stop.y - start.y) / (n - 1).toFloat
    let arr := Array.range n |>.map fun i =>
      ComplexFloat.mk 
        (start.x + i.toFloat * step_r)
        (start.y + i.toFloat * step_i)
    let complexArr := ComplexFloatArray.ofArray arr
    ComplexFloatArray.toComplexFloat64Array complexArr

/-- Create a complex array with random values (for testing) -/
def ComplexFloat64Array.random (n : Nat) (seed : Nat := 42) : ComplexFloat64Array :=
  -- Simple linear congruential generator for reproducible randomness
  let lcg := fun (seed : Nat) =>
    let seed' := (1103515245 * seed + 12345) % (2^31)
    let r := (seed' % 1000).toFloat / 1000.0 - 0.5  -- Range [-0.5, 0.5]
    let seed'' := (1103515245 * seed' + 12345) % (2^31)
    let i := (seed'' % 1000).toFloat / 1000.0 - 0.5
    (seed'', ComplexFloat.mk r i)
  let (_, values) := (List.range n).foldl (fun (seed, acc) _ =>
    let (newSeed, val) := lcg seed
    (newSeed, val :: acc)
  ) (seed, [])
  let arr := ComplexFloatArray.ofArray values.reverse.toArray
  ComplexFloatArray.toComplexFloat64Array arr

/-- Get the size of a ComplexFloat64Array -/
def ComplexFloat64Array.length (arr : ComplexFloat64Array) : Nat :=
  arr.size

/-- Convert ComplexFloat64Array to a readable string (for debugging) -/
def ComplexFloat64Array.toString (arr : ComplexFloat64Array) (maxElems : Nat := 10) : String :=
  let complexArr := arr.toComplexFloatArray
  let n := min complexArr.size maxElems
  let elems := Array.range n |>.map fun i => 
    s!"{complexArr.get! i}"
  let elemsStr := String.intercalate ", " elems.toList
  if complexArr.size > maxElems then
    s!"[{elemsStr}, ... ({complexArr.size} total)]"
  else
    s!"[{elemsStr}]"

instance : ToString ComplexFloat64Array where
  toString := ComplexFloat64Array.toString

/-- Extract real parts as Float64Array -/
def ComplexFloat64Array.realParts (arr : ComplexFloat64Array) : Float64Array :=
  let complexArr := arr.toComplexFloatArray
  let reals := Array.range complexArr.size |>.map fun i =>
    (complexArr.get! i).x
  FloatArray.mk reals |>.toFloat64Array

/-- Extract imaginary parts as Float64Array -/
def ComplexFloat64Array.imagParts (arr : ComplexFloat64Array) : Float64Array :=
  let complexArr := arr.toComplexFloatArray
  let imags := Array.range complexArr.size |>.map fun i =>
    (complexArr.get! i).y
  FloatArray.mk imags |>.toFloat64Array

/-- Create identity matrix in complex array format (row-major) -/
def ComplexFloat64Array.eye (n : Nat) : ComplexFloat64Array :=
  let arr := Array.replicate (n * n) ComplexFloat.zero
  let arr' := Array.range n |>.foldl (fun acc i =>
    acc.set! (i * n + i) ComplexFloat.one
  ) arr
  let complexArr := ComplexFloatArray.ofArray arr'
  ComplexFloatArray.toComplexFloat64Array complexArr

/-- Create a diagonal matrix from a vector -/
def ComplexFloat64Array.diag (v : ComplexFloat64Array) : ComplexFloat64Array :=
  let vec := v.toComplexFloatArray
  let n := vec.size
  let arr := Array.replicate (n * n) ComplexFloat.zero
  let arr' := Array.range n |>.foldl (fun acc i =>
    acc.set! (i * n + i) (vec.get! i)
  ) arr
  let complexArr := ComplexFloatArray.ofArray arr'
  ComplexFloatArray.toComplexFloat64Array complexArr

end BLAS