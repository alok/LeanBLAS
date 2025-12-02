import LeanBLAS.FFI.CBLASLevelOneComplexFloat64
import LeanBLAS.Spec.LevelOne

/-!
# CBLAS Level 1 Complex Implementation

This module provides the CBLAS implementation of Level 1 BLAS operations
for ComplexFloat64Array types. These are vector-vector operations on complex numbers.

## Complex Number Operations

Complex BLAS operations often have multiple variants:
- Standard operations (e.g., `zdotu`): No conjugation
- Conjugate operations (e.g., `zdotc`): Conjugates first vector
- Mixed precision (e.g., `dznrm2`): Complex input, real output

## Implementation Notes

The FFI bindings handle the interleaved storage format used by BLAS:
- Complex numbers stored as [real, imaginary] pairs
- Offset calculations must account for 2× factor
-/

namespace BLAS.CBLAS

open Sorry

set_option linter.unusedVariables false

/-- CBLAS implementation of Level 1 BLAS operations for ComplexFloat64Array.

This instance provides efficient complex vector operations through FFI calls
to optimized BLAS libraries. Complex conjugation is handled appropriately
for operations like dot products. -/
instance : LevelOneData ComplexFloat64Array Float ComplexFloat where
  size x := x.size
  get x i := 
    -- Extract ComplexFloat from ByteArray at position i
    -- Each complex number is 16 bytes (8 for real, 8 for imaginary)
    if h : i < x.size then
      -- Convert ByteArray to ComplexFloatArray to extract values
      let complexArray := x.toComplexFloatArray
      -- Access using ComplexFloatArray's get! method
      complexArray.get! i
    else
      ComplexFloat.zero
  
  -- Use conjugate dot product (zdotc) as the default dot product for complex numbers
  dot N X offX incX Y offY incY := zdotc N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  
  -- Euclidean norm returns real value
  nrm2 N X offX incX := dznrm2 N.toUSize X offX.toUSize incX.toUSize
  
  -- Sum of absolute values returns real value  
  asum N X offX incX := dzasum N.toUSize X offX.toUSize incX.toUSize
  
  -- Index of maximum absolute value
  iamax N X offX incX := izamax N.toUSize X offX.toUSize incX.toUSize |>.toNat
  
  -- Vector operations
  swap N X offX incX Y offY incY := zswap N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  copy N X offX incX Y offY incY := zcopy N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  axpy N a X offX incX Y offY incY := zaxpy N.toUSize a X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize
  
  -- Givens rotations for complex numbers - not implemented
  -- CBLAS provides zrotg but the interface differs from real rotg
  rotg _ _ := panic! "Complex Givens rotation (zrotg) not yet implemented"
  rotmg _ _ _ _ := panic! "Complex modified Givens rotation not supported by CBLAS"
  rot _ _ _ _ _ _ _ _ _ := panic! "Complex plane rotation (zrot/zdrot) not yet implemented"
  
  -- Scaling operations
  scal N a X offX incX := zscal N.toUSize a X offX.toUSize incX.toUSize

-- Additional complex-specific operations exposed at the Nat interface level

/-- Unconjugated dot product for complex vectors: Σᵢ Xᵢ * Yᵢ (without conjugation).
    Use conjugate dot product (zdotc, the default) for most applications.
    This is a Nat-based wrapper around the FFI function. -/
def unconjugatedDot (N : Nat) (X : ComplexFloat64Array) (offX incX : Nat) (Y : ComplexFloat64Array) (offY incY : Nat) : ComplexFloat :=
  CBLAS.zdotu N.toUSize X offX.toUSize incX.toUSize Y offY.toUSize incY.toUSize

/-- Scale a complex vector by a real scalar: X := a*X where a ∈ ℝ.
    More efficient than zscal when the scalar is real.
    This is a Nat-based wrapper around the FFI function. -/
def scaleByReal (N : Nat) (a : Float) (X : ComplexFloat64Array) (offX incX : Nat) : ComplexFloat64Array :=
  CBLAS.zdscal N.toUSize a X offX.toUSize incX.toUSize

instance : LevelOneDataExt ComplexFloat64Array Float ComplexFloat where
  const N a :=
    -- Create a constant array by converting from ComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let floatArr := FloatArray.mk (floatData.mapIdx fun i _ =>
      if i % 2 = 0 then a.re else a.im)
    let complexFloatArr : ComplexFloatArray := { data := floatArr }
    ComplexFloatArray.toComplexFloat64Array complexFloatArr
    
  sum N X offX incX := 
    -- Sum all elements by iterating through the array
    let arr := X.toComplexFloatArray
    let rec sumHelper (i : Nat) (acc : ComplexFloat) : ComplexFloat :=
      if i >= N then acc
      else 
        let idx := offX + i * incX
        if idx < arr.size then
          sumHelper (i + 1) (acc + arr.get! idx)
        else acc
    sumHelper 0 ComplexFloat.zero
    
  axpby N a X offX incX b Y offY incY := 
    -- Y := a*X + b*Y
    -- First scale Y by b
    let Y' := zscal N.toUSize b Y offY.toUSize incY.toUSize
    -- Then add a*X
    zaxpy N.toUSize a X offX.toUSize incX.toUSize Y' offY.toUSize incY.toUSize
    
  scaladd N a X offX incX b := 
    -- Return a*X + b (where b is broadcast to all elements)
    -- Create an array for the result - using same approach as const
    let floatData := Array.replicate (N * 2) 0.0
    let floatArr := FloatArray.mk floatData
    let Y := ComplexFloatArray.toComplexFloat64Array { data := floatArr }
    -- Copy X to Y  
    let Y' := zcopy N.toUSize X offX.toUSize incX.toUSize Y 0 1
    -- Scale by a
    let Y'' := zscal N.toUSize a Y' 0 1
    -- Add constant b
    let floatDataB := Array.replicate (N * 2) 0.0
    let floatArrB := FloatArray.mk (floatDataB.mapIdx fun i _ =>
      if i % 2 = 0 then b.re else b.im)
    let bVec := ComplexFloatArray.toComplexFloat64Array { data := floatArrB }
    zaxpy N.toUSize ComplexFloat.one bVec 0 1 Y'' 0 1
    
  imaxRe N X offX incX _ :=
    -- Find index with maximum real part
    let arr := X.toComplexFloatArray
    let rec findMax (i : Nat) (maxIdx : Nat) (maxVal : Float) : Nat :=
      if i >= N then maxIdx
      else
        let idx := offX + i * incX
        if idx < arr.size then
          let val := (arr.get! idx).re
          if val > maxVal then findMax (i + 1) i val
          else findMax (i + 1) maxIdx maxVal
        else maxIdx
    findMax 0 0 (-(1.0 / 0.0))

  imaxIm N X offX incX _ :=
    -- Find index with maximum imaginary part
    let arr := X.toComplexFloatArray
    let rec findMaxIm (i : Nat) (maxIdx : Nat) (maxVal : Float) : Nat :=
      if i >= N then maxIdx
      else
        let idx := offX + i * incX
        if idx < arr.size then
          let val := (arr.get! idx).im
          if val > maxVal then findMaxIm (i + 1) i val
          else findMaxIm (i + 1) maxIdx maxVal
        else maxIdx
    findMaxIm 0 0 (-(1.0 / 0.0))

  iminRe N X offX incX _ :=
    -- Find index with minimum real part
    let arr := X.toComplexFloatArray
    let rec findMin (i : Nat) (minIdx : Nat) (minVal : Float) : Nat :=
      if i >= N then minIdx
      else
        let idx := offX + i * incX
        if idx < arr.size then
          let val := (arr.get! idx).re
          if val < minVal then findMin (i + 1) i val
          else findMin (i + 1) minIdx minVal
        else minIdx
    findMin 0 0 (1.0 / 0.0)

  iminIm N X offX incX _ :=
    -- Find index with minimum imaginary part
    let arr := X.toComplexFloatArray
    let rec findMinIm (i : Nat) (minIdx : Nat) (minVal : Float) : Nat :=
      if i >= N then minIdx
      else
        let idx := offX + i * incX
        if idx < arr.size then
          let val := (arr.get! idx).im
          if val < minVal then findMinIm (i + 1) i val
          else findMinIm (i + 1) minIdx minVal
        else minIdx
    findMinIm 0 0 (1.0 / 0.0)
    
  mul N X offX incX Y offY incY :=
    -- Element-wise multiplication: Z[i] = X[i] * Y[i]
    let xArr := X.toComplexFloatArray
    let yArr := Y.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        let yi := offY + i * incY
        if xi < xArr.size && yi < yArr.size then
          let x := xArr.get! xi
          let y := yArr.get! yi
          let prod := x * y
          if isReal then prod.re else prod.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  div N X offX incX Y offY incY :=
    -- Element-wise division: Z[i] = X[i] / Y[i]
    let xArr := X.toComplexFloatArray
    let yArr := Y.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        let yi := offY + i * incY
        if xi < xArr.size && yi < yArr.size then
          let x := xArr.get! xi
          let y := yArr.get! yi
          let quot := x / y
          if isReal then quot.re else quot.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }
    
  inv N X offX incX :=
    -- Element-wise reciprocal: Z[i] = 1 / X[i]
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          let inv := ComplexFloat.one / x
          if isReal then inv.re else inv.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  abs N X offX incX :=
    -- Element-wise absolute value: Z[i] = |X[i]| (returns real array in complex format)
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          if isReal then ComplexFloat.abs x else 0.0  -- Store magnitude in real part
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  sqrt N X offX incX :=
    -- Element-wise square root: Z[i] = sqrt(X[i])
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          let sqrtVal := ComplexFloat.sqrt x
          if isReal then sqrtVal.re else sqrtVal.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  exp N X offX incX :=
    -- Element-wise exponential: Z[i] = exp(X[i])
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          let expVal := ComplexFloat.exp x
          if isReal then expVal.re else expVal.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  log N X offX incX :=
    -- Element-wise logarithm: Z[i] = log(X[i])
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          let logVal := ComplexFloat.log x
          if isReal then logVal.re else logVal.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  sin N X offX incX :=
    -- Element-wise sine: Z[i] = sin(X[i])
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          let sinVal := ComplexFloat.sin x
          if isReal then sinVal.re else sinVal.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

  cos N X offX incX :=
    -- Element-wise cosine: Z[i] = cos(X[i])
    let xArr := X.toComplexFloatArray
    let floatData := Array.replicate (N * 2) 0.0
    let result := FloatArray.mk (floatData.mapIdx fun idx _ =>
      let i := idx / 2
      let isReal := idx % 2 = 0
      if i < N then
        let xi := offX + i * incX
        if xi < xArr.size then
          let x := xArr.get! xi
          let cosVal := ComplexFloat.cos x
          if isReal then cosVal.re else cosVal.im
        else 0.0
      else 0.0)
    ComplexFloatArray.toComplexFloat64Array { data := result }

end BLAS.CBLAS