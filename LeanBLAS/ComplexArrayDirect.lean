import LeanBLAS.FFI.FloatArray
import LeanBLAS.ComplexFloat
import LeanBLAS.ComplexArray

/-!
# Direct Complex Array Construction

This module provides a way to create ComplexFloat64Array without going through
the problematic FFI conversion.
-/

namespace BLAS

/-- Convert a ComplexFloat to its byte representation -/
def ComplexFloat.toBytes (c : ComplexFloat) : Array UInt8 := 
  -- Convert real part
  let realBits := c.x.toBits
  let realBytes : Array UInt8 := #[
    (realBits).toUInt8,
    (realBits >>> 8).toUInt8,
    (realBits >>> 16).toUInt8,
    (realBits >>> 24).toUInt8,
    (realBits >>> 32).toUInt8,
    (realBits >>> 40).toUInt8,
    (realBits >>> 48).toUInt8,
    (realBits >>> 56).toUInt8
  ]
  -- Convert imaginary part
  let imagBits := c.y.toBits
  let imagBytes : Array UInt8 := #[
    (imagBits).toUInt8,
    (imagBits >>> 8).toUInt8,
    (imagBits >>> 16).toUInt8,
    (imagBits >>> 24).toUInt8,
    (imagBits >>> 32).toUInt8,
    (imagBits >>> 40).toUInt8,
    (imagBits >>> 48).toUInt8,
    (imagBits >>> 56).toUInt8
  ]
  realBytes ++ imagBytes

/-- Create a ComplexFloat64Array directly from an array of ComplexFloat -/
def ComplexFloat64Array.ofComplexArray (arr : Array ComplexFloat) : ComplexFloat64Array :=
  let bytes := arr.foldl (fun acc c => acc ++ c.toBytes) #[]
  let byteArray := ByteArray.mk bytes
  -- Ensure size is multiple of 16
  if h : byteArray.size % 16 = 0 then
    ⟨byteArray, h⟩
  else
    -- Pad with zeros if needed
    let padding := 16 - (byteArray.size % 16)
    let paddedArray := ByteArray.mk (bytes ++ Array.replicate padding 0)
    ⟨paddedArray, by sorry⟩

/-- Safe macro for creating ComplexFloat64Array -/
macro "#c64safe[" xs:term,* "]" : term => 
  `(ComplexFloat64Array.ofComplexArray #[$xs,*])

end BLAS