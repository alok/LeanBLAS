#include <lean/lean.h>
#include <cblas.h>
#include <string.h>
#include <stdio.h>
#include "util.h"


void ensure_exclusive_float_array(lean_object ** X){
  if (!lean_is_exclusive(*X)) {
    /* printf("warning: making array copy!\n"); */
    *X = lean_copy_float_array(*X);
  }
}

void ensure_exclusive_byte_array(lean_object ** X){
  if (!lean_is_exclusive(*X)) {
    /* printf("warning: making array copy!\n"); */
    *X = lean_copy_byte_array(*X);
  }
}

static void leanblas_invalid_enum(const char* which, const uint8_t tag, const char* expected) {
  char msg[128];
  snprintf(msg, sizeof(msg), "LeanBLAS FFI: invalid %s tag %u (expected %s)", which, (unsigned)tag, expected);
  lean_internal_panic(msg);
}


CBLAS_ORDER leanblas_cblas_order(const uint8_t order) {
  // `BLAS.Order` is defined in Lean as:
  //   | RowMajor
  //   | ColMajor
  // and nullary inductive constructors are encoded with tags 0,1,... in order.
  switch (order) {
    case 0:
      return CblasRowMajor;
    case 1:
      return CblasColMajor;
    default:
      leanblas_invalid_enum("Order", order, "0=RowMajor, 1=ColMajor");
      return CblasRowMajor;  // unreachable
  }
}

CBLAS_TRANSPOSE leanblas_cblas_transpose(const uint8_t trans) {
  switch (trans) {
    // `BLAS.Transpose` is defined in Lean as:
    //   | NoTrans
    //   | Trans
    //   | ConjTrans
    case 0:
      return CblasNoTrans;
    case 1:
      return CblasTrans;
    case 2:
      return CblasConjTrans;
    default:
      leanblas_invalid_enum("Transpose", trans, "0=NoTrans, 1=Trans, 2=ConjTrans");
      return CblasNoTrans;  // unreachable
  }
}

CBLAS_UPLO leanblas_cblas_uplo(const uint8_t uplo) {
  switch (uplo) {
    // `BLAS.UpLo` is defined in Lean as:
    //   | Upper
    //   | Lower
    case 0:
      return CblasUpper;
    case 1:
      return CblasLower;
    default:
      leanblas_invalid_enum("UpLo", uplo, "0=Upper, 1=Lower");
      return CblasUpper;  // unreachable
  }
}

CBLAS_DIAG leanblas_cblas_diag(const uint8_t diag) {
  switch (diag) {
    // `BLAS.Diag` is defined in Lean as:
    //   | NonUnit
    //   | Unit
    case 0:
      return CblasNonUnit;
    case 1:
      return CblasUnit;
    default:
      leanblas_invalid_enum("Diag", diag, "0=NonUnit, 1=Unit");
      return CblasNonUnit;  // unreachable
  }
}

CBLAS_SIDE leanblas_cblas_side(const uint8_t side) {
  switch (side) {
    // `BLAS.Side` is defined in Lean as:
    //   | Left
    //   | Right
    case 0:
      return CblasLeft;
    case 1:
      return CblasRight;
    default:
      leanblas_invalid_enum("Side", side, "0=Left, 1=Right");
      return CblasLeft;  // unreachable
  }
}

LEAN_EXPORT lean_obj_res leanblas_float_array_to_byte_array(lean_obj_arg a){
  lean_obj_res r;
  if (lean_is_exclusive(a)) r = a;
  else r = lean_copy_float_array(a);
  lean_sarray_object * o = lean_to_sarray(r);
  o->m_size *= 8;
  o->m_capacity *= 8;
  lean_set_st_header((lean_object*)o, LeanScalarArray, 1);
  return r;
}

LEAN_EXPORT lean_obj_res leanblas_byte_array_to_float_array(lean_obj_arg a){
  lean_obj_res r;
  if (lean_is_exclusive(a)) r = a;
  else r = lean_copy_byte_array(a);
  lean_sarray_object * o = lean_to_sarray(r);
  o->m_size /= 8;
  o->m_capacity /= 8;
  lean_set_st_header((lean_object*)o, LeanScalarArray, 8);
  return r;
}

LEAN_EXPORT lean_obj_res leanblas_complex_float_array_to_byte_array(lean_obj_arg a){
  // ComplexFloatArray is a structure with a FloatArray field
  // However, Lean may optimize single-field structures and pass the field directly
  
  lean_object* float_array;
  
  // Check if we received the FloatArray directly (optimization for single-field structures)
  if (lean_is_sarray(a)) {
    // We received the FloatArray directly
    float_array = a;
  } else if (lean_is_ctor(a)) {
    // We received the ComplexFloatArray structure
    float_array = lean_ctor_get(a, 0);
  } else {
    lean_internal_panic("leanblas_complex_float_array_to_byte_array: unexpected object type");
  }
  
  if (!float_array) {
    lean_internal_panic("leanblas_complex_float_array_to_byte_array: null float_array");
  }
  
  // Convert FloatArray to ByteArray by reinterpreting the header
  // This is similar to leanblas_float_array_to_byte_array
  lean_obj_res r;
  if (lean_is_exclusive(float_array)) r = float_array;
  else r = lean_copy_float_array(float_array);
  
  lean_sarray_object * o = lean_to_sarray(r);
  o->m_size *= 8;  // FloatArray elements are 8 bytes each
  o->m_capacity *= 8;
  lean_set_st_header((lean_object*)o, LeanScalarArray, 1);
  
  // Debug output
  // printf("DEBUG: Created ComplexFloat64Array\n");
  // printf("  ByteArray size: %zu\n", lean_sarray_size(r));
  // printf("  Expected complex count: %zu\n", lean_sarray_size(r) / 16);
  
  // For single-field structures, Lean may expect just the field
  // So we return the ByteArray directly
  return r;
}

LEAN_EXPORT lean_obj_res leanblas_byte_array_to_complex_float_array(lean_obj_arg a){
  // First extract the ByteArray from ComplexFloat64Array
  lean_object* byte_array;

  if (lean_is_sarray(a)) {
    // We received the ByteArray directly
    byte_array = a;
  } else if (lean_is_ctor(a)) {
    // We received the ComplexFloat64Array structure
    byte_array = lean_ctor_get(a, 0);
  } else {
    lean_internal_panic("leanblas_byte_array_to_complex_float_array: unexpected object type");
  }

  // Convert ByteArray to FloatArray
  lean_obj_res float_array;
  if (lean_is_exclusive(byte_array)) float_array = byte_array;
  else float_array = lean_copy_byte_array(byte_array);

  lean_sarray_object * o = lean_to_sarray(float_array);
  o->m_size /= 8;
  o->m_capacity /= 8;
  lean_set_st_header((lean_object*)o, LeanScalarArray, 8);

  // For single-field structures, Lean may expect just the field
  // So we return the FloatArray directly
  return float_array;
}

// ============================================================================
// Float32Array conversion functions
// ============================================================================

// Create a Float32Array with the given number of elements
LEAN_EXPORT lean_obj_res leanblas_float32_array_mk(size_t n) {
  size_t byte_size = n * 4;  // 4 bytes per float
  lean_obj_res arr = lean_alloc_sarray(1, byte_size, byte_size);
  // Zero-initialize
  memset(lean_sarray_cptr(arr), 0, byte_size);
  return arr;
}

// Create a Float32Array filled with a constant value
LEAN_EXPORT lean_obj_res leanblas_float32_array_const(size_t n, float value) {
  size_t byte_size = n * 4;
  lean_obj_res arr = lean_alloc_sarray(1, byte_size, byte_size);
  float* ptr = (float*)lean_sarray_cptr(arr);
  for (size_t i = 0; i < n; i++) {
    ptr[i] = value;
  }
  return arr;
}

// Get the size of a Float32Array (number of float elements)
LEAN_EXPORT size_t leanblas_float32_array_size(b_lean_obj_arg arr) {
  if (lean_is_sarray(arr)) {
    return lean_sarray_size(arr) / 4;
  } else if (lean_is_ctor(arr)) {
    lean_object* byte_array = lean_ctor_get(arr, 0);
    return lean_sarray_size(byte_array) / 4;
  }
  return 0;
}

// Get an element from Float32Array
LEAN_EXPORT double leanblas_float32_array_get(b_lean_obj_arg arr, size_t idx) {
  float* ptr = lean_float32_array_cptr(arr);
  return (double)ptr[idx];  // Return as double for Lean's Float type
}

// Set an element in Float32Array (returns new array)
LEAN_EXPORT lean_obj_res leanblas_float32_array_set(lean_obj_arg arr, size_t idx, double value) {
  ensure_exclusive_byte_array(&arr);
  float* ptr = lean_float32_array_cptr(arr);
  ptr[idx] = (float)value;
  return arr;
}
