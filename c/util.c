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


CBLAS_ORDER leanblas_cblas_order(const uint8_t order) {
  if (order == 0) {
    return CblasRowMajor;
  } else {
    return CblasColMajor;
  }
}

CBLAS_TRANSPOSE leanblas_cblas_transpose(const uint8_t trans) {
  switch (trans) {
    case 0:
      return CblasNoTrans;
    case 1:
      return CblasTrans;
    case 2:
      return CblasConjTrans;
    default:
      return CblasNoTrans;
  }
}

CBLAS_UPLO leanblas_cblas_uplo(const uint8_t uplo) {
  switch (uplo) {
    case 0:
      return CblasUpper;
    case 1:
      return CblasLower;
    default:
      return CblasUpper;
  }
}

CBLAS_DIAG leanblas_cblas_diag(const uint8_t diag) {
  switch (diag) {
    case 0:
      return CblasNonUnit;
    case 1:
      return CblasUnit;
    default:
      return CblasNonUnit;
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
  printf("DEBUG: Created ComplexFloat64Array\n");
  printf("  ByteArray size: %zu\n", lean_sarray_size(r));
  printf("  Expected complex count: %zu\n", lean_sarray_size(r) / 16);
  
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




