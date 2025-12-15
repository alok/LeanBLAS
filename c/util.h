#include <lean/lean.h>
#include <stdio.h>
#include <cblas.h>
#include <stdint.h>


void ensure_exclusive_float_array(lean_object ** X);
void ensure_exclusive_byte_array(lean_object ** X);

CBLAS_ORDER leanblas_cblas_order(const uint8_t order);
CBLAS_TRANSPOSE leanblas_cblas_transpose(const uint8_t trans);
CBLAS_UPLO leanblas_cblas_uplo(const uint8_t uplo);
CBLAS_DIAG leanblas_cblas_diag(const uint8_t diag);
CBLAS_SIDE leanblas_cblas_side(const uint8_t side);

// Helper to extract ComplexFloat (two inline doubles) parts.
// ComplexFloat is a constructor with two Float fields stored as raw doubles
// immediately after the object header.
static inline void leanblas_complexfloat_parts(b_lean_obj_arg cf, double* re, double* im) {
    const double* fields = (const double*)((const uint8_t*)cf + sizeof(void*));
    *re = fields[0];
    *im = fields[1];
}

// Helper function to get pointer to complex data from ComplexFloat64Array
static inline double* lean_complex_float64_array_cptr(b_lean_obj_arg arr) {
    // ComplexFloat64Array is a structure with a ByteArray at field 0
    // But Lean may optimize single-field structures and pass the ByteArray directly

    if (lean_is_sarray(arr)) {
        // We received the ByteArray directly (single-field structure optimization)
        return (double*)lean_sarray_cptr(arr);
    } else if (lean_is_ctor(arr)) {
        // We received the ComplexFloat64Array structure
        lean_object* byte_array = lean_ctor_get(arr, 0);
        return (double*)lean_sarray_cptr(byte_array);
    } else {
        // This shouldn't happen
        return NULL;
    }
}

// Helper function to get pointer to Float32Array data
// Float32Array is a structure with a ByteArray at field 0
static inline float* lean_float32_array_cptr(b_lean_obj_arg arr) {
    if (lean_is_sarray(arr)) {
        // We received the ByteArray directly (single-field structure optimization)
        return (float*)lean_sarray_cptr(arr);
    } else if (lean_is_ctor(arr)) {
        // We received the Float32Array structure
        lean_object* byte_array = lean_ctor_get(arr, 0);
        return (float*)lean_sarray_cptr(byte_array);
    } else {
        // This shouldn't happen
        return NULL;
    }
}

// Helper function to get pointer to Float64Array data.
// Float64Array is a structure with a ByteArray at field 0 (Prop fields are erased),
// so at runtime it may be passed either as the ByteArray directly or as a ctor.
static inline double* lean_float64_array_cptr(b_lean_obj_arg arr) {
    if (lean_is_sarray(arr)) {
        return (double*)lean_sarray_cptr(arr);
    } else if (lean_is_ctor(arr)) {
        lean_object* byte_array = lean_ctor_get(arr, 0);
        return (double*)lean_sarray_cptr(byte_array);
    } else {
        return NULL;
    }
}
