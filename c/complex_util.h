#ifndef LEANBLAS_COMPLEX_UTIL_H
#define LEANBLAS_COMPLEX_UTIL_H

#include <lean/lean.h>

// Helper function to get pointer to complex data from ComplexFloat64Array
static inline double* lean_complex_float64_array_cptr(b_lean_obj_arg arr) {
    // ComplexFloat64Array is a structure with a ByteArray at field 0
    lean_object* byte_array = lean_ctor_get(arr, 0);
    return (double*)lean_sarray_cptr(byte_array);
}

// Helper function to get pointer to complex data from ComplexFloat32Array
static inline float* lean_complex_float32_array_cptr(b_lean_obj_arg arr) {
    // ComplexFloat32Array is a structure with a ByteArray at field 0
    lean_object* byte_array = lean_ctor_get(arr, 0);
    return (float*)lean_sarray_cptr(byte_array);
}

#endif // LEANBLAS_COMPLEX_UTIL_H