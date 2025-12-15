#include <lean/lean.h>
#include <cblas.h>
#include <math.h>
#include "util.h"



/** ddot
 *
 * Computes the dot product of two vectors.
 *
 * @param N Number of elements in input vectors
 * @param X Pointer to first input vector
 * @param offX starting index of X
 * @param incX Increment for the elements of X
 * @param Y Pointer to second input vector
 * @param offY starting index of Y
 * @param incY Increment for the elements of Y
 *
 * @return Dot product of X and Y
 */
LEAN_EXPORT double leanblas_cblas_ddot(const size_t N,
                                 const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                 const b_lean_obj_arg Y, const size_t offY, const size_t incY){
  return cblas_ddot((int)N, lean_float64_array_cptr(X) + offX, (int)incX, lean_float64_array_cptr(Y) + offY, (int)incY);
}



/** zdot
 *
  * Computes the dot product of two complex vectors.
  *
  * @param N Number of elements in input vectors
  * @param X Pointer to first input vector
  * @param offX starting index of X
  * @param incX Increment for the elements of X
  * @param Y Pointer to second input vector
  * @param offY starting index of Y
  * @param incY Increment for the elements of Y
  *
  * @return Dot product of X and Y
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_zdot(const size_t N,
                                       const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                       const b_lean_obj_arg Y, const size_t offY, const size_t incY){

  double r[2];
  cblas_zdotc_sub((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX,
                          (void *)(lean_complex_float64_array_cptr(Y) + 2*offY), (int)incY, r);

  lean_obj_res lean_res = lean_alloc_ctor(0, 0, 2*sizeof(double));
  lean_ctor_set_float(lean_res, 0*sizeof(double), r[0]);
  lean_ctor_set_float(lean_res, 1*sizeof(double), r[1]);
  return lean_res;
}

// Rename existing zdot to zdotc for consistency
LEAN_EXPORT lean_obj_res leanblas_cblas_zdotc(const size_t N,
                                      const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                      const b_lean_obj_arg Y, const size_t offY, const size_t incY){
  double r[2];
  cblas_zdotc_sub((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX,
                          (void *)(lean_complex_float64_array_cptr(Y) + 2*offY), (int)incY, r);

  lean_obj_res lean_res = lean_alloc_ctor(0, 0, 2*sizeof(double));
  lean_ctor_set_float(lean_res, 0*sizeof(double), r[0]);
  lean_ctor_set_float(lean_res, 1*sizeof(double), r[1]);
  return lean_res;
}

LEAN_EXPORT lean_obj_res leanblas_cblas_zdotu(const size_t N,
                                      const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                      const b_lean_obj_arg Y, const size_t offY, const size_t incY){
  double r[2];
  cblas_zdotu_sub((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX,
                          (void *)(lean_complex_float64_array_cptr(Y) + 2*offY), (int)incY, r);

  lean_obj_res lean_res = lean_alloc_ctor(0, 0, 2*sizeof(double));
  lean_ctor_set_float(lean_res, 0*sizeof(double), r[0]);
  lean_ctor_set_float(lean_res, 1*sizeof(double), r[1]);
  return lean_res;
}

LEAN_EXPORT double leanblas_cblas_dznrm2(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_dznrm2((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX);
}

LEAN_EXPORT double leanblas_cblas_dzasum(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_dzasum((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX);
}

LEAN_EXPORT size_t leanblas_cblas_izamax(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_izamax((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX);
}

LEAN_EXPORT lean_obj_res leanblas_cblas_zswap(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                              lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&X);
  ensure_exclusive_byte_array(&Y);
  cblas_zswap((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX,
                      (void *)(lean_complex_float64_array_cptr(Y) + 2*offY), (int)incY);
  lean_obj_res result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, X);
  lean_ctor_set(result, 1, Y);
  return result;
}

LEAN_EXPORT lean_obj_res leanblas_cblas_zcopy(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                                              lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&Y);
  cblas_zcopy((int)N, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX,
                      (void *)(lean_complex_float64_array_cptr(Y) + 2*offY), (int)incY);
  return Y;
}

LEAN_EXPORT lean_obj_res leanblas_cblas_zaxpy(const size_t N, const b_lean_obj_arg alpha, 
                                              const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                              lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&Y);
  double alpha_arr[2];
  leanblas_complexfloat_parts(alpha, &alpha_arr[0], &alpha_arr[1]);
  
  cblas_zaxpy((int)N, alpha_arr, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX,
                                 (void *)(lean_complex_float64_array_cptr(Y) + 2*offY), (int)incY);
  return Y;
}

LEAN_EXPORT lean_obj_res leanblas_cblas_zscal(const size_t N, const b_lean_obj_arg alpha, 
                                              lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double alpha_arr[2];
  leanblas_complexfloat_parts(alpha, &alpha_arr[0], &alpha_arr[1]);
  
  cblas_zscal((int)N, alpha_arr, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX);
  return X;
}

LEAN_EXPORT lean_obj_res leanblas_cblas_zdscal(const size_t N, const double alpha, 
                                               lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  cblas_zdscal((int)N, alpha, (void *)(lean_complex_float64_array_cptr(X) + 2*offX), (int)incX);
  return X;
}




/** dnrm2
 *
 * Computes the Euclidean norm of a vector.
 *
 * @param N Number of elements in input vector
 * @param X Pointer to input vector
 * @param offX starting index of X
 * @param incX Increment for the elements of X
 *
 * @return Euclidean norm of X
 */
LEAN_EXPORT double leanblas_cblas_dnrm2(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_dnrm2((int)N, lean_float64_array_cptr(X) + offX, (int)incX);
}


/** dasum
 *
 * Computes the sum of the absolute values of the elements of a vector.
 *
 * @param N Number of elements in input vector
 * @param X Pointer to input vector
 * @param offX starting index of X
 * @param incX Increment for the elements of X
 *
 * @return Sum of the absolute values of the elements of X
 */
LEAN_EXPORT double leanblas_cblas_dasum(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_dasum((int)N, lean_float64_array_cptr(X) + offX, (int)incX);
}

/** idamax
 *
 * Finds the index of the first element with maximum absolute value.
 *
 * @param N Number of elements in input vector
 * @param X Pointer to input vector
 * @param offX starting index of X
 * @param incX Increment for the elements of X
 *
 * @return Index of the first element with maximum absolute value
 */
LEAN_EXPORT size_t leanblas_cblas_idamax(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_idamax((int)N, lean_float64_array_cptr(X) + offX, (int)incX);
}


/** dswap
  *
  * Interchanges two vectors.
  *
  * @param N Number of elements in input vectors
  * @param X Pointer to first input vector
  * @param offX starting index of X
  * @param incX Increment for the elements of X
  * @param Y Pointer to second input vector
  * @param offY starting index of Y
  * @param incY Increment for the elements of Y

  * @return X and Y with their elements interchanged
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_dswap(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&X);
  ensure_exclusive_byte_array(&Y);
  cblas_dswap((int)N, lean_float64_array_cptr(X) + offX, (int)incX, lean_float64_array_cptr(Y) + offY, (int)incY);

  lean_obj_res res = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(res, 0, X);
  lean_ctor_set(res, 1, Y);
  return res;
}


/** dcopy
  *
  * Copies a vector, X, to a vector, Y.
  *
  * @param N Number of elements in input vectors
  * @param X Pointer to input vector
  * @param offX starting index of X
  * @param incX Increment for the elements of X
  * @param Y Pointer to output vector
  * @param offY starting index of Y
  * @param incY Increment for the elements of Y
  *
  * @return Y with the elements of X copied to it
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_dcopy(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&Y);
  cblas_dcopy((int)N, lean_float64_array_cptr(X) + offX, (int)incX, lean_float64_array_cptr(Y) + offY, (int)incY);
  return Y;
}

/** daxpy
  *
  * Computes a vector, Y, plus a scalar multiple of a vector, X.
  *
  * @param N Number of elements in input vectors
  * @param alpha Scalar multiplier
  * @param X Pointer to input vector
  * @param offX starting index of X
  * @param incX Increment for the elements of X
  * @param Y Pointer to output vector
  * @param offY starting index of Y
  * @param incY Increment for the elements of Y
  *
  * @return Y with the elements of alpha*X added to it
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_daxpy(const size_t N, const double alpha, const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&Y);
  cblas_daxpy((int)N, alpha, lean_float64_array_cptr(X) + offX, (int)incX, lean_float64_array_cptr(Y) + offY, (int)incY);
  return Y;
}



/** drotg
  *
  * Constructs a Givens plane rotation.
  *
  * @param a First input scalar
  * @param b Second input scalar
  * @param c Cosine of the angle of rotation
  * @param s Sine of the angle of rotation
  *
  * @return a, b, c, and s with the Givens plane rotation constructed
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_drotg(double a, double b){
  double c, s;
  cblas_drotg(&a, &b, &c, &s);

  lean_obj_res res = lean_alloc_ctor(0, 4, 0);
  lean_ctor_set(res, 0, lean_box(a));
  lean_ctor_set(res, 1, lean_box(b));
  lean_ctor_set(res, 2, lean_box(c));
  lean_ctor_set(res, 3, lean_box(s));
  return res;
}


/** drotmg
  *
  * Constructs a modified Givens plane rotation.
  *
  * @param d1 First input scalar
  * @param d2 Second input scalar
  * @param x1 First input vector
  * @param y1 Second input vector
  * @param param Pointer to output vector
  *
  * @return d1, d2, x1, y1, and param with the modified Givens plane rotation constructed
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_drotmg(const double d1, const double d2, const double x1, const double y1){
  double d1_out, d2_out, x1_out, y1_out;
  double param[5];
  cblas_drotmg(&d1_out, &d2_out, &x1_out, y1_out, param);

  printf("fix implementation of drotmg\n");

  lean_obj_res res = lean_alloc_ctor(0, 5, 0);
  lean_ctor_set(res, 0, lean_box(d1_out));
  lean_ctor_set(res, 1, lean_box(d2_out));
  lean_ctor_set(res, 2, lean_box(x1_out));
  lean_ctor_set(res, 3, lean_box(y1_out));
  lean_ctor_set(res, 4, lean_box(0));
  return res;
}


/** drot
  *
  * Applies a Givens plane rotation to a pair of vectors.
  *
  * @param N Number of elements in input vectors
  * @param X Pointer to first input vector
  * @param offX starting index of X
  * @param incX Increment for the elements of X
  * @param Y Pointer to second input vector
  * @param offY starting index of Y
  * @param incY Increment for the elements of Y
  * @param c Cosine of the angle of rotation
  * @param s Sine of the angle of rotation
  *
  * @return X and Y with the Givens plane rotation applied
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_drot(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                               lean_obj_arg Y, const size_t offY, const size_t incY, const double c, const double s){
  ensure_exclusive_byte_array(&X);
  ensure_exclusive_byte_array(&Y);
  cblas_drot((int)N, lean_float64_array_cptr(X) + offX, (int)incX, lean_float64_array_cptr(Y) + offY, (int)incY, c, s);
 
  lean_obj_res res = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(res, 0, X);
  lean_ctor_set(res, 1, Y);
  return res;
}
 

/** dscal
  *
  * Scales a vector by a constant.
  *
  * @param N Number of elements in input vector
  * @param alpha Scalar multiplier
  * @param X Pointer to input vector
  * @param offX starting index of X
  * @param incX Increment for the elements of X
  *
  * @return X with the elements scaled by alpha
  */
LEAN_EXPORT lean_obj_res leanblas_cblas_dscal(const size_t N, const double alpha, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  cblas_dscal((int)N, alpha, lean_float64_array_cptr(X) + offX, (int)incX);
  return X;
}




/* class LevelOneDataExt (R K : outParam Type) (Array : Type) [Scalar R K] where */
/*   const (N : Nat) (a : K) : Array */
/*   sum (N : Nat) (X : Array) (offX incX : Nat) : K */
/*   axpby (N : Nat) (α β : K) (X : Array) (offX incX : Nat) (Y : Array) (offY incY : Nat) : Array */

/*   imaxRe (N : Nat) (X : Array) (offX incX : Nat) : R */
/*   imaxIm (N : Nat) (X : Array) (offX incX : Nat) : R */
/*   iminRe (N : Nat) (X : Array) (offX incX : Nat) : R */
/*   iminIm (N : Nat) (X : Array) (offX incX : Nat) : R */

/*   /- Element wise operations -/ */
/*   mul (N : Nat) (X : Array) (offX incX : Nat) (Y : Array) (offY incY : Nat) : Array */
/*   div (N : Nat) (X : Array) (offX incX : Nat) (Y : Array) (offY incY : Nat) : Array */
/*   inv (N : Nat) (X : Array) (offX incX : Nat) : Array */
/*   abs (N : Nat) (X : Array) (offX incX : Nat) : Array */
/*   sqrt (N : Nat) (X : Array) (offX incX : Nat) : Array */
/*   exp (N : Nat) (X : Array) (offX incX : Nat) : Array */
/*   log (N : Nat) (X : Array) (offX incX : Nat) : Array */
/*   sin (N : Nat) (X : Array) (offX incX : Nat) : Array */
/*   cos (N : Nat) (X : Array) (offX incX : Nat) : Array */



LEAN_EXPORT lean_obj_res leanblas_cblas_dconst(const size_t N, const double a){

  size_t s = sizeof(double)/sizeof(char);
  lean_obj_res arr = lean_alloc_sarray(sizeof(char), s*N, s*N);
  double * ptr = lean_float64_array_cptr(arr);

  for (size_t i = 0; i < N; i++){
    ptr[i] = a;
  }

  return arr;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dsum(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  double sum = 0;
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    sum += xptr[offX + i*incX];
  }
  return lean_box(sum);
}


LEAN_EXPORT lean_obj_res leanblas_cblas_daxpby(const size_t N, const double alpha, lean_obj_arg X, const size_t offX, const size_t incX,
                                                               const double beta,  lean_obj_arg Y, const size_t offY, const size_t incY){
  // modify `X` in place only iff we are supposed to modify *all* elements of `Y`
  if (lean_is_exclusive(X) && !lean_is_exclusive(Y) &&
      lean_sarray_size(X)*sizeof(double) == N && offX == 0 && incX == 1 &&
      lean_sarray_size(Y)*sizeof(double) == N && offY == 0 && incY == 1){
    // daxpby is not standard CBLAS, implement using dscal and daxpy
    // X = beta*Y + alpha*X
    cblas_dscal((int)N, alpha, lean_float64_array_cptr(X) + offX, (int)incX);
    cblas_daxpy((int)N, beta, lean_float64_array_cptr(Y) + offY, (int)incY, 
                lean_float64_array_cptr(X) + offX, (int)incX);
    lean_dec(Y);
    return X;
  } else {
    ensure_exclusive_byte_array(&Y);
    // daxpby is not standard CBLAS, implement using dscal and daxpy
    // Y = alpha*X + beta*Y
    cblas_dscal((int)N, beta, lean_float64_array_cptr(Y) + offY, (int)incY);
    cblas_daxpy((int)N, alpha, lean_float64_array_cptr(X) + offX, (int)incX,
                lean_float64_array_cptr(Y) + offY, (int)incY);
    lean_dec(X);
    return Y;
  }
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dscaladd(const size_t N, const double alpha, lean_obj_arg X, const size_t offX, const size_t incX,
                                                                  const double beta){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = alpha*xptr[offX + i*incX] + beta;
  }
  return X;
}


LEAN_EXPORT size_t leanblas_cblas_dimax_re(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  double * xptr = lean_float64_array_cptr(X);
  double max = xptr[offX];
  size_t max_index = 0;
  for (size_t i = 1; i < N; i++){
    if (xptr[offX + i*incX] > max){
      max = xptr[offX + i*incX];
      max_index = i;
    }
  }
  return offX + max_index*incX;
}


LEAN_EXPORT size_t leanblas_cblas_dimin_re(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  double * xptr = lean_float64_array_cptr(X);
  double min = xptr[offX];
  size_t min_index = 0;
  for (size_t i = 1; i < N; i++){
    if (xptr[offX + i*incX] < min){
      min = xptr[offX + i*incX];
      min_index = i;
    }
  }
  return offX + min_index*incX;
}



LEAN_EXPORT lean_obj_res leanblas_cblas_dmul(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                             lean_obj_arg Y, const size_t offY, const size_t incY){

  // modify `X` in place only iff we are supposed to modify *all* elements of `Y`
  if (lean_is_exclusive(X) && !lean_is_exclusive(Y) &&
      lean_sarray_size(X)*sizeof(double) == N && offX == 0 && incX == 1 &&
      lean_sarray_size(Y)*sizeof(double) == N && offY == 0 && incY == 1){
    double * xptr = lean_float64_array_cptr(X);
    double * yptr = lean_float64_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      xptr[offX + i*incX] *= yptr[offY + i*incY];
    }
    lean_dec(Y);
    return X;
  } else {
    ensure_exclusive_byte_array(&Y);
    double * xptr = lean_float64_array_cptr(X);
    double * yptr = lean_float64_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      yptr[offY + i*incY] *= xptr[offX + i*incX];
    }
    lean_dec(X);
    return Y;
  }
}


LEAN_EXPORT lean_obj_res leanblas_cblas_ddiv(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                             lean_obj_arg Y, const size_t offY, const size_t incY){
  // modify `X` in place only iff we are supposed to modify *all* elements of `Y`
  if (lean_is_exclusive(X) && !lean_is_exclusive(Y) &&
      lean_sarray_size(X)*sizeof(double) == N && offX == 0 && incX == 1 &&
      lean_sarray_size(Y)*sizeof(double) == N && offY == 0 && incY == 1){
    double * xptr = lean_float64_array_cptr(X);
    double * yptr = lean_float64_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      xptr[offX + i*incX] /= yptr[offY + i*incY];
    }
    lean_dec(Y);
    return X;
  } else {
    ensure_exclusive_byte_array(&Y);
    double * xptr = lean_float64_array_cptr(X);
    double * yptr = lean_float64_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      yptr[offY + i*incY] = xptr[offX + i*incX] / yptr[offY + i*incY];
    }
    lean_dec(X);
    return Y;
  }
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dinv(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = 1.0 / xptr[offX + i*incX];
  }
  return X;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dabs(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = fabs(xptr[offX + i*incX]);
  }
  return X;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dsqrt(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = sqrt(xptr[offX + i*incX]);
  }
  return X;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dexp(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = exp(xptr[offX + i*incX]);
  }
  return X;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dlog(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = log(xptr[offX + i*incX]);
  }
  return X;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dsin(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = sin(xptr[offX + i*incX]);
  }
  return X;
}


LEAN_EXPORT lean_obj_res leanblas_cblas_dcos(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  double * xptr = lean_float64_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = cos(xptr[offX + i*incX]);
  }
  return X;
}

// ============================================================================
// Float32 (Single Precision) BLAS Level 1 Operations
// ============================================================================

/** sdot - Single precision dot product
 * @param N Number of elements
 * @param X First vector (Float32Array)
 * @param offX Starting offset in X
 * @param incX Increment for X
 * @param Y Second vector (Float32Array)
 * @param offY Starting offset in Y
 * @param incY Increment for Y
 * @return Dot product as double (Lean's Float type)
 */
LEAN_EXPORT double leanblas_cblas_sdot(const size_t N,
                                 const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                 const b_lean_obj_arg Y, const size_t offY, const size_t incY){
  return (double)cblas_sdot((int)N, lean_float32_array_cptr(X) + offX, (int)incX,
                                    lean_float32_array_cptr(Y) + offY, (int)incY);
}

/** snrm2 - Single precision Euclidean norm */
LEAN_EXPORT double leanblas_cblas_snrm2(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return (double)cblas_snrm2((int)N, lean_float32_array_cptr(X) + offX, (int)incX);
}

/** sasum - Single precision sum of absolute values */
LEAN_EXPORT double leanblas_cblas_sasum(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return (double)cblas_sasum((int)N, lean_float32_array_cptr(X) + offX, (int)incX);
}

/** isamax - Index of max absolute value (single precision) */
LEAN_EXPORT size_t leanblas_cblas_isamax(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  return cblas_isamax((int)N, lean_float32_array_cptr(X) + offX, (int)incX);
}

/** sswap - Swap two single precision vectors */
LEAN_EXPORT lean_obj_res leanblas_cblas_sswap(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                              lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&X);
  ensure_exclusive_byte_array(&Y);
  cblas_sswap((int)N, lean_float32_array_cptr(X) + offX, (int)incX,
                      lean_float32_array_cptr(Y) + offY, (int)incY);
  lean_obj_res result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, X);
  lean_ctor_set(result, 1, Y);
  return result;
}

/** scopy - Copy single precision vector */
LEAN_EXPORT lean_obj_res leanblas_cblas_scopy(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                                              lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&Y);
  cblas_scopy((int)N, lean_float32_array_cptr(X) + offX, (int)incX,
                      lean_float32_array_cptr(Y) + offY, (int)incY);
  return Y;
}

/** saxpy - Single precision: Y := alpha*X + Y */
LEAN_EXPORT lean_obj_res leanblas_cblas_saxpy(const size_t N, const double alpha,
                                              const b_lean_obj_arg X, const size_t offX, const size_t incX,
                                              lean_obj_arg Y, const size_t offY, const size_t incY){
  ensure_exclusive_byte_array(&Y);
  cblas_saxpy((int)N, (float)alpha, lean_float32_array_cptr(X) + offX, (int)incX,
                                    lean_float32_array_cptr(Y) + offY, (int)incY);
  return Y;
}

/** sscal - Single precision: X := alpha*X */
LEAN_EXPORT lean_obj_res leanblas_cblas_sscal(const size_t N, const double alpha,
                                              lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  cblas_sscal((int)N, (float)alpha, lean_float32_array_cptr(X) + offX, (int)incX);
  return X;
}

/** srotg - Construct Givens rotation (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_srotg(double a, double b){
  float fa = (float)a, fb = (float)b, fc, fs;
  cblas_srotg(&fa, &fb, &fc, &fs);
  lean_obj_res result = lean_alloc_ctor(0, 0, 4*sizeof(double));
  lean_ctor_set_float(result, 0*sizeof(double), (double)fa);
  lean_ctor_set_float(result, 1*sizeof(double), (double)fb);
  lean_ctor_set_float(result, 2*sizeof(double), (double)fc);
  lean_ctor_set_float(result, 3*sizeof(double), (double)fs);
  return result;
}

/** srot - Apply Givens rotation (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_srot(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                              lean_obj_arg Y, const size_t offY, const size_t incY,
                                                              const double c, const double s){
  ensure_exclusive_byte_array(&X);
  ensure_exclusive_byte_array(&Y);
  cblas_srot((int)N, lean_float32_array_cptr(X) + offX, (int)incX,
                     lean_float32_array_cptr(Y) + offY, (int)incY, (float)c, (float)s);
  lean_obj_res result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, X);
  lean_ctor_set(result, 1, Y);
  return result;
}

/** sconst - Create constant single precision vector (non-standard) */
LEAN_EXPORT lean_obj_res leanblas_cblas_sconst(const size_t N, const double alpha){
  size_t byte_size = N * 4;
  lean_obj_res arr = lean_alloc_sarray(1, byte_size, byte_size);
  float* ptr = (float*)lean_sarray_cptr(arr);
  float val = (float)alpha;
  for (size_t i = 0; i < N; i++) {
    ptr[i] = val;
  }
  return arr;
}

/** ssum - Sum of elements (single precision, non-standard) */
LEAN_EXPORT double leanblas_cblas_ssum(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  const float* ptr = lean_float32_array_cptr(X) + offX;
  double sum = 0.0;
  for (size_t i = 0; i < N; i++) {
    sum += (double)ptr[i * incX];
  }
  return sum;
}

/** srotmg - Construct modified Givens rotation (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_srotmg(const double d1, const double d2, const double x1, const double y1){
  float fd1 = (float)d1, fd2 = (float)d2, fx1 = (float)x1;
  float param[5];
  cblas_srotmg(&fd1, &fd2, &fx1, (float)y1, param);

  lean_obj_res res = lean_alloc_ctor(0, 0, 5*sizeof(double));
  lean_ctor_set_float(res, 0*sizeof(double), (double)fd1);
  lean_ctor_set_float(res, 1*sizeof(double), (double)fd2);
  lean_ctor_set_float(res, 2*sizeof(double), (double)fx1);
  lean_ctor_set_float(res, 3*sizeof(double), (double)param[0]);
  lean_ctor_set_float(res, 4*sizeof(double), (double)param[1]);
  return res;
}

/** saxpby - Single precision: Y := alpha*X + beta*Y (non-standard) */
LEAN_EXPORT lean_obj_res leanblas_cblas_saxpby(const size_t N, const double alpha, lean_obj_arg X, const size_t offX, const size_t incX,
                                                               const double beta,  lean_obj_arg Y, const size_t offY, const size_t incY){
  if (lean_is_exclusive(X) && !lean_is_exclusive(Y) &&
      lean_sarray_size(X)/4 == N && offX == 0 && incX == 1 &&
      lean_sarray_size(Y)/4 == N && offY == 0 && incY == 1){
    cblas_sscal((int)N, (float)alpha, lean_float32_array_cptr(X) + offX, (int)incX);
    cblas_saxpy((int)N, (float)beta, lean_float32_array_cptr(Y) + offY, (int)incY,
                lean_float32_array_cptr(X) + offX, (int)incX);
    lean_dec(Y);
    return X;
  } else {
    ensure_exclusive_byte_array(&Y);
    cblas_sscal((int)N, (float)beta, lean_float32_array_cptr(Y) + offY, (int)incY);
    cblas_saxpy((int)N, (float)alpha, lean_float32_array_cptr(X) + offX, (int)incX,
                lean_float32_array_cptr(Y) + offY, (int)incY);
    lean_dec(X);
    return Y;
  }
}

/** sscaladd - Single precision: X := alpha*X + beta (non-standard) */
LEAN_EXPORT lean_obj_res leanblas_cblas_sscaladd(const size_t N, const double alpha, lean_obj_arg X, const size_t offX, const size_t incX,
                                                                  const double beta){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  float a = (float)alpha, b = (float)beta;
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = a*xptr[offX + i*incX] + b;
  }
  return X;
}

/** simax_re - Index of max value (single precision) */
LEAN_EXPORT size_t leanblas_cblas_simax_re(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  const float* xptr = lean_float32_array_cptr(X);
  float max = xptr[offX];
  size_t max_index = 0;
  for (size_t i = 1; i < N; i++){
    if (xptr[offX + i*incX] > max){
      max = xptr[offX + i*incX];
      max_index = i;
    }
  }
  return offX + max_index*incX;
}

/** simin_re - Index of min value (single precision) */
LEAN_EXPORT size_t leanblas_cblas_simin_re(const size_t N, const b_lean_obj_arg X, const size_t offX, const size_t incX){
  const float* xptr = lean_float32_array_cptr(X);
  float min = xptr[offX];
  size_t min_index = 0;
  for (size_t i = 1; i < N; i++){
    if (xptr[offX + i*incX] < min){
      min = xptr[offX + i*incX];
      min_index = i;
    }
  }
  return offX + min_index*incX;
}

/** smul - Element-wise multiply (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_smul(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                             lean_obj_arg Y, const size_t offY, const size_t incY){
  if (lean_is_exclusive(X) && !lean_is_exclusive(Y) &&
      lean_sarray_size(X)/4 == N && offX == 0 && incX == 1 &&
      lean_sarray_size(Y)/4 == N && offY == 0 && incY == 1){
    float* xptr = lean_float32_array_cptr(X);
    const float* yptr = lean_float32_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      xptr[offX + i*incX] *= yptr[offY + i*incY];
    }
    lean_dec(Y);
    return X;
  } else {
    ensure_exclusive_byte_array(&Y);
    const float* xptr = lean_float32_array_cptr(X);
    float* yptr = lean_float32_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      yptr[offY + i*incY] *= xptr[offX + i*incX];
    }
    lean_dec(X);
    return Y;
  }
}

/** sdiv - Element-wise divide (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_sdiv(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX,
                                                             lean_obj_arg Y, const size_t offY, const size_t incY){
  if (lean_is_exclusive(X) && !lean_is_exclusive(Y) &&
      lean_sarray_size(X)/4 == N && offX == 0 && incX == 1 &&
      lean_sarray_size(Y)/4 == N && offY == 0 && incY == 1){
    float* xptr = lean_float32_array_cptr(X);
    const float* yptr = lean_float32_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      xptr[offX + i*incX] /= yptr[offY + i*incY];
    }
    lean_dec(Y);
    return X;
  } else {
    ensure_exclusive_byte_array(&Y);
    const float* xptr = lean_float32_array_cptr(X);
    float* yptr = lean_float32_array_cptr(Y);
    for (size_t i = 0; i < N; i++){
      yptr[offY + i*incY] = xptr[offX + i*incX] / yptr[offY + i*incY];
    }
    lean_dec(X);
    return Y;
  }
}

/** sinv - Element-wise inverse (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_sinv(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = 1.0f / xptr[offX + i*incX];
  }
  return X;
}

/** sabs - Element-wise absolute value (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_sabs(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = fabsf(xptr[offX + i*incX]);
  }
  return X;
}

/** ssqrt - Element-wise square root (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_ssqrt(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = sqrtf(xptr[offX + i*incX]);
  }
  return X;
}

/** sexp - Element-wise exponential (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_sexp(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = expf(xptr[offX + i*incX]);
  }
  return X;
}

/** slog - Element-wise natural log (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_slog(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = logf(xptr[offX + i*incX]);
  }
  return X;
}

/** ssin - Element-wise sine (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_ssin(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = sinf(xptr[offX + i*incX]);
  }
  return X;
}

/** scos - Element-wise cosine (single precision) */
LEAN_EXPORT lean_obj_res leanblas_cblas_scos(const size_t N, lean_obj_arg X, const size_t offX, const size_t incX){
  ensure_exclusive_byte_array(&X);
  float* xptr = lean_float32_array_cptr(X);
  for (size_t i = 0; i < N; i++){
    xptr[offX + i*incX] = cosf(xptr[offX + i*incX]);
  }
  return X;
}
