import LeanBLAS.FFI.FloatArray

namespace BLAS.CBLAS

/-! # CBLAS Level 2 FFI Bindings for ComplexFloat64

This module provides low-level FFI (Foreign Function Interface) bindings to CBLAS Level 2
operations for double-precision complex floating-point numbers (ComplexFloat64).

## Overview

Complex Level 2 BLAS operations are matrix-vector operations with O(n²) complexity.
This module exposes the C interface for complex matrix-vector arithmetic.

## Matrix Storage

Complex matrices use the same layout conventions as real matrices, but each
element occupies 16 bytes (real + imaginary parts). The leading dimension
parameter must account for this when working with submatrices.

## Hermitian vs Symmetric

For complex matrices, we distinguish between:
- **Symmetric**: A = A^T (transpose without conjugation)
- **Hermitian**: A = A^H (conjugate transpose)

Most applications use Hermitian matrices for complex numbers as they have
real eigenvalues and better numerical properties.
-/

/-- zgemv

summary: computes y = alpha*A*x + beta*y for general complex matrices

inputs:
- order: row-major (CblasRowMajor) or column-major (CblasColMajor)
- trans: operation to apply (NoTrans, Trans, or ConjTrans)
- M: number of rows of A
- N: number of columns of A
- alpha: scalar multiplier
- A: the matrix
- offA: starting offset in A
- lda: leading dimension of A
- X: input vector
- offX: starting offset in X
- incX: stride of X
- beta: scalar multiplier for Y
- Y: input/output vector
- offY: starting offset in Y
- incY: stride of Y

outputs: Y = alpha*op(A)*X + beta*Y

C interface:
```
void cblas_zgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans,
                 const int M, const int N, const void *alpha,
                 const void *A, const int lda,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY);
```
-/
@[extern "leanblas_cblas_zgemv"]
opaque zgemv (order : Order) (trans : Transpose) (M N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : @& ComplexFloat64Array) (offX incX : USize) (beta : ComplexFloat)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array

/-- zhemv

summary: computes y = alpha*A*x + beta*y for Hermitian complex matrices

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part stored
- N: order of matrix A
- alpha: scalar multiplier
- A: Hermitian matrix
- offA: starting offset in A
- lda: leading dimension of A
- X: input vector
- offX: starting offset in X
- incX: stride of X
- beta: scalar multiplier for Y
- Y: input/output vector
- offY: starting offset in Y
- incY: stride of Y

outputs: Y = alpha*A*X + beta*Y where A is Hermitian

C interface:
```
void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const int N, const void *alpha,
                 const void *A, const int lda,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY);
```
-/
@[extern "leanblas_cblas_zhemv"]
opaque zhemv (order : Order) (uplo : Uplo) (N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : @& ComplexFloat64Array) (offX incX : USize) (beta : ComplexFloat)
             (Y : ComplexFloat64Array) (offY incY : USize) : ComplexFloat64Array

/-- ztrmv

summary: computes x = A*x or x = A^T*x or x = A^H*x for triangular complex matrices

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular
- trans: operation to apply (NoTrans, Trans, or ConjTrans)
- diag: unit or non-unit diagonal
- N: order of matrix A
- A: triangular matrix
- offA: starting offset in A
- lda: leading dimension of A
- X: input/output vector
- offX: starting offset in X
- incX: stride of X

outputs: X = op(A)*X

C interface:
```
void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const int N, const void *A, const int lda,
                 void *X, const int incX);
```
-/
@[extern "leanblas_cblas_ztrmv"]
opaque ztrmv (order : Order) (uplo : Uplo) (trans : Transpose) (diag : Diag)
             (N : USize) (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : ComplexFloat64Array) (offX incX : USize) : ComplexFloat64Array

/-- ztrsv

summary: solves A*x = b or A^T*x = b or A^H*x = b for triangular complex matrices

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular
- trans: operation to apply (NoTrans, Trans, or ConjTrans)
- diag: unit or non-unit diagonal
- N: order of matrix A
- A: triangular matrix
- offA: starting offset in A
- lda: leading dimension of A
- X: right-hand side vector on input, solution on output
- offX: starting offset in X
- incX: stride of X

outputs: X = inv(op(A))*X

C interface:
```
void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const int N, const void *A, const int lda,
                 void *X, const int incX);
```
-/
@[extern "leanblas_cblas_ztrsv"]
opaque ztrsv (order : Order) (uplo : Uplo) (trans : Transpose) (diag : Diag)
             (N : USize) (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (X : ComplexFloat64Array) (offX incX : USize) : ComplexFloat64Array

/-- zgerc

summary: performs rank-1 update A = alpha*x*y^H + A (conjugate of y)

inputs:
- order: row-major or column-major
- M: number of rows of A
- N: number of columns of A
- alpha: scalar multiplier
- X: first vector
- offX: starting offset in X
- incX: stride of X
- Y: second vector (will be conjugated)
- offY: starting offset in Y
- incY: stride of Y
- A: matrix to update
- offA: starting offset in A
- lda: leading dimension of A

outputs: A = alpha*X*conj(Y)^T + A

C interface:
```
void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY,
                 void *A, const int lda);
```
-/
@[extern "leanblas_cblas_zgerc"]
opaque zgerc (order : Order) (M N : USize) (alpha : ComplexFloat)
             (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize)
             (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

/-- zgeru

summary: performs rank-1 update A = alpha*x*y^T + A (no conjugation)

inputs:
- order: row-major or column-major
- M: number of rows of A
- N: number of columns of A
- alpha: scalar multiplier
- X: first vector
- offX: starting offset in X
- incX: stride of X
- Y: second vector (not conjugated)
- offY: starting offset in Y
- incY: stride of Y
- A: matrix to update
- offA: starting offset in A
- lda: leading dimension of A

outputs: A = alpha*X*Y^T + A

C interface:
```
void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY,
                 void *A, const int lda);
```
-/
@[extern "leanblas_cblas_zgeru"]
opaque zgeru (order : Order) (M N : USize) (alpha : ComplexFloat)
             (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize)
             (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

/-- zher

summary: performs Hermitian rank-1 update A = alpha*x*x^H + A

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part stored
- N: order of matrix A
- alpha: real scalar multiplier (imaginary part ignored)
- X: vector
- offX: starting offset in X
- incX: stride of X
- A: Hermitian matrix to update
- offA: starting offset in A
- lda: leading dimension of A

outputs: A = alpha*X*conj(X)^T + A (only upper or lower part updated)

C interface:
```
void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                const int N, const double alpha,
                const void *X, const int incX,
                void *A, const int lda);
```
-/
@[extern "leanblas_cblas_zher"]
opaque zher (order : Order) (uplo : Uplo) (N : USize) (alpha : Float)
            (X : @& ComplexFloat64Array) (offX incX : USize)
            (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

/-- zher2

summary: performs Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part stored
- N: order of matrix A
- alpha: scalar multiplier
- X: first vector
- offX: starting offset in X
- incX: stride of X
- Y: second vector
- offY: starting offset in Y
- incY: stride of Y
- A: Hermitian matrix to update
- offA: starting offset in A
- lda: leading dimension of A

outputs: A = alpha*X*conj(Y)^T + conj(alpha)*Y*conj(X)^T + A

C interface:
```
void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const int N, const void *alpha,
                 const void *X, const int incX,
                 const void *Y, const int incY,
                 void *A, const int lda);
```
-/
@[extern "leanblas_cblas_zher2"]
opaque zher2 (order : Order) (uplo : Uplo) (N : USize) (alpha : ComplexFloat)
             (X : @& ComplexFloat64Array) (offX incX : USize)
             (Y : @& ComplexFloat64Array) (offY incY : USize)
             (A : ComplexFloat64Array) (offA : USize) (lda : USize) : ComplexFloat64Array

end BLAS.CBLAS