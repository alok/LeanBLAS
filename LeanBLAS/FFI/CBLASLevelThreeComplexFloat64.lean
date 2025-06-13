import LeanBLAS.FFI.FloatArray

namespace BLAS.CBLAS

/-! # CBLAS Level 3 FFI Bindings for ComplexFloat64

This module provides low-level FFI (Foreign Function Interface) bindings to CBLAS Level 3
operations for double-precision complex floating-point numbers (ComplexFloat64).

## Overview

Complex Level 3 BLAS operations are matrix-matrix operations with O(n³) complexity.
This module exposes the C interface for complex matrix arithmetic.

## Matrix Storage

Complex matrices use the same layout conventions as real matrices, but each
element occupies 16 bytes (8 bytes real + 8 bytes imaginary). The leading dimension
parameter must account for this when working with submatrices.

## Hermitian vs Symmetric

For complex matrices, we distinguish between:
- **Symmetric**: A = A^T (transpose without conjugation)  
- **Hermitian**: A = A^H (conjugate transpose)

Most applications use Hermitian matrices for complex numbers as they have
real eigenvalues and better numerical properties.
-/

/-- zgemm

summary: computes C = alpha*op(A)*op(B) + beta*C for general complex matrices

inputs:
- order: row-major (CblasRowMajor) or column-major (CblasColMajor)
- transA: operation to apply to A (NoTrans, Trans, or ConjTrans)
- transB: operation to apply to B (NoTrans, Trans, or ConjTrans)
- M: number of rows of op(A) and C
- N: number of columns of op(B) and C
- K: number of columns of op(A) and rows of op(B)
- alpha: scalar multiplier
- A: first matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: second matrix
- offB: starting offset in B
- ldb: leading dimension of B
- beta: scalar multiplier for C
- C: input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*op(A)*op(B) + beta*C

C interface:
```
void cblas_zgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zgemm"]
opaque zgemm (order : Order) (transA transB : Transpose) (M N K : USize) 
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize) 
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- zsymm

summary: computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is symmetric

inputs:
- order: row-major or column-major
- side: left (A*B) or right (B*A)
- uplo: upper or lower triangular part of A stored
- M: number of rows of C
- N: number of columns of C
- alpha: scalar multiplier
- A: symmetric matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: general matrix
- offB: starting offset in B
- ldb: leading dimension of B
- beta: scalar multiplier for C
- C: input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*A*B + beta*C or C = alpha*B*A + beta*C

C interface:
```
void cblas_zsymm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                 const enum CBLAS_UPLO uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zsymm"]
opaque zsymm (order : Order) (side : Side) (uplo : Uplo) (M N : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- zhemm

summary: computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is Hermitian

inputs:
- order: row-major or column-major
- side: left (A*B) or right (B*A)
- uplo: upper or lower triangular part of A stored
- M: number of rows of C
- N: number of columns of C
- alpha: scalar multiplier
- A: Hermitian matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: general matrix
- offB: starting offset in B
- ldb: leading dimension of B
- beta: scalar multiplier for C
- C: input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*A*B + beta*C or C = alpha*B*A + beta*C

C interface:
```
void cblas_zhemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                 const enum CBLAS_UPLO uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zhemm"]
opaque zhemm (order : Order) (side : Side) (uplo : Uplo) (M N : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- zsyrk

summary: performs symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part of C stored
- trans: NoTrans (A*A^T) or Trans (A^T*A)
- N: order of matrix C
- K: number of columns of A (NoTrans) or rows of A (Trans)
- alpha: scalar multiplier
- A: matrix
- offA: starting offset in A
- lda: leading dimension of A
- beta: scalar multiplier for C
- C: symmetric input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C

C interface:
```
void cblas_zsyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zsyrk"]
opaque zsyrk (order : Order) (uplo : Uplo) (trans : Transpose) (N K : USize)
             (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (beta : ComplexFloat)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- zherk

summary: performs Hermitian rank-k update C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part of C stored
- trans: NoTrans (A*A^H) or ConjTrans (A^H*A)
- N: order of matrix C
- K: number of columns of A (NoTrans) or rows of A (ConjTrans)
- alpha: real scalar multiplier
- A: matrix
- offA: starting offset in A
- lda: leading dimension of A
- beta: real scalar multiplier for C
- C: Hermitian input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C

C interface:
```
void cblas_zherk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const int N, const int K,
                 const double alpha, const void *A, const int lda,
                 const double beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zherk"]
opaque zherk (order : Order) (uplo : Uplo) (trans : Transpose) (N K : USize)
             (alpha : Float)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (beta : Float)
             (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- zsyr2k

summary: performs symmetric rank-2k update C = alpha*A*B^T + alpha*B*A^T + beta*C

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part of C stored
- trans: NoTrans or Trans
- N: order of matrix C
- K: number of columns of A and B (NoTrans) or rows (Trans)
- alpha: scalar multiplier
- A: first matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: second matrix
- offB: starting offset in B
- ldb: leading dimension of B
- beta: scalar multiplier for C
- C: symmetric input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*A*B^T + alpha*B*A^T + beta*C or C = alpha*A^T*B + alpha*B^T*A + beta*C

C interface:
```
void cblas_zsyr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                  const enum CBLAS_TRANSPOSE trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb,
                  const void *beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zsyr2k"]
opaque zsyr2k (order : Order) (uplo : Uplo) (trans : Transpose) (N K : USize)
              (alpha : ComplexFloat)
              (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
              (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
              (beta : ComplexFloat)
              (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- zher2k

summary: performs Hermitian rank-2k update C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C

inputs:
- order: row-major or column-major
- uplo: upper or lower triangular part of C stored
- trans: NoTrans or ConjTrans
- N: order of matrix C
- K: number of columns of A and B (NoTrans) or rows (ConjTrans)
- alpha: scalar multiplier
- A: first matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: second matrix
- offB: starting offset in B
- ldb: leading dimension of B
- beta: real scalar multiplier for C
- C: Hermitian input/output matrix
- offC: starting offset in C
- ldc: leading dimension of C

outputs: C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C

C interface:
```
void cblas_zher2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                  const enum CBLAS_TRANSPOSE trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda,
                  const void *B, const int ldb,
                  const double beta, void *C, const int ldc);
```
-/
@[extern "leanblas_cblas_zher2k"]
opaque zher2k (order : Order) (uplo : Uplo) (trans : Transpose) (N K : USize)
              (alpha : ComplexFloat)
              (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
              (B : @& ComplexFloat64Array) (offB : USize) (ldb : USize)
              (beta : Float)
              (C : ComplexFloat64Array) (offC : USize) (ldc : USize) : ComplexFloat64Array

/-- ztrmm

summary: computes B = alpha*op(A)*B or B = alpha*B*op(A) where A is triangular

inputs:
- order: row-major or column-major
- side: left (A*B) or right (B*A)
- uplo: upper or lower triangular
- transA: operation to apply to A (NoTrans, Trans, or ConjTrans)
- diag: unit or non-unit diagonal
- M: number of rows of B
- N: number of columns of B
- alpha: scalar multiplier
- A: triangular matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: input/output matrix
- offB: starting offset in B
- ldb: leading dimension of B

outputs: B = alpha*op(A)*B or B = alpha*B*op(A)

C interface:
```
void cblas_ztrmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                 const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_DIAG diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb);
```
-/
@[extern "leanblas_cblas_ztrmm"]
opaque ztrmm (order : Order) (side : Side) (uplo : Uplo) (transA : Transpose) (diag : Diag)
             (M N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : ComplexFloat64Array) (offB : USize) (ldb : USize) : ComplexFloat64Array

/-- ztrsm

summary: solves op(A)*X = alpha*B or X*op(A) = alpha*B for X where A is triangular

inputs:
- order: row-major or column-major
- side: left (A*X = alpha*B) or right (X*A = alpha*B)
- uplo: upper or lower triangular
- transA: operation to apply to A (NoTrans, Trans, or ConjTrans)
- diag: unit or non-unit diagonal
- M: number of rows of B
- N: number of columns of B
- alpha: scalar multiplier
- A: triangular matrix
- offA: starting offset in A
- lda: leading dimension of A
- B: right-hand side on input, solution X on output
- offB: starting offset in B
- ldb: leading dimension of B

outputs: B = X where op(A)*X = alpha*B or X*op(A) = alpha*B

C interface:
```
void cblas_ztrsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                 const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_DIAG diag, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 void *B, const int ldb);
```
-/
@[extern "leanblas_cblas_ztrsm"]
opaque ztrsm (order : Order) (side : Side) (uplo : Uplo) (transA : Transpose) (diag : Diag)
             (M N : USize) (alpha : ComplexFloat)
             (A : @& ComplexFloat64Array) (offA : USize) (lda : USize)
             (B : ComplexFloat64Array) (offB : USize) (ldb : USize) : ComplexFloat64Array

end BLAS.CBLAS