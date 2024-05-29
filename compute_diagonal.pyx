
# cython: language_level=3
# cython: infer_types=False

import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport zhemm, zdotc, zhpmv
from cython cimport boundscheck, wraparound, parallel

# @boundscheck(False)
# @wraparound(False)
# def compute_diagonal_blas(np.ndarray[np.complex128_t, ndim=2, mode="fortran"] U,
#                           np.ndarray[np.complex128_t, ndim=2, mode="fortran"] A):
#     cdef int n = U.shape[0]
#     cdef np.ndarray[np.complex128_t, ndim=1, mode="fortran"] C_diag = np.zeros(n, dtype=np.complex128, order='F')
#     cdef np.ndarray[np.complex128_t, ndim=2, mode="fortran"] B = np.zeros((n, n), dtype=np.complex128, order='F')
#     cdef int i

#     cdef double complex alpha = 1.0 + 0.0j
#     cdef double complex beta = 0.0 + 0.0j
#     cdef int one = 1

#     # Compute B = A * U using BLAS zgemm
#     with nogil:
#         zgemm(b"N", b"N", &n, &n, &n, &alpha, &A[0, 0], &n, &U[0, 0], &n, &beta, &B[0, 0], &n)
#         # Compute the diagonal elements of C = U^H * B
#         for i in range(n):
#                 C_diag[i] = zdotc(&n, &U[0, i], &one, &B[0, i], &one)

#     return C_diag


# @boundscheck(False)
# @wraparound(False)
# def compute_diagonal_blas(np.ndarray[np.complex128_t, ndim=2, mode="fortran"] U,
#                           np.ndarray[np.complex128_t, ndim=1, mode="fortran"] A) -> np.ndarray[np.float64_t]:
#     cdef int n = U.shape[0]
#     cdef np.npy_intp *dim = [n]
#     cdef np.npy_intp *dims = [n, n]
#     cdef np.ndarray[np.float64_t, ndim=1, mode="fortran"] C_diag = np.PyArray_EMPTY(1, dim, np.NPY_FLOAT64, 1)
#     cdef np.ndarray[np.complex128_t, ndim=2, mode="fortran"] B = np.PyArray_EMPTY(2, dims, np.NPY_COMPLEX128, 1)
#     cdef int i

#     cdef double complex alpha = 1.0 + 0.0j
#     cdef double complex beta = 0.0 + 0.0j
#     cdef int one = 1

#     with nogil:
#         for i in parallel.prange(n):
#             zhpmv(b"U", &n, &alpha, &A[0], &U[0, i], &one, &beta, &B[0,i], &one)
#         for i in parallel.prange(n):
#             C_diag[i] = zdotc(&n, &U[0, i], &one, &B[0,i], &one).real
#     return C_diag


@boundscheck(False)
@wraparound(False)
def compute_diagonal_blas(np.ndarray[np.complex128_t, ndim=2, mode="fortran"] U,
                          np.ndarray[np.complex128_t, ndim=2, mode="fortran"] A) -> np.ndarray[np.float64_t]:
    cdef int n = U.shape[0]
    cdef np.npy_intp *dims = [n, n]
    cdef np.npy_intp *dim = [n]
    cdef np.ndarray[np.float64_t, ndim=1, mode="fortran"] C_diag = np.PyArray_EMPTY(1, dim, np.NPY_FLOAT64, 1)
    cdef np.ndarray[np.complex128_t, ndim=2, mode="fortran"] B = np.PyArray_EMPTY(2, dims, np.NPY_COMPLEX128, 1)

    cdef int i
    cdef double complex alpha = 1.0 + 0.0j
    cdef double complex beta = 0.0 + 0.0j
    cdef int one = 1

    # Compute B = A * U using BLAS zgemm
    with nogil:
        zhemm(b"L", b"U", &n, &n, &alpha, &A[0, 0], &n, &U[0, 0], &n, &beta, &B[0, 0], &n)
        # Compute the diagonal elements of C = U^H * B
        for i in range(n):
                C_diag[i] = zdotc(&n, &U[0, i], &one, &B[0, i], &one).real

    return C_diag
