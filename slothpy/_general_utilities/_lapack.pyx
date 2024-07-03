# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

cimport numpy as np
import numpy as np
from scipy.linalg.cython_blas cimport zaxpy, zhemm, zdotc, caxpy, chemm, cdotc
from scipy.linalg.cython_lapack cimport zheevr, cheevr
from cython cimport boundscheck, wraparound

# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


#Double precision


@boundscheck(False)
@wraparound(False)
def _zheevr_lwork(int n, jobz ='N', range='A', int il=0, int iu=0, np.float64_t vl=-1e30, np.float64_t vu=0):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int lda = n
        int ldz = n
        int m
        int info
        np.ndarray[np.complex128_t, ndim=2, mode='fortran'] a
        np.ndarray[np.complex128_t, ndim=2, mode='fortran'] z
        np.ndarray[np.float64_t, ndim=1] w
        np.ndarray[np.complex128_t, ndim=1] work
        int lwork
        np.ndarray[np.float64_t, ndim=1] rwork
        int lrwork
        np.ndarray[int, ndim=1] iwork
        int liwork
        np.ndarray[int, ndim=1] isuppz
        np.npy_intp *dims = [n, n]
        np.float64_t abs_tol = -1

    a = np.PyArray_EMPTY(2, dims, np.NPY_COMPLEX128, 1)
    if range == 'I':
        w = np.PyArray_EMPTY(1, [iu-il+1], np.NPY_FLOAT64, 1)
    else:
        w = np.PyArray_EMPTY(1, [n], np.NPY_FLOAT64, 1)
    isuppz = np.PyArray_EMPTY(1, [2*n], np.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            z = np.PyArray_EMPTY(2, [n, iu-il+1], np.NPY_COMPLEX128, 1)
        else:
            z = np.PyArray_EMPTY(2, [n, n], np.NPY_COMPLEX128, 1)
    else: 
        z = np.PyArray_EMPTY(2, [1, 1], np.NPY_COMPLEX128, 1)

    lwork = -1
    lrwork = -1
    liwork = -1
    work = np.PyArray_EMPTY(1, [1], np.NPY_COMPLEX128, 1)
    rwork = np.PyArray_EMPTY(1, [1], np.NPY_FLOAT64, 1)
    iwork = np.PyArray_EMPTY(1, [1], np.NPY_INT32, 1)
    
    with nogil:
        zheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)
 
    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")

    lwork = int(work[0].real)
    lrwork = int(rwork[0])
    liwork = int(iwork[0])
    
    return lwork, lrwork, liwork


@boundscheck(False)
@wraparound(False)
def _zheevr(np.ndarray[np.complex128_t, ndim=2, mode='fortran'] a, int lwork, int lrwork, int liwork, jobz='N', range='A', int il=0, int iu=0, np.float64_t vl=-1e30, np.float64_t vu=0):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int n = a.shape[0]
        int lda = n
        int ldz = n
        int m
        int info
        np.ndarray[np.complex128_t, ndim=2, mode='fortran'] z
        np.ndarray[np.float64_t, ndim=1] w
        np.ndarray[np.complex128_t, ndim=1] work
        np.ndarray[np.float64_t, ndim=1] rwork
        np.ndarray[int, ndim=1] iwork
        np.ndarray[int, ndim=1] isuppz
        np.float64_t abs_tol = -1

    if a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix must be square")

    if range == 'I':
        w = np.PyArray_EMPTY(1, [iu-il+1], np.NPY_FLOAT64, 1)
    else:
        w = np.PyArray_EMPTY(1, [n], np.NPY_FLOAT64, 1)
    isuppz = np.PyArray_EMPTY(1, [2*n], np.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            z = np.PyArray_EMPTY(2, [n, iu-il+1], np.NPY_COMPLEX128, 1)
        else:
            z = np.PyArray_EMPTY(2, [n, n], np.NPY_COMPLEX128, 1)
    else:
        z = np.PyArray_EMPTY(2, [1, 1], np.NPY_COMPLEX128, 1)
    
    work = np.PyArray_EMPTY(1, [lwork], np.NPY_COMPLEX128, 1)
    rwork = np.PyArray_EMPTY(1, [lrwork], np.NPY_FLOAT64, 1)
    iwork = np.PyArray_EMPTY(1, [liwork], np.NPY_INT32, 1)

    with nogil:
        zheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)

    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")
    
    if jobz == 'V':
        return w[:m], z[:, :m]
    else:
        return w[:m]


@boundscheck(False)
@wraparound(False)
def _zutmud(np.ndarray[np.complex128_t, ndim=2, mode="fortran"] U,
                          np.ndarray[np.complex128_t, ndim=2, mode="fortran"] M) -> np.ndarray[np.float64_t]:
    cdef int m = U.shape[0]
    cdef int n = U.shape[1]
    cdef np.npy_intp *dims = [m, n]
    cdef np.npy_intp *dim = [n]
    cdef np.ndarray[np.float64_t, ndim=1, mode="fortran"] C_diag = np.PyArray_EMPTY(1, dim, np.NPY_FLOAT64, 1)
    cdef np.ndarray[np.complex128_t, ndim=2, mode="fortran"] B = np.PyArray_EMPTY(2, dims, np.NPY_COMPLEX128, 1)

    cdef int i
    cdef np.complex128_t alpha = 1.0 + 0.0j
    cdef np.complex128_t beta = 0.0 + 0.0j
    cdef int one = 1

    with nogil:
        zhemm(b"L", b"U", &m, &n, &alpha, &M[0, 0], &m, &U[0, 0], &m, &beta, &B[0, 0], &m)
        for i in range(n):
                C_diag[i] = zdotc(&n, &U[0, i], &one, &B[0, i], &one).real

    return C_diag


@boundscheck(False)
@wraparound(False)
def _zdot3d(np.ndarray[np.complex128_t, ndim=3] M, np.ndarray[np.float64_t, ndim=1] V) -> np.ndarray[np.complex128_t]:
    cdef int m = M.shape[1]
    cdef int n = M.shape[2]
    cdef np.npy_intp *dims = [m, n]
    cdef np.ndarray[np.complex128_t, ndim=2] result = np.PyArray_ZEROS(2, dims, np.NPY_COMPLEX128, 0)

    cdef int one = 1
    cdef np.complex128_t alpha0 = V[0] + 0.0j
    cdef np.complex128_t alpha1 = V[1] + 0.0j
    cdef np.complex128_t alpha2 = V[2] + 0.0j
    cdef int size = m * n

    with nogil:
        zaxpy(&size, &alpha0, &M[0, 0, 0], &one, &result[0, 0], &one)
        zaxpy(&size, &alpha1, &M[1, 0, 0], &one, &result[0, 0], &one)
        zaxpy(&size, &alpha2, &M[2, 0, 0], &one, &result[0, 0], &one)

    return result


#Single precision


@boundscheck(False)
@wraparound(False)
def _cheevr_lwork(int n, jobz='N', range='A', int il=0, int iu=0, np.float32_t vl=-1e30, np.float32_t vu=0):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int lda = n
        int ldz = n
        int m
        int info
        np.ndarray[np.complex64_t, ndim=2, mode='fortran'] a
        np.ndarray[np.complex64_t, ndim=2, mode='fortran'] z
        np.ndarray[np.float32_t, ndim=1] w
        np.ndarray[np.complex64_t, ndim=1] work
        int lwork
        np.ndarray[np.float32_t, ndim=1] rwork
        int lrwork
        np.ndarray[int, ndim=1] iwork
        int liwork
        np.ndarray[int, ndim=1] isuppz
        np.npy_intp *dims = [n, n]
        np.float32_t abs_tol = -1

    a = np.PyArray_EMPTY(2, dims, np.NPY_COMPLEX64, 1)
    if range == 'I':
        w = np.PyArray_EMPTY(1, [iu-il+1], np.NPY_FLOAT32, 1)
    else:
        w = np.PyArray_EMPTY(1, [n], np.NPY_FLOAT32, 1)
    isuppz = np.PyArray_EMPTY(1, [2*n], np.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            z = np.PyArray_EMPTY(2, [n, iu-il+1], np.NPY_COMPLEX64, 1)
        else:
            z = np.PyArray_EMPTY(2, [n, n], np.NPY_COMPLEX64, 1)
    else:
        z = np.PyArray_EMPTY(2, [1, 1], np.NPY_COMPLEX64, 1)
    
    lwork = -1
    lrwork = -1
    liwork = -1
    work = np.PyArray_EMPTY(1, [1], np.NPY_COMPLEX64, 1)
    rwork = np.PyArray_EMPTY(1, [1], np.NPY_FLOAT32, 1)
    iwork = np.PyArray_EMPTY(1, [1], np.NPY_INT32, 1)
    
    with nogil:
        cheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)
       
    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")
    
    lwork = int(work[0].real)
    lrwork = int(rwork[0])
    liwork = int(iwork[0])
    
    return lwork, lrwork, liwork


@boundscheck(False)
@wraparound(False)
def _cheevr(np.ndarray[np.complex64_t, ndim=2, mode='fortran'] a, int lwork, int lrwork, int liwork, jobz='N', range='A', int il=0, int iu=0, np.float32_t vl=-1e30, np.float32_t vu=0):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int n = a.shape[0]
        int lda = n
        int ldz = n
        int m
        int info
        np.ndarray[np.complex64_t, ndim=2, mode='fortran'] z
        np.ndarray[np.float32_t, ndim=1] w
        np.ndarray[np.complex64_t, ndim=1] work
        np.ndarray[np.float32_t, ndim=1] rwork
        np.ndarray[int, ndim=1] iwork
        np.ndarray[int, ndim=1] isuppz
        np.float32_t abs_tol = -1

    if a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix must be square")

    if range == 'I':
        w = np.PyArray_EMPTY(1, [iu-il+1], np.NPY_FLOAT32, 1)
    else:
        w = np.PyArray_EMPTY(1, [n], np.NPY_FLOAT32, 1)
    isuppz = np.PyArray_EMPTY(1, [2*n], np.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            z = np.PyArray_EMPTY(2, [n, iu-il+1], np.NPY_COMPLEX64, 1)
        else:
            z = np.PyArray_EMPTY(2, [n, n], np.NPY_COMPLEX64, 1)
    else:
        z = np.PyArray_EMPTY(2, [1, 1], np.NPY_COMPLEX64, 1)
    
    work = np.PyArray_EMPTY(1, [lwork], np.NPY_COMPLEX64, 1)
    rwork = np.PyArray_EMPTY(1, [lrwork], np.NPY_FLOAT32, 1)
    iwork = np.PyArray_EMPTY(1, [liwork], np.NPY_INT32, 1)
    
    with nogil:
        cheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)
        
    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")
        
    if jobz == 'V':
        return w[:m], z[:, :m]
    else:
        return w[:m]


@boundscheck(False)
@wraparound(False)
def _cutmud(np.ndarray[np.complex64_t, ndim=2, mode="fortran"] U,
                          np.ndarray[np.complex64_t, ndim=2, mode="fortran"] M) -> np.ndarray[np.float32_t]:
    cdef int m = U.shape[0]
    cdef int n = U.shape[1]
    cdef np.npy_intp *dims = [m, n]
    cdef np.npy_intp *dim = [n]
    cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] C_diag = np.PyArray_EMPTY(1, dim, np.NPY_FLOAT64, 1)
    cdef np.ndarray[np.complex64_t, ndim=2, mode="fortran"] B = np.PyArray_EMPTY(2, dims, np.NPY_COMPLEX128, 1)

    cdef int i
    cdef np.complex64_t alpha = 1.0 + 0.0j
    cdef np.complex64_t beta = 0.0 + 0.0j
    cdef int one = 1

    with nogil:
        chemm(b"L", b"U", &m, &n, &alpha, &M[0, 0], &m, &U[0, 0], &m, &beta, &B[0, 0], &m)
        for i in range(n):
                C_diag[i] = cdotc(&n, &U[0, i], &one, &B[0, i], &one).real

    return C_diag


@boundscheck(False)
@wraparound(False)
def _cdot3d(np.ndarray[np.complex64_t, ndim=3] M, np.ndarray[np.float32_t, ndim=1] V) -> np.ndarray[np.complex64_t]:
    cdef int m = M.shape[1]
    cdef int n = M.shape[2]
    cdef np.npy_intp *dims = [m, n]
    cdef np.ndarray[np.complex64_t, ndim=2] result = np.PyArray_ZEROS(2, dims, np.NPY_COMPLEX64, 0)

    cdef int one = 1
    cdef np.complex64_t alpha0 = V[0] + 0.0j
    cdef np.complex64_t alpha1 = V[1] + 0.0j
    cdef np.complex64_t alpha2 = V[2] + 0.0j
    cdef int size = m * n

    with nogil:
        caxpy(&size, &alpha0, &M[0, 0, 0], &one, &result[0, 0], &one)
        caxpy(&size, &alpha1, &M[1, 0, 0], &one, &result[0, 0], &one)
        caxpy(&size, &alpha2, &M[2, 0, 0], &one, &result[0, 0], &one)

    return result