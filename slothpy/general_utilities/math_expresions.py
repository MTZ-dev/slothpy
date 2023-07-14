import numpy as np
from numba import jit
from math import factorial

@jit('float64(float64, float64)', nopython=True, cache=True, nogil=True)
def binom(n, k):
    if k > n - k:
        k = n - k
    res = 1
    for i in range(k):
        res *= (n - i)
        res /= (i + 1)
    return res

@jit('float64(float64, float64, float64, float64, float64, float64)', nopython=True, cache=True, nogil=True)
def Clebsh_Gordan(j1,m1,j2,m2,j3,m3):

    cg_coeff = 0

    if (m1 + m2 != m3) or (j1 < 0.0) or (j2 < 0.0) or (j3 < 0.0) or np.abs(m1) > j1 or np.abs(m2) > j2 or np.abs(m3) > j3 or (np.abs(j1 - j2) > j3) or ((j1 + j2) < j3) or (np.abs(j2 - j3) > j1) or ((j2 + j3) < j1) or (np.abs(j3 - j1) > j2) or ((j3 + j1) < j2) or (np.mod(int(2.0 * j1), 2) != np.mod(int(2.0 * np.abs(m1)), 2)) or (np.mod(int(2.0 * j2), 2) != np.mod(int(2.0 * np.abs(m2)), 2)) or (np.mod(int(2.0 * j3), 2) != np.mod(int(2.0 * np.abs(m3)), 2)):
        return cg_coeff

    J = j1 + j2 + j3
    C = np.sqrt(binom(2*j1,J-2*j2)*binom(2*j2,J-2*j3)/(binom(J+1,J-2*j3)*binom(2*j1,j1-m1)*binom(2*j2,j2-m2)*binom(2*j3,j3-m3)))
    z_min = np.max(np.array([0,j1-m1-J+2*j2,j2+m2-J+2*j1]))
    z_max = np.min(np.array([J-2*j3,j1-m1,j2+m2]))
    for z in range(z_min,z_max+1):
        cg_coeff  += (-1)**z * binom(J-2*j3,z) * binom(J-2*j2,j1-m1-z) * binom(J-2*j1,j2+m2-z)
    
    return cg_coeff * C

@jit('float64(float64, float64, float64, float64, float64, float64)', nopython=True, cache=True, nogil=True)
def Wigner_3j(j1, j2, j3, m1, m2, m3):

    return (-1)**(j1 - j2 - m3)/np.sqrt(2*j3 + 1) * Clebsh_Gordan(j1, m1, j2, m2, j3, -m3)

def finite_diff_stencil(diff_order: int, num_of_points: int, step: np.float64):

    stencil_len = 2 * num_of_points + 1

    if diff_order >= stencil_len:
        raise ValueError(f"Insufficient number of points to evaluate coefficients. Provide number of points greater than (derivative order - 1) / 2.")
    
    stencil_matrix = np.tile(np.arange(-num_of_points, num_of_points + 1).astype(np.int64), (stencil_len,1))
    stencil_matrix = stencil_matrix ** np.arange(0, stencil_len).reshape(-1, 1)

    order_vector = np.zeros(stencil_len)
    order_vector[diff_order] = factorial(diff_order)/np.power(step, diff_order)

    stencil_coeff = np.linalg.inv(stencil_matrix) @ order_vector.T

    return stencil_coeff


def hermitian_x_in_basis_of_hermitian_y(x_matrix, y_matrix):
   
   _, eigenvectors = np.linalg.eigh(y_matrix)

   return eigenvectors.conj().T @ x_matrix @ eigenvectors


def decomposition_of_hermitian_matrix(matrix):

    _, eigenvectors = np.linalg.eigh(matrix)

    return (eigenvectors * eigenvectors.conj()).real.T * 100


def normalize_grid_vectors(grid):
    pass