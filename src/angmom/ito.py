import numpy as np
from numba import jit
import os
import h5py
import math


@jit('float64(float64)', nopython=True, cache=True, nogil=True)
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


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


#@jit('float64[:,:](float64, float64, float64)', nopython=True, cache=True, nogil=True)
def ITO_matrix(J,k,q):

    dim = np.int64(2*J + 1)

    matrix = np.zeros((dim, dim), dtype = np.float64)

    for i in range(dim):

        mj = i - J
        v = np.int64(i+q)

        if v >= 0 and v < dim:
            matrix[v,i] = (-1)**(J-mj+q) * Wigner_3j(J, k, J, -mj-q, q, mj)

    coeff = np.float64(1.0)

    for i in range(-k, k+1):
            coeff *= (2*J + 1 + i)
            
    coeff /= math.factorial(2*k)
    coeff = np.sqrt(coeff)
    coeff *= ((-1)**k) * math.factorial(k)

    if True:
        N_k_k = ((-1)**k)/(2**(k/2))

    return matrix * N_k_k * coeff


def calculate_B_k_q(matrix: np.ndarray, k: np.int32, q: np.int32):

    J = (matrix.shape[0] - 1)/2

    matrix = np.ascontiguousarray(matrix)
    ITO_plus = ITO_matrix(J, k, q)
    ITO_plus = np.ascontiguousarray(ITO_plus).astype(np.complex128)
    ITO_minus = ITO_matrix(J, k, -q)
    ITO_minus = np.ascontiguousarray(ITO_minus).astype(np.complex128)

    numerator = np.trace(matrix @ ITO_minus)
    denominator = np.trace(ITO_plus @ ITO_minus)

    return numerator/denominator


def get_SOC_matrix_in_J_basis(path: str, hdf5_file: str, num_of_states: int):

    ge = 2.00231930436256

    # Construct input file path
    input_file = os.path.join(path, hdf5_file)

    # Read data from HDF5 file
    with h5py.File(input_file, 'r') as file:
        soc_matrix = file['ORCA']['SOC'][:]
        sx = 0.5 * file['ORCA']['SX'][:]
        lx = 1j * file['ORCA']['LX'][:]
        sy = 0.5j * file['ORCA']['SY'][:]
        ly = 1j * file['ORCA']['LY'][:]
        sz = 0.5 * file['ORCA']['SZ'][:]
        lz = 1j * file['ORCA']['LZ'][:]

    # Perform diagonalization on SOC matrix
    soc_energies, eigenvectors = np.linalg.eigh(soc_matrix)
    soc_energies = np.ascontiguousarray(soc_energies.astype(np.float64))
    eigenvectors = np.ascontiguousarray(eigenvectors.astype(np.complex128))

    # Apply transformations to spin and orbital operators
    sx = eigenvectors.conj().T @ sx.astype(np.complex128) @ eigenvectors
    sy = eigenvectors.conj().T @ sy.astype(np.complex128) @ eigenvectors
    sz = eigenvectors.conj().T @ sz.astype(np.complex128) @ eigenvectors
    lx = eigenvectors.conj().T @ lx.astype(np.complex128) @ eigenvectors
    ly = eigenvectors.conj().T @ ly.astype(np.complex128) @ eigenvectors
    lz = eigenvectors.conj().T @ lz.astype(np.complex128) @ eigenvectors

    # Slice arrays based on states_cutoff
    sx = sx[:num_of_states, :num_of_states]
    sy = sy[:num_of_states, :num_of_states]
    sz = sz[:num_of_states, :num_of_states]
    lx = lx[:num_of_states, :num_of_states]
    ly = ly[:num_of_states, :num_of_states]
    lz = lz[:num_of_states, :num_of_states]
    soc_energies = (soc_energies[:num_of_states] - soc_energies[0]) * 219474.6

    #TO DO implement rotation of Jz

    Jz = lz + sz

    # Perform diagonalization on Jz matrix
    _, eigenvectors = np.linalg.eigh(Jz)
    eigenvectors = np.ascontiguousarray(eigenvectors.astype(np.complex128))

    # Apply transformations to SOC_matrix
    soc_matrix = eigenvectors.conj().T @ np.diag(soc_energies).astype(np.complex128) @ eigenvectors

    eigenvalues, eigenvectors = np.linalg.eigh(soc_matrix)

    return soc_matrix #-(lz + ge * sz)



def ITO_complex_decomp_matrix(matrix: np.ndarray, order: int):

    result = []

    for k in range(0,order+1):
        for q in range(-k,k+1):
                B_k_q = calculate_B_k_q(matrix, k, q)
                result.append([k, q, B_k_q])
    
    return result


def matrix_from_ITO(J, coefficients):

    dim = np.int64(2*J + 1)

    matrix = np.zeros((dim, dim), dtype=np.complex128)

    for i in coefficients:
        matrix += ITO_matrix(J, i[0], i[1]) * i[2]

    return matrix


hamiltonian = get_SOC_matrix_in_J_basis('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 16)

eigenvalues_ham = np.linalg.eigvalsh(hamiltonian)

decomposition = ITO_complex_decomp_matrix(hamiltonian, 14)

# print(decomposition)

matrix = matrix_from_ITO(15/2, decomposition)

eigenvalues_mat = np.linalg.eigvalsh(matrix)

print(eigenvalues_ham - eigenvalues_mat)

print(hamiltonian - matrix)
