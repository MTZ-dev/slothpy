import numpy as np
import math
from slothpy.general_utilities.io import (get_soc_momenta_and_energies_from_hdf5, get_soc_total_angular_momenta_and_energies_from_hdf5)
from slothpy.general_utilities.math_expresions import (hermitian_x_in_basis_of_hermitian_y, decomposition_of_hermitian_matrix, Wigner_3j)
from slothpy.magnetism.zeeman import calculate_zeeman_matrix


def get_soc_matrix_in_z_magnetic_momentum_basis(filename, group, start_state, stop_state):

    magnetic_momenta, soc_energies  = get_soc_momenta_and_energies_from_hdf5(filename, group, stop_state+1)
    magnetic_momenta = magnetic_momenta[:][start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    soc_matrix = np.diag(soc_energies)
    soc_matrix = hermitian_x_in_basis_of_hermitian_y(soc_matrix, magnetic_momenta[2])

    return soc_matrix 


def get_soc_matrix_in_z_total_angular_momentum_basis(filename, group, start_state, stop_state):

    total_angular_momenta, soc_energies  = get_soc_total_angular_momenta_and_energies_from_hdf5(filename, group, stop_state+1)
    magnetic_momenta = magnetic_momenta[:][start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    soc_matrix = np.diag(soc_energies)
    soc_matrix = hermitian_x_in_basis_of_hermitian_y(soc_matrix, total_angular_momenta[2])
    
    return soc_matrix 


def get_zeeman_matrix_in_z_magnetic_momentum_basis(filename, group, field, orientation, start_state, stop_state):

    magnetic_momenta, soc_energies  = get_soc_momenta_and_energies_from_hdf5(filename, group, stop_state+1)
    magnetic_momenta = magnetic_momenta[:][start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)
    zeeman_matrix = hermitian_x_in_basis_of_hermitian_y(zeeman_matrix, magnetic_momenta[2])

    return zeeman_matrix


def get_zeeman_matrix_in_z_total_angular_momentum_basis(filename, group, field, orientation, start_state, stop_state):

    total_angular_momenta, soc_energies  = get_soc_total_angular_momenta_and_energies_from_hdf5(filename, group, stop_state+1)
    magnetic_momenta, _ = get_soc_momenta_and_energies_from_hdf5(filename, group, stop_state+1)
    magnetic_momenta = magnetic_momenta[:][start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)
    zeeman_matrix = hermitian_x_in_basis_of_hermitian_y(zeeman_matrix, total_angular_momenta[2])
    
    return zeeman_matrix 


def get_decomposition_in_z_magnetic_momentum_basis(filename, group, number_of_states):

    soc_matrix = get_soc_matrix_in_z_magnetic_momentum_basis(filename, group, number_of_states)
    decopmosition = decomposition_of_hermitian_matrix(soc_matrix)

    return decopmosition


def get_decomposition_in_z_total_angular_momentum_basis(filename, group, number_of_states):
    
    soc_matrix = get_soc_matrix_in_z_total_angular_momentum_basis(filename, group, number_of_states)
    decopmosition = decomposition_of_hermitian_matrix(soc_matrix)

    return decopmosition


#@jit('float64[:,:](float64, float64, float64)', nopython=True, cache=True, nogil=True)
def ito_matrix(J,k,q):

    dim = np.int64(2*J + 1)

    matrix = np.zeros((dim, dim), dtype = np.float64)

    for i in range(dim):

        mj = i - J
        v = np.int64(i+q)

        if v >= 0 and v < dim:
            matrix[v,i] = (-1)**(J-mj+q) * Wigner_3j(J, k, J, -mj-q, q, mj)

    coeff = np.float64(1.0)

    for i in range(int(-k), int(k+1)):
            coeff *= (2*J + 1 + i)
            
    coeff /= math.factorial(2*k)
    coeff = np.sqrt(coeff)
    coeff *= ((-1)**k) * math.factorial(k)

    #you can implement conventions from the article here
    N_k_k = ((-1)**k)/(2**(k/2))

    return matrix * N_k_k * coeff


def calculate_b_k_q(matrix: np.ndarray, k: np.int32, q: np.int32):

    J = (matrix.shape[0] - 1)/2

    matrix = np.ascontiguousarray(matrix)
    ITO_plus = ito_matrix(J, k, q)
    ITO_plus = np.ascontiguousarray(ITO_plus).astype(np.complex128)
    ITO_minus = ito_matrix(J, k, -q)
    ITO_minus = np.ascontiguousarray(ITO_minus).astype(np.complex128)

    numerator = np.trace(matrix @ ITO_minus)
    denominator = np.trace(ITO_plus @ ITO_minus)

    return numerator/denominator


def ito_complex_decomp_matrix(matrix: np.ndarray, order: int, even_order: bool = False):

    step = 1

    if even_order:
        step = 2

    result = []

    for k in range(0,order+1):
        for q in range(-k,k+1):
                B_k_q = calculate_b_k_q(matrix, k, q)
                result.append([k, q, B_k_q])
    
    return result


def matrix_from_ito_complex(J, coefficients):

    dim = np.int64(2*J + 1)

    matrix = np.zeros((dim, dim), dtype=np.complex128)

    for i in coefficients:
        matrix += ito_matrix(J, int(i[0].real), int(i[1].real)) * i[2]

    return matrix



def ito_real_decomp_matrix(matrix: np.ndarray, order: int, even_order: bool = False):

    step = 1

    if even_order:
        step = 2

    result = []

    for k in range(0,order+1, step):
        for q in range(-k,0):
                B_k_q = -1j * (calculate_b_k_q(matrix, k, q) - ((-1)**(-q)) * calculate_b_k_q(matrix, k, -q))
                result.append([k, q, B_k_q.real])

        B_k_q = calculate_b_k_q(matrix, k, 0)
        result.append([k, 0, B_k_q.real])

        for q in range(1,k+1):
            B_k_q = (calculate_b_k_q(matrix, k, -q) + ((-1)**q) * calculate_b_k_q(matrix, k, q))
            result.append([k, q, B_k_q.real])

    return result



def matrix_from_ito_real(J, coefficients):

    dim = np.int64(2*J + 1)

    matrix = np.zeros((dim, dim), dtype=np.complex128)

    for i in coefficients:

        k = np.int64(i[0])
        q = np.int64(i[1])
        
        if q < 0:
            matrix += 1j * 0.5 * (((-1)**(-q+1)) * ito_matrix(J, k, -q) +  ito_matrix(J, k, q)) * i[2]
        if q > 0:
            matrix += 0.5 * (ito_matrix(J, k, -q) +  ((-1)**q) * ito_matrix(J, k, q)) * i[2]
        if q == 0:
            matrix += ito_matrix(J, k, q) * i[2]

    return matrix