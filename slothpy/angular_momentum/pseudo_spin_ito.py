import numpy as np
import math
from slothpy.general_utilities.io import (get_soc_magnetic_momenta_and_energies_from_hdf5, get_soc_total_angular_momenta_and_energies_from_hdf5)
from slothpy.general_utilities.math_expresions import (hermitian_x_in_basis_of_hermitian_y, decomposition_of_hermitian_matrix, Wigner_3j)
from slothpy.magnetism.zeeman import calculate_zeeman_matrix
#from sympy.physics.quantum.cg import (CG, Wigner3j)


def set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(momenta_matrix, matrix):
    """Convention:
     Jx[i,i+1] = real, negative
     Jy[i,i+1] = imag, positive
     J+/- = real (Condon_Shortley)
     Jz = real, diag"""

    # Transform momenta to "z" basis
    _, eigenvectors = np.linalg.eigh(momenta_matrix[2,:,:])
    for i in range(3):
        momenta_matrix[i,:,:] = eigenvectors.conj().T @ momenta_matrix[i,:,:] @ eigenvectors

    # Initialize phases of vectors with the first one = 1
    c = np.zeros(momenta_matrix.shape[1], dtype=np.complex128)
    c[0] = 1.

    # Set Jx[i,i+1] to real negative and collect phases of vectors in c[:]
    for i in range(momenta_matrix[0,:,:].shape[0]-1):
        if np.real(momenta_matrix[0,i,i+1]).any() > 1e-12 or np.abs(np.imag(momenta_matrix[1,i,i+1])).any() > 1e-12:
            c[i+1] = momenta_matrix[0,i,i+1].conj()/np.abs(momenta_matrix[0,i,i+1])/c[i].conj()
        else:
            c[i+1] = 1.
    
    for i in range(momenta_matrix[0].shape[0]-1):
        if momenta_matrix[0,i,i+1] * c[i].conj() * c[i+1] > 0:
            c[i+1] = -c[i+1]
    
    matrix_out = np.zeros_like(matrix)

    for i in range(matrix_out.shape[0]):
        for j in range(matrix_out.shape[0]):
            matrix_out[i,j] = matrix[i,j] * c[i].conj() * c[j]

    return matrix_out


def get_soc_matrix_in_z_magnetic_momentum_basis(filename, group, start_state, stop_state, rotation = None):

    magnetic_momenta, soc_energies  = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, stop_state+1, rotation)
    magnetic_momenta = magnetic_momenta[:, start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    soc_matrix = np.diag(soc_energies).astype(np.complex128)
    soc_matrix = hermitian_x_in_basis_of_hermitian_y(soc_matrix, magnetic_momenta[2,:,:])
    soc_matrix = set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(magnetic_momenta, soc_matrix)
    
    return soc_matrix 


def get_soc_matrix_in_z_total_angular_momentum_basis(filename, group, start_state, stop_state, rotation = None):

    total_angular_momenta, soc_energies  = get_soc_total_angular_momenta_and_energies_from_hdf5(filename, group, stop_state+1, rotation)
    total_angular_momenta = total_angular_momenta[:, start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    soc_matrix = np.diag(soc_energies).astype(np.complex128)
    soc_matrix = hermitian_x_in_basis_of_hermitian_y(soc_matrix, total_angular_momenta[2,:,:])
    soc_matrix = set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(total_angular_momenta, soc_matrix)
    
    return soc_matrix


def get_zeeman_matrix_in_z_magnetic_momentum_basis(filename, group, field, orientation, start_state, stop_state, rotation = None):

    magnetic_momenta, soc_energies  = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, stop_state+1, rotation)
    magnetic_momenta = magnetic_momenta[:, start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)
    zeeman_matrix = hermitian_x_in_basis_of_hermitian_y(zeeman_matrix, magnetic_momenta[2,:,:])
    zeeman_matrix = set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(magnetic_momenta, zeeman_matrix)

    return zeeman_matrix


def get_zeeman_matrix_in_z_total_angular_momentum_basis(filename, group, field, orientation, start_state, stop_state, rotation = None):

    total_angular_momenta, soc_energies  = get_soc_total_angular_momenta_and_energies_from_hdf5(filename, group, stop_state+1)
    magnetic_momenta, _ = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, stop_state+1, rotation)
    magnetic_momenta = magnetic_momenta[:, start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)
    zeeman_matrix = hermitian_x_in_basis_of_hermitian_y(zeeman_matrix, total_angular_momenta[2,:,:])
    zeeman_matrix = set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(total_angular_momenta, zeeman_matrix)
    
    return zeeman_matrix 


def get_decomposition_in_z_magnetic_momentum_basis(filename, group, start_state, stop_state, rotation = None):

    soc_matrix = get_soc_matrix_in_z_magnetic_momentum_basis(filename, group, start_state, stop_state, rotation)
    decopmosition = decomposition_of_hermitian_matrix(soc_matrix)

    return decopmosition


def get_decomposition_in_z_total_angular_momentum_basis(filename, group, start_state, stop_state, rotation = None):
    
    soc_matrix = get_soc_matrix_in_z_total_angular_momentum_basis(filename, group, start_state, stop_state, rotation)
    decopmosition = decomposition_of_hermitian_matrix(soc_matrix)

    return decopmosition


#@jit('float64[:,:](float64, float64, float64)', nopython=True, cache=True, nogil=True)
def ito_matrix(J,k,q):

    dim = np.int64(2*J + 1)

    matrix = np.zeros((dim, dim), dtype = np.float64)

    for i in range(dim):
        mj1 = i - J
        for p in range(dim):
            mj2 = p - J
            matrix[i,p] = (-1)**(J-mj1) * Wigner_3j(J, k, J, -mj1, q, mj2)

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

    for k in range(0,order+1, step):
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



# def ito_real_decomp_matrix(matrix: np.ndarray, order: int, even_order: bool = False):

#     step = 1

#     if even_order:
#         step = 2

#     result = []

#     for k in range(0,order+1, step):
#         for q in range(-k,0):
#                 B_k_q = 1j * 1/np.sqrt(2) * (calculate_b_k_q(matrix, k, q) + ((-1)**(-q)) * calculate_b_k_q(matrix, k, -q))
#                 result.append([k, q, B_k_q.real])

#         B_k_q = calculate_b_k_q(matrix, k, 0)
#         result.append([k, 0, B_k_q.real])

#         for q in range(1,k+1):
#             B_k_q = 1/np.sqrt(2) * (calculate_b_k_q(matrix, k, -q) + (-1)**(q+1)) * (calculate_b_k_q(matrix, k, q))
#             result.append([k, q, B_k_q.real])

#     return result


def ito_real_decomp_matrix(matrix: np.ndarray, order: int, even_order: bool = False):

    step = 1

    if even_order:
        step = 2

    result = []

    J = (matrix.shape[0] - 1)/2
    matrix = np.ascontiguousarray(matrix)

    for k in range(0,order+1, step):
        for q in range(k,0,-1):

            ITO_plus = ito_matrix(J, k, q)
            ITO_plus = np.ascontiguousarray(ITO_plus).astype(np.complex128)
            ITO_minus = ito_matrix(J, k, -q)
            ITO_minus = np.ascontiguousarray(ITO_minus).astype(np.complex128)
            B_k_q = -1j * (np.trace(matrix @ ITO_plus) - ((-1)**(-q)) * np.trace(matrix @ ITO_minus))/np.trace(ITO_plus @ ITO_minus)  #1/np.sqrt(2) *

            result.append([k, -q, B_k_q.real])

        B_k_q = calculate_b_k_q(matrix, k, 0)
        result.append([k, 0, B_k_q.real])

        for q in range(1,k+1):

            ITO_plus = ito_matrix(J, k, q)
            ITO_plus = np.ascontiguousarray(ITO_plus).astype(np.complex128)
            ITO_minus = ito_matrix(J, k, -q)
            ITO_minus = np.ascontiguousarray(ITO_minus).astype(np.complex128)
            B_k_q = (np.trace(matrix @ ITO_plus) + ((-1)**(-q)) * np.trace(matrix @ ITO_minus))/np.trace(ITO_plus @ ITO_minus) #1/np.sqrt(2) *

            result.append([k, q, B_k_q.real])

    return result



# def matrix_from_ito_real(J, coefficients):

#     dim = np.int64(2*J + 1)

#     matrix = np.zeros((dim, dim), dtype=np.complex128)

#     for i in coefficients:

#         k = np.int64(i[0])
#         q = np.int64(i[1])
        
#         if q < 0:
#             matrix += 1j * 1/np.sqrt(2) * (((-1)**(-q+1)) * ito_matrix(J, k, -q) +  ito_matrix(J, k, q)) * i[2]
#         if q > 0:
#             matrix += 1/np.sqrt(2) * (ito_matrix(J, k, -q) +  ((-1)**q) * ito_matrix(J, k, q)) * i[2]
#         if q == 0:
#             matrix += ito_matrix(J, k, q) * i[2]

#     return matrix


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