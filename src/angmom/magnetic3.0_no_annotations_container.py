import os
os.environ['OMP_NUM_THREADS'] = '2'
import multiprocessing
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit, cfunc
import timeit
from mpl_toolkits.mplot3d import Axes3D
import math


#TO DO - print the elipsoid of main magnetic axes
# None passing



def get_soc_energies_cm_1(path: str, hdf5_file: str, num_of_states: int = None) -> np.ndarray:

    hartree_to_cm_1 = 219474.6 #atomic units to wavenumbers

    # Construct input file path
    input_file = os.path.join(path, hdf5_file)

    # Read data from HDF5 file
    with h5py.File(input_file, 'r') as file:
        soc_matrix = file['ORCA']['SOC'][:]

    # Perform diagonalization on SOC matrix
    soc_energies = np.linalg.eigvalsh(soc_matrix)

    # Set frist state to zero energy
    soc_energies = (soc_energies - soc_energies[0]) * hartree_to_cm_1

    # Get the first num_of_states
    if num_of_states is not None:
        soc_energies = soc_energies[:num_of_states]

    return soc_energies


def get_states_magnetic_momenta(path: str, hdf5_file: str, states: np.ndarray, J_moment: bool = False):
    
    # Convert states to ndarray without repetitions
    states = np.unique(np.array(states).astype(np.int64))

    # Number of states desired
    num_of_states = states.size

    # Construct input file path
    input_file = os.path.join(path, hdf5_file)

    # Check matrix size
    with h5py.File(input_file, 'r') as file:
        dataset = file['ORCA']['SOC']
        shape = dataset.shape[0]
    
    if num_of_states > shape:
        raise ValueError(f'States cutoff is larger than the number of SO-states ({shape}). Please set it less or equal.')

    ge = 2.00231930436256 #Electron g factor

    #  Initialize the result array
    magnetic_moment = np.ascontiguousarray(np.zeros((3,num_of_states), dtype=np.float64))

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
    _, eigenvectors = np.linalg.eigh(soc_matrix)
    eigenvectors = np.ascontiguousarray(eigenvectors.astype(np.complex128))

    # Apply transformations to spin and orbital operators
    sx = eigenvectors.conj().T @ sx.astype(np.complex128) @ eigenvectors
    sy = eigenvectors.conj().T @ sy.astype(np.complex128) @ eigenvectors
    sz = eigenvectors.conj().T @ sz.astype(np.complex128) @ eigenvectors
    lx = eigenvectors.conj().T @ lx.astype(np.complex128) @ eigenvectors
    ly = eigenvectors.conj().T @ ly.astype(np.complex128) @ eigenvectors
    lz = eigenvectors.conj().T @ lz.astype(np.complex128) @ eigenvectors

    # Slice arrays based on states_cutoff
    sx = sx[states, states]
    sy = sy[states, states]
    sz = sz[states, states]
    lx = lx[states, states]
    ly = ly[states, states]
    lz = lz[states, states]

    if J_moment:
        # Compute total momenta
        magnetic_moment[0] =  (sx + lx).real
        magnetic_moment[1] =  (sy + ly).real
        magnetic_moment[2] =  (sz + lz).real

    else:
        # Compute and save magnetic momenta in a.u.
        magnetic_moment[0] =  -(ge * sx + lx).real
        magnetic_moment[1] =  -(ge * sy + ly).real
        magnetic_moment[2] =  -(ge * sz + lz).real

    return magnetic_moment





#add here jit
def calculate_zeeman_splitting(magnetic_moment: np.ndarray, soc_energies: np.ndarray, field: np.float64, grid: np.ndarray, num_of_states: int):

    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T
    hartree_to_cm_1 = 219474.6 #atomic units to wavenumbers

    # Initialize arrays and scale energy to the ground SOC state
    zeeman_array = np.zeros((grid.shape[0], num_of_states), dtype=np.float64)
    magnetic_moment = np.ascontiguousarray(magnetic_moment)
    soc_energies = np.ascontiguousarray(soc_energies - soc_energies[0])

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        # Construct Zeeman matrix
        orient = -field * bohr_magneton * grid[j, :3]
        zeeman_matrix = magnetic_moment[0] * orient[0] + magnetic_moment[1] * orient[1] + magnetic_moment[2] * orient[2]

        # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
        for k in range(zeeman_matrix.shape[0]):
            zeeman_matrix[k, k] += soc_energies[k]

        # Diagonalize full Zeeman Hamiltonian
        energies = np.linalg.eigvalsh(zeeman_matrix)

        # Get only desired number of states in cm-1
        energies = energies[:num_of_states] * hartree_to_cm_1

        # Collect the results
        zeeman_array[j,:] = energies 
    
    return zeeman_array


def caculate_zeeman_splitting_wrapper(args):

    zeeman_array = calculate_zeeman_splitting(*args)

    return zeeman_array


def arg_iter_zeeman(magnetic_moment, soc_energies, fields, grid, num_of_states):
    
    # Iterator generator for arguments with different field values to be distributed along num_process processes
    for i in range(fields.shape[0]):
      yield (magnetic_moment, soc_energies, fields[i], grid, num_of_states)


def zeeman_splitting(path: str, hdf5_file: str, states_cutoff: int, num_of_states: int, fields: np.ndarray, grid: np.ndarray, num_cpu: int, average: bool = False) -> np.ndarray:
    
    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu)

    # Initialize the result array
    zeeman_array = np.zeros((grid.shape[0], fields.shape[0], num_of_states), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_moment_energies_from_hdf5_orca(path, hdf5_file, states_cutoff)

    # Parallel M(T) calculation over different field values
    with multiprocessing.Pool(num_process) as p:
        zeeman = p.map(caculate_zeeman_splitting_wrapper, arg_iter_zeeman(magnetic_moment, soc_energies, fields, grid, num_of_states))

    # Collecting results
    for i in range(fields.shape[0]):
        zeeman_array[:,i,:] = zeeman[i]

    # Average over directions
    if average == True:

        zeeman_array_av = np.zeros((fields.shape[0], num_of_states), dtype=np.float64)
        for i in range(fields.shape[0]):
            for j in range(grid.shape[0]):
                zeeman_array_av[i, :] += zeeman_array[j, i, :] * grid[j, 3]

        return zeeman_array_av

    return zeeman_array


@jit('float64(complex128[:,:,:], float64[:], float64, float64[:], float64)', nopython=True, cache=True, nogil=True)
def calculate_mth_grid(magnetic_moment: np.ndarray, soc_energies: np.ndarray, field: np.float64, grid: np.ndarray, temperature: np.float64) -> np.float64:

    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T

    # Initialize arrays as contiguous
    magnetic_moment = np.ascontiguousarray(magnetic_moment)
    soc_energies = np.ascontiguousarray(soc_energies)


    # Construct Zeeman matrix
    orient = -field * bohr_magneton * grid[:3]
    zeeman_matrix = magnetic_moment[0] * orient[0] + magnetic_moment[1] * orient[1] + magnetic_moment[2] * orient[2]

    # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
    for k in range(zeeman_matrix.shape[0]):
        zeeman_matrix[k, k] += soc_energies[k]

    # Diagonalize full Hamiltonian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
    eigenvalues = np.ascontiguousarray(eigenvalues)
    eigenvectors = np.ascontiguousarray(eigenvectors)

    # Transform momenta according to the new eigenvectors
    states_momenta = eigenvectors.conj().T @ (
        grid[0] * magnetic_moment[0]
        + grid[1] * magnetic_moment[1]
        + grid[2] * magnetic_moment[2]
    ) @ eigenvectors

    # Get diagonal momenta of the new states
    states_momenta = np.diag(states_momenta).real.astype(np.float64)

    # Compute partition function and magnetization
    mth = calculate_magnetization(eigenvalues, states_momenta, temperature)

    return mth


def calculate_mth_grid_wrapper(args):

    # Unpack arguments and call the function
    mth = calculate_mth_grid(*args)

    return mth


def arg_iter_mag_3d(magnetic_moment, soc_energies, field, theta, phi, temperature):
    
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            yield (magnetic_moment, soc_energies, field, np.array([np.sin(phi[i, j]) * np.cos(theta[i, j]), np.sin(phi[i, j]) * np.sin(theta[i, j]), np.cos(phi[i, j])]), temperature)


def mag_3d(path: str, hdf5_file: str, states_cutoff: int, field: np.ndarray, grid: int, temperature: np.float64, num_cpu: int) -> np.ndarray:

    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu)

    # Create a gird
    theta = np.linspace(0, 2*np.pi, grid)
    phi = np.linspace(0, np.pi, grid)
    theta, phi = np.meshgrid(theta, phi)

    # Initialize the result array
    mag_3d_array = np.zeros_like(phi, dtype=np.float64)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_moment_energies_from_hdf5_orca(path, hdf5_file, states_cutoff)

    # Parallel M(T,H) calculation over different grid points
    with multiprocessing.Pool(num_process) as p:
        mth = p.map(calculate_mth_grid_wrapper, arg_iter_mag_3d(magnetic_moment, soc_energies, field, theta, phi, temperature))

    index = 0

    # Collecting results
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            mag_3d_array[i, j] = mth[index]
            index += 1

    x = np.sin(phi) * np.cos(theta) * mag_3d_array
    y = np.sin(phi) * np.sin(theta) * mag_3d_array
    z = np.cos(phi) * mag_3d_array

    return x, y, z

# zastanów się czy numerycznie to robić też
def chi_exp_3d(path: str, hdf5_file: str, states_cutoff: int, field: np.ndarray, grid: int, temperature: np.float64, num_cpu: int) -> np.ndarray:

    x, y, z = mag_3d(path, hdf5_file, states_cutoff, field, grid, temperature, num_cpu)

    x /= field
    y /= field
    z /= field

    return x, y, z
    

def calculate_chi_grid(magnetic_moment, soc_energies, field, temperatures, num_of_points, delta_h):

    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T
    bohr_magneton_to_cm3 = 0.5584938904 # Conversion factor for chi in cm3

    # Initialize the result array for M(T,H)
    mth = np.zeros((temperatures.shape[0], 2 * num_of_points + 1))
    chi = np.zeros_like(temperatures)

    # Set fields for finite difference method
    fields = np.arange(-num_of_points, num_of_points + 1).astype(np.int64) * delta_h + field
    fields = fields.astype(np.float64)

    # Initialize arrays as contiguous
    magnetic_moment = np.ascontiguousarray(magnetic_moment)
    soc_energies = np.ascontiguousarray(soc_energies)

    # Iterate over field values for finite difference method
    for i in range(fields.shape[0]):

        # Construct Zeeman matrix
        zeeman_matrix = -fields[i] * bohr_magneton * magnetic_moment[0]

        # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
        for k in range(zeeman_matrix.shape[0]):
            zeeman_matrix[k, k] += soc_energies[k]

        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
        eigenvalues = np.ascontiguousarray(eigenvalues)
        eigenvectors = np.ascontiguousarray(eigenvectors)

        # Transform momenta according to the new eigenvectors
        states_momenta = eigenvectors.conj().T @ magnetic_moment[1] @ eigenvectors

        # Get diagonal momenta of the new states
        states_momenta = np.diag(states_momenta).real.astype(np.float64)

        # Compute partition function and magnetization
        for t in range(temperatures.shape[0]):
            mth[t, i] = calculate_magnetization(eigenvalues, states_momenta, temperatures[t])

    stencil_coeff = finite_diff_stencil(1, num_of_points, delta_h)

    # Numerical derivative of M(T,H) around given field value 
    for t in range(temperatures.shape[0]):
        chi[t] = np.dot(mth[t, :], stencil_coeff)

    return chi * bohr_magneton_to_cm3


def calculate_chi_grid_wrapper(args):

    # Unpack arguments and call the function
    chi = calculate_chi_grid(*args)

    return chi


def arg_iter_chi_tensor(magnetic_moment, soc_energies, field, temperatures, num_of_points, delta_h):
    
    for i in range(3):
        for j in range(3):
            yield (np.array([magnetic_moment[i], magnetic_moment[j]]), soc_energies, field, temperatures, num_of_points, delta_h)



def chi_tensor(path: str, hdf5_file: str, field: np.float64, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64):

    # Initialize the result array
    sus_tensor = np.zeros((3,3,temperatures.shape[0]), dtype=np.float64)

    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_moment_energies_from_hdf5_orca(path, hdf5_file, states_cutoff)

    # Parallel M(T,H) calculation over different grid points
    with multiprocessing.Pool(num_process) as p:
        chi = p.map(calculate_chi_grid_wrapper, arg_iter_chi_tensor(magnetic_moment, soc_energies, field, temperatures, num_of_points, delta_h))

    # Collect results in (3,3) tensor
    chi_reshape = np.array(chi).reshape((3,3,temperatures.shape[0]))
    sus_tensor = np.transpose(chi_reshape, axes=(2, 0, 1))

    return sus_tensor


def calculate_B_k_q(matrix: np.ndarray, k: np.int32, q: np.int32):

    matrix = np.ascontiguousarray(matrix)

    J = (matrix.shape[0] - 1)/2

    numerator = np.complex128(0)
    denominator = np.complex128(0)

    for i in range(int(2*J+1)):
        for j in range(int(2*J+1)):
            numerator += matrix[i,j] * ((-1)**(q)) * Clebsh_Gordan(J, J - i, k, -q, J, J - j)
            denominator += Clebsh_Gordan(J, J - j, k, q, J, J - i) * Clebsh_Gordan(J, J - i, k, -q, J, J - j) 

    B = 1.0

    for i in range(-k, k+1):
            B = B * (2*J + 1 + i)
            
    B = B / ((2**k) * math.factorial(2*k))
    B = math.sqrt(B)
    B = B * math.factorial(k)

    #denominator = (2*J + 1) / (2*k + 1)

    return numerator * (-1)**(k)/np.sqrt(2*k + 1)/denominator/B



def denom_check(J: int, k: np.int32, q: np.int32):

    denominator = np.complex128(0)

    for i in range(int(2*J+1)):
        for j in range(int(2*J+1)):
            denominator += Clebsh_Gordan(J, J - j, k, q, J, J - i) * Clebsh_Gordan(J, J - i, k, -q, J, J - j)

    print(denominator)

    denominator1 = (2*J + 1) / (2*k + 1)

    print(denominator1)

    return denominator1 - denominator


def ITO_decomp_matrix(matrix: np.ndarray, order: int):

    for k in range(0,order+1):
        for q in range(-k,k+1):
            if q >= 0:
                B_k_q = ((-1)**q * calculate_B_k_q(matrix, k, q) + calculate_B_k_q(matrix, k, -q))
                print(f"{k} {q} {B_k_q.real}")
            if q < 0:
                B_k_q = -1j * (-(-1)**q * calculate_B_k_q(matrix, k, q) + calculate_B_k_q(matrix, k, -q))
                print(f"{k} {q} {B_k_q.real}")


def ITO_complex_decomp_matrix(matrix: np.ndarray, order: int):

    for k in range(0,order+1):
        for q in range(-k,k+1):
                B_k_q = calculate_B_k_q(matrix, k, q)
                print(f"{k} {q} {B_k_q}")


def matrix_from_ITO(filename, J):

    dim = int(2*J + 1)

    matrix = np.zeros((dim, dim),dtype=np.complex128)

    ITO = np.loadtxt(filename, dtype = np.complex128)

    for cfp in ITO:
        for i in range(dim):
            for j in range(dim):
                        B = 1.0
                        for i in range(-int(cfp[0]), int(cfp[0])+1):
                                B = B * (2*J + 1 + i)        
                        B = B / ((2**int(cfp[0])) * math.factorial(2*int(cfp[0])))
                        B = math.sqrt(B)
                        B = B * math.factorial(int(cfp[0]))

                        matrix[i,j] += ((-1)**(-int(cfp[1])) * Clebsh_Gordan(J, J - i, int(cfp[0]), int(cfp[1]), J, J - j)) * B * (-1)**(-int(cfp[0]))/np.sqrt(2*int(cfp[0])+1) * cfp[2]

    eigenvalues, eigenvectors =  np.linalg.eigh(matrix)

    print(eigenvalues)
    #print(eigenvectors)
    


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
