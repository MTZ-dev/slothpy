import threadpoolctl
import multiprocessing
import numpy as np
from numba import jit
from slothpy.general_utilities.system import get_num_of_processes
from slothpy.general_utilities.io import get_soc_magnetic_momenta_and_energies_from_hdf5


@jit('complex128[:,:](complex128[:,:,:], float64[:], float64, float64[:])', nopython=True, cache=True, nogil=True)
def calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation):

    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T

    orientation = -field * bohr_magneton * orientation
    zeeman_matrix = magnetic_momenta[0] * orientation[0] + magnetic_momenta[1] * orientation[1] + magnetic_momenta[2] * orientation[2]

    # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
    for k in range(zeeman_matrix.shape[0]):
        zeeman_matrix[k, k] += soc_energies[k]

    return zeeman_matrix


@jit('float64[:,:](complex128[:,:,:], float64[:], float64, float64[:,:], int64)', nopython=True, cache=True, nogil=True)
def calculate_zeeman_splitting(magnetic_momenta: np.ndarray, soc_energies: np.ndarray, field: np.float64, grid: np.ndarray, num_of_states: int) -> np.ndarray:

    hartree_to_cm_1 = 219474.6 #atomic units to wavenumbers

    # Initialize arrays and scale energy to the ground SOC state
    zeeman_array = np.zeros((grid.shape[0], num_of_states), dtype=np.float64)
    magnetic_momenta = np.ascontiguousarray(magnetic_momenta)
    soc_energies = np.ascontiguousarray(soc_energies - soc_energies[0])

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):

        orientation = grid[j, :3]

        zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)

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


def zeeman_splitting(filename: str, group: str, states_cutoff: int, num_of_states: int, fields: np.ndarray, grid: np.ndarray, num_cpu: int, num_threads: int, average: bool = False) -> np.ndarray:
    
    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu, num_threads)

    # Initialize the result array
    zeeman_array = np.zeros((grid.shape[0], fields.shape[0], num_of_states), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, states_cutoff)

    with threadpoolctl.threadpool_limits(limits=num_threads, user_api='blas'):
        with threadpoolctl.threadpool_limits(limits=num_threads, user_api='openmp'):
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


def get_zeeman_matrix(filename: str, group: str, states_cutoff: int, field: np.float64, orientation: np.ndarray) -> np.ndarray:

    magnetic_momenta, soc_energies  = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, states_cutoff)
    zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)

    return zeeman_matrix


@jit('float64(float64[:], float64, boolean)', nopython=True, cache=True, nogil=True)
def calculate_hemholtz_energy(energies: np.ndarray, temperature: np.float64, internal_energy: False) -> np.float64:

    kB = 3.166811563e-6  # Boltzmann constant a.u./K
    hartree_to_cm_1 = 219474.6 #atomic units to wavenumbers
    energies = (energies[1:] - energies[0])

    # Boltzman weights
    exp_diff = np.exp(-(energies) / (kB * temperature))

    # Partition function
    z = np.sum(exp_diff)

    if internal_energy:
        e = np.sum((energies * hartree_to_cm_1) * exp_diff)
        return e / z
    else:
        return -kB * temperature * np.log(z) * hartree_to_cm_1


@jit('float64[:](complex128[:,:,:], float64[:], float64, float64[:,:], float64[:], boolean)', nopython=True, cache=True, nogil=True)
def calculate_hemholtz_energyt(magnetic_momenta: np.ndarray, soc_energies: np.ndarray, field: np.float64, grid: np.ndarray, temperatures: np.ndarray, internal_energy: False) -> np.ndarray:

    # Initialize arrays
    energyt_array = np.ascontiguousarray(np.zeros((temperatures.shape[0]), dtype=np.float64))
    magnetic_momenta = np.ascontiguousarray(magnetic_momenta)
    soc_energies = np.ascontiguousarray(soc_energies)

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):

        # Construct Zeeman matrix
        orientation = grid[j, :3]
        
        zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)

        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
        eigenvalues = np.ascontiguousarray(eigenvalues)
        eigenvectors = np.ascontiguousarray(eigenvectors)

        # Compute Hemholtz energy for each T
        for t in range(temperatures.shape[0]):
            energyt_array[t] += (calculate_hemholtz_energy(eigenvalues, temperatures[t], internal_energy) * grid[j, 3])

    return energyt_array


def calculate_hemholtz_energyt_wrapper(args):
    """Wrapper function for parallel use of E(T) calulations

    Args:
        args (tuple): Tuple of arguments for calculate_mt function

    Returns:
        np.ndarray[np.float64]: E(T) array.
    """
    # Unpack arguments and call the function
    et = calculate_hemholtz_energyt(*args)

    return et


def arg_iter_hemholtz_energyth(magnetic_momenta, soc_energies, fields, grid, temperatures, internal_energy: False):
    
    # Iterator generator for arguments with different field values to be distributed along num_process processes
    for i in range(fields.shape[0]):
      yield (magnetic_momenta, soc_energies, fields[i], grid, temperatures, internal_energy)


def hemholtz_energyth(filename: str, group: str, states_cutoff: int, fields: np.ndarray, grid: np.ndarray, temperatures: np.ndarray, num_cpu: int, num_threads: int, internal_energy: False) -> np.ndarray:

    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu, num_threads)

    # Initialize the result array
    eth_array = np.zeros((temperatures.shape[0], fields.shape[0]), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)
    temperatures = np.ascontiguousarray(temperatures)

    # Read data from HDF5 file
    magnetic_momenta, soc_energies = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, states_cutoff)

    with threadpoolctl.threadpool_limits(limits=num_threads, user_api='blas'):
        with threadpoolctl.threadpool_limits(limits=num_threads, user_api='openmp'):
            # Parallel M(T) calculation over different field values
            with multiprocessing.Pool(num_process) as p:
                et = p.map(calculate_hemholtz_energyt_wrapper, arg_iter_hemholtz_energyth(magnetic_momenta, soc_energies, fields, grid, temperatures, internal_energy))

    # Collecting results in plotting-friendly convention for M(H)
    for i in range(fields.shape[0]):
        eth_array[:,i] = et[i]

    return eth_array 


def arg_iter_hemholtz_energy_3d(magnetic_moment, soc_energies, field, theta, phi, temperatures, internal_energy: False):

    field = np.float64(field)
    
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            yield (magnetic_moment, soc_energies, field, np.array([[np.sin(phi[i, j]) * np.cos(theta[i, j]), np.sin(phi[i, j]) * np.sin(theta[i, j]), np.cos(phi[i, j]), 1.]]), temperatures, internal_energy)


def hemholtz_energy_3d(filename: str, group: str, states_cutoff: int, fields: np.ndarray, spherical_grid: int, temperatures: np.ndarray, num_cpu: int, num_threads: int, internal_energy: False) -> np.ndarray:

    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu, num_threads)

    # Create a gird
    theta = np.linspace(0, 2*np.pi, 2 * spherical_grid)
    phi = np.linspace(0, np.pi, spherical_grid)
    theta, phi = np.meshgrid(theta, phi)

    # Initialize the result array
    hemholtz_energy_3d_array = np.zeros((fields.shape[0], temperatures.shape[0], phi.shape[0], phi.shape[1]), dtype=np.float64)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_magnetic_momenta_and_energies_from_hdf5(filename, group, states_cutoff)

    for field_index, field in enumerate(fields):

        with threadpoolctl.threadpool_limits(limits=num_threads, user_api='blas'):
            with threadpoolctl.threadpool_limits(limits=num_threads, user_api='openmp'):
                # Parallel M(T,H) calculation over different grid points
                with multiprocessing.Pool(num_process) as p:
                    eth = p.map(calculate_hemholtz_energyt_wrapper, arg_iter_hemholtz_energy_3d(magnetic_moment, soc_energies, field, theta, phi, temperatures, internal_energy))

        pool_index = 0

        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                hemholtz_energy_3d_array[field_index,:,i,j] = eth[pool_index][:]
                pool_index += 1


    x = (np.sin(phi) * np.cos(theta))[np.newaxis,np.newaxis,:,:] * hemholtz_energy_3d_array
    y = (np.sin(phi) * np.sin(theta))[np.newaxis,np.newaxis,:,:] * hemholtz_energy_3d_array
    z = (np.cos(phi))[np.newaxis,np.newaxis,:,:] * hemholtz_energy_3d_array

    return x, y, z