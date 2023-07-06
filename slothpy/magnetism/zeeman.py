import numpy as np
import multiprocessing
from numba import jit
from slothpy.general_utilities.system import get_num_of_processes
from slothpy.general_utilities.io import get_soc_momenta_and_energies_from_hdf5


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


def zeeman_splitting(filename: str, group: str, states_cutoff: int, num_of_states: int, fields: np.ndarray, grid: np.ndarray, num_cpu: int, average: bool = False) -> np.ndarray:
    
    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu)

    # Initialize the result array
    zeeman_array = np.zeros((grid.shape[0], fields.shape[0], num_of_states), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_momenta_and_energies_from_hdf5(filename, group, states_cutoff)

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

    magnetic_momenta, soc_energies  = get_soc_momenta_and_energies_from_hdf5(filename, group, states_cutoff)
    zeeman_matrix = calculate_zeeman_matrix(magnetic_momenta, soc_energies, field, orientation)

    return zeeman_matrix