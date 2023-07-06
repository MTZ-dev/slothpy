import numpy as np
import multiprocessing
from slothpy.magnetism.magnetisation import (mth, calculate_magnetization)
from slothpy.general_utilities.math_expresions import finite_diff_stencil
from slothpy.general_utilities.system import get_num_of_processes
from slothpy.general_utilities.io import get_soc_momenta_and_energies_from_hdf5

def chitht(filename: str, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, exp: bool = False, T: bool = True, grid: np.ndarray = None) -> np.ndarray:
    """
    Calculates chiT(H,T) using data from a HDF5 file for given field, states cutoff, temperatures, and optional grid (XYZ if not present).

    Args:
        path (str): Path to the file.
        hdf5_file (str): Name of the HDF5 file.
        field (np.ndarray[np.float64]): Array of fields.
        states_cutoff (int): Number of states cutoff value.
        temperatures (np.ndarray[np.float64]): Array of temperatures.
        num_cpu (int): Number of CPU used for to call mth function for M(T,H) calculation
        grid (np.ndarray[np.float64], optional): Grid array for direction averaging. Defaults to XYZ.

    Returns:
        np.ndarray[np.float64]: Array of chit values.

    """

    if num_of_points < 0 or (not isinstance(num_of_points, int)):

        raise ValueError(f'Number of points for finite difference method has to be a possitive integer!')
    
    bohr_magneton_to_cm3 = 0.5584938904 # Conversion factor for chi in cm3
    
    # Comments here modyfied!!!!
    chitht_array = np.zeros((fields.shape[0], temperatures.shape[0]))

    # Default XYZ grid
    if grid is None or grid == None:
        grid = np.array([[1., 0., 0., 0.3333333333333333], [0., 1., 0., 0.3333333333333333], [0., 0., 1., 0.3333333333333333]], dtype=np.float64)

    # Experimentalist model
    if (exp == True) or (num_of_points == 0):

        for index_field, field in enumerate(fields):

            mth_array = mth(filename, group, states_cutoff, np.array([field]), grid, temperatures, num_cpu)

            if T:
                for index, temp in enumerate(temperatures):
                    chit[index] = temp * mth_array[index] / field
            else:
                chit = mth_array / field
            
            chitht_array[index_field, :] = chit * bohr_magneton_to_cm3

    else:

        for index_field, field in enumerate(fields):

            # Set fields for finite difference method
            fields_diff = np.arange(-num_of_points, num_of_points + 1).astype(np.int64) * delta_h + field
            fields_diff = fields_diff.astype(np.float64)

            # Initialize result array
            chit = np.zeros_like(temperatures)

            # Get M(t,H) for two adjacent values of field
            mth_array = mth(filename, group, states_cutoff, fields_diff, grid, temperatures, num_cpu)

            stencil_coeff = finite_diff_stencil(1, num_of_points, delta_h)

            if T:
                # Numerical derivative of M(T,H) around given field value 
                for index, temp in enumerate(temperatures):
                    chit[index] = temp * np.dot(mth_array[index], stencil_coeff)
            else:
                for index in range(temperatures.shape[0]):
                    chit[index] = np.dot(mth_array[index], stencil_coeff)
            
            chitht_array[index_field, :] = chit * bohr_magneton_to_cm3

    return chitht_array


def calculate_chi_grid(magnetic_momenta, soc_energies, field, temperatures, num_of_points, delta_h):

    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T
    bohr_magneton_to_cm3 = 0.5584938904 # Conversion factor for chi in cm3

    # Initialize the result array for M(T,H)
    mth = np.zeros((temperatures.shape[0], 2 * num_of_points + 1))
    chi = np.zeros_like(temperatures)

    # Set fields for finite difference method
    fields = np.arange(-num_of_points, num_of_points + 1).astype(np.int64) * delta_h + field
    fields = fields.astype(np.float64)

    # Initialize arrays as contiguous
    magnetic_momenta = np.ascontiguousarray(magnetic_momenta)
    soc_energies = np.ascontiguousarray(soc_energies)

    # Iterate over field values for finite difference method
    for i in range(fields.shape[0]):

        # Construct Zeeman matrix
        zeeman_matrix = -fields[i] * bohr_magneton * magnetic_momenta[0]

        # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
        for k in range(zeeman_matrix.shape[0]):
            zeeman_matrix[k, k] += soc_energies[k]

        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
        eigenvalues = np.ascontiguousarray(eigenvalues)
        eigenvectors = np.ascontiguousarray(eigenvectors)

        # Transform momenta according to the new eigenvectors
        states_momenta = eigenvectors.conj().T @ magnetic_momenta[1] @ eigenvectors

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


def arg_iter_chi_tensor(magnetic_momenta, soc_energies, field, temperatures, num_of_points, delta_h):
    
    for i in range(3):
        for j in range(3):
            yield (np.array([magnetic_momenta[i], magnetic_momenta[j]]), soc_energies, field, temperatures, num_of_points, delta_h)


def chit_tensorht(filename: str, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, T: bool = True):

    chi_tensor_array = np.zeros((fields.shape[0],temperatures.shape[0],3,3), dtype=np.float64)

    for index, field in enumerate(fields):

        # Get number of parallel proceses to be used
        num_process = get_num_of_processes(num_cpu)

        # Read data from HDF5 file
        magnetic_momenta, soc_energies = get_soc_momenta_and_energies_from_hdf5(filename, group, states_cutoff)

        # Parallel M(T,H) calculation over different grid points
        with multiprocessing.Pool(num_process) as p:
            chi = p.map(calculate_chi_grid_wrapper, arg_iter_chi_tensor(magnetic_momenta, soc_energies, field, temperatures, num_of_points, delta_h))

        # Collect results in (3,3) tensor
        chi_reshape = np.array(chi).reshape((3,3,temperatures.shape[0]))
        sus_tensor = np.transpose(chi_reshape, axes=(2, 0, 1))

        if T:
            sus_tensor = sus_tensor * temperatures[:, np.newaxis, np.newaxis]

        chi_tensor_array[index,:,:,:] = sus_tensor[:,:,:]

    return chi_tensor_array

