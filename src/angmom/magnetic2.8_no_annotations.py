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


def get_num_of_processes(num_cpu):

    # Check CPUs number considering the desired number of threads and assign number of processes
    if num_cpu < int(os.getenv('OMP_NUM_THREADS')):
        raise ValueError(f"Insufficient number of CPU cores assigned. Desired threads: {int(os.getenv('OMP_NUM_THREADS'))}, Actual processors: {num_cpu}")
    else:
        num_process = num_cpu//int(os.getenv('OMP_NUM_THREADS'))
    
    return num_process


def grep_to_file(path_inp: str, inp_file: str, pattern: str, path_out: str, out_file: str, lines: int = 1) -> None:
    """
    Extracts lines from an input file that match a specified pattern and saves them to an output file.

    Args:
        path_inp (str): Path to the input file.
        inp_file (str): Name of the input file.
        pattern (str): Pattern to search for in the input file.
        path_out (str): Path to the output file.
        out_file (str): Name of the output file.
        lines (int, optional): Number of lines to save after a matching pattern is found. Defaults to 1 (save only the line with pattern occurrence).

    Raises:
        ValueError: If the pattern is not found in the input file.
    """
    regex = re.compile(pattern)

    input_file = os.path.join(path_inp, inp_file)
    output_file = os.path.join(path_out, out_file)

    with open(input_file, 'r') as file, open(output_file, 'w') as output:
        save_lines = False
        pattern_found = False

        for line in file:
            if regex.search(line):
                save_lines = True
                pattern_found = True
                remaining_lines = lines

            if save_lines:
                output.write(line)
                remaining_lines -= 1

                if remaining_lines <= 0:
                    save_lines = False

        if not pattern_found:
            raise ValueError("Pattern not found in the input file.")


def get_orca_so_blocks_size(path: str, orca_file: str) -> tuple[int, int, int]:
    """
    Retrieves the dimensions and block sizes for spin-orbit calculations from an ORCA file.

    Args:
        path (str): Path to the ORCA file.
        orca_file (str): Name of the ORCA file.

    Returns:
        tuple[int, int, int, int]: A tuple containing the spin-orbit dimension, block size, number of whole blocks,
            and remaining columns.

    Raises:
        ValueError: If the spin-orbit dimension is not found in the ORCA file.
    """
    orca_file_path = os.path.join(path, orca_file)

    with open(orca_file_path, 'r') as file:
        content = file.read()

    so_dim_match = re.search(r'Dim\(SO\)\s+=\s+(\d+)', content)
    if so_dim_match:
        so_dim = int(so_dim_match.group(1))
    else:
        raise ValueError("Dim(SO) not found in the ORCA file.")

    num_blocks = so_dim // 6

    if so_dim % 6 != 0:
        num_blocks += 1

    block_size = ((so_dim + 1) * num_blocks) + 1
    num_of_whole_blocks = so_dim // 6
    remaining_columns = so_dim % 6

    return so_dim, block_size, num_of_whole_blocks, remaining_columns


def orca_spin_orbit_to_hdf5(path_orca: str, inp_orca: str, path_out: str, hdf5_output: str, pt2: bool = False) -> None:
    """
    Converts spin-orbit calculations from an ORCA file to a HDF5 file format.

    Args:
        path_orca (str): Path to the ORCA file.
        inp_orca (str): Name of the ORCA file.
        path_out (str): Path for the output files.
        hdf5_output (str): Name of the HDF5 output file.
        pt2 (bool): Get results from the second-order perturbation-corrected states.

    Raises:
        ValueError: If the spin-orbit dimension is not found in the ORCA file.
        ValueError: If the pattern is not found in the input file.
    """
    # Retrieve dimensions and block sizes for spin-orbit calculations
    so_dim, block_size, num_of_whole_blocks, remaining_columns = get_orca_so_blocks_size(path_orca, inp_orca)

    # Create HDF5 file and ORCA group
    output = h5py.File(f'{hdf5_output}.hdf5', 'x')
    orca = output.create_group("ORCA")
    orca.attrs['Description'] = 'Group containing results of relativistic SOC ORCA calculations - angular momenta and SOC matrix in CI basis'

    # Extract and process matrices (SX, SY, SZ, LX, LY, LZ)
    matrices = ['SX', 'SY', 'SZ', 'LX', 'LY', 'LZ']
    descriptions = ['real part', 'imaginary part', 'real part', 'real part', 'real part', 'real part']

    for matrix_name, description in zip(matrices, descriptions):
        out_file = f'{matrix_name}.tmp'
        pattern = f'{matrix_name} MATRIX IN CI BASIS\n'

        # Extract lines matching the pattern to a temporary file
        grep_to_file(path_orca, inp_orca, pattern, path_out, out_file, block_size + 4)  # The first 4 lines are titles

        with open(out_file, 'r') as file:
            if pt2:
                for _ in range(block_size + 4):
                    file.readline()  #Skip non-pt2 block
            for _ in range(4):
                file.readline()  # Skip the first 4 lines
            matrix = np.empty((so_dim, so_dim), dtype=np.float64)
            l = 0
            for _ in range(num_of_whole_blocks):
                file.readline()  # Skip a line before each block of 6 columns
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(6):
                        matrix[i, l + j] = np.float64(line[j + 1])
                l += 6

            if remaining_columns > 0:
                file.readline()  # Skip a line before the remaining columns
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(remaining_columns):
                        matrix[i, l + j] = np.float64(line[j + 1])

            # Create dataset in HDF5 file and assign the matrix
            dataset = orca.create_dataset(matrix_name, shape=(so_dim, so_dim), dtype=np.float64)
            dataset[:, :] = matrix[:, :]
            dataset.attrs['Description'] = f'Dataset containing {description} of {matrix_name} matrix in CI basis'

        # Remove the temporary file
        os.remove(out_file)

    # Extract and process SOC matrix
    grep_to_file(path_orca, inp_orca, r'SOC MATRIX \(A\.U\.\)\n', path_out, 'SOC.tmp', 2 * block_size + 4 + 2)  # The first 4 lines are titles, two lines in the middle for Im part

    with open('SOC.tmp', "r") as file:
        content = file.read()

    # Replace "-" with " -"
    modified_content = content.replace("-", " -")

    # Write the modified content to the same file
    with open('SOC.tmp', "w") as file:
        file.write(modified_content)

    with open('SOC.tmp', 'r') as file:
        if pt2:
            for _ in range(2 * block_size + 4 + 2):
                file.readline()  #Skip non-pt2 block
        for _ in range(4):
            file.readline()  # Skip the first 4 lines
        matrix_real = np.empty((so_dim, so_dim), dtype=np.float64)
        l = 0
        for _ in range(num_of_whole_blocks):
            file.readline()  # Skip a line before each block of 6 columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_real[i, l + j] = np.float64(line[j + 1])
            l += 6

        if remaining_columns > 0:
            file.readline()  # Skip a line before the remaining columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(remaining_columns):
                    matrix_real[i, l + j] = np.float64(line[j + 1])

        for _ in range(2):
            file.readline()  # Skip 2 lines separating real and imaginary part

        matrix_imag = np.empty((so_dim, so_dim), dtype=np.float64)
        l = 0
        for _ in range(num_of_whole_blocks):
            file.readline()  # Skip a line before each block of 6 columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_imag[i, l + j] = np.float64(line[j + 1])
            l += 6

        if remaining_columns > 0:
            file.readline()  # Skip a line before the remaining columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(remaining_columns):
                    matrix_imag[i, l + j] = np.float64(line[j + 1])

    # Create a dataset in HDF5 file for SOC matrix
    dataset = orca.create_dataset('SOC', shape=(so_dim, so_dim), dtype=np.complex128)
    complex_matrix = np.array(matrix_real + 1j * matrix_imag, dtype=np.complex128)
    dataset[:, :] = complex_matrix[:, :]
    dataset.attrs['Description'] = 'Dataset containing complex SOC matrix in CI basis'

    # Remove the temporary file
    os.remove('SOC.tmp')

    # Close the HDF5 file
    output.close()


def get_soc_moment_energies_from_hdf5_orca(path: str, hdf5_file: str, states_cutoff: int) -> tuple[np.ndarray, np.ndarray]:

    # Construct input file path
    input_file = os.path.join(path, hdf5_file)

    # Check matrix size
    with h5py.File(input_file, 'r') as file:
        dataset = file['ORCA']['SOC']
        shape = dataset.shape[0]
    
    if states_cutoff > shape:
        raise ValueError(f'States cutoff is larger than the number of SO-states ({shape}). Please set it less or equal.')

    ge = 2.00231930436256 #Electron g factor

    #  Initialize the result array
    magnetic_moment = np.ascontiguousarray(np.zeros((3,states_cutoff,states_cutoff), dtype=np.complex128))

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
    sx = sx[:states_cutoff, :states_cutoff]
    sy = sy[:states_cutoff, :states_cutoff]
    sz = sz[:states_cutoff, :states_cutoff]
    lx = lx[:states_cutoff, :states_cutoff]
    ly = ly[:states_cutoff, :states_cutoff]
    lz = lz[:states_cutoff, :states_cutoff]
    soc_energies = soc_energies[:states_cutoff] - soc_energies[0]

    # Compute and save magnetic momenta in a.u.
    magnetic_moment[0] =  -(ge * sx + lx)
    magnetic_moment[1] =  -(ge * sy + ly)
    magnetic_moment[2] =  -(ge * sz + lz)

    return magnetic_moment, soc_energies


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


@jit('float64(float64[:], float64[:], float64)', nopython=True, cache=True, nogil=True)
def calculate_magnetization(energies: np.ndarray, states_momenta: np.ndarray, temperature: np.float64) -> np.float64:
    """
    Calculates the magnetization for a given array of states energies, momenta, and temperature.

    Args:
        energies (np.ndarray[np.float64]): Array of energies.
        states_momenta (np.ndarray[np.float64]): Array of states momenta.
        temperature (np.float64): Temperature value.

    Returns:
        np.float64: Magnetization value.

    """
    kB = 3.166811563e-6  # Boltzmann constant a.u./K

    # Boltzman weights
    exp_diff = np.exp(-(energies - energies[0]) / (kB * temperature))

    # Partition function
    z = np.sum(exp_diff)

    # Weighted magnetic moments of microstates
    m = np.sum(states_momenta * exp_diff)

    return m / z


@jit('float64[:](complex128[:,:,:], float64[:], float64, float64[:,:], float64[:])', nopython=True, cache=True, nogil=True)
def calculate_mt(magnetic_moment: np.ndarray, soc_energies: np.ndarray, field: np.float64, grid: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    """
    Calculates the M(T) array for a given array of magnetic moments, SOC energies, directional grid for powder averaging,
    and temperatures for a particular value of magnetic field.

    Args:
        magnetic_moment (np.ndarray[np.complex128]): Array of magnetic moments.
        soc_energies (np.ndarray[np.float64]): Array of SOC energies.
        field (np.float64): Value of magnetic field.
        grid (np.ndarray[np.float64]): Grid array.
        temperatures (np.ndarray[np.float64]): Array of temperatures.

    Returns:
        np.ndarray[np.float64]: M(T) array.

    """
    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T

    # Initialize arrays
    mt_array = np.ascontiguousarray(np.zeros((temperatures.shape[0]), dtype=np.float64))
    magnetic_moment = np.ascontiguousarray(magnetic_moment)
    soc_energies = np.ascontiguousarray(soc_energies)

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        # Construct Zeeman matrix
        orient = -field * bohr_magneton * grid[j, :3]
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
            grid[j, 0] * magnetic_moment[0]
            + grid[j, 1] * magnetic_moment[1]
            + grid[j, 2] * magnetic_moment[2]
        ) @ eigenvectors

        # Get diagonal momenta of the new states
        states_momenta = np.diag(states_momenta).real.astype(np.float64)

        # Compute partition function and magnetization for each T
        for t in range(temperatures.shape[0]):
            mt_array[t] += (calculate_magnetization(eigenvalues, states_momenta, temperatures[t]) * grid[j, 3])

    return mt_array


def calculate_mt_wrapper(args):
    """Wrapper function for parallel use of M(T) calulations

    Args:
        args (tuple): Tuple of arguments for calculate_mt function

    Returns:
        np.ndarray[np.float64]: M(T) array.
    """
    # Unpack arguments and call the function
    mt = calculate_mt(*args)

    return mt


def arg_iter_mth(magnetic_moment, soc_energies, fields, grid, temperatures):
    
    # Iterator generator for arguments with different field values to be distributed along num_process processes
    for i in range(fields.shape[0]):
      yield (magnetic_moment, soc_energies, fields[i], grid, temperatures)


def mth(path: str, hdf5_file: str, states_cutoff: int, fields: np.ndarray, grid: np.ndarray, temperatures: np.ndarray, num_cpu: int) -> np.ndarray:
    """
    Calculates the M(T,H) array using magnetic moments and SOC energies for given fields, grid (for powder direction averaging), and temperatures.
    The function is parallelized across num_cpu CPUs for simultaneous calculations over different magnetic field values.

    Args:
        path (str): Path to the file.
        hdf5_file (str): Name of the HDF5 file.
        states_cutoff (int): States cutoff value.
        fields (np.ndarray[np.float64]): Array of fields.
        grid (np.ndarray[np.float64]): Grid array of directions for powder averaging
        temperatures (np.ndarray[np.float64]): Array of temperatures
        num_cpu (int): Number of CPU used for calculations. (work will be distributed across num_cpu//num_threads processes)

    Raises:
        ValueError: If insufficient number of CPUs were assigned considering number of the desired threads.

    Returns:
        np.ndarray[np.float64]: M(T,H) array.

    """
    
    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu)

    # Initialize the result array
    mth_array = np.zeros((temperatures.shape[0], fields.shape[0]), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)
    temperatures = np.ascontiguousarray(temperatures)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_moment_energies_from_hdf5_orca(path, hdf5_file, states_cutoff)

    # Parallel M(T) calculation over different field values
    with multiprocessing.Pool(num_process) as p:
        mt = p.map(calculate_mt_wrapper, arg_iter_mth(magnetic_moment, soc_energies, fields, grid, temperatures))

    # Collecting results in plotting-friendly convention for M(H)
    for i in range(fields.shape[0]):
        mth_array[:,i] = mt[i]

    return mth_array # Returning values in Bohr magnetons


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


#add here jit
def finite_diff_stencil(diff_order: int, num_of_points: int, step: np.float64):

    stencil_len = 2 * num_of_points + 1

    if diff_order >= stencil_len:
        raise ValueError(f"Insufficient number of points to evaluate coefficients. Provide number of points greater than (derivative order - 1) / 2.")
    
    stencil_matrix = np.tile(np.arange(-num_of_points, num_of_points + 1).astype(np.int64), (stencil_len,1))
    stencil_matrix = stencil_matrix ** np.arange(0, stencil_len).reshape(-1, 1)

    order_vector = np.zeros(stencil_len)
    order_vector[diff_order] = math.factorial(diff_order)/np.power(step, diff_order)

    stencil_coeff = np.linalg.inv(stencil_matrix) @ order_vector.T

    return stencil_coeff


def chit(path: str, hdf5_file: str, field: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, grid: np.ndarray = None, exp: bool = False) -> np.ndarray:
    """
    Calculates chiT(T) using data from a HDF5 file for given field, states cutoff, temperatures, and optional grid (XYZ if not present).

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

    bohr_magneton_to_cm3 = 0.5584938904 # Conversion factor for chi in cm3

    # Set fields for finite difference method
    fields = np.arange(-num_of_points, num_of_points + 1).astype(np.int64) * delta_h + field
    fields = fields.astype(np.float64)

    # Default XYZ grid
    if grid is None:
        grid = np.array([[1., 0., 0., 0.3333333333333333], [0., 1., 0., 0.3333333333333333], [0., 0., 1., 0.3333333333333333]]).astype(np.float64)

    # Initialize result array
    chit = np.zeros_like(temperatures)

    # Experimentalist model
    if (exp == True) or (num_of_points == 0):

        mth_array = mth(path, hdf5_file, states_cutoff, np.array([field]), grid, temperatures, num_cpu)

        for index, temp in enumerate(temperatures):
            chit[index] = temp * mth_array[index] / field
        
        return chit * bohr_magneton_to_cm3


    # Get M(t,H) for two adjacent values of field
    mth_array = mth(path, hdf5_file, states_cutoff, fields, grid, temperatures, num_cpu)

    stencil_coeff = finite_diff_stencil(1, num_of_points, delta_h)

    # Numerical derivative of M(T,H) around given field value 
    for index, temp in enumerate(temperatures):
        chit[index] = temp * np.dot(mth_array[index], stencil_coeff)

    return chit * bohr_magneton_to_cm3


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
            numerator += matrix[i,j] * Clebsh_Gordan(J, -J + i, k, -q, J, -J + j)
            denominator += Clebsh_Gordan(J, -J + j, k, q, J, -J + i) * Clebsh_Gordan(J, -J + i, k, -q, J, -J + j)

    B = 1.0

    for i in range(-k, k+1):
            B = B * (2*J + 1 + i)
            
    B = B / ((2**k) * math.factorial(2*k))
    B = math.sqrt(B)
    B = (-1)**k * B * math.factorial(k)

    #denominator = (2*J + 1) / (2*k + 1)

    return numerator*np.sqrt(J)/denominator/B


def ITO_decomp_matrix(matrix: np.ndarray, order: int):

    for k in range(0,order+1):
        for q in range(-k,k+1):
            if q >= 0:
                B_k_q = ((-1)**q * calculate_B_k_q(matrix, k, q) + calculate_B_k_q(matrix, k, -q))
                print(f"{k} {q} {B_k_q.real}")
            if q < 0:
                B_k_q = -1j * (-(-1)**q * calculate_B_k_q(matrix, k, q) + calculate_B_k_q(matrix, k, -q))
                print(f"{k} {q} {B_k_q.real}")


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

    return soc_matrix #-(lz + 2 * sz)


def matrix_from_ITO():
    pass


if __name__ == '__main__':


    hamiltonian = get_SOC_matrix_in_J_basis('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 16)

    ITO_decomp_matrix(hamiltonian, 14)


    fields = np.linspace(0.001, 7, 64)
    temperatures1 = np.linspace(1.8, 1.8, 1)
    temperatures2 = np.linspace(1, 300, 300)
    grid = np.loadtxt('grid.txt', usecols = (1,2,3,4))
    grid2 = np.loadtxt('grid2.txt', usecols = (1,2,3,4))
    grid3 = np.loadtxt('grid3.txt', usecols = (1,2,3,4))
    temperatures3 = np.linspace(1,5,5)

    #print("\033[90mTitle") #kolory w terminalu

    #print(get_states_magnetic_momenta('.', 'DyCo_cif_nevpt2_new_basis.hdf5', np.arange(16), J_moment=True))
    # sus_tensor = chi_tensor('.', 'DyCo_cif.hdf5', 0.1, 2003, temperatures3, 32, 7, 0.0001)

    # print(sus_tensor * temperatures3[:, np.newaxis, np.newaxis])

    # zs = zeeman_splitting('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 16, 16, fields, grid, 2, average = True)

    # zs = zeeman_splitting('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 16, 16, fields, np.array([[0.,0.,1.,1.]]), 2)


    # plt.plot(fields, zs[0], "-")
    # plt.show()

    



    #print(get_soc_energies_cm_1('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 16))

    # x, y, z = mag_3d('.', 'NdCoNO2_cas_super_tight_cas.hdf5', 364, 1., 100, 2., 32)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.plot_wireframe(x, y, z)
    # ax.set_xlim(-2,2)
    # ax.set_ylim(-2,2)
    # ax.set_zlim(-2,2)
    # ax.set_box_aspect([1, 1, 1])
    # plt.show()


    # def mth_function_wrapper():
    #     mth('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 512, fields, grid, temperatures1, 32)

    # repetitions = 1


    #Measure execution time
    # execution_times = timeit.repeat(stmt=mth_function_wrapper, repeat=1, number=repetitions)

    # print("Execution times mth:", str(np.array(execution_times)/repetitions), "seconds")


    # def chit_function_wrapper():
    #     chit('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 0.1, 2002, temperatures2, 64)

    # repetitions = 1


    # #Measure execution time
    # execution_times = timeit.repeat(stmt=mth_function_wrapper, repeat=1, number=repetitions)

    # print("Execution times chit:", str(np.array(execution_times)/repetitions), "seconds")


    # mth1 = mth('.', 'DyCo.hdf5', 64, fields, grid, temperatures1, 64)
    # mth2 = mth('.', 'DyCo_nevpt2.hdf5', 64, fields, grid, temperatures1, 64)
    # mth3 = mth('.', 'DyCo_nevpt2_trun.hdf5', 64, fields, grid, temperatures1, 64)
    # mth4 = mth('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 32, fields, grid, temperatures1, 2)
    # mth5 = mth('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 32, fields, grid2, temperatures1, 2)
    #mth6 = mth('.', 'NdCoNO2_cas_super_tight_cas.hdf5', 364, fields, grid3, temperatures1, 32)

    # for i in fields:
    #     print(i)

    # for mh in mth4:
    #     plt.plot(fields, mh)
    # for mh in mth5:
    #     plt.plot(fields, mh)
    # for mh in mth6:
    #     plt.plot(fields, mh)
    #     for i in mh:
    #         print(i)

    # fields, temperatures1 = np.meshgrid(fields, temperatures1)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    #Plot the surface.
    # surf = ax.plot_surface(fields, temperatures1, mth5, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # ax.set_box_aspect([1, 1, 1])

    # plt.show()

    # for mh in mth2:
    #     plt.plot(fields, mh, "b-")
    #     # for i in mh:
    #     #     print(i)
    # for mh in mth3:
    #     plt.plot(fields, mh, "y-")
    # for mh in mth4:
    #     plt.plot(fields, mh, "g-")
    # plt.ylim(0,6)
    # plt.show()



    # chit1 = chit('.', 'DyCo.hdf5', 0.1, 512, temperatures2, 64, grid)
    # chit2 = chit('.', 'DyCo_nevpt2.hdf5', 0.1, 512, temperatures2, 64, grid)
    # chit3 = chit('.', 'DyCo_nevpt2_trun.hdf5', 0.1, 512, temperatures2, 64, grid)

    # plt.plot(temperatures2, chit1, "r-")
    # plt.plot(temperatures2, chit2, "b-")
    # plt.plot(temperatures2, chit3, "y-")
    


    # chit4 = chit('.', 'DyCo_cif.hdf5', 0.1, 128, temperatures2, 2, 1, 0.00001)
    # chit5 = chit('.', 'DyCo_nevpt2.hdf5', 0.1, 512, temperatures2, 64)
    # chit6 = chit('.', 'DyCo_nevpt2_trun.hdf5', 0.1, 512, temperatures2, 64)
    # chit7 = chit('.', 'NdCoNO2_cas_super_tight_cas.hdf5', 0.1, 364, temperatures2, 32, 1, 0.0001, exp=False)

    # plt.plot(temperatures2, chit4, "r-")
    # plt.plot(temperatures2, chit5, "b-")
    # plt.plot(temperatures2, chit6, "y-")
    # plt.plot(temperatures2, chit7, "g-")
    # plt.ylim(0, 15)
    # plt.show()

    # for t in temperatures2:
    #     print(t)

    # for c in chit7:
    #     print(c)

    #orca_spin_orbit_to_hdf5('.', 'NdCoNO2_cas_super_tight_cas.out', '.', 'NdCoNO2_cas_super_tight_cas', pt2 = False)