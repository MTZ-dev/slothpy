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

    # Create dataset in HDF5 file for SOC matrix
    dataset = orca.create_dataset('SOC', shape=(so_dim, so_dim), dtype=np.complex128)
    complex_matrix = np.array(matrix_real + 1j * matrix_imag, dtype=np.complex128)
    dataset[:, :] = complex_matrix[:, :]
    dataset.attrs['Description'] = 'Dataset containing complex SOC matrix in CI basis'

    # Remove the temporary file
    os.remove('SOC.tmp')

    # Close the HDF5 file
    output.close()


def get_soc_moment_energies_from_hdf5_orca(path: str, hdf5_file: str, states_cutoff: int) -> tuple[np.ndarray, np.ndarray]:

    ge = 2.00231930436256 #Electron g factor

    #  Initialize the result array
    magnetic_moment = np.ascontiguousarray(np.zeros((3,states_cutoff,states_cutoff), dtype=np.complex128))

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
    
    # Check CPUs number considering the desired number of threads and assign number of processes
    if num_cpu < int(os.getenv('OMP_NUM_THREADS')):
        raise ValueError(f"Insufficient number of CPU cores assigned. Desired threads: {int(os.getenv('OMP_NUM_THREADS'))}, Actual processors: {num_cpu}")
    else:
        num_process = num_cpu//int(os.getenv('OMP_NUM_THREADS'))

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


def chit(path: str, hdf5_file: str, field: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, grid: np.ndarray = None) -> np.ndarray:
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

    delta_h = 0.0001  # Set dH = 1 Oe

    # Set fields for finite difference method
    fields = np.array([field - delta_h, field + delta_h])

    # Default XYZ grid
    if grid is None:
        grid = np.array([[1., 0., 0., 0.3333333333333333], [0., 1., 0., 0.3333333333333333], [0., 0., 1., 0.3333333333333333]]).astype(np.float64)

    # Initialize result array
    chit = np.zeros_like(temperatures)

    # Get M(t,H) for two adjacent values of field
    mth_array = mth(path, hdf5_file, states_cutoff, fields, grid, temperatures, num_cpu)

    # Numerical derivative of M(T,H) around given field value 
    for index, temp in enumerate(temperatures):
        chit[index] = temp * (mth_array[index][1] - mth_array[index][0]) / (2 * delta_h)

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

    # Check CPUs number considering the desired number of threads and assign number of processes
    if num_cpu < int(os.getenv('OMP_NUM_THREADS')):
        raise ValueError(f"Insufficient number of CPU cores assigned. Desired threads: {int(os.getenv('OMP_NUM_THREADS'))}, Actual processors: {num_cpu}")
    else:
        num_process = num_cpu//int(os.getenv('OMP_NUM_THREADS'))

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





if __name__ == '__main__':


    fields = np.linspace(0.001, 7, 64)
    temperatures1 = np.linspace(1, 300, 300)
    temperatures2 = np.linspace(1, 300, 300)
    grid = np.loadtxt('grid.txt', usecols = (1,2,3,4))


    # x, y, z = mag_3d('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 256, 0.1, 300, 2.0, 32)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.plot_wireframe(x, y, z)
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
    # mth4 = mth('.', 'DyCo_cif.hdf5', 64, fields, grid, temperatures1, 64)
    mth5 = mth('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 256, fields, grid, temperatures1, 32)

    # for i in fields:
    #     print(i)

    # for mh in mth5:
    #     plt.plot(fields, mh)
    #     for i in mh:
    #         print(i)

    fields, temperatures1 = np.meshgrid(fields, temperatures1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(fields, temperatures1, mth5, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_box_aspect([1, 1, 1])

    plt.show()

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
    


    # chit4 = chit('.', 'DyCo.hdf5', 0.1, 512, temperatures2, 64)
    # chit5 = chit('.', 'DyCo_nevpt2.hdf5', 0.1, 512, temperatures2, 64)
    # chit6 = chit('.', 'DyCo_nevpt2_trun.hdf5', 0.1, 512, temperatures2, 64)
    #chit7 = chit('.', 'DyCo_cif_nevpt2_new_basis.hdf5', 0.1, 2002, temperatures2, 64)

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

    #orca_spin_orbit_to_hdf5('.', 'DyCo_cif_cas_nevpt2_new_basis.out', '.', 'DyCo_cif_nevpt2_new_basis', pt2 = True)