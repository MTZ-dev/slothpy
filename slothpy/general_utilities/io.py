import os
import re
import numpy as np
import h5py

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

def orca_spin_orbit_to_slt(path_orca: str, inp_orca: str, path_out: str, hdf5_output: str, name: str, pt2: bool = False) -> None:
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
    output = h5py.File(f'{hdf5_output}.slt', 'a')
    orca = output.create_group(str(name))
    orca.attrs['Description'] = f'Group({name}) containing results of relativistic SOC ORCA calculations - angular momenta and SOC matrix in CI basis'

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

def get_soc_energies_and_soc_angular_momenta_from_hdf5(filename: str, group: str) -> tuple: #named tuple, see numpy for example linalg.eig, they return class from typing

    # Read data from HDF5 file
    with h5py.File(filename, 'r') as file:
        soc_matrix = file[str(group)]['SOC'][:]
        sx = 0.5 * file[str(group)]['SX'][:]
        lx = 1j * file[str(group)]['LX'][:]
        sy = 0.5j * file[str(group)]['SY'][:]
        ly = 1j * file[str(group)]['LY'][:]
        sz = 0.5 * file[str(group)]['SZ'][:]
        lz = 1j * file[str(group)]['LZ'][:]

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

    # Return operators in SOC basis
    return soc_energies, sx, sy, sz, lx, ly, lz

