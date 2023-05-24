import os
import re
import numpy as np
import h5py

def grep(path_inp, inp_file, pattern, path_out, out_file, lines_after=0):
    # Compile the regular expression pattern
    regex = re.compile(pattern)
    
    input_file = os.path.join(path_inp, inp_file)
    output_file = os.path.join(path_out, out_file)

    # Open the input and output files
    with open(input_file, 'r') as file, open(output_file, 'w') as output:
        # Flag to determine if we should save lines
        save_lines = False

        # Iterate over each line in the file
        for line in file:
            # Check if the line matches the pattern
            if regex.search(line):
                # Match found, set the flag to start saving lines
                save_lines = True

            # Save lines after the matching line
            if save_lines:
                output.write(line)

                # Decrease the number of lines to save
                lines_after -= 1

                # Check if we have saved enough lines
                if lines_after <= 0:
                    save_lines = False

def get_orca_so_dim_blocks(path: str, orca_file: str) -> "tuple(int, int)":
    
    orca_file = os.path.join(path, orca_file)
    
    with open(orca_file, 'r') as file:
        content = file.read()
    so_dim = int(re.search(r'Dim\(SO\)\s+=\s+(\d+)', content).group(1))
    num_blocks = so_dim // 6
    if so_dim % 6 != 0:
        num_blocks += 1
    block_size = ((so_dim + 1) * num_blocks) + 1
    
    return so_dim, num_blocks, block_size

import os
import re
import h5py

def orca_spin_orbit_to_hdf5(path_orca: str, inp_orca: str, path_out: str, hdf5_output: str) -> None:
    
    so_dim, num_blocks, block_size = get_orca_so_dim_blocks(path_orca, inp_orca)
    N1 = so_dim // 6
    N2 = so_dim % 6
    output = h5py.File(f'{hdf5_output}.hdf5','x')
    orca = output.create_group("orca") 
    
    for pattern, out_file in zip([r'SX MATRIX IN CI BASIS\n', r'SY MATRIX IN CI BASIS\n', r'SZ MATRIX IN CI BASIS\n',
                                 r'LX MATRIX IN CI BASIS\n', r'LY MATRIX IN CI BASIS\n', r'LZ MATRIX IN CI BASIS\n'],['SX', 'SY', 'SZ', 'LX', 'LY', 'LZ']): 
        grep(path_orca, inp_orca, pattern, path_out, out_file, block_size + 4) #The first 4 lines are titles
        with open(out_file, 'r') as file:
            for i in range(4):
                file.readline() #Skip the first 4 lines
            matrix = np.empty((so_dim, so_dim), dtype=np.float64)
            l = 0
            for k in range(N1):
                file.readline()  # Skip a line before each block of 6 rows
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(6):
                        matrix[i, l+j] = np.float64(line[j+1])
                l += 6

            if N2 > 0:
                file.readline()  # Skip a line before the remaining rows
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(N2):
                        matrix[i, l+j] = np.float64(line[j+1])
            
            dataset = orca.create_dataset(out_file, shape=(so_dim, so_dim), dtype=np.float64)
            dataset[:,:] = matrix[:,:]
    
    grep(path_orca, inp_orca, r'SOC MATRIX \(A\.U\.\)\n', path_out, 'SOC', 2*block_size + 4 + 2) #The first 4 lines are titles, two lines in the middle for Im part
    
    with open('SOC', "r") as file:
        content = file.read()

    # Replace "-" with " -"
    modified_content = content.replace("-", " -")

    # Write the modified content to the same file
    with open('SOC', "w") as file:
        file.write(modified_content)
    
    with open('SOC', 'r') as file:
        for i in range(4):
            file.readline() #Skip the first 4 lines
        matrix_real = np.empty((so_dim, so_dim), dtype=np.float64)
        l = 0
        for k in range(N1):
            file.readline()  # Skip a line before each block of 6 rows
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_real[i, l+j] = np.float64(line[j+1])
            l += 6

        if N2 > 0:
            file.readline()  # Skip a line before the remaining rows
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(N2):
                    matrix_real[i, l+j] = np.float64(line[j+1])
                    
        for i in range(2):
            file.readline() #Skip 2 lines separating real and imaginary part
            
        matrix_imag = np.empty((so_dim, so_dim), dtype=np.float64)
        l = 0
        for k in range(N1):
            file.readline()  # Skip a line before each block of 6 rows
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_imag[i, l+j] = np.float64(line[j+1])
            l += 6

        if N2 > 0:
            file.readline()  # Skip a line before the remaining rows
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(N2):
                    matrix_imag[i, l+j] = np.float64(line[j+1])

        dataset = orca.create_dataset('SOC', shape=(so_dim, so_dim), dtype=np.complex128)
        complex_matrix = np.array(matrix_real + 1j * matrix_imag, dtype=np.complex128)
        #complex_matrix = complex_matrix.astype(np.complex128)
        dataset[:,:] = complex_matrix[:,:]
    
    # output.close()

    # orca_spin_orbit_to_hdf5('.', 'DyCo_CF_34_-2_cas.out', '.', 'test')

    # file = h5py.File('test.hdf5', 'r')
    # soc = file['orca']['SOC']
    # soc_matrix = soc[:,:]
    # file.close()
    # eigenvalues, eigenvectors = np.linalg.eigh(soc_matrix)
    # eigenvectors = np.matrix(eigenvectors)
    # eigenvectors.H@soc_matrix@eigenvectors