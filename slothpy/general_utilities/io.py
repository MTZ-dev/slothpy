import os
import re
import numpy as np
import h5py
from typing import Tuple
from slothpy.general_utilities._constants import YELLOW, RESET
from slothpy.core._slothpy_exceptions import SltFileError, SltReadError
from slothpy.angular_momentum.rotation import _rotate_vector_operator
from slothpy.general_utilities._math_expresions import _mag_mom_from_ang_mom


def grep_to_file(
    path_inp: str,
    inp_file: str,
    pattern: str,
    path_out: str,
    out_file: str,
    lines: int = 1,
) -> None:
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

    with open(input_file, "r") as file, open(output_file, "w") as output:
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


def _dataset_exists(hdf5_file, group_name, dataset_name):
    with h5py.File(hdf5_file, "r") as file:
        if group_name in file:
            group = file[group_name]
            return dataset_name in group

        return False


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

    with open(orca_file_path, "r") as file:
        content = file.read()

    so_dim_match = re.search(r"Dim\(SO\)\s+=\s+(\d+)", content)
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


def orca_spin_orbit_to_slt(
    path_orca: str,
    inp_orca: str,
    path_out: str,
    hdf5_output: str,
    name: str,
    pt2: bool = False,
) -> (
    None
):  # tutaj with open raczej dla hdf5, żeby zabezpieczyć plik! juz raz się zepsuł
    """
    Converts spin-orbit calculations from an ORCA .out file to a HDF5 file format.

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
    hdf5_file = os.path.join(path_out, hdf5_output)

    # Retrieve dimensions and block sizes for spin-orbit calculations
    (
        so_dim,
        block_size,
        num_of_whole_blocks,
        remaining_columns,
    ) = get_orca_so_blocks_size(path_orca, inp_orca)

    # Create HDF5 file and ORCA group
    output = h5py.File(f"{hdf5_file}.slt", "a")
    orca = output.create_group(str(name))
    orca.attrs["Description"] = (
        f"Group({name}) containing results of relativistic SOC ORCA"
        " calculations - angular momenta and SOC matrix in CI basis"
    )

    # Extract and process matrices (SX, SY, SZ, LX, LY, LZ)
    matrices = ["SX", "SY", "SZ", "LX", "LY", "LZ"]
    descriptions = [
        "real part",
        "imaginary part",
        "real part",
        "real part",
        "real part",
        "real part",
    ]

    for matrix_name, description in zip(matrices, descriptions):
        out_file = f"{matrix_name}.tmp"
        pattern = f"{matrix_name} MATRIX IN CI BASIS\n"

        # Extract lines matching the pattern to a temporary file
        grep_to_file(
            path_orca, inp_orca, pattern, path_out, out_file, block_size + 4
        )  # The first 4 lines are titles

        with open(out_file, "r") as file:
            if pt2:
                for _ in range(block_size + 4):
                    file.readline()  # Skip non-pt2 block
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
            dataset = orca.create_dataset(
                f"SF_{matrix_name}", shape=(so_dim, so_dim), dtype=np.float64
            )
            dataset[:, :] = matrix[:, :]
            dataset.attrs["Description"] = (
                f"Dataset containing {description} of SF_{matrix_name} matrix"
                " in CI basis (spin-free)"
            )

        # Remove the temporary file
        os.remove(out_file)

    # Extract and process SOC matrix
    grep_to_file(
        path_orca,
        inp_orca,
        r"SOC MATRIX \(A\.U\.\)\n",
        path_out,
        "SOC.tmp",
        2 * block_size + 4 + 2,
    )  # The first 4 lines are titles, two lines in the middle for Im part

    with open("SOC.tmp", "r") as file:
        content = file.read()

    # Replace "-" with " -"
    modified_content = content.replace("-", " -")

    # Write the modified content to the same file
    with open("SOC.tmp", "w") as file:
        file.write(modified_content)

    with open("SOC.tmp", "r") as file:
        if pt2:
            for _ in range(2 * block_size + 4 + 2):
                file.readline()  # Skip non-pt2 block
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
    dataset = orca.create_dataset(
        "SOC", shape=(so_dim, so_dim), dtype=np.complex128
    )
    complex_matrix = np.array(
        matrix_real + 1j * matrix_imag, dtype=np.complex128
    )
    dataset[:, :] = complex_matrix[:, :]
    dataset.attrs[
        "Description"
    ] = "Dataset containing complex SOC matrix in CI basis (spin-free)"

    # Remove the temporary file
    os.remove("SOC.tmp")

    # Close the HDF5 file
    output.close()


def molcas_spin_orbit_to_slt(
    path_molcas: str,
    inp_molcas: str,
    path_out: str,
    hdf5_output: str,
    name: str,
) -> None:
    rassi_path_name = os.path.join(path_molcas, inp_molcas)

    with h5py.File(f"{rassi_path_name}.rassi.h5", "a") as rassi:
        soc_energies = rassi["SOS_ENERGIES"][:]
        sx = (
            rassi["SOS_SPIN_REAL"][0, :, :]
            + 1j * rassi["SOS_SPIN_IMAG"][0, :, :]
        )
        sy = (
            rassi["SOS_SPIN_REAL"][1, :, :]
            + 1j * rassi["SOS_SPIN_IMAG"][1, :, :]
        )
        sz = (
            rassi["SOS_SPIN_REAL"][2, :, :]
            + 1j * rassi["SOS_SPIN_IMAG"][2, :, :]
        )
        lx = (
            1j * rassi["SOS_ANGMOM_REAL"][0, :, :]
            - rassi["SOS_ANGMOM_IMAG"][0, :, :]
        )
        ly = (
            1j * rassi["SOS_ANGMOM_REAL"][1, :, :]
            - rassi["SOS_ANGMOM_IMAG"][1, :, :]
        )
        lz = (
            1j * rassi["SOS_ANGMOM_REAL"][2, :, :]
            - rassi["SOS_ANGMOM_IMAG"][2, :, :]
        )

    hdf5_file = os.path.join(path_out, hdf5_output)

    with h5py.File(f"{hdf5_file}.slt", "a") as output:
        molcas = output.create_group(str(name))
        molcas.attrs["Description"] = (
            f"Group({name}) containing results of relativistic SOC MOLCAS"
            " calculations - angular momenta, spin in SOC basis and SOC"
            " energies"
        )

        rassi_soc = molcas.create_dataset(
            "SOC_energies", shape=(soc_energies.shape[0],), dtype=np.float64
        )
        rassi_soc[:] = soc_energies[:]
        rassi_soc.attrs[
            "Description"
        ] = f"Dataset containing SOC energies from MOLCAS RASSI calculation"
        rassi_sx = molcas.create_dataset(
            "SOC_SX", shape=sx.shape, dtype=np.complex128
        )
        rassi_sx[:] = sx[:]
        rassi_sx.attrs[
            "Description"
        ] = f"Dataset containing Sx matrix in SOC basis"
        rassi_sy = molcas.create_dataset(
            "SOC_SY", shape=sy.shape, dtype=np.complex128
        )
        rassi_sy[:] = sy[:]
        rassi_sy.attrs[
            "Description"
        ] = f"Dataset containing Sy matrix in SOC basis"
        rassi_sz = molcas.create_dataset(
            "SOC_SZ", shape=sz.shape, dtype=np.complex128
        )
        rassi_sz[:] = sz[:]
        rassi_sz.attrs[
            "Description"
        ] = f"Dataset containing Sz matrix in SOC basis"
        rassi_lx = molcas.create_dataset(
            "SOC_LX", shape=lx.shape, dtype=np.complex128
        )
        rassi_lx[:] = lx[:]
        rassi_lx.attrs[
            "Description"
        ] = f"Dataset containing Lx matrix in SOC basis"
        rassi_ly = molcas.create_dataset(
            "SOC_LY", shape=ly.shape, dtype=np.complex128
        )
        rassi_ly[:] = ly[:]
        rassi_ly.attrs[
            "Description"
        ] = f"Dataset containing Ly matrix in SOC basis"
        rassi_lz = molcas.create_dataset(
            "SOC_LZ", shape=lz.shape, dtype=np.complex128
        )
        rassi_lz[:] = lz[:]
        rassi_lz.attrs[
            "Description"
        ] = f"Dataset containing Lz matrix in SOC basis"


def _load_orca_hdf5(filename, group, rotation):
    try:
        # Read data from HDF5 file
        with h5py.File(filename, "r") as file:
            shape = file[str(group)]["SOC"][:].shape[0]
            ang_mom = np.zeros((6, shape, shape), dtype=np.complex128)
            soc_mat = file[str(group)]["SOC"][:]
            ang_mom[0][:] = 0.5 * file[str(group)]["SF_SX"][:]
            ang_mom[1][:] = 0.5j * file[str(group)]["SF_SY"][:]
            ang_mom[2][:] = 0.5 * file[str(group)]["SF_SZ"][:]
            ang_mom[3][:] = 1j * file[str(group)]["SF_LX"][:]
            ang_mom[4][:] = 1j * file[str(group)]["SF_LY"][:]
            ang_mom[5][:] = 1j * file[str(group)]["SF_LZ"][:]

        # Perform diagonalization of SOC matrix
        soc_ener, eigenvect = np.linalg.eigh(soc_mat)

        ang_mom = np.ascontiguousarray(ang_mom)
        soc_ener = np.ascontiguousarray(soc_ener.astype(np.float64))
        eigenvect = np.ascontiguousarray(eigenvect.astype(np.complex128))

        # Apply transformations to spin and orbital operators
        ang_mom = eigenvect.conj().T @ ang_mom @ eigenvect

        if (rotation is not None) or (rotation != None):
            ang_mom[0:3, :, :] = _rotate_vector_operator(
                ang_mom[0:3, :, :], rotation
            )
            ang_mom[3:6, :, :] = _rotate_vector_operator(
                ang_mom[3:6, :, :], rotation
            )

        # Return operators in SOC basis
        return soc_ener, ang_mom

    except Exception as exc:
        raise SltFileError(
            filename,
            exc,
            message=(
                "Failed to load SOC, spin, and angular momenta data in"
                " the ORCA format from the .slt file."
            ),
        ) from None


def _load_molcas_hdf5(filename, group, rotation):
    try:
        with h5py.File(filename, "r") as file:
            shape = file[str(group)]["SOC_energies"][:].shape[0]
            ang_mom = np.zeros((6, shape, shape), dtype=np.complex128)
            soc_ener = file[str(group)]["SOC_energies"][:]
            ang_mom[0][:] = file[str(group)]["SOC_SX"][:]
            ang_mom[1][:] = file[str(group)]["SOC_SY"][:]
            ang_mom[2][:] = file[str(group)]["SOC_SZ"][:]
            ang_mom[3][:] = file[str(group)]["SOC_LX"][:]
            ang_mom[4][:] = file[str(group)]["SOC_LY"][:]
            ang_mom[5][:] = file[str(group)]["SOC_LZ"][:]

        if (rotation is not None) or (rotation != None):
            ang_mom[0:3, :, :] = _rotate_vector_operator(
                ang_mom[0:3, :, :], rotation
            )
            ang_mom[3:6, :, :] = _rotate_vector_operator(
                ang_mom[3:6, :, :], rotation
            )

        return soc_ener, ang_mom

    except Exception as exc:
        raise SltFileError(
            filename,
            exc,
            message=(
                "Failed to load SOC, spin, and angular momenta data in"
                " the MOLCAS format from the .slt file."
            ),
        ) from None


def _get_soc_ener_and_soc_ang_mom_from_hdf5(
    filename: str, group: str, rotation=None
) -> Tuple[np.ndarray, np.ndarray]:
    if _dataset_exists(filename, group, "SOC"):
        return _load_orca_hdf5(filename, group, rotation)

    if _dataset_exists(filename, group, "SOC_energies"):
        return _load_molcas_hdf5(filename, group, rotation)

    else:
        raise SltReadError(
            filename,
            ValueError(""),
            message=(
                "Incorect data format. The program was unable to read SOC,"
                " spin, and angular momenta."
            ),
        )


def _get_soc_mag_mom_and_ener_from_hdf5(
    filename: str, group: str, states_cutoff: int, rotation=None
) -> tuple[np.ndarray, np.ndarray]:
    soc_ener, ang_mom = _get_soc_ener_and_soc_ang_mom_from_hdf5(
        filename, group, rotation
    )

    shape = soc_ener.size

    if (not isinstance(states_cutoff, int)) or (states_cutoff < 0):
        raise ValueError(
            "Invalid states cutoff. Set it to positive integer less or equal"
            " to the number of SO-states (or zero for all states)."
        )

    if states_cutoff > shape:
        raise ValueError(
            f"States cutoff is larger than the number of SO-states ({shape})."
            " Please set it less or equal (or zero for all states)."
        )

    if states_cutoff == 0:
        states_cutoff = shape

    soc_ener = soc_ener[:states_cutoff] - soc_ener[0]
    mag_mom = _mag_mom_from_ang_mom(ang_mom, stop=states_cutoff)

    return mag_mom, soc_ener


def get_soc_total_angular_momenta_and_energies_from_hdf5(
    filename: str, group: str, states_cutoff: int, rotation=None
) -> tuple[np.ndarray, np.ndarray]:
    shape = -1

    # Check matrix size
    with h5py.File(filename, "r") as file:
        try:
            dataset = file[group]["SOC"]
        except Exception as e:
            error_type_1 = type(e).__name__
            error_message_1 = str(e)
            error_print_1 = f"{error_type_1}: {error_message_1}"
            try:
                dataset = file[group]["SOC_energies"]
            except Exception as e:
                error_type_2 = type(e).__name__
                error_message_2 = str(e)
                error_print_2 = f"{error_type_2}: {error_message_2}"
                raise Exception(
                    "Failed to acces SOC data sets due to the following"
                    f" errors: {error_print_1}, {error_print_2}"
                )

        shape = dataset.shape[0]

    if shape < 0:
        raise Exception(
            f"Failed to read size of SOC matrix from file {filename} due to"
            f" the following errors:\n {error_print_1}, {error_print_2}"
        )

    if (not isinstance(states_cutoff, int)) or (states_cutoff < 0):
        raise ValueError(
            f"Invalid states cutoff. Set it to positive integer less or equal"
            f" to the number of SO-states"
        )

    if states_cutoff > shape:
        raise ValueError(
            f"States cutoff is larger than the number of SO-states ({shape})."
            " Please set it less or equal (or zero) for all states."
        )

    if states_cutoff == 0:
        states_cutoff = shape

    #  Initialize the result array
    total_angular_momenta = np.ascontiguousarray(
        np.zeros((3, states_cutoff, states_cutoff), dtype=np.complex128)
    )

    (
        soc_energies,
        sx,
        sy,
        sz,
        lx,
        ly,
        lz,
    ) = get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group, rotation
    )

    # Slice arrays based on states_cutoff
    sx = sx[:states_cutoff, :states_cutoff]
    sy = sy[:states_cutoff, :states_cutoff]
    sz = sz[:states_cutoff, :states_cutoff]
    lx = lx[:states_cutoff, :states_cutoff]
    ly = ly[:states_cutoff, :states_cutoff]
    lz = lz[:states_cutoff, :states_cutoff]
    soc_energies = soc_energies[:states_cutoff] - soc_energies[0]

    # Compute and save magnetic momenta in a.u.
    total_angular_momenta[0] = sx + lx
    total_angular_momenta[1] = sy + ly
    total_angular_momenta[2] = sz + lz

    return total_angular_momenta, soc_energies


def get_soc_energies_cm_1(
    filename: str, group: str, num_of_states: int = None
) -> np.ndarray:
    hartree_to_cm_1 = 219474.6

    if num_of_states < 0 or (not isinstance(num_of_states, int)):
        raise ValueError(
            f"Invalid number of states. Set it to positive integer or 0 for"
            f" all states."
        )

    try:
        # Read data from HDF5 file
        with h5py.File(filename, "r") as file:
            soc_matrix = file[str(group)]["SOC"][:]

        # Perform diagonalization on SOC matrix
        soc_energies = np.linalg.eigvalsh(soc_matrix)

        if (
            isinstance(num_of_states, int)
            and num_of_states > 0
            and num_of_states <= soc_energies.shape[0]
        ):
            soc_energies = soc_energies[:num_of_states]

        # Return operators in SOC basis
        return (soc_energies - soc_energies[0]) * hartree_to_cm_1

    except Exception as e:
        error_type_1 = type(e).__name__
        error_message_1 = str(e)
        error_print_1 = f"{error_type_1}: {error_message_1}"

    try:
        with h5py.File(filename, "r") as file:
            soc_energies = file[str(group)]["SOC_energies"][:]

        if (
            isinstance(num_of_states, int)
            and num_of_states > 0
            and num_of_states <= soc_energies.shape[0]
        ):
            soc_energies = soc_energies[:num_of_states]

        return (soc_energies - soc_energies[0]) * hartree_to_cm_1

    except Exception as e:
        error_type_2 = type(e).__name__
        error_message_2 = str(e)
        error_print_2 = f"{error_type_2}: {error_message_2}"

    raise Exception(
        "Failed to load SOC, spin and angular momenta data from HDF5 file.\n"
        f" Error(s) encountered while trying read the data: {error_print_1},"
        f" {error_print_2}"
    )


def get_states_magnetic_momenta(
    filename: str, group: str, states: np.ndarray = None, rotation=None
):
    ge = 2.00231930436256  # Electron g factor

    (
        _,
        sx,
        sy,
        sz,
        lx,
        ly,
        lz,
    ) = get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group, rotation
    )

    if (
        (np.all(states is not None))
        and (np.all(states != None))
        and (np.any(states != 0))
    ):
        if (
            np.any(states < 0)
            or not np.issubdtype(states.dtype, np.integer)
            or np.any(states > sx.shape[0])
        ):
            raise ValueError(
                "States list contains negative values, non-integer elements"
                " or indexes greater than the number of states:"
                f" {sx.shape[0]}!"
            )

        if states.size == 1 and (np.any(states != 0)):
            magnetic_momenta = np.ascontiguousarray(
                np.zeros((3, states), dtype=np.complex128)
            )

            # Slice arrays based on states_cutoff
            sx = sx[:states, :states]
            sy = sy[:states, :states]
            sz = sz[:states, :states]
            lx = lx[:states, :states]
            ly = ly[:states, :states]
            lz = lz[:states, :states]

            # Compute and save magnetic momenta in a.u.
            magnetic_momenta[0] = np.diagonal(-(ge * sx + lx))
            magnetic_momenta[1] = np.diagonal(-(ge * sy + ly))
            magnetic_momenta[2] = np.diagonal(-(ge * sz + lz))

        elif np.any(states != 0):
            # Convert states to ndarray without repetitions
            states = np.unique(np.array(states).astype(np.int64))

            # Number of states desired
            num_of_states = states.size

            #  Initialize the result array
            magnetic_momenta = np.ascontiguousarray(
                np.zeros((3, num_of_states), dtype=np.complex128)
            )

            # Slice arrays based on states_cutoff
            sx = sx[states, states]
            sy = sy[states, states]
            sz = sz[states, states]
            lx = lx[states, states]
            ly = ly[states, states]
            lz = lz[states, states]

            # Compute and save magnetic momenta in a.u.
            magnetic_momenta[0] = -(ge * sx + lx)
            magnetic_momenta[1] = -(ge * sy + ly)
            magnetic_momenta[2] = -(ge * sz + lz)

    else:
        states = np.arange(sx.shape[0], dtype=np.int64)

        #  Initialize the result array
        magnetic_momenta = np.ascontiguousarray(
            np.zeros((3, sx.shape[0]), dtype=np.complex128)
        )

        # Compute and save magnetic momenta in a.u.
        magnetic_momenta[0] = np.diagonal(-(ge * sx + lx))
        magnetic_momenta[1] = np.diagonal(-(ge * sy + ly))
        magnetic_momenta[2] = np.diagonal(-(ge * sz + lz))

    return states, magnetic_momenta.real


def get_states_total_angular_momneta(
    filename: str, group: str, states: np.ndarray = None, rotation=None
):
    (
        _,
        sx,
        sy,
        sz,
        lx,
        ly,
        lz,
    ) = get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group, rotation
    )

    if (
        (np.all(states is not None))
        and (np.all(states != None))
        and (np.any(states != 0))
    ):
        if (
            np.any(states < 0)
            or not np.issubdtype(states.dtype, np.integer)
            or np.any(states > sx.shape[0])
        ):
            raise ValueError(
                "States list contains negative values, non-integer elements"
                " or indexes greater than the number of states:"
                f" {sx.shape[0]}!"
            )

        if states.size == 1 and (np.any(states != 0)):
            total_angular_momenta = np.ascontiguousarray(
                np.zeros((3, states), dtype=np.complex128)
            )

            # Slice arrays based on states_cutoff
            sx = sx[:states, :states]
            sy = sy[:states, :states]
            sz = sz[:states, :states]
            lx = lx[:states, :states]
            ly = ly[:states, :states]
            lz = lz[:states, :states]

            # Compute and save magnetic momenta in a.u.
            total_angular_momenta[0] = np.diagonal(sx + lx)
            total_angular_momenta[1] = np.diagonal(sy + ly)
            total_angular_momenta[2] = np.diagonal(sz + lz)

        elif np.any(states != 0):
            # Convert states to ndarray without repetitions
            states = np.unique(np.array(states).astype(np.int64))

            # Number of states desired
            num_of_states = states.size

            #  Initialize the result array
            total_angular_momenta = np.ascontiguousarray(
                np.zeros((3, num_of_states), dtype=np.complex128)
            )

            # Slice arrays based on states_cutoff
            sx = sx[states, states]
            sy = sy[states, states]
            sz = sz[states, states]
            lx = lx[states, states]
            ly = ly[states, states]
            lz = lz[states, states]

            # Compute and save magnetic momenta in a.u.
            total_angular_momenta[0] = sx + lx
            total_angular_momenta[1] = sy + ly
            total_angular_momenta[2] = sz + lz

    else:
        states = np.arange(sx.shape[0], dtype=np.int64)

        #  Initialize the result array
        total_angular_momenta = np.ascontiguousarray(
            np.zeros((3, sx.shape[0]), dtype=np.complex128)
        )

        # Compute and save magnetic momenta in a.u.
        total_angular_momenta[0] = np.diagonal(sx + lx)
        total_angular_momenta[1] = np.diagonal(sy + ly)
        total_angular_momenta[2] = np.diagonal(sz + lz)

    return states, total_angular_momenta.real


def get_magnetic_momenta_matrix(
    filename: str, group: str, states_cutoff: np.ndarray, rotation=None
):
    magnetic_momenta, _ = get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    return magnetic_momenta


def get_total_angular_momneta_matrix(
    filename: str, group: str, states_cutoff: np.int64, rotation=None
):
    (
        _,
        sx,
        sy,
        sz,
        lx,
        ly,
        lz,
    ) = get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group, rotation
    )

    if (
        (not isinstance(states_cutoff, np.int))
        or (states_cutoff < 0)
        or (states_cutoff > sx.shape[0])
    ):
        raise ValueError(
            "Invalid states cutoff, set it to positive integer less than the"
            f" number of states: {sx.shape[0]} or 0 for all states."
        )

    if states_cutoff != 0:
        #  Initialize the result array
        total_angular_momenta = np.ascontiguousarray(
            np.zeros((3, states_cutoff, states_cutoff), dtype=np.complex128)
        )

        # Slice arrays based on states_cutoff
        sx = sx[:states_cutoff, :states_cutoff]
        sy = sy[:states_cutoff, :states_cutoff]
        sz = sz[:states_cutoff, :states_cutoff]
        lx = lx[:states_cutoff, :states_cutoff]
        ly = ly[:states_cutoff, :states_cutoff]
        lz = lz[:states_cutoff, :states_cutoff]

    elif states_cutoff == 0:
        #  Initialize the result array
        total_angular_momenta = np.ascontiguousarray(
            np.zeros((3, sx.shape[0], sx.shape[1]), dtype=np.complex128)
        )

    # Compute and save magnetic momenta in a.u.
    total_angular_momenta[0] = sx + lx
    total_angular_momenta[1] = sy + ly
    total_angular_momenta[2] = sz + lz

    return total_angular_momenta
