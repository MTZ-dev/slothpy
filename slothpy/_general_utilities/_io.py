# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from os import remove
from os.path import join
from re import compile, search
from h5py import File
from numpy import (
    ndarray,
    array,
    ascontiguousarray,
    zeros,
    empty,
    any,
    diagonal,
    int64,
    float64,
    complex128,
)
from numpy.linalg import eigh, eigvalsh
from typing import Tuple
from slothpy._general_utilities._constants import YELLOW, RESET, H_CM_1
from slothpy.core._slothpy_exceptions import SltFileError, SltReadError
from slothpy._angular_momentum._rotation import _rotate_vector_operator
from slothpy._general_utilities._math_expresions import (
    _magnetic_momenta_from_angular_momenta,
    _total_angular_momenta_from_angular_momenta,
)
from slothpy.core._config import settings

def _grep_to_file(
    path_inp: str,
    inp_file: str,
    pattern: str,
    path_out: str,
    out_file: str,
    lines: int = 1,
) -> None:
    """
    Extracts lines from an input file that match a specified pattern and saves
    them to an output file.

    Args:
        path_inp (str): Path to the input file.
        inp_file (str): Name of the input file.
        pattern (str): Pattern to search for in the input file.
        path_out (str): Path to the output file.
        out_file (str): Name of the output file.
        lines (int, optional): Number of lines to save after a matching pattern
        is found. Defaults to 1 (save only the line with pattern occurrence).

    Raises:
        ValueError: If the pattern is not found in the input file.
    """
    regex = compile(pattern)

    input_file = join(path_inp, inp_file)
    output_file = join(path_out, out_file)

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


def _group_exists(hdf5_file, group_name: str):
    with File(hdf5_file, "r") as file:
        return group_name in file


def _dataset_exists(hdf5_file, group_name, dataset_name):
    with File(hdf5_file, "r") as file:
        if group_name in file:
            group = file[group_name]
            return dataset_name in group

        return False


def _get_orca_so_blocks_size(
    path: str, orca_file: str
) -> tuple[int, int, int, int]:
    """
    Retrieves the dimensions and block sizes for spin-orbit calculations from
    an ORCA file.

    Args:
        path (str): Path to the ORCA file.
        orca_file (str): Name of the ORCA file.

    Returns:
        tuple[int, int, int, int]: A tuple containing the spin-orbit dimension,
        block size, number of whole blocks, and remaining columns.

    Raises:
        ValueError: If the spin-orbit dimension is not found in the ORCA file.
    """
    orca_file_path = join(path, orca_file)

    with open(orca_file_path, "r") as file:
        content = file.read()

    so_dim_match = search(r"Dim\(SO\)\s+=\s+(\d+)", content)
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


def _orca_spin_orbit_to_slt(
    path_orca: str,
    inp_orca: str,
    path_out: str,
    hdf5_output: str,
    name: str,
    pt2: bool = False,
) -> None:
    """
    Converts spin-orbit calculations from an ORCA .out file to
    a HDF5 file format.

    Args:
        path_orca (str): Path to the ORCA file.
        inp_orca (str): Name of the ORCA file.
        path_out (str): Path for the output files.
        hdf5_output (str): Name of the HDF5 output file.
        pt2 (bool): Get results from the second-order perturbation-corrected
                    states.

    Raises:
        ValueError: If the spin-orbit dimension is not found in the ORCA file.
        ValueError: If the pattern is not found in the input file.
    """
    hdf5_file = join(path_out, hdf5_output)

    # Retrieve dimensions and block sizes for spin-orbit calculations
    (
        so_dim,
        block_size,
        num_of_whole_blocks,
        remaining_columns,
    ) = _get_orca_so_blocks_size(path_orca, inp_orca)

    # Create HDF5 file and ORCA group
    output = File(f"{hdf5_file}.slt", "a")
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
        out_file_name = f"{matrix_name}.tmp"
        out_file = join(path_out, out_file_name)
        pattern = f"{matrix_name} MATRIX IN CI BASIS\n"

        # Extract lines matching the pattern to a temporary file
        _grep_to_file(
            path_orca,
            inp_orca,
            pattern,
            path_out,
            out_file_name,
            block_size + 4,
        )  # The first 4 lines are titles

        with open(out_file, "r") as file:
            if pt2:
                for _ in range(block_size + 4):
                    file.readline()  # Skip non-pt2 block
            for _ in range(4):
                file.readline()  # Skip the first 4 lines
            matrix = empty((so_dim, so_dim), dtype=float64)
            l = 0
            for _ in range(num_of_whole_blocks):
                file.readline()  # Skip a line before each block of 6 columns
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(6):
                        matrix[i, l + j] = float64(line[j + 1])
                l += 6

            if remaining_columns > 0:
                file.readline()  # Skip a line before the remaining columns
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(remaining_columns):
                        matrix[i, l + j] = float64(line[j + 1])

            # Create dataset in HDF5 file and assign the matrix
            dataset = orca.create_dataset(
                f"SF_{matrix_name}", shape=(so_dim, so_dim), dtype=float64
            )
            dataset[:, :] = matrix[:, :]
            dataset.attrs["Description"] = (
                f"Dataset containing {description} of SF_{matrix_name} matrix"
                " in CI basis (spin-free)"
            )

        # Remove the temporary file
        remove(out_file)

    # Extract and process SOC matrix
    _grep_to_file(
        path_orca,
        inp_orca,
        r"SOC MATRIX \(A\.U\.\)\n",
        path_out,
        "SOC.tmp",
        2 * block_size + 4 + 2,
    )  # The first 4 lines are titles, two lines in the middle for Im part

    soc_file = join(path_out, "SOC.tmp")

    with open(soc_file, "r") as file:
        content = file.read()

    # Replace "-" with " -"
    modified_content = content.replace("-", " -")

    # Write the modified content to the same file
    with open(soc_file, "w") as file:
        file.write(modified_content)

    with open(soc_file, "r") as file:
        if pt2:
            for _ in range(2 * block_size + 4 + 2):
                file.readline()  # Skip non-pt2 block
        for _ in range(4):
            file.readline()  # Skip the first 4 lines
        matrix_real = empty((so_dim, so_dim), dtype=float64)
        l = 0
        for _ in range(num_of_whole_blocks):
            file.readline()  # Skip a line before each block of 6 columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_real[i, l + j] = float64(line[j + 1])
            l += 6

        if remaining_columns > 0:
            file.readline()  # Skip a line before the remaining columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(remaining_columns):
                    matrix_real[i, l + j] = float64(line[j + 1])

        for _ in range(2):
            file.readline()  # Skip 2 lines separating real and imaginary part

        matrix_imag = empty((so_dim, so_dim), dtype=float64)
        l = 0
        for _ in range(num_of_whole_blocks):
            file.readline()  # Skip a line before each block of 6 columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_imag[i, l + j] = float64(line[j + 1])
            l += 6

        if remaining_columns > 0:
            file.readline()  # Skip a line before the remaining columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(remaining_columns):
                    matrix_imag[i, l + j] = float64(line[j + 1])

    # Create a dataset in HDF5 file for SOC matrix
    dataset = orca.create_dataset(
        "SOC", shape=(so_dim, so_dim), dtype=complex128
    )
    complex_matrix = array(matrix_real + 1j * matrix_imag, dtype=complex128)
    dataset[:, :] = complex_matrix[:, :]
    dataset.attrs[
        "Description"
    ] = "Dataset containing complex SOC matrix in CI basis (spin-free)"

    # Remove the temporary file
    remove(soc_file)

    # Close the HDF5 file
    output.close()


def _molcas_spin_orbit_to_slt(
    path_molcas: str,
    inp_molcas: str,
    path_out: str,
    hdf5_output: str,
    group_name: str,
    edipmom: bool = False,
) -> None:
    rassi_path_name = join(path_molcas, inp_molcas)
    hdf5_file = join(path_out, hdf5_output)

    with File(f"{rassi_path_name}.rassi.h5", "r") as rassi:
        with File(f"{hdf5_file}.slt", "a") as output:
            group = output.create_group(group_name)
            group.attrs["Type"] = "Hamiltonian"
            group.attrs["Kind"] = "MOLCAS"
            group.attrs["Precision"] = settings.precision
            if edipmom:
                group.attrs["Additional"] = "Edipmom"
            group.attrs["Description"] = "Group containing relativistic SOC MOLCAS results."

            dataset_rassi = rassi["SOS_ENERGIES"][:]
            group.attrs["States"] = dataset_rassi.shape[0]
            dataset_out = group.create_dataset("SOC_ENERGIES", shape=dataset_rassi.shape, dtype=settings.float, data=dataset_rassi.astype(settings.float), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing SOC energies."

            dataset_rassi = rassi["SOS_SPIN_REAL"][0, :, :] + 1j * rassi["SOS_SPIN_IMAG"][0, :, :]
            dataset_out = group.create_dataset("SOC_SX", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing the Sx matrix in the SOC basis."
            dataset_rassi = rassi["SOS_SPIN_REAL"][1, :, :] + 1j * rassi["SOS_SPIN_IMAG"][1, :, :]
            dataset_out = group.create_dataset("SOC_SY", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing the Sy matrix in the SOC basis."
            dataset_rassi = rassi["SOS_SPIN_REAL"][2, :, :] + 1j * rassi["SOS_SPIN_IMAG"][2, :, :]
            dataset_out = group.create_dataset("SOC_SZ", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing the Sz matrix in the SOC basis."

            dataset_rassi = 1j * rassi["SOS_ANGMOM_REAL"][0, :, :] - rassi["SOS_ANGMOM_IMAG"][0, :, :]
            dataset_out = group.create_dataset("SOC_LX", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing the Lx matrix in the SOC basis."
            dataset_rassi = rassi["SOS_ANGMOM_REAL"][1, :, :] - rassi["SOS_ANGMOM_IMAG"][1, :, :]
            dataset_out = group.create_dataset("SOC_LY", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing the Ly matrix in the SOC basis."
            dataset_rassi = rassi["SOS_ANGMOM_REAL"][2, :, :] - rassi["SOS_ANGMOM_IMAG"][2, :, :]
            dataset_out = group.create_dataset("SOC_LZ", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Dataset containing the Lz matrix in the SOC basis."

            if edipmom:
                dataset_rassi = rassi["SOS_EDIPMOM_REAL"][0, :, :] + 1j * rassi["SOS_EDIPMOM_REAL"][0, :, :]
                dataset_out = group.create_dataset("SOC_EDIPMOMX", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
                dataset_out.attrs["Description"] = "Dataset containing the Px electric dipole moment matrix in the SOC basis."
                dataset_rassi = rassi["SOS_EDIPMOM_REAL"][1, :, :] + 1j * rassi["SOS_EDIPMOM_REAL"][1, :, :]
                dataset_out = group.create_dataset("SOC_EDIPMOMY", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
                dataset_out.attrs["Description"] = "Dataset containing the Py electric dipole moment matrix in the SOC basis."
                dataset_rassi = rassi["SOS_EDIPMOM_REAL"][2, :, :] + 1j * rassi["SOS_EDIPMOM_REAL"][2, :, :]
                dataset_out = group.create_dataset("SOC_EDIPMOMZ", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
                dataset_out.attrs["Description"] = "Dataset containing the Pz electric dipole moment matrix in the SOC basis."


def _load_orca_hdf5(filename, group, rotation):
    try:
        # Read data from HDF5 file
        with File(filename, "r") as file:
            shape = file[str(group)]["SOC"][:].shape[0]
            angular_momenta = zeros((6, shape, shape), dtype=complex128)
            soc_mat = file[str(group)]["SOC"][:]
            angular_momenta[0][:] = 0.5 * file[str(group)]["SF_SX"][:]
            angular_momenta[1][:] = 0.5j * file[str(group)]["SF_SY"][:]
            angular_momenta[2][:] = 0.5 * file[str(group)]["SF_SZ"][:]
            angular_momenta[3][:] = 1j * file[str(group)]["SF_LX"][:]
            angular_momenta[4][:] = 1j * file[str(group)]["SF_LY"][:]
            angular_momenta[5][:] = 1j * file[str(group)]["SF_LZ"][:]

        # Perform diagonalization of SOC matrix
        soc_energies, eigenvectors = eigh(soc_mat)

        angular_momenta = ascontiguousarray(angular_momenta)
        soc_energies = ascontiguousarray(soc_energies.astype(float64))
        eigenvectors = ascontiguousarray(eigenvectors.astype(complex128))

        # Apply transformations to spin and orbital operators
        angular_momenta = (
            eigenvectors.conj().T @ angular_momenta @ eigenvectors
        )

        if (rotation is not None) or (rotation != None):
            angular_momenta[0:3, :, :] = _rotate_vector_operator(
                angular_momenta[0:3, :, :], rotation
            )
            angular_momenta[3:6, :, :] = _rotate_vector_operator(
                angular_momenta[3:6, :, :], rotation
            )

        # Return operators in SOC basis
        return soc_energies, angular_momenta

    except Exception as exc:
        raise SltFileError(
            filename,
            exc,
            message=(
                "Failed to load SOC, spin, and angular momenta data in"
                " the ORCA format from the .slt file."
            ),
        ) from None


def _load_molcas_hdf5(filename, group, rotation, edipmom = False):
    try:
        with File(filename, "r") as file:
            shape = file[group]["SOC_ENERGIES"][:].shape[0]
            angular_momenta = zeros((6, shape, shape), dtype=settings.complex)
            soc_energies = file[group]["SOC_ENERGIES"][:]
            angular_momenta[0][:] = file[group]["SOC_SX"][:]
            angular_momenta[1][:] = file[group]["SOC_SY"][:]
            angular_momenta[2][:] = file[group]["SOC_SZ"][:]
            angular_momenta[3][:] = file[group]["SOC_LX"][:]
            angular_momenta[4][:] = file[group]["SOC_LY"][:]
            angular_momenta[5][:] = file[group]["SOC_LZ"][:]

            angular_momenta = ascontiguousarray(angular_momenta)
            soc_energies = ascontiguousarray(soc_energies.astype(settings.float))

            if (rotation is not None) or (rotation != None):
                angular_momenta[0:3, :, :] = _rotate_vector_operator(
                    angular_momenta[0:3, :, :], rotation
                )
                angular_momenta[3:6, :, :] = _rotate_vector_operator(
                    angular_momenta[3:6, :, :], rotation
                )
            
            if edipmom:
                if file[group].attrs["Additional"] == "Edipmom":
                    shape = file[group]["SOC_EDIPMOMX"][:].shape[0]
                    electric_dipolar_momenta = zeros((3, shape, shape), dtype=settings.complex)
                    electric_dipolar_momenta[0][:] = file[group]["SOC_EDIPMOMX"][:]
                    electric_dipolar_momenta[1][:] = file[group]["SOC_EDIPMOMY"][:]
                    electric_dipolar_momenta[2][:] = file[group]["SOC_EDIPMOMZ"][:]

                    if (rotation is not None) or (rotation != None):
                        electric_dipolar_momenta = _rotate_vector_operator(
                            electric_dipolar_momenta, rotation
                        )

                    electric_dipolar_momenta = ascontiguousarray(electric_dipolar_momenta)

                    return soc_energies, angular_momenta, electric_dipolar_momenta

                else:
                    raise KeyError(f"Group '{group}' does not contain electric dipolar momenta in a valid format.")

            return soc_energies, angular_momenta

    except Exception as exc:
        raise SltFileError(
            filename,
            exc,
            message=(
                "Failed to load SOC, spin, angular momenta or electric dipole momenta data in"
                " the MOLCAS format from the .slt file."
            ),
        ) from None


def _get_soc_energies_and_soc_angular_momenta_from_hdf5(
    filename: str, group: str, rotation: ndarray = None
) -> Tuple[ndarray, ndarray]:
    if _dataset_exists(filename, group, "SOC"):
        return _load_orca_hdf5(filename, group, rotation)

    if _dataset_exists(filename, group, "SOC_energies"):
        return _load_molcas_hdf5(filename, group, rotation)

    else:
        raise SltReadError(
            filename,
            ValueError(""),
            message=(
                "Incorrect group name or data format. The program was unable"
                " to read SOC, spin, and angular momenta."
            ),
        )


def _get_soc_magnetic_momenta_and_energies_from_hdf5(
    filename: str, group: str, states_cutoff: int, rotation: ndarray = None
) -> Tuple[ndarray, ndarray]:
    (
        soc_energies,
        angular_momenta,
    ) = _get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group, rotation
    )

    soc_energies = soc_energies[:states_cutoff] - soc_energies[0]
    magnetic_momenta = _magnetic_momenta_from_angular_momenta(
        angular_momenta, 0, states_cutoff
    )

    return magnetic_momenta, soc_energies


def _get_soc_total_angular_momenta_and_energies_from_hdf5(
    filename: str, group: str, states_cutoff: int, rotation=None
) -> Tuple[ndarray, ndarray]:
    (
        soc_energies,
        angular_momenta,
    ) = _get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group, rotation
    )

    soc_energies = soc_energies[:states_cutoff] - soc_energies[0]
    total_angular_momenta = _total_angular_momenta_from_angular_momenta(
        angular_momenta, 0, states_cutoff
    )

    return total_angular_momenta, soc_energies


def _get_soc_energies_cm_1(
    filename: str, group: str, num_of_states: int = 0
) -> ndarray:
    if num_of_states < 0 or (not isinstance(num_of_states, int)):
        raise ValueError(
            "Invalid number of states. Set it to positive integer or 0 for"
            " all states."
        )
    with File(filename, "r") as file:
        try:
            soc_matrix = file[str(group)]["SOC"][:]
            soc_energies = eigvalsh(soc_matrix)
        except Exception as exc1:
            error_message_1 = str(exc1)
            try:
                soc_energies = file[str(group)]["SOC_energies"][:]
            except Exception as exc2:
                error_message_2 = str(exc2)
                if error_message_1 == error_message_2:
                    error_message_2 = ""
                raise RuntimeError(
                    f"{error_message_1}. {error_message_2}."
                ) from None

        if num_of_states > soc_energies.shape[0]:
            raise ValueError(
                "Invalid number of states. Set it less or equal to the overal"
                f" number of SOC states: {soc_energies.shape[0]}"
            )

        if num_of_states != 0:
            soc_energies = soc_energies[:num_of_states]

        # Return operators in SOC basis
        return (soc_energies - soc_energies[0]) * H_CM_1


def _get_states_magnetic_momenta(
    filename: str, group: str, states: ndarray = None, rotation: ndarray = None
):
    if any(states < 0):
        raise ValueError("States list contains negative values.")

    if isinstance(states, int):
        states_cutoff = states
        (
            magnetic_momenta,
            _,
        ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
            filename, group, states_cutoff, rotation
        )
        magnetic_momenta = diagonal(magnetic_momenta, axis1=1, axis2=2)
    else:
        states = array(states, dtype=int64)
        if states.ndim == 1:
            states_cutoff = max(states)
            (
                magnetic_momenta,
                _,
            ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
                filename, group, states_cutoff, rotation
            )
            magnetic_momenta = magnetic_momenta[:, states, states]
        else:
            raise ValueError("The list of states has to be a 1D array.")

    return magnetic_momenta.real


def _get_states_total_angular_momenta(
    filename: str, group: str, states: ndarray = None, rotation: ndarray = None
):
    if any(states < 0):
        raise ValueError("States list contains negative values.")

    if isinstance(states, int):
        states_cutoff = states
        (
            total_angular_momenta,
            _,
        ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
            filename, group, states_cutoff, rotation
        )
        total_angular_momenta = diagonal(
            total_angular_momenta, axis1=1, axis2=2
        )
    else:
        states = array(states, dtype=int64)
        if states.ndim == 1:
            states_cutoff = max(states)
            (
                total_angular_momenta,
                _,
            ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
                filename, group, states_cutoff, rotation
            )
            total_angular_momenta = total_angular_momenta[:, states, states]
        else:
            raise ValueError("The list of states has to be a 1D array.")

    return total_angular_momenta.real


def _get_magnetic_momenta_matrix(
    filename: str, group: str, states_cutoff: int, rotation: ndarray = None
):
    magnetic_momenta, _ = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    return magnetic_momenta


def _get_total_angular_momneta_matrix(
    filename: str, group: str, states_cutoff: int, rotation: ndarray = None
):
    (
        total_angular_momenta,
        _,
    ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    return total_angular_momenta
