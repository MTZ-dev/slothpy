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

from os.path import join, splitext
from math import cos, sin, radians, sqrt
from ase.io import read
from slothpy.core._slothpy_exceptions import SltFileError, SltInputError
from slothpy.core.slt_file_object import SltFile
from slothpy._general_utilities._io import _orca_spin_orbit_to_slt, _molcas_to_slt, _xyz_to_slt


def hamiltonian_from_orca(
    orca_filepath: str,
    orca_filename: str,
    slt_filepath: str,
    slt_filename: str,
    group_name: str,
    pt2: bool = False,
) -> SltFile:
    """
    Create or append data to a SltFile from ORCA output file.

    Parameters
    ----------
    orca_filepath : str
        Path to the ORCA output file.
    orca_filename : str
        Name of the ORCA output file.
    slt_filepath : str
        Path of the existing or new .slt file to which the results will
        be saved.
    slt_filename : str
        Name of the .slt file to be created/accessed.
    group_name : str
        Name of a group to which results of relativistic ab initio calculations
        will be saved.
    pt2 : bool, optional
        If True the results of CASPT2/NEVPT2 second-order perturbative
        corrections will be loaded to the file., by default False

    Returns
    -------
    SltFile
        An instance of SltFile class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a SltFile from given files.

    Note
    ----
    ORCA calculations have to be done with the "printlevel 5" keyword in the
    "rel" section for outputs to be readable by SlothPy.
    """

    if slt_filename.endswith(".slt"):
        slt_filename = slt_filename[:-4]
    if not isinstance(group_name, str):
        raise SltInputError(f"The group name has to be a string not {type(group_name)}.")
    try:
        _orca_spin_orbit_to_slt(
            orca_filepath,
            orca_filename,
            slt_filepath,
            slt_filename,
            group_name,
            pt2,
        )

        return SltFile._new(slt_filepath, slt_filename)
    
    except Exception as exc:
        file = join(slt_filepath, slt_filename)
        raise SltFileError(
            file,
            exc,
            message="Failed to create a .slt file from the ORCA output file",
        ) from None


def hamiltonian_from_molcas(molcas_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool = False) -> SltFile:
    """
    Create a SltFile from MOLCAS rassi.h5 file.

    Parameters
    ----------
    molcas_filepath : str
        Path to the MOLCAS .rassi.h5 file.
    slt_filepath : str
        Path of the existing or new .slt file to which the results will
        be saved.
    group_name : str
        Name of a group to which results of relativistic ab initio calculations
        will be saved.
    edipmom : bool
        If set to True, electric dipole moment integrals will be read from
        the MOLCAS .rassi.h5 file for simulations of spectroscopic properties.

    Returns
    -------
    SltFile
        An instance of SltFile class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a SltFile from given files.

    Note
    ----
    MOLCAS calculations have to be done with the "MESO" keyword within the
    RASSI section and the installation has to support HDF5 files for .rassi.h5
    files to be readable by SlothPy.
    """

    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    if not molcas_filepath.endswith(".rassi.h5"):
        raise SltInputError(ValueError("The molcas file to be loaded must have a .rassi.h5 extension."))
    if not isinstance(group_name, str):
        raise SltInputError(ValueError(f"The group name has to be a string not {type(group_name)}."))
    try:
        _molcas_to_slt(molcas_filepath, slt_filepath, group_name, electric_dipole_momenta)

        return SltFile._new(slt_filepath)
    
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message=("Failed to create a .slt file from the MOLCAS rassi.h5 file")) from None


def slt_file(slt_filepath: str) -> SltFile:
    """
    Create a SltFile object from the existing .slt file.

    Parameters
    ----------
    slt_filepath : str
        Path to the existing .slt file to be loaded.

    Returns
    -------
    SltFile
        An instance of SltFile class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a SltFile from a given file.
    """

    if not slt_filepath.endswith(".slt"):
        raise SltInputError(ValueError("The file to be loaded must have a .slt extension."))
    try:
        return SltFile._new(slt_filepath)
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message="Failed to load SltFIle from the .slt file.") from None


def xyz(input_filepath: str, slt_filepath: str, group_name: str) -> SltFile:
    """
    Reads a .xyz or .cif file along with cell parameters and saves the data
    as a xyz group in a .slt file.

    Parameters:
    ----------
    input_filepath : str
        Path to the input .xyz or .cif file.
    slt_filepath : str
        Path of the existing or new .slt file to which the results will
        be saved.
    group_name : str
        Name of a group to which unit cell will be saved.

    Raises:
    ------
    SltInputError:
        If input_file has an unsupported extension
    SltFileError:
        If there are issues reading the input file or writing to the .slt file.
    """

    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    try:
        _, ext = splitext(input_filepath)
        ext = ext.lower()
        if ext not in ['.xyz', '.cif']:
            raise ValueError("Unsupported file extension. Only .cif and .xyz files are supported.")
        try:
            atoms = read(input_filepath)
            elements = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
        except Exception as exc:
            raise IOError(f"Error reading the input file '{input_filepath}': {exc}")
    except Exception as exc:
        raise SltInputError(exc, message="Failed to save xyz to .slt file.") from None
    
    try:
        _xyz_to_slt(slt_filepath, group_name, elements, positions)

        return SltFile._new(slt_filepath)
    
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message="Failed to save unit cell to .slt file.") from None


def unit_cell(input_file, slt_file, group_name, cell_vectors=None, cell_params=None):
    """
    Reads a .cif or .xyz file along with cell parameters and saves the data
    as a unit cell group in a .slt file.

    Parameters:
    ----------
    input_file : str
        Path to the input .cif or .xyz file.
    slt_file : str
        Path of the existing or new .slt file to which the results will
        be saved.
    group_name : str
        Name of a group to which unit cell will be saved.
    cell_vectors : list or ndarray, optional
        A 3x3 list or NumPy array (ArrayLike structure) representing the unit
        cell vectors. Required if input_file is a .xyz file and cell_params is
        not provided.
    cell_params : list, tuple, or ndarray optional
        An ArrayLike structure containing [a, b, c, alpha, beta, gamma],
        where a, b, c are the unit cell lengths and alpha, beta, gamma are the
        unit cell angles in degrees. Required if input_file is a .xyz file
        and cell_vectors is not provided.

    Raises:
    ------
    SltInputError:
        If input_file has an unsupported extension or required cell parameters
        are not provided for .xyz files.
    SltFileError:
        If there are issues reading the input file or writing to the .slt file.
    """
    # try:
    #     _, ext = splitext(input_file)
    #     ext = ext.lower()
    #     if ext not in ['.cif', '.xyz']:
    #         raise ValueError("Unsupported file extension. Only .cif and .xyz files are supported.")

    #     try:
    #         atoms = read(input_file)
    #     except Exception as exc:
    #         raise IOError(f"Error reading the input file '{input_file}': {exc}")

    #     cell = None

    #     if ext == '.cif':
    #         cell = atoms.get_cell().array
    #     elif ext == '.xyz':
    #         if cell_params is not None:
    #             if len(cell_params) != 6:
    #                 raise ValueError("The lattice_params must contain exactly six values: [a, b, c, alpha, beta, gamma].")
    #             a, b, c, alpha_deg, beta_deg, gamma_deg = cell_params
    #             alpha = radians(alpha_deg)
    #             beta = radians(beta_deg)
    #             gamma = radians(gamma_deg)

    #             cell = np.zeros((3, 3), dtype=np.float64)
    #             cell[0, 0] = a
    #             cell[0, 1] = 0.0
    #             cell[0, 2] = 0.0

    #             cell[1, 0] = b * cos(gamma)
    #             cell[1, 1] = b * sin(gamma)
    #             cell[1, 2] = 0.0

    #             # Calculate the z-component of the third cell vector
    #             c_x = c * cos(beta)
    #             c_y = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
    #             c_z_squared = c**2 - c_x**2 - c_y**2
    #             if c_z_squared < 0:
    #                 raise ValueError("Invalid lattice parameters leading to a negative z-component squared.")
    #             c_z = sqrt(c_z_squared)
    #             cell[2, 0] = c_x
    #             cell[2, 1] = c_y
    #             cell[2, 2] = c_z
    #         elif cell_params is not None:
    #             cell = np.array(cell_params, dtype=np.float64)
    #             if cell.shape != (3, 3):
    #                 raise ValueError("cell_params must be a 3x3 matrix.")
    #         else:
    #             raise ValueError("For .xyz files, either cell_params or lattice_params must be provided.")
    #     else:
    #         # This should not happen due to earlier check
    #         raise ValueError("Unsupported file extension.")

    #     if cell is None:
    #         raise ValueError("Unit cell parameters could not be determined.")

    #     # Extract atomic information
    #     elements = atoms.get_chemical_symbols()
    #     positions = atoms.get_positions()  # Nx3 array

    # except Exception as exc:
    #     raise SltInputError(exc, message="Failed to save unit cell to .slt file.") from None
    
    # # Create or open the HDF5 file
    # try:
    #     with h5py.File(output_hdf, 'a') as hdf:
    #         # Create the group if it doesn't exist
    #         if group_name in hdf:
    #             print(f"Group '{group_name}' already exists in '{output_hdf}'. Overwriting datasets.")
    #             grp = hdf[group_name]
    #         else:
    #             grp = hdf.create_group(group_name)

    #         # Save cell parameters
    #         if 'cell' in grp:
    #             print("Dataset 'cell' already exists. Overwriting.")
    #             del grp['cell']
    #         grp.create_dataset('cell', data=cell, dtype='f8')

    #         # Save elements
    #         if 'elements' in grp:
    #             print("Dataset 'elements' already exists. Overwriting.")
    #             del grp['elements']
    #         dt = h5py.string_dtype(encoding='utf-8')
    #         grp.create_dataset('elements', data=np.array(elements, dtype='S'), dtype=dt)

    #         # Save positions
    #         if 'positions' in grp:
    #             print("Dataset 'positions' already exists. Overwriting.")
    #             del grp['positions']
    #         grp.create_dataset('positions', data=positions, dtype='f8')

    #         print(f"Unit cell data successfully saved to group '{group_name}' in '{output_hdf}'.")

    # except Exception as exc:
    #     raise SltFileError(slt_file, exc, message="Failed to save unit cell to .slt file.") from None
    pass
