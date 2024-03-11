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

from os.path import join
from slothpy.core.compound_object import Compound
from slothpy._general_utilities._io import (
    _orca_spin_orbit_to_slt,
    _molcas_spin_orbit_to_slt,
)
from slothpy.core._slothpy_exceptions import SltFileError


def compound_from_orca(
    orca_filepath: str,
    orca_filename: str,
    slt_filepath: str,
    slt_filename: str,
    group_name: str,
    pt2: bool = False,
) -> Compound:
    """
    Create a Compound from ORCA output file.

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
    Compound
        An instance of Compound class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a Compound from given files.

    Note
    ----
    ORCA calculations have to be done with the "printlevel 3" keyword for
    outputs to be readable by SlothPy.
    """

    if slt_filename.endswith(".slt"):
        slt_filename = slt_filename[:-4]
    try:
        _orca_spin_orbit_to_slt(
            orca_filepath,
            orca_filename,
            slt_filepath,
            slt_filename,
            group_name,
            pt2,
        )
    except Exception as exc:
        file = join(slt_filepath, slt_filename)
        raise SltFileError(
            file,
            exc,
            message="Failed to create a .slt file from the ORCA output file",
        ) from None

    return Compound._new(slt_filepath, slt_filename)


def compound_from_molcas(
    molcas_filepath: str,
    molcas_filename: str,
    slt_filepath: str,
    slt_filename: str,
    name: str,
) -> Compound:
    """
    Create a Compound from MOLCAS rassi.h5 file.

    Parameters
    ----------
    molcas_filepath : str
        Path to the MOLCAS .rassi.h5 file.
    molcas_filename : str
        Name of the MOLCAS .rassi.h5 file (without the suffix).
    slt_filepath : str
        Path of the existing or new .slt file to which the results will
        be saved.
    slt_filename : str
        Name of the .slt file to be created/accessed.
    name : str
        Name of a group to which results of relativistic ab initio calculations
        will be saved.

    Returns
    -------
    Compound
        An instance of Compound class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a Compound from given files.

    Note
    ----
    MOLCAS calculations have to be done with the "MESO" keyword within the
    RASSI section and the installation has to support HDF5 files for .rassi.h5
    files to be readable by SlothPy.
    """

    if slt_filename.endswith(".slt"):
        slt_filename = slt_filename[:-4]
    try:
        _molcas_spin_orbit_to_slt(
            molcas_filepath,
            molcas_filename,
            slt_filepath,
            slt_filename,
            name,
        )
    except Exception as exc:
        file = join(slt_filepath, slt_filename)
        raise SltFileError(
            file,
            exc,
            message=(
                "Failed to create a .slt file from the MOLCAS rassi.h5 file"
            ),
        ) from None

    return Compound._new(slt_filepath, slt_filename)


def compound_from_slt(slt_filepath: str, slt_filename: str) -> Compound:
    """
    Create a Compound from the existing .slt file.

    Parameters
    ----------
    slt_filepath : str
        Path to the existing .slt file to be loaded.
    slt_filename : str
        Name of an existing .slt file to be loaded.

    Returns
    -------
    Compound
        An instance of Compound class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a Compound from a given file.
    """

    if slt_filename.endswith(".slt"):
        slt_filename = slt_filename[:-4]
    try:
        return Compound._new(slt_filepath, slt_filename)
    except Exception as exc:
        file = join(slt_filepath, slt_filename)
        raise SltFileError(
            file,
            exc,
            message="Failed to load Compound from the .slt file.",
        ) from None
