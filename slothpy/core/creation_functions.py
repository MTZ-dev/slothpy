from slothpy.core.compound_object import Compound
from slothpy.general_utilities._io import (
    _orca_spin_orbit_to_slt,
    _molcas_spin_orbit_to_slt,
)


def compound_from_orca(
    hdf5_filepath: str,
    hdf5_filename: str,
    name: str,
    orca_filepath: str,
    orca_filename: str,
    pt2: bool = False,
) -> Compound:
    try:
        _orca_spin_orbit_to_slt(
            orca_filepath,
            orca_filename,
            hdf5_filepath,
            hdf5_filename,
            name,
            pt2,
        )
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(
            f"Failed to create a HDF5 file from the ORCA output file or acces"
            f" the exsiting one."
        )
        raise Exception(
            f"Error encountered while trying to create Compound: {error_type}:"
            f" {error_message}"
        )

    obj = Compound._new(hdf5_filepath, hdf5_filename)

    return obj


def compound_from_molcas(
    hdf5_filepath: str,
    hdf5_filename: str,
    name: str,
    molcas_filepath: str,
    molcas_filename: str,
) -> Compound:
    try:
        _molcas_spin_orbit_to_slt(
            molcas_filepath,
            molcas_filename,
            hdf5_filepath,
            hdf5_filename,
            name,
        )
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(
            f"Failed to create a HDF5 file from the MOLCAS files or acces the"
            f" exsiting one."
        )
        raise Exception(
            f"Error encountered while trying to create Compound: {error_type}:"
            f" {error_message}"
        )

    obj = Compound._new(hdf5_filepath, hdf5_filename)

    return obj


def compound_from_slt(hdf5_filepath, hdf5_filename):
    try:
        obj = Compound._new(hdf5_filepath, hdf5_filename)
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(f"Failed to load data from a HDF5 file.")
        raise Exception(
            f"Error encountered while trying to create Compound: {error_type}:"
            f" {error_message}"
        )

    return obj
