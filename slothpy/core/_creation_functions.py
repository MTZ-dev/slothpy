from slothpy.core._compound_object import Compound
from slothpy.general_utilities.io import orca_spin_orbit_to_slt

def compound_from_orca( hdf5_filepath: str, hdf5_filename: str, name: str, orca_filepath: str, orca_filename: str, pt2: bool = False) -> Compound:
    
    try:
        orca_spin_orbit_to_slt(orca_filepath, orca_filename, hdf5_filepath, hdf5_filename, name, pt2)
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(f'Failed to create a HDF5 file from the ORCA output file or acces the exsiting one.')
        print(f'Error encountered while trying to create Compound: {error_type}: {error_message}')
        return
    
    obj = Compound._new(hdf5_filepath, hdf5_filename)

    return obj
    

def compound_from_slt(hdf5_filepath, hdf5_filename):
    
    try:
        obj = Compound._new(hdf5_filepath, hdf5_filename)
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(f'Failed to load data from a HDF5 file.')
        print(f'Error encountered while trying to create Compound: {error_type}: {error_message}')
        return
    
    return obj



