import h5py
import numpy as np
import slothpy as slt


# Create the HDF5 file and save the arrays in a group
with h5py.File('data', 'w') as f:
    group = f.create_group('lebedev_laikov_hemisphere')
    for i in range(12):
        array = slt.lebedev_laikov_grid_over_hemisphere(i).astype(np.float64)
        group.create_dataset(f'{i}', data=array, chunks=True)

print("Arrays have been successfully saved to arrays.h5 in 'arrays_group'")
