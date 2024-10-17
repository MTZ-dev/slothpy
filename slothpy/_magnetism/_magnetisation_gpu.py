from typing import Literal
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
from numpy import (
    ndarray,
    dtype,
    array,
    vdot,
    sum,
    zeros,
    ascontiguousarray,
    diag,
    newaxis,
    exp,
    float64,
    complex128,
    array_equal,
    concatenate,
)
from numpy.linalg import eigh
from numba import jit, set_num_threads
from slothpy._general_utilities._constants import KB, MU_B
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._magnetism._zeeman import _calculate_zeeman_matrix
from slothpy._general_utilities._grids_over_sphere import (
    _fibonacci_over_sphere,
    _meshgrid_over_sphere_flatten,
)
import cupy as cp


# def _mth(
#     filename: str,
#     group: str,
#     fields: ndarray[float64],
#     grid: ndarray[float64],
#     temperatures: ndarray[float64],
#     states_cutoff: int,
#     num_cpu: int,
#     num_threads: int,
#     rotation: ndarray[float64] = None,
# ) -> ndarray:
#     # Read data from HDF5 file
#     (
#         magnetic_momenta,
#         soc_energies,
#     ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
#         filename, group, states_cutoff, rotation
#     )

#     #  Allocate GPU arrays as contiguous
#     magnetic_momenta_cp = cp.array(magnetic_momenta, dtype=cp.complex128)
#     soc_energies_cp = cp.array(soc_energies, dtype=cp.float64)
#     fields_cp = cp.array(fields, dtype=cp.float64)
#     temperatures_cp = cp.array(temperatures, dtype=cp.float64)
#     grid_cp = cp.array(grid, dtype=cp.float64)
#     magnetic_momenta_cp = cp.ascontiguousarray(
#         magnetic_momenta_cp, dtype=cp.complex128
#     )
#     soc_energies_cp = cp.ascontiguousarray(soc_energies_cp, dtype=cp.float64)
#     fields_cp = cp.ascontiguousarray(fields_cp, dtype=cp.float64)
#     temperatures_cp = cp.ascontiguousarray(temperatures_cp, dtype=cp.float64)
#     grid_cp = cp.ascontiguousarray(grid_cp, dtype=cp.float64)
#     mth_array = cp.zeros(
#         (temperatures.shape[0], fields.shape[0]), dtype=cp.float64
#     )

#     del magnetic_momenta
#     del soc_energies
#     del fields
#     del grid
#     del temperatures

#     streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_cpu)]

#     for i in range(fields_cp.shape[0]):
#         for j in range(grid_cp.shape[0]):
#             with streams[(i * grid_cp.shape[0] + j) % num_cpu]:
#                 _magnetisation(
#                     magnetic_momenta_cp,
#                     soc_energies_cp,
#                     fields_cp,
#                     temperatures_cp,
#                     grid_cp,
#                     mth_array,
#                     i,
#                     j,
#                 )

#         # Synchronize all streams
#         for stream in streams:
#             stream.synchronize()

#     return mth_array  # Returning values in Bohr magnetons


# def _magnetisation(
#     magnetic_momenta,
#     soc_energies,
#     fields,
#     temperatures,
#     grid,
#     mth_array,
#     field_i,
#     grid_j,
# ):
#     zeeman_matrix = cp.einsum(
#         "ijk,i->jk",
#         magnetic_momenta,
#         -fields[field_i] * MU_B * grid[grid_j, :3],
#     )
#     cp.fill_diagonal(zeeman_matrix, cp.diag(zeeman_matrix) + soc_energies)
#     zeeman_energies, eigenvectors = cp.linalg.eigh(zeeman_matrix)
#     zeeman_matrix = eigenvectors.conj().T @ -zeeman_matrix @ eigenvectors
#     KB_temps = KB * temperatures[cp.newaxis, :]
#     zeeman_expanded = cp.exp(
#         -(zeeman_energies[:, cp.newaxis] - zeeman_energies[0]) / KB_temps
#     )
#     mth_array[:, field_i] += cp.dot(
#         cp.diag(zeeman_matrix).real.astype(cp.float64), zeeman_expanded
#     ) / cp.sum(zeeman_expanded, axis=0)
#     print(f"{field_i}, {grid_j}")


# import cupy as cp


# def _magnetisation(magnetic_momenta, soc_energies, fields, temperatures, grid):
#     # Reshape and broadcast arrays for vectorized operations
#     fields_reshaped = fields[:, cp.newaxis, cp.newaxis]  # Shape: (F, 1, 1)
#     grid_reshaped = grid[:, :3][cp.newaxis, :, :]  # Shape: (1, G, 3)
#     # Perform vectorized calculations
#     orientation = -fields_reshaped * MU_B * grid_reshaped  # Shape: (F, G, 3)
#     zeeman_matrix = cp.einsum(
#         "ijk,fgk->fgij", magnetic_momenta, orientation
#     )  # Shape: (F, N, N, G)
#     cp.fill_diagonal(
#         zeeman_matrix,
#         cp.diag(zeeman_matrix) + soc_energies[cp.newaxis, cp.newaxis, :],
#     )

#     # Eigenvalues calculation (requires loop if eigenvectors are needed)
#     zeeman_energies = cp.array(
#         [
#             cp.linalg.eigvalsh(zeeman_matrix[f, g, :, :])
#             for f in range(fields.shape[0])
#             for g in range(grid.shape[0])
#         ]
#     )
#     zeeman_energies = zeeman_energies.reshape(
#         fields.shape[0], grid.shape[0], -1
#     )  # Reshape to (F, G, N)

#     temperatures_reshaped = temperatures[
#         cp.newaxis, cp.newaxis, cp.newaxis, :
#     ]  # Shape: (1, 1, T)
#     # More vectorized operations
#     zeeman_expanded = cp.exp(
#         -(
#             zeeman_energies[:, :, :, cp.newaxis]
#             - zeeman_energies[:, :, 0, cp.newaxis]
#         )
#         / (KB * temperatures_reshaped)
#     )

#     # Compute magnetisation
#     mth_array = cp.dot(zeeman_matrix, zeeman_expanded) / cp.sum(
#         zeeman_expanded, axis=2
#     )

#     return mth_array


# def _mth(
#     filename,
#     group,
#     fields,
#     grid,
#     temperatures,
#     states_cutoff,
#     num_cpu,
#     num_threads,
#     rotation=None,
# ):
#     # Read data from HDF5 file
#     (
#         magnetic_momenta,
#         soc_energies,
#     ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
#         filename, group, states_cutoff, rotation
#     )

#     # Convert to CuPy arrays
#     magnetic_momenta_cp = cp.array(magnetic_momenta, dtype=cp.complex128)
#     soc_energies_cp = cp.array(soc_energies, dtype=cp.float64)
#     fields_cp = cp.array(fields, dtype=cp.float64)
#     temperatures_cp = cp.array(temperatures, dtype=cp.float64)
#     grid_cp = cp.array(grid, dtype=cp.float64)

#     # Call the magnetisation function
#     mth_array = _magnetisation(
#         magnetic_momenta_cp,
#         soc_energies_cp,
#         fields_cp,
#         temperatures_cp,
#         grid_cp,
#     )

#     return mth_array
