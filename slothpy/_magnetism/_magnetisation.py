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
from slothpy._general_utilities._system import (
    _get_num_of_processes,
    _distribute_chunks,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._magnetism._zeeman import _calculate_zeeman_matrix
from slothpy._general_utilities._grids_over_sphere import (
    _fibonacci_over_sphere,
    _meshgrid_over_sphere_flatten,
)


@jit(
    "float64(float64[:], float64[:], float64)",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _calculate_magnetization(
    energies: ndarray, states_momenta: ndarray, temperature: float64
) -> float64:
    energies = ascontiguousarray(energies)
    states_momenta = ascontiguousarray(states_momenta)
    # Boltzman weights
    exp_diff = exp(-(energies - energies[0]) / (KB * temperature))

    # Partition function
    z = sum(exp_diff)

    # Weighted magnetic moments of microstates
    m = sum(states_momenta * exp_diff)
    # m = vdot(states_momenta, exp_diff)

    return m / z


@jit(
    "float64[:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
)
def _mt_over_fields_grid(
    magnetic_momenta, soc_energies, fields, grid, temperatures
):
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    mht_array = zeros((fields_shape_0, temperatures_shape_0), dtype=float64)

    for i in range(fields_shape_0):
        mt_array = zeros(temperatures_shape_0, dtype=float64)

        for j in range(grid_shape_0):
            orientation = grid[j, :3]
            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[i], orientation
            )

            magnetic_momenta_cont = ascontiguousarray(magnetic_momenta)
            eigenvalues, eigenvectors = eigh(zeeman_matrix)

            states_momenta = (
                eigenvectors.conj().T
                @ (
                    grid[j, 0] * magnetic_momenta_cont[0]
                    + grid[j, 1] * magnetic_momenta_cont[1]
                    + grid[j, 2] * magnetic_momenta_cont[2]
                )
                @ eigenvectors
            )

            states_momenta_diag = diag(states_momenta).real.astype(float64)

            for t in range(temperatures_shape_0):
                mt_array[t] += (
                    _calculate_magnetization(
                        eigenvalues, states_momenta_diag, temperatures[t]
                    )
                    * grid[j, 3]
                )

        mht_array[i, :] = mt_array[:]

    return mht_array


@jit(
    "float64[:,:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _mt_over_fields_tensor(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    temperatures: ndarray,
):
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]

    mht_tensor_array = ascontiguousarray(
        zeros((fields_shape_0, temperatures_shape_0, 3, 3), dtype=float64)
    )

    for f in range(fields_shape_0):
        # Initialize arrays
        mt_tensor_array = ascontiguousarray(
            zeros((temperatures_shape_0, 3, 3), dtype=float64)
        )

        # Perform calculations for each tensor component
        for i in range(3):
            for j in range(3):
                # Construct Zeeman matrix
                zeeman_matrix = -fields[f] * MU_B * magnetic_momenta[j]
                for k in range(zeeman_matrix.shape[0]):
                    zeeman_matrix[k, k] += soc_energies[k]

                # Diagonalize full Hamiltonian matrix
                eigenvalues, eigenvectors = eigh(zeeman_matrix)
                magnetic_momenta = ascontiguousarray(magnetic_momenta)

                # Transform momentum according to the new eigenvectors
                states_momenta = (
                    eigenvectors.conj().T @ magnetic_momenta[i] @ eigenvectors
                )

                # Get diagonal momenta of the new states
                states_momenta = diag(states_momenta).real.astype(float64)

                # Compute partition function and magnetization for each T
                for t in range(temperatures.shape[0]):
                    mt_tensor_array[t, i, j] = _calculate_magnetization(
                        eigenvalues, states_momenta, temperatures[t]
                    )
        mht_tensor_array[f, :, :, :] = mt_tensor_array[:, :, :]

    return mht_tensor_array


@jit(
    "float64[:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _mt_over_grid_fields(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
):
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    mght_array = zeros(
        (grid_shape_0, fields_shape_0, temperatures_shape_0), dtype=float64
    )

    for g in range(grid_shape_0):
        mht_array = zeros(
            (fields_shape_0, temperatures_shape_0), dtype=float64
        )
        grid_momenta = (
            grid[g, 0] * magnetic_momenta[0]
            + grid[g, 1] * magnetic_momenta[1]
            + grid[g, 2] * magnetic_momenta[2]
        )

        for f in range(fields_shape_0):
            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[f], grid[g]
            )
            eigenvalues, eigenvectors = eigh(zeeman_matrix)

            states_momenta = (
                eigenvectors.conj().T @ grid_momenta @ eigenvectors
            )

            states_momenta_diag = diag(states_momenta).real.astype(float64)

            for t in range(temperatures_shape_0):
                mht_array[f, t] = _calculate_magnetization(
                    eigenvalues, states_momenta_diag, temperatures[t]
                )

        mght_array[g, :, :] = mht_array[:, :]

    return mght_array


def _calculate_mht(
    magnetic_momenta_name: str,
    soc_energies_name: str,
    fields_name: str,
    grid_name: str,
    temperatures_name: str,
    magnetic_momenta_shape: tuple,
    soc_energies_shape: tuple,
    grid_shape: tuple,
    temperatures_shape: tuple,
    field_chunk: tuple,
) -> ndarray:
    magnetic_momenta_shared = SharedMemory(magnetic_momenta_name)
    magnetic_momenta_array = ndarray(
        magnetic_momenta_shape,
        complex128,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float64, soc_energies_shared.buf
    )
    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray(grid_shape, float64, grid_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float64,
        temperatures_shared.buf,
    )

    offset = dtype(float64).itemsize * field_chunk[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray((chunk_length,), float64, fields_shared.buf, offset)

    # Switch to tensor calculation
    if array_equal(grid_array, array([1.0])):
        return _mt_over_fields_tensor(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            temperatures_array,
        )

    return _mt_over_fields_grid(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        temperatures_array,
    )


def _calculate_mht_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mht(*args)

    return mt


def _arg_iter_mht(
    magnetic_momenta_name,
    soc_energies_name,
    fields_name,
    grid_name,
    temperatures_name,
    magnetic_momenta_shape,
    soc_energies_shape,
    grid_shape,
    temperatures_shape,
    fields_chunks,
):
    # Iterator generator for arguments with different field values to be
    # distributed along num_process processes
    for field_chunk in fields_chunks:
        yield (
            magnetic_momenta_name,
            soc_energies_name,
            fields_name,
            grid_name,
            temperatures_name,
            magnetic_momenta_shape,
            soc_energies_shape,
            grid_shape,
            temperatures_shape,
            field_chunk,
        )


def _mth(
    filename: str,
    group: str,
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    rotation: ndarray[float64] = None,
) -> ndarray:
    # Flag for tensor calculation
    tensor_calc = False
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    # Get number of parallel proceses to be used
    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0]
    )

    #  Allocate arrays as contiguous
    fields = ascontiguousarray(fields, dtype=float64)
    temperatures = ascontiguousarray(temperatures, dtype=float64)
    grid = ascontiguousarray(grid, dtype=float64)

    if array_equal(grid, array([1.0])):
        tensor_calc = True

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)

        # Copy data to shared memory
        magnetic_momenta_shared_arr = ndarray(
            magnetic_momenta.shape,
            dtype=magnetic_momenta.dtype,
            buffer=magnetic_momenta_shared.buf,
        )
        soc_energies_shared_arr = ndarray(
            soc_energies.shape,
            dtype=soc_energies.dtype,
            buffer=soc_energies_shared.buf,
        )
        fields_shared_arr = ndarray(
            fields.shape, dtype=fields.dtype, buffer=fields_shared.buf
        )
        grid_shared_arr = ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )
        temperatures_shared_arr = ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]

        del magnetic_momenta
        del soc_energies
        del fields
        del grid
        del temperatures

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                set_num_threads(num_threads)
                with Pool(num_process) as p:
                    mht = p.map(
                        _calculate_mht_wrapper,
                        _arg_iter_mht(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            grid_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            _distribute_chunks(
                                fields_shared_arr.shape[0], num_process
                            ),
                        ),
                    )

    # Hidden option for susceptibility tensor calculation.
    if tensor_calc:
        return concatenate(mht)

    # Collecting results in plotting-friendly convention for M(H)
    mth_array = concatenate(mht).T

    return mth_array  # Returning values in Bohr magnetons


def _arg_iter_mght(
    magnetic_momenta_name,
    soc_energies_name,
    fields_name,
    grid_name,
    temperatures_name,
    magnetic_momenta_shape,
    soc_energies_shape,
    fields_shape,
    temperatures_shape,
    grids_chunks,
):
    # Iterator generator for arguments with different grid values to be
    # distributed along num_process processes
    for grid_chunk in grids_chunks:
        yield (
            magnetic_momenta_name,
            soc_energies_name,
            fields_name,
            grid_name,
            temperatures_name,
            magnetic_momenta_shape,
            soc_energies_shape,
            fields_shape,
            temperatures_shape,
            grid_chunk,
        )


def _calculate_mght(
    magnetic_momenta_name: str,
    soc_energies_name: str,
    fields_name: str,
    grid_name: str,
    temperatures_name: str,
    magnetic_momenta_shape: tuple,
    soc_energies_shape: tuple,
    fields_shape: tuple,
    temperatures_shape: tuple,
    grid_chunk: tuple,
) -> ndarray:
    magnetic_momenta_shared = SharedMemory(magnetic_momenta_name)
    magnetic_momenta_array = ndarray(
        magnetic_momenta_shape,
        complex128,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float64, soc_energies_shared.buf
    )
    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray(fields_shape, float64, fields_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float64,
        temperatures_shared.buf,
    )

    offset = dtype(float64).itemsize * grid_chunk[0] * 3
    chunk_length = grid_chunk[1] - grid_chunk[0]

    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray((chunk_length, 3), float64, grid_shared.buf, offset)

    return _mt_over_grid_fields(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        temperatures_array,
    )


def _calculate_mght_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mght(*args)

    return mt


def _mag_3d(
    filename: str,
    group: str,
    fields: ndarray,
    grid_type: Literal["mesh", "fibonacci"],
    grid_number: int,
    temperatures: ndarray,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    rotation: ndarray = None,
    sus_3d_num: bool = False,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    if grid_type != "mesh" and grid_type != "fibonacci":
        raise ValueError(
            'The only allowed grid types are "mesh" or "fibonacci".'
        ) from None
    if grid_type == "mesh":
        grid = _meshgrid_over_sphere_flatten(grid_number)
        num_points = 2 * grid_number**2
    elif grid_type == "fibonacci":
        grid = _fibonacci_over_sphere(grid_number)
        num_points = grid_number
    else:
        raise (
            ValueError('Grid type can only be set to "mesh" or "fibonacci".')
        )
    # Get number of parallel proceses to be used
    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, num_points
    )

    #  Allocate arrays as contiguous
    fields = ascontiguousarray(fields, dtype=float64)
    temperatures = ascontiguousarray(temperatures, dtype=float64)
    grid = ascontiguousarray(grid, dtype=float64)

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)

        # Copy data to shared memory
        magnetic_momenta_shared_arr = ndarray(
            magnetic_momenta.shape,
            dtype=magnetic_momenta.dtype,
            buffer=magnetic_momenta_shared.buf,
        )
        soc_energies_shared_arr = ndarray(
            soc_energies.shape,
            dtype=soc_energies.dtype,
            buffer=soc_energies_shared.buf,
        )
        fields_shared_arr = ndarray(
            fields.shape, dtype=fields.dtype, buffer=fields_shared.buf
        )
        grid_shared_arr = ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )
        temperatures_shared_arr = ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]

        del magnetic_momenta
        del soc_energies
        del fields
        del grid
        del temperatures

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                set_num_threads(num_threads)
                with Pool(num_process) as p:
                    mght = p.map(
                        _calculate_mght_wrapper,
                        _arg_iter_mght(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            fields_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            _distribute_chunks(
                                grid_shared_arr.shape[0], num_process
                            ),
                        ),
                    )

        mght = concatenate(mght)
        grid = grid_shared_arr[:].copy()
        fields_shape = fields_shared_arr.shape[0]
        temperatures_shape = temperatures_shared_arr.shape[0]

    if grid_type == "mesh":
        mag_3d = mght.reshape(
            (
                grid_number,
                grid_number * 2,
                fields_shape,
                temperatures_shape,
            )
        )
        mag_3d = mag_3d.transpose((2, 3, 0, 1))
        grid = grid.reshape(grid_number, grid_number * 2, 3)
        grid = grid.transpose((2, 0, 1))
        if sus_3d_num:
            return mag_3d, grid
        mag_3d_array = grid[:, newaxis, newaxis, :, :] * mag_3d
    elif grid_type == "fibonacci":
        if sus_3d_num:
            return mght, grid
        mag_3d_array = grid[:, :, newaxis, newaxis] * mght[:, newaxis, :, :]
        mag_3d_array = mag_3d_array.transpose((2, 3, 0, 1))

    return mag_3d_array
