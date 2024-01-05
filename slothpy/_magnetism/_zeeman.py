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
    sum,
    vdot,
    zeros,
    ascontiguousarray,
    concatenate,
    newaxis,
    exp,
    log,
    float64,
    complex128,
)
from numpy.linalg import eigvalsh
from numba import jit, set_num_threads
from slothpy._general_utilities._constants import KB, MU_B, H_CM_1
from slothpy._general_utilities._system import (
    _get_num_of_processes,
    _distribute_chunks,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._grids_over_sphere import (
    _fibonacci_over_sphere,
    _meshgrid_over_sphere_flatten,
)


@jit(
    "complex128[:,:](complex128[:,:,:], float64[:], float64, float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _calculate_zeeman_matrix(
    magnetic_momenta, soc_energies, field, orientation
):
    orientation = -field * MU_B * orientation
    zeeman_matrix = ascontiguousarray(
        magnetic_momenta[0] * orientation[0]
        + magnetic_momenta[1] * orientation[1]
        + magnetic_momenta[2] * orientation[2]
    )

    # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
    for k in range(zeeman_matrix.shape[0]):
        zeeman_matrix[k, k] += soc_energies[k]

    return zeeman_matrix


@jit(
    "float64[:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " int64, boolean)",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _zeeman_over_fields_grid(
    magnetic_momenta,
    soc_energies,
    fields,
    grid,
    num_of_states,
    average: bool = False,
):
    fields_shape_0 = fields.shape[0]
    grid_shape_0 = grid.shape[0]
    magnetic_momenta = ascontiguousarray(magnetic_momenta)
    # Initialize arrays and scale energy to the ground SOC state
    soc_energies = ascontiguousarray(soc_energies - soc_energies[0])
    if average:
        zeeman_h_array = zeros(
            (fields_shape_0, 1, num_of_states), dtype=float64
        )
    else:
        zeeman_h_array = zeros(
            (fields_shape_0, grid.shape[0], num_of_states), dtype=float64
        )

    for i in range(fields_shape_0):
        zeeman_array = zeros(
            (zeeman_h_array.shape[1], zeeman_h_array.shape[2]), dtype=float64
        )

        # Perform calculations for each magnetic field orientation
        for j in range(grid_shape_0):
            orientation = grid[j, :3]

            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[i], orientation
            )

            # Diagonalize full Zeeman Hamiltonian
            energies = eigvalsh(zeeman_matrix)

            # Get only desired number of states in cm-1
            energies = energies[:num_of_states] * H_CM_1

            if average:
                zeeman_array[0, :] += energies * grid[j, 3]
            # Collect the results
            else:
                zeeman_array[j, :] = energies

        zeeman_h_array[i] = zeeman_array

    return zeeman_h_array


def _calculate_zeeman_splitting(
    magnetic_momenta_name: str,
    soc_energies_name: str,
    fields_name: str,
    grid_name: str,
    magnetic_momenta_shape: tuple,
    soc_energies_shape: tuple,
    grid_shape: tuple,
    field_chunk: tuple,
    num_of_states,
    average: bool = False,
) -> ndarray:
    # Option to enable calculations with only a single grid point.

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

    offset = dtype(float64).itemsize * field_chunk[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray((chunk_length,), float64, fields_shared.buf, offset)

    return _zeeman_over_fields_grid(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        num_of_states,
        average,
    )


def _caculate_zeeman_splitting_wrapper(args):
    zeeman_array = _calculate_zeeman_splitting(*args)

    return zeeman_array


def _arg_iter_zeeman(
    magnetic_momenta_name,
    soc_energies_name,
    fields_name,
    grid_name,
    magnetic_momenta_shape,
    soc_energies_shape,
    grid_shape,
    fields_chunks,
    num_of_states,
    average: bool = False,
):
    # Iterator generator for arguments with different field values to be
    # distributed along num_process processes
    for field_chunk in fields_chunks:
        yield (
            magnetic_momenta_name,
            soc_energies_name,
            fields_name,
            grid_name,
            magnetic_momenta_shape,
            soc_energies_shape,
            grid_shape,
            field_chunk,
            num_of_states,
            average,
        )


def _zeeman_splitting(
    filename: str,
    group: str,
    num_of_states: int,
    fields: ndarray,
    grid: ndarray,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    average: bool = False,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    # Get number of parallel proceses to be used
    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0]
    )

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = ascontiguousarray(fields)
    grid = ascontiguousarray(grid)
    fields_shape_0 = fields.shape[0]
    grid_shape_0 = grid.shape[0]

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)

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

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]

        del magnetic_momenta
        del soc_energies
        del fields
        del grid

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                set_num_threads(num_threads)
                with Pool(num_process) as p:
                    zeeman = p.map(
                        _caculate_zeeman_splitting_wrapper,
                        _arg_iter_zeeman(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            grid_shared_arr.shape,
                            _distribute_chunks(
                                fields_shared_arr.shape[0], num_process
                            ),
                            num_of_states,
                            average,
                        ),
                    )

    zeeman_array = concatenate(zeeman)
    zeeman_array = zeeman_array.transpose((1, 0, 2))

    if average:
        zeeman_array = zeeman_array.reshape((1, fields_shape_0, num_of_states))
    else:
        zeeman_array = zeeman_array.reshape(
            (grid_shape_0, fields_shape_0, num_of_states)
        )

    return zeeman_array


def _get_zeeman_matrix(
    filename: str,
    group: str,
    states_cutoff: int,
    fields: ndarray[float64],
    orientations: ndarray,
    rotation: ndarray = None,
) -> ndarray:
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    zeeman_matrix = zeros(
        (
            fields.shape[0],
            orientations.shape[0],
            magnetic_momenta.shape[1],
            magnetic_momenta.shape[2],
        ),
        dtype=complex128,
    )

    for f, field in enumerate(fields):
        for o, orientation in enumerate(orientations):
            zeeman_matrix[f, o, :, :] = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, field, orientation
            )

    return zeeman_matrix


@jit(
    "float64(float64[:], float64)",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _calculate_helmholtz_energy(
    energies: ndarray, temperature: float64
) -> float64:
    # Boltzman weights
    exp_diff = exp(-(energies - energies[0]) / (KB * temperature))

    # Partition function
    z = sum(exp_diff)

    return -KB * temperature * log(z) * H_CM_1


@jit(
    "float64(float64[:], float64)",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _calculate_internal_energy(
    energies: ndarray, temperature: float64
) -> float64:
    # Boltzman weights
    exp_diff = exp(-(energies - energies[0]) / (KB * temperature))

    # Partition function
    z = sum(exp_diff)

    e = vdot((energies * H_CM_1), exp_diff)
    # e = sum((energies * H_CM_1) * exp_diff)

    return e / z


@jit(
    "float64[:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _helmholtz_energyt_over_fields_grid(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
) -> ndarray:
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    eht_array = zeros((fields_shape_0, temperatures_shape_0), dtype=float64)

    for i in range(fields_shape_0):
        et_array = ascontiguousarray(
            zeros((temperatures.shape[0]), dtype=float64)
        )

        # Perform calculations for each magnetic field orientation
        for j in range(grid_shape_0):
            # Construct Zeeman matrix
            orientation = grid[j, :3]

            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[i], orientation
            )

            # Diagonalize full Hamiltonian matrix
            eigenvalues = eigvalsh(zeeman_matrix)
            eigenvalues = ascontiguousarray(eigenvalues)

            # Compute Helmholtz energy for each T
            for t in range(temperatures.shape[0]):
                et_array[t] += (
                    _calculate_helmholtz_energy(eigenvalues, temperatures[t])
                    * grid[j, 3]
                )

        eht_array[i, :] = et_array[:]

    return eht_array


@jit(
    "float64[:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _internal_energyt_over_fields_grid(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
) -> ndarray:
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    eht_array = zeros((fields_shape_0, temperatures_shape_0), dtype=float64)

    for i in range(fields_shape_0):
        et_array = ascontiguousarray(
            zeros((temperatures.shape[0]), dtype=float64)
        )

        # Perform calculations for each magnetic field orientation
        for j in range(grid_shape_0):
            # Construct Zeeman matrix
            orientation = grid[j, :3]

            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[i], orientation
            )

            # Diagonalize full Hamiltonian matrix
            eigenvalues = eigvalsh(zeeman_matrix)
            eigenvalues = ascontiguousarray(eigenvalues)

            # Compute Helmholtz energy for each T
            for t in range(temperatures.shape[0]):
                et_array[t] += (
                    _calculate_internal_energy(eigenvalues, temperatures[t])
                    * grid[j, 3]
                )

        eht_array[i, :] = et_array[:]

    return eht_array


@jit(
    "float64[:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _helmholtz_energyt_over_grid_fields(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
) -> ndarray:
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    eght_array = zeros(
        (grid_shape_0, fields_shape_0, temperatures_shape_0), dtype=float64
    )

    for g in range(grid_shape_0):
        eht_array = ascontiguousarray(
            zeros((fields_shape_0, temperatures.shape[0]), dtype=float64)
        )

        # Perform calculations for each magnetic field orientation
        for f in range(fields_shape_0):
            # Construct Zeeman matrix

            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[f], grid[g]
            )

            # Diagonalize full Hamiltonian matrix
            eigenvalues = eigvalsh(zeeman_matrix)
            eigenvalues = ascontiguousarray(eigenvalues)

            # Compute Helmholtz energy for each T
            for t in range(temperatures_shape_0):
                eht_array[f, t] += _calculate_helmholtz_energy(
                    eigenvalues, temperatures[t]
                )

        eght_array[g, :, :] = eht_array[:, :]

    return eght_array


@jit(
    "float64[:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _internal_energyt_over_grid_fields(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
) -> ndarray:
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    eght_array = zeros(
        (grid_shape_0, fields_shape_0, temperatures_shape_0), dtype=float64
    )

    for g in range(grid_shape_0):
        eht_array = ascontiguousarray(
            zeros((fields_shape_0, temperatures.shape[0]), dtype=float64)
        )

        # Perform calculations for each magnetic field orientation
        for f in range(fields_shape_0):
            # Construct Zeeman matrix

            zeeman_matrix = _calculate_zeeman_matrix(
                magnetic_momenta, soc_energies, fields[f], grid[g]
            )

            # Diagonalize full Hamiltonian matrix
            eigenvalues = eigvalsh(zeeman_matrix)
            eigenvalues = ascontiguousarray(eigenvalues)

            # Compute Helmholtz energy for each T
            for t in range(temperatures_shape_0):
                eht_array[f, t] += _calculate_internal_energy(
                    eigenvalues, temperatures[t]
                )

        eght_array[g, :, :] = eht_array[:, :]

    return eght_array


def _calculate_eht(
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
    energy_type: Literal["helmholtz", "internal"],
) -> ndarray:
    # Option to enable calculations with only a single grid point.
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

    if energy_type == "helmholtz":
        return _helmholtz_energyt_over_fields_grid(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )
    elif energy_type == "internal":
        return _internal_energyt_over_fields_grid(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )


def _calculate_eht_wrapper(args):
    # Unpack arguments and call the function
    et = _calculate_eht(*args)

    return et


def _arg_iter_eht(
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
    energy_type,
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
            energy_type,
        )


def _eth(
    filename: str,
    group: str,
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
    energy_type: Literal["helmholtz", "internal"],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    # Get number of parallel proceses to be used
    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0]
    )

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = ascontiguousarray(fields)
    temperatures = ascontiguousarray(temperatures)
    grid = ascontiguousarray(grid)

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
                    eht = p.map(
                        _calculate_eht_wrapper,
                        _arg_iter_eht(
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
                            energy_type,
                        ),
                    )

    # Collecting results in plotting-friendly convention as for the M(H)
    eth_array = concatenate(eht).T

    return eth_array  # Returning values in cm-1


def _arg_iter_eght(
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
    energy_type,
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
            energy_type,
        )


def _calculate_eght(
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
    energy_type: Literal["helmholtz", "internal"],
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

    if energy_type == "helmholtz":
        return _helmholtz_energyt_over_grid_fields(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )

    elif energy_type == "internal":
        return _internal_energyt_over_grid_fields(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )


def _calculate_eght_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_eght(*args)

    return mt


def _energy_3d(
    filename: str,
    group: str,
    fields: ndarray,
    grid_type: Literal["mesh", "fibonacci"],
    grid_number: int,
    temperatures: ndarray,
    energy_type: Literal["helmholtz", "internal"],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    rotation: ndarray = None,
) -> ndarray:
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

    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

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
                # Parallel M(T,H) calculation over different grid points
                with Pool(num_process) as p:
                    eght = p.map(
                        _calculate_eght_wrapper,
                        _arg_iter_eght(
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
                            energy_type,
                        ),
                    )

        eght = concatenate(eght)
        grid = grid_shared_arr[:].copy()
        fields_shape = fields_shared_arr.shape[0]
        temperatures_shape = temperatures_shared_arr.shape[0]

    if grid_type == "mesh":
        energy_3d = eght.reshape(
            (
                grid_number,
                grid_number * 2,
                fields_shape,
                temperatures_shape,
            )
        )
        energy_3d = energy_3d.transpose((2, 3, 0, 1))
        grid = grid.reshape(grid_number, grid_number * 2, 3)
        grid = grid.transpose((2, 0, 1))
        energy_3d_array = grid[:, newaxis, newaxis, :, :] * energy_3d
    elif grid_type == "fibonacci":
        energy_3d_array = grid[:, :, newaxis, newaxis] * eght[:, newaxis, :, :]
        energy_3d_array = energy_3d_array.transpose((2, 3, 0, 1))

    return energy_3d_array
