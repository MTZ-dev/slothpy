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
from multiprocessing import Process, Pool
from threadpoolctl import threadpool_limits
from numpy import (
    ndarray,
    dtype,
    array,
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
    int64,
)
from numpy.linalg import eigvalsh
from numba import jit, set_num_threads, prange
from slothpy._general_utilities._constants import KB, MU_B, H_CM_1
from slothpy._general_utilities._system import (
    _get_num_of_processes,
    _to_shared_memory,
    _from_shared_memory,
    _chunk_from_shared_memory,
    _distribute_chunks,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._grids_over_sphere import (
    _fibonacci_over_sphere,
    _meshgrid_over_sphere_flatten,
)
from slothpy._general_utilities._math_expresions import _3d_dot
from slothpy.core._config import settings
from slothpy._gui._monitor_gui import _run_monitor_gui


@jit(
    f"{settings.numba_complex}[:,:]({settings.numba_complex}[:,:,:], {settings.numba_float}[:], {settings.numba_float}, {settings.numba_float}[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _calculate_zeeman_matrix(
    magnetic_momenta, soc_energies, field, orientation
):
    orientation = (-field * MU_B * orientation)
    magnetic_momenta = _3d_dot(magnetic_momenta, orientation)
    soc_energies = soc_energies.astype(magnetic_momenta.dtype)

    for k in prange(magnetic_momenta.shape[0]):
        magnetic_momenta[k, k] += soc_energies[k]

    return magnetic_momenta


@jit([
        f"({settings.numba_float}[:], {settings.numba_complex}[:,:,:], {settings.numba_float}[:], {settings.numba_float}[:,:], int64, int64[:], {settings.numba_float}[:,:])",
        f"({settings.numba_float}[:], {settings.numba_complex}[:,:,:], {settings.numba_float}[:], {settings.numba_float}[:,:], int64, int64[:], {settings.numba_float}[:,:,:])"
    ],
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
    inline="always",
)
def _zeeman_over_fields_orientations(
    soc_energies: ndarray,
    magnetic_momenta: ndarray,
    magnetic_fields: ndarray,
    orientations: ndarray,
    num_of_states: int,
    progress: ndarray,
    zeeman_array: ndarray,
):
    magnetic_fields_shape_0 = magnetic_fields.shape[0]
    grid_shape_0 = orientations.shape[0]
    zeeman_dim = zeeman_array.ndim

    for i in range(magnetic_fields_shape_0):
        for j in range(grid_shape_0):
            orientation = orientations[j, :3]
            zeeman_matrix = _calculate_zeeman_matrix(magnetic_momenta, soc_energies, magnetic_fields[i], orientation)
            energies = eigvalsh(zeeman_matrix).astype(soc_energies.dtype)
            energies = energies[:num_of_states] * H_CM_1
            if zeeman_dim == 2:
                zeeman_array[i, :] += energies * orientations[j, 3]
            else:
                zeeman_array[i, j, :] = energies
            progress += 1


def _zeeman_splitting_process(
    soc_energies_info: tuple,
    magnetic_momenta_info: tuple,
    magnetic_fields_info: tuple,
    orientations_info: tuple,
    progress_array_info: tuple,
    zeeman_array_info: tuple,
    magnetic_fields_chunk: tuple,
    process_index: int,
    number_of_states: int,
    average: bool,
    number_threads: int,
):  
    sm_soc_energies = SharedMemory(soc_energies_info[0])
    soc_energies = _from_shared_memory(sm_soc_energies, soc_energies_info)
    sm_magnetic_momenta = SharedMemory(magnetic_momenta_info[0])
    magnetic_momenta = _from_shared_memory(sm_magnetic_momenta, magnetic_momenta_info)
    sm_orientations = SharedMemory(orientations_info[0])
    orientations = _from_shared_memory(sm_orientations, orientations_info)
    sm_magnetic_fields = SharedMemory(magnetic_fields_info[0])
    magnetic_fields = _chunk_from_shared_memory(sm_magnetic_fields, magnetic_fields_info, magnetic_fields_chunk)
    sm_progress = SharedMemory(progress_array_info[0])
    progress = _chunk_from_shared_memory(sm_progress, progress_array_info, (process_index, process_index+1))

    sm_zeeman= SharedMemory(zeeman_array_info[0])

    if average:
        offset = magnetic_fields_chunk[0] * number_of_states * dtype(zeeman_array_info[2]).itemsize
        zeeman_array = ndarray((magnetic_fields_chunk[1] - magnetic_fields_chunk[0], number_of_states), zeeman_array_info[2], sm_zeeman.buf, offset)
    else:
        offset = magnetic_fields_chunk[0] * orientations_info[1][0] * number_of_states * dtype(zeeman_array_info[2]).itemsize
        zeeman_array = ndarray((magnetic_fields_chunk[1] - magnetic_fields_chunk[0], orientations_info[1][0], number_of_states), zeeman_array_info[2], sm_zeeman.buf, offset)

    with threadpool_limits(limits=number_threads):
        set_num_threads(number_threads)
        _zeeman_over_fields_orientations(soc_energies, magnetic_momenta, magnetic_fields, orientations, number_of_states, progress, zeeman_array)


def _zeeman_splitting_process_wrapper(args):
    return _zeeman_splitting_process(*args)


def _zeeman_splitting(
    filename: str,
    group: str,
    number_of_states: int,
    magnetic_fields: ndarray,
    orientations: ndarray,
    states_cutoff: int,
    number_cpu: int,
    number_threads: int,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    # Get number of parallel proceses to be used
    number_processes, number_threads = _get_num_of_processes(number_cpu, number_threads, magnetic_fields.shape[0])

    if orientations.shape[1] == 4:
        average = True
        zeeman_array = zeros((magnetic_fields.shape[0], number_of_states), dtype=magnetic_fields.dtype, order="C")
    elif orientations.shape[1] == 3:
        average = False
        zeeman_array = zeros((magnetic_fields.shape[0], orientations.shape[0], number_of_states), dtype=magnetic_fields.dtype, order="C")
    else:
        raise ValueError("Wrong orientations' array dimensions.")

    soc_energies = ascontiguousarray(soc_energies)
    magnetic_momenta = ascontiguousarray(magnetic_momenta)
    magnetic_fields = ascontiguousarray(magnetic_fields)
    orientations = ascontiguousarray(orientations)
    zeeman_array = ascontiguousarray(zeeman_array)
    progress_array = zeros((number_processes,), dtype=int64)

    magnetic_fields_chunks = _distribute_chunks(magnetic_fields.shape[0], number_processes)

    with SharedMemoryManager() as smm:
        soc_energies_info = _to_shared_memory(smm, soc_energies)
        magnetic_momenta_info = _to_shared_memory(smm, magnetic_momenta)
        magnetic_fields_info = _to_shared_memory(smm, magnetic_fields)
        orientations_info = _to_shared_memory(smm, orientations)
        zeeman_array_info = _to_shared_memory(smm, zeeman_array)
        progress_array_info = _to_shared_memory(smm, progress_array)

        if settings.monitor:
            monitor = Process(target=_run_monitor_gui, args=(progress_array_info, magnetic_fields_chunks, orientations_info[1][0], "zeeman_splitting"))
            monitor.start()

        with Pool(number_processes) as p:
            p.map(_zeeman_splitting_process_wrapper,[(soc_energies_info, magnetic_momenta_info, magnetic_fields_info, orientations_info, progress_array_info, zeeman_array_info, chunk, process_index, number_of_states, average, number_threads) for process_index, chunk in enumerate(magnetic_fields_chunks)])

        sm_zeeman_array = SharedMemory(zeeman_array_info[0])
        zeeman_array = array(_from_shared_memory(sm_zeeman_array, zeeman_array_info), copy=True, order="C")
        if not average:
            zeeman_array = zeeman_array.transpose((1,0,2))

        if settings.monitor:
            monitor.join()
            monitor.close()

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
                magnetic_momenta,
                soc_energies,
                field,
                orientation.astype(complex128),
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
            orientation = grid[j, :3].astype(complex128)

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
            orientation = grid[j, :3].astype(complex128)

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
                magnetic_momenta,
                soc_energies,
                fields[f],
                grid[g].astype(complex128),
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
                magnetic_momenta,
                soc_energies,
                fields[f],
                grid[g].astype(complex128),
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
