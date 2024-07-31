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
    arange,
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
# from scipy.linalg import eigh
from numba import jit, set_num_threads, prange, types, int64, float32, float64, complex64, complex128
from slothpy._general_utilities._constants import KB, H_CM_1
from slothpy._general_utilities._system import (
    SharedMemoryArrayInfo,
    _load_shared_memory_arrays,
    _get_num_of_processes,
    _distribute_chunks,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._grids_over_sphere import (
    _fibonacci_over_sphere,
    _meshgrid_over_sphere,
)
from slothpy._general_utilities._math_expresions import _3d_dot
from slothpy.core._hamiltonian_object import Hamiltonian

@jit([
    types.Array(complex64, 2, 'C')(
        types.Array(complex64, 3, 'C', True), 
        types.Array(float32, 1, 'C', True), 
        float32, 
        types.Array(float32, 1, 'C', True)
    ),
    types.Array(complex128, 2, 'C')(
        types.Array(complex128, 3, 'C', True), 
        types.Array(float64, 1, 'C', True), 
        float64, 
        types.Array(float64, 1, 'C', True)
    )
],
nopython=True,
nogil=True,
cache=True,
fastmath=True,
parallel=True,
)
def _calculate_zeeman_matrix(
    magnetic_momenta, states_energies, field, orientation
):
    magnetic_momenta = _3d_dot(magnetic_momenta, -field * orientation)

    for k in prange(magnetic_momenta.shape[0]):
        magnetic_momenta[k, k] += states_energies[k]

    return magnetic_momenta


def _zeeman_splitting(hamiltonian: Hamiltonian, magnetic_fields: ndarray, orientations: ndarray, progress_array: ndarray, zeeman_array: ndarray, number_of_states: int, electric_field_vector: ndarray, process_index: int, start: int, end: int):  
    magnetic_fields_shape_0 = magnetic_fields.shape[0]
    h_cm_1 = array(H_CM_1, dtype=magnetic_fields.dtype)
    hamiltonian._electric_field = electric_field_vector
    
    for i in range(start, end):
        hamiltonian._magnetic_field = orientations[i//magnetic_fields_shape_0, :3] * magnetic_fields[i%magnetic_fields_shape_0]
        zeeman_array[i, :] = hamiltonian.zeeman_energies(number_of_states) * h_cm_1
        progress_array[process_index] += 1


def _zeeman_splitting_average(hamiltonian: Hamiltonian, magnetic_fields: ndarray, orientations: ndarray, progress_array: ndarray, number_of_states: int, electric_field_vector: ndarray, process_index: int, start: int, end: int):
    orientations_shape_0 = orientations.shape[0]
    start_field_index = start // orientations_shape_0
    end_field_index = (end - 1) // orientations_shape_0 + 1
    previous_field_index = start_field_index
    zeeman_array = zeros((end_field_index - start_field_index, number_of_states), dtype=magnetic_fields.dtype)
    zeeman_index = 0
    h_cm_1 = array(H_CM_1, dtype=magnetic_fields.dtype)
    hamiltonian._electric_field = electric_field_vector

    for i in range(start, end):
        current_field_index = i // orientations_shape_0
        orientation_index = i % orientations_shape_0
        if current_field_index != previous_field_index:
            zeeman_index += 1
            previous_field_index = current_field_index
        hamiltonian._magnetic_field = orientations[orientation_index, :3] * magnetic_fields[current_field_index]
        zeeman_array[zeeman_index, :] += hamiltonian.zeeman_energies(number_of_states) * h_cm_1 * orientations[orientation_index, 3]
        progress_array[process_index] += 1
 
    return start_field_index, end_field_index, zeeman_array


def _zeeman_splitting_proxy(slt_hamiltonian_info, sm_arrays_info_list: list[SharedMemoryArrayInfo], args_list, process_index, start: int, end: int, returns: bool = False):
    hamiltonian = Hamiltonian(sm_arrays_info_list, slt_hamiltonian_info[0], slt_hamiltonian_info[1])
    sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list[hamiltonian._shared_memory_index:])
    if returns:
        return _zeeman_splitting_average(hamiltonian, *arrays, *args_list, process_index, start, end)
    else:
        _zeeman_splitting(hamiltonian, *arrays, *args_list, process_index, start, end)


def _get_zeeman_matrix(
    filename: str,
    group: str,
    states_cutoff: int,
    fields: ndarray,
    orientations: ndarray,
    rotation: ndarray = None,
) -> ndarray:
    (
        magnetic_momenta,
        states_energies,
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
                states_energies,
                field,
                orientation,
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
    energies: ndarray, temperature
):
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
    energies: ndarray, temperature
):
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
    states_energies: ndarray,
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

            zeeman_matrix = magnetic_momenta[0]#_calculate_zeeman_matrix(
            #     magnetic_momenta, states_energies, fields[i], orientation ###nie działało z compatibility
            # )

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
    states_energies: ndarray,
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

            zeeman_matrix = magnetic_momenta[0] # _calculate_zeeman_matrix( nie działało
            #     magnetic_momenta, states_energies, fields[i], orientation
            # )

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
    states_energies: ndarray,
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

            zeeman_matrix = magnetic_momenta[0] #_calculate_zeeman_matrix( nie działało
            #     magnetic_momenta,
            #     states_energies,
            #     fields[f],
            #     grid[g],
            # )

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
    states_energies: ndarray,
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

            zeeman_matrix = magnetic_momenta[0] #_calculate_zeeman_matrix( nie działało
            #     magnetic_momenta,
            #     states_energies,
            #     fields[f],
            #     grid[g],
            # )

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
    states_energies_name: str,
    fields_name: str,
    grid_name: str,
    temperatures_name: str,
    magnetic_momenta_shape: tuple,
    states_energies_shape: tuple,
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
    states_energies_shared = SharedMemory(states_energies_name)
    states_energies_array = ndarray(
        states_energies_shape, float64, states_energies_shared.buf
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
            states_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )
    elif energy_type == "internal":
        return _internal_energyt_over_fields_grid(
            magnetic_momenta_array,
            states_energies_array,
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
    states_energies_name,
    fields_name,
    grid_name,
    temperatures_name,
    magnetic_momenta_shape,
    states_energies_shape,
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
            states_energies_name,
            fields_name,
            grid_name,
            temperatures_name,
            magnetic_momenta_shape,
            states_energies_shape,
            grid_shape,
            temperatures_shape,
            field_chunk,
            energy_type,
        )


def _eth(
    filename: str,
    group: str,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
    energy_type: Literal["helmholtz", "internal"],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        states_energies,
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
        states_energies_shared = smm.SharedMemory(size=states_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)

        # Copy data to shared memory
        magnetic_momenta_shared_arr = ndarray(
            magnetic_momenta.shape,
            dtype=magnetic_momenta.dtype,
            buffer=magnetic_momenta_shared.buf,
        )
        states_energies_shared_arr = ndarray(
            states_energies.shape,
            dtype=states_energies.dtype,
            buffer=states_energies_shared.buf,
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
        states_energies_shared_arr[:] = states_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]

        del magnetic_momenta
        del states_energies
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
                            states_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            states_energies_shared_arr.shape,
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
    states_energies_name,
    fields_name,
    grid_name,
    temperatures_name,
    magnetic_momenta_shape,
    states_energies_shape,
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
            states_energies_name,
            fields_name,
            grid_name,
            temperatures_name,
            magnetic_momenta_shape,
            states_energies_shape,
            fields_shape,
            temperatures_shape,
            grid_chunk,
            energy_type,
        )


def _calculate_eght(
    magnetic_momenta_name: str,
    states_energies_name: str,
    fields_name: str,
    grid_name: str,
    temperatures_name: str,
    magnetic_momenta_shape: tuple,
    states_energies_shape: tuple,
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
    states_energies_shared = SharedMemory(states_energies_name)
    states_energies_array = ndarray(
        states_energies_shape, float64, states_energies_shared.buf
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
            states_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )

    elif energy_type == "internal":
        return _internal_energyt_over_grid_fields(
            magnetic_momenta_array,
            states_energies_array,
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
        states_energies,
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
        states_energies_shared = smm.SharedMemory(size=states_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)

        # Copy data to shared memory
        magnetic_momenta_shared_arr = ndarray(
            magnetic_momenta.shape,
            dtype=magnetic_momenta.dtype,
            buffer=magnetic_momenta_shared.buf,
        )
        states_energies_shared_arr = ndarray(
            states_energies.shape,
            dtype=states_energies.dtype,
            buffer=states_energies_shared.buf,
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
        states_energies_shared_arr[:] = states_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]

        del magnetic_momenta
        del states_energies
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
                            states_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            states_energies_shared_arr.shape,
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
