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

from os import cpu_count
from time import perf_counter_ns
from typing import Tuple, Literal
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from numpy import (
    ndarray,
    dtype,
    ones,
    ascontiguousarray,
    float32,
    complex64,
)
from threadpoolctl import threadpool_limits
from numba import set_num_threads
from slothpy._magnetism._magnetisation import (
    _mt_over_fields_grid,
    _mt_over_grid_fields,
    _arg_iter_mht,
    _arg_iter_mght,
)
from slothpy._magnetism._zeeman import (
    _zeeman_over_fields_grid,
    _helmholtz_energyt_over_fields_grid,
    _internal_energyt_over_fields_grid,
    _helmholtz_energyt_over_grid_fields,
    _internal_energyt_over_grid_fields,
    _arg_iter_eht,
    _arg_iter_zeeman,
    _arg_iter_eght,
)
from slothpy._general_utilities._system import _get_num_of_processes
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._system import _distribute_chunks
from slothpy._general_utilities._constants import (
    YELLOW,
    BLUE,
    PURPLE,
    GREEN,
    RED,
    RESET,
)


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
        complex64,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float32, soc_energies_shared.buf
    )
    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray(grid_shape, float32, grid_shared.buf)

    offset = dtype(float32).itemsize * field_chunk[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray((chunk_length,), float32, fields_shared.buf, offset)

    setup_time_end = perf_counter_ns()

    # autotune_size = 4
    grid_array = ones((4, grid_array.shape[1]), dtype=float32)
    fields_array = ones((4,), dtype=float32)

    exec_time_start = perf_counter_ns()

    _zeeman_over_fields_grid(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        num_of_states,
        average,
    )

    exec_time_end = perf_counter_ns()

    return [setup_time_end, exec_time_end - exec_time_start]


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
        complex64,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float32, soc_energies_shared.buf
    )
    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray(grid_shape, float32, grid_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float32,
        temperatures_shared.buf,
    )

    offset = dtype(float32).itemsize * field_chunk[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray((chunk_length,), float32, fields_shared.buf, offset)

    setup_time_end = perf_counter_ns()

    # autotune_size = 4
    grid_array = ones((4, grid_array.shape[1]), dtype=float32)
    fields_array = ones((4,), dtype=float32)

    if energy_type == "helmholtz":
        exec_time_start = perf_counter_ns()
        _helmholtz_energyt_over_fields_grid(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )
        exec_time_end = perf_counter_ns()
    elif energy_type == "internal":
        exec_time_start = perf_counter_ns()
        _internal_energyt_over_fields_grid(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )
        exec_time_end = perf_counter_ns()

    return [setup_time_end, exec_time_end - exec_time_start]


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
        complex64,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float32, soc_energies_shared.buf
    )
    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray(grid_shape, float32, grid_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float32,
        temperatures_shared.buf,
    )

    offset = dtype(float32).itemsize * field_chunk[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray((chunk_length,), float32, fields_shared.buf, offset)

    setup_time_end = perf_counter_ns()

    # autotune_size = 4
    grid_array = ones((4, grid_array.shape[1]), dtype=float32)
    fields_array = ones((4,), dtype=float32)

    exec_time_start = perf_counter_ns()

    _mt_over_fields_grid(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        temperatures_array,
    )

    exec_time_end = perf_counter_ns()

    return [setup_time_end, exec_time_end - exec_time_start]


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
        complex64,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float32, soc_energies_shared.buf
    )
    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray(fields_shape, float32, fields_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float32,
        temperatures_shared.buf,
    )

    offset = dtype(float32).itemsize * grid_chunk[0] * 3
    chunk_length = grid_chunk[1] - grid_chunk[0]

    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray((chunk_length, 3), float32, grid_shared.buf, offset)

    setup_time_end = perf_counter_ns()

    # autotune_size = 4
    grid_array = ones((4, grid_array.shape[1]), dtype=float32)
    fields_array = ones((4,), dtype=float32)

    exec_time_start = perf_counter_ns()

    _mt_over_grid_fields(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        temperatures_array,
    )

    exec_time_end = perf_counter_ns()

    return [setup_time_end, exec_time_end - exec_time_start]


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
        complex64,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float32, soc_energies_shared.buf
    )
    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray(fields_shape, float32, fields_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float32,
        temperatures_shared.buf,
    )

    offset = dtype(float32).itemsize * grid_chunk[0] * 3
    chunk_length = grid_chunk[1] - grid_chunk[0]

    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray((chunk_length, 3), float32, grid_shared.buf, offset)

    setup_time_end = perf_counter_ns()

    # autotune_size = 4
    grid_array = ones((4, grid_array.shape[1]), dtype=float32)
    fields_array = ones((4,), dtype=float32)

    if energy_type == "helmholtz":
        exec_time_start = perf_counter_ns()
        _helmholtz_energyt_over_grid_fields(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )
        exec_time_end = perf_counter_ns()
    elif energy_type == "internal":
        exec_time_start = perf_counter_ns()
        _internal_energyt_over_grid_fields(
            magnetic_momenta_array,
            soc_energies_array,
            fields_array,
            grid_array,
            temperatures_array,
        )
        exec_time_end = perf_counter_ns()

    return [setup_time_end, exec_time_end - exec_time_start]


def _calculate_eght_wrapper(args):
    zeeman_array = _calculate_eght(*args)

    return zeeman_array


def _caculate_zeeman_splitting_wrapper(args):
    zeeman_array = _calculate_zeeman_splitting(*args)

    return zeeman_array


def _calculate_eht_wrapper(args):
    # Unpack arguments and call the function
    et = _calculate_eht(*args)

    return et


def _calculate_mht_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mht(*args)

    return mt


def _calculate_mght_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mght(*args)

    return mt


def _benchmark(
    filename: str,
    group: str,
    fields: ndarray[float32],
    grid: ndarray[float32],
    temperatures: ndarray[float32],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    benchamark_type: Literal[
        "zeeman", "energy", "magnetisation", "energy_3d", "magnetisation_3d"
    ] = "magnetisation",
    num_of_states: int = 0,
    average: bool = False,
    energy_type: Literal["helmholtz", "internal"] = "helmholtz",
) -> ndarray:
    setup_time_start = perf_counter_ns()
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    if benchamark_type in ["zeeman", "energy", "magnetisation"]:
        # Get number of parallel proceses to be used
        num_process, num_threads = _get_num_of_processes(
            num_cpu, num_threads, fields.shape[0]
        )
    elif benchamark_type in ["energy_3d", "magnetisation_3d"]:
        num_process, num_threads = _get_num_of_processes(
            num_cpu, num_threads, grid.shape[0]
        )

    #  Allocate arrays as contiguous
    fields = ascontiguousarray(fields, dtype=float32)
    temperatures = ascontiguousarray(temperatures, dtype=float32)
    grid = ascontiguousarray(grid, dtype=float32)

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
                if benchamark_type == "zeeman":
                    with Pool(num_process) as p:
                        timings = p.map(
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
                elif benchamark_type == "energy":
                    with Pool(num_process) as p:
                        timings = p.map(
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
                elif benchamark_type == "magnetisation":
                    with Pool(num_process) as p:
                        timings = p.map(
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
                elif benchamark_type == "energy_3d":
                    with Pool(num_process) as p:
                        timings = p.map(
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
                elif benchamark_type == "magnetisation_3d":
                    with Pool(num_process) as p:
                        timings = p.map(
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

    setup_times = [sublist[0] for sublist in timings]
    exec_times = [sublist[1] for sublist in timings]

    setup_time = max(setup_times) - setup_time_start
    exec_time = max(exec_times)

    return setup_time, exec_time


def _auto_tune(
    filename: str,
    group: str,
    fields: ndarray[float32],
    grid: ndarray[float32],
    temperatures: ndarray[float32],
    states_cutoff: int,
    num_cpu: int,
    num_to_parallelize: int,
    inner_loop_size: int,
    benchamark_type: Literal[
        "zeeman", "energy", "magnetisation", "energy_3d", "magnetisation_3d"
    ] = "magnetisation",
    num_of_states: int = 0,
    average: bool = False,
    energy_type: Literal["helmholtz", "internal"] = "helmholtz",
) -> Tuple[int, int]:
    if num_cpu == 0:
        num_cpu = int(cpu_count())
    final_num_of_processes = num_cpu
    final_num_of_threads = 1
    best_time = float("inf")

    old_processes = num_cpu
    new_processes = 1
    num_processes = 1
    num_threads = 1

    for threads in range(1, num_cpu + 2):
        new_processes = num_cpu // threads
        if new_processes >= num_to_parallelize:
            new_processes = num_to_parallelize
            threads = num_cpu // new_processes
        if (
            new_processes != old_processes
            and old_processes <= num_to_parallelize
        ):
            num_processes = old_processes
            num_threads = threads - 1

            setup_time, exec_time = _benchmark(
                filename,
                group,
                fields,
                grid,
                temperatures,
                states_cutoff,
                num_cpu,
                num_threads,
                benchamark_type,
                num_of_states,
                average,
                energy_type,
            )
            # autotune_size ** 2 = 16
            current_time = (
                setup_time
                + num_to_parallelize
                * inner_loop_size
                / (16 * num_processes)
                * exec_time
            )

            info = (
                "Processes:"
                f" {num_processes}, threads: {num_threads}."
                " Estimated maximal execution time: "
            )

            if current_time < best_time:
                best_time = current_time
                final_num_of_processes = num_processes
                final_num_of_threads = num_threads
                info += GREEN + f"{current_time/1e9} s" + RESET + "."
            else:
                info += RED + f"{current_time/1e9} s" + RESET + "."

            info += (
                " The best time: " + GREEN + f"{best_time/1e9} s" + RESET + "."
            )

            print(info)

        old_processes = new_processes

    print(
        "Job will run using"
        + YELLOW
        + f" {final_num_of_processes * final_num_of_threads}"
        + RESET
        + " logical"
        + YELLOW
        + " Processors"
        + RESET
        + " with"
        + BLUE
        + f" {final_num_of_processes}"
        + RESET
        + " parallel"
        + BLUE
        + " Processes"
        + RESET
        + " each utilizing"
        + PURPLE
        + f" {final_num_of_threads} threads"
        + RESET
        + ".\n"
        + "The calculation time (starting from now) is estimated to be at"
        " most: "
        + GREEN
        + f"{best_time/1e9} s"
        + RESET
        + "."
    )
    return final_num_of_threads
