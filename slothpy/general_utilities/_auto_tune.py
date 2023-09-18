from os import cpu_count
from time import perf_counter
from statistics import mean
from math import ceil
from typing import Tuple
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from numpy import (
    ndarray,
    array,
    ones,
    ascontiguousarray,
    linspace,
    float64,
    complex128,
)
from threadpoolctl import threadpool_limits
from slothpy.magnetism._magnetisation import _mt_over_grid
from slothpy.general_utilities.system import _get_num_of_processes
from slothpy.general_utilities.io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy.general_utilities._constants import (
    YELLOW,
    BLUE,
    PURPLE,
    GREEN,
    RED,
    RESET,
)


def _arg_iter_mth_benchmark(
    magnetic_momenta,
    soc_energies,
    fields,
    grid,
    temperatures,
    m_s,
    s_s,
    t_s,
    g_s,
):
    # Iterator generator for arguments with different field values to be
    # distributed along num_process processes
    for i in range(fields.shape[0]):
        yield (
            magnetic_momenta,
            soc_energies,
            fields[i],
            grid,
            temperatures,
            m_s,
            s_s,
            t_s,
            g_s,
        )


def _dummy_function(
    magnetic_momenta: str,
    soc_energies: str,
    field: float64,
    grid: str,
    temperatures: str,
    m_s: int,
    s_s: int,
    t_s: int,
    g_s: int = 0,
) -> float64:
    grid_s = SharedMemory(name=grid)
    grid_a = ndarray(g_s, dtype=float64, buffer=grid_s.buf)
    temperatures_s = SharedMemory(name=temperatures)
    temperatures_a = ndarray(
        t_s,
        dtype=float64,
        buffer=temperatures_s.buf,
    )
    magnetic_momenta_s = SharedMemory(name=magnetic_momenta)
    magnetic_momenta_a = ndarray(
        m_s,
        dtype=complex128,
        buffer=magnetic_momenta_s.buf,
    )
    soc_energies_s = SharedMemory(name=soc_energies)
    soc_energies_a = ndarray(s_s, dtype=float64, buffer=soc_energies_s.buf)

    dummy = soc_energies_a + 1
    dummy = magnetic_momenta_a + 1
    dummy = temperatures_a + 1
    dummy = grid_a + 1
    field = field + 1

    if dummy.any():
        return ones(temperatures_a.shape, dtype=float64)
    else:
        return ones(temperatures_a.shape, dtype=float64)


def _dummy_function_wrapper(args):
    # Unpack arguments and call the function
    mt = _dummy_function(*args)

    return mt


def _get_mt_exec_time(
    magnetic_momenta: str,
    soc_energies: str,
    field: float64,
    grid: str,
    temperatures: str,
    m_s: int,
    s_s: int,
    t_s: int,
    g_s: int = 0,
) -> float64:
    grid_s = SharedMemory(name=grid)
    grid_a = ndarray(g_s, dtype=float64, buffer=grid_s.buf)
    temperatures_s = SharedMemory(name=temperatures)
    temperatures_a = ndarray(
        t_s,
        dtype=float64,
        buffer=temperatures_s.buf,
    )
    magnetic_momenta_s = SharedMemory(name=magnetic_momenta)
    magnetic_momenta_a = ndarray(
        m_s,
        dtype=complex128,
        buffer=magnetic_momenta_s.buf,
    )
    soc_energies_s = SharedMemory(name=soc_energies)
    soc_energies_a = ndarray(s_s, dtype=float64, buffer=soc_energies_s.buf)

    start_time = perf_counter()

    _mt_over_grid(
        magnetic_momenta_a, soc_energies_a, field, grid_a, temperatures_a
    )

    end_time = perf_counter()

    return end_time - start_time


def _get_mt_exec_time_wrapper(args):
    start_time = perf_counter()

    # Unpack arguments and call the function
    mt = _get_mt_exec_time(*args)

    end_time = perf_counter()

    return [mt, end_time - start_time]


def _mth_load(
    filename: str,
    group: str,
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
) -> float64:
    start_time_load = perf_counter()
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Get magnetic field in a.u. and allocate arrays as contiguous
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
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)
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
        temperatures_shared_arr = ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )
        grid_shared_arr = ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        temperatures_shared_arr[:] = temperatures[:]
        grid_shared_arr[:] = grid[:]

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                with Pool(num_process) as p:
                    exec_time = p.map(
                        _dummy_function_wrapper,
                        _arg_iter_mth_benchmark(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared_arr,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            grid_shared_arr.shape,
                        ),
                    )
    end_time_load = perf_counter()

    return end_time_load - start_time_load


def _mth_benchmark(
    filename: str,
    group: str,
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
) -> float64:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Get magnetic field in a.u. and allocate arrays as contiguous
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
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)
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
        temperatures_shared_arr = ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )
        grid_shared_arr = ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        temperatures_shared_arr[:] = temperatures[:]
        grid_shared_arr[:] = grid[:]

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                with Pool(num_process) as p:
                    exec_time = p.map(
                        _get_mt_exec_time_wrapper,
                        _arg_iter_mth_benchmark(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared_arr,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            grid_shared_arr.shape,
                        ),
                    )

    exec_time = array(exec_time)
    setup_time = mean(
        exec_time[:, 1] - exec_time[:, 0]
    )  # Four as in the chunksize 4 * len(self._pool)

    return setup_time, mean(
        exec_time[:, 0]
    )  # Can return min, max or mean for different approaches depending on
    # internal_loop_samples size in _auto_tune so be careful what you are
    # doing.


def _auto_tune(
    filename,
    group,
    num_to_parallelize: int,
    matrix_size: int,
    internal_loop_size: int,
    internal_task_size: int,
    num_cpu: int = 0,
    internal_loop_samples: int = 1,
) -> Tuple[int, int]:
    temperatures = linspace(1, 600, internal_task_size, dtype=float64)
    grid = ones(
        (internal_loop_samples, 4),
        dtype=float64,
    )
    fields_load = ones(num_to_parallelize, dtype=float64)

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
        if new_processes != old_processes:
            num_processes = old_processes
            num_threads = threads - 1
            fields = linspace(1, 10, num_processes, dtype=float64)

            setup_time, exec_time = _mth_benchmark(
                filename,
                group,
                fields,
                grid,
                temperatures,
                matrix_size,
                num_cpu,
                num_threads,
            )
            load_time = _mth_load(
                filename,
                group,
                fields_load,
                grid,
                temperatures,
                matrix_size,
                num_cpu,
                num_threads,
            )

            print(load_time)

            current_time = (
                exec_time * internal_loop_size / internal_loop_samples
                + setup_time
            ) * ceil(num_to_parallelize / num_processes) + load_time

            info = (
                "Processes:"
                f" {num_processes}, threads: {num_threads}."
                " Estimated execution time: "
            )

            if current_time < best_time:
                best_time = current_time
                final_num_of_processes = num_processes
                final_num_of_threads = num_threads
                info += GREEN + f"{current_time} s" + RESET + "."
            else:
                info += RED + f"{current_time} s" + RESET + "."

            info += " The best time: " + GREEN + f"{best_time}s" + RESET + "."

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
        + "The calculation time (starting from now) is estimated to be: "
        + GREEN
        + f"{best_time} s"
        + RESET
        + "."
    )
    return num_cpu, final_num_of_threads
