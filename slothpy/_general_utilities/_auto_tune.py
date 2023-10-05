from os import cpu_count
from time import perf_counter_ns
from math import ceil
from statistics import mean
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
    zeros,
    diag,
    float64,
    complex128,
)
from numpy.linalg import eigh, eigvalsh
from threadpoolctl import threadpool_limits
from slothpy._magnetism._magnetisation import (
    _mt_over_grid,
    _calculate_magnetization,
)
from slothpy._magnetism._zeeman import (
    _calculate_helmholtz_energy,
    _calculate_zeeman_matrix,
)
from slothpy._general_utilities._system import _get_num_of_processes
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._constants import (
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
    energy: bool = False,
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
            energy,
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
    energy: bool = False,
) -> ndarray:
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


def _helmholtz_energyt_over_grid(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    grid: ndarray,
    temperatures: ndarray,
    internal_energy: bool = False,
) -> int:
    # Initialize arrays
    energyt_array = ascontiguousarray(
        zeros((temperatures.shape[0]), dtype=float64)
    )

    times = []

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        start = perf_counter_ns()
        # Construct Zeeman matrix
        orientation = grid[j, :3]

        zeeman_matrix = _calculate_zeeman_matrix(
            magnetic_momenta, soc_energies, field, orientation
        )

        # Diagonalize full Hamiltonian matrix
        eigenvalues = eigvalsh(zeeman_matrix)
        eigenvalues = ascontiguousarray(eigenvalues)

        # Compute Helmholtz energy for each T
        for t in range(temperatures.shape[0]):
            energyt_array[t] += (
                _calculate_helmholtz_energy(
                    eigenvalues, temperatures[t], internal_energy
                )
                * grid[j, 3]
            )
        end = perf_counter_ns()

        times.append(end - start)

    return min(times) * grid.shape[0]


def _mt_over_grid(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    grid: ndarray,
    temperatures: ndarray,
) -> int:
    # Initialize arrays
    mt_array = ascontiguousarray(zeros((temperatures.shape[0]), dtype=float64))

    times = []

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        start = perf_counter_ns()
        # Construct Zeeman matrix
        orientation = grid[j, :3]

        zeeman_matrix = _calculate_zeeman_matrix(
            magnetic_momenta, soc_energies, field, orientation
        )

        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = eigh(zeeman_matrix)
        magnetic_momenta = ascontiguousarray(magnetic_momenta)

        # Transform momenta according to the new eigenvectors
        states_momenta = (
            eigenvectors.conj().T
            @ (
                grid[j, 0] * magnetic_momenta[0]
                + grid[j, 1] * magnetic_momenta[1]
                + grid[j, 2] * magnetic_momenta[2]
            )
            @ eigenvectors
        )

        # Get diagonal momenta of the new states
        states_momenta = diag(states_momenta).real.astype(float64)

        # Compute partition function and magnetization for each T
        for t in range(temperatures.shape[0]):
            mt_array[t] += (
                _calculate_magnetization(
                    eigenvalues, states_momenta, temperatures[t]
                )
                * grid[j, 3]
            )
        end = perf_counter_ns()

        times.append(end - start)

    return min(times) * grid.shape[0]


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
    energy: bool = False,
) -> list[int]:
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

    if energy:
        start_time = perf_counter_ns()

        exec_time = _helmholtz_energyt_over_grid(
            magnetic_momenta_a,
            soc_energies_a,
            field,
            grid_a,
            temperatures_a,
            False,
        )

        mt = array(exec_time)

        end_time = perf_counter_ns()

    else:
        start_time = perf_counter_ns()

        exec_time = _mt_over_grid(
            magnetic_momenta_a, soc_energies_a, field, grid_a, temperatures_a
        )

        mt = array(exec_time)

        end_time = perf_counter_ns()

    setup_time = end_time - start_time - exec_time

    return [exec_time, setup_time]


def _get_mt_exec_time_wrapper(args):
    # Unpack arguments and call the function
    mt = _get_mt_exec_time(*args)

    return mt


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
    start_time_load = perf_counter_ns()
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0]
    )

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
                    dummy_return = p.map(
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
    dummy_return = array(dummy_return)
    end_time_load = perf_counter_ns()

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
    energy: bool = False,
) -> float64:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0]
    )

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
                    times = p.map(
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
                            energy,
                        ),
                    )

    times = array(times)
    exec_time = sorted(times[:, 0])
    setup_time = sorted(times[:, 1])
    return (
        mean(exec_time[: ceil(len(exec_time) / 4)]),
        mean(setup_time[: ceil(len(exec_time) / 4)]),
    )  # Can return min, max, median, or mean for different approaches
    # depending on internal_loop_samples size in _auto_tune so be careful what
    # you are doing.


def _auto_tune(
    filename,
    group,
    num_to_parallelize: int,
    matrix_size: int,
    internal_loop_size: int,
    internal_task_size: int,
    num_cpu: int = 0,
    internal_loop_samples: int = 1,
    energy: bool = False,
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
        if (
            new_processes != old_processes
            and old_processes <= num_to_parallelize
        ):
            num_processes = old_processes
            num_threads = threads - 1
            if num_processes == num_to_parallelize:
                fields = linspace(1, 10, num_processes)
            else:
                fields = linspace(1, 10, 2 * num_processes, dtype=float64)

            exec_setup_time = _mth_benchmark(
                filename,
                group,
                fields,
                grid,
                temperatures,
                matrix_size,
                num_cpu,
                num_threads,
                energy,
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

            current_time = (
                exec_setup_time[0] * internal_loop_size / internal_loop_samples
                + exec_setup_time[1]
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
        + "The calculation time (starting from now) is estimated to be: "
        + GREEN
        + f"{best_time/1e9} s"
        + RESET
        + "."
    )
    return num_cpu, final_num_of_threads
