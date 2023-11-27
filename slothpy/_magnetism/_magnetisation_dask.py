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
from time import perf_counter
from typing import Union
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
from numpy import (
    ndarray,
    array,
    sum,
    zeros,
    ascontiguousarray,
    diag,
    linspace,
    meshgrid,
    pi,
    sin,
    cos,
    newaxis,
    exp,
    float64,
    complex128,
    array_equal,
    concatenate,
)
from numpy.linalg import eigh
from numba import jit
from slothpy._general_utilities._constants import KB, MU_B
from slothpy._general_utilities._system import _get_num_of_processes
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._magnetism._zeeman import _calculate_zeeman_matrix
from dask.array import from_array
from dask.distributed import Client, as_completed
from tqdm import tqdm


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
    # Boltzman weights
    exp_diff = exp(-(energies - energies[0]) / (KB * temperature))

    # Partition function
    z = sum(exp_diff)

    # Weighted magnetic moments of microstates
    m = sum(states_momenta * exp_diff)

    return m / z


@jit(
    "float64[:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    nogil=True,
    fastmath=True,
)
def _mt_over_grid(magnetic_momenta, soc_energies, fields, grid, temperatures):
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

            eigenvalues, eigenvectors = eigh(zeeman_matrix)
            magnetic_momenta_cont = ascontiguousarray(magnetic_momenta)

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

        mht_array[i, :] = mt_array

    return mht_array


@jit(
    "float64[:,:,:](complex128[:,:,:], float64[:], float64, float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _mt_over_tensor(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    temperatures: ndarray,
):
    # Initialize arrays
    mt_tensor_array = ascontiguousarray(
        zeros((temperatures.shape[0], 3, 3), dtype=float64)
    )

    # Perform calculations for each tensor component
    for i in range(3):
        for j in range(3):
            # Construct Zeeman matrix
            zeeman_matrix = -field * MU_B * magnetic_momenta[j]
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

    return mt_tensor_array


def _calculate_mt(
    magnetic_momenta: str,
    soc_energies: str,
    field: float64,
    grid: Union[str, ndarray],
    temperatures: str,
    num_threads,
) -> ndarray:
    print("Starting MT")
    start = perf_counter()
    with threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpool_limits(limits=num_threads, user_api="openmp"):
            mt = _mt_over_grid(
                magnetic_momenta, soc_energies, field, grid, temperatures
            )
    stop = perf_counter()
    print(f"{stop - start}")

    return mt


def distribute_chunks(data, num_workers):
    """
    Yields chunks of data for distribution across workers.

    Args:
    - data (list): The dataset to be distributed.
    - num_workers (int): Number of workers (N).

    Yields:
    - Chunks of data, one for each worker.
    """
    n = len(data)
    chunk_size = n // num_workers
    remainder = n % num_workers

    for i in range(num_workers):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        yield data[start:end]


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

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = ascontiguousarray(fields, dtype=float64)
    temperatures = ascontiguousarray(temperatures, dtype=float64)
    grid = ascontiguousarray(grid, dtype=float64)

    with Client(
        n_workers=num_process, threads_per_worker=num_threads
    ) as client:
        magnetic_momenta_future = client.scatter(
            magnetic_momenta, broadcast=True
        )
        soc_energies_future = client.scatter(soc_energies, broadcast=True)
        grid_future = client.scatter(grid, broadcast=True)
        temperatures_future = client.scatter(temperatures, broadcast=True)

        fields_futures = client.scatter(
            [chunk for chunk in distribute_chunks(fields, num_process)]
        )

        # futures = []

        # workers = client.scheduler_info()['workers']

        # for worker, info in workers.items():
        #     print(f"Worker {worker}:")
        #     print(f"  - Threads: {info['nthreads']}")
        #     print(f"  - Memory limit: {info['memory_limit']}")

        # print(f"Dask dashboard available at: {client.dashboard_link}")

        # args_list = list[_arg_iter_mth(magnetic_momenta, soc_energies, fields_da, grid_da, temperatures_da)]

        # with threadpool_limits(limits=num_threads, user_api="blas"):
        #     with threadpool_limits(limits=num_threads, user_api="openmp"):
        # tutaj if else grid, tensor i jakei.s inne co do 3d wymyslisz i map directly do jit numby;
        # if
        # Using Dask's map function to distribute the tasks
        # for field in tqdm(fields, desc="Submitting tasks"):
        #     future = client.submit(_mt_over_grid, magnetic_momenta_future, soc_energies_future, field, grid_future, temperatures_future)
        #     futures.append(future)

        futures = [
            client.submit(
                _calculate_mt,
                magnetic_momenta_future,
                soc_energies_future,
                fields_chunk,
                grid_future,
                temperatures_future,
                num_threads,
            )
            for fields_chunk in fields_futures
        ]

        # else

        # progress_bar = tqdm(total=len(futures), desc="Processing tasks")

        # for future in as_completed(futures):
        #     progress_bar.update(1)

        # progress_bar.close()
        # Gather results
        # print("Collecting results...")
        mht = client.gather(futures)

    ######## To the description: Scales from laptops to clusters
    # Dask is convenient on a laptop. It installs trivially with conda or pip and extends the size of convenient datasets from “fits in memory” to “fits on disk”.

    # Dask can scale to a cluster of 100s of machines. It is resilient, elastic, data local, and low latency. For more information, see the documentation about the distributed scheduler.

    # This ease of transition between single-machine to moderate cluster enables users to both start simple and grow when necessary.

    # Hidden option for susceptibility tensor calculation.
    if array_equal(grid, array([1])):
        return array(mht)

    # Collecting results in plotting-friendly convention for M(H)
    mth_array = concatenate(mht).T

    return mth_array  # Returning values in Bohr magnetons
