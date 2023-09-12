import time
import multiprocessing
import multiprocessing.managers
import threadpoolctl
import numpy as np
from numba import jit
from slothpy.general_utilities.system import _get_num_of_processes
from slothpy.general_utilities.io import (
    _get_soc_mag_mom_and_ener_from_hdf5,
)
from slothpy.magnetism.zeeman import calculate_zeeman_matrix


@jit(
    "float64(float64[:], float64[:], float64)",
    nopython=True,
    cache=True,
    nogil=True,
)
def _calculate_magnetization(
    energies: np.ndarray, states_momenta: np.ndarray, temperature: np.float64
) -> np.float64:
    """
    Calculates the magnetization for a given array of states energies, momenta, and temperature.

    Args:
        energies (np.ndarray[np.float64]): Array of energies.
        states_momenta (np.ndarray[np.float64]): Array of states momenta.
        temperature (np.float64): Temperature value.

    Returns:
        np.float64: Magnetization value.

    """
    kB = 3.166811563e-6  # Boltzmann constant a.u./K

    # Boltzman weights
    exp_diff = np.exp(-(energies - energies[0]) / (kB * temperature))

    # Partition function
    z = np.sum(exp_diff)

    # Weighted magnetic moments of microstates
    m = np.sum(states_momenta * exp_diff)

    return m / z


# @jit(
#     (
#         "float64[:](complex128[:,:,:], float64[:], float64, float64[:,:],"
#         " float64[:])"
#     ),
#     nopython=True,
#     cache=True,
#     nogil=True,
# )
def _calculate_mt(
    magnetic_momenta: np.ndarray,
    soc_energies: np.ndarray,
    field: np.float64,
    grid: np.ndarray,
    temperatures: np.ndarray,
    m_s,
    s_s,
    g_s,
    t_s,
) -> np.ndarray:
    """
    Calculates the M(T) array for a given array of magnetic moments, SOC energies, directional grid for powder averaging,
    and temperatures for a particular value of magnetic field.

    Args:
        magnetic_moment (np.ndarray[np.complex128]): Array of magnetic moments.
        soc_energies (np.ndarray[np.float64]): Array of SOC energies.
        field (np.float64): Value of magnetic field.
        grid (np.ndarray[np.float64]): Grid array.
        temperatures (np.ndarray[np.float64]): Array of temperatures.

    Returns:
        np.ndarray[np.float64]: M(T) array.

    """

    grid = np.ndarray(g_s, dtype=np.float64, buffer=grid.buf)
    temperatures = np.ndarray(
        t_s,
        dtype=np.float64,
        buffer=temperatures.buf,
    )
    magnetic_momenta = np.ndarray(
        m_s,
        dtype=np.complex128,
        buffer=magnetic_momenta.buf,
    )
    soc_energies = np.ndarray(s_s, dtype=np.float64, buffer=soc_energies.buf)

    # Initialize arrays
    mt_array = np.ascontiguousarray(
        np.zeros((temperatures.shape[0]), dtype=np.float64)
    )
    magnetic_momenta = np.ascontiguousarray(magnetic_momenta)
    soc_energies = np.ascontiguousarray(soc_energies)

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        # Construct Zeeman matrix
        orientation = grid[j, :3]

        zeeman_matrix = calculate_zeeman_matrix(
            magnetic_momenta, soc_energies, field, orientation
        )

        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
        eigenvalues = np.ascontiguousarray(eigenvalues)
        eigenvectors = np.ascontiguousarray(eigenvectors)

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
        states_momenta = np.diag(states_momenta).real.astype(np.float64)

        # Compute partition function and magnetization for each T
        for t in range(temperatures.shape[0]):
            mt_array[t] += (
                _calculate_magnetization(
                    eigenvalues, states_momenta, temperatures[t]
                )
                * grid[j, 3]
            )

    return mt_array


def _calculate_mt_wrapper(args):
    """Wrapper function for parallel use of M(T) calulations

    Args:
        args (tuple): Tuple of arguments for calculate_mt function

    Returns:
        np.ndarray[np.float64]: M(T) array.
    """
    # Unpack arguments and call the function
    mt = _calculate_mt(*args)

    return mt


def _arg_iter_mth(
    fields,
    magnetic_momenta,
    soc_energies,
    grid,
    temperatures,
    m_s,
    s_s,
    g_s,
    t_s,
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
            g_s,
            t_s,
        )


def _mth(
    filename: str,
    group: str,
    fields: np.ndarray[np.float64],
    grid: np.ndarray[np.float64],
    temperatures: np.ndarray[np.float64],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
) -> np.ndarray:
    # Get number of parallel proceses to be used
    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Initialize the result array
    mth_array = np.zeros((temperatures.size, fields.size), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)
    temperatures = np.ascontiguousarray(temperatures)

    # Read data from HDF5 file
    (
        mag_mom,
        soc_ener,
    ) = _get_soc_mag_mom_and_ener_from_hdf5(filename, group, states_cutoff)

    with multiprocessing.managers.SharedMemoryManager() as smm:
        # Create shared memory for arrays
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)
        mag_mom_shared = smm.SharedMemory(size=mag_mom.nbytes)
        soc_ener_shared = smm.SharedMemory(size=soc_ener.nbytes)

        # Copy data to shared memory
        fields_shared_arr = np.ndarray(
            fields.shape, dtype=fields.dtype, buffer=fields_shared.buf
        )
        grid_shared_arr = np.ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )
        temperatures_shared_arr = np.ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )
        mag_mom_shared_arr = np.ndarray(
            mag_mom.shape, dtype=mag_mom.dtype, buffer=mag_mom_shared.buf
        )
        soc_ener_shared_arr = np.ndarray(
            soc_ener.shape, dtype=soc_ener.dtype, buffer=soc_ener_shared.buf
        )

        g_shape = grid.shape
        t_shape = temperatures.shape
        s_shape = soc_ener.shape
        m_shape = mag_mom.shape

        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]
        soc_ener_shared_arr[:] = soc_ener[:]
        mag_mom_shared_arr[:] = mag_mom[:]

        with threadpoolctl.threadpool_limits(
            limits=num_threads, user_api="blas"
        ):
            with threadpoolctl.threadpool_limits(
                limits=num_threads, user_api="openmp"
            ):
                with multiprocessing.Pool(num_process) as p:
                    mt = p.map(
                        _calculate_mt_wrapper,
                        _arg_iter_mth(
                            fields_shared_arr,
                            mag_mom_shared,
                            soc_ener_shared,
                            grid_shared,
                            temperatures_shared,
                            m_shape,
                            s_shape,
                            g_shape,
                            t_shape,
                        ),
                    )

        # Collecting results in plotting-friendly convention for M(H)
        for i in range(fields.shape[0]):
            mth_array[:, i] = mt[i]

    # # Clean up shared memory
    # fields_shared.close()
    # fields_shared.unlink()
    # grid_shared.close()
    # grid_shared.unlink()
    # temperatures_shared.close()
    # temperatures_shared.unlink()
    # mag_mom_shared.close()
    # mag_mom_shared.unlink()
    # soc_ener_shared.close()
    # soc_ener_shared.unlink()

    return mth_array  # Returning values in Bohr magnetons


def _arg_iter_mag_3d(
    magnetic_moment, soc_energies, fields, theta, phi, temperatures
):
    for k in range(fields.shape[0]):
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                yield (
                    magnetic_moment,
                    soc_energies,
                    fields[k],
                    np.array(
                        [
                            [
                                np.sin(phi[i, j]) * np.cos(theta[i, j]),
                                np.sin(phi[i, j]) * np.sin(theta[i, j]),
                                np.cos(phi[i, j]),
                                1.0,
                            ]
                        ],
                        dtype=np.float64,
                    ),
                    temperatures,
                )


def _mag_3d(
    filename: str,
    group: str,
    states_cutoff: int,
    fields: np.ndarray,
    spherical_grid: int,
    temperatures: np.ndarray,
    num_cpu: int,
    num_threads: int,
) -> np.ndarray:
    # Get number of parallel proceses to be used
    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Create a gird
    theta = np.linspace(0, 2 * np.pi, 2 * spherical_grid)
    phi = np.linspace(0, np.pi, spherical_grid)
    theta, phi = np.meshgrid(theta, phi)

    # Initialize the result array
    mag_3d_array = np.zeros(
        (fields.shape[0], temperatures.shape[0], phi.shape[0], phi.shape[1]),
        dtype=np.float64,
    )

    # Read data from HDF5 file
    (
        magnetic_moment,
        soc_energies,
    ) = _get_soc_mag_mom_and_ener_from_hdf5(filename, group, states_cutoff)

    with threadpoolctl.threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpoolctl.threadpool_limits(
            limits=num_threads, user_api="openmp"
        ):
            # Parallel M(T,H) calculation over different grid points
            with multiprocessing.Pool(num_process) as p:
                mth = p.map(
                    _calculate_mt_wrapper,
                    _arg_iter_mag_3d(
                        magnetic_moment,
                        soc_energies,
                        fields,
                        theta,
                        phi,
                        temperatures,
                    ),
                )

            pool_index = 0
            for k in range(fields.shape[0]):
                for i in range(phi.shape[0]):
                    for j in range(phi.shape[1]):
                        mag_3d_array[k, :, i, j] = mth[pool_index][:]
                        pool_index += 1

    x = (np.sin(phi) * np.cos(theta))[
        np.newaxis, np.newaxis, :, :
    ] * mag_3d_array
    y = (np.sin(phi) * np.sin(theta))[
        np.newaxis, np.newaxis, :, :
    ] * mag_3d_array
    z = (np.cos(phi))[np.newaxis, np.newaxis, :, :] * mag_3d_array

    return x, y, z
