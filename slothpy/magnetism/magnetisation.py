from multiprocessing.managers import SharedMemoryManager
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
)
from numpy.linalg import eigh
from numba import jit
from slothpy.general_utilities.system import _get_num_of_processes
from slothpy.general_utilities.io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy.magnetism.zeeman import calculate_zeeman_matrix


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
    exp_diff = exp(-(energies - energies[0]) / (kB * temperature))

    # Partition function
    z = sum(exp_diff)

    # Weighted magnetic moments of microstates
    m = sum(states_momenta * exp_diff)

    return m / z


@jit(
    (
        "float64[:](complex128[:,:,:], float64[:], float64, float64[:,:],"
        " float64[:])"
    ),
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _mt_over_grid(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    grid: ndarray,
    temperatures: ndarray,
):
    # Initialize arrays
    mt_array = ascontiguousarray(zeros((temperatures.shape[0]), dtype=float64))

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        # Construct Zeeman matrix
        orientation = grid[j, :3]

        zeeman_matrix = calculate_zeeman_matrix(
            magnetic_momenta, soc_energies, field, orientation
        )
        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = eigh(zeeman_matrix)
        # eigenvalues = np.ascontiguousarray(eigenvalues)
        # eigenvectors = np.ascontiguousarray(eigenvectors)

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

    return mt_array


def _calculate_mt(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    grid: ndarray,
    temperatures: ndarray,
    m_s,
    s_s,
    g_s,
    t_s,
) -> ndarray:
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

    grid = ndarray(g_s, dtype=float64, buffer=grid.buf)
    temperatures = ndarray(
        t_s,
        dtype=float64,
        buffer=temperatures.buf,
    )
    magnetic_momenta = ndarray(
        m_s,
        dtype=complex128,
        buffer=magnetic_momenta.buf,
    )
    soc_energies = ndarray(s_s, dtype=float64, buffer=soc_energies.buf)

    return _mt_over_grid(
        magnetic_momenta, soc_energies, field, grid, temperatures
    )


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
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
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

    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Initialize the result array
    mth_array = ascontiguousarray(
        zeros((temperatures.size, fields.size), dtype=float64)
    )

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = ascontiguousarray(fields)
    grid = ascontiguousarray(grid)
    temperatures = ascontiguousarray(temperatures)

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)

        # Copy data to shared memory
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

        g_shape = grid.shape
        t_shape = temperatures.shape
        s_shape = soc_energies.shape
        m_shape = magnetic_momenta.shape

        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                with Pool(num_process) as p:
                    mt = p.map(
                        _calculate_mt_wrapper,
                        _arg_iter_mth(
                            fields_shared_arr,
                            magnetic_momenta_shared,
                            soc_energies_shared,
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
                    array(
                        [
                            [
                                sin(phi[i, j]) * cos(theta[i, j]),
                                sin(phi[i, j]) * sin(theta[i, j]),
                                cos(phi[i, j]),
                                1.0,
                            ]
                        ],
                        dtype=float64,
                    ),
                    temperatures,
                )


def _mag_3d(
    filename: str,
    group: str,
    states_cutoff: int,
    fields: ndarray,
    spherical_grid: int,
    temperatures: ndarray,
    num_cpu: int,
    num_threads: int,
) -> ndarray:
    # Get number of parallel proceses to be used
    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Create a gird
    theta = linspace(0, 2 * pi, 2 * spherical_grid)
    phi = linspace(0, pi, spherical_grid)
    theta, phi = meshgrid(theta, phi)

    # Initialize the result array
    mag_3d_array = zeros(
        (fields.shape[0], temperatures.shape[0], phi.shape[0], phi.shape[1]),
        dtype=float64,
    )

    # Read data from HDF5 file
    (
        magnetic_moment,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    with threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpool_limits(limits=num_threads, user_api="openmp"):
            # Parallel M(T,H) calculation over different grid points
            with Pool(num_process) as p:
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

    x = (sin(phi) * cos(theta))[newaxis, newaxis, :, :] * mag_3d_array
    y = (sin(phi) * sin(theta))[newaxis, newaxis, :, :] * mag_3d_array
    z = (cos(phi))[newaxis, newaxis, :, :] * mag_3d_array

    return x, y, z
