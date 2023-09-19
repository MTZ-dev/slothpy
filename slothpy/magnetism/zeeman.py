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
    log,
    float64,
    complex128,
)
from numpy.linalg import eigh, eigvalsh
from numba import jit
from slothpy.general_utilities._constants import KB, MU_B, H_CM_1
from slothpy.general_utilities.system import _get_num_of_processes
from slothpy.general_utilities.io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
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


# @jit(
#     (
#         "float64[:,:](complex128[:,:,:], float64[:], float64, float64[:,:],"
#         " int64)"
#     ),
#     nopython=True,
#     cache=True,
#     nogil=True,
#     fastmath=True,
# )
def calculate_zeeman_splitting(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    grid: ndarray,
    num_of_states: int,
) -> ndarray:
    pass
    # hartree_to_cm_1 = 219474.6  # atomic units to wavenumbers

    # # Initialize arrays and scale energy to the ground SOC state
    # zeeman_array = np.zeros((grid.shape[0], num_of_states), dtype=np.float64)
    # magnetic_momenta = np.ascontiguousarray(magnetic_momenta)
    # soc_energies = np.ascontiguousarray(soc_energies - soc_energies[0])

    # # Perform calculations for each magnetic field orientation
    # for j in range(grid.shape[0]):
    #     orientation = grid[j, :3]

    #     zeeman_matrix = _calculate_zeeman_matrix(
    #         magnetic_momenta, soc_energies, field, orientation
    #     )

    #     # Diagonalize full Zeeman Hamiltonian
    #     energies = np.linalg.eigvalsh(zeeman_matrix)

    #     # Get only desired number of states in cm-1
    #     energies = energies[:num_of_states] * hartree_to_cm_1

    #     # Collect the results
    #     zeeman_array[j, :] = energies

    # return zeeman_array


def caculate_zeeman_splitting_wrapper(args):
    zeeman_array = calculate_zeeman_splitting(*args)

    return zeeman_array


def arg_iter_zeeman(
    magnetic_moment, soc_energies, fields, grid, num_of_states
):
    # Iterator generator for arguments with different field values to be distributed along num_process processes
    for i in range(fields.shape[0]):
        yield (magnetic_moment, soc_energies, fields[i], grid, num_of_states)


def zeeman_splitting(
    filename: str,
    group: str,
    states_cutoff: int,
    num_of_states: int,
    fields: ndarray,
    grid: ndarray,
    num_cpu: int,
    num_threads: int,
    average: bool = False,
) -> ndarray:
    pass
    # # Get number of parallel proceses to be used
    # num_process = get_num_of_processes(num_cpu, num_threads)

    # # Initialize the result array
    # zeeman_array = np.zeros(
    #     (grid.shape[0], fields.shape[0], num_of_states), dtype=np.float64
    # )

    # # Get magnetic field in a.u. and allocate arrays as contiguous
    # fields = np.ascontiguousarray(fields)
    # grid = np.ascontiguousarray(grid)

    # # Read data from HDF5 file
    # (
    #     magnetic_moment,
    #     soc_energies,
    # ) = get_soc_magnetic_momenta_and_energies_from_hdf5(
    #     filename, group, states_cutoff
    # )

    # with threadpoolctl.threadpool_limits(limits=num_threads, user_api="blas"):
    #     with threadpoolctl.threadpool_limits(
    #         limits=num_threads, user_api="openmp"
    #     ):
    #         # Parallel M(T) calculation over different field values
    #         with multiprocessing.Pool(num_process) as p:
    #             zeeman = p.map(
    #                 caculate_zeeman_splitting_wrapper,
    #                 arg_iter_zeeman(
    #                     magnetic_moment,
    #                     soc_energies,
    #                     fields,
    #                     grid,
    #                     num_of_states,
    #                 ),
    #             )

    # # Collecting results
    # for i in range(fields.shape[0]):
    #     zeeman_array[:, i, :] = zeeman[i]

    # # Average over directions
    # if average == True:
    #     zeeman_array_av = np.zeros(
    #         (fields.shape[0], num_of_states), dtype=np.float64
    #     )
    #     for i in range(fields.shape[0]):
    #         for j in range(grid.shape[0]):
    #             zeeman_array_av[i, :] += zeeman_array[j, i, :] * grid[j, 3]

    #     return zeeman_array_av

    # return zeeman_array


def get_zeeman_matrix(
    filename: str,
    group: str,
    states_cutoff: int,
    field: float64,
    orientation: ndarray,
) -> ndarray:
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )
    zeeman_matrix = _calculate_zeeman_matrix(
        magnetic_momenta, soc_energies, field, orientation
    )

    return zeeman_matrix


@jit(
    "float64(float64[:], float64, boolean)",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _calculate_hemholtz_energy(
    energies: ndarray, temperature: float64, internal_energy: False
) -> float64:
    energies = energies[1:] - energies[0]

    # Boltzman weights
    exp_diff = exp(-(energies) / (KB * temperature))

    # Partition function
    z = sum(exp_diff)

    # Float64 precision
    z = max(z, 1e-307)

    if internal_energy:
        e = sum((energies * H_CM_1) * exp_diff)
        return e / z
    else:
        return -KB * temperature * log(z) * H_CM_1


@jit(
    (
        "float64[:](complex128[:,:,:], float64[:], float64, float64[:,:],"
        " float64[:], boolean)"
    ),
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _hemholtz_energyt_over_grid(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    field: float64,
    grid: ndarray,
    temperatures: ndarray,
    internal_energy: bool = False,
) -> ndarray:
    # Initialize arrays
    energyt_array = ascontiguousarray(
        zeros((temperatures.shape[0]), dtype=float64)
    )

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        # Construct Zeeman matrix
        orientation = grid[j, :3]

        zeeman_matrix = _calculate_zeeman_matrix(
            magnetic_momenta, soc_energies, field, orientation
        )

        # Diagonalize full Hamiltonian matrix
        eigenvalues = eigvalsh(zeeman_matrix)
        eigenvalues = ascontiguousarray(eigenvalues)

        # Compute Hemholtz energy for each T
        for t in range(temperatures.shape[0]):
            energyt_array[t] += (
                _calculate_hemholtz_energy(
                    eigenvalues, temperatures[t], internal_energy
                )
                * grid[j, 3]
            )

    return energyt_array


def _calculate_hemholtz_energyt(
    magnetic_momenta: str,
    soc_energies: str,
    field: float64,
    grid: str,
    temperatures: str,
    m_s: int,
    s_s: int,
    t_s: int,
    g_s: int = 0,
    internal_energy: bool = False,
) -> ndarray:
    # Option to enable calculations with only a single grid point.
    if g_s != 0:
        grid_s = SharedMemory(name=grid)
        grid_a = ndarray(g_s, dtype=float64, buffer=grid_s.buf)
    else:
        grid_a = grid

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

    return _hemholtz_energyt_over_grid(
        magnetic_momenta_a,
        soc_energies_a,
        field,
        grid_a,
        temperatures_a,
        internal_energy,
    )


def _calculate_hemholtz_energyt_wrapper(args):
    # Unpack arguments and call the function
    et = _calculate_hemholtz_energyt(*args)

    return et


def _arg_iter_hemholtz_energyth(
    magnetic_momenta,
    soc_energies,
    fields,
    grid,
    temperatures,
    m_s,
    s_s,
    t_s,
    g_s,
    internal_energy,
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
            internal_energy,
        )


def _hemholtz_energyth(
    filename: str,
    group: str,
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    internal_energy: bool = False,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    # Get number of parallel proceses to be used
    num_process = _get_num_of_processes(num_cpu, num_threads)

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
                    eht = p.map(
                        _calculate_hemholtz_energyt_wrapper,
                        _arg_iter_hemholtz_energyth(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared_arr,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            grid_shared_arr.shape,
                            internal_energy,
                        ),
                    )

    # Collecting results in plotting-friendly convention for as for the M(H)
    eth_array = array(eht).T

    return eth_array  # Returning values in cm-1


def _arg_iter_hemholtz_energy_3d(
    magnetic_moment,
    soc_energies,
    fields,
    theta,
    phi,
    temperatures,
    m_shape,
    s_shape,
    t_shape,
    internal_energy,
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
                    m_shape,
                    s_shape,
                    t_shape,
                    0,
                    internal_energy,
                )


def _hemholtz_energy_3d(
    filename: str,
    group: str,
    fields: ndarray,
    spherical_grid: int,
    temperatures: ndarray,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    internal_energy: bool = False,
) -> ndarray:
    # Get number of parallel proceses to be used
    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Create a gird
    theta = linspace(0, 2 * pi, 2 * spherical_grid, dtype=float64)
    phi = linspace(0, pi, spherical_grid, dtype=float64)
    theta, phi = meshgrid(theta, phi)

    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    fields = ascontiguousarray(fields, dtype=float64)
    temperatures = ascontiguousarray(temperatures, dtype=float64)

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)
        theta_shared = smm.SharedMemory(size=theta.nbytes)
        phi_shared = smm.SharedMemory(size=phi.nbytes)

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
        theta_shared_arr = ndarray(
            theta.shape, dtype=theta.dtype, buffer=theta_shared.buf
        )
        phi_shared_arr = ndarray(
            phi.shape, dtype=phi.dtype, buffer=phi_shared.buf
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        temperatures_shared_arr[:] = temperatures[:]
        theta_shared_arr[:] = theta[:]
        phi_shared_arr[:] = phi[:]

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                # Parallel M(T,H) calculation over different grid points
                with Pool(num_process) as p:
                    mht = p.map(
                        _calculate_hemholtz_energyt_wrapper,
                        _arg_iter_hemholtz_energy_3d(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared_arr,
                            theta_shared_arr,
                            phi_shared_arr,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            internal_energy=internal_energy,
                        ),
                    )

    energy_3d = array(mht).reshape(
        (fields.shape[0], phi.shape[0], phi.shape[1], temperatures.shape[0])
    )
    energy_3d = energy_3d.transpose((0, 3, 1, 2))

    energy_3d_array = zeros((3, *energy_3d.shape), dtype=float64)

    energy_3d_array[0] = (sin(phi) * cos(theta))[
        newaxis, newaxis, :, :
    ] * energy_3d
    energy_3d_array[1] = (sin(phi) * sin(theta))[
        newaxis, newaxis, :, :
    ] * energy_3d
    energy_3d_array[2] = (cos(phi))[newaxis, newaxis, :, :] * energy_3d

    return energy_3d_array
