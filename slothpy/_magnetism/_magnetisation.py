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
)
from numpy.linalg import eigh
from numba import jit
from slothpy._general_utilities._constants import KB, MU_B
from slothpy._general_utilities._system import _get_num_of_processes
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._magnetism._zeeman import _calculate_zeeman_matrix


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
    "float64[:](complex128[:,:,:], float64[:], float64, float64[:,:],"
    " float64[:])",
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
) -> ndarray:
    # Initialize arrays
    mt_array = ascontiguousarray(zeros((temperatures.shape[0]), dtype=float64))

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
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

    return mt_array


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
    m_s: tuple,
    s_s: tuple,
    t_s: tuple,
    g_s: tuple = 0,
) -> ndarray:
    # g_s == 0 only if functions don't provide it and then they must pass
    # a single grid point directly from the wrapper as grid variable (not name)
    if g_s != 0:
        # Here grid can be equal to array([1]) which activates sus-tensor calc
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

    # Hidden option for susceptibility tensor calculation.
    if array_equal(grid_a, array([1])):
        return _mt_over_tensor(
            magnetic_momenta_a, soc_energies_a, field, temperatures_a
        )

    return _mt_over_grid(
        magnetic_momenta_a, soc_energies_a, field, grid_a, temperatures_a
    )


def _calculate_mt_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mt(*args)

    return mt


def _arg_iter_mth(
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
                    mht = p.map(
                        _calculate_mt_wrapper,
                        _arg_iter_mth(
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

    # Hidden option for susceptibility tensor calculation.
    if array_equal(grid, array([1])):
        return array(mht)

    # Collecting results in plotting-friendly convention for M(H)
    mth_array = array(mht).T

    return mth_array  # Returning values in Bohr magnetons


def _arg_iter_mag_3d(
    magnetic_moment,
    soc_energies,
    fields,
    theta,
    phi,
    temperatures,
    m_shape,
    s_shape,
    t_shape,
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
                )


def _mag_3d(
    filename: str,
    group: str,
    fields: ndarray,
    spherical_grid: int,
    temperatures: ndarray,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    sus_3d_num: bool = False,
    rotation: ndarray = None,
) -> ndarray:
    # Get number of parallel proceses to be used
    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0] * 2 * spherical_grid**2
    )

    # Create a gird
    theta = linspace(0, 2 * pi, 2 * spherical_grid, dtype=float64)
    phi = linspace(0, pi, spherical_grid, dtype=float64)
    theta, phi = meshgrid(theta, phi)

    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename,
        group,
        states_cutoff,
        rotation,
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
                        _calculate_mt_wrapper,
                        _arg_iter_mag_3d(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared_arr,
                            theta_shared_arr,
                            phi_shared_arr,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            temperatures_shared_arr.shape,
                        ),
                    )

    mag_3d = array(mht).reshape(
        (fields.shape[0], phi.shape[0], phi.shape[1], temperatures.shape[0])
    )
    mag_3d = mag_3d.transpose((0, 3, 1, 2))

    if sus_3d_num:
        return mag_3d

    mag_3d_array = zeros((3, *mag_3d.shape), dtype=float64)

    mag_3d_array[0] = (sin(phi) * cos(theta))[newaxis, newaxis, :, :] * mag_3d
    mag_3d_array[1] = (sin(phi) * sin(theta))[newaxis, newaxis, :, :] * mag_3d
    mag_3d_array[2] = (cos(phi))[newaxis, newaxis, :, :] * mag_3d

    return mag_3d_array
