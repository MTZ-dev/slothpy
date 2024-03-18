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

from math import factorial
from numpy import (
    ndarray,
    array,
    zeros,
    ascontiguousarray,
    arange,
    tile,
    abs,
    mod,
    sqrt,
    min,
    max,
    power,
    float64,
    int64,
    complex128,
)
from numpy.linalg import eigh, inv
from numba import jit
from slothpy._general_utilities._constants import GE, MU_B
from slothpy.core._slothpy_exceptions import SltInputError
from slothpy.core._config import settings


@jit(
    f"complex128[:, :](complex128[:, :, :], {settings.numba_float}[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _3d_dot(u, m):
    return u[0] * m[0] + u[1] * m[1] + u[2] * m[2]


@jit("float64(float64, float64)", nopython=True, cache=True, nogil=True)
def _binom(n, k):
    if k > n - k:
        k = n - k
    res = 1
    for i in range(k):
        res *= n - i
        res /= i + 1
    return res


@jit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nopython=True,
    cache=True,
    nogil=True,
)
def Clebsh_Gordan(j1, m1, j2, m2, j3, m3):
    cg_coeff = 0

    if (
        (m1 + m2 != m3)
        or (j1 < 0.0)
        or (j2 < 0.0)
        or (j3 < 0.0)
        or abs(m1) > j1
        or abs(m2) > j2
        or abs(m3) > j3
        or (abs(j1 - j2) > j3)
        or ((j1 + j2) < j3)
        or (abs(j2 - j3) > j1)
        or ((j2 + j3) < j1)
        or (abs(j3 - j1) > j2)
        or ((j3 + j1) < j2)
        or (mod(int(2.0 * j1), 2) != mod(int(2.0 * abs(m1)), 2))
        or (mod(int(2.0 * j2), 2) != mod(int(2.0 * abs(m2)), 2))
        or (mod(int(2.0 * j3), 2) != mod(int(2.0 * abs(m3)), 2))
    ):
        return cg_coeff

    J = j1 + j2 + j3
    C = sqrt(
        _binom(2 * j1, J - 2 * j2)
        * _binom(2 * j2, J - 2 * j3)
        / (
            _binom(J + 1, J - 2 * j3)
            * _binom(2 * j1, j1 - m1)
            * _binom(2 * j2, j2 - m2)
            * _binom(2 * j3, j3 - m3)
        )
    )
    z_min = max(array([0, j1 - m1 - J + 2 * j2, j2 + m2 - J + 2 * j1]))
    z_max = min(array([J - 2 * j3, j1 - m1, j2 + m2]))
    for z in range(z_min, z_max + 1):
        cg_coeff += (
            (-1) ** z
            * _binom(J - 2 * j3, z)
            * _binom(J - 2 * j2, j1 - m1 - z)
            * _binom(J - 2 * j1, j2 + m2 - z)
        )

    return cg_coeff * C


@jit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nopython=True,
    cache=True,
    nogil=True,
)
def _Wigner_3j(j1, j2, j3, m1, m2, m3):
    return (
        (-1) ** (j1 - j2 - m3)
        / sqrt(2 * j3 + 1)
        * Clebsh_Gordan(j1, m1, j2, m2, j3, -m3)
    )


def _finite_diff_stencil(diff_order: int, num_of_points: int, step: float64):
    stencil_len = 2 * num_of_points + 1

    if diff_order >= stencil_len:
        raise SltInputError(
            ValueError(
                f"Insufficient number of points to evaluate coefficients."
                f" Provide number of points greater than (derivative order -"
                f" 1) / 2."
            )
        )

    stencil_matrix = tile(
        arange(-num_of_points, num_of_points + 1).astype(int64),
        (stencil_len, 1),
    )
    stencil_matrix = stencil_matrix ** arange(0, stencil_len).reshape(-1, 1)

    order_vector = zeros(stencil_len)
    order_vector[diff_order] = factorial(diff_order) / power(step, diff_order)

    stencil_coeff = inv(stencil_matrix) @ order_vector.T

    return stencil_coeff


@jit(
    "complex128[:,:](complex128[:,:], complex128[:,:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _hermitian_x_in_basis_of_hermitian_y(x_matrix, y_matrix):
    x_matrix = ascontiguousarray(x_matrix)
    _, eigenvectors = eigh(y_matrix)

    return eigenvectors.conj().T @ x_matrix @ eigenvectors


@jit(
    "float64[:,:](complex128[:,:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _decomposition_of_hermitian_matrix(matrix):
    _, eigenvectors = eigh(matrix)

    return (eigenvectors * eigenvectors.conj()).real.T * 100


def _normalize_grid_vectors(grid: ndarray):

    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError(
                "A custom grid has to be a (n,4) array in the format:"
                " [[direction_x, direction_y, direction_z, weight],...]."
        )

    norm = 0

    for vector_index in range(grid.shape[0]):
        length = sqrt(
            grid[vector_index][0] ** 2
            + grid[vector_index][1] ** 2
            + grid[vector_index][2] ** 2
        )
        norm += grid[vector_index][3]
        if length == 0:
            raise ValueError("Vector of length zero detected in the input grid.")

        grid[vector_index][:3] = grid[vector_index][:3] / length

    if norm != 0:
        grid[:, 3] = grid[:, 3] / norm

    return grid


def _normalize_orientations(orientations: ndarray):

    if orientations.ndim != 2 or orientations.shape[1] != 3:
        raise ValueError("Orientations has to be (n,3) array in the format: [[direction_x, direction_y, direction_z],...].")
            
    for vector_index in range(orientations.shape[0]):
        length = sqrt(
            orientations[vector_index][0] ** 2
            + orientations[vector_index][1] ** 2
            + orientations[vector_index][2] ** 2
        )
        if length == 0:
            raise ValueError("Vector of length zero detected in the input orientations.")
        
        orientations[vector_index] = orientations[vector_index] / length

    return orientations


def _normalize_orientation(orientation):
    try:
        orientation = array(orientation, dtype=settings.float)
    except Exception as exc:
        raise SltInputError(exc) from None

    if orientation.ndim != 1 or orientation.shape[0] != 3:
        raise SltInputError(
            ValueError(
                "Orientation has to be a 1D array in the format:"
                " [direction_x, direction_y, direction_z]."
            )
        )

    length = sqrt(
        orientation[0] ** 2 + orientation[1] ** 2 + orientation[2] ** 2
    )
    if length == 0:
        raise SltInputError(
            ValueError(
                "Vector of length zero detected in the input orientation."
            )
        )
    orientation = orientation / length

    return orientation


@jit(
    [f"complex64[:](complex64[:], complex64[:])", f"complex64[:,:](complex64[:,:], complex64[:,:])", f"complex64[:,:,:](complex64[:,:,:], complex64[:,:,:])",
     f"complex128[:](complex128[:], complex128[:])", f"complex128[:,:](complex128[:,:], complex128[:,:])", f"complex128[:,:,:](complex128[:,:,:], complex128[:,:,:])"],
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
    parallel=True,
)
def _magnetic_momenta_from_spin_angular_momenta(spin: ndarray, angular_momenta: ndarray):
    mu_b = array(MU_B, dtype=spin.dtype)
    ge = array(GE, dtype=spin.dtype)
    return -mu_b*(ge * spin + angular_momenta)


@jit(
    [f"complex64[:](complex64[:], complex64[:])", f"complex64[:,:](complex64[:,:], complex64[:,:])", f"complex64[:,:,:](complex64[:,:,:], complex64[:,:,:])",
     f"complex128[:](complex128[:], complex128[:])", f"complex128[:,:](complex128[:,:], complex128[:,:])", f"complex128[:,:,:](complex128[:,:,:], complex128[:,:,:])"],
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
    parallel=True,
)
def _total_angular_momenta_from_spin_angular_momenta(spin: ndarray, angular_momenta: ndarray):
    return spin + angular_momenta
