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
from slothpy.general_utilities._constants import GE
from slothpy.core._slothpy_exceptions import SltInputError


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
def Wigner_3j(j1, j2, j3, m1, m2, m3):
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


def hermitian_x_in_basis_of_hermitian_y(x_matrix, y_matrix):
    _, eigenvectors = eigh(y_matrix)

    return eigenvectors.conj().T @ x_matrix @ eigenvectors


def decomposition_of_hermitian_matrix(matrix):
    _, eigenvectors = eigh(matrix)

    return (eigenvectors * eigenvectors.conj()).real.T * 100


def _normalize_grid_vectors(grid):
    grid = array(grid, dtype=float64)

    if grid.ndim != 2 or grid.shape[1] != 4:
        raise SltInputError(
            ValueError(
                "Custom grid has to be (n,4) array in the format:"
                " [[direction_x, direction_y, direction_z, weight],...]."
            )
        )

    for vector_index in range(grid.shape[0]):
        length = sqrt(
            grid[vector_index][0] ** 2
            + grid[vector_index][1] ** 2
            + grid[vector_index][2] ** 2
        )
        if length == 0:
            raise SltInputError(
                ValueError("Vector of length zero detected in the input grid.")
            )
        grid[vector_index][:3] = grid[vector_index][:3] / length

    return grid


@jit(
    "complex128[:,:,:](complex128[:,:,:], int64, int64)",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _magnetic_momenta_from_angular_momenta(
    angular_momenta: ndarray[complex128], start: int = 0, stop: int = 0
):
    if stop == 0:
        stop = angular_momenta.shape[0]
    size = stop - start
    magnetic_momenta = zeros((3, size, size), dtype=complex128)

    # Compute and save magnetic momenta in a.u.
    magnetic_momenta[0] = -(
        GE * angular_momenta[0, start:stop, start:stop]
        + angular_momenta[3, start:stop, start:stop]
    )
    magnetic_momenta[1] = -(
        GE * angular_momenta[1, start:stop, start:stop]
        + angular_momenta[4, start:stop, start:stop]
    )
    magnetic_momenta[2] = -(
        GE * angular_momenta[2, start:stop, start:stop]
        + angular_momenta[5, start:stop, start:stop]
    )
    magnetic_momenta = ascontiguousarray(magnetic_momenta)

    return magnetic_momenta


@jit(
    "complex128[:,:,:](complex128[:,:,:], int64, int64)",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _total_angular_momenta_from_angular_momenta(
    angular_momenta: ndarray[complex128], start: int = 0, stop: int = 0
):
    if stop == 0:
        stop = angular_momenta.shape[0]
    size = stop - start
    total_angular_momenta = zeros((3, size, size), dtype=complex128)

    # Compute and save magnetic momenta in a.u.
    total_angular_momenta[0] = (
        angular_momenta[0, start:stop, start:stop]
        + angular_momenta[3, start:stop, start:stop]
    )
    total_angular_momenta[1] = (
        angular_momenta[1, start:stop, start:stop]
        + angular_momenta[4, start:stop, start:stop]
    )
    total_angular_momenta[2] = (
        angular_momenta[2, start:stop, start:stop]
        + angular_momenta[5, start:stop, start:stop]
    )

    return total_angular_momenta
