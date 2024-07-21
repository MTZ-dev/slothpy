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

from pkg_resources import resource_filename

from numpy import ndarray, arange, ascontiguousarray, linspace, meshgrid, zeros, abs, pi, sqrt, cos, sin, float64, int64
from numba import jit
from h5py import File

# TODO Reconsider making those public as lebedev-laikov
@jit(
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _fibonacci_over_hemisphere(num_points, precision):
    if precision == "double":
        indices = arange(0, num_points, dtype=float64)
    else:
        indices = arange(0, num_points, dtype=float32)
    phi = pi * (3.0 - sqrt(5.0))  # golden angle in radians
    xyz_trans = zeros((3, num_points))

    if precision == "double":
        y = 1 - (indices / float64(num_points - 1))  # y goes from 1 to 0
    else:
        y = 1 - (indices / float32(num_points - 1))  # y goes from 1 to 0
    
    radius = sqrt(abs(1.0 - y * y))  # radius at y
    theta = phi * indices  # golden angle increment

    x = cos(theta) * radius
    z = sin(theta) * radius

    xyz_trans[0] = x
    xyz_trans[1] = y
    xyz_trans[2] = z

    return ascontiguousarray(xyz_trans.T)


def _meshgrid_over_hemisphere_flatten(grid_number, precision):
    if precision == "double":
        theta = linspace(0, pi, grid_number, dtype=float64)
        phi = linspace(0, pi, grid_number, dtype=float64)
    else:
        theta = linspace(0, pi, grid_number, dtype=float32)
        phi = linspace(0, pi, grid_number, dtype=float32)
    
    theta, phi = meshgrid(theta, phi)

    xyz_mesh = zeros((phi.shape[0], phi.shape[1], 3), dtype=phi.dtype)

    xyz_mesh[:, :, 0] = sin(phi) * cos(theta)
    xyz_mesh[:, :, 1] = sin(phi) * sin(theta)
    xyz_mesh[:, :, 2] = cos(phi)

    xyz = xyz_mesh.reshape((-1, 3))

    return ascontiguousarray(xyz)


def lebedev_laikov_grid_over_hemisphere(grid_number: int, precision) -> ndarray:
    """
    Returns Lebedev-Laikov-Grids [1] over a hemisphere.

    Parameters
    ----------
    grid_number : int
        An integer from 0 to 11 controling the number of points in the grid
        over the hemisphere.

        0 - 17 points,
        1 - 29 points,
        2 - 93 points,
        3 - 161 points,
        4 - 185 points,
        5 - 229 points,
        6 - 309 points,
        7 - 401 points,
        8 - 505 points,
        9 - 889 points,
        10 - 1381 points,
        11 - 2949 points

    Returns
    -------
    ndarray
        Grid array in the form [[direction_x, direction_y, direction_z,
        weight], ...].

    Raises
    ------
    ValueError
        If grid is not an integer from 0 to 11.

    Notes
    -----
    Grids 0-2 can be considered as insufficient in most cases (only should be
    used for experimental purposes), 3-5 minimal to standard, 6-8 very precise,
    but costly, 9-11 very dense - only for particular use-cases (extremely
    costly - especially 10-11).

    References
    ----------
    .. [1] V.I. Lebedev, and D.N. Laikov
           "A quadrature formula for the sphere of the 131st
           algebraic order of accuracy"
           Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    """

    if not isinstance(grid_number, (int, int64)) or grid_number < 0 or grid_number > 11:
        raise ValueError("Input grid number must be an integer in the range 0 to 11.")
    
    data_file = resource_filename("slothpy", "static/data")

    with File(data_file, 'r') as file:
        dataset = file['lebedev_laikov_hemisphere'][f'{grid_number}']
        if precision == 'double':
            return dataset.astype(float64)[:]
        else:
            return dataset.astype(float32)[:]

    