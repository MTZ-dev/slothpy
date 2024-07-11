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

from numpy import (
    arange,
    ascontiguousarray,
    linspace,
    meshgrid,
    zeros,
    pi,
    sqrt,
    cos,
    sin,
    float64,
    abs,
)
from numba import jit, types, float64, float32, int64


@jit([
        types.Array(float32, 1, 'C')(int64),
        types.Array(float64, 1, 'C')(int64),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _fibonacci_over_sphere(num_points):
    indices = arange(0, num_points, dtype=float64)
    phi = pi * (3.0 - sqrt(5.0))  # golden angle in radians
    xyz_trans = zeros((3, num_points))

    y = 1 - (indices / float64(num_points - 1)) * 2  # y goes from 1 to -1
    radius = sqrt(abs(1.0 - y * y))  # radius at y

    theta = phi * indices  # golden angle increment

    x = cos(theta) * radius
    z = sin(theta) * radius

    xyz_trans[0] = x
    xyz_trans[1] = y
    xyz_trans[2] = z

    return ascontiguousarray(xyz_trans.T)


def _meshgrid_over_sphere_flatten(grid_number):
    theta = linspace(0, 2 * pi, 2 * grid_number, dtype=float64)
    phi = linspace(0, pi, grid_number, dtype=float64)
    theta, phi = meshgrid(theta, phi)

    xyz_mesh = zeros((phi.shape[0], phi.shape[1], 3), dtype=float64)

    xyz_mesh[:, :, 0] = sin(phi) * cos(theta)
    xyz_mesh[:, :, 1] = sin(phi) * sin(theta)
    xyz_mesh[:, :, 2] = cos(phi)

    xyz = xyz_mesh.reshape((-1, 3))

    return ascontiguousarray(xyz)
