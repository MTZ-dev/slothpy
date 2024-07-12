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

from numpy import ndarray, empty_like
from numba import jit, types, complex128, complex64, float64, float32, int64
from scipy.spatial.transform import Rotation


@jit([
        types.Array(complex64, 3, 'C')(types.Array(complex64, 3, 'C', True), types.Array(float32, 2, 'C', True)),
        types.Array(complex128, 3, 'C')(types.Array(complex128, 3, 'C', True), types.Array(float64, 2, 'C', True)),
        types.Array(float64, 2, 'C')(types.Array(float64, 2, 'C', True), types.Array(float64, 2, 'C', True)),
        types.Array(float32, 2, 'C')(types.Array(float32, 2, 'C', True), types.Array(float32, 2, 'C', True)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _rotate_vector_operator(vect_oper: ndarray, rotation: ndarray):
    rotated_operator = empty_like(vect_oper)
    rotated_operator[0] = rotation[0, 0] * vect_oper[0] + rotation[0, 1] * vect_oper[1] + rotation[0, 2] * vect_oper[2]
    rotated_operator[1] = rotation[1, 0] * vect_oper[0] + rotation[1, 1] * vect_oper[1] + rotation[1, 2] * vect_oper[2]
    rotated_operator[2] = rotation[2, 0] * vect_oper[0] + rotation[2, 1] * vect_oper[1] + rotation[2, 2] * vect_oper[2]

    return rotated_operator


@jit([
        types.Array(complex64, 2, 'C')(types.Array(complex64, 3, 'C', True),types.Array(float32, 2, 'C', True), int64),
        types.Array(complex128, 2, 'C')(types.Array(complex128, 3, 'C', True), types.Array(float64, 2, 'C', True), int64),
        types.Array(float64, 1, 'C')(types.Array(float64, 2, 'C', True), types.Array(float64, 2, 'C', True), int64),
        types.Array(float32, 1, 'C')(types.Array(float32, 2, 'C', True), types.Array(float32, 2, 'C', True), int64),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _rotate_vector_operator_component(vect_oper: ndarray, rotation: ndarray, xyz: int):

    return rotation[xyz, 0] * vect_oper[0] + rotation[xyz, 1] * vect_oper[1] + rotation[xyz, 2] * vect_oper[2]


@jit([
        types.Array(complex64, 2, 'C')(types.Array(complex64, 3, 'C', True), types.Array(float32, 2, 'C', True), types.Array(float32, 1, 'C', True)),
        types.Array(complex128, 2, 'C')(types.Array(complex128, 3, 'C', True), types.Array(float64, 2, 'C', True), types.Array(float64, 1, 'C', True)),
        types.Array(float64, 1, 'C')(types.Array(float64, 2, 'C', True), types.Array(float64, 2, 'C', True), types.Array(float64, 1, 'C', True)),
        types.Array(float32, 1, 'C')(types.Array(float32, 2, 'C', True), types.Array(float32, 2, 'C', True), types.Array(float32, 1, 'C', True)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _rotate_vector_operator_orintation(vect_oper: ndarray, rotation: ndarray, orientation: ndarray):
    rotation_oriented = empty_like(rotation)
    rotated_operator = empty_like(vect_oper[0])
    rotation_oriented[0, :] = rotation[0, :] * orientation[0]
    rotation_oriented[1, :] = rotation[1, :] * orientation[1]
    rotation_oriented[2, :] = rotation[2, :] * orientation[2]
    rotated_operator = rotation_oriented[0, 0] * vect_oper[0] + rotation_oriented[0, 1] * vect_oper[1] + rotation_oriented[0, 2] * vect_oper[2]
    rotated_operator += rotation_oriented[1, 0] * vect_oper[0] + rotation_oriented[1, 1] * vect_oper[1] + rotation_oriented[1, 2] * vect_oper[2]
    rotated_operator += rotation_oriented[2, 0] * vect_oper[0] + rotation_oriented[2, 1] * vect_oper[1] + rotation_oriented[2, 2] * vect_oper[2]

    return rotated_operator


class SltRotation(Rotation):
    """
    This is a wrapper for the scipy.spatial.transform.Rotation class, you can
    use it or the scipy's Rotation directly, but SlothPy only supports single
    rotation in a single instance. For various methods of creating and using
    the instances, please refer to the scipy's docs at:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation.

    See also
    ----------
    scipy.spatial.transform.Rotation
    """
    pass