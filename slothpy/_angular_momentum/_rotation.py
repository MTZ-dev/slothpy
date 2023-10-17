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

from numpy import ndarray, allclose, identity, zeros_like


# TODO: Eventually incorporate it into class as classmethod probably
def _rotate_vector_operator(vect_oper: ndarray, rotation: ndarray):
    # rotation = Rotation.matrix (from class)

    if rotation.shape != (3, 3):
        raise ValueError("Input rotation matrix must be a 3x3 matrix.")

    product = rotation.T @ rotation

    if not allclose(product, identity(3), atol=1e-2, rtol=0):
        raise ValueError("Input rotation matrix must be orthogonal.")

    rotated_oper = zeros_like(vect_oper)

    rotated_oper[0] = (
        rotation[0, 0] * vect_oper[0]
        + rotation[0, 1] * vect_oper[1]
        + rotation[0, 2] * vect_oper[2]
    )
    rotated_oper[1] = (
        rotation[1, 0] * vect_oper[0]
        + rotation[1, 1] * vect_oper[1]
        + rotation[1, 2] * vect_oper[2]
    )
    rotated_oper[2] = (
        rotation[2, 0] * vect_oper[0]
        + rotation[2, 1] * vect_oper[1]
        + rotation[2, 2] * vect_oper[2]
    )

    return rotated_oper


class Rotation:
    pass
