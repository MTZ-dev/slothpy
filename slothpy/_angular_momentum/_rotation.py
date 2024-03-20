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

from numpy import ndarray, zeros_like
from numba import jit

# TODO: Eventually incorporate it into class as classmethod probably
@jit(["complex64[:,:,:](complex64[:,:,:], float32[:,:])", "complex128[:,:,:](complex128[:,:,:], float64[:,:])"],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _rotate_vector_operator(vect_oper: ndarray, rotation: ndarray):

    rotated_operator = zeros_like(vect_oper)

    rotated_operator[0] = (
        rotation[0, 0] * vect_oper[0]
        + rotation[0, 1] * vect_oper[1]
        + rotation[0, 2] * vect_oper[2]
    )
    rotated_operator[1] = (
        rotation[1, 0] * vect_oper[0]
        + rotation[1, 1] * vect_oper[1]
        + rotation[1, 2] * vect_oper[2]
    )
    rotated_operator[2] = (
        rotation[2, 0] * vect_oper[0]
        + rotation[2, 1] * vect_oper[1]
        + rotation[2, 2] * vect_oper[2]
    )

    return rotated_operator


class Rotation:
    pass
