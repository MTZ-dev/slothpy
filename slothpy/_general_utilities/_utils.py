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

from numpy import ndarray, diagonal
from slothpy.core._config import settings
from slothpy._angular_momentum._rotation import _rotate_vector_operator, _rotate_vector_operator_component
from slothpy._general_utilities._math_expresions import _3d_dot

def _rotate_and_return_components(xyz, array, rotation): 
    if isinstance(xyz, ndarray):
        return _3d_dot(_rotate_vector_operator(array, rotation), xyz)
    match xyz:
        case "xyz":
            array = _rotate_vector_operator(array, rotation)
        case "x":
            array =  _rotate_vector_operator_component(array, rotation, 0)
        case "y":
            array =  _rotate_vector_operator_component(array, rotation, 1)
        case "z":
            array =  _rotate_vector_operator_component(array, rotation, 2)
    return array


def _return_components(array_xyz, array_x, array_y, array_z, xyz, start_state, stop_state):
    if isinstance(xyz, ndarray):
        return _3d_dot(array_xyz[:, start_state:stop_state, start_state:stop_state], xyz)
    match xyz:
        case "xyz":
            array = array_xyz[:, start_state:stop_state, start_state:stop_state]
        case "x":
            array = array_x[start_state:stop_state, start_state:stop_state]
        case "y":
            array = array_y[start_state:stop_state, start_state:stop_state]
        case "z":
            array = array_z[start_state:stop_state, start_state:stop_state]    
    return array


def _return_components_diag(array_xyz, array_x, array_y, array_z, xyz, start_state, stop_state):
    if isinstance(xyz, ndarray):
        return _3d_dot(diagonal(array_xyz[:, start_state:stop_state, start_state:stop_state].real, axis1=1, axis2=2).astype(settings.float, order="C"), xyz)
    match xyz:
        case "xyz":
            array = diagonal(array_xyz[:, start_state:stop_state, start_state:stop_state].real, axis1=1, axis2=2).astype(settings.float, order="C")
        case "x":
            array = diagonal(array_x[start_state:stop_state, start_state:stop_state].real).astype(settings.float, order="C")
        case "y":
            array = diagonal(array_y[start_state:stop_state, start_state:stop_state].real).astype(settings.float, order="C")
        case "z":
            array = diagonal(array_z[start_state:stop_state, start_state:stop_state].real).astype(settings.float, order="C")      
    return array