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
from typing import Literal

from numpy import ndarray

from slothpy._angular_momentum._rotation import _rotate_vector_operator, _rotate_vector_operator_component, _rotate_vector_operator_orintation
from slothpy._general_utilities._math_expresions import _3d_dot


def _return_components(slt_group, which: Literal["s", "l", "p", "j", "m"], xyz, start_state, stop_state):
    if isinstance(xyz, ndarray):
        return _3d_dot(getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state], xyz)
    match xyz:
        case "xyz":
            array = getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state]
        case "x":
            array = getattr(slt_group, f"{which}x")[start_state:stop_state, start_state:stop_state]
        case "y":
            array = getattr(slt_group, f"{which}y")[start_state:stop_state, start_state:stop_state]
        case "z":
            array = getattr(slt_group, f"{which}z")[start_state:stop_state, start_state:stop_state]    
    return array


def _rotate_and_return_components(slt_group, which: Literal["s", "l", "p", "j", "m"], xyz, start_state, stop_state, rotation): 
    if isinstance(xyz, ndarray):
        return _rotate_vector_operator_orintation(getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state], rotation, xyz)
    match xyz:
        case "xyz":
            array = _rotate_vector_operator(getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state], rotation)
        case "x":
            array =  _rotate_vector_operator_component(getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state], rotation, 0)
        case "y":
            array =  _rotate_vector_operator_component(getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state], rotation, 1)
        case "z":
            array =  _rotate_vector_operator_component(getattr(slt_group, f"{which}")[:, start_state:stop_state, start_state:stop_state], rotation, 2)
    return array


def _return_components_diag(slt_group, which: Literal["s", "l", "p", "j", "m"], xyz, start_state, stop_state):
    if isinstance(xyz, ndarray):
        return _3d_dot(getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state), xyz)
    match xyz:
        case "xyz":
            array = getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state)
        case "x":
            array = getattr(slt_group, f"{which}x")._get_diagonal(start_state, stop_state)
        case "y":
            array = getattr(slt_group, f"{which}y")._get_diagonal(start_state, stop_state)
        case "z":
            array = getattr(slt_group, f"{which}z")._get_diagonal(start_state, stop_state)  
    return array


def _rotate_and_return_components_diag(slt_group, which: Literal["s", "l", "p", "j", "m"], xyz, start_state, stop_state, rotation): 
    if isinstance(xyz, ndarray):
        return _rotate_vector_operator_orintation(getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state), rotation, xyz)
    match xyz:
        case "xyz":
            array = _rotate_vector_operator(getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state), rotation)
        case "x":
            array =  _rotate_vector_operator_component(getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state), rotation, 0)
        case "y":
            array =  _rotate_vector_operator_component(getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state), rotation, 1)
        case "z":
            array =  _rotate_vector_operator_component(getattr(slt_group, f"{which}")._get_diagonal(start_state, stop_state), rotation, 2)
    return array


def slpjm_components_driver(slt_group, kind: Literal["diagonal", "full"], which: Literal["s", "l", "p", "j", "m"], xyz='xyz', start_state=0, stop_state=0, rotation=None):
    if rotation is None:
        if kind == "diagonal":
            return _return_components_diag(slt_group, which, xyz, start_state, stop_state)
        if kind == "full":
            return _return_components(slt_group, which, xyz, start_state, stop_state)
    else:
        if kind == "diagonal":
            return _rotate_and_return_components_diag(slt_group, which, xyz, start_state, stop_state, rotation)
        if kind == "full":
            return _rotate_and_return_components(slt_group, which, xyz, start_state, stop_state, rotation)

