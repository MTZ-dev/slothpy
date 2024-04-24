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

import inspect 
from functools import wraps
from typing import Literal
from os import cpu_count
from numpy import array, allclose, identity
from slothpy.core._slothpy_exceptions import SltInputError, SltFileError, SltSaveError, SltReadError
from slothpy._general_utilities._grids_over_hemisphere import lebedev_laikov_grid
from slothpy._general_utilities._math_expresions import _normalize_grid_vectors, _normalize_orientations, _normalize_orientation
from slothpy._general_utilities._constants import GREEN, BLUE, RESET
from slothpy._general_utilities._io import _group_exists
from slothpy.core._config import settings

def validate_input(group_type: Literal["HAMILTONIAN"]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            self = bound_args.arguments["self"]

            if not self._exists:
                raise SltFileError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_path}' does not exist in the {GREEN}File{RESET}: '{self._hdf5}'.")

            try:
                self.attributes["Type"]
                self.attributes["States"]
            except SltFileError as exc:
                raise SltReadError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_path}' is not a valid SlothPy group.") from None

            if self.attributes["Type"] != group_type:
                raise SltReadError(self._hdf5, None, f"Wrong group type: '{self.attributes['Type']}' of {BLUE}Group{RESET}: '{self._group_path}' from the {GREEN}File{RESET}: '{self._hdf5}'. Expected '{group_type}' type.")

            if "slt_save" in bound_args.arguments.keys() and bound_args.arguments["slt_save"] is not None:
                if _group_exists(self._hdf5, bound_args.arguments["slt_save"]):
                    raise SltSaveError(
                        self._hdf5,
                        NameError(""),
                        message=f"Unable to save the results. {BLUE}Group{RESET} '{bound_args.arguments['slt_save']}' already exists in the {GREEN}File{RESET}: '{self._hdf5}'. Delete it manually.",
                    ) from None

            try:
                for name, value in bound_args.arguments.items():
                    match name:
                        case "number_cpu":
                            if value is None:
                                value = settings.number_cpu
                            if value == 0:
                                value = int(cpu_count())    
                            elif not (isinstance(value, int) and value > 0 and value <= int(cpu_count())):
                                raise ValueError(f"The number of CPUs has to be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
                        case "number_threads":
                            if value is None:
                                value = settings.number_threads
                            if value == 0:
                                value = int(cpu_count())
                            elif not (isinstance(value, int) and value > 0 and value <= int(cpu_count())):
                                raise ValueError(f"The number of CPUs has to be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
                        case "magnetic_fields":
                            value = array(value, copy=False, order='C', dtype=settings.float)
                            if value.ndim != 1:
                                raise ValueError("The list of fields has to be a 1D array.")
                        case "temperatures":
                            value = array(value, copy=False, order='C', dtype=settings.float)
                            if value.ndim != 1:
                                raise ValueError("The list of temperatures has to be a 1D array.")
                            if (value <= 0).any():
                                raise ValueError("Zero or negative temperatures were detected in the input.")
                        case "grid":
                            if isinstance(value, int):
                                value = lebedev_laikov_grid(value)
                            else:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.ndim != 2:
                                    raise ValueError("The grid array has to be a 2D array in the form [[direction_x, direction_y, direction_z, weight],...].")
                                if value.shape[1] == 3:
                                    value = _normalize_grid_vectors(value)
                                else:
                                    raise ValueError("The grid has to be set to an integer from 0-11, or a custom one has to be in the form [[direction_x, direction_y, direction_z, weight],...].")
                        case "orientations":
                            if isinstance(value, int):
                                value = lebedev_laikov_grid(value)
                            else:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.ndim != 2:
                                    raise ValueError("The array of orientations has to be a 2D array in the form: [[direction_x, direction_y, direction_z],...] or [[direction_x, direction_y, direction_z, weight],...] for powder-averaging (or integer from 0-11).")
                                if value.shape[1] == 4:
                                    value = _normalize_grid_vectors(value)
                                elif value.shape[1] == 3:
                                    value = _normalize_orientations(value)
                                else:
                                    raise ValueError("The orientations' array has to be (n,3) in the form: [[direction_x, direction_y, direction_z],...] or (n,4) array in the form: [[direction_x, direction_y, direction_z, weight],...] for powder-averaging (or integer from 0-11).")
                        case "states_cutoff":
                            if value == 0:
                                value = int(self.attributes["States"])
                            elif not isinstance(value, int) or value < 0:
                                raise ValueError(f"The states' cutoff has to be a nonnegative integer less than or equal to the overall number of available states: {self[bound_args.arguments['group_name']].attributes['States']} (or 0 for all the states).")
                            elif value > self.attributes["States"]:
                                raise ValueError(f"Set the states' cutoff to a nonnegative integer less than or equal to the overall number of available states: {self[bound_args.arguments['group_name']].attributes['States']} (or 0 for all the states).")
                        case "number_of_states":
                            if not isinstance(value, int) or value < 0:
                                raise ValueError("The number of states has to be a positive integer.")
                            max_states = int(self.attributes["States"])
                            if isinstance(bound_args.arguments["states_cutoff"], int) and (bound_args.arguments["states_cutoff"] > 0) and bound_args.arguments["states_cutoff"] < self.attributes["States"]:
                                if value == 0:
                                    value = bound_args.arguments["states_cutoff"]
                                max_states = bound_args.arguments["states_cutoff"]
                            elif bound_args.arguments["states_cutoff"] == 0 and value == 0:
                                value = max_states
                            if value > max_states:
                                raise ValueError("The number of states has to be less or equal to the states' cutoff or overall number of states.")
                        case "start_state":
                            if not (isinstance(value, int) and value >= 0 and value < self.attributes["States"]):
                                raise ValueError(f"The first (start) state's number has to be a nonnegative integer less than or equal to the overall number of states - 1: {self.attributes['States'] - 1}.")
                        case "stop_state":
                            if value == 0:
                                value = int(self.attributes["States"])
                            if not isinstance(value, int) or value < 0 or value > self.attributes["States"]:
                                raise ValueError(f"The last (stop) state's number has to be a nonnegative integer less than or equal to the overall number of states: {self.attributes['States']}.")
                            if "start_state" in bound_args.arguments.keys():
                                if isinstance(bound_args.arguments["start_state"], int) and (bound_args.arguments["start_state"] >= 0) and bound_args.arguments["start_state"] <= self.attributes["States"] and value < bound_args.arguments["start_state"]:
                                    raise ValueError(f"The last (stop) state's number has to be equal or greater than the first (start) state's number: {bound_args.arguments['start_state']}.")
                        case "xyz":
                            if value not in ["xyz", "x", "y", "z"]:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.ndim != 1 or value.size != 3:
                                    raise ValueError(f"The xyz argument has to be one of 'xyz', 'x', 'y', 'z' or it can be an orientation in the form [x,y,z].")
                                value = _normalize_orientation(value)
                        case "rotation":
                            if value is not None:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.shape != (3, 3):
                                    raise ValueError("The rotation matrix must be a 3x3 array.")
                                product = value.T @ value
                                if not allclose(product, identity(3), atol=1e-2, rtol=0):
                                    raise ValueError("Input rotation matrix must be orthogonal.")
                    bound_args.arguments[name] = value
                    
            except Exception as exc:
                raise SltInputError(exc) from None

            return func(**bound_args.arguments)
        
        return wrapper
    return decorator



