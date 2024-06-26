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
from numpy import array, allclose, identity, log
from slothpy.core._slothpy_exceptions import SltInputError, SltFileError, SltSaveError, SltReadError, slothpy_exc
from slothpy._general_utilities._grids_over_hemisphere import lebedev_laikov_grid
from slothpy._general_utilities._math_expresions import _normalize_grid_vectors, _normalize_orientations, _normalize_orientation
from slothpy._general_utilities._constants import GREEN, BLUE, RESET, KB
from slothpy._general_utilities._io import _group_exists
from slothpy.core._config import settings

def validate_input(group_type: Literal["HAMILTONIAN"], direct_acces: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            self = bound_args.arguments["self"]

            if not self._exists:
                raise SltFileError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_name}' does not exist in the {GREEN}File{RESET}: '{self._hdf5}'.")

            try:
                self.attributes["Type"]
                self.attributes["States"]
            except SltFileError as exc:
                raise SltReadError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_name}' is not a valid SlothPy group.") from None

            if self.attributes["Type"] != group_type:
                raise SltReadError(self._hdf5, None, f"Wrong group type: '{self.attributes['Type']}' of {BLUE}Group{RESET}: '{self._group_name}' from the {GREEN}File{RESET}: '{self._hdf5}'. Expected '{group_type}' type.")

            if self.attributes["Type"] == "HAMILTONIAN" and direct_acces:
                if self.attributes["Kind"] == "SLOTHPY":
                    raise SltFileError(self._hdf5, None, "Custom SlothPy Hamiltonians do not support direct access to their properties and they cannot be used to construct other SlothPy Hamiltonians. For all the supported methods, use them as input in place of the slt_group argument.")

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
                                raise ValueError(f"The number of CPUs must be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
                        case "number_threads":
                            if value is None:
                                value = settings.number_threads
                            if value == 0:
                                value = int(cpu_count())
                            elif not (isinstance(value, int) and value > 0 and value <= int(cpu_count())):
                                raise ValueError(f"The number of CPUs must be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
                        case "magnetic_fields":
                            value = array(value, copy=False, order='C', dtype=settings.float)
                            if value.ndim != 1:
                                raise ValueError("The list of fields must be a 1D array.")
                        case "temperatures":
                            value = array(value, copy=False, order='C', dtype=settings.float)
                            if value.ndim != 1:
                                raise ValueError("The list of temperatures must be a 1D array.")
                            if (value <= 0).any():
                                raise ValueError("Zero or negative temperatures were detected in the input.")
                        case "grid":
                            if isinstance(value, int):
                                value = lebedev_laikov_grid(value)
                            else:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.ndim != 2:
                                    raise ValueError("The grid array must be a 2D array in the form [[direction_x, direction_y, direction_z, weight],...].")
                                if value.shape[1] == 3:
                                    value = _normalize_grid_vectors(value)
                                else:
                                    raise ValueError("The grid must be set to an integer from 0-11, or a custom one must be in the form [[direction_x, direction_y, direction_z, weight],...].")
                        case "orientations":
                            if isinstance(value, int):
                                value = lebedev_laikov_grid(value)
                            else:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.ndim != 2:
                                    raise ValueError("The array of orientations must be a 2D array in the form: [[direction_x, direction_y, direction_z],...] or [[direction_x, direction_y, direction_z, weight],...] for powder-averaging (or integer from 0-11).")
                                if value.shape[1] == 4:
                                    value = _normalize_grid_vectors(value)
                                elif value.shape[1] == 3:
                                    value = _normalize_orientations(value)
                                else:
                                    raise ValueError("The orientations' array must be (n,3) in the form: [[direction_x, direction_y, direction_z],...] or (n,4) array in the form: [[direction_x, direction_y, direction_z, weight],...] for powder-averaging (or integer from 0-11).")
                        case "states_cutoff":
                            if not isinstance(value, list) or len(value) != 2:
                                raise ValueError("The states' cutoff must be a Python's list of length 2.")
                            if value[0] == 0:
                                value[0] = int(self.attributes["States"])
                            elif not isinstance(value[0], int) or value[0] < 0:
                                raise ValueError(f"The states' cutoff must be a nonnegative integer less than or equal to the overall number of available states: {self[bound_args.arguments['group_name']].attributes['States']} (or 0 for all the states).")
                            elif value[0] > self.attributes["States"]:
                                raise ValueError(f"Set the states' cutoff to a nonnegative integer less than or equal to the overall number of available states: {self[bound_args.arguments['group_name']].attributes['States']} (or 0 for all the states).")
                            if value[1] == 0:
                                value[1] = value[0]
                            if value[1] == "auto":
                                if "number_of_states" in bound_args.arguments.keys() and bound_args.arguments["number_of_states"] == 0:
                                    value[1] = value[0]
                                elif "number_of_states" in bound_args.arguments.keys() and isinstance(bound_args.arguments["number_of_states"], int) and bound_args.arguments["number_of_states"] <= value[0]:
                                    value[1] = bound_args.arguments["number_of_states"]
                                if "temperatures" in bound_args.arguments.keys():
                                    value[1] = settings.float(max(bound_args.arguments["temperatures"]) * KB * log(1e-16 if settings.precision == "double" else 1e-8))
                            elif not isinstance(value[1], int) or value[1] < 0 or value[1] > value[0]:
                                raise ValueError("Set the second entry of states' cutoff to a nonnegative integer less or equal to the first entry or 0 for all the states from the first entry or 'auto' to let the SlothPy decide on a suitable cutoff.")
                        case "number_of_states":
                            if not isinstance(value, int) or value < 0:
                                raise ValueError("The number of states must be a positive integer or 0 for all of the calculated states.")
                            if not isinstance(bound_args.arguments["states_cutoff"], list) or len(bound_args.arguments["states_cutoff"]) != 2:
                                raise ValueError("The states' cutoff must be a Python's list of length 2.")
                            max_states = int(self.attributes["States"]) if bound_args.arguments["states_cutoff"][0] == 0 else bound_args.arguments["states_cutoff"][0]
                            if isinstance(bound_args.arguments["states_cutoff"][1], int) and (bound_args.arguments["states_cutoff"][1] > 0) and (bound_args.arguments["states_cutoff"][1] <= bound_args.arguments["states_cutoff"][0]) if isinstance(bound_args.arguments["states_cutoff"][0], int) else False:
                                if value == 0:
                                    value = bound_args.arguments["states_cutoff"][1]
                                max_states = bound_args.arguments["states_cutoff"][1]
                            elif bound_args.arguments["states_cutoff"][1] in [0, "auto"] and value == 0:
                                value = max_states
                            if value > max_states:
                                raise ValueError("The number of states must be less or equal to the states' cutoff or overall number of states.")
                        case "start_state":
                            if not (isinstance(value, int) and value >= 0 and value < self.attributes["States"]):
                                raise ValueError(f"The first (start) state's number must be a nonnegative integer less than or equal to the overall number of states - 1: {self.attributes['States'] - 1}.")
                        case "stop_state":
                            if value == 0:
                                value = int(self.attributes["States"])
                            if not isinstance(value, int) or value < 0 or value > self.attributes["States"]:
                                raise ValueError(f"The last (stop) state's number must be a nonnegative integer less than or equal to the overall number of states: {self.attributes['States']}.")
                            if "start_state" in bound_args.arguments.keys():
                                if isinstance(bound_args.arguments["start_state"], int) and (bound_args.arguments["start_state"] >= 0) and bound_args.arguments["start_state"] <= self.attributes["States"] and value < bound_args.arguments["start_state"]:
                                    raise ValueError(f"The last (stop) state's number must be equal or greater than the first (start) state's number: {bound_args.arguments['start_state']}.")
                        case "xyz":
                            if value not in ["xyz", "x", "y", "z"]:
                                value = array(value, copy=False, order='C', dtype=settings.float)
                                if value.ndim != 1 or value.size != 3:
                                    raise ValueError(f"The xyz argument must be one of 'xyz', 'x', 'y', 'z' or it can be an orientation in the form [x,y,z].")
                                value = _normalize_orientation(value)
                        case "rotation": # TODO: Tutaj przerobić do nowej klasy i zwracac to as array!
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

@slothpy_exc("SltInputError")
def _parse_hamiltonian_dicts(compound, magnetic_centers: dict, exchange_interactions: dict):

    if not isinstance(magnetic_centers, dict) or not isinstance(exchange_interactions, dict):
        raise ValueError("Magnetic centers and exchange interactions parameters must be dictionaries.")
    
    if not all(isinstance(key, int) for key in magnetic_centers.keys()):
        raise ValueError("Magnetic centers in the dictionary must be enumerated by integers.")
    
    n = len(magnetic_centers)
    expected_keys = set(range(n))
    actual_keys = set(magnetic_centers.keys())

    if expected_keys != actual_keys:
        raise ValueError("Magnetic centers in the dictionary must be enumerated by integers from 0 to the number of centers - 1 without repetitions.")
    
    states = 1
    exchange_states = 1

    for value in magnetic_centers.values():
        if not isinstance(value, list):
            raise ValueError("The values of the magnetic centers dictionary must be Python's lists.")
        if compound[value[0]].attributes["Type"] != "HAMILTONIAN" or compound[value[0]].attributes["Kind"] == "SLOTHPY":
            raise ValueError(f"Group {value[0]} either does not exist or has a wrong type: expected HAMILTONIAN (it cannot be a custom SlothPy Hamiltonian).")
        if (len(value[1]) != 3 or not all(isinstance(x, int) for x in value[1])):
            raise ValueError("States cutoff must be an interable of length 3 [local_cutoff, mixing_cutoff, exchange_cutoff] with integer values.")
        if value[1][0] < value[1][1] or value[1][1] < value[1][2]:
            raise ValueError("The cutoff parameters must satisfy local_cutoff >= mixing_cutoff >= exchange_cutoff.")
        states *= value[1][1]
        exchange_states *= value[1][2]
        if value[3] != None and (len(value[3]) != 3 or not all(isinstance(x, (int, float)) for x in value[3])):
            raise ValueError("Coordinates must be None or iterable of length 3 with numerical values in Angstrom [x,y,z].")
        value[3] = array(value[3], copy=False, order='C', dtype=settings.float)
       #TODO: value[4] hyperfine add checks when implemented

    if states >= 10000 or (states >= 9000 and exchange_states >= 100):
        print(f"You created a custom Hamiltonian with {states}x{states} exchange space, and computations will require to find {exchange_states} eigenvalues/eigenvectors, which is considered very expensive. You must know what you are doing. All computations will be very lengthy, if possible at all.")

    for key, value in exchange_interactions.items():
        if len(key) != 2 or not all(isinstance(x, int) and 0 <= x < n for x in key):
            raise ValueError("Two-center exchange interactions must be enumerated with iterables of length 2 [centerA_number, centerB_number] containing integers from 0 to the number of centers - 1.")
        value = array(value, copy=False, order='C', dtype=settings.float)
        if value.ndim != 2 or value.shape != (3,3):
            raise ValueError("The J exchange interaction parameters must be real arrays with shape(3,3).")
        exchange_interactions[key] = value

    return states


      





