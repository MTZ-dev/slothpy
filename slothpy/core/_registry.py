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

from abc import ABCMeta
from functools import wraps

from slothpy._general_utilities._constants import BLUE, RESET
from slothpy.core._slothpy_exceptions import SltFileError, SltInputError
from slothpy.core._input_parser import validate_input

type_registry = {}

class MethodTypeMeta(ABCMeta):
    def __new__(metacls, name, bases, class_dict):
        cls = super().__new__(metacls, name, bases, class_dict)
        method_type = class_dict.get('_method_type')
        if method_type:
            if method_type not in type_registry:
                type_registry[method_type] = cls
        return cls


def delegate_method_to_slt_group(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._exists:
            raise SltFileError(self._hdf5, IOError(f"{BLUE}Group{RESET} '{self._group_name}' does not exist in the .slt file.")) from None
        obj = type_registry.get(self.type)
        if hasattr(obj, '_from_slt_file'):
            obj = obj._from_slt_file(self)
        else:
            obj = obj(self)
        delegated_method = getattr(obj, method.__name__, None)
        if delegated_method is None:
            raise SltInputError(AttributeError(f"'{obj._method_type}' object has no method '{method.__name__}'.")) from None
        return delegated_method(*args, **kwargs)
    
    return wrapper


class MethodDelegateMeta(ABCMeta):
    def __new__(cls, name, bases, class_dict):
        for attr_name, attr_value in class_dict.items():
            if isinstance(attr_value, property) and not attr_name in ["attributes", "attrs", "type"]:
                getter = delegate_method_to_slt_group(attr_value.fget) if attr_value.fget else None
                setter = delegate_method_to_slt_group(attr_value.fset) if attr_value.fset else None
                deleter = delegate_method_to_slt_group(attr_value.fdel) if attr_value.fdel else None
                class_dict[attr_name] = property(getter, setter, deleter)
            elif callable(attr_value) and not (attr_name.startswith("__") and attr_name.endswith("__")):
                class_dict[attr_name] = validate_input(delegate_method_to_slt_group(attr_value))

        return super().__new__(cls, name, bases, class_dict)