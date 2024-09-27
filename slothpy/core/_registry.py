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

type_registry = {}

class MethodTypeMeta(ABCMeta):
    def __new__(metacls, name, bases, class_dict):
        cls = super().__new__(metacls, name, bases, class_dict)
        method_type = class_dict.get('_method_type')
        if method_type:
            if method_type not in type_registry:
                type_registry[method_type] = cls
        return cls