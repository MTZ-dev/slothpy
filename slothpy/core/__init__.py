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

from .creation_functions import (
    compound_from_molcas,
    compound_from_orca,
    compound_from_slt,
)

from .compound_object import Compound

from ._config import (
    settings,
    turn_on_monitor,
    turn_off_monitor,
    set_default_error_reporting_mode,
    set_plain_error_reporting_mode,
    set_double_precision,
    set_single_precision,
    set_log_level,
)

__all__ = [
    "compound_from_slt",
    "compound_from_molcas",
    "compound_from_orca",
    "Compound",
    "settings",
    "turn_on_monitor",
    "turn_off_monitor",
    "set_default_error_reporting_mode",
    "set_plain_error_reporting_mode",
    "set_double_precision",
    "set_single_precision",
    "set_log_level",
]
