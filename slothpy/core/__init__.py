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
    hamiltonian_from_molcas,
    hamiltonian_from_orca,
    slt_file,
    xyz,
    unit_cell,
    supercell,
)

from .slt_file_object import SltFile

from ._config import (
    settings,
    set_sysexit_on_sigint,
    set_default_on_sigint,
    turn_on_monitor,
    turn_off_monitor,
    set_default_error_reporting_mode,
    set_plain_error_reporting_mode,
    set_double_precision,
    set_single_precision,
    set_log_level,
    set_print_level,
    set_number_threads,
    set_number_cpu,
)

__all__ = [
    "slt_file",
    "hamiltonian_from_molcas",
    "hamiltonian_from_orca",
    "xyz",
    "unit_cell",
    "supercell",
    "SltFile",
    "settings",
    "set_sysexit_on_sigint",
    "set_default_on_sigint",
    "turn_on_monitor",
    "turn_off_monitor",
    "set_default_error_reporting_mode",
    "set_plain_error_reporting_mode",
    "set_double_precision",
    "set_single_precision",
    "set_log_level",
    "set_print_level",
    "set_num_threads",
    "set_number_cpu",
]
