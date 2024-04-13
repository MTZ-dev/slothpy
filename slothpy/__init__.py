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

from multiprocessing import current_process, set_start_method

if current_process().name == 'MainProcess':
    set_start_method('spawn')

import os

os.environ['NUMBA_OPT'] = 'max'
os.environ['NUMBA_LOOP_VECTORIZE'] = '1'
os.environ['NUMBA_ENABLE_AVX'] = '1'

from multiprocessing import current_process

from .core import (
    compound_from_orca,
    compound_from_slt,
    compound_from_molcas,
    settings,
    turn_on_monitor,
    turn_off_monitor,
    set_plain_error_reporting_mode,
    set_default_error_reporting_mode,
    set_double_precision,
    set_single_precision,
    set_log_level,
)
from ._general_utilities import (
    lebedev_laikov_grid,
    color_map,
)
from .core import Compound
from . import exporting

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
    "lebedev_laikov_grid",
    "colour_map",
    "exporting",
]

if current_process().name == "MainProcess":
    from matplotlib import rcParams, use
    from matplotlib.style import use as mplstyle_use

    # Only used for greetings
    from ._general_utilities._constants import (
        GREEN,
        BLUE,
        YELLOW,
        RESET,
    )

    # Set environment for fast plotting of 3D models
    mplstyle_use("fast")
    rcParams["path.simplify"] = True
    rcParams["path.simplify_threshold"] = 1.0
    use("Qt5Agg")

    # Greeting message
    print(
        BLUE
        + "                  ____  _       _   _     "
        + YELLOW
        + "____\n"
        + BLUE
        + "                 / ___|| | ___ | |_| |__ "
        + YELLOW
        + "|  _ \ _   _\n"
        + BLUE
        + "                 \___ \| |/ _ \| __| '_ \\"
        + YELLOW
        + "| |_) | | | |\n"
        + BLUE
        + "                  ___) | | (_) | |_| | | "
        + YELLOW
        + "|  __/| |_| |\n"
        + BLUE
        + "                 |____/|_|\___/ \__|_| |_"
        + YELLOW
        + "|_|    \__, |\n"
        + "                                                |___/"
        + GREEN
        + "  by MTZ \n"
        + RESET
    )
    print(
        "SlothPy Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ).\nThis"
        " program comes with ABSOLUTELY NO WARRANTY.\nThis is free software,"
        " and you are welcome to redistribute it.\nThe default is chosen to"
        " omit the tracebacks completely. To change it use"
        " slt.set_default_error_reporting_mode() method for the printing of"
        " tracebacks.\nTurn on the SlothPy Monitor utility using"
        " slt.turn_on_monitor()."
    )

if settings.traceback == False:
    set_plain_error_reporting_mode()
