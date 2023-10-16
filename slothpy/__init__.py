from .core import compound_from_orca, compound_from_slt, compound_from_molcas
from ._general_utilities import (
    lebedev_laikov_grid,
    colour_map,
    set_default_error_reporting_mode,
    set_plain_error_reporting_mode,
)
from .core import Compound
from . import exporting

__all__ = [
    "compound_from_slt",
    "compound_from_molcas",
    "compound_from_orca",
    "Compound",
    "lebedev_laikov_grid",
    "colour_map",
    "set_default_error_reporting_mode",
    "set_plain_error_reporting_mode",
    "exporting",
]


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
    + "  by MTZ"
    + RESET
)
print(
    "The default is chosen to omit the tracebacks completely. To change it"
    " use slt.set_default_error_reporting_mode method for the printing of"
    " tracebacks."
)
