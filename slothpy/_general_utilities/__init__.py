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

from ._grids_over_hemisphere import lebedev_laikov_grid
from ._system import (
    set_default_error_reporting_mode,
    set_plain_error_reporting_mode,
)
from ._ploting_utilities import color_map

__all__ = [
    "lebedev_laikov_grid",
    "colour_map",
    "set_default_error_reporting_mode",
    "set_plain_error_reporting_mode",
]
