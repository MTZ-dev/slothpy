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

"""Module for stroing conastants used by other Sloth modules."""

# ANSI escape codes for text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
RESET = "\033[0m"

# Physical constants
GE = 2.00231930436256  # Electron g-factor
KB = 3.166811563e-6  # Boltzmann constant a.u. / K
MU_B_AU_T = 2.127191078656686e-6  # Bohr magneton in a.u. / T
MU_B_AU = 0.5 # Bohr magneton in a.u.
H_CM_1 = 219474.6  # Atomic units (Hartree) to wavenumbers
MU_B_CM_3 = 0.5584938904  # Conversion factor from Bohr magneton to cm3 for chi
U_PI_T_A_AU = 435974.82 # Vacuum magnetic permeability / 4pi in T^2 * A^3 / a.u.
E_PI_A_AU = 0.01179216466 # 1 / Vacuum electric permittivity / 4pi in a.u. * A^3
B_AU_T = 2.35051757077e5 # Magnetic field in a.u. to T
F_AU_VM = 5.14220675112e11 # Electric field in a.u. to V / m
