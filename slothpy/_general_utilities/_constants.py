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
KB = 3.166811563e-6  # Boltzmann constant a.u./K
MU_B = 2.127191078656686e-06  # Bohr magneton in a.u./T
H_CM_1 = 219474.6  # Atomic units (Hartree) to wavenumbers
MU_B_CM_3 = 0.5584938904  # Conversion factor from Bohr magneton to cm3 for chi
MU_T = 1.97276072296918e-06 # Conversion factor magnetic momenta in a.u. to field in T produced by the magnetic dipole (MU_B_T * MU_B ** 2)
MU_B_T = 435974.8198 # Conversion factor from Bohr magneton in a.u./T to field in T produced by the magnetic dipole
F_AU_VM = 5.14220675112e11 # Electric field in a.u. to V/m