# SlothPy
# Copyright (C) 2025 Mikolaj Tadeusz Zychowicz (MTZ)

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

import numpy as np
from h5py import File
from numba import njit

from slothpy._general_utilities._constants import MU_B_AU, B_AU_T, MU_B_CM_3, KB
from input_models import AppConfig
from utils import dofs_with_complete_displacements

def get_hamiltonian_magnetic_momenta_dof_array(cfg: AppConfig):
    with File(cfg.relacs.slt_filepath) as f:
        group = f[cfg.spin_phonon.group_name]
        dof_array = dofs_with_complete_displacements(group,
                                            cfg.spin_phonon.displacement_number)
        magnetic_momenta = group["0/MAGNETIC_DIPOLE_MOMENTA"][:]
        hamiltonian = group["0/HAMILTONIAN_MATRIX"][:]
        return hamiltonian, magnetic_momenta, dof_array

@njit(nogil=True, cache=True, fastmath=True)
def get_chi_T(magnetic_momenta_au: np.ndarray, energies_au: np.ndarray,
              temperatures: np.ndarray, field: np.float64):
    t_shape = temperatures.shape[0]
    chi_T = np.empty(t_shape, dtype=np.float64)
    magnetic_moment_ub = np.diag(magnetic_momenta_au).real / MU_B_AU * B_AU_T

    for t in range(t_shape):
        exp_diff = np.exp(-(energies_au - energies_au[0]) / (KB * temperatures[t]))
        z = np.sum(exp_diff)
        m = np.sum(magnetic_moment_ub * exp_diff)
        chi_T[t] = m / z / field * MU_B_CM_3
    
    return chi_T

@njit(nogil=True, cache=True, fastmath=True)
def get_chi_S(temperatures: np.ndarray):
    return np.zeros_like(temperatures)