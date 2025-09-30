#!/usr/bin/env python3

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

from slothpy._general_utilities._constants import H_CM_1

from input_models import AppConfig
from lattice import get_hessian_recip_axes_spin_phonon
from spin_system import get_hamiltonian_magnetic_momenta_dof_array, get_chi_T, get_chi_S
from utils import get_normalized_orientations_weights, dot_3d

def run_relacs(cfg: AppConfig):
    hessian, recip_axes = get_hessian_recip_axes_spin_phonon(cfg)
    hamiltonian, magnetic_momenta, dof_array = get_hamiltonian_magnetic_momenta_dof_array(cfg)
    orientations, orientations_weights = get_normalized_orientations_weights(cfg)
    fields = cfg.relacs.fields
    temperatures = cfg.relacs.temperatures

    for orientation_index, orientation in enumerate(orientations):
        oriented_momenta = dot_3d(magnetic_momenta, orientation)
        for field_index, field in enumerate(fields):
            field_vector = field * orientation
            hamiltonian_total = hamiltonian - dot_3d(magnetic_momenta, field_vector)
            energies_total, U_total = np.linalg.eigh(hamiltonian_total)
            A = U_total.conj().T @ oriented_momenta @ U_total
            chi_T = get_chi_T(A, energies_total, temperatures, field)
            chi_S = get_chi_S(temperatures) #TODO: find ab initio model
            A *= H_CM_1
            B = - A

    return 0