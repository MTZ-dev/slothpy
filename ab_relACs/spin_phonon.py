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
import tqdm
import h5py

from slothpy._general_utilities._constants import H_CM_1, A_BOHR
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil
from input_models import AppConfig
from utils import dot_3d

def k_mch(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray):
    k_mch = []
    U_R0_T = U_R0.conj().T
    print("Calculating K_MCH term:")
    for dof in tqdm.tqdm(dof_array):
        k_mch.append(U_R0_T @ group[f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}"][:] @ U_R0)

    return np.asarray(k_mch, dtype=np.complex128)

def E_grad_k_U(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray,
               field_vector: np.ndarray, displacement_number: int, step: float,
               degeneracy_tolerance: float = 1e-9):
    E_grad = []
    k_U = []
    finite_difference_stencil = _central_finite_difference_stencil(1,
                                                displacement_number, step * A_BOHR)

    print("Calculating dE + K_U terms:")
    for dof in tqdm.tqdm(dof_array):
        E_grad_component = np.zeros(U_R0.shape[0], dtype=np.float64)
        k_U_component = np.zeros_like(U_R0)
        stencil_index = -1
        for displacement in range(-displacement_number, displacement_number + 1):
            stencil_index += 1
            if displacement == 0:
                continue
            group_name = f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}_{displacement}"
            magnetic_momenta = group[f"{group_name}/MAGNETIC_DIPOLE_MOMENTA"][:]
            hamiltonian = (group[f"{group_name}/HAMILTONIAN_MATRIX"][:]
                            - dot_3d(magnetic_momenta, field_vector))
            E, U_R0_delta = np.linalg.eigh(hamiltonian)

            S = U_R0_delta.conj().T @ U_R0

            projection_mask = np.abs(E[:, None] - E[None, :]) < degeneracy_tolerance
            M = np.where(projection_mask, S, 0.0) 
            u, s, vt = np.linalg.svd(M)
            U_R0_delta = U_R0_delta @ u @ vt

            E_grad_component += E * finite_difference_stencil[stencil_index]
            k_U_component += U_R0_delta * finite_difference_stencil[stencil_index]

        E_grad.append(E_grad_component)
        k_U.append(U_R0.conj().T @ k_U_component)
    
    return np.asarray(E_grad, dtype=np.float64), np.asarray(k_U, dtype=np.complex128)

def spin_phonon_derivatives(dof_array: np.ndarray, B_vec: np.ndarray,
                             E_tot_0: np.ndarray, U_R0: np.ndarray, cfg: AppConfig):
    
    with h5py.File(cfg.relacs.slt_filepath) as f:
        group = f[cfg.spin_phonon.group_name]
        k_mch_array = k_mch(dof_array, group, U_R0)
        E_grad_array, k_U_array = E_grad_k_U(dof_array, group, U_R0, B_vec,
                                              cfg.spin_phonon.displacement_number,
                                              cfg.spin_phonon.step,
                                              cfg.relacs.degeneracy_tolerance)
    anti_symm_energy = E_tot_0[None, :] - E_tot_0[:, None]
    H_grad = np.empty_like(k_mch_array)
    print("Calculating the whole H_grad array:")

    for i in tqdm.tqdm(range(H_grad.shape[0])):
        grad_mch = anti_symm_energy * k_mch_array[i]
        grad_mch = (grad_mch + grad_mch.conj().T) * 0.5
        grad_ku = anti_symm_energy * k_U_array[i]
        grad_ku = (grad_ku + grad_ku.conj().T) * 0.5
        grad_full = grad_mch + grad_ku
        np.fill_diagonal(grad_full, E_grad_array[i])
        H_grad[i] = grad_full
    
    states_number = cfg.relacs.states_number
    H_grad = np.ascontiguousarray(H_grad[:, :states_number, :states_number]) * H_CM_1
    
    print("Applying the translational symmetry constraint:")
    dir_idx = dof_array[:, 0] % 3
    for l in (0, 1, 2):
        mask = dir_idx == l
        if not mask.any():
            continue
        avg = H_grad[mask].mean(axis=0)
        H_grad[mask] -= avg

    return H_grad