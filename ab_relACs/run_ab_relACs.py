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
import scipy.special.cython_special
from threadpoolctl import threadpool_limits
from numba import set_num_threads

from slothpy._general_utilities._constants import H_CM_1

from input_models import AppConfig
from lattice import get_hessian_recip_axes_masses_inv_sqrt_spin_phonon
from spin_system import get_hamiltonian_magnetic_momenta_dof_array, get_chi_T, get_chi_S
from utils import get_normalized_orientations_weights, dot_3d, multigrid_aniso
from spin_phonon import spin_phonon_derivatives
from susceptibility_relax import _DAWSN_ALIASES, _register_dawsn_symbols, make_susceptibility_relax_time
from constants import T_FILED_OE
from exporting import export_susceptibility_csv, export_tau_csv
from plotting import plot_chi_vs_freq, plot_tau_vs_inv_T

def run_relacs(cfg: AppConfig):
    _register_dawsn_symbols()
    set_num_threads(cfg.relacs.number_cpus)
    with threadpool_limits(1):
        susceptibility_relax_time = make_susceptibility_relax_time(cfg)

    hessian, recip_axes, masses_inv_sqrt = get_hessian_recip_axes_masses_inv_sqrt_spin_phonon(cfg)
    hamiltonian, magnetic_momenta, dof_array = get_hamiltonian_magnetic_momenta_dof_array(cfg)
    orientations, orientations_weights = get_normalized_orientations_weights(cfg.relacs.orientations)
    fields = cfg.relacs.fields
    temperatures = cfg.relacs.temperatures
    n_points_array = cfg.relacs.n_points
    fwhm_array = cfg.relacs.fwhm
    omega_Hz = np.concatenate((np.array([1e-4]),cfg.relacs.frequencies))
    omega_angular = omega_Hz * 2 * np.pi

    sus_orient_H_T = np.zeros((n_points_array.shape[0],fwhm_array.shape[0],orientations.shape[0],fields.shape[0],temperatures.shape[0], omega_angular.shape[0]), dtype=np.complex128)
    tau_R21_orient_H_T = np.zeros((n_points_array.shape[0],fwhm_array.shape[0],orientations.shape[0],fields.shape[0],temperatures.shape[0]), dtype=np.float64)
    tau_R41_orient_H_T = np.zeros((n_points_array.shape[0],fwhm_array.shape[0],orientations.shape[0],fields.shape[0],temperatures.shape[0]), dtype=np.float64)
    
    for orientation_index, orientation in enumerate(orientations):
        oriented_momenta = dot_3d(magnetic_momenta, orientation)
        for field_index, field in enumerate(fields):
            field_vector = field * orientation
            hamiltonian_total = hamiltonian - dot_3d(magnetic_momenta, field_vector)
            energies_total, U_total = np.linalg.eigh(hamiltonian_total)
            A = U_total.conj().T @ oriented_momenta @ U_total
            chi_T = get_chi_T(A, energies_total, temperatures, field)
            chi_S = get_chi_S(temperatures) #TODO: find an ab initio model
            A *= H_CM_1
            B = - A
            hamiltonian_gradients = spin_phonon_derivatives(dof_array, field_vector,
                                                       energies_total, U_total, cfg)
            energies_total *= H_CM_1
            with threadpool_limits(1):
                set_num_threads(cfg.relacs.number_cpus)
                for n_points_index, n_points in enumerate(n_points_array):
                    grid, weights = multigrid_aniso(recip_axes, n_points, cfg.relacs.q_ranges)
                    for fwhm_index, fwhm in enumerate(fwhm_array):
                        sus_T, relax_time_R21_T, relax_time_R41_T = susceptibility_relax_time(
                                            omega_angular, energies_total, A, B, hamiltonian_gradients,
                                            temperatures, hessian, masses_inv_sqrt, dof_array, grid,
                                            weights, fwhm, chi_T, chi_S)
                        
                        sus_orient_H_T[n_points_index,fwhm_index,orientation_index,field_index,:,:] = sus_T * orientations_weights[orientation_index]
                        tau_R21_orient_H_T[n_points_index,fwhm_index,orientation_index,field_index,:] = relax_time_R21_T
                        tau_R41_orient_H_T[n_points_index,fwhm_index,orientation_index,field_index,:] = relax_time_R41_T             
    
    sus_H_T = np.sum(sus_orient_H_T, axis=2)
    B_array = fields * T_FILED_OE
    export_susceptibility_csv(temperatures, B_array, omega_Hz, sus_H_T[0,0], cfg.relacs.chi_csv_path) # Only the first n_point, fwhm for now
    export_tau_csv(temperatures, B_array, tau_R21_orient_H_T[0,0,0], cfg.relacs.tau_21_csv_path) # Only the first n_point, fwhm, orient for now
    export_tau_csv(temperatures, B_array, tau_R41_orient_H_T[0,0,0], cfg.relacs.tau_41_csv_path) # Only the first n_point, fwhm, orient for now

    if cfg.relacs.show_plot:
        import matplotlib.pyplot as plt
        plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T, part="imag", title="χ''(ν)")
        plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T, part="real", title="χ'(ν)")
        plot_tau_vs_inv_T(temperatures, fields, n_points_array, fwhm_array, tau_R21_orient_H_T[:,:,0,:,:], tau_R41_orient_H_T[:,:,0,:,:], which="R21", title="τ(T)")
        plt.show()
  
    return 0