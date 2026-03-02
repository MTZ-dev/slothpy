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

from itertools import product
import time

import numpy as np
import scipy.special.cython_special
from threadpoolctl import threadpool_limits
from numba import set_num_threads

from slothpy._general_utilities._constants import H_CM_1, B_AU_T, MU_B_AU

from input_models import AppConfig
from lattice import get_hessian_recip_axes_masses_inv_sqrt_spin_phonon
from spin_system import get_hamiltonian_magnetic_momenta_dof_array, get_chi_T, get_chi_S
from utils import get_normalized_orientations_weights, dot_3d, multigrid_aniso, make_npoints_fwhm_filename, make_npoints_fwhm_orient_filename, make_wdos_path, make_energy_lines_from_levels
from spin_phonon import spin_phonon_derivatives
from susceptibility_relax import susceptibility_relax_time, plot_spinphonon_weighted_phonon_dos
from constants import T_FILED_OE
from exporting import export_susceptibility_csv, export_tau_csv
from plotting import plot_chi_vs_freq, plot_tau_vs_inv_T

def run_relacs(cfg: AppConfig):
    start = time.perf_counter()

    hessian, recip_axes, masses_inv_sqrt, hessian_group = get_hessian_recip_axes_masses_inv_sqrt_spin_phonon(cfg)
    hamiltonian, magnetic_momenta, dof_array = get_hamiltonian_magnetic_momenta_dof_array(cfg)
    orientations, orientations_weights = get_normalized_orientations_weights(cfg.relacs.orientations)
    fields = cfg.relacs.fields
    temperatures = cfg.relacs.temperatures
    n_points_array = cfg.relacs.n_points
    fwhm_array = cfg.relacs.fwhm
    omega_Hz = np.concatenate((np.array([np.min(cfg.relacs.frequencies)*0.1]),cfg.relacs.frequencies))
    omega_angular = omega_Hz * 2 * np.pi
    cutoff_mult = cfg.relacs.cutoff_fwhm
    degeneracy_tolerance = cfg.relacs.degeneracy_tolerance
    states_number = cfg.relacs.states_number
    modes_low = cfg.relacs.modes_low
    modes_high = cfg.relacs.modes_high
    threads = cfg.relacs.number_cpus

    kind = 1 if cfg.relacs.broadening == "gaussian" else 0
    direct = 1 if cfg.relacs.direct else 0
    run_KR = 1 if cfg.relacs.chi_csv_path else 0
    run_PSI = 1 if cfg.relacs.psi_frequency_shift else 0
    run_rho0 = 1 if cfg.relacs.initial_correlation else 0
    run_R21 = 1 if cfg.relacs.tau_21_csv_path else 0
    run_R41 = 1 if cfg.relacs.tau_41_csv_path else 0
    symm = 1 if cfg.relacs.symm else 0

    sus_orient_H_T = np.zeros((4,n_points_array.shape[0],fwhm_array.shape[0],orientations.shape[0],fields.shape[0],temperatures.shape[0], omega_angular.shape[0]), dtype=np.complex128)
    tau_R21_orient_H_T = np.zeros((n_points_array.shape[0],fwhm_array.shape[0],orientations.shape[0],fields.shape[0],temperatures.shape[0]), dtype=np.float64)
    tau_R41_orient_H_T = np.zeros((n_points_array.shape[0],fwhm_array.shape[0],orientations.shape[0],fields.shape[0],temperatures.shape[0]), dtype=np.float64)

    # ---- DOS-only shortcut -------------------------------------------------
    dos_cfg = cfg.weighted_dos
    run_only_wdos = (dos_cfg is not None) and bool(dos_cfg.save_path)

    if run_only_wdos:
        dos_temperature = dos_cfg.temperature
        if dos_cfg.weight_mode.startswith("thermal") and dos_temperature is None:
            dos_temperature = float(temperatures[0])

        for orientation_index, orientation in enumerate(orientations):
            oriented_momenta = dot_3d(magnetic_momenta, orientation)

            for field_index, field in enumerate(fields):
                field_vector = field * orientation

                hamiltonian_total = hamiltonian - dot_3d(magnetic_momenta, field_vector)
                energies_total, U_total = np.linalg.eigh(hamiltonian_total)

                hamiltonian_gradients = spin_phonon_derivatives(
                    dof_array, field_vector, energies_total, U_total, cfg
                )

                energies_total_cm1 = energies_total * H_CM_1

                energy_lines_auto = make_energy_lines_from_levels(
                    energies_total_cm1[:states_number],
                    modes_low=cfg.relacs.modes_low,
                    modes_high=cfg.relacs.modes_high,
                    tol=cfg.relacs.degeneracy_tolerance,
                    max_lines=50,                     
                    )

                for n_points in n_points_array:
                    grid_q, w_q, n_k = multigrid_aniso(recip_axes, int(n_points), cfg.relacs.q_ranges)

                    save_path = make_wdos_path(
                        dos_cfg.save_path,
                        n_points=int(n_points),
                        field=float(field),
                        orientation=orientation * B_AU_T,
                        suffix=".pdf"
                    )

                    energy_lines = (
                        energy_lines_auto
                        if (len(dos_cfg.energy_lines) == 0)
                        else dos_cfg.energy_lines
                        )

                    with threadpool_limits(1):
                        plot_spinphonon_weighted_phonon_dos(
                            E=energies_total_cm1,
                            H_grad=hamiltonian_gradients,
                            hessian=hessian,
                            masses_inv_sqrt=masses_inv_sqrt,
                            dof_array=dof_array,
                            grid=grid_q,
                            weights=w_q,
                            n_k=n_k,
                            states_number=states_number,
                            threads=threads,

                            modes_low=cfg.relacs.modes_low,
                            modes_high=cfg.relacs.modes_high,

                            resolution=dos_cfg.resolution,
                            convolution=dos_cfg.convolution,
                            fwhm=dos_cfg.fwhm,
                            density=dos_cfg.density,

                            weight_mode=dos_cfg.weight_mode,
                            temperature=dos_temperature,

                            dos_freq=dos_cfg.dos_freq,
                            eps_freq_cm1=dos_cfg.eps_freq_cm1,

                            save_path=save_path,
                            show=bool(dos_cfg.show),
                            energy_lines=energy_lines,
                            title=dos_cfg.title,
                        )


        end = time.perf_counter()
        print(f"Weighted DOS-only runtime: {end - start} s")
        return
    
    for orientation_index, orientation in enumerate(orientations):
        oriented_momenta = dot_3d(magnetic_momenta, orientation)
        for field_index, field in enumerate(fields):
            field_vector = field * orientation
            hamiltonian_total = hamiltonian - dot_3d(magnetic_momenta, field_vector)
            energies_total, U_total = np.linalg.eigh(hamiltonian_total)
            A = U_total.conj().T @ oriented_momenta @ U_total
            chi_T = get_chi_T(A, energies_total, temperatures, field)
            chi_S = get_chi_S(temperatures)
            B = A * B_AU_T / MU_B_AU
            A *= H_CM_1
            hamiltonian_gradients = spin_phonon_derivatives(dof_array, field_vector, energies_total, U_total, cfg)
            energies_total *= H_CM_1

            for n_points_index, n_points in enumerate(n_points_array):
                grid, weights, n_k = multigrid_aniso(recip_axes, n_points, cfg.relacs.q_ranges)
                for fwhm_index, fwhm in enumerate(fwhm_array):
                    if kind == 1:
                        gamma_fwhm = fwhm / (2 * np.sqrt(2*np.log(2)))
                    elif kind == 0:
                        gamma_fwhm = fwhm / 2
                    with threadpool_limits(1):
                        sus_T, relax_time_R21_T, relax_time_R41_T = susceptibility_relax_time(
                                            omega_angular, energies_total, A, B, hamiltonian_gradients,
                                            temperatures, hessian, masses_inv_sqrt, dof_array, grid,
                                            weights, gamma_fwhm, chi_T, chi_S, cutoff_mult, degeneracy_tolerance,
                                            states_number, modes_low, modes_high, threads, kind, direct,
                                            run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k, symm)
                    sus_orient_H_T[:,n_points_index,fwhm_index,orientation_index,field_index,:,:] = sus_T
                    tau_R21_orient_H_T[n_points_index,fwhm_index,orientation_index,field_index,:] = relax_time_R21_T
                    tau_R41_orient_H_T[n_points_index,fwhm_index,orientation_index,field_index,:] = relax_time_R41_T             
        
    sus_H_T = np.sum(sus_orient_H_T * orientations_weights[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis], axis=3)
    B_array = fields * T_FILED_OE
    orientations *= B_AU_T
    for n, f, o in product(range(n_points_array.shape[0]), range(fwhm_array.shape[0]), range(orientations.shape[0])):
        if cfg.relacs.chi_csv_path:
                if o == 0:
                    save_filepath = make_npoints_fwhm_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f])
                    export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_H_T[0,n,f], save_filepath)
                    save_filepath = make_npoints_fwhm_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], "no_chit")
                    export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_H_T[1,n,f], save_filepath)
                    if cfg.relacs.psi_frequency_shift:
                        save_filepath = make_npoints_fwhm_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], "psi")
                        export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_H_T[2,n,f], save_filepath)
                        if cfg.relacs.initial_correlation:
                            save_filepath = make_npoints_fwhm_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], "psi_init")
                            export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_H_T[3,n,f], save_filepath)
                save_filepath = make_npoints_fwhm_orient_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], orientations[o], sig=6, int_tol=1e-12)
                export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_orient_H_T[0,n,f,o], save_filepath)
                save_filepath = make_npoints_fwhm_orient_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], orientations[o], sig=6, int_tol=1e-12, additional="no_chit")
                export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_orient_H_T[1,n,f,o], save_filepath)
                if cfg.relacs.psi_frequency_shift:
                    save_filepath = make_npoints_fwhm_orient_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], orientations[o], sig=6, int_tol=1e-12, additional="psi")
                    export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_orient_H_T[2,n,f,o], save_filepath)
                    if cfg.relacs.initial_correlation:
                        save_filepath = make_npoints_fwhm_orient_filename(cfg.relacs.chi_csv_path, n_points_array[n], fwhm_array[f], orientations[o], sig=6, int_tol=1e-12, additional="psi_init")
                        export_susceptibility_csv(B_array, temperatures, omega_Hz, sus_orient_H_T[3,n,f,o], save_filepath)
        if cfg.relacs.tau_21_csv_path:
                save_filepath = make_npoints_fwhm_orient_filename(cfg.relacs.tau_21_csv_path, n_points_array[n], fwhm_array[f], orientations[o], sig=6, int_tol=1e-12)
                export_tau_csv(B_array, temperatures, tau_R21_orient_H_T[n,f,o], save_filepath)
        if cfg.relacs.tau_41_csv_path:
                save_filepath = make_npoints_fwhm_orient_filename(cfg.relacs.tau_41_csv_path, n_points_array[n], fwhm_array[f], orientations[o], sig=6, int_tol=1e-12)
                export_tau_csv(B_array, temperatures, tau_R41_orient_H_T[n,f,o], save_filepath)

    end = time.perf_counter()
    print(f"Running time: {end - start} s")

    if cfg.relacs.show_plot:
        import matplotlib.pyplot as plt
        if cfg.relacs.chi_csv_path:
            plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[0], part="imag", title="χ''(ν)")
            plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[0], part="real", title="χ'(ν)")
            plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[1], part="imag", title="χ''(no_chiT)(ν)")
            plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[1], part="real", title="χ'(no_chiT)(ν)")
            if cfg.relacs.psi_frequency_shift:
                plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[2], part="imag", title="χ''(psi)(ν)")
                plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[2], part="real", title="χ'(psi)(ν)")
                if cfg.relacs.initial_correlation:
                    plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[3], part="imag", title="χ''(init)(ν)")
                    plot_chi_vs_freq(omega_Hz, temperatures, fields, n_points_array, fwhm_array, sus_H_T[3], part="real", title="χ'(init)(ν)")
        if cfg.relacs.tau_21_csv_path or cfg.relacs.tau_41_csv_path:
            plot_tau_vs_inv_T(temperatures, fields, n_points_array, fwhm_array, tau_R21_orient_H_T[:,:,0,:,:], tau_R41_orient_H_T[:,:,0,:,:], which="both", title="τ(T)")
        plt.show()