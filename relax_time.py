from __future__ import annotations

import numpy as np
from numpy import pi
from numpy.linalg import eigvals
from numba import njit, prange

import os
import re
import posixpath
from collections import defaultdict
import itertools
from typing import Callable, Iterable, Tuple, Sequence

import numpy as np
import h5py
import numpy as np
from numba import njit, prange

import slothpy as slt
from slothpy._general_utilities._constants import A_BOHR, H_CM_1, B_AU_T, AU_BOHR_CM_1
from slothpy._general_utilities._io import _hamiltonian_derivatives_from_dir_to_slt
from slothpy._general_utilities._lapack import _zdot3d
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil
from slothpy.core._slt_file import SltHessian
from slothpy.core._hessian_object import Hessian

import matplotlib.pyplot as plt


# ─── physical constants (a.u.) ───────────────────────────────────────────────
KB        = 3.166811563e-6          # Eh K⁻¹
H_BAR     = 1.0                     # ℏ
AU_TIME_S = 2.4188843265857e-17     # 1 au time → s

# ─── helpers ─────────────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def liou_idx(a: int, b: int, N: int) -> int:
    return a * N + b

@njit(cache=True, inline="always")
def _lorentz(dE: float, lw: float) -> float:
    return 0.0 if lw <= 0.0 else lw / (dE*dE + lw*lw) / pi

@njit(cache=True, inline="always")
def _gauss(dE: float, lw: float) -> float:
    return 0.0 if lw <= 0.0 else np.exp(-(dE/lw)**2) / (lw*np.sqrt(pi))

@njit(cache=True)
def delta_line(kind: int, dE: float, lw: float) -> float:
    return _lorentz(dE, lw) if kind == 0 else _gauss(dE, lw)

@njit(cache=True, inline="always")
def bose_occ(freq: float, beta: float) -> float:
    u = beta * H_BAR * freq
    return 0.0 if u > 700.0 else 1.0/(np.exp(u) - 1.0)

# ─── second-order (one-phonon) kernel ───────────────────────────────────────
@njit(parallel=True)
def _add_R21_mode(R21: np.ndarray, Ener: np.ndarray, V: np.ndarray,
                  freq: float, nB: float, lw: float,
                  smear: int, sec_tol: float):
    if not np.isfinite(freq) or freq <= 0.0:
        return

    N, N2 = Ener.size, Ener.size*Ener.size
    prefc = pi / H_BAR

    for ab in prange(N2):
        a, b = ab // N, ab % N
        for cd in range(N2):
            c, d = cd // N, cd % N
            if abs(Ener[a]-Ener[c]+Ener[d]-Ener[b]) > sec_tol:
                continue

            val = 0.0 + 0.0j

            # branch (b→d)
            dE = Ener[b]-Ener[d]
            s  = 1.0 if dE > 0.0 else -1.0
            g  = (nB if s > 0 else nB+1.0) * pi \
                 * delta_line(smear, dE - s*freq, lw)
            val += V[a,c]*np.conj(V[b,d]) * g

            # branch (a→c)
            dE = Ener[a]-Ener[c]
            s  = 1.0 if dE > 0.0 else -1.0
            g  = (nB if s > 0 else nB+1.0) * pi \
                 * delta_line(smear, dE - s*freq, lw)
            val += V[a,c]*np.conj(V[b,d]) * g

            # counter terms
            if d == b:
                for k in range(N):
                    dE = Ener[k]-Ener[c]
                    s  = 1.0 if dE > 0.0 else -1.0
                    g  = (nB if s > 0 else nB+1.0) * pi \
                         * delta_line(smear, dE - s*freq, lw)
                    val -= np.conj(V[k,a])*V[k,c] * g
            if c == a:
                for k in range(N):
                    dE = Ener[k]-Ener[d]
                    s  = 1.0 if dE > 0.0 else -1.0
                    g  = (nB if s > 0 else nB+1.0) * pi \
                         * delta_line(smear, dE - s*freq, lw)
                    val -= V[k,b]*np.conj(V[k,d]) * g

            R21[ab, cd] -= prefc * val     # minus sign → negative rates

def build_R21(Ener: np.ndarray, temp: float, lw: float, smear: int,
              gen: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]],
              *, sec_tol: float = 1e-6) -> np.ndarray:
    N = Ener.size
    R = np.zeros((N*N, N*N), np.complex128)
    beta = 1.0 / (KB*temp)
    for Y, w in gen():
        for V, f in zip(Y, w):
            _add_R21_mode(R, Ener, V, f, bose_occ(f, beta),
                          lw, smear, sec_tol)
    return R

# ─── fourth-order helper: resolvent matrix ──────────────────────────────────
@njit(cache=True)
def _R_pm(V1, V2, Ener, sign, freq, lw):
    N = Ener.size
    out = np.zeros((N, N), np.complex128)
    for i in range(N):
        for j in range(N):
            acc = 0.0+0.0j
            for k in range(N):
                acc += V1[i,k]*V2[k,j] / (Ener[k]-Ener[j]+sign*freq-1j*lw)
            out[i,j] = acc
    return out

def make_R41(Ener, V1, V2, T, f1, f2, lw1, lw2, smear, *, sec_tol=1e-6):
    N, N2 = Ener.size, Ener.size*Ener.size
    R = np.zeros((N2,N2), np.complex128)
    beta = 1.0 / (KB*T)
    n1, n2 = bose_occ(f1,beta), bose_occ(f2,beta)
    lw = lw1+lw2
    Rabp = _R_pm(V1,V2,Ener,+1,f2,lw2)
    Rabm = _R_pm(V1,V2,Ener,-1,f2,lw2)
    Rbap = _R_pm(V2,V1,Ener,+1,f1,lw1)
    Rbam = _R_pm(V2,V1,Ener,-1,f1,lw1)
    prefc = pi*pi / H_BAR
    Δ = lambda dE: delta_line(smear,dE,lw)

    for a in range(N):
        for b in range(N):
            ab = liou_idx(a,b,N)
            for c in range(N):
                for d in range(N):
                    cd = liou_idx(c,d,N)
                    if abs(Ener[a]-Ener[c]+Ener[d]-Ener[b]) > sec_tol:
                        continue
                    val = 0.0+0.0j
                    # ω1 emit / ω2 absorb
                    G = n1*(n2+1)*Δ(Ener[a]-Ener[c]-f1+f2)
                    val += (np.conj(Rabp[b,d])*Rabp[a,c]
                          + np.conj(Rabp[b,d])*Rbam[a,c]
                          + np.conj(Rbam[b,d])*Rabp[a,c]
                          + np.conj(Rbam[b,d])*Rbam[a,c]) * G
                    # ω1 absorb / ω2 emit
                    G = n2*(n1+1)*Δ(Ener[a]-Ener[c]+f1-f2)
                    val += (np.conj(Rabm[b,d])*Rabm[a,c]
                          + np.conj(Rabm[b,d])*Rbap[a,c]
                          + np.conj(Rbap[b,d])*Rabm[a,c]
                          + np.conj(Rbap[b,d])*Rbap[a,c]) * G
                    # double absorption
                    G = n1*n2*Δ(Ener[a]-Ener[c]-f1-f2)
                    val += (np.conj(Rabm[b,d])*Rabm[a,c]
                          + np.conj(Rbam[b,d])*Rbam[a,c]
                          + np.conj(Rbam[b,d])*Rabm[a,c]
                          + np.conj(Rabm[b,d])*Rbam[a,c]) * G
                    # double emission
                    G = (n1+1)*(n2+1)*Δ(Ener[a]-Ener[c]+f1+f2)
                    val += (np.conj(Rabp[b,d])*Rabp[a,c]
                          + np.conj(Rbap[b,d])*Rbap[a,c]
                          + np.conj(Rbap[b,d])*Rabp[a,c]
                          + np.conj(Rabp[b,d])*Rbap[a,c]) * G
                    R[ab,cd] -= prefc*val     # minus sign
    return R

def build_R41(Ener, T, lw, smear, gen, *, sec_tol=1e-6):
    N = Ener.size
    R = np.zeros((N*N,N*N), np.complex128)
    for Y, w in gen():
        J = len(w)
        for i in range(J):
            for j in range(i, J):
                R += make_R41(Ener, Y[i], Y[j], T,
                              w[i], w[j], lw, lw,
                              smear, sec_tol=sec_tol)
    return R

# ─── Lindbladian + relaxation time ──────────────────────────────────────────
def redfield_lindbladian(Ener, T, lw, smear, gen, *,
                         include_R41=False, sec_tol=1e-6):
    R21 = build_R21(Ener, T, lw, smear, gen, sec_tol=sec_tol)
    R41 = build_R41(Ener, T, lw, smear, gen, sec_tol=sec_tol) if include_R41 else np.zeros_like(R21)

    N = Ener.size
    iL = np.zeros_like(R21)
    ω  = Ener / H_BAR
    for a in range(N):
        for b in range(N):
            iL[liou_idx(a,b,N), liou_idx(a,b,N)] = 1j*(ω[a]-ω[b])

    return iL + R21 + R41, R21, R41

def relaxation_time(R, *, tol=0):
    lam = -eigvals(R)
    lam = lam[np.isfinite(lam)]
    neg = lam.real[lam.real < -tol]
    if neg.size == 0:
        raise ValueError("Lindbladian has no negative eigenvalues")
    return -1.0 / neg.max()               # au

# ─── plotting helpers ───────────────────────────────────────────────────────
def make_T1_accumulator():
    bank: dict[tuple[str,float,float], list[tuple[float,float]]] = defaultdict(list)
    def add(T: float, label: str, B: float, γ: float, T1_s: float):
        bank[(label,B,γ)].append((T,T1_s))
    def finish(invT=True):
        if not bank:
            print("nothing to plot"); return
        plt.figure(figsize=(6,4))
        for (lab,B,γ), data in bank.items():
            data.sort()
            T,T1 = map(np.array, zip(*data))
            x = 1.0/T if invT else T
            plt.plot(x, T1, marker='o', lw=1,
                     label=f"{lab}  B={B:g} T γ={γ:.0e}")
        plt.yscale('log')
        plt.ylabel("T₁ (s)")
        if invT:
            plt.xlabel("1 / T  (K⁻¹)")
        else:
            plt.xlabel("Temperature (K)")
            plt.xscale('log')
        plt.grid(True, ls=':')
        plt.legend(frameon=False, fontsize='x-small')
        plt.tight_layout(); plt.show()
    return add, finish



@njit(cache=True, parallel=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt):
    for i in range(dof_array.shape[0]):
        dof = dof_array[i]
        for j in prange(normal_modes.shape[1]):
            Y_q[j] +=  H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))

def dofs_with_complete_displacements(h5_group: h5py.Group, displacement_number: int):
    
    base_regex = re.compile(r'^(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)$')

    complete = []

    def visitor(rel_name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        leaf_name = posixpath.basename(rel_name)
        m = base_regex.fullmatch(leaf_name)
        if not m:
            return

        dof, nx, ny, nz = map(int, m.groups())
        base = leaf_name
        parent = obj.parent

        missing = [d for d in range(-displacement_number, displacement_number + 1) if d != 0 and f"{base}_{d}" not in parent]

        if not missing:
            complete.append([dof, nx, ny, nz])

    h5_group.visititems(visitor)
    complete = np.asarray(complete, dtype=np.int64)

    return complete


def k_mch(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray):
    k_mch = []
    U_R0_T = U_R0.T
    U_R0_conj = U_R0.conj()

    for dof in dof_array:
        k_mch.append(U_R0_T @ group[f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}"][:] @ U_R0_conj)

    return np.asarray(k_mch, dtype=np.complex128)


def E_grad_k_U(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray, magnetic_field_vector: np.ndarray, displacement_number: int, degeneracy_tolerance: float = 1e-8):
    E_grad = []
    k_U = []
    finite_difference_stencil = _central_finite_difference_stencil(1, displacement_number, step * A_BOHR)

    for dof in dof_array:
        E_grad_component = np.zeros(U_R0.shape[0], dtype=np.float64)
        k_U_component = np.zeros_like(U_R0)
        stencil_index = -1
        for displacement in range(-displacement_number, displacement_number + 1):
            stencil_index += 1
            if displacement == 0:
                continue
            group_name = f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}_{displacement}"
            hamiltonian = group[f"{group_name}/HAMILTONIAN_MATRIX"][:]+_zdot3d(group[f"{group_name}/MAGNETIC_DIPOLE_MOMENTA"][:], -magnetic_field_vector)
            E, U_R0_delta = np.linalg.eigh(hamiltonian)

            S = U_R0_delta.conj().T @ U_R0

            projection_mask = np.abs(E[:, None] - E[None, :]) < degeneracy_tolerance
            M = np.where(projection_mask, S, 0.0) 
            u, s, vt = np.linalg.svd(M)
            U_R0_delta = U_R0_delta @ u @ vt

            E_grad_component += E * finite_difference_stencil[stencil_index]
            k_U_component += U_R0_delta * finite_difference_stencil[stencil_index]

        E_grad.append(E_grad_component)
        k_U.append(k_U_component.conj().T @ U_R0)
    
    return np.asarray(E_grad, dtype=np.float64), np.asarray(k_U, dtype=np.complex128)


if __name__ == "__main__":

    # ── USER-CONFIGURABLE SWEEP LISTS & PARAMETERS ──────────────────────────
    npoints_list    = [5]
    gamma_fwhm_list = [4e-5]          # FWHM in a.u.
    T_list          = [5,8,10,20,30,50,80,100]           # Kelvin
    B_list          = [0.1]            # Tesla
    states_number   = 8                    # electronic sub-space size
    modes_number    = 30                   # phonon modes per k-point
    # ────────────────────────────────────────────────────────────────────────

    # one-shot data that never changes over the sweep -----------------------
    orca_fragovl_path   = "/home/mikolaj/orca_6_0_1_avx2/orca_fragovl"
    dirpath             = "/home/mikolaj/Data/Displacements_small/CeCo_displ" # "/home/mikolaj/Data/Displacements_cluster/CeCo_displ_cluster"
    slt_filepath        = "./seminarium/import.slt"
    group_name          = "xxx"
    displacement_number = 1
    step                = 0.025
    omega_SI            = np.logspace(0.0001, 6, 500)
    omega_au            = np.logspace(-7, 0, 100)

    # refresh the .slt file
    if os.path.exists(slt_filepath):
        os.remove(slt_filepath)
    slt.set_default_error_reporting_mode()
    _hamiltonian_derivatives_from_dir_to_slt(
        dirpath, slt_filepath, group_name,
        displacement_number, step, 64, 1, "ORCA",
        False, False, False, orca_fragovl_path
    )

    # phonon part -------------------------------------------------------
    Dy = slt.supercell(
        "./seminarium/YCo_supercell_from_cell/dof_0_disp_0.xyz",
        slt_filepath, "YCo_supercell",
        3, 3, 2,
        supercell_params=[22.663134149075237,
                            22.663134149075233,
                            25.14851428466812,
                            90.0, 90.0, 120.0],
        multiplicity=1,
    )
    hessian = Dy["YCo_supercell"].hessian_from_finite_displacements(
        "./seminarium/YCo_supercell_from_cell",
        "CP2K", "YCo_hessian",
        1, 0.01, born_charges=True
    )
    slt_hessian     = SltHessian(hessian)
    masses_inv_sqrt = slt_hessian._masses_inv_sqrt
    recip_axes = slt_hessian.atoms_object().cell.reciprocal().cellpar()[:3]
    hess_obj        = Hessian(
        slt_hessian.hessian()[:],
        np.outer(masses_inv_sqrt, masses_inv_sqrt),
        np.array([0., 0., 0.])
    )

    add, finish = make_T1_accumulator()

    for npoints in npoints_list:

        # -------- data that depend *only* on npoints -----------------------
        with h5py.File(slt_filepath, "r") as f:
            grp          = f[group_name]
            dof_array    = dofs_with_complete_displacements(grp, displacement_number)
            hamiltonian0 = grp["0/HAMILTONIAN_MATRIX"][:]
            AB           = _zdot3d(grp["0/MAGNETIC_DIPOLE_MOMENTA"][:],
                                   np.asarray([0, 1, 1], dtype = np.float64))
        
        # ------------------------------------------------------------------ #
        #  INNER SWEEP over linewidth γ, temperature T and field B           #
        # ------------------------------------------------------------------ #
        for gamma_fwhm, T, B in itertools.product(
                gamma_fwhm_list, T_list, B_list):

            # correct B-vector *and* B-dependent electronic objects ---------
            orient = np.array([0, 1, 1], np.float64)
            orient /= np.linalg.norm(orient)
            B_vec  = B / B_AU_T * orient

            q = np.linspace(-0.5, 0.5, npoints, endpoint=True)
            q_grid = np.meshgrid(q, q, q, indexing='ij')
            grid = np.ascontiguousarray(np.vstack([grid.ravel() for grid in q_grid]).T)

            with h5py.File(slt_filepath, "r") as f:
                grp = f[group_name]
                H_total = grp["0/HAMILTONIAN_MATRIX"][:] + \
                          _zdot3d(grp["0/MAGNETIC_DIPOLE_MOMENTA"][:], -B_vec)

                # field-dependent eigenvectors
                E_tot, U_R0 = np.linalg.eigh(H_total)

                # k-matrix & gradients with **CORRECT** B-vector
                k_mch_arr = k_mch(dof_array, grp, U_R0)
                E_grad, k_U_arr = E_grad_k_U(
                    dof_array, grp, U_R0,
                    B_vec, displacement_number, 1e-10
                )

            # build H_grad (field dependent) and truncate -------------------
            H_grad = np.empty_like(k_mch_arr)
            for i in range(H_grad.shape[0]):
                Ek = (E_tot[None, :] - E_tot[:, None]) * (k_mch_arr[i] + k_U_arr[i])
                np.fill_diagonal(Ek, E_grad[i])
                H_grad[i] = Ek
            H_grad = H_grad[:, :states_number, :states_number]

            # closure capturing *field-dependent* H_grad --------------------
            def get_Y_q_and_freq():
                n_k_inv = 1.0 / np.sqrt(len(grid))
                for q in grid:
                    print(q)
                    hess_obj.kpoint = q
                    freq, modes = hess_obj.frequencies_eigenvectors

                    # --- keep only the requested number of modes --------------------
                    freq  = freq[:modes_number] * (AU_BOHR_CM_1 / H_CM_1/ H_BAR)
                    modes = modes[:, :modes_number]

                    nz    = np.where(freq <= 0)[0]
                    start = nz.max() + 1 if nz.size else 0

                    Y_q = np.zeros(
                        (freq.size, states_number, states_number),
                        dtype=np.complex128,
                    )

                    get_Y_q(
                        Y_q,
                        H_grad,
                        modes,
                        q,
                        dof_array,
                        masses_inv_sqrt,
                        n_k_inv,
                    )

                    yield (
                        np.ascontiguousarray(Y_q[start:]),
                        np.ascontiguousarray(freq[start:]),
                    )

            label = (f"np={npoints}, γ={gamma_fwhm:.0e}, "
                    f"T={T:g} K, B={B:g} T")


            Rtot, R21, R41 = redfield_lindbladian(
                E_tot[:states_number], T, gamma_fwhm, 0, get_Y_q_and_freq,
                include_R41=True)

            T1_R21_s = AU_TIME_S * relaxation_time(R21)
            print(T1_R21_s)
            T1_R41_s = AU_TIME_S * relaxation_time(R41)
            print(T1_R41_s)

            add(T, "R21", B, gamma_fwhm, T1_R21_s)
            add(T, "R41", B, gamma_fwhm, T1_R41_s)



            # print(T, gamma_fwhm)
            # print("R21 =\n", R21)
            # print("T1  =", relaxation_time(R21))

    finish(invT=True)
