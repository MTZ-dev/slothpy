from __future__ import annotations

import tqdm

import numpy as np
from numpy import pi
from scipy.linalg import eigvals
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
KB        = 0.6950347291
H         = 33.3571775619
H_BAR     = 33.3571775619 / (2 * pi)
AU_TIME_S = 1e-12
M_AU = 1822.89


@njit(cache=True)
def Jhat_p(omega, w_ab, wq, n_q, d):
    if np.abs(w_ab) < 0.1:
        return 0.0 + 0.0 * 1j
    if w_ab > 0:
        z = -1j / (w_ab - wq - 1j * d) * n_q
        return z if z.imag < 0 else np.conjugate(z)
    if w_ab < 0:
        z = -1j / (w_ab + wq - 1j * d) * (n_q + 1)
        return z if z.imag < 0 else np.conjugate(z)

@njit(cache=True)
def Jhat_m(omega, w_ab, wq, n_q, d):
    if np.abs(w_ab) < 0.1:
        return 0.0 + 0.0 * 1j
    if w_ab < 0:
        z = -1j / (w_ab + wq - 1j * d) * n_q
        return z if z.imag > 0 else np.conjugate(z)
    if w_ab > 0:
        z = -1j / (w_ab - wq - 1j * d) * (n_q + 1)
        return z if z.imag > 0 else np.conjugate(z)
    
@njit(cache=True)
def zeta(x: float, beta: float) -> float:
    eps = 1e-40
    if abs(x) < eps:
        return beta
    u = beta * x
    return np.expm1(u) / (x)

@njit(cache=True)
def Jcorr(omega, omega_p, w_ab, wq, n_q, d, beta):
    if np.abs(w_ab) < 0.1:
        return 0.0 + 0.0 * 1j
    if w_ab < 0:
        z = -1j / (omega_p + wq - 1j * d) * n_q * zeta(w_ab + wq, beta)
        return z if z.imag < 0 else np.conjugate(z)
    if w_ab > 0:
        z = -1j / (omega_p - wq - 1j * d) * (n_q + 1) * zeta(w_ab - wq, beta)
        return z if z.imag > 0 else np.conjugate(z)

@njit(cache=True)
def add_PSI_bundle(out, omega, A, Yb, wb, nb, delta, beta, w_n, q_0):
    N=w_n.size; N2=N*N; J=wb.size
    coeff = 1 / H_BAR
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh=np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    for d in range(N):
                        cd=liou(c,d,N)
                        val=0.0+0.0j
                        for e in range(N):
                            val+=Jcorr(omega,w_n[e]-w_n[d],w_n[d]-w_n[b],wq,n_q,delta,beta)*A[e,c]*(Yh[d,b]*Y[a,e]+Y[d,b]*Yh[a,e])
                            val-=Jcorr(omega,w_n[a]-w_n[d],w_n[d]-w_n[e],wq,n_q,delta,beta)*A[a,c]*(Y[d,e]*Yh[e,b]+Yh[d,e]*Y[e,b])
                            if a==c:
                                for f in range(N):
                                    val+=Jcorr(omega,w_n[c]-w_n[d]+w_n[e]-w_n[f],w_n[d]-w_n[e],wq,n_q,delta,beta)*A[e,f]*(Yh[f,b]*Y[d,e]+Y[f,b]*Yh[d,e])
                            val-=Jcorr(omega,w_n[c]-w_n[d]+w_n[e]-w_n[b],w_n[d]-w_n[e],wq,n_q,delta,beta)*(Y[a,c]*Yh[d,e]+Yh[a,c]*Y[d,e])*A[e,b]
                        out[ab,cd]+=val*coeff

@njit(cache=True)
def add_KR_bundle(out, omega, Yb, wb, nb, delta, w_n, q_0):
    N = w_n.size; N2 = N * N; J = wb.size
    coeff = 1 / H_BAR
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh = np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab = liou(a,b,N)
                for c in range(N):
                    for d in range(N):
                        cd = liou(c,d,N)
                        val = 0.0+0.0j
                        if d==b:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat_p(omega,w_n[e]-w_n[d],wq,n_q,delta)*(Y[a,e]*Yh[e,c] + Yh[a,e]*Y[e,c])
                            val+=tmp
                        val-=Jhat_p(omega,w_n[a]-w_n[d],wq,n_q,delta)*(Y[a,c]*Yh[d,b] + Yh[a,c]*Y[d,b])
                        if a==c:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat_m(omega,w_n[c]-w_n[e],wq,n_q,delta)*(Y[d,e]*Yh[e,b] + Yh[d,e]*Y[e,b])
                            val+=tmp
                        val-=Jhat_m(omega,w_n[c]-w_n[b],wq,n_q,delta)*(Y[a,c]*Yh[d,b] + Yh[a,c]*Y[d,b])
                        out[ab,cd]+=val * coeff

@njit(cache=True)
def add_rho0_bundle(out, A, Yb, wb, nb, beta, w_n, q_0):
    N=w_n.size; J=wb.size
    coeff = -1 * H_BAR * H_BAR / (2.0)
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh=np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    for d in range(N):
                        cd=liou(c,d,N)
                        corr=0.0+0.0j
                        for e in range(N):
                            w_de=w_n[d]-w_n[e]
                            corr+=(n_q*Iint(w_de+wq,w_n[e]-w_n[b]-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_n[e]-w_n[b]+wq,beta))*A[a,c]*(Y[d,e]*Yh[e,b]+Yh[d,e]*Y[e,b])
                            for f in range(N):
                                corr-=(n_q*Iint(w_de+wq,w_n[e]-w_n[f]-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_n[e]-w_n[f]+wq,beta))*(1.0 if a==c else 0.0)*(Y[d,e]*Yh[e,f] + Yh[d,e]*Y[e,f])*A[f,b]
                        out[ab,cd]+=coeff*corr

@njit(cache=True, inline="always")
def Iint(w1: float, w2: float, beta: float) -> float:
    eps = 1e-40
    if abs(w1) < eps and abs(w2) < eps:
        return 0.5 * beta * beta

    def expm1_drop(u):
        return np.expm1(u)

    if abs(w2) < eps:
        u = w1 * beta
        num = np.exp(u)
        return beta * num / (1 * w1) - expm1_drop(u) / (1**2 * w1**2)
    if abs(w1) < eps:
        u = w2 * beta
        return expm1_drop(u) / (1**2 * w2**2) - beta / (1 * w2)
    if abs(w1 + w2) < eps:
        u = w1 * beta
        return -(beta - expm1_drop(u) / (1 * w1)) / (1 * w1)

    u1  = 1 * w1 * beta
    u12 = 1 * (w1 + w2) * beta
    term1 = expm1_drop(u12) / (1**2 * w2 * (w1 + w2))
    term2 = expm1_drop(u1)  / (1**2 * w1 * w2)
    return term1 - term2

@njit(cache=True, inline="always")
def bose_occ(freq: float, beta: float) -> float:
    u = beta * freq
    return 1.0/(np.exp(u) - 1.0)

@njit(cache=True)
def liou(a, b, N):
    return a * N + b

def build_KR(Ener: np.ndarray, temp: float, lw: float,
              gen: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    N = Ener.size
    R = np.zeros((N*N, N*N), np.complex128)
    beta = 1.0 / (KB*temp)
    for Y, w, q_0 in gen():
        add_KR_bundle(R, 0.0, Y, w, bose_occ(w, beta), lw, Ener, q_0)

    return R

def build_M_PSI(Ener: np.ndarray, temp: float, lw: float,
              gen: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]], A) -> np.ndarray:
    N = Ener.size
    R = np.zeros((N*N, N*N), np.complex128)
    beta = 1.0 / (KB*temp)
    for Y, w, q_0 in gen():
        add_PSI_bundle(R, 0.0, A, Y, w, bose_occ(w, beta), lw, beta, Ener, q_0)

    return R

def half_bz_grid_aniso(
    b_len: Sequence[float],
    n_ref: int,
    *,
    endpoint: bool = False,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Anisotropic first-BZ mesh with equal point density in Cartesian q-space.

    Parameters
    ----------
    b_len    : (3,) sequence
        Lengths |b1|, |b2|, |b3| of the reciprocal-lattice vectors.
    n_ref    : int (odd)
        Points along the *shortest* axis in the *full* mesh (must be odd).
    endpoint : bool, optional
        If True,  +0.5  is included on every axis (closed grid);
        if False (default), the grid is half-open on the +0.5 side.
        In both cases 0 is centred on all axes.
    tol      : float, optional
        Tolerance for zero / symmetry tests.

    Returns
    -------
    q : (M, 3) ndarray
        Unique q-points (fractional coords), one per {+q, –q}, Γ included.
    """
    # ---- 0. sanity checks -------------------------------------------------
    if n_ref % 2 == 0:
        raise ValueError("n_ref must be odd so that 0 is on the grid.")

    b_len = np.asarray(b_len, float)
    if b_len.size != 3 or np.any(b_len <= 0):
        raise ValueError("b_len must contain three positive numbers.")

    # ---- 1. choose n_i so that |b_i|/(n_i – 1) ≈ const -------------------
    b_min = b_len.min()
    n_axis = []
    for L in b_len:
        n = int(round(n_ref * L / b_min))    # proportional to length
        if n % 2 == 0:                       # force odd → includes 0
            n += 1
        n_axis.append(n)

    # ---- 2. build the full tensor-product grid ---------------------------
    ax = []
    for n in n_axis:
        if endpoint:                         # closed grid: ±0.5 both included
            ax.append(np.linspace(-0.5, 0.5, n, endpoint=True, dtype=float))
        else:                                # half-open: +0.5 excluded, 0 centred
            k  = n // 2                      # n = 2·k + 1  (odd!)
            ax.append(np.arange(-k, k + 1, dtype=float) / n)  # step = 1/n
    full = np.array(np.meshgrid(*ax, indexing="ij")).reshape(3, -1).T  # (N, 3)

    # ---- 3. inversion-symmetry reduction ---------------------------------
    keep = np.zeros(full.shape[0], dtype=bool)
    for i, (x, y, z) in enumerate(full):
        # Γ is always kept
        if abs(x) < tol and abs(y) < tol and abs(z) < tol:
            keep[i] = True
            continue
        # first non-zero component decides the (+) half
        if   x >  tol: keep[i] = True
        elif x < -tol: continue
        elif y >  tol: keep[i] = True
        elif y < -tol: continue
        elif z >  tol: keep[i] = True
        # (z < -tol) → partner already kept

    q_unique = full[keep]

    # ---- 4. sort for reproducibility -------------------------------------
    idx = np.lexsort(q_unique.T[::-1])
    return q_unique[idx]


def phase_correction_eigenvectors(momenta_matrix):

    _, eigenvectors = np.linalg.eigh(momenta_matrix[2, :, :])

    momenta_matrix_x = eigenvectors.conjugate().T @ np.ascontiguousarray(momenta_matrix[0, :, :]) @ eigenvectors

    # Initialize phases of vectors with the first one = 1
    c = np.zeros(momenta_matrix.shape[1], dtype=np.complex128)
    c[0] = 1.0

    # Set Jx[i,i+1] to real negative and collect phases of vectors in c[:]
    for i in range(momenta_matrix_x.shape[1] - 1):
        if (
            np.real(momenta_matrix_x[i, i + 1]) > 1e-17
            or abs(np.imag(momenta_matrix_x[i, i + 1])) > 1e-17
        ):
            c[i + 1] = (
                momenta_matrix_x[i, i + 1] * c[i].conjugate()
            ).conjugate() / abs(momenta_matrix_x[i, i + 1])
            if (
                (momenta_matrix_x[i, i + 1] * c[i].conjugate()) * c[i + 1]
            ).real > 0.0:
                c[i + 1] = -c[i + 1]
        else:
            c[i + 1] = 1.0

    # Apply the phases for eigenvecotrs
    eigenvectors = np.ascontiguousarray(eigenvectors * c)

    return eigenvectors


def crystal_field_derivatives(dof_array, group, magnetic_field_vector, displacement_number, step, states_number):
    H_grad = []
    finite_difference_stencil = _central_finite_difference_stencil(1, displacement_number, step * A_BOHR)

    momenta_matrix0 = group[f"0/MAGNETIC_DIPOLE_MOMENTA"][:]
    hamiltonian0 = group[f"0/HAMILTONIAN_MATRIX"][:] - (momenta_matrix0[0] * magnetic_field_vector[0] + momenta_matrix0[1] * magnetic_field_vector[1] + momenta_matrix0[2] * magnetic_field_vector[2])
    E_0, U_0 = np.linalg.eigh(hamiltonian0)
    momenta_matrix0 = (U_0.conj().T[np.newaxis, :, :] @ momenta_matrix0 @ U_0[np.newaxis, :, :])[:, :states_number, :states_number]
    eigenvectors0 = phase_correction_eigenvectors(momenta_matrix0)

    for dof in dof_array:
        stencil_index = -1
        H_grad_component = np.zeros((states_number, states_number), dtype=np.complex128)
        for displacement in range(-displacement_number, displacement_number + 1):
            stencil_index += 1
            if displacement == 0:
                continue
            group_name = f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}_{displacement}"
            momenta_matrix = group[f"{group_name}/MAGNETIC_DIPOLE_MOMENTA"][:]
            hamiltonian = group[f"{group_name}/HAMILTONIAN_MATRIX"][:] - (momenta_matrix[0] * magnetic_field_vector[0] + momenta_matrix[1] * magnetic_field_vector[1] + momenta_matrix[2] * magnetic_field_vector[2])
            E, U = np.linalg.eigh(hamiltonian)
            momenta_matrix = (U.conj().T[np.newaxis, :, :] @ momenta_matrix @ U[np.newaxis, :, :])[:, :states_number, :states_number]
            eigenvectors = phase_correction_eigenvectors(momenta_matrix)
            hamiltonian_z_basis = eigenvectors.conj().T @ np.diag(E[:states_number]) @ eigenvectors
            
            H_grad_component += hamiltonian_z_basis * finite_difference_stencil[stencil_index]

        H_grad.append(eigenvectors0 @ H_grad_component @ eigenvectors0.conj().T)

    H_grad = np.asarray(H_grad, dtype=np.complex128)
    dir_idx = (dof_array[:, 0] + 2) % 3

    for l in (0, 1, 2):
        mask = dir_idx == l
        if not mask.any():
            continue
        avg = H_grad[mask].mean(axis=0)
        H_grad[mask] -= avg

    return H_grad * H_CM_1

@njit(cache=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt, freq):
    for i in range(dof_array.shape[0]):
        dof = dof_array[i]
        for j in range(normal_modes.shape[1]):
            Y_q[j] += (1 / np.sqrt(freq[j]/H_CM_1)) * H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * 1/np.sqrt(M_AU) * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))


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

def susceptibility(
    omega_grid: np.ndarray,
    Hs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    T: float,
    gamma_fwhm: float,
    get_Y_q_and_freq: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]],
    *,
    states_number: int = 0,
    include_init_corr: bool = False,
    on_step: Callable[[int, float, complex], None] | None = None,
):

    beta = 1.0 / (KB * T)

    # ── diagonalise and truncate to the requested sub-space ─────────────────
    E, U = np.linalg.eigh(Hs)
    A_e, B_e = U.conj().T @ A @ U, U.conj().T @ B @ U
    A_e = A_e[:states_number, :states_number]
    B_e = B_e[:states_number, :states_number]

    E = E[:states_number]

    rho_eq = np.exp(-beta * (E-E[0]))
    rho_eq /= rho_eq.sum()

    N  = states_number
    N2 = N * N

    rho_vec = np.zeros((N2), dtype=np.complex128)

    rho_mat = np.diag(rho_eq)

    for c in range(N):
        for d in range(N):
            rho_vec[liou(c, d, N)] += rho_mat[c,d]

    H_s = np.diag(E)

    M_L = np.zeros((N2, N2), dtype=np.complex128)
    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    if d == b:
                        M_L[liou(a,b,N), liou(c,d,N)] += H_s[a,c]
                    if a == c:
                        M_L[liou(a,b,N), liou(c,d,N)] -= H_s[d,b]

    M_rho0 = np.zeros((N2, N2), dtype=np.complex128)
    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    if d == b:
                        M_rho0[liou(a,b,N), liou(c,d,N)] += A_e[a,c]
                    if a == c:
                        M_rho0[liou(a,b,N), liou(c,d,N)] -= A_e[d,b]

    eye = np.eye(N2, dtype=np.complex128)
    chi = np.empty_like(omega_grid, dtype=np.complex128)

    # Neglect omega for now out of the loop
    M_KR  = build_KR(E, T, gamma_fwhm, get_Y_q_and_freq)
    M_PSI = np.zeros((N2, N2), dtype=np.complex128)
    M_PSI = build_M_PSI(E, T, gamma_fwhm, get_Y_q_and_freq, A_e)

    if include_init_corr:
        for Yb, wb, q_0 in get_Y_q_and_freq():
            add_rho0_bundle(M_rho0, A_e, Yb, wb, bose_occ(wb, beta), beta, E, q_0)

    M_rho0 /= np.max(np.abs(M_rho0))

    # frequency loop ---------------------------------------------------------
    for k, omega in enumerate(omega_grid):

        Xi       = 1j / H_BAR * M_L + M_KR - 1j * omega * eye
        num      = (M_rho0 + 1j * H_BAR * M_PSI) @ rho_vec
        rho_hat  = np.linalg.solve(Xi, num).reshape((N, N))
        chi[k]   = 1j / H_BAR * np.trace(B_e @ rho_hat)

        if on_step is not None:
            on_step(k, omega*1e12, chi[k])  # ω back in SI

    return chi

if __name__ == "__main__":

    # ── USER-CONFIGURABLE SWEEP LISTS & PARAMETERS ──────────────────────────
    npoints_list    = [5]
    gamma_fwhm_list = [3]          # FWHM in cm-1
    T_list          = [2.5,2.55,2.6,2.7,2.8,2.9,3,3.2,3.5,4,4.5,5,5.5,6]           # Kelvin 
    B_list          = [0.01]            # Tesla
    states_number   = 6                    # electronic sub-space size
    modes_mult      = 1.1
    mode_threshold  = 1e-30
    modes_low       = 3    #cm-1
    modes_high      = 1200 #cm-1
    secular_tolerance = 1e-9    
    # ────────────────────────────────────────────────────────────────────────

    # one-shot data that never changes over the sweep -----------------------
    lanthanide          = "Yb"
    orca_fragovl_path   = "/home/mikolaj/orca_6_0_1_avx2/orca_fragovl"
    dirpath             = f"/home/mikolaj/Data/Displacements_small/{lanthanide}Co_displ" # "/home/mikolaj/Data/Displacements_cluster/CeCo_displ_cluster"
    slt_filepath        = "./seminarium/import.slt"
    group_name          = "xxx"
    displacement_number = 1
    step                = 0.025
    omega_SI            = np.logspace(0.0001, 6, 500)
    omega_au            = np.logspace(0, 6, 110)

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
    Dy["YCo_supercell"].replace_atoms([0], [lanthanide])
    hessian = Dy["YCo_supercell"].hessian_from_finite_displacements(
        "./seminarium/YCo_supercell_from_cell",
        "CP2K", "YCo_hessian",
        1, 0.01, born_charges=True
    )

    # Dy = slt.unit_cell("/home/mikolaj/InputOutput_CP2K/LaCu_pymon_no_symm/LaCu_pymon_opt.xyz", "./seminarium/Dy.slt", "guju", [[1.2795564780516631E+001, 0.0000000000000000E+000, 0.0000000000000000E+000],[0.0000000000000000E+000,  1.0688348682773828E+001, 0.0000000000000000E+000],[-8.0938444693724920E-001,  0.0000000000000000E+000,  6.2893542237840405E+000]])
    # Dy["guju"].supercell(2,2,3,"slt", slt_group_name="guju_super")
    # hessian = Dy["guju_super"].hessian_from_finite_displacements("/home/mikolaj/SpinDynamics/Phonons/LaCu_pymon_no_symm", "CP2K", "guju_hessian", 1, 0.01, born_charges=True)

    slt_hessian     = SltHessian(hessian)
    masses_inv_sqrt = slt_hessian._masses_inv_sqrt
    recip_axes = slt_hessian.atoms_object().cell.reciprocal().cellpar()[:3]
    hess_obj        = Hessian(slt_hessian.hessian()[:], np.outer(masses_inv_sqrt, masses_inv_sqrt), np.array([0., 0., 0.]))
    
    plt.ion()       # turn interactive mode on once

    def make_step_plotter(ax_re, ax_im, label):
        # fresh empty lines, one per curve
        line_re, = ax_re.plot([], [], marker='o', markersize=0.7, lw=1, label=label)
        line_im, = ax_im.plot([], [], marker='o', markersize=0.7, lw=1, label=label)

        # pre-allocate containers (faster than re-fetching get_xdata each step)
        xs, ys_re, ys_im = [], [], []

        def _update(k, omega_si, chi_k):
            if np.abs(chi_k.imag) < np.inf:
                xs.append(omega_si)
                ys_re.append(np.abs(chi_k.real))
                ys_im.append(np.abs(chi_k.imag))

                line_re.set_data(xs, ys_re)
                line_im.set_data(xs, ys_im)

                # keep autoscaling inexpensive
                for ax in (ax_re, ax_im):
                    ax.relim()
                    ax.autoscale_view()
                plt.pause(0.001)   # let the GUI breathe

        return _update
        
    # ---------------------------------------------------------------------- #
    #  PLOTTING CANVAS (two sub-plots: Re and Im)                            #
    # ---------------------------------------------------------------------- #
    fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax in (ax_re, ax_im):
        ax.set_xscale("log")
        ax.set_xlabel(r"ω  (rad s$^{-1}$)")
        ax.grid(True, which="both", ls=":")
    ax_re.set_ylabel(r"Re χ(ω)")
    ax_im.set_ylabel(r"Im χ(ω)")

    for npoints in npoints_list:
        for B in B_list:

            orient = np.array([1, 1, 1], np.float64) 
            orient /= (np.linalg.norm(orient) * B_AU_T)
            B_vec  = B * orient

            grid = half_bz_grid_aniso(recip_axes, npoints)

            with h5py.File(slt_filepath, "r") as f:
                grp = f[group_name]
                dof_array = dofs_with_complete_displacements(grp, displacement_number)
                magnetic_momenta = grp["0/MAGNETIC_DIPOLE_MOMENTA"][:]
                AB = (magnetic_momenta[0] * orient[0] + magnetic_momenta[1] * orient[1] + magnetic_momenta[2] * orient[2])
                H_total = (grp["0/HAMILTONIAN_MATRIX"][:] - (magnetic_momenta[0] * B_vec[0] + magnetic_momenta[1] * B_vec[1] + magnetic_momenta[2] * B_vec[2])) * H_CM_1

                H_grad = crystal_field_derivatives(dof_array, grp, B_vec, 1, step, states_number)

            def get_Y_q_and_freq():
                n_k_inv = 1.0 / np.sqrt(len(grid))
                for q in grid:
                    q_0 = np.allclose(q, np.asarray([0.0, 0.0, 0.0]), atol=0.000001)
                    hess_obj.kpoint = q
                    freq, modes = hess_obj.frequencies_eigenvectors

                    # --- keep only the requested number of modes --------------------
                    freq  *= AU_BOHR_CM_1

                    mask = (freq >= modes_low) & (freq <= modes_high)
                    idx  = np.where(mask)[0]

                    Y_q = np.zeros((freq.size, states_number, states_number), dtype=np.complex128)

                    get_Y_q(Y_q, H_grad, modes, q, dof_array, masses_inv_sqrt, n_k_inv, freq)

                    # for index, i in enumerate(np.ascontiguousarray(Y_q[idx], dtype=np.complex128)):
                    #     print(index, np.trace(i@i.conj().T))

                    yield np.ascontiguousarray(Y_q[idx], dtype=np.complex128), np.ascontiguousarray(freq[idx], dtype=np.float64), q_0


            for gamma_fwhm, T in itertools.product(
                gamma_fwhm_list, T_list):
                    label = (f"np={npoints}, γ={gamma_fwhm:.0e}, "
                            f"T={T:g} K, B={B:g} T")
                    step_plotter = make_step_plotter(ax_re, ax_im, label)

                    chi = susceptibility(
                        omega_au/1e12, H_total, AB, AB,
                        T, gamma_fwhm, get_Y_q_and_freq,
                        states_number=states_number,
                        include_init_corr=True,
                        on_step=step_plotter            # ← live updates happen here
                    )
 

                    # # plotting ------------------------------------------------------
                    # label = (f"np={npoints}, γ={gamma_fwhm:.0e}, "
                    #          f"T={T:g} K, B={B:g} T")
                    # ax_re.plot(omega_SI, chi.real, label=label)
                    # ax_im.plot(omega_SI, chi.imag, label=label)

    # figure cosmetics -------------------------------------------------------
    ax_re.legend(fontsize="x-small", frameon=False, ncols=2)
    ax_im.legend(fontsize="x-small", frameon=False, ncols=2)
    plt.ioff()                   # stop live updates
    plt.show()
              
# R21 = build_R21(Ener, T, lw, smear, gen, sec_tol=sec_tol)