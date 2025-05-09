from __future__ import annotations

import os
import re
import posixpath
import math
import itertools
from typing import Callable, Iterable, Tuple

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


from typing import Tuple, Sequence

def half_bz_grid_aniso(
    b_len: Sequence[float],
    n_ref: int,
    *,
    endpoint: bool = True,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Anisotropic first-BZ mesh with equal point density in Cartesian q-space.

    Parameters
    ----------
    b_len    : (3,) sequence
        Lengths |b1|, |b2|, |b3| of the reciprocal-lattice vectors.
    n_ref    : int (odd)
        Number of points along the **shortest** axis in the *full* mesh.
        Must be odd so that 0 is on every axis.
    endpoint : bool, optional
        If True, +0.5 is included on each axis (default False – half-open).
    tol      : float, optional
        Tolerance for zero tests / symmetry checks.

    Returns
    -------
    q : (M, 3) ndarray
        Unique q-points (fractional coordinates), one per {+q, −q}, Γ included.
    """
    if n_ref % 2 == 0:
        raise ValueError("n_ref must be odd so that 0 is on the grid.")

    b_len = np.asarray(b_len, float)
    if b_len.size != 3 or np.any(b_len <= 0):
        raise ValueError("b_len must contain three positive numbers.")

    # --- 1. choose n_i so that |b_i|/(n_i-1) ≈ const -----------------------
    b_min = b_len.min()
    n_axis = []
    for L in b_len:
        n = int(round(n_ref * L / b_min))         # proportional to length
        if n % 2 == 0:                            # force odd → includes 0
            n += 1
        n_axis.append(n)

    # --- 2. build the full tensor product grid ----------------------------
    ax = [np.linspace(-0.5, 0.5, n, endpoint=endpoint, dtype=float)
          for n in n_axis]
    full = np.array(np.meshgrid(*ax, indexing="ij")).reshape(3, -1).T  # (N,3)

    # --- 3. inversion-symmetry reduction ----------------------------------
    keep = np.zeros(full.shape[0], bool)
    for i, (x, y, z) in enumerate(full):
        # Γ always in
        if abs(x) < tol and abs(y) < tol and abs(z) < tol:
            keep[i] = True
            continue
        # first non-zero component decides
        if   x >  tol: keep[i] = True
        elif x < -tol: continue
        elif y >  tol: keep[i] = True
        elif y < -tol: continue
        elif z >  tol: keep[i] = True
        # (z < −tol) -> partner already kept

    q_unique = full[keep]

    # --- 4. sorted for reproducibility ------------------------------------
    idx = np.lexsort(q_unique.T[::-1])
    return q_unique[idx]


def plot_susceptibility(freq, chi, *, ax=None, label=None):
    """
    Semilog-x plot of the complex susceptibility.

    Parameters
    ----------
    freq : 1-D array
        Positive frequency grid (rad s⁻¹).  
    chi  : 1-D complex array
        χ(ω) evaluated on the same grid.
    ax   : matplotlib.axes.Axes, optional
        Existing axis – if None, a new figure is created.
    label : str, optional
        Suffix for legend labels (useful when comparing curves).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xscale("log")

    lab_re = f"Re χ{f' ({label})' if label else ''}"
    lab_im = f"Im χ{f' ({label})' if label else ''}"

    ax.plot(freq, chi.real, "-", label=lab_re)
    ax.plot(freq, chi.imag, "-", label=lab_im)

    ax.set_xlabel(r"ω  (rad s$^{-1}$)")
    ax.set_ylabel(r"χ(ω)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    plt.show()

    return ax


def plot_complex_matrix(
    M: np.ndarray,
    *,
    title: str = "Complex Matrix",
    cmap: str = "bwr",
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (10, 4)
):
    """
    Plot real and imaginary parts of a 2D complex matrix as heatmaps.

    Parameters
    ----------
    M : (N,N) ndarray
        Complex matrix to visualize.
    title : str, optional
        Global title for the figure.
    cmap : str, optional
        Colormap to use for heatmaps (default 'bwr' — blue-white-red).
    vmin, vmax : float, optional
        Fixed limits for color scaling; if None, autoscale separately for Re and Im.
    figsize : (width, height) tuple, optional
        Size of the figure.
    """
    if M.ndim != 2 or not np.iscomplexobj(M):
        raise ValueError("Input must be a 2D complex matrix.")

    real_part = M.real
    imag_part = M.imag

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    im0 = axes[0].imshow(real_part, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Real part")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(imag_part, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Imaginary part")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    for ax in axes:
        ax.set_xlabel('j')
        ax.set_ylabel('i')

    plt.tight_layout()
    plt.show()

AU_ANGULAR_FREQUENCY = 4.13413733359e16                       

def omega_to_au(omega_si):
    return np.asarray(omega_si) / AU_ANGULAR_FREQUENCY


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

KB   = 3.166811563e-6

# -------------------------------------------------- scalar helpers ----------
# --------------------------- numerical guards --------------------------------
MAX_EXP  = 300.0   # absolute overflow guard (double precision)
DROP_EXP = 36    # exp(36)  ≈  1e16  → treat as numerically zero

# --------------------------- scalar helpers ----------------------------------
@njit(cache=True)
def bose(omega: float, beta: float) -> float:
    """Bose occupation with overflow protection."""
    u = beta * 1 * omega
    if u > DROP_EXP:
        # print("DROP", u)
        return 0.0              # e^{u} huge → n_B ≈ 0
    # if u < -DROP_EXP:
    #     return -1.0             # exp(u)≈0 ⇒ expm1(u)≈-1
    return 1.0 / math.expm1(u)

@njit(cache=True)
def zeta(x: float, beta: float) -> float:
    """ζ(x) = (e^{βħx}−1)/(ħx) with drop‑rule."""
    eps = 1e-14
    if abs(x) < eps:
        return beta
    u = beta * 1 * x
    if u > DROP_EXP:
        # print("DROP", u)            # divergent, partner factor will be ~0 → drop
        return 0.0
    if u < -DROP_EXP:           # e^{u}≈0 ⇒ ζ≈−1/(ħx)
        return -1.0 / (1 * x)
    return math.expm1(u) / (1 * x)

@njit(cache=True)
def Iint(w1: float, w2: float, beta: float) -> float:
    """Imaginary‑time double integral with exponent dropping."""
    eps = 1e-14
    if abs(w1) < eps and abs(w2) < eps:
        return 0.5 * beta * beta

    def expm1_drop(u):
        # if u > DROP_EXP:
        #     # print("DROP", u)
        #     return 0.0          # treat large exponent as cancelled
        # if u < -DROP_EXP:
        #     return -1.0
        return math.expm1(u)

    if abs(w2) < eps:
        u = 1 * w1 * beta
        num = 0.0 if u > DROP_EXP else math.exp(u)
        return beta * num / (1 * w1) - expm1_drop(u) / (1**2 * w1**2)
    if abs(w1) < eps:
        u = 1 * w2 * beta
        return expm1_drop(u) / (1**2 * w2**2) - beta / (1 * w2)
    if abs(w1 + w2) < eps:
        u = 1 * w1 * beta
        return -(beta - expm1_drop(u) / (1 * w1)) / (1 * w1)

    u1  = 1 * w1 * beta
    u12 = 1 * (w1 + w2) * beta
    term1 = expm1_drop(u12) / (1**2 * w2 * (w1 + w2))
    term2 = expm1_drop(u1)  / (1**2 * w1 * w2)
    return term1 - term2

@njit(cache=True)
def lor_pref(x, d):
    return d / (x * x + d * d)

@njit(cache=True)
def lor_hilb(x, d):
    return x / (x * x + d * d)


@njit(cache=True)
def Jhat(omega, w_ab, wq, n_q, d):
    """Lorentz–broadened spectral density with **sign‑filtered evaluation**.

    * |w_ab| ≃ 0  → 0 (no transition).
    * w_ab > 0    → absorption only  (phonon taken from bath).
    * w_ab < 0    → emission   only  (phonon emitted).
    The unused branch is not evaluated, so no extra lorentzian is formed.
    """
    if np.abs(w_ab) < 1e-8:
        return 0.0 + 0.0j

    if w_ab > 0.0:                     # absorption (n_q term)
        x = omega - (w_ab - wq)
        if (w_ab - wq) < 0:
            return 0.0 + 0.0j
        L = lor_pref(x, d) + 1j * lor_hilb(x, d)
        return 0.5 / wq * (n_q * L)
    else:                              # emission ((n_q+1) term)
        x = omega - (w_ab + wq)
        if (w_ab + wq) < 0:
            return 0.0 + 0.0j
        L = lor_pref(x, d) + 1j * lor_hilb(x, d)
        return 0.5 / wq * ((n_q + 1.0) * L)

# @njit(cache=True)
# def Jhat(omega, w_ab, wq, n_q, d):
#     x1 = omega - (w_ab - wq)
#     x2 = omega - (w_ab + wq)
#     L1 = lor_pref(x1, d) - 1j * lor_hilb(x1, d)
#     L2 = lor_pref(x2, d) - 1j * lor_hilb(x2, d)
#     return 0.5 * 1 / wq * (n_q * L1 + (n_q + 1.0) * L2)

@njit(cache=True)
def Jcorr(omega, omega_p, w_ab, wq, n_q, d, beta):
    z1 = zeta(w_ab + wq, beta)
    z2 = zeta(w_ab - wq, beta)
    x1 = omega - omega_p - wq
    x2 = omega - omega_p + wq
    L1 = lor_pref(x1, d) + 1j * lor_hilb(x1, d)
    L2 = lor_pref(x2, d) + 1j * lor_hilb(x2, d)
    return 0.5 * 1 / wq * (n_q * z1 * L1 + (n_q + 1.0) * z2 * L2)

    # if np.abs(w_ab) < 1e-8:
    #     return 0.0 + 0.0j

    # if w_ab > 0.0:                     # absorption (n_q term)
    #     x = omega - (w_ab - wq)
    #     if (w_ab - wq) < 0:
    #         return 0.0 + 0.0j
    #     L = lor_pref(x, d) + 1j * lor_hilb(x, d)
    #     return 0.5 / wq * (n_q * L)
    # else:                        # emission ((n_q+1) term)
    #     x = omega - (w_ab + wq)
    #     if (w_ab + wq) < 0:
    #         return 0.0 + 0.0j
    #     L = lor_pref(x, d) + 1j * lor_hilb(x, d)
    #     return 0.5 / wq * ((n_q + 1.0) * L)

@njit(cache=True)
def liou(a, b, N):
    return a * N + b

# -------------------- add matrices per (q,J‑bundle) -------------------------
@njit(cache=True, parallel=True)
def add_KR_bundle(out, omega, Yb, wb, nb, delta, w_n):
    N = w_n.size; N2 = N * N; J = wb.size
    for j in prange(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh = np.conjugate(Y)
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
                                tmp+=Jhat(omega,w_n[e]-w_n[d],wq,n_q,delta)*Y[a,e]*Yh[e,c]
                            val+=tmp
                        val-=Jhat(omega,w_n[a]-w_n[d],wq,n_q,delta)*Y[a,c]*Yh[d,b]
                        if a==c:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat(omega,w_n[c]-w_n[e],-wq,n_q,delta)*Y[d,e]*Yh[e,b]
                            val+=tmp
                        val-=Jhat(omega,w_n[c]-w_n[b],-wq,n_q,delta)*Y[a,c]*Yh[d,b]
                        out[ab,cd]+=val

@njit(cache=True, parallel=True)
def add_PSI_bundle(out, omega, A, Yb, wb, nb, d, beta, w_n):
    N=w_n.size; N2=N*N; J=wb.size
    for j in prange(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh=np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    for d_ in range(N):
                        cd=liou(c,d_,N)
                        val=0.0+0.0j
                        for e in range(N):
                            w_ed=w_n[e]-w_n[d_]
                            val+=Jcorr(omega,w_ed,w_n[d_]-w_n[b],wq,n_q,d,beta)*Y[a,e]*A[e,c]*Yh[d_,b]
                            val-=Jcorr(omega,w_ed,w_n[a]-w_n[d_],wq,n_q,d,beta)*A[a,c]*Y[d_,e]*Yh[e,b]
                        if a==c:
                            for e in range(N):
                                w_ed=w_n[e]-w_n[d_]
                                for f in range(N):
                                    om_p=w_n[c]-w_n[d_]+w_n[e]-w_n[f]
                                    val+=Jcorr(omega,w_ed,om_p,wq,n_q,d,beta)*Y[d_,e]*A[e,f]*Yh[f,b]
                        for e in range(N):
                            w_ed=w_n[e]-w_n[d_]
                            om_p=w_n[c]-w_n[d_]+w_n[e]-w_n[b]
                            val-=Jcorr(omega,w_ed,om_p,wq,n_q,d,beta)*Y[a,c]*Yh[d_,e]*A[e,b]
                        out[ab,cd]+=val

@njit(cache=True, parallel=True)
def add_rho0_bundle(out, A, Yb, wb, nb, beta, w_n):
    """Second‑order Δρ_S term with correct ½ħ/ω_q prefactor."""
    N=w_n.size; J=wb.size
    for j in prange(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        coeff = -1 / (2.0 * wq)
        Yh=np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    for d_ in range(N):
                        cd=liou(c,d_,N)
                        corr=0.0+0.0j
                        for e in range(N):
                            w_de=w_n[d_]-w_n[e]
                            corr+=(n_q*Iint(w_de+wq,w_n[e]-w_n[b]-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_n[e]-w_n[b]+wq,beta))*A[a,c]*Y[d_,e]*Yh[e,b]
                            for f in range(N):
                                corr-=(n_q*Iint(w_de+wq,w_n[e]-w_n[f]-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_n[e]-w_n[f]+wq,beta))*(1.0 if a==c else 0.0)*Y[d_,e]*Yh[e,f]*A[f,b]
                        out[ab,cd]+=coeff*corr

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
    include_init_corr: bool = True,
    on_step: Callable[[int, float, complex], None] | None = None,
):

    beta = 1.0 / (KB * T)

    # ── diagonalise and truncate to the requested sub-space ─────────────────
    E, U = np.linalg.eigh(Hs)
    A_e, B_e = U.conj().T @ A @ U, U.conj().T @ B @ U
    A_e = A_e[:states_number, :states_number]
    B_e = B_e[:states_number, :states_number]

    E_shift = E[:states_number] - E.min()
    w_n = E_shift

    rho_eq = np.exp(-beta * E_shift)
    rho_eq /= rho_eq.sum()
    rho_eq[rho_eq < 1e-12] = 0
    rho_vec = np.diag(rho_eq).flatten()

    N  = states_number
    N2 = N * N
    d  = 0.5 * gamma_fwhm

    # Get omega ####################################################

    # omega_grid = []
    # row, cols = np.tril_indices(N, k=-1)

    # E_diff = (w_n[:, np.newaxis] - w_n[np.newaxis, :])[row, cols].flatten()
    # for _, wb in get_Y_q_and_freq():
    #     for w in wb:
    #         diff = E_diff - w
    #         print(diff)
    #         omega_grid.append(np.abs(diff))

    # omega_grid = np.concatenate(omega_grid)
    # print(omega_grid.shape)
    
    # etol = gamma_fwhm  # cluster distance tolerance

    # # Sort array for easier clustering
    # arr_sorted = np.sort(omega_grid)

    # # Start clustering
    # clusters = []
    # i = 0
    # n = len(arr_sorted)
    # while i < n:
    #     start = arr_sorted[i]
    #     j = i
    #     # grow the cluster while distance to the start point ≤ etol
    #     while j < n and arr_sorted[j] - start <= etol:
    #         j += 1
    #     clusters.append(arr_sorted[i:j])
    #     i = j

    # omega_grid = np.array([np.mean(c) for c in clusters])

    # print(omega_grid.shape)

    ####################################################################

    # Liouvillian and “initial” matrices ------------------------------------
    M_L   = np.kron(np.diag(w_n), np.eye(N)) - np.kron(np.eye(N), np.diag(w_n))
    M_rho0 = np.kron(A_e, np.eye(N)) - np.kron(np.eye(N), A_e)

    if include_init_corr:
        for Yb, wb in get_Y_q_and_freq():
            nb = np.array([bose(w, beta) for w in wb])
            add_rho0_bundle(M_rho0, A_e, Yb, wb, nb, beta, w_n)

    eye = np.eye(N2, dtype=np.complex128)
    chi = np.empty_like(omega_grid, dtype=np.complex128)

    # frequency loop ---------------------------------------------------------
    for k, omega in enumerate(omega_grid):
        print(k)
        M_KR  = np.zeros((N2, N2), dtype=np.complex128)
        M_PSI = np.zeros((N2, N2), dtype=np.complex128)

        for Yb, wb in get_Y_q_and_freq():
            nb = np.array([bose(w, beta) for w in wb])
            add_KR_bundle(M_KR, omega, Yb, wb, nb, d, w_n)
            add_PSI_bundle(M_PSI, omega, A_e, Yb, wb, nb, d, beta, w_n)

        Xi       = 1j * M_L + 2* M_KR.real - 1j * omega * eye
        num      = (M_rho0 + M_PSI) @ rho_vec
        rho_hat  = np.linalg.solve(Xi, num).reshape((N, N))
        chi[k]   = 1j * np.trace(B_e @ rho_hat)

        if on_step is not None:
            on_step(k, omega, chi[k])  # ω back in SI

    return chi



@njit(cache=True, parallel=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt):
    for i in range(dof_array.shape[0]):
        dof = dof_array[i]
        for j in prange(normal_modes.shape[1]):
            Y_q[j] +=  H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))


# --------------------------------------------------------------------------- #
#  MAIN — parameter sweep + multi-curve plotting                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # ── USER-CONFIGURABLE SWEEP LISTS & PARAMETERS ──────────────────────────
    npoints_list    = [3,5,7,9,11]
    gamma_fwhm_list = [3e-5]          # FWHM in a.u.
    T_list          = [20.0]           # Kelvin
    B_list          = [0.1]            # Tesla
    states_number   = 4                    # electronic sub-space size
    modes_number    = 30                    # phonon modes per k-point
    # ────────────────────────────────────────────────────────────────────────

    import itertools

    # one-shot data that never changes over the sweep -----------------------
    orca_fragovl_path   = "/home/mikolaj/orca_6_0_1_avx2/orca_fragovl"
    dirpath             = "/home/mikolaj/Data/Displacements_small/NdCo_displ" # "/home/mikolaj/Data/Displacements_cluster/CeCo_displ_cluster"
    slt_filepath        = "./seminarium/import.slt"
    group_name          = "xxx"
    displacement_number = 1
    step                = 0.025
    omega_SI            = np.logspace(0.0001, 6, 500)
    omega_au            = np.logspace(-7, -3, 100)

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

    plt.ion()       # turn interactive mode on once

    def make_step_plotter(ax_re, ax_im, label):
        # fresh empty lines, one per curve
        line_re, = ax_re.plot([], [], marker='o', markersize=0.7, lw=1, label=label)
        line_im, = ax_im.plot([], [], marker='o', markersize=0.7, lw=1, label=label)

        # pre-allocate containers (faster than re-fetching get_xdata each step)
        xs, ys_re, ys_im = [], [], []

        def _update(k, omega_si, chi_k):
            xs.append(omega_si)
            ys_re.append(chi_k.real)
            ys_im.append(chi_k.imag)

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

    # ---------------------------------------------------------------------- #
    #  OUTER LOOP: different band-path densities (npoints)                   #
    # ---------------------------------------------------------------------- #
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

            grid = half_bz_grid_aniso(recip_axes, npoints)

            # q = np.linspace(-0.5, 0.5, npoints, endpoint=True)
            # q = np.array([0], dtype=np.float64)
            # q = np.array([-0.5, -0.25, 0, 0.25, 0.5], dtype=np.float64)
            # q = np.array([-0.5, -0.333333,-0.166666, 0, 0.166666, 0.333333, 0.5], dtype=np.float64)
            # q_grid = np.meshgrid(q, q, q, indexing='ij')
            # grid = np.ascontiguousarray(np.vstack([grid.ravel() for grid in q_grid]).T)

            # grid            = slt_hessian.atoms_object().cell.get_bravais_lattice.bandpath(
            #                 npoints=npoints
            #             ).kpts.astype(np.float64)

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
                    hess_obj.kpoint = q
                    freq, modes = hess_obj.frequencies_eigenvectors

                    # --- keep only the requested number of modes --------------------
                    freq  = freq[:modes_number] * (AU_BOHR_CM_1 / H_CM_1)
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

            step_plotter = make_step_plotter(ax_re, ax_im, label)

            chi = susceptibility(
                omega_au, H_total, AB, AB,
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