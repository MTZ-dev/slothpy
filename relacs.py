from __future__ import annotations

import os
import re
import posixpath
import math
from typing import Callable, Iterable, Tuple

import numpy as np
import h5py
import numpy as np
from numba import njit, prange

import slothpy as slt
from slothpy._general_utilities._constants import A_BOHR, H_CM_1, B_AU_T
from slothpy._general_utilities._io import _hamiltonian_derivatives_from_dir_to_slt
from slothpy._general_utilities._lapack import _zdot3d
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil
from slothpy.core._slt_file import SltHessian
from slothpy.core._hessian_object import Hessian

import matplotlib.pyplot as plt


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

    ax.plot(freq, chi.real, label=lab_re)
    ax.plot(freq, chi.imag, "--", label=lab_im)

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
@njit(cache=True)
def bose(omega: float, beta: float) -> float:
    x = beta * 1 * omega
    if x > 700.0:
        return 0.0
    return 1.0 / (math.exp(x) - 1.0)

@njit(cache=True)
def zeta(x: float, beta: float) -> float:
    print(x, beta, x * beta)
    a = beta if abs(x) < 1e-14 else (math.exp(1 * x * beta) - 1.0) / (1 * x)
    return a

@njit(cache=True)
def Iint(w1: float, w2: float, beta: float) -> float:
    eps = 1e-14
    if abs(w1) < eps and abs(w2) < eps:
        return 0.5 * beta * beta
    if abs(w2) < eps:
        return beta * math.exp(1 * w1 * beta) / (1 * w1) - (
            math.exp(1 * w1 * beta) - 1.0) / (1**2 * w1**2)
    if abs(w1) < eps:
        return (math.exp(1 * w2 * beta) - 1.0) / (1**2 * w2**2) - beta / (1 * w2)
    if abs(w1 + w2) < eps:
        return -1.0 / (1 * w1) * (beta - (math.exp(1 * w1 * beta) - 1.0) / (1 * w1))
    return (
        (math.exp(1 * (w1 + w2) * beta) - 1.0) / (1**2 * w2 * (w1 + w2))
        - (math.exp(1 * w1 * beta) - 1.0) / (1**2 * w1 * w2)
    )

@njit(cache=True)
def lor_pref(x, d):
    return d / (x * x + d * d)

@njit(cache=True)
def lor_hilb(x, d):
    return -x / (x * x + d * d)

@njit(cache=True)
def Jhat(omega, w_ab, wq, n_q, d):
    x1 = omega - (w_ab - wq)
    x2 = omega - (w_ab + wq)
    L1 = lor_pref(x1, d) + 1j * lor_hilb(x1, d)
    L2 = lor_pref(x2, d) + 1j * lor_hilb(x2, d)
    return 0.5 * 1 / wq * (n_q * L1 + (n_q + 1.0) * L2)

@njit(cache=True)
def Jcorr(omega, omega_p, w_ab, wq, n_q, d, beta):
    z1 = zeta(w_ab + wq, beta)
    z2 = zeta(w_ab - wq, beta)
    x1 = omega - omega_p - wq
    x2 = omega - omega_p + wq
    L1 = lor_pref(x1, d) + 1j * lor_hilb(x1, d)
    L2 = lor_pref(x2, d) + 1j * lor_hilb(x2, d)
    return 0.5 * 1 / wq * (n_q * z1 * L1 + (n_q + 1.0) * z2 * L2)

@njit(cache=True)
def liou(a, b, N):
    return a * N + b

# -------------------- add matrices per (q,J‑bundle) -------------------------
@njit(cache=True)
def add_KR_bundle(out, omega, Yb, wb, nb, d, w_n):
    N = w_n.size; N2 = N * N; J = wb.size
    for j in range(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh = np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab = liou(a,b,N)
                for c in range(N):
                    for d_ in range(N):
                        cd = liou(c,d_,N)
                        val = 0.0+0.0j
                        if d_==b:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat(omega,w_n[e]-w_n[d_],wq,n_q,d)*Y[a,e]*Yh[e,c]
                            val+=tmp
                        val-=Jhat(omega,w_n[a]-w_n[d_],wq,n_q,d)*Y[a,c]*Yh[d_,b]
                        if a==c:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat(-omega,w_n[e]-w_n[c],wq,n_q,d)*Y[d_,e]*Yh[e,b]
                            val+=tmp
                        val-=Jhat(-omega,w_n[b]-w_n[c],wq,n_q,d)*Y[a,c]*Yh[d_,b]
                        out[ab,cd]+=val

@njit(cache=True)
def add_PSI_bundle(out, omega, A, Yb, wb, nb, d, beta, w_n):
    N=w_n.size; N2=N*N; J=wb.size
    for j in range(J):
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

@njit(cache=True)
def add_rho0_bundle(out, A, Yb, wb, nb, beta, w_n):
    """Second‑order Δρ_S term with correct ½ħ/ω_q prefactor."""
    N=w_n.size; J=wb.size
    for j in range(J):
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

# ---------------------------------------------------- main -------------------
def susceptibility(
    omega_grid: np.ndarray,
    Hs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    T: float,
    gamma_fwhm: float,
    get_Y_q_and_freq: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]],
    *,
    include_init_corr: bool = True,
):
    beta = 1.0/(KB*T)
    E,U=np.linalg.eigh(Hs)
    w_n=E/1
    A_e,B_e=U.conj().T@A@U, U.conj().T@B@U
    E_shift = E - E.min()            # shift to avoid overflow
    rho_eq  = np.exp(-beta * E_shift)
    rho_eq /= rho_eq.sum()
    rho_vec = np.diag(rho_eq).flatten()
    N=Hs.shape[0]; N2=N*N
    d=0.5*gamma_fwhm

    # Liouvillian
    M_L=np.kron(np.diag(w_n),np.eye(N))-np.kron(np.eye(N),np.diag(w_n))

    # initial matrix
    M_rho0=np.kron(A_e,np.eye(N))-np.kron(np.eye(N),A_e)
    if include_init_corr:
        for Ybundle, wbundle in get_Y_q_and_freq():
            nbundle=np.array([bose(w,beta) for w in wbundle])
            add_rho0_bundle(M_rho0,A_e,Ybundle,wbundle,nbundle,beta,w_n)

    eye=np.eye(N2,dtype=np.complex128)
    chi=np.empty_like(omega_grid,dtype=np.complex128)

    for k,omega in enumerate(omega_grid):
        M_KR=np.zeros((N2,N2),dtype=np.complex128)
        M_PSI=np.zeros((N2,N2),dtype=np.complex128)
        for Ybundle, wbundle in get_Y_q_and_freq():
            nbundle=np.array([bose(w,beta) for w in wbundle])
            add_KR_bundle(M_KR,omega,Ybundle,wbundle,nbundle,d,w_n)
            add_PSI_bundle(M_PSI,omega,A_e,Ybundle,wbundle,nbundle,d,beta,w_n)
        num=(M_rho0+1j/1*M_PSI)@rho_vec
        Xi=1j/1*M_L+1.0/(1**2)*M_KR-1j*omega*eye
        rho_hat=np.linalg.solve(Xi,num).reshape((N,N))
        chi[k]=1j/1*np.trace(B_e@rho_hat)
    return chi


@njit(cache=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt):
    for i in range(dof_array.shape[0]):
        dof = dof_array[i]
        for j in range(normal_modes.shape[1]):
            Y_q[j] +=  H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))


if __name__ == "__main__":

    orca_fragovl_path = "/home/mikolaj/orca_6_0_1_avx2/orca_fragovl"
    dirpath = "/home/mikolaj/Data/Displacements_small/CeCo_displ"
    slt_filepath = "./seminarium/import.slt"
    group_name = "xxx"
    displacement_number = 1
    step = 0.025
    kpoints_grid_number = 1

    os.remove(slt_filepath)
    slt.set_default_error_reporting_mode()

    _hamiltonian_derivatives_from_dir_to_slt(dirpath, slt_filepath, group_name, displacement_number, step, 64, 1, "ORCA", False, False, False, orca_fragovl_path)
    
    magnetic_field = 0.1
    orientation = np.asarray([1,1,1], dtype=np.float64)
    magnetic_field_vector = magnetic_field / B_AU_T * orientation / np.linalg.vector_norm(orientation)

    with h5py.File(slt_filepath, "r") as file:
        group = file[group_name]
        dof_array = dofs_with_complete_displacements(group, displacement_number)
        hamiltonian = group["0/HAMILTONIAN_MATRIX"][:]
        AB = _zdot3d(group["0/MAGNETIC_DIPOLE_MOMENTA"][:], orientation)

        hamiltonian_total = group["0/HAMILTONIAN_MATRIX"][:]+_zdot3d(group["0/MAGNETIC_DIPOLE_MOMENTA"][:], -magnetic_field_vector)
        E, U_R0 = np.linalg.eigh(hamiltonian_total)

        k_mch_array = k_mch(dof_array, group, U_R0)
        E_grad, k_U_array = E_grad_k_U(dof_array, group, U_R0, magnetic_field_vector, displacement_number, 1e-8)
        
        H_grad = np.empty_like(k_mch_array)

        for i in range(H_grad.shape[0]):
            E_k_mch_k_U = (E[None, :]-E[:, None])*(k_mch_array[i] + k_U_array[i])
            np.fill_diagonal(E_k_mch_k_U, E_grad[i])
            H_grad[i] = E_k_mch_k_U
        
    Dy = slt.supercell("./seminarium/YCo_supercell_from_cell/dof_0_disp_0.xyz", slt_filepath, "YCo_supercell", 3, 3, 2, supercell_params=[22.663134149075237, 22.663134149075233, 25.14851428466812, 90.0, 90.0, 120.00000000000001], multiplicity=1)
    hessian = Dy["YCo_supercell"].hessian_from_finite_displacements("./seminarium/YCo_supercell_from_cell", "CP2K", "YCo_hessian", 1, 0.01, born_charges=True)

    slt_hessian = SltHessian(hessian)
    masses_inv_sqrt = slt_hessian._masses_inv_sqrt
    hessian_object = Hessian(slt_hessian.hessian()[:], np.outer(masses_inv_sqrt, masses_inv_sqrt), np.asarray([0.,0.,0.], dtype=np.float64))

    q = np.linspace(-1, 1, kpoints_grid_number, endpoint=False)
    q_grid = np.meshgrid(q, q, q, indexing='ij')
    kpoints_grid = np.ascontiguousarray(np.vstack([grid.ravel() for grid in q_grid]).T)

    def get_Y_q_and_freq():
        number_of_kpoints_inv_sqrt = 1 / np.sqrt(kpoints_grid.shape[0])

        for q in kpoints_grid:
            hessian_object.kpoint = q
            freq, normal_modes = hessian_object.frequencies_eigenvectors
            indicies = np.where(freq <= 0)[0]
            if indicies.size != 0:
                freq_start_index = np.max(indicies) + 1
            else:
                freq_start_index = 0

            Y_q = np.zeros((freq.shape[0], H_grad.shape[1], H_grad.shape[2]), dtype=np.complex128)
            get_Y_q(Y_q, H_grad, normal_modes, q, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt)

            yield Y_q[freq_start_index:], freq[freq_start_index:]


    omega = np.logspace(1, 8, 20)
    omega_au = omega_to_au(omega)

    chi = susceptibility(omega, hamiltonian_total, AB, AB, 8, 1e-5, get_Y_q_and_freq)
    print(chi)

    plot_susceptibility(omega, chi)