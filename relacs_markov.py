from __future__ import annotations

import numpy as np
from numpy import pi

import os
import re
import posixpath
import itertools
import csv
import ctypes
from time import perf_counter
from pathlib import Path
from typing import Sequence, Union
ArrayLike = Union[Sequence, np.ndarray]

import h5py
import scipy.special.cython_special
from numba import njit, prange, set_num_threads, get_thread_id
from numba.extending import get_cython_function_address
from threadpoolctl import threadpool_limits
import tqdm

import slothpy as slt
from slothpy._general_utilities._constants import A_BOHR, H_CM_1, B_AU_T, AU_BOHR_CM_1, MU_B_CM_3
from slothpy._general_utilities._io import _hamiltonian_derivatives_from_dir_to_slt
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil
from slothpy.core._slt_file import SltHessian
from slothpy.core._hessian_object import Hessian

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cmc      # <-- contains `cmc.managua`
# register under its own name so that later get_cmap("managua") works
mpl.colormaps.register(cmc.managua, name = "managua")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterExponent


KB        = 0.6950347291
H         = 33.3571775619
H_BAR     = 33.3571775619 / (2 * pi)
S_TIME_PS = 1e12
T_FILED_OE = 10000
M_AU = 1822.89

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1dawsn")
c_dawsn = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)(addr)

@njit(nogil=True, cache=True, fastmath=True)
def dawsn(x):
    return c_dawsn(x, 0)

def k_mch(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray):
    k_mch = []
    U_R0_T = U_R0.conj().T
    print("Calculating K_MCH term:")
    for dof in tqdm.tqdm(dof_array):
        k_mch.append(U_R0_T @ group[f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}"][:] @ U_R0)

    return np.asarray(k_mch, dtype=np.complex128)

def E_grad_k_U(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray, B_vec: np.ndarray, displacement_number: int, step: float, degeneracy_tolerance: float = 1e-9):
    E_grad = []
    k_U = []
    finite_difference_stencil = _central_finite_difference_stencil(1, displacement_number, step * A_BOHR)

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
            hamiltonian = group[f"{group_name}/HAMILTONIAN_MATRIX"][:] - (magnetic_momenta[0] * B_vec[0] + magnetic_momenta[1] * B_vec[1] + magnetic_momenta[2] * B_vec[2])
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

def full_derivatives(dof_array, group, B_vec, displacement_number, step, degeneracy_tolerance, states_number):

    magnetic_momenta = grp["0/MAGNETIC_DIPOLE_MOMENTA"][:]

    hamiltonian0 = group[f"0/HAMILTONIAN_MATRIX"][:] - (magnetic_momenta[0] * B_vec[0] + magnetic_momenta[1] * B_vec[1] + magnetic_momenta[2] * B_vec[2])
    E_tot_0, U_R0 = np.linalg.eigh(hamiltonian0)

    k_mch_array = k_mch(dof_array, group, U_R0)
    E_grad_array, k_U_array = E_grad_k_U(dof_array, group, U_R0, B_vec, displacement_number, step, degeneracy_tolerance)
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

def export_susceptibility_csv(
    temperatures: ArrayLike,
    fields: ArrayLike,
    freqs: ArrayLike,
    chi: np.ndarray,
    filename: str | Path = "susceptibility_data.csv",
) -> None:

    T  = np.asarray(temperatures, dtype=np.float64)
    H  = np.asarray(fields, dtype=np.float64)
    f  = np.asarray(freqs, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.complex128)

    if chi.shape != (H.size, T.size, f.size):
        raise ValueError(
            f"`chi` must have shape (n_fields, n_temps, n_freqs) = "
            f"({H.size}, {T.size}, {f.size}); got {chi.shape!r}."
        )

    H_grid, T_grid, f_grid = np.meshgrid(H, T, f, indexing="ij")
    rows = np.column_stack([
        T_grid.ravel(),        
        H_grid.ravel(),
        f_grid.ravel(),
        np.abs(chi.real).ravel(),
        np.abs(chi.imag).ravel(),
    ])

    filename = Path(filename)
    with filename.open(mode="w", newline="") as fp:
        fp.write("[Data]\n")
        fp.write("Temperature (K),Magnetic Field (Oe),AC Frequency (Hz),AC X'  (emu/Oe),AC X\" (emu/Oe)\n")
        writer = csv.writer(fp, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerows(rows)

def export_tau_csv(
    temperatures: ArrayLike,
    fields: ArrayLike,
    tau: np.ndarray,
    filename: str | Path = "tau_data.csv",
) -> None:

    T  = np.asarray(temperatures, dtype=np.float64)
    H  = np.asarray(fields, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if tau.shape != (H.size, T.size):
        raise ValueError(
            f"`tau` must have shape (n_fields, n_temps) = "
            f"({H.size}, {T.size}); got {chi.shape!r}."
        )

    H_grid, T_grid = np.meshgrid(H, T, indexing="ij")
    rows = np.column_stack([
        T_grid.ravel(),
        H_grid.ravel(),
        tau.ravel(),
    ])

    filename = Path(filename)
    with filename.open(mode="w", newline="") as fp:
        fp.write("[Data]\n")
        fp.write("T,H,tau\n")
        writer = csv.writer(fp, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerows(rows)

def plot_susceptibility_curves(
        omega_rad_s : np.ndarray,        # (M,)
        chi_complex : np.ndarray,        # (K,M)
        temps_K     : np.ndarray,        # (K,)
        *,
        plot_type    : str = "freq",          # "freq"  or  "colecole"
        part         : str = "imag",          # only used if plot_type=="freq"
        colormap     : str = "viridis",
        reverse      : bool = False,
        color_mode   : str = "value",         # "value" or "index"
        legend_style : str = "list",          # "list" or "colorbar"
        ax           : mpl.axes.Axes | None = None,
        title        : str | None = None,
        xlabel       : str | None = None,
        ylabel       : str | None = None,
        savepath     : str | None = None,
        dpi          : int = 300,
):
    """
    plot_type:
        "freq"     – χ'(ν), χ''(ν) or |χ|(ν) on logarithmic ν axis
        "colecole" – Cole–Cole (χ'' vs χ') plot

    color_mode:
        "value"    – colour ∝ temperature value (continuous scale)
        "index"    – colour uses evenly spaced positions in the colormap,
                     one hue per temperature curve
    """

    # ------------------------------------------------------------------ #
    # 0. global style                                                    #
    # ------------------------------------------------------------------ #
    mpl.rcParams.update({
        "font.family"      : "serif",
        "font.size"        : 10,
        "mathtext.fontset" : "cm",
        "axes.labelsize"   : 10,
        "axes.titlesize"   : 10,
        "xtick.direction"  : "in",
        "ytick.direction"  : "in",
        "xtick.top"        : True,
        "ytick.right"      : True,
        "axes.spines.right": True,
        "axes.spines.top"  : True,
        "figure.dpi"       : dpi,
        "legend.handletextpad": 0.4,
        "legend.handlelength": 1.2,
        "legend.columnspacing": 0.6,
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5/1.15), constrained_layout=True)
    else:
        fig = ax.figure

    # ------------------------------------------------------------------ #
    # 1. prepare, sort, normalise                                        #
    # ------------------------------------------------------------------ #
    sorter    = np.argsort(temps_K)
    T_sorted  = np.asarray(temps_K, float)[sorter]
    χ_sorted  = np.asarray(np.abs(chi_complex.real)+ 1j * np.abs(chi_complex.imag))[sorter]

    # dimensionless normalisation (keeps sign)
    χ_sorted = χ_sorted / np.max(χ_sorted.imag)


    # ------------------------------------------------------------------ #
    # 2. build colour mapping                                            #
    # ------------------------------------------------------------------ #
    base_cmap = mpl.colormaps.get_cmap(colormap)
    if reverse:
        base_cmap = base_cmap.reversed()

    if color_mode == "value":
        norm      = mpl.colors.Normalize(vmin=T_sorted.min(),
                                         vmax=T_sorted.max())
        cmap_line = lambda T: base_cmap(norm(T))
        cbar_cmap, cbar_norm = base_cmap, norm

    elif color_mode == "index":
        palette   = base_cmap(np.linspace(0, 1, len(T_sorted)))
        cmap_idx  = mpl.colors.ListedColormap(palette,
                                              name=f"{colormap}_indexed")
        idx_norm  = mpl.colors.Normalize(vmin=0, vmax=len(T_sorted) - 1)
        cmap_line = lambda T: palette[np.where(T_sorted == T)[0][0]]

        cbar_cmap, cbar_norm = cmap_idx, idx_norm
    else:
        raise ValueError("color_mode must be 'value' or 'index'")

    # ------------------------------------------------------------------ #
    # 3. draw curves                                                     #
    # ------------------------------------------------------------------ #
    lines = []

    if plot_type.lower() == "freq":
        part_funcs = {"real": np.real, "imag": np.imag, "abs": np.abs}
        if part not in part_funcs:
            raise ValueError(f"'part' must be one of {list(part_funcs)}")
        f_part = part_funcs[part]

        for T, χ in zip(T_sorted, χ_sorted):
            line, = ax.plot(
                omega_rad_s,
                f_part(χ),
                lw=0.9,
                color=cmap_line(T),
            )
            lines.append(line)

        ax.set_xscale("log")
        xlabel = xlabel or r"$\nu / \mathrm{Hz}$"
        if ylabel is None:
            label_map = {"real": r"$\chi'$", "imag": r"$\chi''$", "abs": r"|$\chi$|"}
            ylabel = rf"{label_map[part]} / a.u."

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    elif plot_type.lower() == "colecole":
        for T, χ in zip(T_sorted, χ_sorted):
            line, = ax.plot(
                np.real(χ),
                np.imag(χ),
                lw=0.9,
                color=cmap_line(T),
            )
            lines.append(line)

        xlabel = xlabel or r"$\chi' / \mathrm{a.u.}$"
        ylabel = ylabel or r"$\chi'' / \mathrm{a.u.}$"
        # ax.set_aspect('equal', adjustable='datalim')

    else:
        raise ValueError("plot_type must be 'freq' or 'colecole'")

    # ------------------------------------------------------------------ #
    # 4. axis decoration                                                 #
    # ------------------------------------------------------------------ #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.4)
    if title:
        ax.set_title(title)

    # ------------------------------------------------------------------ #
    # 5. legend / colour-key                                             #
    # ------------------------------------------------------------------ #
    if legend_style == "list":
        ax.legend(
            list(reversed(lines)),
            [f"{T:g}" for T in T_sorted[::-1]],
            title=r"$T$ / K",
            title_fontsize=10,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
            labelspacing=0.3,
        )

    elif legend_style == "colorbar":
        sm = mpl.cm.ScalarMappable(cmap=cbar_cmap, norm=cbar_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$T$ / K")
        if color_mode == "value":
            cbar.set_ticks([T_sorted.min(), T_sorted.max()])
            cbar.set_ticklabels([f"{T_sorted.min():g}", f"{T_sorted.max():g}"])
        else:  # index mode → discrete bar
            cbar.set_ticks([0, len(T_sorted) - 1])
            cbar.set_ticklabels([f"{T_sorted[0]:g}", f"{T_sorted[-1]:g}"])
    else:
        raise ValueError("legend_style must be 'list' or 'colorbar'")

    # ------------------------------------------------------------------ #
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return ax

def make_tau_plotter(ax, *, label=None):
    line, = ax.plot([], [], "o", lw=1, ms=4, label=label or "τ(T)")
    xs, ys = [], []                 # pre‑allocate containers

    def _update(npts, tau):
        xs.append(npts)
        ys.append(np.log10(tau))    # store log10(τ)
        line.set_data(xs, ys)

        ax.relim();  ax.autoscale_view()
        plt.pause(0.0001)            # let the GUI breathe
    return _update

@njit(nogil=True, cache=True, fastmath=True)
def Jhat_p_sec(w_ab, wq, n_q, d, cutoff):
    if w_ab > 0:
        return gaussian(w_ab - wq, d, cutoff) * n_q
    if w_ab < 0:
        return gaussian(w_ab + wq, d, cutoff) * (n_q + 1)
    return 0.0 + 0.0 * 1j 

@njit(nogil=True, cache=True, fastmath=True)
def lorentz_hilbert(E, d):
    return -1j / (E - 1j * d)

@njit(nogil=True, cache=True, fastmath=True)
def Jhat_p(w_ab, wq, n_q, d, cutoff):
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff:
        return lorentz_hilbert(w_ab - wq, d) * n_q
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff:
        return lorentz_hilbert(w_ab + wq, d) * (n_q + 1)
    return 0.0 + 0.0 * 1j 

@njit(nogil=True, cache=True, fastmath=True)
def Jhat_m(w_ab, wq, n_q, d, cutoff):
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff:
        return lorentz_hilbert(w_ab + wq, d) * n_q
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff:
        return lorentz_hilbert(w_ab - wq, d) * (n_q + 1)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True)
def Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff):
    u = w_cd + w_ab
    if u < 0 and np.abs(u + wq) < cutoff:
        return lorentz_hilbert(u + wq, d) * n_q * zeta(w_ab + wq, beta)
    if u > 0 and np.abs(u - wq) < cutoff:
        return lorentz_hilbert(u - wq, d) * (n_q + 1) * zeta(w_ab - wq, beta)
    return 0.0 + 0.0 * 1j

# @njit(nogil=True, cache=True, fastmath=True)
# def gauss_hilbert(E, d):
#     factor_sqrt = E / (np.sqrt(2)*d)
#     return np.sqrt(np.pi*0.5) / d * (np.exp(-(factor_sqrt*factor_sqrt)) - 2j / np.sqrt(np.pi) * dawsn(factor_sqrt))

# @njit(nogil=True, cache=True, fastmath=True)
# def Jhat_p(w_ab, wq, n_q, d, cutoff):
#     if w_ab > 0 and np.abs(w_ab - wq) < cutoff:
#         return gauss_hilbert(w_ab - wq, d) * n_q
#     if w_ab < 0 and np.abs(w_ab + wq) < cutoff:
#         return gauss_hilbert(w_ab + wq, d) * (n_q + 1)
#     return 0.0 + 0.0 * 1j 

# @njit(nogil=True, cache=True, fastmath=True)
# def Jhat_m(w_ab, wq, n_q, d, cutoff):
#     if w_ab < 0 and np.abs(w_ab + wq) < cutoff:
#         return gauss_hilbert(w_ab + wq, d) * n_q
#     if w_ab > 0 and np.abs(w_ab - wq) < cutoff:
#         return gauss_hilbert(w_ab - wq, d) * (n_q + 1)
#     return 0.0 + 0.0 * 1j

# @njit(nogil=True, cache=True, fastmath=True)
# def Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff):
#     u = w_cd + w_ab
#     if u < 0 and np.abs(u + wq) < cutoff:
#         return gauss_hilbert(u + wq, d) * n_q * zeta(w_ab + wq, beta)
#     if u > 0 and np.abs(u - wq) < cutoff:
#         return gauss_hilbert(u - wq, d) * (n_q + 1) * zeta(w_ab - wq, beta)
#     return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True)
def add_PSI_bundle(out, A, Yb, wb, nb, delta, beta, w_n, q_0, i, cutoff, weight):
    N=w_n.shape[0]; J=wb.size
    coeff = H_BAR * 0.5 * weight
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y, wq, n_q, delta_j, cutoff_j = Yb[j], wb[j], nb[j], delta[j], cutoff[j]
        Yh=np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    d = c
                    cd=liou(c,d,N)
                    val=0.0+0.0j
                    for e in range(N):
                        val+=Jcorr(w_n[e,c],w_n[d,b],wq,n_q,delta_j,beta,cutoff_j)*A[e,c]*(Yh[d,b]*Y[a,e]+Y[d,b]*Yh[a,e])
                        val-=Jcorr(w_n[a,d],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j)*A[a,c]*(Y[d,e]*Yh[e,b]+Yh[d,e]*Y[e,b])
                        if a == c:
                            for f in range(N):
                                val+=Jcorr(w_n[e,f],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j)*A[e,f]*(Yh[f,b]*Y[d,e]+Y[f,b]*Yh[d,e])
                        val-=Jcorr(w_n[e,b],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j)*(Y[a,c]*Yh[d,e]+Yh[a,c]*Y[d,e])*A[e,b]
                    out[ab,cd,i]+=val*coeff

@njit(nogil=True, cache=True, fastmath=True)
def add_KR_bundle(out, Yb, wb, nb, delta, w_n, q_0, i, cutoff, weight):
    N = w_n.shape[0]; J = wb.size
    coeff = H_BAR * 0.5 * weight
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y, wq, n_q, delta_j, cutoff_j = Yb[j], wb[j], nb[j], delta[j], cutoff[j]
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
                                tmp+=Jhat_p(w_n[e,d],wq,n_q,delta_j,cutoff_j)*(Y[a,e]*Yh[e,c] + Yh[a,e]*Y[e,c])
                            val+=tmp
                        val-=Jhat_p(w_n[a,d],wq,n_q,delta_j,cutoff_j)*(Y[a,c]*Yh[d,b] + Yh[a,c]*Y[d,b])
                        if a==c:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat_m(w_n[c,e],wq,n_q,delta_j,cutoff_j)*(Y[d,e]*Yh[e,b] + Yh[d,e]*Y[e,b])
                            val+=tmp
                        val-=Jhat_m(w_n[c,b],wq,n_q,delta_j,cutoff_j)*(Y[a,c]*Yh[d,b] + Yh[a,c]*Y[d,b])
                        out[ab,cd,i]+=val * coeff

@njit(nogil=True, cache=True, fastmath=True)
def add_rho0_bundle(out, trace, A, Yb, wb, nb, beta, w_n, q_0, rho, i):
    N=w_n.shape[0]; J=wb.size
    coeff = 0.5
    if q_0:
        coeff *= 0.5
    for j in prange(J):
        Y, wq, n_q = Yb[j], wb[j], nb[j]
        Yh=np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    d = c
                    corr=0.0+0.0j
                    for e in range(N):
                        w_de=w_n[d,e]
                        w_eb=w_n[e,b]
                        if b == a:
                            trace[i]+=(n_q*Iint(w_de+wq,w_eb-wq,beta)+(n_q+1.0)*Iint(w_de-wq,+wq,beta))*rho[c,d]*(Y[d,e]*Yh[e,b]+Yh[d,e]*Y[e,b])
                        corr+=(n_q*Iint(w_de+wq,w_eb-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_eb+wq,beta))*A[a,c]*(Y[d,e]*Yh[e,b]+Yh[d,e]*Y[e,b])*rho[c,d]
                        for f in range(N):
                            w_ef=w_n[e,f]
                            corr-=(n_q*Iint(w_de+wq,w_ef-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_ef+wq,beta))*(1.0 if a==c else 0.0)*(Y[d,e]*Yh[e,f] + Yh[d,e]*Y[e,f])*A[f,b]*rho[c,d]
                    out[ab,i]+=coeff*corr

@njit(nogil=True, cache=True, fastmath=True)
def zeta(x: float, beta: float) -> float:
    eps = 1e-14
    u = beta * x
    if abs(u) < eps:
        return beta
    if abs(u) > 650:
        return 0.0 + 1j * 0.0
    return np.expm1(u) / (x)

@njit(nogil=True, cache=True, fastmath=True)
def Iint(w1: float, w2: float, beta: float) -> float:
    eps = 1e-14
    u1 = w1 * beta
    u2 = w2 * beta
    u12 = u1 + u2
    if abs(u1) > 650 or abs(u2) > 650 or abs(u12) > 650:
        return 0.0 + 1j * 0.0
    elif abs(u1) < eps and abs(u2) < eps:
        return 0.5 * beta * beta
    elif abs(u2) < eps:
        num = np.exp(u1)
        return beta * num / (w1) - np.expm1(u1) / (w1**2)
    elif abs(u1) < eps:
        return np.expm1(u2) / (w2**2) - beta / (w2)
    elif abs(u1 + u2) < eps:
        return -(beta - np.expm1(u1) / (1 * w1)) / (w1)
    term1 = np.expm1(u12) / (w2 * (w1 + w2))
    term2 = np.expm1(u1)  / (w1 * w2)
    return np.abs(term1 - term2)

@njit(nogil=True, cache=True, fastmath=True)
def bose_occ(freq: float, beta: float) -> float:
    u = beta * freq
    return 1.0/np.expm1(u)

@njit(nogil=True, cache=True, fastmath=True)
def liou(a, b, N):
    return a * N + b

def half_bz_grid_aniso(
    b_len: Sequence[float],
    n_ref: int,
    start_q: float,
    end_q: float = 0.0,
    *,
    endpoint: bool = True,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Anisotropic first-BZ mesh with equal point density in Cartesian q-space,
    optionally excluding the central region per axis.

    Parameters
    ----------
    b_len    : (3,) sequence
        Lengths |b1|, |b2|, |b3| of the reciprocal-lattice vectors.
    n_ref    : int (odd)
        Points along the *shortest* axis in the *full* mesh (must be odd).
    start_q  : float
        Outer limit per axis; grid spans [-start_q, start_q] on each axis.
    end_q    : float, optional
        Inner exclusion half-width per axis. If 0 (default), nothing is
        excluded and the grid is continuous. If > 0, per-axis domain is
        [-start_q,-end_q] ∪ [end_q,start_q] (Γ excluded unless end_q≈0).
    endpoint : bool, optional
        If True, the positive end (+start_q) is included (closed grid);
        if False, the grid is half-open on the +start_q side.
        In both cases 0 is centred on all axes (since n is odd).
    tol      : float, optional
        Tolerance for zero / symmetry tests.

    Returns
    -------
    q : (M, 3) ndarray
        Unique q-points (fractional coords), one per {+q, –q}.
        Γ is included only when end_q <= tol.
    """
    # ---- 0. sanity checks -------------------------------------------------
    if n_ref % 2 == 0:
        raise ValueError("n_ref must be odd so that 0 is on the grid.")

    b_len = np.asarray(b_len, float)
    if b_len.size != 3 or np.any(b_len <= 0):
        raise ValueError("b_len must contain three positive numbers.")

    if not (start_q > 0):
        raise ValueError("start_q must be positive.")
    if end_q < 0 or end_q >= start_q:
        raise ValueError("Require 0 <= end_q < start_q.")

    # ---- 1. choose n_i so that |b_i|/(n_i – 1) ≈ const -------------------
    b_min = b_len.min()
    n_axis = []
    for L in b_len:
        n = int(round(n_ref * L / b_min))    # proportional to length
        if n % 2 == 0:                       # force odd → includes 0
            n += 1
        n_axis.append(n)

    # ---- 2. build per-axis arrays over [-start_q, start_q] ---------------
    # Use linspace for both closed and half-open (endpoint=False) variants.
    # Then, if end_q > 0, drop the inner segment (-end_q, end_q) per axis.
    ax = []
    for n in n_axis:
        arr = np.linspace(-start_q, start_q, n, endpoint=endpoint, dtype=np.float64)
        if end_q > tol:
            # keep values in [-start_q,-end_q] ∪ [end_q,start_q] (within tol)
            mask = (arr <= -end_q + tol) | (arr >= end_q - tol)
            arr = arr[mask]
            if arr.size == 0:
                raise ValueError(
                    "After excluding the inner region, an axis became empty. "
                    "Increase n_ref or reduce end_q."
                )
        ax.append(arr)

    # ---- 3. full tensor-product grid -------------------------------------
    full = np.array(np.meshgrid(*ax, indexing="ij")).reshape(3, -1).T  # (N, 3)

    # ---- 4. inversion-symmetry reduction ---------------------------------
    keep = np.zeros(full.shape[0], dtype=bool)
    include_gamma = (end_q <= tol)

    for i, (x, y, z) in enumerate(full):
        # Γ kept only when inner region is not excluded
        if include_gamma and abs(x) < tol and abs(y) < tol and abs(z) < tol:
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

    # ---- 5. sort for reproducibility -------------------------------------
    idx = np.lexsort(q_unique.T[::-1])
    return q_unique[idx]

def multigrid_aniso(
    b_len: Sequence[float],
    n_ref: int,
    q_ranges: Sequence[float],
    *,
    endpoint: bool = True,
    tol: float = 1e-12
) -> np.ndarray:
    grids_list = []
    weights_list = []
    q_ranges.insert(0, 0.0)
    for i_q in range(1,len(q_ranges)):
        aniso_grid = half_bz_grid_aniso(b_len, n_ref, q_ranges[i_q], q_ranges[i_q-1], endpoint=endpoint, tol=tol)
        grid_weight = (2*q_ranges[i_q])**3
        aniso_weights = np.full(aniso_grid.shape[0], grid_weight)
        grids_list.append(aniso_grid)
        weights_list.append(aniso_weights)
    
    return np.vstack(grids_list), np.concatenate(weights_list)

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

@njit(nogil=True, cache=True, fastmath=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt, freq, modes_low):
    for j in range(normal_modes.shape[1]):
        for i in range(dof_array.shape[0]):
            dof = dof_array[i]
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

@njit(nogil=True, cache=True, fastmath=True)
def get_relax_time(R_mat):
    try:
        lam = np.linalg.eigvals(R_mat).real
    except Exception:
        return 0.0

    lam_size = lam.size
    i_min = 0
    best = abs(lam[0])
    for i in range(1, lam_size):
        ai = abs(lam[i])
        if ai < best:
            best = ai
            i_min = i
    max_neg = -np.inf
    for i in range(lam_size):
        if i == i_min:
            continue
        xi = lam[i]
        if xi < 0.0 and xi > max_neg:
            max_neg = xi

    return (-1.0/max_neg)/S_TIME_PS

def susceptibility_relax_time(
    omega_grid: np.ndarray,
    Hs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H_grad: np.ndarray,
    T: np.ndarray,
    gamma_fwhm: float,
    cutoff: float,
    hessian, masses_inv_sqrt, dof_array, grid, weights, modes_low, modes_high,
    *,
    states_number: int = 0,
    degeneracy_tolerance: float = 1e-6,
    secular_tolerance: float = 1e-6,
    threads: int = 64,
):

    omega_grid = omega_grid / S_TIME_PS
    beta = 1.0 / (KB * T)
    temp_size = beta.shape[0]

    # ── diagonalise and truncate to the requested sub-space ─────────────────
    E, U = np.linalg.eigh(Hs)
    A_e, B_e = U.conj().T @ A @ U, U.conj().T @ B @ U
    A_e = np.ascontiguousarray(A_e[:states_number, :states_number])
    B_e = np.ascontiguousarray(B_e[:states_number, :states_number])

    E = E[:states_number]

    E = (E - np.min(E)) * H_CM_1

    for i in range(E.shape[0]):
        for j in range(i+1, E.shape[0]):
            if np.isclose(E[i], E[j], atol=degeneracy_tolerance):
                E[i] = (E[i] + E[j]) * 0.5
                E[j] = E[i]

    E = (E - np.min(E))

    N  = states_number
    N2 = N * N

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
                        M_rho0[liou(a,b,N),liou(c,d,N)] += A_e[a,c]
                    if a == c:
                        M_rho0[liou(a,b,N),liou(c,d,N)] -= A_e[d,b]

    rho_vec = np.zeros((temp_size, N2), dtype=np.complex128)
    rho_mat = np.zeros((temp_size, N, N), dtype=np.complex128)
    for t in range(temp_size):
        rho_eq = np.exp(-beta[t] * E)
        rho_eq /= rho_eq.sum()
        rho_mat[t] = np.diag(rho_eq)
        for c in range(N):
            for d in range(N):
                rho_vec[t,liou(c, d, N)] += rho_mat[t,c,d]

    M_KR = np.zeros((temp_size, N2, N2, threads), dtype=np.complex128)
    M_PSI = np.zeros((temp_size, N2, N2, threads), dtype=np.complex128)
    rho_vec_init = np.zeros((temp_size, N2, threads), dtype=np.complex128)
    M_rho0_trace = np.zeros((temp_size, threads), dtype=np.complex128)
    R21 = np.zeros((temp_size, N2, N2, threads), dtype=np.complex128)
    R41 = np.zeros((temp_size, N2, N2, threads), dtype=np.complex128)

    w_n = E[:,np.newaxis] - E[np.newaxis,:]
    
    set_num_threads(threads)

    with threadpool_limits(1):
        start = perf_counter()
        build_matrices(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, modes_low, modes_high, beta, gamma_fwhm, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat, cutoff, R21, secular_tolerance, R41)
        stop = perf_counter()
        print((stop-start)/grid.shape[0]/T.shape[0])

    M_KR = np.sum(M_KR, axis=3)
    M_PSI = np.sum(M_PSI, axis=3)
    rho_vec_init = np.sum(rho_vec_init, axis=2)
    M_rho0_trace = 1.0 + np.sum(M_rho0_trace, axis=1)
    R21 = np.sum(R21, axis=3)
    R41 = np.sum(R41, axis=3)

    eye = np.eye(N2, dtype=np.complex128)

    chi_T = np.empty((temp_size, omega_grid.shape[0]), dtype=np.complex128)
    relax_time_R21_T = np.empty(temp_size, dtype=np.float64)
    relax_time_R41_T = np.empty(temp_size, dtype=np.float64)

    for t in range(temp_size):
        relax_time_R21_T[t] = get_relax_time(R21[t])
        print("R21:", relax_time_R21_T[t], np.log10(relax_time_R21_T[t]))
        relax_time_R41_T[t] = get_relax_time(R41[t])
        print("R41:", relax_time_R41_T[t], np.log10(relax_time_R41_T[t]))

        for k, omega in enumerate(omega_grid):
            Xi       = 1j / H_BAR * M_L + M_KR[t] / (H_BAR ** 2) - 1j * omega * eye
            num      = (1j / H_BAR * M_PSI[t]) @ rho_vec[t] + (M_rho0 @ rho_vec[t] + rho_vec_init[t]) / M_rho0_trace[t].real
            rho_hat  = np.linalg.solve(Xi, num).reshape((N, N))
            chi_T[t,k]   = 1j / H_BAR * np.trace(B_e @ rho_hat) / H_CM_1 * MU_B_CM_3

    return chi_T, relax_time_R21_T, relax_time_R41_T

@njit(nogil=True, cache=True, fastmath=True)
def _build_dynamical_matrix(hessian: np.ndarray, masses_inv_sqrt: np.ndarray, kpoint: np.ndarray):
    dyn_mat = np.zeros(masses_inv_sqrt.shape, dtype=np.complex128)

    for nx in prange(hessian.shape[0]):
        for ny in range(hessian.shape[1]):
            for nz in range(hessian.shape[2]):
                dyn_mat += hessian[nx, ny, nz, :, :] * np.exp(2j * pi * (kpoint[0] * nx + kpoint[1] * ny + kpoint[2] * nz))

    dyn_mat *= masses_inv_sqrt
    dyn_mat += dyn_mat.conj().T
    
    return -0.5 * dyn_mat

@njit(nogil=True, cache=True, fastmath=True)
def frequencies_eigenvectors(dynamical_matrix):
        frequencies_squared, eigenvectors = np.linalg.eigh(dynamical_matrix)

        return np.where(frequencies_squared >= 0, np.sqrt(np.abs(frequencies_squared)), -np.sqrt(np.abs(frequencies_squared))), eigenvectors

@njit(nogil=True, cache=True, fastmath=True)
def add_R21_bundle(out, Yb, wb, nb, delta, w_n, q_0, i, sec_tol, cutoff, weight):
    N, J = w_n.shape[0], wb.shape[0]
    prefc = pi / H * weight # pi * pi / H with Jhat_p_sec
    if q_0:
        prefc *= 0.5
    for j in range(J):
        Y, wq, n_q, delta_j, cutoff_j = Yb[j], wb[j], nb[j], delta[j], cutoff[j]
        Yh = np.conjugate(Y.T)
        for a in range(N):
            for b in range(N):
                ab = liou(a,b,N)
                for c in range(N):
                    for d in range(N):
                        cd = liou(c,d,N)
                        if abs(w_n[a,c]+w_n[d,b]) > sec_tol:
                            continue
                        val = 0.0 + 0.0j
                        val += (Y[a,c]*Yh[d,b] + Yh[a,c]*Y[d,b]) * Jhat_p(w_n[b,d], wq, n_q, delta_j, cutoff_j)
                        val += (Y[a,c]*Yh[d,b] + Yh[a,c]*Y[d,b]) * Jhat_p(w_n[a,c], wq, n_q, delta_j, cutoff_j)
                        if d == b:
                            for j in range(N):
                                val -= (Yh[a,j]*Y[j,c] + Y[a,j]*Yh[j,c]) * Jhat_p(w_n[j,c], wq, n_q, delta_j, cutoff_j)
                        if c == a:
                            for j in range(N):
                                val -= (Y[j,b]*Yh[d,j] + Yh[j,b]*Y[d,j]) * Jhat_p(w_n[j,d], wq, n_q, delta_j, cutoff_j)
                        out[ab,cd,i] += prefc * val

@njit(cache=True)
def _R_pm(V1, V2, w_n, sign, freq, lw):
    N = w_n.shape[0]
    out = np.zeros((N, N), np.complex128)
    for i in range(N):
        for j in range(N):
            acc = 0.0+0.0j
            for k in range(N):
                    acc += V1[i,k]*V2[k,j] / (w_n[k,j]+sign*freq-1j*lw)
            out[i,j] = acc
    return out

@njit(cache=True, inline="always", fastmath=True)
def gaussian(dE: float, lw: float, cutoff: float) -> float:
    if np.abs(dE) >= cutoff:
        return 0.0
    prefactor = 1 / (np.sqrt(2 * np.pi) * lw)
    exponent = -0.5 * dE * dE / (lw * lw)
    return prefactor * np.exp(exponent)

@njit(cache=True)
def add_R41(out: np.ndarray, w_n: np.ndarray, V1: np.ndarray, V2: np.ndarray, n1: float, n2: float, f1: float, f2: float, lw1: float, lw2: float, ind: int, cutoff: float, A: float, B: float, weight: float, sec_tol: float = 1e-6) -> np.ndarray:
    N = w_n.shape[0]
    prefc = pi * pi / H * weight
    lw_total = lw1

    Rabp = _R_pm(V1, V2, w_n, +1, f2, lw2)
    Rabm = _R_pm(V1, V2, w_n, -1, f2, lw2)
    Rbap = _R_pm(V2, V1, w_n, +1, f1, lw1)
    Rbam = _R_pm(V2, V1, w_n, -1, f1, lw1)

    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    if np.abs(w_n[a,c] + w_n[d,b]) >= sec_tol:
                        continue
                    ab = a * N + b
                    cd = c * N + d
                    val = 0.0 + 0.0j

                    G = n1 * (n2 + 1.0) * gaussian(w_n[a,c] - f1 + f2, lw_total, cutoff)
                    val += (np.conj(Rabp[b,d]) * Rabp[a,c] + np.conj(Rabp[b,d]) * Rbam[a,c] + np.conj(Rbam[b,d]) * Rabp[a,c] + np.conj(Rbam[b,d]) * Rbam[a,c]) * G * B
                    G = n2 * (n1 + 1.0) * gaussian(w_n[a,c] + f1 - f2, lw_total, cutoff)
                    val += (np.conj(Rabm[b,d]) * Rabm[a,c] + np.conj(Rabm[b,d]) * Rbap[a,c] + np.conj(Rbap[b,d]) * Rabm[a,c] + np.conj(Rbap[b,d]) * Rbap[a,c]) * G * B
                    G = n1 * n2 * gaussian(w_n[a,c] - f1 - f2, lw_total, cutoff)
                    val += (np.conj(Rabm[b,d]) * Rabm[a,c] + np.conj(Rbam[b,d]) * Rbam[a,c] + np.conj(Rbam[b,d]) * Rabm[a,c] + np.conj(Rabm[b,d]) * Rbam[a,c]) * G * A
                    G = (n1 + 1.0) * (n2 + 1.0) * gaussian(w_n[a,c] + f1 + f2, lw_total, cutoff)
                    val += (np.conj(Rabp[b,d]) * Rabp[a,c] + np.conj(Rbap[b,d]) * Rbap[a,c] + np.conj(Rbap[b,d]) * Rabp[a,c] + np.conj(Rabp[b,d]) * Rbap[a,c]) * G * A

                    # if a == c:
                    #     for k in range(N):
                    #         G = n1 * (n2 + 1.0) * gaussian(w_n[k,d] - f1 + f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabp[k,d]) * Rabp[k,b] + np.conj(Rabp[k,d]) * Rbam[k,b] + np.conj(Rbam[k,d]) * Rabp[k,b] + np.conj(Rbam[k,d]) * Rbam[k,b]) * G
                    #         G = n2 * (n1 + 1.0) * gaussian(w_n[k,d] + f1 - f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabm[k,d]) * Rabm[k,b] + np.conj(Rabm[k,d]) * Rbap[k,b] + np.conj(Rbap[k,d]) * Rabm[k,b] + np.conj(Rbap[k,d]) * Rbap[k,b]) * G
                    #         G = n1 * n2 * gaussian(w_n[k,d] - f1 - f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabm[k,d]) * Rabm[k,b] + np.conj(Rabm[k,d]) * Rbam[k,b] + np.conj(Rbam[k,d]) * Rabm[k,b] + np.conj(Rbam[k,d]) * Rbam[k,b]) * G
                    #         G = (n1 + 1.0) * (n2 + 1.0) * gaussian(w_n[k,d] + f1 + f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabp[k,d]) * Rabp[k,b] + np.conj(Rabp[k,d]) * Rbap[k,b] + np.conj(Rbap[k,d]) * Rabp[k,b] + np.conj(Rbap[k,d]) * Rbap[k,b]) * G
                    # if b == d:
                    #     for k in range(N):
                    #         G = n1 * (n2 + 1.0) * gaussian(w_n[k,c] - f1 + f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabp[k,a]) * Rabp[k,c] + np.conj(Rabp[k,a]) * Rbam[k,c] + np.conj(Rbam[k,a]) * Rabp[k,c] + np.conj(Rbam[k,a]) * Rbam[k,c]) * G
                    #         G = n2 * (n1 + 1.0) * gaussian(w_n[k,c] + f1 - f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabm[k,a]) * Rabm[k,c] + np.conj(Rabm[k,a]) * Rbap[k,c] + np.conj(Rbap[k,a]) * Rabm[k,c] + np.conj(Rbap[k,a]) * Rbap[k,c]) * G
                    #         G = n1 * n2 * gaussian(w_n[k,c] - f1 - f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabm[k,a]) * Rabm[k,c] + np.conj(Rabm[k,a]) * Rbam[k,c] + np.conj(Rbam[k,a]) * Rabm[k,c] + np.conj(Rbam[k,a]) * Rbam[k,c]) * G
                    #         G = (n1 + 1.0) * (n2 + 1.0) * gaussian(w_n[k,c] + f1 + f2, lw_total, cutoff)
                    #         val -= 0.5 * (np.conj(Rabp[k,a]) * Rabp[k,c] + np.conj(Rabp[k,a]) * Rbap[k,c] + np.conj(Rbap[k,a]) * Rabp[k,c] + np.conj(Rbap[k,a]) * Rbap[k,c]) * G

                    out[ab, cd, ind] += prefc * val

@njit(nogil=True, cache=True, fastmath=True, parallel=True)
def build_matrices(hessian: np.ndarray, masses_inv_sqrt: np.ndarray, dof_array: np.ndarray, H_grad: np.ndarray, grid: np.ndarray, weights: np.ndarray, modes_low: float, modes_high: float, beta: float, gamma_fwhm, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat, cutoff, R21, sec_tol, R41):
    n_k_inv = 1.0 / np.sqrt(grid.shape[0])
    masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)

    # print(w_n)

    # modes_high = np.log(1e40)/beta

    gamma = np.asarray([0.0, 0.0, 0.0])
    freq0, modes0 = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, gamma))
    freq_shape = freq0.shape[0]
    scale_freq = np.min(freq0)
    arr = np.diag(w_n, k=1)
    w_n_qtm_max = arr[0]
    for i in range(2, arr.size, 2):  # step of 2
        if arr[i] < w_n_qtm_max:
            w_n_qtm_max = arr[i]
    # print((freq0-scale_freq)*AU_BOHR_CM_1)

    # Raman ----------------------------------------------------------------------------------
    # threads_number = get_num_threads()
    # max_grid_per_thread = np.int64(np.ceil(grid.shape[0]/threads_number))
    # Yb_array = np.zeros((threads_number,2*max_grid_per_thread*freq_shape, H_grad.shape[1], H_grad.shape[2]), np.complex128)
    # wb_array = np.zeros((threads_number,2*max_grid_per_thread*freq_shape), np.float64)
    # raman_counter = np.zeros(threads_number, dtype=np.int64)
    # raman_counter_wb = np.zeros(threads_number, dtype=np.int64)
    # raman_counter_2wb = np.zeros(threads_number, dtype=np.int64)
    # ----------------------------------------------------------------------------------------

    # freq_min = np.inf

    for i in prange(grid.shape[0]):
        thread_id = get_thread_id()
        q = grid[i]
        weight = weights[i]
        q_0 = np.allclose(q, gamma, atol=1e-6)
        freq, modes = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, q))
        freq = freq - scale_freq
        freq *= AU_BOHR_CM_1
        # if thread_id == 0 and not q_0:
        #     freq_min = np.minimum(np.min(freq), freq_min)
        #     print(freq_min, np.min(freq))

        if q_0:
            freq, modes = freq[3:], modes[:,3:]
        mask = (freq >= modes_low) & (freq <= modes_high)
        idx  = np.where(mask)[0]
        wb, modes = np.ascontiguousarray(freq[idx]), np.ascontiguousarray(modes[:,idx])
        Yb = np.zeros((wb.size, H_grad.shape[1], H_grad.shape[2]), dtype=np.complex128)
        get_Y_q(Yb, H_grad, modes, q, dof_array, masses_inv_sqrt, n_k_inv, freq, modes_low)

        if q_0:
            max_freq_acoustic = np.min(wb)
        else:
            max_freq_acoustic = np.min(wb[3:])

        # Raman ----------------------------------------------------------------------------------
        # raman_counter_wb[thread_id] = raman_counter[thread_id] + wb.shape[0]
        # raman_counter_2wb[thread_id] = raman_counter_wb[thread_id] + wb.shape[0]
        # Yb_array[thread_id,raman_counter[thread_id]:raman_counter_wb[thread_id]] = Yb
        # wb_array[thread_id,raman_counter[thread_id]:raman_counter_wb[thread_id]] = wb
        # if not q_0:
        #     Yb_array[thread_id,raman_counter_wb[thread_id]:raman_counter_2wb[thread_id]] = np.conjugate(np.transpose(Yb, (0,2,1)))
        #     wb_array[thread_id,raman_counter_wb[thread_id]:raman_counter_2wb[thread_id]] = wb
        # raman_counter[thread_id] = raman_counter_2wb[thread_id]
        # ----------------------------------------------------------------------------------------

        for t_index in range(beta.shape[0]):
            bose = bose_occ(wb, beta[t_index])
            fwhm_j = gamma_fwhm[t_index] / modes_high / (2 / np.expm1(0.5 * modes_high * beta[0]) + 1) * wb * (2 / np.expm1(0.5 * wb * beta[t_index]) + 1) # * weight
            if not q_0:
                fwhm_j[:3] = gamma_fwhm[t_index] / modes_high / (2 / np.expm1(0.5 * modes_high * beta[0]) + 1) * max_freq_acoustic * (2 / np.expm1(0.5 * max_freq_acoustic * beta[t_index]) + 1) / max_freq_acoustic**2 / (1/beta[0]/KB)**3 * wb[:3]**2 * (1/beta[t_index]/KB)**3 # * weight
            cutoff_j = np.minimum(fwhm_j * 45, np.abs(wb + 1.01 * w_n_qtm_max)) 
            # if thread_id == 0:
            #     print(max_freq_acoustic, 1/beta[t_index]/KB, wb, fwhm_j, cutoff_j)
            # if thread_id == 0:
            #     print(wb, 1/beta[t_index]/KB, cutoff_j)
            add_KR_bundle(M_KR[t_index], Yb, wb, bose, fwhm_j, w_n, q_0, thread_id, cutoff_j, weight)
            add_PSI_bundle(M_PSI[t_index], A_e, Yb, wb, bose, fwhm_j, beta[t_index], w_n, q_0, thread_id, cutoff_j, weight)
            add_rho0_bundle(rho_vec_init[t_index], M_rho0_trace[t_index], A_e, Yb, wb, bose, beta[t_index], w_n, q_0, rho_mat[t_index], thread_id)
            # add_R21_bundle(R21[t_index], Yb, wb, bose, fwhm_j, w_n, q_0, thread_id, sec_tol, cutoff_j, weight)

    # Raman ----------------------------------------------------------------------------------
    # wb_array = wb_array.reshape((threads_number*2*max_grid_per_thread*freq_shape))
    # Yb_array = Yb_array.reshape((threads_number*2*max_grid_per_thread*freq_shape, H_grad.shape[1], H_grad.shape[2]))
    # N_pairs = wb_array.size * (wb_array.size + 1) // 2
    # for t_index_raman in range(beta.shape[0]):
    #     print("T start")
    #     bose_raman = bose_occ(wb_array, beta[t_index_raman])
    #     for p in prange(N_pairs):
    #         thread_id = get_thread_id()
    #         k = np.int64((np.sqrt(8*p + 1) - 1) // 2)
    #         l = p - k*(k + 1)//2
    #         if wb_array[k] == 0.0 or wb_array[l] == 0.0:
    #             continue
    #         A = 1 if k!=l else 0.25
    #         B = 1 if k!=l else 0.5
    #         add_R41(R41[t_index_raman], w_n, Yb_array[k], Yb_array[l], bose_raman[k], bose_raman[l], wb_array[k], wb_array[l], gamma_fwhm, gamma_fwhm, thread_id, cutoff, A, B, weight, sec_tol=sec_tol)
    # ----------------------------------------------------------------------------------------

if __name__ == "__main__":

    # ── USER-CONFIGURABLE SWEEP LISTS & PARAMETERS ──────────────────────────
    npoints_list    = [51] # 3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,71,77,81,85,91,101,111,121,131,141,151,161,181,201
    gamma_fwhm_list = [[0.38]*13]# [[0.5,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.7]]          # FWHM in cm-1
    T_list          = [1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,3.0] # [4,10,12,15,20,25,30,35,40] # [1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,3.0] # [1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,3.0] # 2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9
    B_list          = [0.1]  # 0.05,0.1,0.2,0.3        # Tesla 0.001,0.002,0.003,0.004,
    states_number   = 6                   # electronic sub-space size
    modes_low       = 0.00001    #cm-1
    modes_high      = 145 #cm-1
    q_ranges        = [0.125,0.25,0.5] # 0.015625,0.03125,0.0625,0.125,0.25,
    cutoff_list     = [[1]]# [[5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7]]
    degeneracy_tolerance = 1e-5
    secular_tolerance = 1e-5
    correlation = True
    # ────────────────────────────────────────────────────────────────────────

    # one-shot data that never changes over the sweep -----------------------
    lanthanide          = "Yb"
    orca_fragovl_path   = "/home/mikolaj/orca_6_0_1_avx2/orca_fragovl"
    dirpath             = f"/home/mikolaj/Data/Displacements_small_0001/{lanthanide}Co_displ" # "/home/mikolaj/Data/Displacements_cluster/CeCo_displ_cluster"
    slt_filepath        = "./seminarium/import.slt"
    group_name          = "xxx"
    displacement_number = 1
    step                = 0.0001
    omega_Hz            = np.logspace(-4, 7, 300)
    omega_angular       = 2*pi*omega_Hz
    chi_H_T = np.zeros((len(B_list),len(T_list), omega_Hz.shape[0]), dtype=np.complex128)
    tau_R21_H_T = np.zeros((len(B_list),len(T_list)), dtype=np.float64)
    tau_R41_H_T = np.zeros((len(B_list),len(T_list)), dtype=np.float64)

    # refresh the .slt file
    if os.path.exists(slt_filepath):
        os.remove(slt_filepath)
    slt.set_default_error_reporting_mode()
    _hamiltonian_derivatives_from_dir_to_slt(dirpath, slt_filepath, group_name, displacement_number, step, 64, 1, "ORCA",False, False, False, orca_fragovl_path)

    # phonon part -------------------------------------------------------
    Dy = slt.supercell("./seminarium/YCo_supercell_from_cell/dof_0_disp_0.xyz", slt_filepath, "YCo_supercell", 3, 3, 2,
        supercell_params=[22.663134149075237,
                            22.663134149075233,
                            25.14851428466812,
                            90.0, 90.0, 120.0],
        multiplicity=1,
    )
    Dy["YCo_supercell"].replace_atoms([0], [lanthanide])
    hessian = Dy["YCo_supercell"].hessian_from_finite_displacements("./seminarium/YCo_supercell_from_cell", "CP2K", "YCo_hessian", 1, 0.01, born_charges=True)

    slt_hessian     = SltHessian(hessian)
    masses_inv_sqrt = slt_hessian._masses_inv_sqrt
    recip_axes = slt_hessian.atoms_object().cell.reciprocal().cellpar()[:3]
    hess_obj        = Hessian(slt_hessian.hessian()[:], np.outer(masses_inv_sqrt, masses_inv_sqrt), np.array([0., 0., 0.]))

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
                plt.pause(0.0001)   # let the GUI breathe

        return _update
        
    # ---------------------------------------------------------------------- #
    #  PLOTTING CANVAS (two sub-plots: Re and Im)                            #
    # ---------------------------------------------------------------------- #
    fig, (ax_re, ax_im, ax_tau) = plt.subplots(
        1, 3, figsize=(15, 5), sharex=False,
        gridspec_kw={"width_ratios": (1.2, 1.2, 1)})   # ⬅ wider canvas

    for ax in (ax_re, ax_im):
        ax.set_xscale("log")
        ax.set_xlabel(r"ω  (rad s$^{-1}$)")
        ax.grid(True, which="both", ls=":")
    ax_re.set_ylabel(r"Re χ(ω)")
    ax_im.set_ylabel(r"Im χ(ω)")

    ax_tau.set_xlabel(r"$n_\mathrm{points}$")
    ax_tau.set_ylabel(r"$\log_{10}\,\tau$  (ps)")
    ax_tau.set_title("Relaxation time vs. mesh size")
    ax_tau.grid(True, ls=":")
    tau_plotter = make_tau_plotter(ax_tau)

    orients = np.array([[0,0,1]], np.float64) # np.array([[0,0,1], [1,0,0], [0,1,0]], np.float64)

    for orient in orients:
        orient /= (np.linalg.norm(orient) * B_AU_T)
    
        for B in enumerate(B_list):
            B_vec  = B[1] * orient

            with h5py.File(slt_filepath, "r") as f:
                grp = f[group_name]
                dof_array = dofs_with_complete_displacements(grp, displacement_number)
                magnetic_momenta = grp["0/MAGNETIC_DIPOLE_MOMENTA"][:]
                A_op = (magnetic_momenta[0] * orient[0] + magnetic_momenta[1] * orient[1] + magnetic_momenta[2] * orient[2])
                A_op = A_op * H_CM_1
                B_op = - A_op * 2.1142 # (cm-1/T to bohr magneton)
                H_total = (grp["0/HAMILTONIAN_MATRIX"][:] - (magnetic_momenta[0] * B_vec[0] + magnetic_momenta[1] * B_vec[1] + magnetic_momenta[2] * B_vec[2]))

                H_grad = full_derivatives(dof_array, grp, B_vec, 1, step, degeneracy_tolerance, states_number)
            
            for npoints in npoints_list:
                grid, weights = multigrid_aniso(recip_axes, npoints, q_ranges, endpoint=True)
                # grid = np.array([[0.0,0.0,0.0]], dtype=np.float64)

                for gamma_fwhm, cutoff in itertools.product(gamma_fwhm_list, cutoff_list):

                        chi_T, relax_time_R21_T, relax_time_R41_T = susceptibility_relax_time(
                            omega_angular, H_total, A_op, B_op, H_grad,
                            np.asarray(T_list, dtype=np.float64), np.asarray(gamma_fwhm, dtype=np.float64), np.asarray(cutoff, dtype=np.float64), slt_hessian.hessian()[:], masses_inv_sqrt, dof_array, grid, weights, modes_low, modes_high,
                            states_number=states_number,
                            degeneracy_tolerance=degeneracy_tolerance,
                            secular_tolerance=secular_tolerance)

                        chi_H_T[B[0],:,:] = chi_T
                        tau_R21_H_T[B[0],:] = relax_time_R21_T
                        tau_R41_H_T[B[0],:] = relax_time_R41_T

                        for T in enumerate(T_list):
                            label = (f"np={npoints}, γ={gamma_fwhm[T[0]]:.0e}, "
                                    f"T={T[1]:g} K, B={B[1]:g} T, cut={cutoff}")
                            step_plotter = make_step_plotter(ax_re, ax_im, label)
                            for omega in enumerate(omega_Hz):
                                step_plotter(omega[0], omega[1], chi_T[T[0],omega[0]])
                            tau_plotter(npoints, relax_time_R21_T[T[0]])
                            tau_plotter(npoints, relax_time_R41_T[T[0]])

    B_array = np.asarray(B_list)*T_FILED_OE
    export_susceptibility_csv(T_list, B_array, omega_Hz, chi_H_T, "./seminarium/test_ac_relacs.dat")
    export_tau_csv(T_list, B_array, tau_R21_H_T, "./seminarium/test_tau_R21_relacs.dat")
    export_tau_csv(T_list, B_array, tau_R41_H_T, "./seminarium/test_tau_R41_relacs.dat")

                # base_name = f"/home/mikolaj/Documents/PosterECMOLS25/{lanthanide}_{B}_{gamma_fwhm}_{npoints}{"_corr" if correlation else ""}"

                # ax = plot_susceptibility_curves(
                #         omega_au,
                #         chi_all_T,
                #         np.array(T_list),
                #         part="imag",
                #         legend_style="colorbar",
                #         colormap="managua",
                #         reverse=True,
                #         color_mode="index",
                #         title=rf"{lanthanide}Co (B = {B} T, $\Delta$ = {gamma_fwhm_list[0]} cm$^{{-1}}$)",
                #         savepath=f"{base_name}_imag.png",
                # )
                # plt.show()

                # ax = plot_susceptibility_curves(
                #         omega_au,
                #         chi_all_T,
                #         np.array(T_list),
                #         part="real",
                #         legend_style="colorbar",
                #         colormap="managua",
                #         reverse=True,
                #         color_mode="index",
                #         title=rf"{lanthanide}Co (B = {B} T, $\Delta$ = {gamma_fwhm_list[0]} cm$^{{-1}}$)",
                #         savepath=f"{base_name}_real.png",
                # )
                # plt.show()


                # ax = plot_susceptibility_curves(
                #         omega_au,
                #         chi_all_T,
                #         np.array(T_list),
                #         plot_type="colecole",
                #         part="real",
                #         legend_style="colorbar",
                #         colormap="managua",
                #         reverse=True,
                #         color_mode="index",
                #         title=rf"{lanthanide}Co (B = {B} T, $\Delta$ = {gamma_fwhm_list[0]} cm$^{{-1}}$)",
                #         savepath=f"{base_name}_colecole.png",
                # )
                # plt.show()


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