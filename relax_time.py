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
KB        = 0.6950347291  # 3.166811563e-6          # Eh K⁻¹
H         = 33.3571775619 # cm-1*ps
H_BAR     = 33.3571775619 / (2 * pi) # 0.6950347291 # 2.4188843265857e-17                     # ℏ
AU_TIME_S = 1e-12 # ps time → s
M_AU = 1822.89

# ─── helpers ─────────────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def liou_idx(a: int, b: int, N: int) -> int:
    return a * N + b

@njit(cache=True, inline="always")
def _lorentz(dE: float, lw: float) -> float:
    return -1j / (dE - 1j * lw)
    # return lw / (dE*dE + lw*lw)

@njit(cache=True, inline="always")
def _gauss(dE: float, lw: float) -> float:
    return np.exp(-(dE/lw)**2) / (lw*np.sqrt(pi))

@njit(cache=True)
def delta_line(kind: int, dE: float, lw: float) -> float:
    return _lorentz(dE, lw) if kind == 0 else _gauss(dE, lw)

@njit(cache=True, inline="always")
def bose_occ(freq: float, beta: float) -> float:
    u = beta * freq
    return 1.0/(np.exp(u) - 1.0)

# Principal‑value part of 1/(dE ± iΓ) for Lorentzian broadening.
@njit(cache=True, inline="always")
def _pval_lorentz(dE: float, lw: float) -> float:
    return dE / (dE * dE + lw * lw)

# Derivative of Lorentzian δ w.r.t. dE (needed by get_dK21)
@njit(cache=True, inline="always")
def _d_lorentz(dE: float, lw: float) -> float:
    den = dE * dE + lw * lw
    return -2.0 * lw * dE / pi / (den * den)

# Derivative of the principal‑value kernel w.r.t. dE (Lorentz)
@njit(cache=True, inline="always")
def _d_pval_lorentz(dE: float, lw: float) -> float:
    den  = dE * dE + lw * lw
    return (lw * lw - dE * dE) / (den * den)

# Public wrapper that switches on `kind`
@njit(cache=True, inline="always")
def pval(kind: int, dE: float, lw: float) -> float:
    if kind != 0:
        # Gaussian Hilbert transform (Faddeeva) is non‑trivial – not yet needed
        raise NotImplementedError("Gaussian ‘pval’ not implemented – use kind=0 (Lorentzian)")
    return _pval_lorentz(dE, lw)

@njit(cache=True, inline="always")
def d_delta_line(kind: int, dE: float, lw: float) -> float:
    if kind != 0:
        raise NotImplementedError("Gaussian d_delta not implemented – use kind=0 (Lorentzian)")
    return _d_lorentz(dE, lw)

@njit(cache=True, inline="always")
def d_pval(kind: int, dE: float, lw: float) -> float:
    if kind != 0:
        raise NotImplementedError("Gaussian d_pval not implemented – use kind=0 (Lorentzian)")
    return _d_pval_lorentz(dE, lw)

# ─── second-order (one-phonon) kernel ───────────────────────────────────────
@njit(cache=True)
def _add_R21_mode(R21: np.ndarray, Ener: np.ndarray, V: np.ndarray,
                  freq: float, nB: float, lw: float,
                  smear: int, sec_tol: float, q_0: bool):

    N, N2 = Ener.size, Ener.size*Ener.size
    prefc = pi / H

    R21 = np.zeros((N2, N2), np.complex128)

    if q_0:
        prefc *= 0.5

    for ab in range(N2):
        a, b = ab // N, ab % N
        for cd in range(N2):
            c, d = cd // N, cd % N
            if abs(Ener[a]-Ener[c]+Ener[d]-Ener[b]) > sec_tol:
                continue

            val = 0.0 + 0.0j

            dE = Ener[b]-Ener[d]
            s  = 1.0 if dE > 0.0 else -1.0
            g  = (nB if s > 0 else nB+1.0) * pi * delta_line(smear, dE - s*freq, lw)
            val += (V[a,c]*np.conj(V[b,d]) + np.conj(V[c,a])*V[d,b]) * g

            dE = Ener[a]-Ener[c]
            s  = 1.0 if dE > 0.0 else -1.0
            g  = (nB if s > 0 else nB+1.0) * pi * delta_line(smear, dE - s*freq, lw)
            val += (V[a,c]*np.conj(V[b,d]) + np.conj(V[c,a])*V[d,b]) * g

            if d == b:
                for k in range(N):
                    dE = Ener[k]-Ener[c]
                    s  = 1.0 if dE > 0.0 else -1.0
                    g  = (nB if s > 0 else nB+1.0) * pi * delta_line(smear, dE - s*freq, lw)
                    val -= (np.conj(V[k,a])*V[k,c] + V[a,k]*np.conj(V[c,k])) * g
            if c == a:
                for k in range(N):
                    dE = Ener[k]-Ener[d]
                    s  = 1.0 if dE > 0.0 else -1.0
                    g  = (nB if s > 0 else nB+1.0) * pi * delta_line(smear, dE - s*freq, lw)
                    val -= (V[k,b]*np.conj(V[k,d]) + np.conj(V[b,k])*V[d,k]) * g

            R21[ab, cd] += prefc * val
    
    return R21


def build_R21(Ener: np.ndarray, temp: float, lw: float, smear: int,
              gen: Callable[[], Iterable[Tuple[np.ndarray, np.ndarray]]],
              *, sec_tol: float = 1e-12) -> np.ndarray:
    N = Ener.size
    R = np.zeros((N*N, N*N), np.complex128)
    beta = 1.0 / (KB*temp)
    for Y, w, q_0 in gen():
        for V, f in zip(Y, w):
            R += _add_R21_mode(R, Ener, V, f, bose_occ(f, beta), lw, smear, sec_tol, q_0)
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


# -----------------------------------------------------------------------------
#  Main driver  —  exact transcription of Fortran MAKE_R41
# -----------------------------------------------------------------------------
@njit(cache=True)
def make_R41(Ener: np.ndarray,
             V1:   np.ndarray,
             V2:   np.ndarray,
             T:    float,
             f1:   float,
             f2:   float,
             lw1:  float,
             lw2:  float,
             smear: int,
             q_0: bool,
             correction: bool = False,
             sec_tol: float = 1e-6) -> np.ndarray:
    N  = Ener.size
    N2 = N * N

    beta = 1.0 / (KB * T)
    n1, n2 = bose_occ(f1, beta), bose_occ(f2, beta)

    Rabp = _R_pm(V1, V2, Ener, +1, f2, lw2)
    Rabm = _R_pm(V1, V2, Ener, -1, f2, lw2)
    Rbap = _R_pm(V2, V1, Ener, +1, f1, lw1)
    Rbam = _R_pm(V2, V1, Ener, -1, f1, lw1)

    prefc = pi * pi / H

    if q_0 == 1:
        prefc *= 0.5
    elif q_0 == 2:
        prefc *= 0.25

    Δ  = lambda dE: delta_line(smear, dE, lw1)  # Fortran used Γ₁ for δ

    R = np.zeros((N2, N2), np.complex128)

    for ab in range(N2):
        a = ab // N
        b = ab - a * N
        for cd in range(N2):
            c = cd // N
            d = cd - c * N
            if np.abs(Ener[a] - Ener[c] + Ener[d] - Ener[b]) >= sec_tol:
                continue

            val = 0.0 + 0.0j

            # (ω₁ emit, ω₂ absorb)
            G = n1 * (n2 + 1.0) * Δ(Ener[a] - Ener[c] - f1 + f2)
            val += (
                np.conj(Rabp[b, d]) * Rabp[a, c] +
                np.conj(Rabp[b, d]) * Rbam[a, c] +
                np.conj(Rbam[b, d]) * Rabp[a, c] +
                np.conj(Rbam[b, d]) * Rbam[a, c]
            ) * G

            # (ω₁ absorb, ω₂ emit)
            G = n2 * (n1 + 1.0) * Δ(Ener[a] - Ener[c] + f1 - f2)
            val += (
                np.conj(Rabm[b, d]) * Rabm[a, c] +
                np.conj(Rabm[b, d]) * Rbap[a, c] +
                np.conj(Rbap[b, d]) * Rabm[a, c] +
                np.conj(Rbap[b, d]) * Rbap[a, c]
            ) * G

            # double absorption
            G = n1 * n2 * Δ(Ener[a] - Ener[c] - f1 - f2)
            val += (
                np.conj(Rabm[b, d]) * Rabm[a, c] +
                np.conj(Rbam[b, d]) * Rbam[a, c] +
                np.conj(Rbam[b, d]) * Rabm[a, c] +
                np.conj(Rabm[b, d]) * Rbam[a, c]
            ) * G

            # double emission
            G = (n1 + 1.0) * (n2 + 1.0) * Δ(Ener[a] - Ener[c] + f1 + f2)
            val += (
                np.conj(Rabp[b, d]) * Rabp[a, c] +
                np.conj(Rbap[b, d]) * Rbap[a, c] +
                np.conj(Rbap[b, d]) * Rabp[a, c] +
                np.conj(Rabp[b, d]) * Rbap[a, c]
            ) * G

            # ------------------------------------------------------------------
            #  Counter‑terms (la == lc) / (lb == ld)  –  identical to Fortran
            # ------------------------------------------------------------------
            if a == c:
                for k in range(N):
                    G = n1 * (n2 + 1.0) * Δ(Ener[k] - Ener[d] - f1 + f2)
                    val -= 0.5 * (
                        np.conj(Rabp[k, d]) * Rabp[k, b] +
                        np.conj(Rabp[k, d]) * Rbam[k, b] +
                        np.conj(Rbam[k, d]) * Rabp[k, b] +
                        np.conj(Rbam[k, d]) * Rbam[k, b]
                    ) * G

                    G = n2 * (n1 + 1.0) * Δ(Ener[k] - Ener[d] + f1 - f2)
                    val -= 0.5 * (
                        np.conj(Rabm[k, d]) * Rabm[k, b] +
                        np.conj(Rabm[k, d]) * Rbap[k, b] +
                        np.conj(Rbap[k, d]) * Rabm[k, b] +
                        np.conj(Rbap[k, d]) * Rbap[k, b]
                    ) * G

                    G = n1 * n2 * Δ(Ener[k] - Ener[d] - f1 - f2)
                    val -= 0.5 * (
                        np.conj(Rabm[k, d]) * Rabm[k, b] +
                        np.conj(Rabm[k, d]) * Rbam[k, b] +
                        np.conj(Rbam[k, d]) * Rabm[k, b] +
                        np.conj(Rbam[k, d]) * Rbam[k, b]
                    ) * G

                    G = (n1 + 1.0) * (n2 + 1.0) * Δ(Ener[k] - Ener[d] + f1 + f2)
                    val -= 0.5 * (
                        np.conj(Rabp[k, d]) * Rabp[k, b] +
                        np.conj(Rabp[k, d]) * Rbap[k, b] +
                        np.conj(Rbap[k, d]) * Rabp[k, b] +
                        np.conj(Rbap[k, d]) * Rbap[k, b]
                    ) * G

            if b == d:
                for k in range(N):
                    G = n1 * (n2 + 1.0) * Δ(Ener[k] - Ener[c] - f1 + f2)
                    val -= 0.5 * (
                        np.conj(Rabp[k, a]) * Rabp[k, c] +
                        np.conj(Rabp[k, a]) * Rbam[k, c] +
                        np.conj(Rbam[k, a]) * Rabp[k, c] +
                        np.conj(Rbam[k, a]) * Rbam[k, c]
                    ) * G

                    G = n2 * (n1 + 1.0) * Δ(Ener[k] - Ener[c] + f1 - f2)
                    val -= 0.5 * (
                        np.conj(Rabm[k, a]) * Rabm[k, c] +
                        np.conj(Rabm[k, a]) * Rbap[k, c] +
                        np.conj(Rbap[k, a]) * Rabm[k, c] +
                        np.conj(Rbap[k, a]) * Rbap[k, c]
                    ) * G

                    G = n1 * n2 * Δ(Ener[k] - Ener[c] - f1 - f2)
                    val -= 0.5 * (
                        np.conj(Rabm[k, a]) * Rabm[k, c] +
                        np.conj(Rabm[k, a]) * Rbam[k, c] +
                        np.conj(Rbam[k, a]) * Rabm[k, c] +
                        np.conj(Rbam[k, a]) * Rbam[k, c]
                    ) * G

                    G = (n1 + 1.0) * (n2 + 1.0) * Δ(Ener[k] - Ener[c] + f1 + f2)
                    val -= 0.5 * (
                        np.conj(Rabp[k, a]) * Rabp[k, c] +
                        np.conj(Rabp[k, a]) * Rbap[k, c] +
                        np.conj(Rbap[k, a]) * Rabp[k, c] +
                        np.conj(Rbap[k, a]) * Rbap[k, c]
                    ) * G

            # ------------------------------------------------------------------
            #  Higher‑order corrections (K21, dK21)  –  fully implemented
            # ------------------------------------------------------------------
            if correction:
                lw_tot = lw1 + lw2
                for lk in range(N):
                    for ll in range(N):
                        sec2 = Ener[a] - Ener[lk] + Ener[ll] - Ener[b]
                        sec3 = Ener[lk] - Ener[c] + Ener[d] - Ener[ll]
                        if (np.abs(sec2) < sec_tol) and (np.abs(sec3) < sec_tol):
                            #  G2   (built with V2)
                            e_shift = Ener[c] - Ener[d]
                            G2 = get_K21(V2, Ener, T, f2, e_shift, lw_tot, smear, lk, ll, c, d)

                            #  K2   (built with V1) – energy denominator ΔE
                            Ediff  = Ener[lk] - Ener[ll] - Ener[c] + Ener[d]
                            if np.abs(Ediff) >= sec_tol:
                                num  = get_K21(V1, Ener, T, f1, Ener[lk] - Ener[ll], lw_tot, smear, a, b, lk, ll)
                                num -= get_K21(V1, Ener, T, f1, e_shift,            lw_tot, smear, a, b, lk, ll)
                                K2   = num / Ediff
                            else:
                                K2   = get_dK21(V1, Ener, T, f1, e_shift, lw_tot, smear, a, b, lk, ll)
                            val -= K2 * G2

                            #  Swap V1 ↔ V2
                            G2 = get_K21(V1, Ener, T, f1, e_shift, lw_tot, smear, lk, ll, c, d)
                            if np.abs(Ediff) >= sec_tol:
                                num  = get_K21(V2, Ener, T, f2, Ener[lk] - Ener[ll], lw_tot, smear, a, b, lk, ll)
                                num -= get_K21(V2, Ener, T, f2, e_shift,            lw_tot, smear, a, b, lk, ll)
                                K2   = num / Ediff
                            else:
                                K2   = get_dK21(V2, Ener, T, f2, e_shift, lw_tot, smear, a, b, lk, ll)
                            val -= K2 * G2

            R[ab, cd] += prefc * val

    return R

# -----------------------------------------------------------------------------
#  Place‑holders for the higher‑order kernels
# -----------------------------------------------------------------------------

@njit(cache=True)
def get_K21(V: np.ndarray,
            Ener: np.ndarray,
            T: float,
            freq: float,
            e_shift: float,   # Re(ω)   ≡ dble(lw) in Fortran
            width: float,     # Im(ω)   ≡ aimag(lw)
            smear: int,
            la: int, lb: int, lc: int, ld: int) -> complex:
    """Exact Numba port of Fortran *get_K21* (linear system–bath coupling)."""

    N      = Ener.size
    prefc  = pi / H
    beta   = 1.0 / (KB * T)
    n      = bose_occ(freq, beta)

    K21 = 0.0 + 0.0j

    # ------------------------------------------------------------------
    #  Helper : accumulate a pair of spectrum‑shifted terms
    # ------------------------------------------------------------------
    def _spectral_pair(dE_base: float, v_left: complex, v_right: complex):
        """Add contributions with ±ω to K21 (closure over outer scope)."""
        nonlocal K21
        # +ω
        dE = dE_base + freq + e_shift
        Gf = (pi * delta_line(smear, dE, width) + 1j * pval(smear, dE, width)) * n
        # −ω
        dE = dE_base - freq + e_shift
        Gf += (pi * delta_line(smear, dE, width) + 1j * pval(smear, dE, width)) * (n + 1.0)
        K21 += v_left * np.conj(v_right) * Gf * prefc

    # ------------------------------------------------------------------
    #  Core pairs (four indices fixed)
    # ------------------------------------------------------------------
    _spectral_pair(Ener[ld] - Ener[la], V[la, lc], V[lb, ld])
    _spectral_pair(Ener[lb] - Ener[lc], V[la, lc], V[lb, ld])

    # ------------------------------------------------------------------
    #  Additional sums for degenerate secular manifolds
    # ------------------------------------------------------------------
    if ld == lb:
        for kk in range(N):
            _spectral_pair(Ener[lb] - Ener[kk], V[kk, lc], V[kk, la])

    if lc == la:
        for kk in range(N):
            _spectral_pair(Ener[kk] - Ener[la], V[kk, lb], V[kk, ld])

    return K21

# -----------------------------------------------------------------------------
#  Derivative  dK21/d(dE)  (Fortran *get_dK21*)
# -----------------------------------------------------------------------------
@njit(cache=True)
def get_dK21(V: np.ndarray,
             Ener: np.ndarray,
             T: float,
             freq: float,
             e_shift: float,
             width: float,
             smear: int,
             la: int, lb: int, lc: int, ld: int) -> complex:
    """Derivative of K21 with respect to the secular energy (dK21/dΔ)."""

    N      = Ener.size
    prefc  = pi / H
    beta   = 1.0 / (KB * T)
    n      = bose_occ(freq, beta)

    K21 = 0.0 + 0.0j

    # ------------------------------------------------------------------
    def _spectral_pair_deriv(dE_base: float, v_left: complex, v_right: complex):
        nonlocal K21
        # +ω
        dE = dE_base + freq + e_shift
        Gf = (pi * d_delta_line(smear, dE, width) + 1j * d_pval(smear, dE, width)) * n
        # −ω
        dE = dE_base - freq + e_shift
        Gf += (pi * d_delta_line(smear, dE, width) + 1j * d_pval(smear, dE, width)) * (n + 1.0)
        K21 += v_left * np.conj(v_right) * Gf * prefc

    # Main blocks
    _spectral_pair_deriv(Ener[ld] - Ener[la], V[la, lc], V[lb, ld])
    _spectral_pair_deriv(Ener[lb] - Ener[lc], V[la, lc], V[lb, ld])

    if ld == lb:
        for kk in range(N):
            _spectral_pair_deriv(Ener[lb] - Ener[kk], V[kk, lc], V[kk, la])

    if lc == la:
        for kk in range(N):
            _spectral_pair_deriv(Ener[kk] - Ener[la], V[kk, lb], V[kk, ld])

    return K21


def build_R41(Ener,
                             mode_gen,
                             temp: float,
                             lw_ph=0.0,
                             type_smear: int = 0,
                             max_cache: int = 64,
                             correction: bool = False,
                             sec_tol: float = 1e-6):
    
    lw1 = lw2 = np.float64(lw_ph)

    N  = Ener.size
    N2 = N * N

    # ─── gather all modes in a flat list ──────────────────────────────
    modes = []  # each element: (freq, Y_matrix)
    for Y_q, freq_vec, q_0 in mode_gen():
        for j in range(freq_vec.size):
            modes.append((np.float64(freq_vec[j]), Y_q[j].copy(), q_0))
            modes.append((np.float64(freq_vec[j]), Y_q[j].conj().T.copy(), q_0))

    # ─── double sum without repetition (unordered pairs) ──────────────
    R_total = np.zeros((N2, N2), np.complex128)
    for (f1, V1, q_01), (f2, V2, q_02) in itertools.combinations_with_replacement(modes, 2):
        block = make_R41(Ener, V1, V2, temp, f1, f2, lw1, lw2, type_smear, q_01 + q_02, correction, sec_tol)
        if V1 is not V2:
            block = block + block.T

        R_total += block

    return R_total



# ─── Lindbladian + relaxation time ──────────────────────────────────────────
def redfield_lindbladian(Ener, T, lw, smear, gen, *,
                         include_R41=False, sec_tol=1e-12):

    R21 = build_R21(Ener, T, lw, smear, gen, sec_tol=sec_tol)
    R41 = build_R41(Ener, gen, T, lw, smear, correction=False, sec_tol=sec_tol) if include_R41 else np.zeros_like(R21)

    N = Ener.size
    iL = np.zeros_like(R21)
    ω  = Ener
    for a in range(N):
        for b in range(N):
            iL[liou_idx(a,b,N), liou_idx(a,b,N)] = 1j*(ω[a]-ω[b])

    return iL + R21 + R41, R21, R41

def relaxation_time(R, *, tol=0):
    lam = eigvals(R)
    print(lam)
    lam = lam.real
    print(np.sort(lam))
    seq = list(lam)
    idx_min = min(range(len(seq)), key=lambda i: abs(seq[i]))
    seq.pop(idx_min)
    negatives = [x for x in seq if x < 0]
    if not negatives:
        raise ValueError("No negative numbers remain after removal.")
    return -1.0 / max(negatives)

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

def random_H_grad(
    dof_array: np.ndarray,
    n_states: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build a stand-in array of Hamiltonian gradients for testing.

    Parameters
    ----------
    dof_array : (N_dof, 4) int ndarray
        Same structure you pass to `crystal_field_derivatives`
        (column 0 stores the Cartesian direction: −2→x, −1→y, 0→z
        in Sloth / Spiral notation).
    n_states  : int
        Dimension of the electronic sub-space (i.e. `states_number`).
    rng : np.random.Generator, optional
        Random-number generator; pass one if you need reproducibility
        (e.g. `rng=np.random.default_rng(123)`).

    Returns
    -------
    H_grad : (N_dof, n_states, n_states) complex ndarray
        Hermitian matrices with ⟨V_x⟩ = ⟨V_y⟩ = ⟨V_z⟩ = 0
        exactly like the Fortran normalisation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_dof = dof_array.shape[0]
    H_grad = np.empty((n_dof, n_states, n_states), dtype=np.complex128)

    # --- 1. generate independent Hermitian matrices -----------------------
    for i in range(n_dof):
        A = rng.standard_normal((n_states, n_states)) \
            + 1j * rng.standard_normal((n_states, n_states))
        H_grad[i] = (A + A.conj().T) / 2.0          # Hermitian projection

    # --- 2. subtract direction-wise averages (x,y,z) -----------------------
    dir_idx = (dof_array[:, 0] + 2) % 3            # 0→x, 1→y, 2→z
    for l in (0, 1, 2):
        mask = dir_idx == l
        if mask.any():
            avg = H_grad[mask].mean(axis=0)
            H_grad[mask] -= avg

    return H_grad


@njit(cache=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt, freq):
    for i in range(dof_array.shape[0]):
        dof = dof_array[i]
        for j in range(normal_modes.shape[1]):
            Y_q[j] +=  (1 / np.sqrt(freq[j]/H_CM_1)) * H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * 1/np.sqrt(M_AU) * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))


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

if __name__ == "__main__":

    # ── USER-CONFIGURABLE SWEEP LISTS & PARAMETERS ──────────────────────────
    npoints_list    = [5]
    gamma_fwhm_list = [5]          # FWHM in a.u.
    T_list          = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,8,8.5,9,9.5,10,11,12,13,14,15]           # Kelvin
    B_list          = [0.00000001]            # Tesla
    states_number   = 6                    # electronic sub-space size
    modes_mult      = 1.1
    mode_threshold  = 1e-30
    modes_low       = 3    #cm-1
    modes_high      = 1000 #cm-1
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

    add, finish = make_T1_accumulator()

    for npoints in npoints_list:
        for B in B_list:

            orient = np.array([1, 1, 1], np.float64)
            orient /= np.linalg.norm(orient)
            B_vec  = B / B_AU_T * orient

            grid = half_bz_grid_aniso(recip_axes, npoints)
            # grid = slt_hessian.atoms_object().cell.bandpath(npoints=npoints).kpts

            # # print(len(grid))

            # # q = np.asarray([0.])
            # q = np.linspace(-0.5, 0.5, npoints, endpoint=False)
            # q_grid = np.meshgrid(q, q, q, indexing='ij')
            # grid = np.ascontiguousarray(np.vstack([grid.ravel() for grid in q_grid]).T)

            with h5py.File(slt_filepath, "r") as f:
                grp = f[group_name]
                dof_array = dofs_with_complete_displacements(grp, displacement_number)
                magnetic_momenta = grp["0/MAGNETIC_DIPOLE_MOMENTA"][:]
                H_total = grp["0/HAMILTONIAN_MATRIX"][:] - (magnetic_momenta[0] * B_vec[0] + magnetic_momenta[1] * B_vec[1] + magnetic_momenta[2] * B_vec[2])

                E_tot, U_R0 = np.linalg.eigh(H_total)

                # E_shift = modes_mult * np.max(E_tot - E_tot[0]) * H_CM_1 / AU_BOHR_CM_1

                # k-matrix & gradients with **CORRECT** B-vector
                # k_mch_arr = k_mch(dof_array, grp, U_R0)
                # E_grad, k_U_arr = E_grad_k_U(
                #     dof_array, grp, U_R0,
                #     B_vec, displacement_number, 1e-10
                # )

                # # build H_grad (field dependent) and truncate -------------------
                # H_grad = np.empty_like(k_mch_arr)
                # for i in range(H_grad.shape[0]):
                #     Ek = (E_tot[None, :] - E_tot[:, None]) * (k_mch_arr[i] + k_U_arr[i])
                #     np.fill_diagonal(Ek, E_grad[i])
                #     H_grad[i] = Ek
                # H_grad = H_grad[:, :states_number, :states_number]

                E_tot = E_tot[:states_number] * H_CM_1

                H_grad = crystal_field_derivatives(dof_array, grp, B_vec, 1, step, states_number)

                # H_grad = random_H_grad(dof_array, states_number)

            # closure capturing *field-dependent* H_grad --------------------
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

                Rtot, R21, R41 = redfield_lindbladian(
                    E_tot, T, gamma_fwhm, 0, get_Y_q_and_freq,
                    include_R41=False, sec_tol=secular_tolerance)

                T1_R21_s = AU_TIME_S * relaxation_time(R21)
                print(T1_R21_s)
                # T1_R41_s = AU_TIME_S * relaxation_time(R41)
                # print(T1_R41_s)

                add(T, "R21", B, gamma_fwhm, T1_R21_s)
                # add(T, "R41", B, gamma_fwhm, T1_R41_s)

    finish(invT=True)
