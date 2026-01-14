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

from __future__ import annotations

import numpy as np
from numpy import pi

from typing import Sequence, Union
ArrayLike = Union[Sequence, np.ndarray]

from numba import njit, prange, types, get_thread_id, literally, set_num_threads, get_num_threads
from numba.extending import intrinsic, get_cython_function_address, overload
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.errors import TypingError
from llvmlite import ir as llir
from llvmlite import binding as llvm
from llvmlite.ir import IRBuilder

from slothpy._general_utilities._constants import H_CM_1, AU_BOHR_CM_1, MU_B_CM_3, B_AU_T
from constants import KB, H, H_BAR, S_TIME_PS, M_AU

_DAWSN_ALIASES = {} 

def _register_dawsn_symbols():
    try:
        addr_d = int(get_cython_function_address(
            "scipy.special.cython_special", "__pyx_fuse_1dawsn"))
        llvm.add_symbol("pybridge_dawsn_d", addr_d)
        _DAWSN_ALIASES['f64'] = "pybridge_dawsn_d"
    except Exception:
        pass

    try:
        addr_f = int(get_cython_function_address(
            "scipy.special.cython_special", "__pyx_fuse_0dawsn"))
        llvm.add_symbol("pybridge_dawsn_f", addr_f)
        _DAWSN_ALIASES['f32'] = "pybridge_dawsn_f"
    except Exception:
        pass

@intrinsic
def dawsn(typingctx, x_ty):
    if x_ty not in (types.float32, types.float64):
        raise TypingError("dawsn(x): x must be float32 or float64")

    sig = x_ty(x_ty)

    def codegen(ctx: BaseContext, builder: IRBuilder, signature, args):
        (xval,) = args
        i32 = llir.IntType(32)
        zero = llir.Constant(i32, 0)

        if x_ty == types.float32 and 'f32' in _DAWSN_ALIASES:
            fnty = llir.FunctionType(llir.FloatType(),
                                     [llir.FloatType(), i32])
            callee = cgutils.get_or_insert_function(builder.module, fnty,
                                                    name=_DAWSN_ALIASES['f32'])
            return builder.call(callee, [xval, zero])

        fnty = llir.FunctionType(llir.DoubleType(),
                                 [llir.DoubleType(), i32])
        callee = cgutils.get_or_insert_function(builder.module, fnty,
                                                name=_DAWSN_ALIASES.get('f64', 'pybridge_dawsn_d'))

        x_as_f64 = (xval if x_ty == types.float64
                    else builder.fpext(xval, llir.DoubleType()))
        res_f64 = builder.call(callee, [x_as_f64, zero])
        if x_ty == types.float64:
            return res_f64
        else:
            return builder.fptrunc(res_f64, llir.FloatType())

    return sig, codegen

_register_dawsn_symbols()

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def dawsn_f(x):
    return dawsn(x)

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def kronecker_liouville(O: np.ndarray, N: int, N2: int):
    superoperator = np.zeros((N2, N2), dtype=np.complex128)
    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    if d == b:
                        superoperator[liou(a,b,N), liou(c,d,N)] += O[a,c]
                    if a == c:
                        superoperator[liou(a,b,N), liou(c,d,N)] -= O[d,b]
    return superoperator

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jhat_p_sec(w_ab, wq, n_q, d, cutoff):
    if w_ab > 0:
        return gaussian(w_ab - wq, d, cutoff) * n_q
    if w_ab < 0:
        return gaussian(w_ab + wq, d, cutoff) * (n_q + 1)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def lorentz_hilbert(E, d):
    return 1j / (E + 1j * d)

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def gauss_hilbert(E, d):
    factor_sqrt = E / (np.sqrt(2)*d)
    return np.sqrt(np.pi*0.5) / d * (np.exp(-(factor_sqrt*factor_sqrt)) + 2j / np.sqrt(np.pi) * dawsn_f(factor_sqrt))

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def build_Y_table_j(Yb):
    J, N, _ = Yb.shape
    out = np.empty((J, N, N, N, N), np.complex128)
    for j in range(J):
        Y  = Yb[j]
        Yh = np.conjugate(Y.T).copy()
        out_j = out[j]
        for a in range(N):
            Ya  = Y[a]
            Yha = Yh[a]
            for b in range(N):
                s1 = Ya[b]
                s2 = Yha[b]
                out_j[a, b, :, :] = s1 * Yh + s2 * Y
    return out

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def add_KR_bundle(out, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight):
    J = Jhat_p_table.shape[0]; N = Jhat_p_table.shape[1]
    coeff = H_BAR * 0.5 * weight
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y_j = Yb_table[j]
        Jhat_p_j = Jhat_p_table[j]
        Jhat_m_j = Jhat_m_table[j]
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
                                tmp+=Jhat_p_j[e,d]*Y_j[a,e,e,c] 
                            val+=tmp
                        val-=Jhat_p_j[a,d]*Y_j[a,c,d,b]
                        if a==c:
                            tmp=0.0+0.0j
                            for e in range(N):
                                tmp+=Jhat_m_j[c,e]*Y_j[d,e,e,b]
                            val+=tmp
                        val-=Jhat_m_j[c,b]*Y_j[a,c,d,b]
                        out[ab,cd]+=val * coeff

# @njit(nogil=True, cache=True, fastmath=True, inline="never")
# def add_rho0_bundle(out, trace, Yb_table, wb, nb, beta, w_n, q_0, rho, thread_id, t_index, weight, cutoff_j):
#     N=w_n.shape[0]; J=wb.size
#     coeff = 0.5 * weight
#     if q_0:
#         coeff *= 0.5
#     for j in range(J):
#         Y_j, wq, n_q, cutoff = Yb_table[j], wb[j], nb[j], cutoff_j[j]
#         for a in range(N):
#             rho_aa = rho[a, a]
#             for b in range(N):
#                 ab = liou(a, b, N)
#                 corr = 0.0 + 0.0j
#                 for e in range(N):
#                     w_de = w_n[a, e]
#                     w_eb = w_n[e, b]
#                     kernel = n_q * Iint(w_de + wq, w_eb - wq, beta, cutoff) + (n_q + 1.0) * Iint(w_de - wq, w_eb + wq, beta, cutoff)
#                     corr += kernel * Y_j[a, e, e, b] * rho_aa
#                     if b == a:
#                         trace[thread_id, t_index] += coeff * kernel * rho_aa * Y_j[a, e, e, a]
#                 out[ab] += coeff * corr

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def add_rho0_bundle(out, trace, Yb_table, wb, nb, beta, w_n,
                    q_0, rho, thread_id, t_index, weight, cutoff_j):
    N = w_n.shape[0]
    J = wb.size
    coeff = 0.5 * weight
    if q_0:
        coeff *= 0.5

    for j in range(J):
        Y_j = Yb_table[j]
        wq = wb[j]
        n_q = nb[j]
        cutoff = 1000 * cutoff_j[j]

        for a in range(N):
            for b in range(N):
                ab = liou(a, b, N)
                rho_bb = rho[b, b]
                corr = 0.0 + 0.0j
                for e in range(N):
                    w_ae = w_n[a, e]
                    w_eb = w_n[e, b]
                    k = 0.0
                    if np.abs(w_ae + wq) < cutoff and np.abs(w_eb - wq) < cutoff:
                        k += n_q * Iint(w_ae + wq, w_eb - wq, beta)
                    if np.abs(w_ae - wq) < cutoff and np.abs(w_eb + wq) < cutoff:
                        k += (n_q + 1.0) * Iint(w_ae - wq, w_eb + wq, beta)

                    if k != 0.0:
                        corr += k * Y_j[a, e, e, b] * rho_bb
                        if b == a:
                            trace[thread_id, t_index] += coeff * k * rho_bb * Y_j[a, e, e, a]

                out[ab] += coeff * corr



@njit(nogil=True, cache=True, fastmath=True, inline="always")
def zeta(x: float, beta: float) -> float:
    eps = 1e-15
    u = beta * x
    if abs(u) < eps:
        return beta
    if abs(u) > 650:
        return 0.0 + 1j * 0.0
    return np.expm1(u) / (x)

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Iint(w1: float, w2: float, beta: float) -> float:
    eps = 1e-15
    u1 = w1 * beta
    u2 = w2 * beta
    u12 = u1 + u2
    if abs(u1) > 650 or abs(u2) > 650 or abs(u12) > 650:
        return 0.0 + 1j * 0.0
    if abs(u1) < eps and abs(u2) < eps:
        return 0.5 * beta * beta
    elif abs(u2) < eps:
        num = np.exp(u1)
        return beta * num / (w1) - np.expm1(u1) / (w1**2)
    elif abs(u1) < eps:
        return np.expm1(u2) / (w2**2) - beta / (w2)
    elif abs(u12) < eps:
        return -(beta - np.expm1(u1) / (w1)) / (w1)
    term1 = np.expm1(u12) / (w2 * (w1 + w2))
    term2 = np.expm1(u1)  / (w1 * w2)
    return term1 - term2

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def bose_occ(freq: float, beta: float) -> float:
    u = beta * freq
    return 1.0/np.expm1(u)

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def liou(a, b, N):
    return a * N + b

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt, freq):
    for j in range(normal_modes.shape[1]):
        for i in range(dof_array.shape[0]):
            dof = dof_array[i]
            Y_q[j] += (1 / np.sqrt(freq[j]/H_CM_1)) * H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * 1/np.sqrt(M_AU) * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def _build_dynamical_matrix(hessian: np.ndarray, masses_inv_sqrt: np.ndarray, kpoint: np.ndarray):
    dyn_mat = np.zeros(masses_inv_sqrt.shape, dtype=np.complex128)

    for nx in range(hessian.shape[0]):
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

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def add_R21_bundle(out, Yb_table, w_n, Jhat_p_table, q_0, sec_tol, weight):
    N, J = w_n.shape[0], Jhat_p_table.shape[0]
    prefc = pi / H * weight # pi * pi / H with Jhat_p_sec
    if q_0:
        prefc *= 0.5
    for j in range(J):
        Y_j = Yb_table[j]
        Jhat_p_j = Jhat_p_table[j]
        for a in range(N):
            for b in range(N):
                ab = liou(a,b,N)
                for c in range(N):
                    for d in range(N):
                        cd = liou(c,d,N)
                        if abs(w_n[a,c]+w_n[d,b]) > sec_tol:
                            continue
                        val = 0.0 + 0.0j
                        val += Y_j[a,c,d,b]*Jhat_p_j[b,d]
                        val += Y_j[a,c,d,b]*Jhat_p_j[a,c]
                        if d == b:
                            for e in range(N):
                                val -= Y_j[a,e,e,c]*Jhat_p_j[e,c]
                        if c == a:
                            for f in range(N):
                                val -= Y_j[f,b,d,f]*Jhat_p_j[f,d]
                        out[ab,cd] += prefc * val

@njit(nogil=True, cache=True, fastmath=True, inline="never")
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

@njit(nogil=True, cache=True, inline="always", fastmath=True)
def gaussian(dE: float, lw: float, cutoff: float) -> float:
    if np.abs(dE) >= cutoff:
        return 0.0
    prefactor = 1 / (np.sqrt(2 * np.pi) * lw)
    exponent = -0.5 * dE * dE / (lw * lw)
    return prefactor * np.exp(exponent)

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def add_R41(out: np.ndarray, w_n: np.ndarray, V1: np.ndarray, V2: np.ndarray, n1: float, n2: float, f1: float, f2: float, lw1: float, lw2: float, ind: int, cutoff: float, weight: float, sec_tol: float = 1e-6) -> np.ndarray:
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
                    val += (np.conj(Rabp[b,d]) * Rabp[a,c] + np.conj(Rabp[b,d]) * Rbam[a,c] + np.conj(Rbam[b,d]) * Rabp[a,c] + np.conj(Rbam[b,d]) * Rbam[a,c]) * G 
                    G = n2 * (n1 + 1.0) * gaussian(w_n[a,c] + f1 - f2, lw_total, cutoff)
                    val += (np.conj(Rabm[b,d]) * Rabm[a,c] + np.conj(Rabm[b,d]) * Rbap[a,c] + np.conj(Rbap[b,d]) * Rabm[a,c] + np.conj(Rbap[b,d]) * Rbap[a,c]) * G 
                    G = n1 * n2 * gaussian(w_n[a,c] - f1 - f2, lw_total, cutoff)
                    val += (np.conj(Rabm[b,d]) * Rabm[a,c] + np.conj(Rbam[b,d]) * Rbam[a,c] + np.conj(Rbam[b,d]) * Rabm[a,c] + np.conj(Rabm[b,d]) * Rbam[a,c]) * G
                    G = (n1 + 1.0) * (n2 + 1.0) * gaussian(w_n[a,c] + f1 + f2, lw_total, cutoff)
                    val += (np.conj(Rabp[b,d]) * Rabp[a,c] + np.conj(Rbap[b,d]) * Rbap[a,c] + np.conj(Rbap[b,d]) * Rabp[a,c] + np.conj(Rabp[b,d]) * Rbap[a,c]) * G

                    if a == c:
                        for k in range(N):
                            G = n1 * (n2 + 1.0) * gaussian(w_n[k,d] - f1 + f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabp[k,d]) * Rabp[k,b] + np.conj(Rabp[k,d]) * Rbam[k,b] + np.conj(Rbam[k,d]) * Rabp[k,b] + np.conj(Rbam[k,d]) * Rbam[k,b]) * G
                            G = n2 * (n1 + 1.0) * gaussian(w_n[k,d] + f1 - f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabm[k,d]) * Rabm[k,b] + np.conj(Rabm[k,d]) * Rbap[k,b] + np.conj(Rbap[k,d]) * Rabm[k,b] + np.conj(Rbap[k,d]) * Rbap[k,b]) * G
                            G = n1 * n2 * gaussian(w_n[k,d] - f1 - f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabm[k,d]) * Rabm[k,b] + np.conj(Rabm[k,d]) * Rbam[k,b] + np.conj(Rbam[k,d]) * Rabm[k,b] + np.conj(Rbam[k,d]) * Rbam[k,b]) * G
                            G = (n1 + 1.0) * (n2 + 1.0) * gaussian(w_n[k,d] + f1 + f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabp[k,d]) * Rabp[k,b] + np.conj(Rabp[k,d]) * Rbap[k,b] + np.conj(Rbap[k,d]) * Rabp[k,b] + np.conj(Rbap[k,d]) * Rbap[k,b]) * G
                    if b == d:
                        for k in range(N):
                            G = n1 * (n2 + 1.0) * gaussian(w_n[k,c] - f1 + f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabp[k,a]) * Rabp[k,c] + np.conj(Rabp[k,a]) * Rbam[k,c] + np.conj(Rbam[k,a]) * Rabp[k,c] + np.conj(Rbam[k,a]) * Rbam[k,c]) * G
                            G = n2 * (n1 + 1.0) * gaussian(w_n[k,c] + f1 - f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabm[k,a]) * Rabm[k,c] + np.conj(Rabm[k,a]) * Rbap[k,c] + np.conj(Rbap[k,a]) * Rabm[k,c] + np.conj(Rbap[k,a]) * Rbap[k,c]) * G
                            G = n1 * n2 * gaussian(w_n[k,c] - f1 - f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabm[k,a]) * Rabm[k,c] + np.conj(Rabm[k,a]) * Rbam[k,c] + np.conj(Rbam[k,a]) * Rabm[k,c] + np.conj(Rbam[k,a]) * Rbam[k,c]) * G
                            G = (n1 + 1.0) * (n2 + 1.0) * gaussian(w_n[k,c] + f1 + f2, lw_total, cutoff)
                            val -= 0.5 * (np.conj(Rabp[k,a]) * Rabp[k,c] + np.conj(Rabp[k,a]) * Rbap[k,c] + np.conj(Rbap[k,a]) * Rabp[k,c] + np.conj(Rbap[k,a]) * Rbap[k,c]) * G

                    out[ind, ab, cd] += prefc * val

# @njit(nogil=True, cache=True, fastmath=True, inline="never")
# def add_R41(out: np.ndarray,
#             w_n: np.ndarray,
#             V1: np.ndarray,
#             V2: np.ndarray,
#             n1: float,
#             n2: float,
#             f1: float,
#             f2: float,
#             lw1: float,
#             lw2: float,
#             ind: int,
#             cutoff: float,
#             A: float,
#             B: float,
#             weight: float,
#             sec_tol: float = 1e-6) -> np.ndarray:
#     """
#     Raman 4th–order population kernel (secular, Markov) in Liouville space.

#     - out:  R41 slice for all threads, shape (threads, N^2, N^2)
#     - w_n:  energy differences w_n[a,b] = E_a - E_b
#     - V1,V2: spin–phonon coupling matrices for the two modes (complex N×N)
#     - n1,n2: Bose occupancies of modes with frequencies f1,f2
#     - f1,f2: phonon frequencies
#     - lw1,lw2: broadenings (used in denominators and Gaussian δ)
#     - ind:   thread id (first index of `out`)
#     - cutoff: energy cutoff for Gaussian δ
#     - A,B:   degeneracy factors (A for ++/--, B for +-/-+)
#     - weight: integration weight for this phonon pair
#     - sec_tol: secular tolerance on level spacing |ω_ab|
#     """

#     N = w_n.shape[0]

#     # Local population rate matrix R_ab (a,b = 0..N-1), for this phonon pair
#     # R[a,b] is the rate from level b to a.
#     R = np.zeros((N, N), dtype=np.float64)

#     # Loop over level pairs (a,b). Only populations (a,a) ← (b,b) will be non-zero.
#     for a in range(N):
#         for b in range(N):

#             w_ab = w_n[a, b]  # E_a - E_b

#             # Secular approximation: skip nearly degenerate transitions
#             if np.abs(w_ab) < sec_tol:
#                 continue

#             # Build complex amplitudes for the four Raman channels:
#             # Rpm, Rmp, Rpp, Rmm  (following the Fortran logic)
#             Rpm = 0.0 + 0.0j
#             Rmp = 0.0 + 0.0j
#             Rpp = 0.0 + 0.0j
#             Rmm = 0.0 + 0.0j

#             for c in range(N):
#                 V1ac = V1[a, c]
#                 V2cb = V2[c, b]
#                 V2ac = V2[a, c]
#                 V1cb = V1[c, b]

#                 w_cb = w_n[c, b]  # E_c - E_b

#                 # --- "+-" channel (pm) ---
#                 #   1st term: V1 V2 / (E_c - E_b - f2 - i lw2)
#                 #   2nd term: V2 V1 / (E_c - E_b + f1 - i lw1)
#                 Rpm += V1ac * V2cb / (w_cb - f2 - 1j * lw2)
#                 Rpm += V2ac * V1cb / (w_cb + f1 - 1j * lw1)

#                 # --- "-+" channel (mp) ---
#                 #   1st term: V1 V2 / (E_c - E_b + f2 - i lw2)
#                 #   2nd term: V2 V1 / (E_c - E_b - f1 - i lw1)
#                 Rmp += V1ac * V2cb / (w_cb + f2 - 1j * lw2)
#                 Rmp += V2ac * V1cb / (w_cb - f1 - 1j * lw1)

#                 # --- "++" channel (pp) ---
#                 #   both denominators with +f1,+f2
#                 Rpp += V1ac * V2cb / (w_cb + f2 - 1j * lw2)
#                 Rpp += V2ac * V1cb / (w_cb + f1 - 1j * lw1)

#                 # --- "--" channel (mm) ---
#                 #   both denominators with -f1,-f2
#                 Rmm += V1ac * V2cb / (w_cb - f2 - 1j * lw2)
#                 Rmm += V2ac * V1cb / (w_cb - f1 - 1j * lw1)

#             lw_delta = lw1

#             DE_pm = w_ab - f2 + f1
#             G_pm = n2 * (n1 + 1.0) * gaussian(DE_pm, lw_delta, cutoff)

#             DE_mp = w_ab + f2 - f1
#             G_mp = (n2 + 1.0) * n1 * gaussian(DE_mp, lw_delta, cutoff)

#             DE_mm = w_ab - f2 - f1
#             G_mm = n2 * n1 * gaussian(DE_mm, lw_delta, cutoff)

#             DE_pp = w_ab + f2 + f1
#             G_pp = (n2 + 1.0) * (n1 + 1.0) * gaussian(DE_pp, lw_delta, cutoff)

#             Rpm2 = (Rpm.real * Rpm.real) + (Rpm.imag * Rpm.imag)
#             Rmp2 = (Rmp.real * Rmp.real) + (Rmp.imag * Rmp.imag)
#             Rmm2 = (Rmm.real * Rmm.real) + (Rmm.imag * Rmm.imag)
#             Rpp2 = (Rpp.real * Rpp.real) + (Rpp.imag * Rpp.imag)

#             rate_ab = 0.0
#             rate_ab += B * (G_pm * Rpm2 + G_mp * Rmp2)
#             rate_ab += A * (G_mm * Rmm2 + G_pp * Rpp2)

#             R[a, b] += rate_ab

#     prefc = pi * pi / (H*H*H) * weight

#     for a in range(N):
#         for b in range(N):
#             R[a, b] *= prefc

#     # Trace preservation for populations:
#     # for each "source" level b, impose Σ_a R[a,b] = 0
#     for b in range(N):
#         loss = 0.0
#         for a in range(N):
#             if a != b:
#                 loss += R[a, b]
#         R[b, b] -= loss

#     for a in range(N):
#         ab = liou(a, a, N)
#         for b in range(N):
#             cd = liou(b, b, N)
#             out[ind, ab, cd] += R[a, b]

#     return out


@njit(nogil=True, cache=True, fastmath=True, inline="never")
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

def hilbert(x, d, kind):
    raise NotImplementedError

@overload(hilbert, nogil=True, fastmath=True, cache=True, inline="always", prefer_literal=True)
def ov_hilbert(E, d, kind):
    if isinstance(kind, types.Literal):
        if kind.literal_value == 1:
            def impl(E, d, kind):
                return gauss_hilbert(E, d)
            return impl
        elif kind.literal_value == 0:
            def impl(E, d, kind):
                return lorentz_hilbert(E, d)
            return impl

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jhat_p_symm(w_ab, wq, n_q, d, cutoff, kind):
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff:
        return hilbert(w_ab - wq, d, kind) * n_q
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff:
        return hilbert(w_ab + wq, d, kind) * (n_q + 1)
    return 0.0 + 0.0 * 1j 

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jhat_p(w_ab, wq, n_q, d, cutoff, kind):
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff: # and w_ab - wq < 0:
        return hilbert(w_ab - wq, d, kind) * n_q
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff: # and w_ab + wq > 0:
        return hilbert(w_ab + wq, d, kind) * (n_q + 1)
    return 0.0 + 0.0 * 1j 

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jhat_m(w_ab, wq, n_q, d, cutoff, kind):
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff: # and w_ab + wq > 0:
        return hilbert(w_ab + wq, d, kind) * n_q
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff: # and w_ab - wq < 0:
        return hilbert(w_ab - wq, d, kind) * (n_q + 1)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind):
    u = w_cd + w_ab
    if u < 0 and np.abs(u + wq) < cutoff: # and u + wq > 0:
        return hilbert(u + wq, d, kind) * n_q * zeta(w_ab + wq, beta)
    if u > 0 and np.abs(u - wq) < cutoff: # and u - wq < 0:
        return hilbert(u - wq, d, kind) * (n_q + 1) * zeta(w_ab - wq, beta)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def build_Jp_symm_table_j(w_n, wb, nb, delta, cutoff, kind):
    N = w_n.shape[0]
    J = wb.shape[0]
    Jp = np.empty((J, N, N), np.complex128)

    for j in range(J):
        wq, n_q, delta_j, cutoff_j = wb[j], nb[j], delta[j], cutoff[j]
        for a in range(N):
            for b in range(N):
                x = w_n[a, b]
                Jp[j, a, b] = Jhat_p_symm(x,wq,n_q,delta_j,cutoff_j,kind)
    return Jp

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def build_Jp_Jm_tables_j(w_n, wb, nb, delta, cutoff, kind):
    N = w_n.shape[0]
    J = wb.shape[0]
    Jp = np.empty((J, N, N), np.complex128)
    Jm = np.empty((J, N, N), np.complex128)

    for j in range(J):
        wq, n_q, delta_j, cutoff_j = wb[j], nb[j], delta[j], cutoff[j]
        for a in range(N):
            for b in range(N):
                x = w_n[a, b]
                Jp[j, a, b] = Jhat_p(x,wq,n_q,delta_j,cutoff_j,kind)
                Jm[j, a, b] = Jhat_m(x,wq,n_q,delta_j,cutoff_j,kind)
    return Jp, Jm

@njit(nogil=True, cache=True, fastmath=True, inline="never")
def add_PSI_bundle(out, A, Yb_table, wb, nb, delta, beta, w_n, q_0, cutoff, weight, kind):
    N=w_n.shape[0]; J=wb.size
    coeff = H_BAR * 0.5 * weight
    if q_0:
        coeff *= 0.5
    for j in range(J):
        Y_j, wq, n_q, delta_j, cutoff_j = Yb_table[j], wb[j], nb[j], delta[j], cutoff[j]
        for a in range(N):
            for b in range(N):
                ab=liou(a,b,N)
                for c in range(N):
                    d = c
                    cd=liou(c,d,N)
                    val=0.0+0.0j
                    for e in range(N):
                        val+=Jcorr(w_n[e,c],w_n[d,b],wq,n_q,delta_j,beta,cutoff_j,kind)*A[e,c]*Y_j[d,b,a,e]
                        val-=Jcorr(w_n[a,d],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j,kind)*A[a,c]*Y_j[d,e,e,b]
                        if a == c:
                            for f in range(N):
                                val+=Jcorr(w_n[e,f],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j,kind)*A[e,f]*Y_j[f,b,d,e]
                        val-=Jcorr(w_n[e,b],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j,kind)*Y_j[a,c,d,e]*A[e,b]
                    out[ab,cd]+=val*coeff

@njit(nogil=True, fastmath=True, cache=True, inline="always")
def cutoff_j_direct(fwhm, cutoff_mult):
    return fwhm * cutoff_mult

@njit(nogil=True, fastmath=True, cache=True, inline="always")
def cutoff_j_no_direct(fwhm, cutoff_mult, wb, w_n_direct_max):
    return np.minimum(fwhm * cutoff_mult, np.abs(wb + 1.01 * w_n_direct_max))

def get_cutoff_j(fwhm_j, cutoff_mult, wb, w_n_direct_max, direct):
    raise NotImplementedError

@overload(get_cutoff_j, nogil=True, fastmath=True, cache=True, inline="always", prefer_literal=True)
def ov_get_cutoff_j(fwhm, cutoff_mult, wb, w_n_direct_max, direct):
    if isinstance(direct, types.Literal):
        if direct.literal_value == 0:
            def impl(fwhm, cutoff_mult, wb, w_n_direct_max, direct):
                return cutoff_j_no_direct(fwhm, cutoff_mult, wb, w_n_direct_max)
            return impl
        elif direct.literal_value == 1:
            def impl(fwhm, cutoff_mult, wb, w_n_direct_max, direct):
                return cutoff_j_direct(fwhm, cutoff_mult)
            return impl
        
def add_KR_bundle_comp(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight, run_KR):
    raise NotImplementedError

@overload(add_KR_bundle_comp, nogil=True, fastmath=True, cache=True, inline="never", prefer_literal=True)
def ov_add_KR_bundle_comp(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight, run_KR):
    if isinstance(run_KR, types.Literal):
        if run_KR.literal_value == 0:
            def impl(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight, run_KR):
                return
            return impl
        elif run_KR.literal_value == 1:
            def impl(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight, run_KR):
                return add_KR_bundle(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight)
            return impl

def add_PSI_bundle_comp(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind, run_PSI):
    raise NotImplementedError

@overload(add_PSI_bundle_comp, nogil=True, fastmath=True, cache=True, inline="never", prefer_literal=True)
def ov_add_PSI_bundle_comp(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind, run_PSI):
    if isinstance(run_PSI, types.Literal):
        if run_PSI.literal_value == 0:
            def impl(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind, run_PSI):
                return
            return impl
        elif run_PSI.literal_value == 1:
            def impl(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind, run_PSI):
                return add_PSI_bundle(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind)
            return impl

def add_rho0_bundle_comp(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff, run_rho0):
    raise NotImplementedError

@overload(add_rho0_bundle_comp, nogil=True, fastmath=True, cache=True, inline="never", prefer_literal=True)
def ov_add_rho0_bundle_comp(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff, run_rho0):
    if isinstance(run_rho0, types.Literal):
        if run_rho0.literal_value == 0:
            def impl(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff, run_rho0):
                return
            return impl
        elif run_rho0.literal_value == 1:
            def impl(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff, run_rho0):
                return add_rho0_bundle(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff)
            return impl

def add_R21_bundle_comp(R21_i_t, Yb_table, w_n, Jhat_p_table, q_0, degeneracy_tolerance, weight, run_R21):
    raise NotImplementedError

@overload(add_R21_bundle_comp, nogil=True, fastmath=True, cache=True, inline="never", prefer_literal=True)
def ov_add_R21_bundle_comp(R21_i_t, Yb_table, w_n, Jhat_p_table, q_0, degeneracy_tolerance, weight, run_R21):
    if isinstance(run_R21, types.Literal):
        if run_R21.literal_value == 0:
            def impl(R21_i_t, Yb_table, w_n, Jhat_p_table, q_0, degeneracy_tolerance, weight, run_R21):
                return
            return impl
        elif run_R21.literal_value == 1:
            def impl(R21_i_t, Yb_table, w_n, Jhat_p_table, q_0, degeneracy_tolerance, weight, run_R21):
                return add_R21_bundle(R21_i_t, Yb_table, w_n, Jhat_p_table, q_0, degeneracy_tolerance, weight)
            return impl

def build_matrices(
    hessian: np.ndarray,
    masses_inv_sqrt: np.ndarray,
    dof_array: np.ndarray,
    H_grad: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    gamma_fwhm: float,
    beta: float,
    w_n: np.ndarray,
    M_KR: np.ndarray,
    M_PSI: np.ndarray,
    A_e: np.ndarray,
    rho_vec_init: np.ndarray,
    M_rho0_trace: np.ndarray,
    rho_mat: np.ndarray,
    R21: np.ndarray,
    R41: np.ndarray,
    cutoff_mult: float,
    degeneracy_tolerance: float,
    modes_low: float,
    modes_high: float,
    kind: types.Literal,
    direct: types.Literal,
    run_KR: types.Literal,
    run_PSI: types.Literal,
    run_rho0: types.Literal,
    run_R21: types.Literal,
    run_R41: types.Literal,
    n_k: int,
    ):
    raise NotImplementedError

@overload(build_matrices, nogil=True, fastmath=True, cache=True, inline="never", prefer_literal=True)
def ov_build_matrices(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                      R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k):
    if isinstance(run_R41, types.Literal):
        if run_R41.literal_value == 0:
            def impl(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                     R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k):
                return build_matrices_no_R41(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                                             R21, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, n_k)
            return impl
        elif run_R41.literal_value == 1:
            def impl(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                     R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k):
                return build_matrices_R41(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                                          R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, n_k)
            return impl

@njit(nogil=True, fastmath=True, parallel=True, inline="never")
def build_matrices_no_R41(
    hessian: np.ndarray,
    masses_inv_sqrt: np.ndarray,
    dof_array: np.ndarray,
    H_grad: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    gamma_fwhm: float,
    beta: float,
    w_n: np.ndarray,
    M_KR: np.ndarray,
    M_PSI: np.ndarray,
    A_e: np.ndarray,
    rho_vec_init: np.ndarray,
    M_rho0_trace: np.ndarray,
    rho_mat: np.ndarray,
    R21: np.ndarray,
    cutoff_mult: float,
    degeneracy_tolerance: float,
    modes_low: float,
    modes_high: float,
    kind: types.Literal,
    direct: types.Literal,
    run_KR: types.Literal,
    run_PSI: types.Literal,
    run_rho0: types.Literal,
    run_R21: types.Literal,
    n_k: int,
    ):
    n_k_inv = 1.0 / np.sqrt(n_k)
    masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)
    gamma = np.asarray([0.0, 0.0, 0.0])
    freq0, modes0 = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, gamma))
    arr = np.diag(w_n, k=1)
    w_n_direct_max = arr[0]
    for i in range(2, arr.size, 2):
        if arr[i] < w_n_direct_max:
            w_n_direct_max = arr[i]

    for i in prange(grid.shape[0]):
        thread_id = get_thread_id()
        q = grid[i]
        weight = weights[i]
        q_0 = np.allclose(q, gamma, atol=1e-6)
        freq, modes = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, q))
        freq *= AU_BOHR_CM_1

        if q_0:
            freq, modes = freq[3:], modes[:,3:]
        mask = (freq >= modes_low) & (freq <= modes_high)
        idx  = np.where(mask)[0]
        wb, modes = np.ascontiguousarray(freq[idx]), np.ascontiguousarray(modes[:,idx])
        if wb.shape[0] == 0:
            continue
        Yb = np.zeros((wb.size, H_grad.shape[1], H_grad.shape[2]), dtype=np.complex128)
        get_Y_q(Yb, H_grad, modes, q, dof_array, masses_inv_sqrt, n_k_inv, wb)
        Yb_table = build_Y_table_j(Yb)

        fwhm_j = gamma_fwhm * np.ones_like(wb) # TODO: can implement ab inito model for different wb and T (move into the loop) - adaptive_fwhm in config
        cutoff_j = get_cutoff_j(fwhm_j, cutoff_mult, wb, w_n_direct_max, direct)

        for t_index in range(beta.shape[0]):
            beta_t = beta[t_index]
            rho_mat_t = rho_mat[t_index]

            M_KR_i_t = M_KR[thread_id, t_index]
            M_PSI_i_t = M_PSI[thread_id, t_index]
            rho_vec_init_i_t = rho_vec_init[thread_id, t_index]
            R21_i_t = R21[thread_id, t_index]

            bose = bose_occ(wb, beta[t_index])

            Jhat_p_table, Jhat_m_table = build_Jp_Jm_tables_j(w_n, wb, bose, fwhm_j, cutoff_j, kind)
            if run_R21:
                Jhat_p_symm_table = build_Jp_symm_table_j(w_n, wb, bose, fwhm_j, cutoff_j, kind)

            add_KR_bundle_comp(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight, run_KR)
            add_PSI_bundle_comp(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind, run_PSI)
            add_rho0_bundle_comp(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff_j, run_rho0)
            add_R21_bundle_comp(R21_i_t, Yb_table, w_n, Jhat_p_symm_table, q_0, degeneracy_tolerance, weight, run_R21)

@njit(nogil=True, fastmath=True, parallel=True, inline="never")
def build_matrices_R41(
    hessian: np.ndarray,
    masses_inv_sqrt: np.ndarray,
    dof_array: np.ndarray,
    H_grad: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    gamma_fwhm: float,
    beta: float,
    w_n: np.ndarray,
    M_KR: np.ndarray,
    M_PSI: np.ndarray,
    A_e: np.ndarray,
    rho_vec_init: np.ndarray,
    M_rho0_trace: np.ndarray,
    rho_mat: np.ndarray,
    R21: np.ndarray,
    R41: np.ndarray,
    cutoff_mult: float,
    degeneracy_tolerance: float,
    modes_low: float,
    modes_high: float,
    kind: types.Literal,
    direct: types.Literal,
    run_KR: types.Literal,
    run_PSI: types.Literal,
    run_rho0: types.Literal,
    run_R21: types.Literal,
    n_k: int,
    ):
    n_k_inv = 1.0 / np.sqrt(n_k)
    masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)
    gamma = np.asarray([0.0, 0.0, 0.0])
    freq0, modes0 = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, gamma))
    freq_shape = freq0.shape[0]
    arr = np.diag(w_n, k=1)
    w_n_direct_max = arr[0]
    for i in range(2, arr.size, 2):
        if arr[i] < w_n_direct_max:
            w_n_direct_max = arr[i]
    
    threads_number = get_num_threads()
    max_grid_per_thread = np.int64(np.ceil(grid.shape[0]/threads_number))
    max_grid_size = 2*max_grid_per_thread*freq_shape
    reshape_size = threads_number * max_grid_size

    Yb_array = np.zeros((threads_number, max_grid_size, H_grad.shape[1], H_grad.shape[2]), np.complex128)
    wb_array = np.zeros((threads_number, max_grid_size), np.float64)
    wq_array = np.zeros((threads_number, max_grid_size), np.float64)
    raman_counter = np.zeros(threads_number, dtype=np.int64)
    raman_counter_wb = np.zeros(threads_number, dtype=np.int64)
    raman_counter_2wb = np.zeros(threads_number, dtype=np.int64)

    for i in prange(grid.shape[0]):
        thread_id = get_thread_id()
        q = grid[i]
        weight = weights[i]
        q_0 = np.allclose(q, gamma, atol=1e-6)
        freq, modes = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, q))
        freq *= AU_BOHR_CM_1

        if q_0:
            freq, modes = freq[3:], modes[:,3:]
        mask = (freq >= modes_low) & (freq <= modes_high)
        idx  = np.where(mask)[0]
        wb, modes = np.ascontiguousarray(freq[idx]), np.ascontiguousarray(modes[:,idx])
        if wb.shape[0] == 0:
            continue
        Yb = np.zeros((wb.size, H_grad.shape[1], H_grad.shape[2]), dtype=np.complex128)
        get_Y_q(Yb, H_grad, modes, q, dof_array, masses_inv_sqrt, n_k_inv, wb)
        Yb_table = build_Y_table_j(Yb)

        fwhm_j = gamma_fwhm * np.ones_like(wb) # TODO: can implement ab inito model for different wb and T (move into the loop) - adaptive_fwhm in config
        cutoff_j = get_cutoff_j(fwhm_j, cutoff_mult, wb, w_n_direct_max, direct)

        raman_counter_wb[thread_id] = raman_counter[thread_id] + wb.shape[0]
        raman_counter_2wb[thread_id] = raman_counter_wb[thread_id] + wb.shape[0]
        Yb_array[thread_id,raman_counter[thread_id]:raman_counter_wb[thread_id]] = Yb
        wb_array[thread_id,raman_counter[thread_id]:raman_counter_wb[thread_id]] = wb
        if not q_0:
            Yb_array[thread_id,raman_counter_wb[thread_id]:raman_counter_2wb[thread_id]] = np.conjugate(np.transpose(Yb, (0,2,1)))
            wb_array[thread_id,raman_counter_wb[thread_id]:raman_counter_2wb[thread_id]] = wb
        
        # Store the BZ weight for each phonon entry
        s = raman_counter[thread_id]
        m = raman_counter_wb[thread_id]
        e = raman_counter_2wb[thread_id]

        for jj in range(s, m):
            wq_array[thread_id, jj] = weight

        if not q_0:
            for jj in range(m, e):
                wq_array[thread_id, jj] = weight

        raman_counter[thread_id] = raman_counter_2wb[thread_id]

        for t_index in range(beta.shape[0]):
            beta_t = beta[t_index]
            rho_mat_t = rho_mat[t_index]

            M_KR_i_t = M_KR[thread_id, t_index]
            M_PSI_i_t = M_PSI[thread_id, t_index]
            rho_vec_init_i_t = rho_vec_init[thread_id, t_index]
            R21_i_t = R21[thread_id, t_index]

            bose = bose_occ(wb, beta[t_index])

            Jhat_p_table, Jhat_m_table = build_Jp_Jm_tables_j(w_n, wb, bose, fwhm_j, cutoff_j, kind)
            if run_R21:
                Jhat_p_symm_table = build_Jp_symm_table_j(w_n, wb, bose, fwhm_j, cutoff_j, kind)

            add_KR_bundle_comp(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight, run_KR)
            add_PSI_bundle_comp(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight, kind, run_PSI)
            add_rho0_bundle_comp(rho_vec_init_i_t, M_rho0_trace, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index, weight, cutoff_j, run_rho0)
            add_R21_bundle_comp(R21_i_t, Yb_table, w_n, Jhat_p_symm_table, q_0, degeneracy_tolerance, weight, run_R21)

    wb_array = wb_array.reshape((reshape_size))
    Yb_array = Yb_array.reshape((reshape_size, H_grad.shape[1], H_grad.shape[2]))
    wq_array = wq_array.reshape((reshape_size))
    N_pairs = wb_array.size * (wb_array.size + 1) // 2
    cutoff_raman = gamma_fwhm * cutoff_mult
    for t_index_raman in range(beta.shape[0]):
        bose_raman = bose_occ(wb_array, beta[t_index_raman])
        for p in prange(N_pairs):
            thread_id = get_thread_id()
            R_41_t = R41[:,t_index_raman,:,:]
            k = np.int64((np.sqrt(8*p + 1) - 1) // 2)
            l = p - k*(k + 1)//2
            if wb_array[k] == 0.0 or wb_array[l] == 0.0:
                continue
            raman_weight = wq_array[k] * wq_array[l]
            add_R41(R_41_t, w_n, Yb_array[k], Yb_array[l], bose_raman[k], bose_raman[l], wb_array[k],
                    wb_array[l], gamma_fwhm, gamma_fwhm, thread_id, cutoff_raman, raman_weight,
                    sec_tol=degeneracy_tolerance)

@njit(nogil=True, fastmath=True, cache=True, inline="never")
def solve_susceptibility(omega_grid, Xi, num, N, t, B_e, chi_T, chi_isothermal, chi_adiabatic, eye, normalize):
    for k, omega in enumerate(omega_grid):
        Xi_temp = Xi - 1j * omega * eye
        rho_hat  = np.linalg.solve(Xi_temp, num).reshape((N, N))
        chi_T[t,k]   = 1j / H_BAR * np.trace(B_e @ rho_hat) * MU_B_CM_3
        if normalize:    
            if k != 0:
                chi_T[t,k] /= chi_T[t,0].real
                chi_T[t,k] *= (chi_isothermal[t] - chi_adiabatic[t])
                chi_T[t,k] += chi_adiabatic[t]
    if normalize:
        chi_T[t,0] = chi_T[t,0] / chi_T[t,0].real * chi_isothermal[t]

@njit(nogil=True, fastmath=True, inline="never")
def susceptibility_relax_time(
    omega_grid: np.ndarray,
    E: np.ndarray,
    A_e: np.ndarray,
    B_e: np.ndarray,
    H_grad: np.ndarray,
    T: np.ndarray,
    hessian: np.ndarray,
    masses_inv_sqrt: np.ndarray,
    dof_array: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    gamma_fwhm: float,
    chi_isothermal: np.ndarray,
    chi_adiabatic: np.ndarray,
    cutoff_mult: float,
    degeneracy_tolerance: float,
    states_number: int,
    modes_low: float,
    modes_high: float,
    threads: int,
    kind: int,
    direct: int,
    run_KR: int,
    run_PSI: int,
    run_rho0: int,
    run_R21: int,
    run_R41: int,
    n_k: int,
):  
    kind_lit = literally(kind)
    direct_lit = literally(direct)
    run_KR_lit = literally(run_KR)
    run_PSI_lit = literally(run_PSI)
    run_rho0_lit = literally(run_rho0)
    run_R21_lit = literally(run_R21)
    run_R41_lit = literally(run_R41)

    set_num_threads(threads)
    threads = get_num_threads()
    omega_grid = omega_grid / S_TIME_PS
    beta = 1.0 / (KB * T)
    temp_size = beta.shape[0]

    A_e = np.ascontiguousarray(A_e[:states_number, :states_number])
    B_e = np.ascontiguousarray(B_e[:states_number, :states_number])

    E = E[:states_number]
    E = (E - np.min(E))
    for i in range(E.shape[0]):
        for j in range(i+1, E.shape[0]):
            if np.isclose(E[i], E[j], atol=degeneracy_tolerance):
                E[i] = (E[i] + E[j]) * 0.5
                E[j] = E[i]
    E = (E - np.min(E))
    print(E)

    N  = states_number
    N2 = N * N
    H_s = np.diag(E)
    M_L = kronecker_liouville(H_s, N, N2)
    M_rho0 = kronecker_liouville(A_e, N, N2) 
    rho_vec = np.zeros((temp_size, N2), dtype=np.complex128)
    rho_mat = np.zeros((temp_size, N, N), dtype=np.complex128)
    for t in range(temp_size):
        rho_eq = np.exp(-beta[t] * E)
        rho_eq /= rho_eq.sum()
        rho_mat[t] = np.diag(rho_eq)
        for c in range(N):
            for d in range(N):
                rho_vec[t,liou(c, d, N)] += rho_mat[t,c,d]

    M_KR = np.zeros((threads, temp_size, N2, N2), dtype=np.complex128)
    M_PSI = np.zeros((threads, temp_size, N2, N2), dtype=np.complex128)
    rho_vec_init = np.zeros((threads, temp_size, N2), dtype=np.complex128)
    M_rho0_trace = np.zeros((threads, temp_size), dtype=np.complex128)
    R21 = np.zeros((threads, temp_size, N2, N2), dtype=np.complex128)
    R41 = np.zeros((threads, temp_size, N2, N2), dtype=np.complex128)

    w_n = E[:,np.newaxis] - E[np.newaxis,:]
    
    build_matrices(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights,
                    gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace,
                    rho_mat, R21, R41, cutoff_mult, degeneracy_tolerance, modes_low,
                    modes_high, kind_lit, direct_lit, run_KR_lit, run_PSI_lit, run_rho0_lit,
                    run_R21_lit, run_R41_lit, n_k)

    M_KR = np.sum(M_KR, axis=0)
    M_PSI = np.sum(M_PSI, axis=0)
    rho_vec_init = np.sum(rho_vec_init, axis=0)
    M_rho0_trace = 1.0 + np.sum(M_rho0_trace, axis=0)
    R21 = np.sum(R21, axis=0)
    R41 = np.sum(R41, axis=0)

    eye = np.eye(N2, dtype=np.complex128)

    chi_T = np.empty((4,temp_size, omega_grid.shape[0]), dtype=np.complex128)
    relax_time_R21_T = np.empty(temp_size, dtype=np.float64)
    relax_time_R41_T = np.empty(temp_size, dtype=np.float64)

    for t in range(temp_size):
        relax_time_R21_T[t] = get_relax_time(R21[t])
        relax_time_R41_T[t] = get_relax_time(R41[t])

        if run_KR_lit:
            Xi = 1j / H_BAR * M_L + M_KR[t] / (H_BAR ** 2)
            num = M_rho0 @ rho_vec[t]
            solve_susceptibility(omega_grid, Xi, num, N, t, B_e, chi_T[0], chi_isothermal, chi_adiabatic, eye, True)
            solve_susceptibility(omega_grid, Xi, num, N, t, B_e, chi_T[1], chi_isothermal, chi_adiabatic, eye, False)

            if run_PSI_lit:
                num = (1j / H_BAR * M_PSI[t]) @ rho_vec[t] + M_rho0 @ rho_vec[t]
                solve_susceptibility(omega_grid, Xi, num, N, t, B_e, chi_T[2], chi_isothermal, chi_adiabatic, eye, False)

                if run_rho0_lit:
                    rho_vec_t = (rho_vec[t] + rho_vec_init[t]) / M_rho0_trace[t].real
                    num = (1j / H_BAR * M_PSI[t]) @ rho_vec[t] + M_rho0 @ rho_vec_t
                    solve_susceptibility(omega_grid, Xi, num, N, t, B_e, chi_T[3], chi_isothermal, chi_adiabatic, eye, False)

    return chi_T, relax_time_R21_T, relax_time_R41_T
