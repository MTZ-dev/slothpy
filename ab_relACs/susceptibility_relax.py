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

from __future__ import annotations

import numpy as np
from numpy import pi

from typing import Sequence, Union
ArrayLike = Union[Sequence, np.ndarray]

from numba import njit, prange, get_thread_id, set_num_threads, get_num_threads
from numba import types
from numba.extending import intrinsic, get_cython_function_address
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.errors import TypingError
from llvmlite import ir as llir
from llvmlite import binding as llvm
from llvmlite.ir import IRBuilder

from slothpy._general_utilities._constants import H_CM_1, AU_BOHR_CM_1, MU_B_CM_3
from constants import KB, H, H_BAR, S_TIME_PS, M_AU
from input_models import AppConfig

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

@njit(nogil=True, cache=True, fastmath=True)
def dummy_function(*args):
    return

@njit(nogil=True, cache=True, fastmath=True)
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

@njit(nogil=True, cache=True, fastmath=True)
def Jhat_p_sec(w_ab, wq, n_q, d, cutoff):
    if w_ab > 0:
        return gaussian(w_ab - wq, d, cutoff) * n_q
    if w_ab < 0:
        return gaussian(w_ab + wq, d, cutoff) * (n_q + 1)
    return 0.0 + 0.0 * 1j 

@njit(nogil=True, cache=True, fastmath=True)
def lorentz_hilbert(E, d):
    return 1j / (E - 1j * d)

@njit(nogil=True, cache=True, fastmath=True)
def gauss_hilbert(E, d):
    factor_sqrt = E / (np.sqrt(2)*d)
    return np.sqrt(np.pi*0.5) / d * (np.exp(-(factor_sqrt*factor_sqrt)) + 2j / np.sqrt(np.pi) * dawsn(factor_sqrt))

@njit(nogil=True, cache=True, fastmath=True)
def build_Y_table_j(Yb):
    J, N, _ = Yb.shape
    out = np.empty((J, N, N, N, N), np.complex128)

    for j in prange(J):
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

@njit(nogil=True, cache=True, fastmath=True)
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

@njit(nogil=True, cache=True, fastmath=True)
def add_rho0_bundle(out, trace, A, Yb_table, wb, nb, beta, w_n, q_0, rho, thread_id, t_index):
    N=w_n.shape[0]; J=wb.size
    coeff = 0.5
    if q_0:
        coeff *= 0.5
    for j in prange(J):
        Y_j, wq, n_q = Yb_table[j], wb[j], nb[j]
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
                            trace[thread_id,t_index]+=(n_q*Iint(w_de+wq,w_eb-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_eb+wq,beta))*rho[a,d]*Y_j[d,e,e,b]
                        corr+=(n_q*Iint(w_de+wq,w_eb-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_eb+wq,beta))*A[a,c]*Y_j[d,e,e,b]*rho[c,d]
                        for f in range(N):
                            w_ef=w_n[e,f]
                            corr-=(n_q*Iint(w_de+wq,w_ef-wq,beta)+(n_q+1.0)*Iint(w_de-wq,w_ef+wq,beta))*(1.0 if a==c else 0.0)*Y_j[d,e,e,f]*A[f,b]*rho[c,d]
                    out[ab]+=coeff*corr

@njit(nogil=True, cache=True, fastmath=True)
def zeta(x: float, beta: float) -> float:
    eps = 1e-15
    u = beta * x
    if abs(u) < eps:
        return beta
    if abs(u) > 650:
        return 0.0 + 1j * 0.0
    return np.expm1(u) / (x)

@njit(nogil=True, cache=True, fastmath=True)
def Iint(w1: float, w2: float, beta: float) -> float:
    eps = 1e-15
    u1 = w1 * beta
    u2 = w2 * beta
    u12 = u1 + u2
    if abs(u1) > 650 or abs(u12) > 650:
        return 0.0 + 1j * 0.0
    elif abs(u1) < eps and abs(u2) < eps:
        return 0.5 * beta * beta
    elif abs(u2) < eps:
        num = np.exp(u1)
        return beta * num / (w1) - np.expm1(u1) / (w1**2)
    elif abs(u1) < eps:
        return np.expm1(u2) / (w2**2) - beta / (w2)
    elif abs(u12) < eps:
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

@njit(nogil=True, cache=True, fastmath=True)
def get_Y_q(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt, freq):
    for j in range(normal_modes.shape[1]):
        for i in range(dof_array.shape[0]):
            dof = dof_array[i]
            Y_q[j] += (1 / np.sqrt(freq[j]/H_CM_1)) * H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * 1/np.sqrt(M_AU) * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))

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

def make_susceptibility_relax_time(cfg: AppConfig):
    cutoff_mult = cfg.relacs.cutoff_fwhm
    degeneracy_tolerance = cfg.relacs.degeneracy_tolerance
    states_number = cfg.relacs.states_number
    modes_low = cfg.relacs.modes_low
    modes_high= cfg.relacs.modes_high

    hilbert = gauss_hilbert if cfg.relacs.broadening == "gaussian" else lorentz_hilbert

    @njit(nogil=True, cache=True, fastmath=True)
    def Jhat_p(w_ab, wq, n_q, d, cutoff):
        if w_ab > 0 and w_ab - wq > -cutoff and w_ab - wq < 0:
            return hilbert(w_ab - wq, d) * n_q
        if w_ab < 0 and w_ab + wq < cutoff and w_ab + wq > 0:
            return hilbert(w_ab + wq, d) * (n_q + 1)
        return 0.0 + 0.0 * 1j 

    @njit(nogil=True, cache=True, fastmath=True)
    def Jhat_m(w_ab, wq, n_q, d, cutoff):
        if w_ab < 0 and w_ab + wq < cutoff and w_ab + wq > 0:
            return hilbert(w_ab + wq, d) * n_q
        if w_ab > 0 and w_ab - wq > -cutoff and w_ab - wq < 0:
            return hilbert(w_ab - wq, d) * (n_q + 1)
        return 0.0 + 0.0 * 1j

    @njit(nogil=True, cache=True, fastmath=True)
    def Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff):
        u = w_cd + w_ab
        if u < 0 and u + wq < cutoff and u + wq > 0:
            return hilbert(u + wq, d) * n_q * zeta(w_ab + wq, beta)
        if u > 0 and u - wq > -cutoff and u - wq < 0:
            return hilbert(u - wq, d) * (n_q + 1) * zeta(w_ab - wq, beta)
        return 0.0 + 0.0 * 1j

    @njit(nogil=True, cache=True, fastmath=True)
    def build_Jp_Jm_tables_j(w_n, wb, nb, delta, cutoff):
        N = w_n.shape[0]
        J = wb.shape[0]
        Jp = np.empty((J, N, N), np.complex128)
        Jm = np.empty((J, N, N), np.complex128)

        for j in range(J):
            wq, n_q, delta_j, cutoff_j = wb[j], nb[j], delta[j], cutoff[j]
            for a in range(N):
                for b in range(N):
                    x = w_n[a, b]
                    Jp[j, a, b] = Jhat_p(x,wq,n_q,delta_j,cutoff_j)
                    Jm[j, a, b] = Jhat_m(x,wq,n_q,delta_j,cutoff_j)

        return Jp, Jm
    
    @njit(nogil=True, cache=True, fastmath=True)
    def add_PSI_bundle(out, A, Yb_table, wb, nb, delta, beta, w_n, q_0, cutoff, weight):
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
                            val+=Jcorr(w_n[e,c],w_n[d,b],wq,n_q,delta_j,beta,cutoff_j)*A[e,c]*Y_j[d,b,a,e]
                            val-=Jcorr(w_n[a,d],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j)*A[a,c]*Y_j[d,e,e,b]
                            if a == c:
                                for f in range(N):
                                    val+=Jcorr(w_n[e,f],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j)*A[e,f]*Y_j[f,b,d,e]
                            val-=Jcorr(w_n[e,b],w_n[d,e],wq,n_q,delta_j,beta,cutoff_j)*Y_j[a,c,d,e]*A[e,b]
                        out[ab,cd]+=val*coeff

    @njit(nogil=True, fastmath=True, cache=True, inline="always")
    def get_cutoff_j_qtm(fwhm, wb, w_n_qtm_max):
        return fwhm * cutoff_mult
    
    @njit(nogil=True, fastmath=True, cache=True, inline="always")
    def get_cutoff_j_no_qtm(fwhm, wb, w_n_qtm_max):
        return np.minimum(fwhm * 1000, np.abs(wb + 1.01 * w_n_qtm_max))

    get_cutoff_j = get_cutoff_j_qtm if cfg.relacs.qtm else get_cutoff_j_no_qtm

    add_KR_bundle_comp = add_KR_bundle
    add_PSI_bundle_comp = add_PSI_bundle if cfg.relacs.psi_frequency_shift else dummy_function
    add_rho0_bundle_comp = add_rho0_bundle if cfg.relacs.initial_correlation else dummy_function
    add_R21_bundle_comp = add_R21_bundle if cfg.relacs.tau_21_csv_path else dummy_function
    
    @njit(nogil=True, fastmath=True, cache=True, parallel=True)
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
        ):
        n_k_inv = 1.0 / np.sqrt(grid.shape[0])
        masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)
        gamma = np.asarray([0.0, 0.0, 0.0])
        freq0, modes0 = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, gamma))
        freq_shape = freq0.shape[0]
        scale_freq = np.min(freq0)
        arr = np.diag(w_n, k=1)
        w_n_qtm_max = arr[0]
        for i in range(2, arr.size, 2):  # step of 2
            if arr[i] < w_n_qtm_max:
                w_n_qtm_max = arr[i]

        # Raman ----------------------------------------------------------------------------------
        # threads_number = get_num_threads()
        # max_grid_per_thread = np.int64(np.ceil(grid.shape[0]/threads_number))
        # Yb_array = np.zeros((threads_number,2*max_grid_per_thread*freq_shape, H_grad.shape[1], H_grad.shape[2]), np.complex128)
        # wb_array = np.zeros((threads_number,2*max_grid_per_thread*freq_shape), np.float64)
        # raman_counter = np.zeros(threads_number, dtype=np.int64)
        # raman_counter_wb = np.zeros(threads_number, dtype=np.int64)
        # raman_counter_2wb = np.zeros(threads_number, dtype=np.int64)
        # ----------------------------------------------------------------------------------------

        for i in prange(grid.shape[0]):
            thread_id = get_thread_id()
            q = grid[i]
            weight = weights[i]
            q_0 = np.allclose(q, gamma, atol=1e-6)
            freq, modes = frequencies_eigenvectors(_build_dynamical_matrix(hessian, masses_inv_sqrt_outer, q))
            freq = freq - scale_freq
            freq *= AU_BOHR_CM_1

            if q_0:
                freq, modes = freq[3:], modes[:,3:]
            mask = (freq >= modes_low) & (freq <= modes_high)
            idx  = np.where(mask)[0]
            wb, modes = np.ascontiguousarray(freq[idx]), np.ascontiguousarray(modes[:,idx])
            Yb = np.zeros((wb.size, H_grad.shape[1], H_grad.shape[2]), dtype=np.complex128)
            get_Y_q(Yb, H_grad, modes, q, dof_array, masses_inv_sqrt, n_k_inv, wb)
            Yb_table = build_Y_table_j(Yb)

            fwhm_j = gamma_fwhm * np.ones_like(wb) # TODO: can implement ab inito model for different wb and T (move into the loop) - adaptive_fwhm in config
            cutoff_j = get_cutoff_j(fwhm_j, wb, w_n_qtm_max)

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
                beta_t = beta[t_index]
                rho_mat_t = rho_mat[t_index]

                M_KR_i_t = M_KR[thread_id, t_index]
                M_PSI_i_t = M_PSI[thread_id, t_index]
                rho_vec_init_i_t = rho_vec_init[thread_id, t_index]
                R21_i_t = R21[thread_id, t_index]

                bose = bose_occ(wb, beta[t_index])

                Jhat_p_table, Jhat_m_table = build_Jp_Jm_tables_j(w_n, wb, bose, fwhm_j, cutoff_j)

                add_KR_bundle_comp(M_KR_i_t, Yb_table, Jhat_p_table, Jhat_m_table, q_0, weight)
                add_PSI_bundle_comp(M_PSI_i_t, A_e, Yb_table, wb, bose, fwhm_j, beta_t, w_n, q_0, cutoff_j, weight)
                add_rho0_bundle_comp(rho_vec_init_i_t, M_rho0_trace, A_e, Yb_table, wb, bose, beta_t, w_n, q_0, rho_mat_t, thread_id, t_index)
                add_R21_bundle_comp(R21_i_t, Yb_table, w_n, Jhat_p_table, q_0, degeneracy_tolerance, weight)

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

    @njit(nogil=True, fastmath=True)
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
    ):
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
        
        build_matrices(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat, R21, R41)

        M_KR = np.sum(M_KR, axis=0)
        M_PSI = np.sum(M_PSI, axis=0)
        rho_vec_init = np.sum(rho_vec_init, axis=0)
        M_rho0_trace = 1.0 + np.sum(M_rho0_trace, axis=0)
        R21 = np.sum(R21, axis=0)
        R41 = np.sum(R41, axis=0)

        eye = np.eye(N2, dtype=np.complex128)

        chi_T = np.empty((temp_size, omega_grid.shape[0]), dtype=np.complex128)
        relax_time_R21_T = np.empty(temp_size, dtype=np.float64)
        relax_time_R41_T = np.empty(temp_size, dtype=np.float64)

        for t in range(temp_size):
            relax_time_R21_T[t] = get_relax_time(R21[t])
            relax_time_R41_T[t] = get_relax_time(R41[t])
            for k, omega in enumerate(omega_grid):
                Xi       = 1j / H_BAR * M_L + M_KR[t] / (H_BAR ** 2) - 1j * omega * eye
                num      = (1j / H_BAR * M_PSI[t]) @ rho_vec[t] + (M_rho0 @ rho_vec[t] + rho_vec_init[t]) / M_rho0_trace[t].real
                rho_hat  = np.linalg.solve(Xi, num).reshape((N, N))
                chi_T[t,k]   = 1j / H_BAR * np.trace(B_e @ rho_hat) / H_CM_1 * MU_B_CM_3
                
                if k != 0:
                    chi_T[t,k] /= chi_T[t,0].real
                    chi_T[t,k] *= (chi_isothermal[t] - chi_adiabatic[t])
                    chi_T[t,k] += chi_adiabatic[t] 
            chi_T[t,0] = chi_isothermal[t]

        return chi_T, relax_time_R21_T, relax_time_R41_T

    return susceptibility_relax_time