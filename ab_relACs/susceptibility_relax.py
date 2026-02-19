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

from tqdm import tqdm

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
def get_Y_q_no_freq(Y_q, H_grad, normal_modes, k_point, dof_array, masses_inv_sqrt, number_of_kpoints_inv_sqrt):
    for j in range(normal_modes.shape[1]):
        for i in range(dof_array.shape[0]):
            dof = dof_array[i]
            Y_q[j] += H_grad[i] * normal_modes[dof[0], j] * masses_inv_sqrt[dof[0]] * 1/np.sqrt(M_AU) * number_of_kpoints_inv_sqrt * np.exp(-2j * np.pi * (k_point[0] * dof[1] + k_point[1] * dof[2] + k_point[2] * dof[3]))

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
def Jhat_m_symm(w_ab, wq, n_q, d, cutoff, kind):
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff:
        return hilbert(w_ab + wq, d, kind) * n_q
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff:
        return hilbert(w_ab - wq, d, kind) * (n_q + 1)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jcorr_symm(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind):
    u = w_cd + w_ab
    if u < 0 and np.abs(u + wq) < cutoff:
        return hilbert(u + wq, d, kind) * n_q * zeta(w_ab + wq, beta)
    if u > 0 and np.abs(u - wq) < cutoff:
        return hilbert(u - wq, d, kind) * (n_q + 1) * zeta(w_ab - wq, beta)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jhat_p(w_ab, wq, n_q, d, cutoff, kind):
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff and w_ab - wq < 0:
        return hilbert(w_ab - wq, d, kind) * n_q
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff and w_ab + wq > 0:
        return hilbert(w_ab + wq, d, kind) * (n_q + 1)
    return 0.0 + 0.0 * 1j 

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jhat_m(w_ab, wq, n_q, d, cutoff, kind):
    if w_ab < 0 and np.abs(w_ab + wq) < cutoff and w_ab + wq > 0:
        return hilbert(w_ab + wq, d, kind) * n_q
    if w_ab > 0 and np.abs(w_ab - wq) < cutoff and w_ab - wq < 0:
        return hilbert(w_ab - wq, d, kind) * (n_q + 1)
    return 0.0 + 0.0 * 1j

@njit(nogil=True, cache=True, fastmath=True, inline="always")
def Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind):
    u = w_cd + w_ab
    if u < 0 and np.abs(u + wq) < cutoff and u + wq > 0:
        return hilbert(u + wq, d, kind) * n_q * zeta(w_ab + wq, beta)
    if u > 0 and np.abs(u - wq) < cutoff and u - wq < 0:
        return hilbert(u - wq, d, kind) * (n_q + 1) * zeta(w_ab - wq, beta)
    return 0.0 + 0.0 * 1j

def Jhat_p_comp(w_ab, wq, n_q, d, cutoff, kind, symm):
    raise NotImplementedError

@overload(Jhat_p_comp, nogil=True, fastmath=True, cache=True, inline="always", prefer_literal=True)
def ov_Jhat_p_comp(w_ab, wq, n_q, d, cutoff, kind, symm):
    if isinstance(symm, types.Literal):
        if symm.literal_value == 0:
            def impl(w_ab, wq, n_q, d, cutoff, kind, symm):
                return Jhat_p(w_ab, wq, n_q, d, cutoff, kind)
            return impl
        elif symm.literal_value == 1:
            def impl(w_ab, wq, n_q, d, cutoff, kind, symm):
                return Jhat_p_symm(w_ab, wq, n_q, d, cutoff, kind)
            return impl
        
def Jcorr_comp(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind, symm):
    raise NotImplementedError

@overload(Jcorr_comp, nogil=True, fastmath=True, cache=True, inline="always", prefer_literal=True)
def ov_Jcorr_comp(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind, symm):
    if isinstance(symm, types.Literal):
        if symm.literal_value == 0:
            def impl(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind, symm):
                return Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind)
            return impl
        elif symm.literal_value == 1:
            def impl(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind, symm):
                return Jcorr(w_cd, w_ab, wq, n_q, d, beta, cutoff, kind)
            return impl
        
def Jhat_m_comp(w_ab, wq, n_q, d, cutoff, kind, symm):
    raise NotImplementedError

@overload(Jhat_m_comp, nogil=True, fastmath=True, cache=True, inline="always", prefer_literal=True)
def ov_Jhat_m_comp(w_ab, wq, n_q, d, cutoff, kind, symm):
    if isinstance(symm, types.Literal):
        if symm.literal_value == 0:
            def impl(w_ab, wq, n_q, d, cutoff, kind, symm):
                return Jhat_m(w_ab, wq, n_q, d, cutoff, kind)
            return impl
        elif symm.literal_value == 1:
            def impl(w_ab, wq, n_q, d, cutoff, kind, symm):
                return Jhat_m_symm(w_ab, wq, n_q, d, cutoff, kind)
            return impl

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
def build_Jp_Jm_tables_j(w_n, wb, nb, delta, cutoff, kind, symm):
    N = w_n.shape[0]
    J = wb.shape[0]
    Jp = np.empty((J, N, N), np.complex128)
    Jm = np.empty((J, N, N), np.complex128)

    for j in range(J):
        wq, n_q, delta_j, cutoff_j = wb[j], nb[j], delta[j], cutoff[j]
        for a in range(N):
            for b in range(N):
                x = w_n[a, b]
                Jp[j, a, b] = Jhat_p_comp(x,wq,n_q,delta_j,cutoff_j,kind,symm)
                Jm[j, a, b] = Jhat_m_comp(x,wq,n_q,delta_j,cutoff_j,kind,symm)
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
    symm: types.Literal,
    ):
    raise NotImplementedError

@overload(build_matrices, nogil=True, fastmath=True, cache=True, inline="never", prefer_literal=True)
def ov_build_matrices(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                      R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k, symm):
    if isinstance(run_R41, types.Literal):
        if run_R41.literal_value == 0:
            def impl(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                     R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k, symm):
                return build_matrices_no_R41(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                                             R21, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, n_k, symm)
            return impl
        elif run_R41.literal_value == 1:
            def impl(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                     R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, run_R41, n_k, symm):
                return build_matrices_R41(hessian, masses_inv_sqrt, dof_array, H_grad, grid, weights, gamma_fwhm, beta, w_n, M_KR, M_PSI, A_e, rho_vec_init, M_rho0_trace, rho_mat,
                                          R21, R41, cutoff_mult, degeneracy_tolerance, modes_low, modes_high, kind, direct, run_KR, run_PSI, run_rho0, run_R21, n_k, symm)
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
    symm: types.Literal
    ):
    n_k_inv = 1.0 / np.sqrt(n_k)
    masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)
    gamma = np.asarray([0.0, 0.0, 0.0])
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

            Jhat_p_table, Jhat_m_table = build_Jp_Jm_tables_j(w_n, wb, bose, fwhm_j, cutoff_j, kind, symm)
            if run_R21:
                if symm:
                    Jhat_p_symm_table = Jhat_p_table
                else:
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
    symm: types.Literal,
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

            Jhat_p_table, Jhat_m_table = build_Jp_Jm_tables_j(w_n, wb, bose, fwhm_j, cutoff_j, kind, symm)
            if run_R21:
                if symm:
                    Jhat_p_symm_table = Jhat_p_table
                else:
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
                chi_T[t,k] /= np.abs(chi_T[t,0].real)
                chi_T[t,k] *= (chi_isothermal[t] - chi_adiabatic[t])
                chi_T[t,k] += chi_adiabatic[t]
    if normalize:
        chi_T[t,0] = chi_T[t,0] / np.abs(chi_T[t,0].real) * chi_isothermal[t]

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
    symm: int,
):  
    kind_lit = literally(kind)
    direct_lit = literally(direct)
    run_KR_lit = literally(run_KR)
    run_PSI_lit = literally(run_PSI)
    run_rho0_lit = literally(run_rho0)
    run_R21_lit = literally(run_R21)
    run_R41_lit = literally(run_R41)
    symm_lit = literally(symm)

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
    print("Energies", E)

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
                    run_R21_lit, run_R41_lit, n_k, symm_lit)

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


def _gauss_kernel(x, x0, sigma):
    dx = (x - x0) / sigma
    return np.exp(-0.5 * dx * dx) / (sigma * np.sqrt(2.0 * np.pi))

def _lorentz_kernel(x, x0, gamma):
    return (gamma / np.pi) / ((x - x0) ** 2 + gamma ** 2)


def plot_spinphonon_weighted_phonon_dos(
    # --- phonons + spin-phonon coupling inputs ---
    E: np.ndarray,
    H_grad: np.ndarray,
    hessian: np.ndarray,
    masses_inv_sqrt: np.ndarray,
    dof_array: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    n_k: int,
    states_number: int,
    threads: int,
    # --- NEW: mode window (cm^-1) ---
    modes_low: float,
    modes_high: float,
    # --- DOS controls ---
    resolution: int = 2000,
    convolution: str | None = "gaussian",   # None / "gaussian" / "lorentzian"
    fwhm: float = 10.0,                     # cm^-1 (for convolution curve)
    density: bool = False,
    # --- weighting controls ---
    weight_mode: str = "fro_offdiag",       # "fro", "fro_offdiag", "thermal_sym", "thermal_offdiag"
    temperature: float | None = None,       # required for thermal_* modes
    # --- frequency handling ---
    dos_freq: str = "raw",                  # "raw" or "abs"
    eps_freq_cm1: float = 1e-12,            # prevents 1/sqrt(0) in get_Y_q
    # --- plot controls ---
    save_path: str | None = None,
    show: bool = True,
    energy_lines=None,
    title: str = "Spin--phonon weighted phonon DOS",
):
    """
    Compute and plot a phonon DOS weighted by spin–phonon coupling strength,
    restricted to modes_low <= omega <= modes_high (in cm^-1).
    """

    # ---------- setup ----------
    if modes_high <= modes_low:
        raise ValueError("Require modes_high > modes_low (both in cm^-1).")

    N = int(states_number)
    E = np.asarray(E[:N], dtype=np.float64)
    H_grad = np.asarray(H_grad, dtype=np.complex128)[:, :N, :N]
    dof_array = np.asarray(dof_array, dtype=np.int64)
    masses_inv_sqrt = np.asarray(masses_inv_sqrt, dtype=np.float64)

    masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)
    n_k_inv = 1.0 / np.sqrt(float(n_k))

    if temperature is None:
        raise ValueError("temperature must be provided for thermal_* weight_mode.")
    beta = 1.0 / (KB * float(temperature))
    p = np.exp(-beta * (E - E.min()))
    p /= p.sum()

    reduce_W, p, _ = _make_weight_reducer(weight_mode, E, temperature)

    all_freq, all_wts = _gather_spinphonon_weighted_dos_fast(
        E=E[:N],
        H_grad=H_grad,
        hessian=hessian,
        masses_inv_sqrt=masses_inv_sqrt,
        dof_array=dof_array,
        grid=grid,
        weights=weights,
        n_k=n_k,
        modes_low=modes_low,
        modes_high=modes_high,
        weight_mode=weight_mode,
        threads=threads,
        temperature=temperature,
        eps_bose_cm1=eps_freq_cm1,   # re-use your existing parameter
    )

    if all_freq.size == 0:
        raise RuntimeError("No modes found in the requested [modes_low, modes_high] window.")

    # ---------- histogram (restricted range) ----------
    fmin = float(modes_low)
    fmax = float(modes_high)
    pad = (fmax - fmin) / max(resolution, 1)
    fmin_plot = fmin - pad
    fmax_plot = fmax + pad

    hist, bin_edges = np.histogram(
        all_freq,
        bins=int(resolution),
        range=(fmin_plot, fmax_plot),
        density=bool(density),
        weights=all_wts
    )

    hist_norm = hist / hist.max() if hist.max() > 0 else hist

    # ---------- convolution curve ----------
    if convolution is None:
        result = (bin_edges, hist_norm)
        freq_range = None
        conv_norm = None
    else:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        freq_range = np.linspace(fmin_plot, fmax_plot, int(resolution), dtype=np.float64)

        if convolution.lower() == "gaussian":
            sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            conv = np.zeros_like(freq_range)
            for x0, a0 in zip(centers, hist_norm):
                conv += a0 * _gauss_kernel(freq_range, x0, sigma)

        elif convolution.lower() == "lorentzian":
            gamma = float(fwhm) / 2.0
            conv = np.zeros_like(freq_range)
            for x0, a0 in zip(centers, hist_norm):
                conv += a0 * _lorentz_kernel(freq_range, x0, gamma)

        else:
            raise ValueError("convolution must be None, 'gaussian', or 'lorentzian'.")

        conv_norm = conv / conv.max() if conv.max() > 0 else conv
        result = (bin_edges, hist_norm, freq_range, conv_norm)

    # ---------- plot ----------
    if save_path is not None or show:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from numpy import diff

        mpl.rcParams.update({
            "font.family"      : "serif",
            "font.size"        : 12,
            "mathtext.fontset" : "cm",
            "axes.labelsize"   : 12,
            "axes.titlesize"   : 12,
            "xtick.direction"  : "in",
            "ytick.direction"  : "in",
            "xtick.top"        : True,
            "ytick.right"      : True,
            "axes.spines.right": True,
            "axes.spines.top"  : True,
            "figure.dpi"       : 300,
        })

        fig, ax = plt.subplots(figsize=(6.299, 4.1993), constrained_layout=True)

        ax.bar(
            bin_edges[:-1],
            hist_norm,
            width=diff(bin_edges),
            color="#F1B960",
            edgecolor="#F1B960",
            alpha=0.75,
            rasterized=True,
        )

        if convolution is not None:
            ax.plot(freq_range, conv_norm, color="#69A6D7", linewidth=0.9)

        ax.set_xlim(fmin-(0.03*fmax), fmax+(0.03*fmax))
        ax.set_xlabel(r"Frequency / cm$^{-1}$")
        ax.set_ylabel(r"Weighted DOS / a.u.")
        ax.set_title(f"{title}")
        ax.grid(True, linestyle="--", alpha=0.5)

        if energy_lines is not None:
            if isinstance(energy_lines, tuple):
                for x0 in energy_lines[0]:
                    ax.axvline(x=float(x0), color="olivedrab", lw=1.4, alpha=1.0, zorder=10000)
                for x0 in energy_lines[1]:
                    ax.axvline(x=float(x0), color="sienna", lw=1.1, alpha=1.0, zorder=10000)
            else:
                for x0 in energy_lines:
                    ax.axvline(x=float(x0), color="sienna", lw=1.1, alpha=1.0, zorder=10000)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return result

def _make_weight_reducer(weight_mode: str, E: np.ndarray, temperature: float | None):
    """
    Returns:
      reduce_W(absY2) -> 1D array W_j
      (p, P_sym)      -> maybe needed elsewhere (optional)
    absY2 shape: (Jm, N, N)
    """
    wm = str(weight_mode).strip().lower()
    N = E.size

    if wm in ("fro", "fro_offdiag"):
        if wm == "fro":
            def reduce_W(absY2):
                return absY2.sum(axis=(1, 2))
        else:
            def reduce_W(absY2):
                diag = np.diagonal(absY2, axis1=1, axis2=2)  # (Jm, N)
                return absY2.sum(axis=(1, 2)) - diag.sum(axis=1)
        return reduce_W, None, None

    # thermal modes
    if temperature is None:
        raise ValueError("temperature must be provided for thermal_* weight_mode.")
    beta = 1.0 / (KB * float(temperature))
    p = np.exp(-beta * (E - E.min()))
    p /= p.sum()
    P_usym = (p[:, None] * np.ones_like(p)[None, :])

    if wm == "thermal_sym":
        def reduce_W(absY2):
            return (absY2 * P_usym[None, :, :]).sum(axis=(1, 2))
        return reduce_W, p, P_usym

    if wm == "thermal_offdiag":
        def reduce_W(absY2):
            diag = np.diagonal(absY2, axis1=1, axis2=2)  # (Jm, N)
            return (absY2 * P_usym[None, :, :]).sum(axis=(1, 2)) - (diag * p).sum(axis=1)
        return reduce_W, p, P_usym

    raise ValueError("weight_mode must be one of: fro, fro_offdiag, thermal_sym, thermal_offdiag")


# -------------------------------------------------------------------------
# FAST PARALLEL GATHER FOR WEIGHTED PHONON DOS (drop-in for tqdm loop)
# -------------------------------------------------------------------------

def _weight_mode_to_code(weight_mode: str) -> int:
    wm = str(weight_mode).strip().lower()
    if wm == "fro":
        return 0
    if wm == "fro_offdiag":
        return 1
    if wm == "thermal_sym":
        return 2
    if wm == "thermal_offdiag":
        return 3
    raise ValueError("weight_mode must be one of: fro, fro_offdiag, thermal_sym, thermal_offdiag")


@njit(nogil=True, cache=True, fastmath=True, inline="always")
def _abs2(z):
    return z.real * z.real + z.imag * z.imag


@njit(nogil=True, cache=True, fastmath=True, inline="always")
def _phase_from_phi(phi):
    # exp(-1j*phi) = cos(phi) - i sin(phi)
    return np.cos(phi) - 1j * np.sin(phi)


@njit(nogil=True, cache=True, fastmath=True, parallel=True, inline="never")
def _gather_spinphonon_weighted_dos_parallel(
    H_grad: np.ndarray,              # (n_dof, N, N) complex128
    hessian: np.ndarray,             # (nx,ny,nz,ndof,ndof) complex128 (your format)
    masses_inv_sqrt: np.ndarray,     # (ndof,) float64
    masses_inv_sqrt_outer: np.ndarray,# (ndof,ndof) float64
    dof_array: np.ndarray,           # (n_dof, 4) int64 : [dof0, tx, ty, tz]
    grid: np.ndarray,                # (nq, 3) float64
    weights_bz: np.ndarray,          # (nq,) float64
    n_k_inv: float,                  # 1/sqrt(n_k)
    modes_low: float,                # cm^-1
    modes_high: float,               # cm^-1
    weight_mode_code: int,           # 0..3
    p: np.ndarray,                   # (N,) float64 (only used for thermal_*; otherwise ignored)
    beta: float,                     # 1/(kB*T) (only used for thermal_* phonon factor)
    eps_bose_cm1: float,             # avoid Bose blowup at 0
    threads: int,
):
    nq = grid.shape[0]
    ndof = masses_inv_sqrt.shape[0]     # number of phonon modes == dynamical matrix dimension
    N = H_grad.shape[1]                # spin states number used in DOS

    # Big fixed buffers: each q-point writes to its own slice.
    out_freq = np.empty(nq * ndof, dtype=np.float64)
    out_wts  = np.full(nq * ndof, np.nan, dtype=np.float64)  # NaN = unused slot

    inv_sqrt_M_AU = 1.0 / np.sqrt(M_AU)

    # Precompute scale per DOF index used in eigenvectors:
    # modes[dof0, j] * (masses_inv_sqrt[dof0] * inv_sqrt_M_AU * n_k_inv) * phase[ii]
    scale_d = masses_inv_sqrt * (inv_sqrt_M_AU * n_k_inv)

    set_num_threads(threads)

    for iq in prange(nq):
        q0 = grid[iq, 0]
        q1 = grid[iq, 1]
        q2 = grid[iq, 2]
        wq_weight = weights_bz[iq]

        # Build dynamical matrix and diagonalize
        Dq = _build_dynamical_matrix(hessian, masses_inv_sqrt_outer, grid[iq])
        freq_au, modes = frequencies_eigenvectors(Dq)   # freq in AU-ish, modes complex
        # Convert frequencies to cm^-1
        base = iq * ndof

        # Precompute phase factors for each gradient entry ONCE per q:
        # phase[ii] = exp(-2j*pi*(q · T_i))
        phase = np.empty(dof_array.shape[0], dtype=np.complex128)
        for ii in range(dof_array.shape[0]):
            tx = dof_array[ii, 1]
            ty = dof_array[ii, 2]
            tz = dof_array[ii, 3]
            phi = 2.0 * np.pi * (q0 * tx + q1 * ty + q2 * tz)
            phase[ii] = _phase_from_phi(phi)

        # Scratch Y matrix reused for each mode
        Y = np.empty((N, N), dtype=np.complex128)

        k = 0  # index within this q-slice
        for j in range(ndof):
            f_cm1 = freq_au[j] * AU_BOHR_CM_1
            if f_cm1 < modes_low or f_cm1 > modes_high:
                continue

            # zero Y
            for a in range(N):
                for b in range(N):
                    Y[a, b] = 0.0 + 0.0j

            # Build Y for this phonon mode j
            # Y += sum_i H_grad[i] * (modes[dof0,j] * scale[dof0] * phase[i])
            for ii in range(dof_array.shape[0]):
                dof0 = dof_array[ii, 0]
                c = modes[dof0, j] * scale_d[dof0] * phase[ii]
                # accumulate into Y matrix
                for a in range(N):
                    for b in range(N):
                        Y[a, b] += H_grad[ii, a, b] * c

            W = 0.0
            if weight_mode_code == 0:
                # fro: sum_{a,b} |Y_ab|^2
                for a in range(N):
                    for b in range(N):
                        W += _abs2(Y[a, b])

            elif weight_mode_code == 1:
                # fro_offdiag: sum_{a!=b} |Y_ab|^2
                for a in range(N):
                    for b in range(N):
                        if a != b:
                            W += _abs2(Y[a, b])

            elif weight_mode_code == 2:
                # thermal_sym (weights by p[a] for all b)
                for a in range(N):
                    pa = p[a]
                    for b in range(N):
                        W += pa * _abs2(Y[a, b])

            else:
                # thermal_offdiag: (thermal_sym) minus diagonal term
                for a in range(N):
                    pa = p[a]
                    for b in range(N):
                        if a != b:
                            W += pa * _abs2(Y[a, b])

            ################################################################# Excluding Dy fast Direct ################################
            # else:
            #     # thermal_offdiag: (thermal_sym) minus diagonal term
            #     for a in range(N):
            #         pa = p[a]
            #         for b in range(N):
            #             if a != b:
            #                 if f_cm1 < 69:
            #                     if a < 4 and b < 4:
            #                         W += pa * _abs2(Y[a, b])
            #                 else:
            #                     W += pa * _abs2(Y[a, b])

            # Phonon thermal factor (only for thermal_* modes)
            ph = 1.0
            if weight_mode_code >= 2:
                ff = f_cm1
                if ff < 0.0:
                    ff = -ff
                if ff < eps_bose_cm1:
                    ff = eps_bose_cm1
                ph = bose_occ(ff, beta)

            out_freq[base + k] = f_cm1
            out_wts[base + k]  = wq_weight * ph * W
            k += 1

        # remaining positions in this slice stay NaN (unused)

    return out_freq, out_wts


def _gather_spinphonon_weighted_dos_fast(
    E: np.ndarray,
    H_grad: np.ndarray,
    hessian: np.ndarray,
    masses_inv_sqrt: np.ndarray,
    dof_array: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    n_k: int,
    modes_low: float,
    modes_high: float,
    weight_mode: str,
    threads: int,
    temperature: float | None,
    eps_bose_cm1: float = 1e-12,
):
    """
    Python wrapper: prepares p/beta and compresses NaN slots after the Numba gather.
    Returns: all_freq, all_wts (1D float64)
    """
    wm_code = _weight_mode_to_code(weight_mode)

    # Prepare p and beta only if thermal_* modes are requested
    if wm_code >= 2:
        if temperature is None:
            raise ValueError("temperature must be provided for thermal_* weight_mode.")
        beta = 1.0 / (KB * float(temperature))
        E = np.asarray(E, dtype=np.float64)
        p = np.exp(-beta * (E - E.min()))
        p /= p.sum()
    else:
        beta = 0.0
        # dummy p (won't be used)
        E = np.asarray(E, dtype=np.float64)
        p = np.ones(E.shape[0], dtype=np.float64)

    grid = np.asarray(grid, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    masses_inv_sqrt = np.asarray(masses_inv_sqrt, dtype=np.float64)
    dof_array = np.asarray(dof_array, dtype=np.int64)
    H_grad = np.asarray(H_grad, dtype=np.complex128)

    masses_inv_sqrt_outer = np.outer(masses_inv_sqrt, masses_inv_sqrt)
    n_k_inv = 1.0 / np.sqrt(float(n_k))

    freq_big, wts_big = _gather_spinphonon_weighted_dos_parallel(
        H_grad, hessian, masses_inv_sqrt, masses_inv_sqrt_outer,
        dof_array, grid, weights, n_k_inv,
        float(modes_low), float(modes_high),
        wm_code, p, float(beta), float(eps_bose_cm1), threads,
    )

    mask = ~np.isnan(wts_big)
    return freq_big[mask], wts_big[mask]




