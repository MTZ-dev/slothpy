# susceptibility.py  –  streaming, multi-branch-aware (bug‑fix)
"""
Fix: initial‑state second‑order term now carries the **correct prefactor
½ħ / ω₍qj₎** (no Δ broadening in M_{ρ_A(0)}).

* `add_rho0_bundle` removed the stray `Δ/ω_q` factor and now uses
  `-HBAR / (2.0 * wq)` exactly as in eq. (119).
* Signature updated (no need for `d`).  Call sites adjusted.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple

import numpy as np
from numba import njit, prange

HBAR = 1
KB   = 3.166811563e-6

# -------------------------------------------------- scalar helpers ----------
@njit(cache=True)
def bose(omega: float, beta: float) -> float:
    x = beta * HBAR * omega
    if x > 700.0:
        return 0.0
    return 1.0 / (math.exp(x) - 1.0)

@njit(cache=True)
def zeta(x: float, beta: float) -> float:
    return beta if abs(x) < 1e-14 else (math.exp(HBAR * x * beta) - 1.0) / (HBAR * x)

@njit(cache=True)
def Iint(w1: float, w2: float, beta: float) -> float:
    eps = 1e-14
    if abs(w1) < eps and abs(w2) < eps:
        return 0.5 * beta * beta
    if abs(w2) < eps:
        return beta * math.exp(HBAR * w1 * beta) / (HBAR * w1) - (
            math.exp(HBAR * w1 * beta) - 1.0) / (HBAR**2 * w1**2)
    if abs(w1) < eps:
        return (math.exp(HBAR * w2 * beta) - 1.0) / (HBAR**2 * w2**2) - beta / (HBAR * w2)
    if abs(w1 + w2) < eps:
        return -1.0 / (HBAR * w1) * (beta - (math.exp(HBAR * w1 * beta) - 1.0) / (HBAR * w1))
    return (
        (math.exp(HBAR * (w1 + w2) * beta) - 1.0) / (HBAR**2 * w2 * (w1 + w2))
        - (math.exp(HBAR * w1 * beta) - 1.0) / (HBAR**2 * w1 * w2)
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
    return 0.5 * HBAR / wq * (n_q * L1 + (n_q + 1.0) * L2)

@njit(cache=True)
def Jcorr(omega, omega_p, w_ab, wq, n_q, d, beta):
    z1 = zeta(w_ab + wq, beta)
    z2 = zeta(w_ab - wq, beta)
    x1 = omega - omega_p - wq
    x2 = omega - omega_p + wq
    L1 = lor_pref(x1, d) + 1j * lor_hilb(x1, d)
    L2 = lor_pref(x2, d) + 1j * lor_hilb(x2, d)
    return 0.5 * HBAR / wq * (n_q * z1 * L1 + (n_q + 1.0) * z2 * L2)

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
        coeff = -HBAR / (2.0 * wq)
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
    w_n=E/HBAR
    A_e,B_e=U.conj().T@A@U, U.conj().T@B@U
    rho_eq=np.exp(-beta*E); rho_eq/=rho_eq.sum(); rho_vec=np.diag(rho_eq).flatten()
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
        num=(M_rho0+1j/HBAR*M_PSI)@rho_vec
        Xi=1j/HBAR*M_L+1.0/(HBAR**2)*M_KR-1j*omega*eye
        rho_hat=np.linalg.solve(Xi,num).reshape((N,N))
        chi[k]=1j/HBAR*np.trace(B_e@rho_hat)
    return chi
