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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------- small utility ---------------------------------------------------
def _as_indices(values, selection, *, name):
    """
    Turn 'all' | int | sequence[int] into a sorted unique list of indices.
    """
    if selection == "all":
        return list(range(len(values)))
    if isinstance(selection, (int, np.integer)):
        return [int(selection)]
    idx = sorted(set(int(i) for i in selection))
    # cheap sanity
    for i in idx:
        if i < 0 or i >= len(values):
            raise IndexError(f"{name} index {i} out of range [0,{len(values)-1}]")
    return idx

def _label_for_combo(n_points_array, fwhm_array, fields, ni, fi, bi):
    return (f"n={n_points_array[ni]}, "
            f"FWHM={fwhm_array[fi]:g} cm$^{{-1}}$, "
            f"B={fields[bi]:g} T")

# ---------- χ(ω) – frequency plots -----------------------------------------
def plot_chi_vs_freq(
    omega_Hz, temperatures, fields, n_points_array, fwhm_array,
    sus_H_T,                                    # (npts, fwhm, fields, temps, freqs) complex
    *,
    n_points_sel="all", fwhm_sel="all", field_sel="all", temps_sel="all",
    part="imag",                                # "real", "imag", or "abs"
    normalize_to_T0=False,
    figsize=(6,4), dpi=150, legend_cols=2, title=None, savepath=None
):
    part_map = {"real": np.real, "imag": np.imag, "abs": np.abs}
    if part not in part_map:
        raise ValueError("part must be one of: 'real', 'imag', 'abs'")
    take = part_map[part]

    ni = _as_indices(n_points_array, n_points_sel, name="n_points")
    fi = _as_indices(fwhm_array,    fwhm_sel,    name="fwhm")
    bi = _as_indices(fields,        field_sel,   name="field")
    ti = _as_indices(temperatures,  temps_sel,   name="temperature")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in ni:
        for j in fi:
            for b in bi:
                for t in ti:
                    χ = sus_H_T[i, j, b, t]               # (freqs,) complex
                    y = take(χ)
                    if normalize_to_T0 and t != 0:
                        y = y / take(sus_H_T[i, j, b, 0]).max()
                    label = _label_for_combo(n_points_array, fwhm_array, fields, i, j, b) + f", T={temperatures[t]:g} K"
                    ax.plot(omega_Hz, y, lw=1.2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$ / Hz")
    ax.set_ylabel({ "real": r"$\chi'$", "imag": r"$\chi''$", "abs": r"$|\chi|$"}[part] + " (a.u.)")
    ax.grid(True, which="both", ls=":")
    if title: ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncols=legend_cols)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return ax

# ---------- Cole–Cole (χ'' vs χ') ------------------------------------------
def plot_cole_cole(
    temperatures, fields, n_points_array, fwhm_array,
    sus_H_T,                                    # (npts, fwhm, fields, temps, freqs)
    *,
    n_points_sel="all", fwhm_sel="all", field_sel="all", temps_sel="all",
    figsize=(5,5), dpi=150, legend_cols=2, title=None, savepath=None
):
    ni = _as_indices(n_points_array, n_points_sel, name="n_points")
    fi = _as_indices(fwhm_array,    fwhm_sel,    name="fwhm")
    bi = _as_indices(fields,        field_sel,   name="field")
    ti = _as_indices(temperatures,  temps_sel,   name="temperature")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in ni:
        for j in fi:
            for b in bi:
                for t in ti:
                    χ = sus_H_T[i, j, b, t]     # (freqs,)
                    ax.plot(χ.real, χ.imag, lw=1.2,
                            label=_label_for_combo(n_points_array, fwhm_array, fields, i, j, b)+f", T={temperatures[t]:g} K")
    ax.set_xlabel(r"$\chi'$ (a.u.)")
    ax.set_ylabel(r"$\chi''$ (a.u.)")
    ax.grid(True, which="both", ls=":")
    if title: ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncols=legend_cols)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return ax

# ---------- τ plots ---------------------------------------------------------
def plot_tau_vs_T(
    temperatures, fields, n_points_array, fwhm_array,
    tau_R21_H_T, tau_R41_H_T,                   # (npts, fwhm, fields, temps)
    *,
    n_points_sel="all", fwhm_sel="all", field_sel="all",
    which="R21",                                # "R21" or "R41" or "both"
    yscale="log",                               # "log" or "linear"
    figsize=(6,4), dpi=150, legend_cols=2, title=None, savepath=None
):
    tau_src = {"R21": tau_R21_H_T, "R41": tau_R41_H_T}
    if which not in ("R21","R41","both"):
        raise ValueError("which must be 'R21', 'R41', or 'both'")

    ni = _as_indices(n_points_array, n_points_sel, name="n_points")
    fi = _as_indices(fwhm_array,    fwhm_sel,    name="fwhm")
    bi = _as_indices(fields,        field_sel,   name="field")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _plot_one(tauH, label_prefix):
        for i in ni:
            for j in fi:
                for b in bi:
                    y = tauH[i, j, b]  # (temps,)
                    ax.plot(temperatures, y, marker='o', ms=3, lw=1.1,
                            label=f"{label_prefix} " + _label_for_combo(n_points_array, fwhm_array, fields, i, j, b))

    if which in ("R21","both"):
        _plot_one(tau_src["R21"], r"$\tau_{21}$")
    if which in ("R41","both"):
        _plot_one(tau_src["R41"], r"$\tau_{41}$")

    ax.set_xlabel("T / K")
    ax.set_ylabel(r"$\tau$ (ps)")
    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", ls=":")
    if title: ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncols=legend_cols)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return ax

def plot_tau_vs_field(
    temperatures, fields, n_points_array, fwhm_array,
    tau_R21_H_T, tau_R41_H_T,                   # (npts, fwhm, fields, temps)
    *,
    n_points_sel="all", fwhm_sel="all", temps_sel="all",
    which="R21", yscale="log",
    figsize=(6,4), dpi=150, legend_cols=2, title=None, savepath=None
):
    tau_src = {"R21": tau_R21_H_T, "R41": tau_R41_H_T}
    if which not in ("R21","R41","both"):
        raise ValueError("which must be 'R21', 'R41', or 'both'")

    ni = _as_indices(n_points_array, n_points_sel, name="n_points")
    fi = _as_indices(fwhm_array,    fwhm_sel,    name="fwhm")
    ti = _as_indices(temperatures,  temps_sel,   name="temperature")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _plot_one(tauH, label_prefix):
        for i in ni:
            for j in fi:
                for t in ti:
                    y = tauH[i, j, :, t]  # (fields,)
                    ax.plot(fields, y, marker='o', ms=3, lw=1.1,
                            label=f"{label_prefix} " +
                                  f"n={n_points_array[i]}, FWHM={fwhm_array[j]:g} cm$^{{-1}}$, T={temperatures[t]:g} K")

    if which in ("R21","both"):
        _plot_one(tau_src["R21"], r"$\tau_{21}$")
    if which in ("R41","both"):
        _plot_one(tau_src["R41"], r"$\tau_{41}$")

    ax.set_xlabel("B / T")
    ax.set_ylabel(r"$\tau$ (ps)")
    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", ls=":")
    if title: ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncols=legend_cols)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return ax
