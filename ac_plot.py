"""
ac_plotting.py
A matplotlib‑based plotting toolkit for AC magnetic susceptibility datasets and
relaxation‑time analysis of single‑molecule magnets (SMMs).

This module generates publication‑ready figures compliant with Nature
Communications graphical standards.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter

__all__ = [
    # style helpers -------------------------------------------------------
    "set_nature_style",
    # I/O helpers ---------------------------------------------------------
    "read_ac_csv",
    "read_tau_csv",
    # containers ----------------------------------------------------------
    "ACDataset",
    "TauGrid",
    # plotting helpers ----------------------------------------------------
    "plot_chi_prime",
    "plot_chi_bis",
    "plot_cole_cole",
    "plot_tau_vs_temperature",
    "plot_composite_panel",
]

###############################################################################
# 1. Styling helpers
###############################################################################

def set_nature_style():
    """Apply global Matplotlib *rcParams* approximating Nature‑style figures."""
    mpl.rcParams.update(
        {
            # canvas & typography -----------------------------------------
            "figure.figsize": (7, 5),  # ≈90 mm × 65 mm
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "mathtext.fontset": "cm",
            # axes & lines ------------------------------------------------
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            # ticks ------------------------------------------------------
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # legend -----------------------------------------------------
            "legend.fontsize": 7,
            "legend.frameon": False,
        }
    )

###############################################################################
# 2. Data containers
###############################################################################

@dataclass(slots=True)
class ACDataset:
    """Single AC‑susceptibility trace at fixed (*T*, *H*)."""

    T: float  # K
    H: float  # Oe
    nu: np.ndarray  # Hz
    chi_prime: np.ndarray  # cm^3 mol⁻¹
    chi_bis: np.ndarray  # cm^3 mol⁻¹
    # optional fitted model curves (often denser)
    nu_model: Optional[np.ndarray] = None
    chi_prime_model: Optional[np.ndarray] = None
    chi_bis_model: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def max_chi_pp(self) -> float:
        return float(np.nanmax(self.chi_bis)) if self.chi_bis.size else np.nan


@dataclass(slots=True)
class TauGrid:
    T: np.ndarray
    H: np.ndarray
    tau_exp:   np.ma.MaskedArray
    tau_total: np.ma.MaskedArray
    mechanisms: Dict[str, np.ma.MaskedArray]

    def slice_T(self, H_value: float, /, *, atol: float = 1e-6):
        idx = np.where(np.abs(self.H - H_value) <= atol)[0]
        if idx.size == 0:
            raise ValueError("field not found")
        return self.T, self.tau_exp[:, idx[0]], self.tau_total[:, idx[0]]

    def slice_H(self, T_value: float, /, *, atol: float = 1e-6):
        idx = np.where(np.abs(self.T - T_value) <= atol)[0]
        if idx.size == 0:
            raise ValueError("temperature not found")
        return self.H, self.tau_exp[idx[0], :], self.tau_total[idx[0], :]

###############################################################################
# 3. CSV readers
###############################################################################

# ---------------------------------------------------------------------
# 3.1  AC‑susceptibility “*_ac_all.csv”
# ---------------------------------------------------------------------

_TH_PATTERN = re.compile(r"T\s*=\s*([\d.]+).*?H\s*=\s*([\d.]+)", re.I)


def _parse_TH(label: str) -> Tuple[float, float]:
    m = _TH_PATTERN.search(label)
    if m is None:
        raise ValueError(f"Cannot extract T/H from column {label!r}")
    return float(m.group(1)), float(m.group(2))

_EXPERIMENTAL_MAP = {
    "frequency": "nu",
    "chiprimemol": "chi_p",
    "chibismol": "chi_pp",
}
_MODEL_MAP = {f"model{k}": v + "_model" for k, v in _EXPERIMENTAL_MAP.items()}

def read_ac_csv(path: str | Path) -> List[ACDataset]:
    df = pd.read_csv(path, sep=";", engine="python")
    df = df.drop(columns=[c for c in ("Name", "Value", "Error") if c in df.columns])

    groups: Dict[Tuple[float, float], Dict[str, pd.Series]] = {}
    for col in df.columns:
        base = col.split(" ")[0].lower()
        if base.startswith("model"):
            tag_map = _MODEL_MAP
        else:
            tag_map = _EXPERIMENTAL_MAP

        for key, label in tag_map.items():
            if base.startswith(key):
                T, H = _parse_TH(col)
                groups.setdefault((T, H), {})[label] = df[col]
                break

    datasets: List[ACDataset] = []
    for (T, H), blk in groups.items():
        nu = blk["nu"].to_numpy(float)
        chi_p = blk["chi_p"].to_numpy(float)
        chi_pp = blk["chi_pp"].to_numpy(float)
        L = min(len(nu), len(chi_p), len(chi_pp))
        nu, chi_p, chi_pp = nu[:L], chi_p[:L], chi_pp[:L]

        nu_m = chi_p_m = chi_pp_m = None
        if "nu_model" in blk:
            nu_m = blk["nu_model"].to_numpy(float)
            chi_p_m = blk["chi_p_model"].to_numpy(float)
            chi_pp_m = blk["chi_pp_model"].to_numpy(float)
            Lm = min(len(nu_m), len(chi_p_m), len(chi_pp_m))
            nu_m, chi_p_m, chi_pp_m = nu_m[:Lm], chi_p_m[:Lm], chi_pp_m[:Lm]

        datasets.append(
            ACDataset(T, H, nu, chi_p, chi_pp, nu_m, chi_p_m, chi_pp_m)
        )

    datasets.sort(key=lambda d: (d.T, d.H))
    return datasets

# ---------------------------------------------------------------------
# 3.2  Relaxation‑time “*_tau_all.csv”
# ---------------------------------------------------------------------

_BASE_MECHS = ["Orbach", "Raman", "Raman_2", "QTM", "Direct", "V_d"]


def read_tau_csv(path: str | Path, *, mechanisms: Sequence[str] | None = None) -> TauGrid:
    df = pd.read_csv(path, sep=";", engine="python")
    mech_names = [m for m in _BASE_MECHS if mechanisms is None or m.lower() in {x.lower() for x in mechanisms}]

    if not {"T", "H", "tau"} <= set(df.columns):
        raise ValueError("Columns T, H, tau not found")

    exp_df = df[["T", "H", "tau"]].dropna().astype(float)

    def _collect_grid(suffix: str | None):
        cols = [f"Temp{suffix or ''}", f"Field{suffix or ''}"] + [f"{m}{suffix or ''}" for m in [*mech_names, "Tau"]]
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame(columns=["Temp", "Field", *mech_names, "Tau"])
        block = (
            df[present]
            .rename(columns=lambda c: c.split(".")[0])
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=["Temp", "Field"], how="any")
        )
        return block

    ct_df = _collect_grid("")
    cf_df = _collect_grid(".1")

    T_vals = np.unique(np.concatenate([exp_df["T"].values, ct_df["Temp"].values if not ct_df.empty else [], cf_df["Temp"].values if not cf_df.empty else []]))
    H_vals = np.unique(np.concatenate([exp_df["H"].values, ct_df["Field"].values if not ct_df.empty else [], cf_df["Field"].values if not cf_df.empty else []]))

    shape = (T_vals.size, H_vals.size)
    def _empty():
        return np.ma.masked_all(shape, dtype=float)

    tau_total = _empty()
    tau_exp = _empty()
    mech_grids = {m: _empty() for m in mech_names}

    def _put(t, h, val, grid):
        if np.isnan(val):
            return
        ti = np.where(T_vals == t)[0]
        hi = np.where(H_vals == h)[0]
        if ti.size and hi.size:
            grid[ti[0], hi[0]] = val

    for row in exp_df.itertuples(index=False):
        _put(row.T, row.H, row.tau, tau_exp)

    for block in (ct_df, cf_df):
        for row in block.itertuples(index=False):
            d = row._asdict()
            T_val, H_val = d["Temp"], d["Field"]
            _put(T_val, H_val, d.get("Tau", np.nan), tau_total)
            for mech in mech_grids:
                _put(T_val, H_val, d.get(mech, np.nan), mech_grids[mech])

    return TauGrid(T_vals, H_vals, tau_exp, tau_total, mech_grids)

###############################################################################
# 4. Internal helpers
###############################################################################

def _get_ax(ax: Optional[plt.Axes] = None):
    return ax if ax is not None else plt.subplots()[1]


def _build_colors(n: int, *, cmap: str | ListedColormap = "turbo", reverse: bool = False):
    base = mpl.colormaps.get_cmap(cmap)
    if reverse:
        base = base.reversed()
    return base(np.linspace(0, 1, n))

###############################################################################
# 5. Plotting functions
###############################################################################

def plot_chi_prime(
    datasets: Sequence[ACDataset],
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str | ListedColormap = "turbo",
    reverse_cmap: bool = False,
    normalize: bool = False,
    logx: bool = True,
    legend_style: str = "colorbar",
):
    """Plot χ′(ν) for a collection of datasets (varying T or H)."""
    ax = _get_ax(ax)
    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse_cmap)

    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0

    for clr, ds in zip(colors, datasets):
        ax.plot(ds.nu, ds.chi_prime / max_norm, ls='', marker='o', ms=3, color=clr)

        if ds.nu_model is not None:
            ax.plot(ds.nu_model, ds.chi_prime_model / max_norm, color=clr, lw=0.9)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$ / $\mathrm{Hz}$")
    ax.set_ylabel(r"$\chi'$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^3 mol^{-1}}$"))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, which="both", ls=":", lw=0.4)

    if legend_style == "colorbar":
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colors),
            norm=mpl.colors.Normalize(vmin=0, vmax=len(datasets) - 1),
        )
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)

        Ts = {d.T for d in datasets}
        Hs = {d.H for d in datasets}
        if len(Ts) > 1:
            tick_labels = [f"{min(Ts):g}", f"{max(Ts):g}"]
            cbar.set_label(r"$T$ / $\mathrm{K}$", labelpad=-8)
        else:
            tick_labels = [f"{min(Hs):g}", f"{max(Hs):g}"]
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=-8)

        cbar.set_ticks([0, len(datasets) - 1])
        cbar.set_ticklabels(tick_labels)

    return ax


def plot_chi_bis(
    datasets: Sequence[ACDataset],
    **kwargs,
):
    ax = kwargs.pop("ax", None)
    ax = _get_ax(ax)
    kwargs = kwargs.copy()
    normalize = kwargs.get("normalize", False)
    cmap = kwargs.get("cmap", "turbo")
    reverse = kwargs.get("reverse_cmap", False)

    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0

    for clr, ds in zip(colors, datasets):
        ax.plot(ds.nu, ds.chi_bis / max_norm, ls='', marker='o', ms=3, color=clr)
        
        if ds.nu_model is not None:
            ax.plot(ds.nu_model, ds.chi_bis_model / max_norm, color=clr, lw=0.9)

    if kwargs.get("logx", True):
        ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$ / $\mathrm{Hz}$")
    ax.set_ylabel(r"$\chi''$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^3 mol^{-1}}$"))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, which="both", ls=":", lw=0.4)

    if kwargs.get("legend_style", "colorbar") == "colorbar":
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colors),
            norm=mpl.colors.Normalize(vmin=0, vmax=len(datasets) - 1),
        )
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)

        Ts = {d.T for d in datasets}
        Hs = {d.H for d in datasets}
        if len(Ts) > 1:
            tick_labels = [f"{min(Ts):g}", f"{max(Ts):g}"]
            cbar.set_label(r"$T$ / $\mathrm{K}$", labelpad=-8)
        else:
            tick_labels = [f"{min(Hs):g}", f"{max(Hs):g}"]
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=-8)

        cbar.set_ticks([0, len(datasets) - 1])
        cbar.set_ticklabels(tick_labels)


def plot_cole_cole(
    datasets: Sequence[ACDataset],
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "turbo",
    reverse_cmap: bool = False,
    normalize: bool = False,
    legend_style: str = "colorbar",
):
    ax = _get_ax(ax)
    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse_cmap)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0


    for clr, ds in zip(colors, datasets):
        ax.plot(ds.chi_prime / max_norm, ds.chi_bis / max_norm, ls='', marker='o', ms=3, color=clr)

        if ds.chi_prime_model is not None:
            ax.plot(ds.chi_prime_model / max_norm, ds.chi_bis_model / max_norm, color=clr, lw=0.9)

    ax.set_xlabel(r"$\chi'$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^3 mol^{-1}}$"))
    ax.set_ylabel(r"$\chi''$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^3 mol^{-1}}$"))
    ax.grid(True, which="both", ls=":", lw=0.4)


    if legend_style == "colorbar":
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colors),
            norm=mpl.colors.Normalize(vmin=0, vmax=len(datasets) - 1),
        )
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)

        Ts = {d.T for d in datasets}
        Hs = {d.H for d in datasets}
        if len(Ts) > 1:
            tick_labels = [f"{min(Ts):g}", f"{max(Ts):g}"]
            cbar.set_label(r"$T$ / $\mathrm{K}$", labelpad=-8)
        else:
            tick_labels = [f"{min(Hs):g}", f"{max(Hs):g}"]
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=-8)

        cbar.set_ticks([0, len(datasets) - 1])
        cbar.set_ticklabels(tick_labels)

    return ax


def plot_tau(tau_exp, tau_mod, *, T=None, H=None, components=None, ax=None, marker="o", cmap: str = "turbo", reverse_cmap: bool = False, legend_style: str = "colorbar"):

    ax = _get_ax(ax)

    τexp: np.ma.MaskedArray = np.ma.asarray(tau_exp)

    if T is not None:
        values = np.asarray(T)    
        x = 1 / np.asarray(T)
    elif H is not None:
        values = np.asarray(H)
        x = np.asarray(H)
    else:
        raise ValueError("Temperature or field value must be provided.")
    
    # keep only where τexp is unmasked
    good   = ~τexp.mask
    x_good = x[good]
    τ_good = τexp.data[good]
    v_good = values[good]

    # sort for a nice gradient (optional)
    order  = np.argsort(v_good)
    x_good, τ_good, v_good = x_good[order], τ_good[order], v_good[order]

    colours = _build_colors(len(v_good), cmap=cmap, reverse=reverse_cmap)

    for v, y, clr in zip(x_good, τ_good, colours):
        ax.plot(v, np.log10(y), ls='', marker=marker, ms=3, color=clr)
    
    # model surface line
    ax.plot(x, np.log10(tau_mod), lw=0.9, color="purple", label="model")

    if components:
        for name, arr in components.items():
            if arr is None:  # skip missing grids
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2:          # (NT,NH) grid – assume same slice
                if T is not None:
                    arr = arr[:, 0]
                elif H is not None:
                    arr = arr[0, :]
            ax.plot(x, np.log10(arr), lw=0.9, label=name, color="purple")

    if T is not None:
        ax.set_xlabel(r"$T^{-1}$ / $\mathrm{K^{-1}}$")
    if H is not None:
        ax.set_xlabel(r"$H$ / $\mathrm{Oe}$")
    ax.set_ylabel(r"$\log_{10}\,\tau$ / $\mathrm{s}$")
    ax.grid(True, which="both", ls=":", lw=0.4)
    ax.legend(frameon=False, fontsize=7)

    if legend_style == "colorbar" and len(v_good) > 1:
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colours),
            norm=mpl.colors.Normalize(vmin=v_good.min(), vmax=v_good.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$T$ / $\mathrm{K}$" if T is not None else r"$H$ / $\mathrm{Oe}", labelpad=-8)
        cbar.set_ticks([v_good.min(), v_good.max()])

    return ax


def plot_composite_panel(
    ac_datasets: Sequence[ACDataset],
    *,
    tau_grid: Tuple[Sequence[float], Sequence[float]] | None = None,
    field_sel: float = None,
    temp_sel: float = None,
    mechanisms: Mapping[str, Sequence[float]] | None = None,
    cmap: str | ListedColormap = "turbo",
    reverse_cmap: bool = False,
    normalize_chi: bool = False,
    suptitle: Optional[str] = None,
    grid: Tuple[int, int] = (2, 2),
    cluster_tol_H: float = 0.5,
    cluster_tol_T: float = 0.05,
):
    """Create the standard 4‑panel figure (χ″, χ′, Cole‑Cole, ln τ).

    *datasets* should all correspond to the same variable sweep (e.g. same H).
    """
    set_nature_style()
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    ax_im, ax_re, ax_cole, ax_tau = axs.flat

    if field_sel is not None:
        datasets = [d for d in ac_datasets if abs(d.H - field_sel) < cluster_tol_H]
        if not datasets:
            raise SystemExit(f"No AC datasets at ≈{field_sel} Oe found")
    elif temp_sel is not None:
        datasets = [d for d in ac_datasets if abs(d.T - temp_sel) < cluster_tol_T]
        if not datasets:
            raise SystemExit(f"No AC datasets at ≈{temp_sel} T found")
    else:
        raise ValueError("One of: field_sel or tem_sel must be provided.")

    plot_chi_bis(datasets, ax=ax_im, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi)
    plot_chi_prime(datasets, ax=ax_re, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi)
    plot_cole_cole(datasets, ax=ax_cole, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi)

    if field_sel is not None:
        T, tau_exp, tau_mod = tau_grid.slice_T(field_sel)
        plot_tau(tau_exp, tau_mod, T=T, components=mechanisms, ax=ax_tau, cmap=cmap, reverse_cmap=reverse_cmap)
    elif temp_sel is not None:
        H, tau_exp, tau_mod = tau_grid.slice_H(temp_sel)
        plot_tau(tau_exp, tau_mod, H=H, components=mechanisms, ax=ax_tau, cmap=cmap, reverse_cmap=reverse_cmap)
    else:
        ax_tau.set_visible(False)

    labels = ["(a)", "(b)", "(c)", "(d)"]
    fig.canvas.draw()                        # make sure positions are known
    offset = 0.02                            # figure‑fraction offset ≈ 2 mm on a 8″ figure

    for label, ax in zip(labels, axs.flat):
        bbox = ax.get_position()
        fig.text(bbox.x0 - offset, bbox.y1 + offset, label, ha="right", va="bottom", fontsize=9, fontweight="bold")

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.05)
    
    single_size = (3.6, 3.2)      # tweak if needed

    fig_im, ax_im_s   = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_chi_bis(datasets, ax=ax_im_s,
                          cmap=cmap, reverse_cmap=reverse_cmap,
                          normalize=normalize_chi, legend_style="colorbar")
    if suptitle:
        fig_im.suptitle(suptitle, fontsize=10)

    fig_re, ax_re_s   = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_chi_prime(datasets, ax=ax_re_s,
                   cmap=cmap, reverse_cmap=reverse_cmap,
                   normalize=normalize_chi, legend_style="colorbar")
    if suptitle:
        fig_re.suptitle(suptitle, fontsize=10)

    fig_cc, ax_cc_s   = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_cole_cole(datasets, ax=ax_cc_s,
                   cmap=cmap, reverse_cmap=reverse_cmap,
                   normalize=normalize_chi, legend_style="colorbar")
    if suptitle:
        fig_cc.suptitle(suptitle, fontsize=10)

    fig_tau, ax_tau_s = plt.subplots(figsize=single_size, constrained_layout=True)
    if field_sel is not None:
        plot_tau(tau_exp, tau_mod, T=T, components=mechanisms, ax=ax_tau_s, cmap=cmap, reverse_cmap=reverse_cmap)
    else:
        plot_tau(tau_exp, tau_mod, H=H, components=mechanisms, ax=ax_tau_s, cmap=cmap, reverse_cmap=reverse_cmap)
    if suptitle:
        fig_tau.suptitle(suptitle, fontsize=10)

    fig_comb = fig

    return fig_im, fig_re, fig_cc, fig_tau, fig_comb


if __name__ == "__main__":
    """Run a quick end‑to‑end test using the sample CSV files.

    Usage:  python ac_plotting.py  (from the directory that contains the CSVs)
    Generates a composite figure in ./seminarium/YbCo_ac_composite.png
    """
    import pathlib as _pl
    import matplotlib as _mpl
    import cmcrameri.cm as _cmc

    # ------------------------------------------------------------------
    # 0. register the custom colormap & styling
    # ------------------------------------------------------------------
    _mpl.colormaps.register(_cmc.managua, name="managua")
    set_nature_style()

    # ------------------------------------------------------------------
    # 1. locate CSVs relative to this file (adjust if needed)
    # ------------------------------------------------------------------
    base_dir = _pl.Path(__file__).with_suffix("").parent
    ac_file = base_dir / "Yb_ac_all.csv"
    tau_file = base_dir / "Yb_tau_all.csv"

    if not ac_file.exists() or not tau_file.exists():
        raise SystemExit("Sample CSV files not found next to ac_plotting.py")

    # ------------------------------------------------------------------
    # 2. load data
    # ------------------------------------------------------------------
    ac_datasets = read_ac_csv(ac_file)
    tau_grid = read_tau_csv(tau_file)

    # pick the 1000 Oe slice -------------------------------------------
    field_sel = 1000.0  # Oe
    temp_sel = 1.8

    # ------------------------------------------------------------------
    # 3. compose figure
    # ------------------------------------------------------------------
    fig_im, fig_re, fig_cc, fig_tau, fig_comb = plot_composite_panel(
        ac_datasets=ac_datasets,
        tau_grid=tau_grid,
        mechanisms={
            "Orbach": tau_grid.mechanisms.get("Orbach"),
        },
        field_sel=field_sel,
        cmap="managua",
        suptitle=f"YbCo  (H = {field_sel:g} Oe)",
        reverse_cmap=True,
    )

    out_dir = base_dir / "seminarium"
    out_dir.mkdir(exist_ok=True)
    outfile_comb = out_dir / "YbCo_ac_composite.png"
    outfile_im = out_dir / "YbCo_ac_bis.png"
    fig_im.savefig(outfile_im, dpi=600, bbox_inches="tight")
    fig_comb.savefig(outfile_comb, dpi=600, bbox_inches="tight")
    print(f"Saved composite figure to {outfile_comb.relative_to(base_dir)}")