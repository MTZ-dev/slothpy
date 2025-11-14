"""
ac_plotting.py — JCP/AIP–style plotting for AC susceptibility & relaxation times

This module generates publication-ready figures aligned with
Journal of Chemical Physics (AIP Publishing) conventions:
- One-column (≈ 3.37 in / 85 mm) and two-column (≈ 7.00 in / 178 mm) widths
- Sans-serif lettering (Helvetica/Arial), 7–8 pt for single-column figures
- Inward ticks; consistent line/marker sizes; no gridlines by default
- Colorblind-safe palettes (cividis/tab10); clear markers + line styles
- Vector export helpers (PDF/EPS) and high-DPI raster (TIFF/PNG)

Public API preserved; `set_nature_style()` remains as an alias to `set_jcp_style()`.
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
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter

###############################################################################
# 0. Figure size presets (inches) commonly used in JCP/AIP
###############################################################################

def single_column_size(height_ratio: float = 0.75) -> Tuple[float, float]:
    """Return (width, height) for a one-column figure. Width ≈ 3.37 in."""
    w = 3.37
    return (w, w * height_ratio)

def double_column_size(height_ratio: float = 0.75) -> Tuple[float, float]:
    """Return (width, height) for a two-column figure. Width ≈ 7.00 in."""
    w = 7.00
    return (w, w * height_ratio)

###############################################################################
# 1. Styling helpers
###############################################################################

# Discrete styles for components/mechanisms (colorblind-safe)
_MECHS_COLOR = {
    "Orbach":  "#4D4D4D",  # dark gray
    "Raman":   "#1F77B4",  # tab:blue
    "Raman_2": "#17BECF",  # tab:cyan
    "QTM":     "#D62728",  # tab:red
    "Direct":  "#2CA02C",  # tab:green
    "V_d":     "#9467BD",  # tab:purple
}

def _base_rcparams():
    # 7–8 pt lettering for one-column; scale gracefully for two-column
    return {
        # canvas & typography ---------------------------------------------
        "savefig.dpi": 600,
        "savefig.transparent": False,
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "mathtext.fontset": "cm",
        # axes & lines -----------------------------------------------------
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
        # ticks (inward; four spines) -------------------------------------
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # legend -----------------------------------------------------------
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        # grids off by default in JCP-style --------------------------------
        "axes.grid": False,
    }

def set_jcp_style(*, figsize: Tuple[float, float] | None = None):
    """
    Apply global Matplotlib rcParams approximating JCP/AIP figure style.

    Parameters
    ----------
    figsize : (float, float), optional
        Default figure size to register (e.g., single_column_size()).
    """
    rc = _base_rcparams()
    if figsize is None:
        rc["figure.figsize"] = single_column_size()
    else:
        rc["figure.figsize"] = figsize
    # use colorblind-safe default colormap
    rc["image.cmap"] = "cividis"
    mpl.rcParams.update(rc)

def set_nature_style(*args, **kwargs):  # alias for compatibility
    """Backward-compatible alias; now maps to set_jcp_style()."""
    return set_jcp_style(*args, **kwargs)

def savefig_jcp(fig: plt.Figure, path: str | Path, *, dpi: int = 600, formats: Sequence[str] = ("pdf",), bbox: str = "tight"):
    """
    Save a figure in vector formats suitable for JCP submission.

    Examples
    --------
    savefig_jcp(fig, "figure1", formats=("pdf","eps"))
    """
    path = Path(path)
    for ext in formats:
        outfile = path.with_suffix("." + ext.lower())
        fig.savefig(outfile, dpi=dpi, bbox_inches=bbox)

# --- NEW helpers --------------------------------------------------------------

def _auto_ylim_from_exp(log10_tau_exp: np.ndarray, *, pad_top: float = 0.1, pad_bot: float = 0.1):
    """Return (ymin, ymax) using ONLY experimental points (in log10 space)."""
    finite = np.isfinite(log10_tau_exp)
    if not np.any(finite):
        return None  # caller keeps current limits
    lo = np.min(log10_tau_exp[finite])
    hi = np.max(log10_tau_exp[finite])
    span = max(1e-12, hi - lo)
    return lo - pad_bot * span, hi + pad_top * span

def _pretty_log_ticks(ax: plt.Axes, plain_until: float = 1e5):
    """
    Major ticks at powers of 10; labels as plain numbers up to `plain_until`,
    then switch to 10^n style. Minor ticks at 2..9 per decade (no labels).
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1), numticks=12))

    def _fmt(val, pos):
        if val <= 0: 
            return ""
        if val < plain_until:
            # format 0.1, 1, 10, 100, 1000, 10000 without trailing .0
            s = f"{val:g}"
            return s
        # scientific above threshold: 10^{n}
        n = int(np.round(np.log10(val)))
        return r"$10^{{{}}}$".format(n)

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))


def _apply_panel_labels(
    axs, *,
    labels=("(a)","(b)","(c)","(d)"),
    pos: str = "inside",              # "inside" | "below" | "above"
    inside_xy=(0.02, 0.02),
    below_y_offset: float = -0.12,
    above_y_offset: float =  0.06,    # distance above the top spine (axes coords)
    fontsize=8,
):
    """
    Place panel labels inside the axes, just below them, or just above them.
    """
    for lab, ax in zip(labels, np.ravel(axs)):
        if pos == "inside":
            ax.text(inside_xy[0], inside_xy[1], lab, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=fontsize, fontweight="bold")
        elif pos == "below":
            ax.text(0.0, below_y_offset, lab, transform=ax.transAxes,
                    ha="left", va="top", fontsize=fontsize, fontweight="bold",
                    clip_on=False)
        elif pos == "above":
            ax.text(0.0, 1.0 + above_y_offset, lab, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=fontsize, fontweight="bold",
                    clip_on=False)


###############################################################################
# 2. Data containers
###############################################################################

@dataclass(slots=True)
class ACDataset:
    """Single AC-susceptibility trace at fixed (T, H)."""
    T: float  # K
    H: float  # Oe
    nu: np.ndarray  # Hz
    chi_prime: np.ndarray  # cm^3 mol^-1
    chi_bis: np.ndarray    # cm^3 mol^-1
    # optional fitted model curves (often denser)
    nu_model: Optional[np.ndarray] = None
    chi_prime_model: Optional[np.ndarray] = None
    chi_bis_model: Optional[np.ndarray] = None

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
            raise ValueError("Requested field slice not found")
        return self.T, self.tau_exp[:, idx[0]], self.tau_total[:, idx[0]]

    def slice_H(self, T_value: float, /, *, atol: float = 1e-6):
        idx = np.where(np.abs(self.T - T_value) <= atol)[0]
        if idx.size == 0:
            raise ValueError("Requested temperature slice not found")
        return self.H, self.tau_exp[idx[0], :], self.tau_total[idx[0], :]

###############################################################################
# 3. CSV readers
###############################################################################

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
        tag_map = _MODEL_MAP if base.startswith("model") else _EXPERIMENTAL_MAP
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

        datasets.append(ACDataset(T, H, nu, chi_p, chi_pp, nu_m, chi_p_m, chi_pp_m))

    datasets.sort(key=lambda d: (d.T, d.H))
    return datasets

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

def _build_colors(n: int, *, cmap: str | ListedColormap = "cividis", reverse: bool = False):
    base = mpl.colormaps.get_cmap(cmap)
    if reverse:
        base = base.reversed()
    return base(np.linspace(0, 1, n))

def _apply_axis_format(ax: plt.Axes):
    # Scientific formatter; show power outside axis when needed
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

###############################################################################
# 5. Plotting functions
###############################################################################

def plot_chi_prime(
    datasets: Sequence[ACDataset],
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str | ListedColormap = "cividis",
    reverse_cmap: bool = False,
    normalize: bool = False,
    logx: bool = True,
    legend_style: str = "colorbar",
    x_min: float | None = None,
    x_max: float | None = None,
    pretty_logx: bool = True,
):
    """Plot χ′(ν) for a collection of datasets (varying T or H) — JCP style."""
    ax = _get_ax(ax)
    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse_cmap)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0

    # use distinct markers to ensure legibility in grayscale
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
    for i, (clr, ds) in enumerate(zip(colors, datasets)):
        ax.plot(ds.nu, ds.chi_prime / max_norm, ls='', marker=markers[i % len(markers)], ms=1.5, mec='none', color=clr)
        if ds.nu_model is not None:
            ax.plot(ds.nu_model, ds.chi_prime_model / max_norm, color=clr, lw=1.0)

    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(r"$\nu$ / $\mathrm{Hz}$")
    ax.set_ylabel(r"$\chi'$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$"))
    _apply_axis_format(ax)

    if legend_style == "colorbar" and len(datasets) > 1:
        sm = mpl.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(colors),
                                   norm=mpl.colors.Normalize(vmin=0, vmax=len(datasets)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        Ts = {d.T for d in datasets}
        Hs = {d.H for d in datasets}
        if len(Ts) > 1:
            cbar.set_label(r"$T$ / $\mathrm{K}$", labelpad=2)
            cbar.set_ticks([0, len(datasets)-1])
            cbar.set_ticklabels([f"{min(Ts):g}", f"{max(Ts):g}"])
        else:
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=2)
            cbar.set_ticks([0, len(datasets)-1])
            cbar.set_ticklabels([f"{min(Hs):g}", f"{max(Hs):g}"])
    if logx:
        if pretty_logx:
            _pretty_log_ticks(ax)
        else:
            ax.set_xscale("log")
    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(left=x_min, right=x_max)

    return ax

def plot_chi_bis(
    datasets: Sequence[ACDataset],
    **kwargs,
):
    """Plot χ″(ν) — JCP style."""
    ax = kwargs.pop("ax", None)
    ax = _get_ax(ax)
    kwargs = kwargs.copy()
    normalize = kwargs.get("normalize", False)
    cmap = kwargs.get("cmap", "cividis")
    reverse = kwargs.get("reverse_cmap", False)
    logx = kwargs.get("logx", True)
    x_min = kwargs.get("x_min", None)
    x_max = kwargs.get("x_max", None)
    pretty_logx = kwargs.get("pretty_logx", True)

    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    for i, (clr, ds) in enumerate(zip(colors, datasets)):
        ax.plot(ds.nu, ds.chi_bis / max_norm, ls='', marker=markers[i % len(markers)], ms=1.5, mec='none', color=clr)
        if ds.nu_model is not None:
            ax.plot(ds.nu_model, ds.chi_bis_model / max_norm, color=clr, lw=1.0)

    if kwargs.get("logx", True):
        ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$ / $\mathrm{Hz}$")
    ax.set_ylabel(r"$\chi''$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$"))
    _apply_axis_format(ax)

    if kwargs.get("legend_style", "colorbar") == "colorbar" and len(datasets) > 1:
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colors),
            norm=mpl.colors.Normalize(vmin=0, vmax=len(datasets) - 1),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        Ts = {d.T for d in datasets}
        Hs = {d.H for d in datasets}
        if len(Ts) > 1:
            cbar.set_label(r"$T$ / $\mathrm{K}$", labelpad=2)
            cbar.set_ticks([0, len(datasets)-1])
            cbar.set_ticklabels([f"{min(Ts):g}", f"{max(Ts):g}"])
        else:
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=2)
            cbar.set_ticks([0, len(datasets)-1])
            cbar.set_ticklabels([f"{min(Hs):g}", f"{max(Hs):g}"])
    if logx:
        if pretty_logx:
            _pretty_log_ticks(ax)
        else:
            ax.set_xscale("log")
    if (x_min is not None) or (x_max is not None):  # NEW
        ax.set_xlim(left=x_min, right=x_max)

    return ax

def plot_cole_cole(
    datasets: Sequence[ACDataset],
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "cividis",
    reverse_cmap: bool = False,
    normalize: bool = False,
    legend_style: str = "colorbar",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
):
    """Cole–Cole plot (χ′ vs χ″) — JCP style. No grid; clear markers."""
    ax = _get_ax(ax)
    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse_cmap)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    for i, (clr, ds) in enumerate(zip(colors, datasets)):
        ax.plot(ds.chi_prime / max_norm, ds.chi_bis / max_norm, ls='', marker=markers[i % len(markers)], ms=1.5, mec='none', color=clr)
        if ds.chi_prime_model is not None:
            ax.plot(ds.chi_prime_model / max_norm, ds.chi_bis_model / max_norm, color=clr, lw=1.0)

    ax.set_xlabel(r"$\chi'$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$"))
    ax.set_ylabel(r"$\chi''$" + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$"))
    _apply_axis_format(ax)

    if legend_style == "colorbar" and len(datasets) > 1:
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colors),
            norm=mpl.colors.Normalize(vmin=0, vmax=len(datasets) - 1),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        Ts = {d.T for d in datasets}
        Hs = {d.H for d in datasets}
        if len(Ts) > 1:
            cbar.set_label(r"$T$ / $\mathrm{K}$", labelpad=2)
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels([f"{min(Ts):g}", f"{max(Ts):g}"])
        else:
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=2)
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels([f"{min(Hs):g}", f"{max(Hs):g}"])
    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(left=x_min, right=x_max)
    if (y_min is not None) or (y_max is not None):
        ax.set_ylim(bottom=y_min, top=y_max)
    return ax

def plot_tau(
    tau_exp,
    tau_mod,
    *,
    T=None,
    H=None,
    components=None,
    ax=None,
    marker="o",
    cmap: str = "cividis",
    reverse_cmap: bool = False,
    legend_style: str = "colorbar",
    model: bool = True,
    # y-limit controls (log10 space)
    ymin: float | None = None,
    ymax: float | None = None,
    pad_top: float = 0.10,
    pad_bottom: float = 0.10,
    autolimit_on: str = "exp",   # "exp" | "all" | "none"
    # NEW: transparency + whether to add a legend at all
    point_alpha: float = 1.0,
    line_alpha: float = 0.9,
    comp_alpha: float = 0.9,
    add_legend: bool = True,
    x_min: float | None = None,
    x_max: float | None = None,
):
    """Plot log10 τ vs T^{-1} or H — JCP style (no grid; clear labeling)."""
    ax = _get_ax(ax)
    τexp: np.ma.MaskedArray = np.ma.asarray(tau_exp)

    if T is not None:
        values = np.asarray(T)
        x = 1.0 / np.asarray(T)
        x_label = r"$T^{-1}$ / $\mathrm{K^{-1}}$"
        cbar_label = r"$T$ / $\mathrm{K}$"
    elif H is not None:
        values = np.asarray(H)
        x = np.asarray(H)
        x_label = r"$H$ / $\mathrm{Oe}$"
        cbar_label = r"$H$ / $\mathrm{Oe}$"
    else:
        raise ValueError("Temperature or field value must be provided.")

    good   = ~τexp.mask
    x_good = x[good]
    τ_good = τexp.data[good]
    v_good = values[good]

    order  = np.argsort(v_good)
    x_good, τ_good, v_good = x_good[order], τ_good[order], v_good[order]

    colours = _build_colors(len(v_good), cmap=cmap, reverse=reverse_cmap)
    y_exp_log10 = np.log10(τ_good)

    # experimental points (slightly larger default in your current code)
    for v, y, clr in zip(x_good, y_exp_log10, colours):
        ax.plot(v, y, ls='', marker=marker, ms=4, mec='none', color=clr, alpha=point_alpha)

    # model and components (with alpha)
    if model and tau_mod is not None:
        ax.plot(x, np.log10(tau_mod), lw=1.2, color="#6A3D9A", label="model", alpha=line_alpha)

    if components:
        for name, arr in components.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr[:, 0] if T is not None else arr[0, :]
            ax.plot(x, np.log10(arr), lw=1.0, label=name,
                    color=_MECHS_COLOR.get(name, None), alpha=comp_alpha)

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log_{10}\,\tau$ / $\mathrm{s}$")
    _apply_axis_format(ax)

    # optional colorbar keyed to the sweep variable
    if legend_style == "colorbar" and len(v_good) > 1:
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colours),
            norm=mpl.colors.Normalize(vmin=v_good.min(), vmax=v_good.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, labelpad=2)
        cbar.set_ticks([v_good.min(), v_good.max()])

    # y-limits (log10) — auto from selected series unless disabled
    if (ymin is None or ymax is None) and autolimit_on != "none":
        y_for_limits = []
        if autolimit_on in ("exp", "all"):
            y_for_limits.append(y_exp_log10)
        if autolimit_on == "all":
            if model and tau_mod is not None:
                y_for_limits.append(np.log10(np.asarray(tau_mod)))
            if components:
                for arr in components.values():
                    if arr is None:
                        continue
                    a = np.asarray(arr)
                    if a.ndim == 2:
                        a = a[:, 0] if T is not None else a[0, :]
                    y_for_limits.append(np.log10(a))
        if y_for_limits:
            ycat = np.concatenate([y for y in y_for_limits if np.size(y)])
            auto = _auto_ylim_from_exp(ycat, pad_top=pad_top, pad_bot=pad_bottom)
            if auto:
                ymin = auto[0] if ymin is None else ymin
                ymax = auto[1] if ymax is None else ymax

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # only add legend when requested (multi-plot path disables this)
    if add_legend and (components or model):
        ax.legend(frameon=False, fontsize=7, handlelength=1.6)
    
    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(left=x_min, right=x_max)

    return ax

def plot_tau_multi(
    entries: Sequence[tuple] | Sequence[Mapping],
    *,
    ax: plt.Axes | None = None,
    legend: bool = True,                  # dataset legend (points)
    show_component_legend: bool = True,   # NEW: mechanisms/model legend (lines)
    component_legend_loc: str = "lower right",
    **tau_kwargs,
):
    """
    Plot multiple τ datasets on one axis, allowing each entry to choose its own
    field/temperature selector (or explicit T/H arrays). Builds TWO legends:
      - dataset legend (points) using per-entry labels
      - component/model legend (lines) aggregated across entries
    """
    from collections.abc import Mapping as _Mapping
    from collections import OrderedDict

    ax = _get_ax(ax)
    default_markers = ["o","s","D","^","v","<",">","P","X","*"]

    # --------- FIRST PASS: compute global y-limits from ALL experimental points
    pad_top     = tau_kwargs.get("pad_top", 0.10)
    pad_bottom  = tau_kwargs.get("pad_bottom", 0.10)
    explicit_ymin = tau_kwargs.get("ymin", None)
    explicit_ymax = tau_kwargs.get("ymax", None)
    explicit_xmin = tau_kwargs.get("x_min", None)
    explicit_xmax = tau_kwargs.get("x_max", None)
    model = tau_kwargs.get("model", None)

    all_yexp = []

    def _collect_yexp_from_spec(spec) -> None:
        if "tg" in spec:
            tg = spec["tg"]
            sel = spec.get("selector") or {}
            if "field" in sel:
                _, tau_exp, _ = tg.slice_T(float(sel["field"]))
                all_yexp.append(np.log10(np.ma.asarray(tau_exp).compressed()))
            elif "temp" in sel:
                _, tau_exp, _ = tg.slice_H(float(sel["temp"]))
                all_yexp.append(np.log10(np.ma.asarray(tau_exp).compressed()))
        else:
            tau_exp = np.ma.asarray(spec["tau_exp"])
            y = np.log10(tau_exp.compressed() if np.ma.isMaskedArray(tau_exp) else tau_exp[np.isfinite(tau_exp)])
            all_yexp.append(y)

    # normalize entries to specs and pre-collect y's
    normalized_specs: list[tuple[dict, dict]] = []  # (spec, style)
    for i, item in enumerate(entries):
        style = {}
        if isinstance(item, _Mapping):
            spec = dict(item)
            style = dict(spec.pop("style", {}) or {})
        else:
            if isinstance(item[0], TauGrid):
                tg = item[0]
                label = item[1] if len(item) > 1 else None
                comps = None
                selector = None
                for part in item[2:]:
                    if isinstance(part, dict) and ("field" in part or "temp" in part):
                        selector = part
                    elif isinstance(part, dict) and comps is None:
                        comps = part
                    elif isinstance(part, dict):
                        style = dict(part)
                spec = dict(tg=tg, label=label, comps=comps, selector=selector)
            else:
                tau_exp = item[0]
                tau_mod = None
                comps = None
                label = None
                axis_map = None
                for part in item[1:]:
                    if isinstance(part, str):
                        label = part
                    elif isinstance(part, dict):
                        if ("T" in part) or ("H" in part):
                            axis_map = part
                        elif comps is None:
                            comps = part
                        else:
                            style = dict(part)
                    else:
                        if tau_mod is None:
                            tau_mod = part
                spec = dict(tau_exp=tau_exp, tau_mod=tau_mod, comps=comps, label=label, axis_map=axis_map)
        style.setdefault("marker", default_markers[i % len(default_markers)])
        normalized_specs.append((spec, style))
        _collect_yexp_from_spec(spec)

    if (explicit_ymin is None or explicit_ymax is None) and len(all_yexp):
        ycat = np.concatenate([y for y in all_yexp if np.size(y)])
        auto = _auto_ylim_from_exp(ycat, pad_top=pad_top, pad_bot=pad_bottom)
        if auto:
            explicit_ymin = auto[0] if explicit_ymin is None else explicit_ymin
            explicit_ymax = auto[1] if explicit_ymax is None else explicit_ymax

    # --------- SECOND PASS: plot each entry; collect legend handles
    dataset_handles, dataset_labels = [], []
    component_handles = OrderedDict()  # name -> handle (first seen)

    for spec, style in normalized_specs:
        before = len(ax.lines)

        local_kwargs = dict(tau_kwargs)
        local_kwargs.setdefault("autolimit_on", "none")  # global limits already set
        local_kwargs["ymin"] = explicit_ymin
        local_kwargs["ymax"] = explicit_ymax
        local_kwargs["x_min"] = explicit_xmin
        local_kwargs["x_max"] = explicit_xmax
        local_kwargs["add_legend"] = False
        local_kwargs.setdefault("point_alpha", tau_kwargs.get("point_alpha", 0.9))
        local_kwargs.setdefault("line_alpha",  tau_kwargs.get("line_alpha", 0.6))
        local_kwargs.setdefault("comp_alpha",  tau_kwargs.get("comp_alpha", 0.6))

        if "tg" in spec:
            tg = spec["tg"]; comps = spec.get("comps"); label = spec.get("label")
            sel = spec.get("selector") or {}
            if "field" in sel:
                T_vals, tau_exp, tau_mod = tg.slice_T(float(sel["field"]))
                plot_tau(tau_exp, tau_mod, T=T_vals, components=comps, ax=ax,
                         legend_style="none", **style, **local_kwargs)
            elif "temp" in sel:
                H_vals, tau_exp, tau_mod = tg.slice_H(float(sel["temp"]))
                plot_tau(tau_exp, tau_mod, H=H_vals, components=comps, ax=ax,
                         legend_style="none", **style, **local_kwargs)
            else:
                raise ValueError("TauGrid entry needs selector {'field': H} or {'temp': T}.")
        else:
            tau_exp = spec["tau_exp"]; tau_mod = spec.get("tau_mod"); comps = spec.get("comps")
            label   = spec.get("label"); axis_map = spec.get("axis_map") or {}
            if "T" in axis_map:
                plot_tau(tau_exp, tau_mod, T=np.asarray(axis_map["T"]), components=comps, ax=ax,
                         legend_style="none", **style, **local_kwargs)
            elif "H" in axis_map:
                plot_tau(tau_exp, tau_mod, H=np.asarray(axis_map["H"]), components=comps, ax=ax,
                         legend_style="none", **style, **local_kwargs)
            else:
                raise ValueError("Explicit-array entry needs axis_map with {'T': ...} or {'H': ...}.")

        # newly added lines:
        new_lines = ax.lines[before:]

        # dataset handle: pick the first point-like line among new lines
        label = spec.get("label")
        if label:
            point_like = None
            for ln in new_lines:
                if (ln.get_linestyle() in ('', 'None')) and (ln.get_marker() not in (None, 'None', '')):
                    point_like = ln
                    break
            if point_like is None and new_lines:
                point_like = new_lines[0]
            if point_like is not None:
                dataset_handles.append(point_like)
                dataset_labels.append(label)

        # component/model handles: collect first occurrence per name
        for ln in new_lines:
            name = ln.get_label()
            if not name or name.startswith("_"):
                continue
            # heuristics: solid/visible lines with no markers are mechanisms/model
            if (ln.get_linestyle() not in ('', 'None')) and (ln.get_marker() in (None, 'None', '')):
                if name not in component_handles:
                    component_handles[name] = ln

    # Apply global limits
    if explicit_ymin is not None and explicit_ymax is not None:
        ax.set_ylim(explicit_ymin, explicit_ymax)

    # Legends
    leg1 = None
    if legend and dataset_handles:
        leg1 = ax.legend(dataset_handles, dataset_labels,
                         frameon=False, fontsize=7, handlelength=1.6, loc="upper left")
    if show_component_legend and component_handles:
        leg2 = ax.legend(list(component_handles.values()), list(component_handles.keys()),
                         frameon=False, fontsize=7, handlelength=1.6, loc=component_legend_loc)
        if leg1 is not None:
            ax.add_artist(leg1)

    return ax

def plot_composite_panel(
    ac_datasets: Sequence[ACDataset],
    *,
    tau_grid: Tuple[Sequence[float], Sequence[float]] | TauGrid | None = None,
    # NEW: entries can now carry different selectors and mechanisms
    tau_grids_multi: Sequence[tuple] | Sequence[Mapping] | None = None,
    field_sel: float = None,
    temp_sel: float = None,
    mechanisms: Mapping[str, Sequence[float]] | None = None,
    cmap: str | ListedColormap = "cividis",
    reverse_cmap: bool = False,
    normalize_chi: bool = False,
    suptitle: Optional[str] = None,
    grid: Tuple[int, int] = (2, 2),
    cluster_tol_H: float = 0.5,
    cluster_tol_T: float = 0.05,
    model: bool = True,
    figure_size: Tuple[float, float] | None = None,
    # label & τ-limit controls
    panel_label_pos: str = "above",
    tau_ylim: Tuple[float, float] | None = None,
    tau_pad_top: float = 0.10,
    tau_pad_bottom: float = 0.10,
    tau_autolimit_on: str = "exp",
    xlim_im:  Tuple[float,float] | None = None,
    xlim_re:  Tuple[float,float] | None = None,
    xlim_cole: Tuple[float,float] | None = None,
    ylim_cole: Tuple[float,float] | None = None,
    xlim_tau: Tuple[float,float] | None = None,
    pretty_logx: bool = True,  
):
    """
    Create the standard 4-panel figure (χ″, χ′, Cole–Cole, log10 τ) — JCP style.
    """
    set_jcp_style(figsize=figure_size or double_column_size(height_ratio=0.78))
    fig, axs = plt.subplots(2, 2, figsize=figure_size or double_column_size(0.78), constrained_layout=True)
    ax_im, ax_re, ax_cole, ax_tau = axs.flat

    # Filter datasets by requested slice
    if field_sel is not None:
        datasets = [d for d in ac_datasets if abs(d.H - field_sel) < cluster_tol_H]
        if not datasets:
            raise SystemExit(f"No AC datasets at ≈{field_sel} Oe found")
    elif temp_sel is not None:
        datasets = [d for d in ac_datasets if abs(d.T - temp_sel) < cluster_tol_T]
        if not datasets:
            raise SystemExit(f"No AC datasets at ≈{temp_sel} K found")
    else:
        raise ValueError("One of: field_sel or temp_sel must be provided.")

    plot_chi_bis(datasets, ax=ax_im, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi,pretty_logx=pretty_logx,
                 x_min=(xlim_im[0] if xlim_im else None), x_max=(xlim_im[1] if xlim_im else None))
    plot_chi_prime(datasets, ax=ax_re, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi, pretty_logx=pretty_logx,
                   x_min=(xlim_re[0] if xlim_re else None), x_max=(xlim_re[1] if xlim_re else None))
    plot_cole_cole(datasets, ax=ax_cole, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi,
                   x_min=(xlim_cole[0] if xlim_cole else None), x_max=(xlim_cole[1] if xlim_cole else None),
                   y_min=(xlim_cole[0] if xlim_cole else None), y_max=(xlim_cole[1] if xlim_cole else None))

    # ---------------- τ panel(s) ----------------
    tau_common_kwargs = dict(
        ymin=(tau_ylim[0] if tau_ylim else None),
        ymax=(tau_ylim[1] if tau_ylim else None),
        x_min=(xlim_tau[0] if xlim_tau else None),
        x_max=(xlim_tau[1] if xlim_tau else None),
        pad_top=tau_pad_top,
        pad_bottom=tau_pad_bottom,
        autolimit_on=tau_autolimit_on,
        model=model,
    )

    # Single-grid path unchanged
    if isinstance(tau_grid, TauGrid):
        tg = tau_grid
        if field_sel is not None:
            T, tau_exp, tau_mod = tg.slice_T(field_sel)
            plot_tau(tau_exp, tau_mod, T=T, components=mechanisms, ax=ax_tau,
                     cmap=cmap, reverse_cmap=reverse_cmap, model=model, **tau_common_kwargs)
        elif temp_sel is not None:
            H, tau_exp, tau_mod = tg.slice_H(temp_sel)
            plot_tau(tau_exp, tau_mod, H=H, components=mechanisms, ax=ax_tau,
                     cmap=cmap, reverse_cmap=reverse_cmap, model=model, **tau_common_kwargs)

    # Multi-grid path now supports per-entry selectors and mechanisms
    elif tau_grids_multi:
        # Just forward the rich specs to plot_tau_multi; it will slice each tg
        plot_tau_multi(
            tau_grids_multi,
            ax=ax_tau,
            **tau_common_kwargs
        )
    else:
        ax_tau.set_visible(False)

    # Panel labels
    _apply_panel_labels(axs, pos=panel_label_pos)

    if suptitle:
        fig.suptitle(suptitle, fontsize=9.0, y=1.02)

    # --- Single-panel exports (τ mirrors the choice above) ---
    single_size = single_column_size(height_ratio=0.82)

    fig_im, ax_im_s   = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_chi_bis(datasets, ax=ax_im_s, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi, legend_style="colorbar")
    if suptitle: fig_im.suptitle(suptitle, fontsize=9.0)

    fig_re, ax_re_s   = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_chi_prime(datasets, ax=ax_re_s, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi, legend_style="colorbar")
    if suptitle: fig_re.suptitle(suptitle, fontsize=9.0)

    fig_cc, ax_cc_s   = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_cole_cole(datasets, ax=ax_cc_s, cmap=cmap, reverse_cmap=reverse_cmap, normalize=normalize_chi, legend_style="colorbar")
    if suptitle: fig_cc.suptitle(suptitle, fontsize=9.0)

    fig_tau_s, ax_tau_s = plt.subplots(figsize=single_size, constrained_layout=True)
    if isinstance(tau_grid, TauGrid):
        if field_sel is not None:
            plot_tau(tau_exp, tau_mod, T=T, components=mechanisms, ax=ax_tau_s,
                     cmap=cmap, reverse_cmap=reverse_cmap, model=model, **tau_common_kwargs)
        else:
            plot_tau(tau_exp, tau_mod, H=H, components=mechanisms, ax=ax_tau_s,
                     cmap=cmap, reverse_cmap=reverse_cmap, model=model, **tau_common_kwargs)
        if suptitle: fig_tau_s.suptitle(suptitle, fontsize=9.0)
    elif tau_grids_multi:
        plot_tau_multi(tau_grids_multi, ax=ax_tau_s, **tau_common_kwargs)
        if suptitle: fig_tau_s.suptitle(suptitle, fontsize=9.0)

    fig_comb = fig
    return fig_im, fig_re, fig_cc, fig_tau_s, fig_comb

# -----------------------------------------------------------------------------
# Optional CLI demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """End-to-end test using sample CSVs; writes vector-format figures."""
    import pathlib as _pl
    set_jcp_style(figsize=double_column_size(0.78))

    base_dir = _pl.Path(__file__).with_suffix("").parent
    ac_file = base_dir / "ac_Tb_npoints_21_fwhm_7.45_no_corr.csv" # "ac_Tb_npoints_21_fwhm_7.45_no_corr.csv" # "ac_TbCo_3000_Oe.csv" "tau_TbCo_0_Oe.csv"
    tau_file = base_dir / "Tb_3000_45_r21.csv" # "tau_TbCo_0_Oe_ab_initio.csv"
    tau_file2 = base_dir / "Tb_3000_45.csv" # "1.csv" # "test_fit_tau2.csv"
    tau_file3 = base_dir / "tau_TbCo_3000_Oe.csv" # "2.csv" #
    tau_file4 = base_dir / "raman_3000_8_3_1_tt.csv"
    tau_file5 = base_dir / "tau_TbCo_0_Oe.csv"
    if not ac_file.exists() or not tau_file.exists():
        raise SystemExit("Sample CSV files not found next to ac_plotting.py")

    ac_datasets = read_ac_csv(ac_file)
    tau_grid = read_tau_csv(tau_file)
    tau_grid2 = read_tau_csv(tau_file2)
    tau_grid3 = read_tau_csv(tau_file3)
    tau_grid4 = read_tau_csv(tau_file4)
    tau_grid5 = read_tau_csv(tau_file5)

    field_sel = 3000.0  # Oe
    suptitle = f"TbCo  (H = {field_sel:g} Oe)"

    figs = plot_composite_panel(
        ac_datasets=ac_datasets,
        # tau_grid=tau_grid, # SINGLE WAY
        tau_grids_multi=[
        # TauGrid + per-entry selector + its own mechanisms + (optional) style
        # (tau_grid,  "Exp",  {"Orbach": tau_grid.mechanisms.get("Orbach"), "V_d": tau_grid.mechanisms.get("V_d")}, {"field": 0.0}, {"marker": "o"}),
        (tau_grid,  "Ab initio R21",  {}, {"field": 3000.0}, {"marker": "s"}),
        (tau_grid4,  "Ab initio R41",  {}, {"field": 3000.0}, {"marker": "v"}),
        (tau_grid2,  "Ab initio",  {}, {"field": 3000.0}, {"marker": "o"}),
        (tau_grid3,  "Exp",  {}, {"field": 3000.0}, {"marker": "X"}),
        # (tau_grid,  "Ab initio",  {}, {"field": 0.0}, {"marker": "s"}),
        # (tau_grid5,  "Exp",  {}, {"field": 0.0}, {"marker": "v"}),
        # (tau_grid2, "Theo", {},  {"field": 3000.0}, {"marker": "s"}),
        # (tau_grid3, "Theo", {},  {"field": 3000.0}, {"marker": "X"}),
        # (tau_grid4, "Theo", {},  {"field": 0.0}, {"marker": "s"}),
        # (tau_grid5, "Theo", {},  {"field": 0.0}, {"marker": "X"}),
        # …or pass explicit arrays if you’ve already sliced them
        # (tau_exp_arr, tau_mod_arr, None, "Sample C", {"T": T_vals}, {"marker":"D"}),
    ],
        # tau_grids_multi=[(tau_grid, "Exp", None), (tau_grid2, "Theo", None)], # # SINGLE WAY
        mechanisms={"Orbach": tau_grid.mechanisms.get("Orbach"), "V_d": tau_grid.mechanisms.get("V_d")},
        field_sel=field_sel,
        cmap="cividis",
        suptitle=suptitle,
        reverse_cmap=False,
        model=False,
        xlim_im=(0.01,1e4),
        xlim_re=(0.01,1e4),
    )

    out_dir = base_dir / "seminarium"
    out_dir.mkdir(exist_ok=True)
    # Save composite in PDF/EPS; single panels in PDF
    _, _, _, _, fig_comb = figs
    savefig_jcp(fig_comb, out_dir / "TbCo_ac_comparison_r41_8_3_1_tt", formats=("pdf",))
    print("Saved composite (PDF/EPS) to", out_dir)
