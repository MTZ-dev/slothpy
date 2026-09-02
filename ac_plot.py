"""
ac_plotting.py — JCP/AIP–style plotting for AC susceptibility & relaxation times

This module generates publication-ready figures aligned with
Journal of Chemical Physics (AIP Publishing) conventions:
- One-column (≈ 3.37 in / 85 mm) and two-column (≈ 7.00 in / 178 mm) widths
- Sans-serif lettering (Helvetica/Arial), 7–8 pt for single-column figures
- Inward ticks; consistent line/marker sizes; no gridlines by default
- Colorblind-safe palettes (cividis/tab10); clear markers + line styles
- Vector export helpers (PDF/EPS) and high-DPI raster (TIFF/PNG)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.colormaps.register(cmc.managua, name="managua")
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter

###############################################################################
# 0. Figure size presets (inches) commonly used in JCP/AIP
###############################################################################


def single_column_size(height_ratio: float = 0.75) -> tuple[float, float]:
    """Return (width, height) for a one-column figure. Width ≈ 3.37 in."""
    w = 3.37  # 8.5598 cm
    return (w, w * height_ratio)


def double_column_size(height_ratio: float = 0.75) -> tuple[float, float]:
    """Return (width, height) for a two-column figure. Width ≈ 7.00 in."""
    w = 7.00  # 17.78 cm
    return (w, w * height_ratio)


###############################################################################
# 1. Styling helpers
###############################################################################

# Discrete styles for components/mechanisms (colorblind-safe)
_MECHS_COLOR = {
    "Orbach": "#D62728",  # tab:red
    "Raman": "#1F77B4",  # tab:blue
    "Raman_2": "#17BECF",  # tab:cyan
    "QTM": "rebeccapurple",  # "#4D4D4D",  # dark gray
    "Direct": "#2CA02C",  # tab:green
    "V_d": "#9467BD",  # tab:purple
    "LMP": "#9467BD",
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


def set_jcp_style(*, figsize: tuple[float, float] | None = None):
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


# def savefig_jcp(fig: plt.Figure, path: str | Path, *, dpi: int = 600, formats: Sequence[str] = ("pdf",), bbox: str = "tight"):
#     """
#     Save a figure in vector formats suitable for JCP submission.

#     Examples
#     --------
#     savefig_jcp(fig, "figure1", formats=("pdf","eps"))
#     """
#     path = Path(path)
#     for ext in formats:
#         outfile = path.with_suffix("." + ext.lower())
#         fig.savefig(outfile, dpi=dpi, bbox_inches=bbox)

### With pdf raster mode ###


def savefig_jcp(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 600,
    formats: Sequence[str] = ("pdf",),
    bbox: str = "tight",
    rasterize_pdf: bool = True,
    raster_dpi: int | None = None,
    raster_pad_inches: float = 0.02,
):
    """
    Save a figure in formats suitable for JCP submission.

    NEW (optional):
    ---------------
    rasterize_pdf : bool
        If True and 'pdf' is requested, the PDF will contain a single raster image
        (rendered at `raster_dpi`), which makes heavy figures MUCH faster/smaller.
        Other formats are saved normally.

    raster_dpi : int | None
        DPI for rasterization into PDF when rasterize_pdf=True.
        Defaults to `dpi` if None.

    raster_pad_inches : float
        Padding used when rasterizing with bbox='tight'.
    """
    path = Path(path)

    # Normalize
    bbox_inches = "tight" if (bbox == "tight") else bbox
    do_tight = bbox == "tight"

    # ---------- helper: write fully-rasterized PDF ----------
    def _save_raster_pdf(outfile: Path):
        rdpi = int(dpi if raster_dpi is None else raster_dpi)

        # Render the original figure to an RGBA array
        img = _figure_to_rgba_image(
            fig,
            dpi=rdpi,
            tight=do_tight,
            pad_inches=raster_pad_inches if do_tight else 0.0,
            facecolor="white",
        )

        # Create a new figure with the SAME physical size as the original
        w_in, h_in = fig.get_size_inches()
        fig_r = plt.figure(figsize=(float(w_in), float(h_in)))
        ax_r = fig_r.add_axes([0, 0, 1, 1])  # full-bleed
        ax_r.imshow(img, aspect="auto")
        ax_r.set_axis_off()

        # Save as a single-image PDF (very light for viewers)
        fig_r.savefig(
            outfile,
            dpi=rdpi,
            bbox_inches="tight" if do_tight else None,
            pad_inches=raster_pad_inches if do_tight else 0.0,
            facecolor="white",
        )
        plt.close(fig_r)

    # ---------- main loop ----------
    for ext in formats:
        ext_l = ext.lower().lstrip(".")
        outfile = path.with_suffix("." + ext_l)

        if ext_l == "pdf" and rasterize_pdf:
            _save_raster_pdf(outfile)
        else:
            fig.savefig(outfile, dpi=dpi, bbox_inches=bbox_inches)


def _fmt_cbar_endpoints(vmin: float, vmax: float, *, decimals: int | None) -> list[str]:
    if decimals is None:
        # default behaviour (keep Matplotlib-like compact formatting)
        return [f"{vmin:g}", f"{vmax:g}"]
    return [f"{vmin:.{decimals}f}", f"{vmax:.{decimals}f}"]


def _auto_ylim_from_exp(
    yvals: np.ndarray, *, pad_top: float = 0.1, pad_bot: float = 0.1
):
    """
    Return (ymin, ymax) from provided y-values (already in transformed space),
    using finite values only. Name preserved for backward compatibility.
    """
    y = np.asarray(yvals, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return None
    lo = float(np.min(y[finite]))
    hi = float(np.max(y[finite]))
    span = max(1e-12, hi - lo)
    return lo - pad_bot * span, hi + pad_top * span


def _tau_transform(arr, mode: str = "log10") -> np.ndarray:
    """
    Transform tau -> y (or z):
      - mode="log10": log10(tau)
      - mode="ln":    ln(tau)
      - mode="tau":   raw tau

    Returns ndarray with NaNs where invalid (non-finite or tau<=0 for logs).
    """
    a = np.asarray(arr, dtype=float)
    m = (mode or "log10").lower()

    if m in ("log10", "lg"):
        out = np.full_like(a, np.nan, dtype=float)
        ok = np.isfinite(a) & (a > 0)
        out[ok] = np.log10(a[ok])
        return out

    if m in ("ln", "log", "log_e"):
        out = np.full_like(a, np.nan, dtype=float)
        ok = np.isfinite(a) & (a > 0)
        out[ok] = np.log(a[ok])
        return out

    if m in ("tau", "raw"):
        out = np.full_like(a, np.nan, dtype=float)
        ok = np.isfinite(a)
        out[ok] = a[ok]
        return out

    raise ValueError("mode must be 'log10', 'ln', or 'tau'")


def _tau_ylabel(mode: str = "log10") -> str:
    m = (mode or "log10").lower()
    if m in ("ln", "log", "log_e"):
        return r"$\ln\,\tau$ / $\mathrm{s}$"
    if m in ("tau", "raw"):
        return r"$\tau$ / $\mathrm{s}$"
    return r"$\log_{10}\,\tau$ / $\mathrm{s}$"


def _pretty_log_ticks(ax: plt.Axes, plain_until: float = 1e6):
    """
    Major ticks at powers of 10; labels as plain numbers up to `plain_until`,
    then switch to 10^n style. Minor ticks at 2..9 per decade (no labels).
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1), numticks=12)
    )

    def _fmt(val, pos):
        if val <= 0:
            return ""
        if val < plain_until:
            # format 0.1, 1, 10, 100, 1000, 10000 without trailing .0
            s = f"{val:g}"
            return s
        # scientific above threshold: 10^{n}
        n = int(np.round(np.log10(val)))
        return rf"$10^{{{n}}}$"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))


def _apply_panel_labels(
    axs,
    *,
    labels=("(a)", "(b)", "(c)", "(d)"),
    pos: str = "inside",  # "inside" | "below" | "above"
    inside_xy=(0.02, 0.02),
    below_y_offset: float = -0.12,
    above_y_offset: float = 0.06,  # distance above the top spine (axes coords)
    fontsize=8,
):
    """
    Place panel labels inside the axes, just below them, or just above them.
    """
    for lab, ax in zip(labels, np.ravel(axs)):
        if pos == "inside":
            ax.text(
                inside_xy[0],
                inside_xy[1],
                lab,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
            )
        elif pos == "below":
            ax.text(
                0.0,
                below_y_offset,
                lab,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=fontsize,
                fontweight="bold",
                clip_on=False,
            )
        elif pos == "above":
            ax.text(
                0.0,
                1.0 + above_y_offset,
                lab,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                clip_on=False,
            )


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
    chi_bis: np.ndarray  # cm^3 mol^-1
    # optional fitted model curves (often denser)
    nu_model: np.ndarray | None = None
    chi_prime_model: np.ndarray | None = None
    chi_bis_model: np.ndarray | None = None

    def max_chi_pp(self) -> float:
        return float(np.nanmax(self.chi_bis)) if self.chi_bis.size else np.nan


@dataclass(slots=True)
class TauGrid:
    T: np.ndarray
    H: np.ndarray
    tau_exp: np.ma.MaskedArray
    tau_total: np.ma.MaskedArray
    mechanisms: dict[str, np.ma.MaskedArray]

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

_TH_PATTERN = re.compile(r"T\s*=\s*([\d.]+).*?H\s*=\s*([\d.]+)", re.IGNORECASE)


def _parse_TH(label: str) -> tuple[float, float]:
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


def read_ac_csv(
    path: str | Path,
    *,
    chi_scale: float = 1.0,
) -> list[ACDataset]:
    """
    Read AC susceptibility CSV exported by ab_relacs

    Parameters
    ----------
    path : str | Path
        CSV file path (semicolon-separated).
    chi_scale : float, optional
        Multiplicative factor applied to ALL susceptibility values:
        chi_prime, chi_bis, and their model counterparts (if present).
        Frequencies are NOT scaled.

    Returns
    -------
    List[ACDataset]
    """
    df = pd.read_csv(path, sep=";", engine="python")
    df = df.drop(
        columns=[c for c in ("Name", "Value", "Error") if c in df.columns],
        errors="ignore",
    )

    # Basic validation / normalization
    try:
        chi_scale = float(chi_scale)
    except Exception as e:
        raise ValueError(f"chi_scale must be a number, got {chi_scale!r}") from e

    groups: dict[tuple[float, float], dict[str, pd.Series]] = {}
    for col in df.columns:
        base = col.split(" ")[0].lower()
        tag_map = _MODEL_MAP if base.startswith("model") else _EXPERIMENTAL_MAP
        for key, label in tag_map.items():
            if base.startswith(key):
                T, H = _parse_TH(col)
                groups.setdefault((T, H), {})[label] = df[col]
                break

    datasets: list[ACDataset] = []
    for (T, H), blk in groups.items():
        # required experimental columns
        if not {"nu", "chi_p", "chi_pp"} <= set(blk.keys()):
            # skip incomplete blocks rather than crashing
            continue

        nu = pd.to_numeric(blk["nu"], errors="coerce").to_numpy(float)
        chi_p = pd.to_numeric(blk["chi_p"], errors="coerce").to_numpy(float) * chi_scale
        chi_pp = (
            pd.to_numeric(blk["chi_pp"], errors="coerce").to_numpy(float) * chi_scale
        )

        # trim to common length
        L = min(len(nu), len(chi_p), len(chi_pp))
        nu, chi_p, chi_pp = nu[:L], chi_p[:L], chi_pp[:L]

        # optional model columns
        nu_m = chi_p_m = chi_pp_m = None
        if {"nu_model", "chi_p_model", "chi_pp_model"} <= set(blk.keys()):
            nu_m = pd.to_numeric(blk["nu_model"], errors="coerce").to_numpy(float)
            chi_p_m = (
                pd.to_numeric(blk["chi_p_model"], errors="coerce").to_numpy(float)
                * chi_scale
            )
            chi_pp_m = (
                pd.to_numeric(blk["chi_pp_model"], errors="coerce").to_numpy(float)
                * chi_scale
            )

            Lm = min(len(nu_m), len(chi_p_m), len(chi_pp_m))
            nu_m, chi_p_m, chi_pp_m = nu_m[:Lm], chi_p_m[:Lm], chi_pp_m[:Lm]

        datasets.append(
            ACDataset(
                T=T,
                H=H,
                nu=nu,
                chi_prime=chi_p,
                chi_bis=chi_pp,
                nu_model=nu_m,
                chi_prime_model=chi_p_m,
                chi_bis_model=chi_pp_m,
            )
        )

    datasets.sort(key=lambda d: (d.T, d.H))
    return datasets


_BASE_MECHS = ["Orbach", "Raman", "Raman_2", "QTM", "Direct", "V_d"]


def read_tau_csv(
    path: str | Path, *, mechanisms: Sequence[str] | None = None
) -> TauGrid:
    df = pd.read_csv(path, sep=";", engine="python")
    mech_names = [
        m
        for m in _BASE_MECHS
        if mechanisms is None or m.lower() in {x.lower() for x in mechanisms}
    ]

    if not {"T", "H", "tau"} <= set(df.columns):
        raise ValueError("Columns T, H, tau not found")

    exp_df = df[["T", "H", "tau"]].dropna().astype(float)

    def _collect_grid(suffix: str | None):
        cols = [f"Temp{suffix or ''}", f"Field{suffix or ''}"] + [
            f"{m}{suffix or ''}" for m in [*mech_names, "Tau"]
        ]
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

    T_vals = np.unique(
        np.concatenate(
            [
                exp_df["T"].values,
                ct_df["Temp"].values if not ct_df.empty else [],
                cf_df["Temp"].values if not cf_df.empty else [],
            ]
        )
    )
    H_vals = np.unique(
        np.concatenate(
            [
                exp_df["H"].values,
                ct_df["Field"].values if not ct_df.empty else [],
                cf_df["Field"].values if not cf_df.empty else [],
            ]
        )
    )

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


def _get_ax(ax: plt.Axes | None = None):
    return ax if ax is not None else plt.subplots()[1]


def _build_colors(
    n: int, *, cmap: str | ListedColormap = "cividis", reverse: bool = False
):
    base = mpl.colormaps.get_cmap(cmap)
    if reverse:
        base = base.reversed()
    return base(np.linspace(0, 1, n))


def _apply_axis_format(ax: plt.Axes, *, sci: bool = False, axis: str = "both"):
    """
    Format tick labels.
    - sci=False: show plain numbers (0.08 instead of 8 × 10^-2), no offset text.
    - sci=True : allow scientific notation when needed, but still no offset text.
    """
    # Choose which axes to format
    axes = []
    if axis in ("both", "x"):
        axes.append(("x", ax.xaxis))
    if axis in ("both", "y"):
        axes.append(("y", ax.yaxis))

    for which, axaxis in axes:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_useOffset(False)  # <- kills the ×10^n offset text
        fmt.set_scientific(bool(sci))  # <- plain numbers by default
        axaxis.set_major_formatter(fmt)

        # Also hide any leftover offset text explicitly
        axaxis.get_offset_text().set_visible(False)

    # Keep your spine linewidths
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


###############################################################################
# 5. Plotting functions
###############################################################################


def plot_chi_prime(
    datasets: Sequence[ACDataset],
    *,
    ax: plt.Axes | None = None,
    cmap: str | ListedColormap = "cividis",
    reverse_cmap: bool = False,
    normalize: bool = False,
    logx: bool = True,
    legend_style: str = "colorbar",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    pretty_logx: bool = True,
    cbar_decimals: int | None = None,
):
    """Plot χ′(ν) for a collection of datasets (varying T or H) — JCP style."""
    ax = _get_ax(ax)
    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse_cmap)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0

    # use distinct markers to ensure legibility in grayscale
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
    for i, (clr, ds) in enumerate(zip(colors, datasets)):
        ax.plot(
            ds.nu,
            ds.chi_prime / max_norm,
            ls="",
            marker=markers[i % len(markers)],
            ms=2.5,
            mec="none",
            color=clr,
        )
        if ds.nu_model is not None:
            ax.plot(ds.nu_model, ds.chi_prime_model / max_norm, color=clr, lw=1.0)

    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(r"$\nu$ / $\mathrm{Hz}$")
    ax.set_ylabel(
        r"$\chi'$"
        + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$")
    )
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
            t0, t1 = float(min(Ts)), float(max(Ts))
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels(_fmt_cbar_endpoints(t0, t1, decimals=cbar_decimals))
        else:
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=2)
            h0, h1 = float(min(Hs)), float(max(Hs))
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels(_fmt_cbar_endpoints(h0, h1, decimals=cbar_decimals))
    if logx:
        if pretty_logx:
            _pretty_log_ticks(ax)
        else:
            ax.set_xscale("log")
    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(left=x_min, right=x_max)
    if (y_min is not None) or (y_max is not None):
        ax.set_ylim(bottom=y_min, top=y_max)

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
    y_min = kwargs.get("y_min", None)
    y_max = kwargs.get("y_max", None)
    pretty_logx = kwargs.get("pretty_logx", True)
    cbar_decimals = kwargs.get("cbar_decimals", None)

    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    for i, (clr, ds) in enumerate(zip(colors, datasets)):
        ax.plot(
            ds.nu,
            ds.chi_bis / max_norm,
            ls="",
            marker=markers[i % len(markers)],
            ms=2.5,
            mec="none",
            color=clr,
        )
        if ds.nu_model is not None:
            ax.plot(ds.nu_model, ds.chi_bis_model / max_norm, color=clr, lw=1.0)

    if kwargs.get("logx", True):
        ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$ / $\mathrm{Hz}$")
    ax.set_ylabel(
        r"$\chi''$"
        + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$")
    )
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
            t0, t1 = float(min(Ts)), float(max(Ts))
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels(_fmt_cbar_endpoints(t0, t1, decimals=cbar_decimals))
        else:
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=2)
            h0, h1 = float(min(Hs)), float(max(Hs))
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels(_fmt_cbar_endpoints(h0, h1, decimals=cbar_decimals))
    if logx:
        if pretty_logx:
            _pretty_log_ticks(ax)
        else:
            ax.set_xscale("log")
    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(left=x_min, right=x_max)
    if (y_min is not None) or (y_max is not None):
        ax.set_ylim(bottom=y_min, top=y_max)

    return ax


def plot_cole_cole(
    datasets: Sequence[ACDataset],
    *,
    ax: plt.Axes | None = None,
    cmap: str = "cividis",
    reverse_cmap: bool = False,
    normalize: bool = False,
    legend_style: str = "colorbar",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    cbar_decimals: int | None = None,
):
    """Cole–Cole plot (χ′ vs χ″) — JCP style. No grid; clear markers."""
    ax = _get_ax(ax)
    colors = _build_colors(len(datasets), cmap=cmap, reverse=reverse_cmap)
    max_norm = max(d.max_chi_pp() for d in datasets) if normalize else 1.0
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    for i, (clr, ds) in enumerate(zip(colors, datasets)):
        ax.plot(
            ds.chi_prime / max_norm,
            ds.chi_bis / max_norm,
            ls="",
            marker=markers[i % len(markers)],
            ms=2.5,
            mec="none",
            color=clr,
        )
        if ds.chi_prime_model is not None:
            ax.plot(
                ds.chi_prime_model / max_norm,
                ds.chi_bis_model / max_norm,
                color=clr,
                lw=1.0,
            )

    ax.set_xlabel(
        r"$\chi'$"
        + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$")
    )
    ax.set_ylabel(
        r"$\chi''$"
        + (r" / $\mathrm{a.u.}$" if normalize else r" / $\mathrm{cm^{3}\ mol^{-1}}$")
    )
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
            t0, t1 = float(min(Ts)), float(max(Ts))
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels(_fmt_cbar_endpoints(t0, t1, decimals=cbar_decimals))
        else:
            cbar.set_label(r"$H$ / $\mathrm{Oe}$", labelpad=2)
            h0, h1 = float(min(Hs)), float(max(Hs))
            cbar.set_ticks([0, len(datasets) - 1])
            cbar.set_ticklabels(_fmt_cbar_endpoints(h0, h1, decimals=cbar_decimals))
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
    point_color: str | None = None,  # <- uniform points color
    component_color: str | None = None,  # <- uniform mechanism color(s)
    component_colors: Mapping[str, str] | None = None,  # <- per-mechanism override
    model_color: str = "grey",  # "#6A3D9A",
    exp_mode: str = "markers",  # "markers" | "line" | "line+markers"
    exp_line_color: str
    | None = None,  # line color (falls back to point_color, else 'k')
    exp_linestyle: str = "-",
    exp_linewidth: float = 1.0,
    # y-limit controls (log10 space)
    ymin: float | None = None,
    ymax: float | None = None,
    pad_top: float = 0.10,
    pad_bottom: float = 0.10,
    autolimit_on: str = "exp",  # "exp" | "all" | "none"
    # NEW: transparency + whether to add a legend at all
    point_alpha: float = 1.0,
    line_alpha: float = 0.95,
    comp_alpha: float = 0.95,
    add_legend: bool = True,
    exp_legend_loc: str = "upper left",
    model_legend_loc: str = "lower right",
    x_min: float | None = None,
    x_max: float | None = None,
    show_total: bool | None = None,  # None => backward-compat (uses `model`)
    total_label: str = "Total",
    total_color: str | None = None,  # None => uses model_color
    total_linestyle: str = "-",
    total_linewidth: float = 1.0,
    y_mode: str = "log10",
    component_linestyle: str = "-",
    component_linewidth: float = 0.9,
    exp_markersize: float = 4.0,
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

    good = ~τexp.mask
    x_good = x[good]
    τ_good = τexp.data[good]
    v_good = values[good]

    order = np.argsort(v_good)
    x_good, τ_good, v_good = x_good[order], τ_good[order], v_good[order]

    if point_color is None:
        colours = _build_colors(len(v_good), cmap=cmap, reverse=reverse_cmap)
    else:
        colours = [point_color] * len(v_good)
    y_exp = _tau_transform(τ_good, y_mode)

    # --- Experimental series rendering ---------------------------------
    exp_mode = (exp_mode or "markers").lower()
    if exp_mode not in {"markers", "line", "line+markers"}:
        raise ValueError("exp_mode must be 'markers', 'line', or 'line+markers'")

    # If user wants a line, use a uniform line color by default
    line_col = exp_line_color or point_color or "k"

    # --- NEW: allow "marker" to act as linestyle override in line modes ---
    # This is intentionally minimal: in exp_mode="line" / "line+markers",
    # if marker is one of the matplotlib linestyles, use it for the exp line.
    _line_styles = {"-", "--", "-.", ":"}
    exp_line_ls = exp_linestyle
    if (
        exp_mode in {"line", "line+markers"}
        and isinstance(marker, str)
        and marker in _line_styles
    ):
        exp_line_ls = marker

    # markers that are drawn as lines; they need markeredgecolor (mec)
    _LINE_MARKERS = {"x", "+", "1", "2", "3", "4", "|", "_"}

    # draw connecting line through experimental points (sorted by x for clean line)
    if exp_mode in {"line", "line+markers"}:
        sx = np.argsort(x_good)
        ax.plot(
            x_good[sx],
            y_exp[sx],
            linestyle=exp_line_ls,
            linewidth=exp_linewidth,
            color=line_col,
            alpha=point_alpha,
            zorder=4,
        )

    # draw markers
    if exp_mode in {"markers", "line+markers"}:
        # In line modes, if point_color is None (colormap), default markers to the line color
        if exp_mode != "markers" and point_color is None:
            colours_mark = [line_col] * len(v_good)
        else:
            colours_mark = colours

        for v, y, clr in zip(x_good, y_exp, colours_mark):
            if marker in _LINE_MARKERS:
                ax.plot(
                    v,
                    y,
                    ls="",
                    marker=marker,
                    ms=exp_markersize,
                    mfc="none",
                    mec=clr,
                    mew=0.9,
                    color=clr,
                    alpha=point_alpha,
                    zorder=4,
                )
            else:
                ax.plot(
                    v,
                    y,
                    ls="",
                    marker=marker,
                    ms=exp_markersize,
                    mec="none",
                    color=clr,
                    alpha=point_alpha,
                    zorder=4,
                )

    # model and components (with alpha)
    if show_total is None:
        show_total = bool(model)

    if show_total and tau_mod is not None:
        col = total_color if total_color is not None else model_color
        ax.plot(
            x,
            _tau_transform(np.asarray(tau_mod), y_mode),
            lw=total_linewidth,
            ls=total_linestyle,
            color=col,
            label=total_label,
            alpha=line_alpha,
            zorder=2.5,
        )

    if components:
        for name, arr in components.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr[:, 0] if T is not None else arr[0, :]

            if component_color is not None:
                c = component_color
            elif component_colors is not None and name in component_colors:
                c = component_colors[name]
            else:
                c = _MECHS_COLOR.get(name, None)

            ax.plot(
                x,
                _tau_transform(arr, y_mode),
                lw=component_linewidth,
                ls=component_linestyle,
                label=name,
                color=c,
                alpha=comp_alpha,
                zorder=3,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(_tau_ylabel(y_mode))
    _apply_axis_format(ax)

    # optional colorbar keyed to the sweep variable
    if legend_style == "colorbar" and len(v_good) > 1:
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(colours),
            norm=mpl.colors.Normalize(vmin=v_good.min(), vmax=v_good.max()),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, labelpad=2)
        cbar.set_ticks([v_good.min(), v_good.max()])

    # y-limits (log10/ln) — auto from selected series unless disabled
    if (ymin is None or ymax is None) and autolimit_on != "none":
        y_for_limits = []
        if autolimit_on in ("exp", "all"):
            y_for_limits.append(y_exp)
        if autolimit_on == "all":
            if model and tau_mod is not None:
                y_for_limits.append(_tau_transform(np.asarray(tau_mod), y_mode))
            if components:
                for arr in components.values():
                    if arr is None:
                        continue
                    a = np.asarray(arr)
                    if a.ndim == 2:
                        a = a[:, 0] if T is not None else a[0, :]
                    y_for_limits.append(_tau_transform(a, y_mode))
        if y_for_limits:
            ycat = np.concatenate([y for y in y_for_limits if np.size(y)])
            auto = _auto_ylim_from_exp(ycat, pad_top=pad_top, pad_bot=pad_bottom)
            if auto:
                ymin = auto[0] if ymin is None else ymin
                ymax = auto[1] if ymax is None else ymax

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # only add legend when requested (multi-plot path disables this)
    if add_legend:
        from matplotlib.lines import Line2D

        # 1) EXP legend proxy (marker/line style) — only meaningful for a single dataset plot
        # Pick a representative exp color
        exp_col = exp_line_color or point_color or "k"
        if exp_mode == "markers" and point_color is None and len(colours) > 0:
            exp_col = colours[0]

        exp_ls = exp_linestyle if exp_mode in ("line", "line+markers") else "None"

        # make sure line-markers are visible in proxy too
        _LINE_MARKERS = {"x", "+", "1", "2", "3", "4", "|", "_"}
        if marker in _LINE_MARKERS:
            exp_proxy = Line2D(
                [0],
                [0],
                linestyle=exp_ls,
                linewidth=exp_linewidth,
                marker=marker,
                markersize=4,
                markerfacecolor="none",
                markeredgecolor=exp_col,
                color=exp_col,
                label="exp",
            )
        else:
            exp_proxy = Line2D(
                [0],
                [0],
                linestyle=exp_ls,
                linewidth=exp_linewidth,
                marker=marker,
                markersize=4,
                markerfacecolor=exp_col,
                markeredgecolor=exp_col,
                color=exp_col,
                label="exp",
            )

        leg_exp = ax.legend(
            handles=[exp_proxy],
            loc=exp_legend_loc,
            frameon=False,
            fontsize=6.5,
            handlelength=1.6,
        )

        # 2) Components/total legend from labeled lines already drawn
        handles, labels = ax.get_legend_handles_labels()
        model_h, model_l = [], []
        for h, lab in zip(handles, labels):
            if not lab or lab.startswith("_") or lab == "exp":
                continue
            model_h.append(h)
            model_l.append(lab)

        if model_h:
            leg_model = ax.legend(
                model_h,
                model_l,
                loc=model_legend_loc,
                frameon=False,
                fontsize=6.5,
                handlelength=1.6,
            )
            ax.add_artist(leg_exp)

    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(left=x_min, right=x_max)

    return ax


def plot_tau_multi(
    entries: Sequence[tuple] | Sequence[Mapping],
    *,
    ax: plt.Axes | None = None,
    legend: bool = True,  # dataset legend (points)
    show_component_legend: bool = True,  # NEW: mechanisms/model legend (lines)
    exp_legend_loc: str = "upper left",
    model_legend_loc: str = "lower right",
    **tau_kwargs,
):
    """
    Plot multiple τ datasets on one axis, allowing each entry to choose its own
    field/temperature selector (or explicit T/H arrays). Builds TWO legends:
      - dataset legend (points) using per-entry labels
      - component/model legend (lines) aggregated across entries
    """
    from collections import OrderedDict
    from collections.abc import Mapping as _Mapping

    ax = _get_ax(ax)
    default_markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    def _slice_components_from_tg(
        tg: TauGrid, comps, *, field=None, temp=None, atol=1e-6
    ):
        """
        Slice component grids to 1D arrays matching the target sweep axis.
        Supports cross-grid components passed as (donor_tg, array) and interpolates
        onto the target axis (H for temp-sweep, T for field-sweep).
        """
        if not comps:
            return comps

        out = {}

        # ---------- helper: resolve donor ----------
        def _resolve_component(v):
            # v can be arr OR (donor_tg, arr)
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], TauGrid):
                return v[0], np.ma.asarray(v[1])
            return tg, np.ma.asarray(v)

        # ---------- temp sweep: x = H ----------
        if temp is not None:
            # target axis
            H_target = np.asarray(tg.H, float)

            # temp index on target
            ti = np.where(np.abs(tg.T - float(temp)) <= atol)[0]
            if ti.size == 0:
                raise ValueError(f"Temp slice {temp} not found in target TauGrid.T")
            # (we don't actually need ti unless someone passes same-grid 2D)

            for name, v in comps.items():
                if v is None:
                    continue
                donor, a = _resolve_component(v)

                # 2D grid -> slice donor at temp, then interpolate along H
                if a.ndim == 2:
                    ti_d = np.where(np.abs(donor.T - float(temp)) <= atol)[0]
                    if ti_d.size == 0:
                        raise ValueError(
                            f"Temp slice {temp} not found in donor grid for {name!r}"
                        )
                    row = np.asarray(a[int(ti_d[0]), :], float)
                    H_d = np.asarray(donor.H, float)

                    ok = np.isfinite(row) & np.isfinite(H_d)
                    if np.count_nonzero(ok) < 2:
                        out[name] = np.full_like(H_target, np.nan, dtype=float)
                    else:
                        # np.interp requires increasing x
                        order = np.argsort(H_d[ok])
                        out[name] = np.interp(
                            H_target,
                            H_d[ok][order],
                            row[ok][order],
                            left=np.nan,
                            right=np.nan,
                        )
                    continue

                # 1D array assumed already on target H grid
                a1 = np.asarray(a, float)
                if a1.size == H_target.size:
                    out[name] = a1
                else:
                    raise ValueError(
                        f"Mechanism {name!r} is 1D length {a1.size} but target H has {H_target.size}. "
                        "If it comes from a different grid, pass (donor_tg, arr2d) instead."
                    )

            return out

        # ---------- field sweep: x = 1/T ----------
        if field is not None:
            # target axis in T
            T_target = np.asarray(tg.T, float)

            # field index on target
            hi = np.where(np.abs(tg.H - float(field)) <= atol)[0]
            if hi.size == 0:
                raise ValueError(f"Field slice {field} not found in target TauGrid.H")

            for name, v in comps.items():
                if v is None:
                    continue
                donor, a = _resolve_component(v)

                # 2D grid -> slice donor at field, then interpolate along T
                if a.ndim == 2:
                    hi_d = np.where(np.abs(donor.H - float(field)) <= atol)[0]
                    if hi_d.size == 0:
                        raise ValueError(
                            f"Field slice {field} not found in donor grid for {name!r}"
                        )
                    col = np.asarray(a[:, int(hi_d[0])], float)
                    T_d = np.asarray(donor.T, float)

                    ok = np.isfinite(col) & np.isfinite(T_d)
                    if np.count_nonzero(ok) < 2:
                        out[name] = np.full_like(T_target, np.nan, dtype=float)
                    else:
                        order = np.argsort(T_d[ok])
                        out[name] = np.interp(
                            T_target,
                            T_d[ok][order],
                            col[ok][order],
                            left=np.nan,
                            right=np.nan,
                        )
                    continue

                # 1D assumed already on target T grid
                a1 = np.asarray(a, float)
                if a1.size == T_target.size:
                    out[name] = a1
                else:
                    raise ValueError(
                        f"Mechanism {name!r} is 1D length {a1.size} but target T has {T_target.size}. "
                        "If it comes from a different grid, pass (donor_tg, arr2d) instead."
                    )

            return out

        return comps

    # --------- FIRST PASS: compute global y-limits from ALL experimental points
    pad_top = tau_kwargs.get("pad_top", 0.10)
    pad_bottom = tau_kwargs.get("pad_bottom", 0.10)
    explicit_ymin = tau_kwargs.get("ymin", None)
    explicit_ymax = tau_kwargs.get("ymax", None)
    explicit_xmin = tau_kwargs.get("x_min", None)
    explicit_xmax = tau_kwargs.get("x_max", None)
    model = tau_kwargs.get("model", None)
    y_mode = tau_kwargs.get("y_mode", "log10")

    all_yexp = []

    def _collect_yexp_from_spec(spec) -> None:
        if "tg" in spec:
            tg = spec["tg"]
            sel = spec.get("selector") or {}
            if "field" in sel:
                _, tau_exp, _ = tg.slice_T(float(sel["field"]))
                y = _tau_transform(np.ma.asarray(tau_exp).compressed(), y_mode)
                all_yexp.append(y[np.isfinite(y)])
            elif "temp" in sel:
                _, tau_exp, _ = tg.slice_H(float(sel["temp"]))
                y = _tau_transform(np.ma.asarray(tau_exp).compressed(), y_mode)
                all_yexp.append(y[np.isfinite(y)])
        else:
            tau_exp = np.ma.asarray(spec["tau_exp"])
            vals = (
                tau_exp.compressed()
                if np.ma.isMaskedArray(tau_exp)
                else tau_exp[np.isfinite(tau_exp)]
            )
            y = _tau_transform(vals, y_mode)
            all_yexp.append(y[np.isfinite(y)])

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
                spec = dict(
                    tau_exp=tau_exp,
                    tau_mod=tau_mod,
                    comps=comps,
                    label=label,
                    axis_map=axis_map,
                )
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
        local_kwargs.setdefault("point_alpha", tau_kwargs.get("point_alpha", 0.95))
        local_kwargs.setdefault("line_alpha", tau_kwargs.get("line_alpha", 0.95))
        local_kwargs.setdefault("comp_alpha", tau_kwargs.get("comp_alpha", 0.95))

        if "tg" in spec:
            tg = spec["tg"]
            comps = spec.get("comps")
            label = spec.get("label")
            sel = spec.get("selector") or {}
            if "field" in sel:
                T_vals, tau_exp, tau_mod = tg.slice_T(float(sel["field"]))
                comps_s = _slice_components_from_tg(tg, comps, field=sel["field"])
                plot_tau(
                    tau_exp,
                    tau_mod,
                    T=T_vals,
                    components=comps_s,
                    ax=ax,
                    legend_style="none",
                    **style,
                    **local_kwargs,
                )
            elif "temp" in sel:
                H_vals, tau_exp, tau_mod = tg.slice_H(float(sel["temp"]))
                comps_s = _slice_components_from_tg(tg, comps, temp=sel["temp"])
                plot_tau(
                    tau_exp,
                    tau_mod,
                    H=H_vals,
                    components=comps_s,
                    ax=ax,
                    legend_style="none",
                    **style,
                    **local_kwargs,
                )
            else:
                raise ValueError(
                    "TauGrid entry needs selector {'field': H} or {'temp': T}."
                )
        else:
            tau_exp = spec["tau_exp"]
            tau_mod = spec.get("tau_mod")
            comps = spec.get("comps")
            label = spec.get("label")
            axis_map = spec.get("axis_map") or {}
            if "T" in axis_map:
                plot_tau(
                    tau_exp,
                    tau_mod,
                    T=np.asarray(axis_map["T"]),
                    components=comps,
                    ax=ax,
                    legend_style="none",
                    **style,
                    **local_kwargs,
                )
            elif "H" in axis_map:
                plot_tau(
                    tau_exp,
                    tau_mod,
                    H=np.asarray(axis_map["H"]),
                    components=comps,
                    ax=ax,
                    legend_style="none",
                    **style,
                    **local_kwargs,
                )
            else:
                raise ValueError(
                    "Explicit-array entry needs axis_map with {'T': ...} or {'H': ...}."
                )

        # newly added lines:
        new_lines = ax.lines[before:]

        # dataset handle: pick the first point-like line among new lines
        label = spec.get("label")
        if label:
            point_like = None
            for ln in new_lines:
                if (ln.get_linestyle() in ("", "None")) and (
                    ln.get_marker() not in (None, "None", "")
                ):
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
            if (ln.get_linestyle() not in ("", "None")) and (
                ln.get_marker() in (None, "None", "")
            ):
                if name not in component_handles:
                    component_handles[name] = ln

    # Apply global limits
    if explicit_ymin is not None and explicit_ymax is not None:
        ax.set_ylim(explicit_ymin, explicit_ymax)

    # Legends
    leg1 = None
    if legend and dataset_handles:
        leg1 = ax.legend(
            dataset_handles,
            dataset_labels,
            frameon=False,
            fontsize=6.5,
            handlelength=1.6,
            loc=exp_legend_loc,
        )

    if show_component_legend and component_handles:
        leg2 = ax.legend(
            list(component_handles.values()),
            list(component_handles.keys()),
            frameon=False,
            fontsize=6.5,
            handlelength=1.6,
            loc=model_legend_loc,
        )
        if leg1 is not None:
            ax.add_artist(leg1)

    return ax


def plot_composite_panel(
    ac_datasets: Sequence[ACDataset],
    *,
    tau_grid: tuple[Sequence[float], Sequence[float]] | TauGrid | None = None,
    # NEW: entries can now carry different selectors and mechanisms
    tau_grids_multi: Sequence[tuple] | Sequence[Mapping] | None = None,
    field_sel: float = None,
    temp_sel: float = None,
    mechanisms: Mapping[str, Sequence[float]] | None = None,
    cmap: str | ListedColormap = "cividis",
    reverse_cmap: bool = False,
    normalize_chi: bool = False,
    suptitle: str | None = None,
    grid: tuple[int, int] = (2, 2),
    cluster_tol_H: float = 0.5,
    cluster_tol_T: float = 0.05,
    model: bool = True,
    figure_size: tuple[float, float] | None = None,
    # label & τ-limit controls
    panel_label_pos: str = "above",
    tau_ylim: tuple[float, float] | None = None,
    tau_pad_top: float = 0.10,
    tau_pad_bottom: float = 0.10,
    tau_autolimit_on: str = "exp",
    xlim_im: tuple[float, float] | None = None,
    xlim_re: tuple[float, float] | None = None,
    ylim_im: tuple[float, float] | None = None,
    ylim_re: tuple[float, float] | None = None,
    xlim_cole: tuple[float, float] | None = None,
    ylim_cole: tuple[float, float] | None = None,
    xlim_tau: tuple[float, float] | None = None,
    pretty_logx: bool = True,
    vertical_1col: bool = False,
    vertical_height_ratio: float = 2.45,  # height = 3.37 * this
    vertical_figure_size: tuple[float, float] | None = None,
    vertical_label_pos: str | None = None,  # if None -> use panel_label_pos
    vertical_rect_top: float = 0.94,  # space reserved for suptitle/labels
    tau_y_mode: str = "log10",
    exp_legend_loc: str = "upper left",
    model_legend_loc: str = "lower right",
    tau_component_linewidth: float = 1.0,
    tau_component_linestyle: str = "-",
    cbar_decimals: int | None = None,
):
    """
    Create the standard 4-panel figure (χ″, χ′, Cole–Cole, log10 τ) — JCP style.
    """
    set_jcp_style(figsize=figure_size or double_column_size(height_ratio=0.78))
    fig, axs = plt.subplots(
        2, 2, figsize=figure_size or double_column_size(0.78), layout="constrained"
    )
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

    plot_chi_bis(
        datasets,
        ax=ax_im,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        normalize=normalize_chi,
        pretty_logx=pretty_logx,
        x_min=(xlim_im[0] if xlim_im else None),
        x_max=(xlim_im[1] if xlim_im else None),
        y_min=(ylim_im[0] if ylim_im else None),
        y_max=(ylim_im[1] if ylim_im else None),
        cbar_decimals=cbar_decimals,
    )
    plot_chi_prime(
        datasets,
        ax=ax_re,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        normalize=normalize_chi,
        pretty_logx=pretty_logx,
        x_min=(xlim_re[0] if xlim_re else None),
        x_max=(xlim_re[1] if xlim_re else None),
        y_min=(ylim_re[0] if ylim_re else None),
        y_max=(ylim_re[1] if ylim_re else None),
        cbar_decimals=cbar_decimals,
    )
    plot_cole_cole(
        datasets,
        ax=ax_cole,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        normalize=normalize_chi,
        x_min=(xlim_cole[0] if xlim_cole else None),
        x_max=(xlim_cole[1] if xlim_cole else None),
        y_min=(ylim_cole[0] if ylim_cole else None),
        y_max=(ylim_cole[1] if ylim_cole else None),
        cbar_decimals=cbar_decimals,
    )

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
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        y_mode=tau_y_mode,
        exp_legend_loc=exp_legend_loc,
        model_legend_loc=model_legend_loc,
        component_linewidth=tau_component_linewidth,
        component_linestyle=tau_component_linestyle,
    )

    # Single-grid path unchanged
    if isinstance(tau_grid, TauGrid):
        tg = tau_grid
        if field_sel is not None:
            T, tau_exp, tau_mod = tg.slice_T(field_sel)
            plot_tau(
                tau_exp,
                tau_mod,
                T=T,
                components=mechanisms,
                ax=ax_tau,
                cmap=cmap,
                reverse_cmap=reverse_cmap,
                model=model,
                **tau_common_kwargs,
            )
        elif temp_sel is not None:
            H, tau_exp, tau_mod = tg.slice_H(temp_sel)
            plot_tau(
                tau_exp,
                tau_mod,
                H=H,
                components=mechanisms,
                ax=ax_tau,
                cmap=cmap,
                reverse_cmap=reverse_cmap,
                model=model,
                **tau_common_kwargs,
            )

    # Multi-grid path now supports per-entry selectors and mechanisms
    elif tau_grids_multi:
        # Just forward the rich specs to plot_tau_multi; it will slice each tg
        plot_tau_multi(tau_grids_multi, ax=ax_tau, **tau_common_kwargs)
    else:
        ax_tau.set_visible(False)

    # Panel labels
    _apply_panel_labels(axs, pos=panel_label_pos)

    eng = fig.get_layout_engine()
    try:
        eng.set(rect=(0.0, 0.0, 1.0, 0.95))  # top=0.93; tweak 0.90–0.96
    except Exception:
        pass

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.0, y=0.995)

    # --- Single-panel exports (τ mirrors the choice above) ---
    single_size = single_column_size(height_ratio=0.82)

    fig_im, ax_im_s = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_chi_bis(
        datasets,
        ax=ax_im_s,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        normalize=normalize_chi,
        legend_style="colorbar",
        pretty_logx=pretty_logx,
        x_min=(xlim_im[0] if xlim_im else None),
        x_max=(xlim_im[1] if xlim_im else None),
        y_min=(ylim_im[0] if ylim_im else None),
        y_max=(ylim_im[1] if ylim_im else None),
        cbar_decimals=cbar_decimals,
    )

    fig_re, ax_re_s = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_chi_prime(
        datasets,
        ax=ax_re_s,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        normalize=normalize_chi,
        legend_style="colorbar",
        pretty_logx=pretty_logx,
        x_min=(xlim_re[0] if xlim_re else None),
        x_max=(xlim_re[1] if xlim_re else None),
        y_min=(ylim_re[0] if ylim_re else None),
        y_max=(ylim_re[1] if ylim_re else None),
        cbar_decimals=cbar_decimals,
    )

    fig_cc, ax_cc_s = plt.subplots(figsize=single_size, constrained_layout=True)
    plot_cole_cole(
        datasets,
        ax=ax_cc_s,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        normalize=normalize_chi,
        legend_style="colorbar",
        x_min=(xlim_cole[0] if xlim_cole else None),
        x_max=(xlim_cole[1] if xlim_cole else None),
        y_min=(ylim_cole[0] if ylim_cole else None),
        y_max=(ylim_cole[1] if ylim_cole else None),
        cbar_decimals=cbar_decimals,
    )

    fig_tau_s, ax_tau_s = plt.subplots(figsize=single_size, constrained_layout=True)
    if isinstance(tau_grid, TauGrid):
        if field_sel is not None:
            plot_tau(
                tau_exp,
                tau_mod,
                T=T,
                components=mechanisms,
                ax=ax_tau_s,
                legend_style="none",
                ymin=(tau_ylim[0] if tau_ylim else None),
                ymax=(tau_ylim[1] if tau_ylim else None),
                x_min=(xlim_tau[0] if xlim_tau else None),
                x_max=(xlim_tau[1] if xlim_tau else None),
                pad_top=tau_pad_top,
                pad_bottom=tau_pad_bottom,
                autolimit_on=tau_autolimit_on,
                model=model,
                cmap=cmap,
                reverse_cmap=reverse_cmap,
                y_mode=tau_y_mode,
                exp_legend_loc=exp_legend_loc,
                model_legend_loc=model_legend_loc,
                component_linewidth=tau_component_linewidth,
                component_linestyle=tau_component_linestyle,
            )
        else:
            plot_tau(
                tau_exp,
                tau_mod,
                H=H,
                components=mechanisms,
                ax=ax_tau_s,
                legend_style="none",
                ymin=(tau_ylim[0] if tau_ylim else None),
                ymax=(tau_ylim[1] if tau_ylim else None),
                x_min=(xlim_tau[0] if xlim_tau else None),
                x_max=(xlim_tau[1] if xlim_tau else None),
                pad_top=tau_pad_top,
                pad_bottom=tau_pad_bottom,
                autolimit_on=tau_autolimit_on,
                model=model,
                cmap=cmap,
                reverse_cmap=reverse_cmap,
                y_mode=tau_y_mode,
                exp_legend_loc=exp_legend_loc,
                model_legend_loc=model_legend_loc,
                component_linewidth=tau_component_linewidth,
                component_linestyle=tau_component_linestyle,
            )

    elif tau_grids_multi:
        plot_tau_multi(
            tau_grids_multi,
            ax=ax_tau_s,
            ymin=(tau_ylim[0] if tau_ylim else None),
            ymax=(tau_ylim[1] if tau_ylim else None),
            x_min=(xlim_tau[0] if xlim_tau else None),
            x_max=(xlim_tau[1] if xlim_tau else None),
            pad_top=tau_pad_top,
            pad_bottom=tau_pad_bottom,
            autolimit_on=tau_autolimit_on,
            model=model,
            cmap=cmap,
            reverse_cmap=reverse_cmap,
            y_mode=tau_y_mode,
            exp_legend_loc=exp_legend_loc,
            model_legend_loc=model_legend_loc,
            component_linewidth=tau_component_linewidth,
            component_linestyle=tau_component_linestyle,
        )

    # ------------------------------------------------------------------
    # OPTIONAL: vertical single-column 3-panel figure (χ″, χ′, τ)
    # ------------------------------------------------------------------
    fig_vert = None
    if vertical_1col:
        w1 = single_column_size(1.0)[0]  # 3.37 in
        if vertical_figure_size is None:
            vertical_figure_size = (w1, w1 * vertical_height_ratio)

        fig_vert, axs_v = plt.subplots(
            3,
            1,
            figsize=vertical_figure_size,
            layout="constrained",
            sharex=False,
        )
        axv_im, axv_re, axv_tau = axs_v

        # χ″
        plot_chi_bis(
            datasets,
            ax=axv_im,
            cmap=cmap,
            reverse_cmap=reverse_cmap,
            normalize=normalize_chi,
            pretty_logx=pretty_logx,
            legend_style="colorbar",
            x_min=(xlim_im[0] if xlim_im else None),
            x_max=(xlim_im[1] if xlim_im else None),
            y_min=(ylim_im[0] if ylim_im else None),
            y_max=(ylim_im[1] if ylim_im else None),
            cbar_decimals=cbar_decimals,
        )

        # χ′
        plot_chi_prime(
            datasets,
            ax=axv_re,
            cmap=cmap,
            reverse_cmap=reverse_cmap,
            normalize=normalize_chi,
            pretty_logx=pretty_logx,
            legend_style="colorbar",
            x_min=(xlim_re[0] if xlim_re else None),
            x_max=(xlim_re[1] if xlim_re else None),
            y_min=(ylim_re[0] if ylim_re else None),
            y_max=(ylim_re[1] if ylim_re else None),
            cbar_decimals=cbar_decimals,
        )

        # τ (reuse the same logic as the main composite τ panel)
        if isinstance(tau_grid, TauGrid):
            tg = tau_grid
            if field_sel is not None:
                T, tau_exp, tau_mod = tg.slice_T(field_sel)
                plot_tau(
                    tau_exp,
                    tau_mod,
                    T=T,
                    components=mechanisms,
                    ax=axv_tau,
                    legend_style="colorbar",
                    **tau_common_kwargs,
                )
            elif temp_sel is not None:
                H, tau_exp, tau_mod = tg.slice_H(temp_sel)
                plot_tau(
                    tau_exp,
                    tau_mod,
                    H=H,
                    components=mechanisms,
                    ax=axv_tau,
                    legend_style="colorbar",
                    **tau_common_kwargs,
                )
            else:
                axv_tau.set_visible(False)

        elif tau_grids_multi:
            plot_tau_multi(tau_grids_multi, ax=axv_tau, **tau_common_kwargs)
        else:
            axv_tau.set_visible(False)

        # Panel labels (a,b,c)
        lab_pos = (
            vertical_label_pos if vertical_label_pos is not None else panel_label_pos
        )
        _apply_panel_labels(axs_v, labels=("(a)", "(b)", "(c)"), pos=lab_pos)

        # Reserve a bit of top space (layout="constrained" compatible)
        eng_v = fig_vert.get_layout_engine()
        try:
            eng_v.set(rect=(0.0, 0.0, 1.0, float(vertical_rect_top)))
        except Exception:
            pass

        if suptitle:
            fig_vert.suptitle(suptitle, fontsize=9.5, y=0.995)

    fig_comb = fig
    if vertical_1col:
        return fig_im, fig_re, fig_cc, fig_tau_s, fig_comb, fig_vert
    return fig_im, fig_re, fig_cc, fig_tau_s, fig_comb


def plot_tau_grid_3d(
    tg: TauGrid,
    *,
    ax: plt.Axes | None = None,
    # what to show
    show_exp: bool = True,
    show_total: bool = True,
    show_mechanisms: bool = True,
    mechanisms: Mapping[str, np.ndarray]
    | None = None,  # overrides tg.mechanisms if given
    which_mechanisms: Sequence[str] | None = None,  # optional filter
    # z-transform
    z_mode: str = "ln",  # "ln" | "log10" | "tau"
    # appearance
    surface_kind: str = "wireframe",  # "wireframe" | "surface"
    surface_alpha: float = 0.75,
    total_color: str = "dimgrey",  # "#6A3D9A",
    mechanism_colors: Mapping[str, str] | None = None,
    exp_color: str = "tab:blue",
    exp_marker: str = "o",
    exp_size: float = 22.0,
    exp_edgecolor: str | None = "k",
    exp_on_top: bool = True,
    # sampling for surfaces only
    stride_T: int = 1,
    stride_H: int = 1,
    # view
    elev: float = 25.0,
    azim: float = -60.0,
    # limits (in transformed space for z)
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    # ticks / grids
    tick_nbins: int = 4,
    show_pane_grid: bool = False,
    # legend
    legend: bool = True,
    legend_loc: str = "upper right",
    legend_fontsize: float = 7.0,
    exp_title: str = None,
):
    """
    3D plot of tau grid: z = ln(tau) or log10(tau), x = 1/T, y = H.

    - Experimental points: ALL points from tg.tau_exp (no striding)
    - Total + mechanisms: surfaces/wireframes (optionally strided for readability)
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, NullLocator
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # -------- transform ----------
    def _z_transform(arr2d: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
        a = np.ma.asarray(arr2d)
        m = (z_mode or "ln").lower()

        if m in ("ln", "log", "log_e"):
            a = np.ma.masked_where(~np.isfinite(a) | (a <= 0), a)
            z = np.ma.log(a)
        elif m in ("log10", "lg"):
            a = np.ma.masked_where(~np.isfinite(a) | (a <= 0), a)
            z = np.ma.log10(a)
        elif m in ("tau", "raw"):
            z = np.ma.masked_where(~np.isfinite(a), a)
        else:
            raise ValueError("z_mode must be 'ln', 'log10', or 'tau'")

        return z.filled(np.nan)

    # -------- surfaces mesh (strided) ----------
    stride_T = max(1, int(stride_T))
    stride_H = max(1, int(stride_H))
    T_s = np.asarray(tg.T)[::stride_T]
    H_s = np.asarray(tg.H)[::stride_H]
    Xs, Ys = np.meshgrid(1.0 / T_s, H_s, indexing="ij")

    def _clip_Z_to_limits(X, Y, Z):
        """Return Z with values outside xlim/ylim/zlim set to NaN (so 3D plots are clipped)."""
        Zc = np.array(Z, copy=True)

        m = np.zeros(Zc.shape, dtype=bool)
        if xlim is not None:
            m |= (X < xlim[0]) | (X > xlim[1])
        if ylim is not None:
            m |= (Y < ylim[0]) | (Y > ylim[1])
        if zlim is not None:
            m |= (Zc < zlim[0]) | (Zc > zlim[1])

        Zc[m] = np.nan
        return Zc

    def _draw_surface(Zs: np.ndarray, *, color: str):
        if surface_kind == "surface":
            ax.plot_surface(
                Xs,
                Ys,
                Zs,
                color=color,
                alpha=surface_alpha,
                linewidth=0.0,
                antialiased=True,
                shade=False,
            )
        elif surface_kind == "wireframe":
            ax.plot_wireframe(
                Xs,
                Ys,
                Zs,
                rstride=1,
                cstride=1,
                color=color,
                alpha=surface_alpha,
                linewidth=0.8,
            )
        else:
            raise ValueError("surface_kind must be 'wireframe' or 'surface'")

    proxy = []  # legend proxies

    # -------- total ----------
    if show_total and tg.tau_total is not None:
        Ztot_full = _z_transform(tg.tau_total)
        Ztot = Ztot_full[::stride_T, ::stride_H]
        Ztot = _clip_Z_to_limits(Xs, Ys, Ztot)
        _draw_surface(Ztot, color=total_color)
        proxy.append(
            Patch(
                facecolor=total_color,
                edgecolor="none",
                alpha=surface_alpha,
                label="Total",
            )
        )

    # -------- mechanisms ----------
    if show_mechanisms:
        mech_src = mechanisms if mechanisms is not None else (tg.mechanisms or {})
        if which_mechanisms is not None:
            allowed = {m.lower() for m in which_mechanisms}
            mech_src = {k: v for k, v in mech_src.items() if k.lower() in allowed}

        for name, grid in mech_src.items():
            if grid is None:
                continue
            col = (
                mechanism_colors[name]
                if (mechanism_colors and name in mechanism_colors)
                else _MECHS_COLOR.get(name, "tab:gray")
            )
            Zm_full = _z_transform(grid)
            Zm = Zm_full[::stride_T, ::stride_H]
            Zm = _clip_Z_to_limits(Xs, Ys, Zm)
            _draw_surface(Zm, color=col)
            proxy.append(
                Patch(
                    facecolor=col,
                    edgecolor="none",
                    alpha=surface_alpha,
                    label=str(name),
                )
            )

    # -------- experimental scatter (ALL points) ----------
    if show_exp and tg.tau_exp is not None:
        T = np.asarray(tg.T)
        H = np.asarray(tg.H)
        Xe, Ye = np.meshgrid(1.0 / T, H, indexing="ij")

        exp = np.ma.asarray(tg.tau_exp)
        Zexp = _z_transform(exp)

        ii, jj = np.where(np.isfinite(Zexp))
        if ii.size:
            xpts = Xe[ii, jj]
            ypts = Ye[ii, jj]
            zpts = Zexp[ii, jj]

            keep = np.isfinite(zpts)
            if xlim is not None:
                keep &= (xpts >= xlim[0]) & (xpts <= xlim[1])
            if ylim is not None:
                keep &= (ypts >= ylim[0]) & (ypts <= ylim[1])
            if zlim is not None:
                keep &= (zpts >= zlim[0]) & (zpts <= zlim[1])

            xpts, ypts, zpts = xpts[keep], ypts[keep], zpts[keep]

            ec = exp_edgecolor if exp_edgecolor is not None else "none"

            ax.scatter(
                xpts,
                ypts,
                zpts,
                s=exp_size,
                c=exp_color,
                marker=exp_marker,
                edgecolors=ec,
                linewidths=0.6 if ec != "none" else 0.0,
                depthshade=not exp_on_top,
            )
            proxy.append(
                Line2D(
                    [0],
                    [0],
                    marker=exp_marker,
                    color="none",
                    markerfacecolor=exp_color,
                    markeredgecolor=(ec if ec != "none" else exp_color),
                    markersize=max(4.0, np.sqrt(exp_size)),
                    label="Exp" if exp_title is None else exp_title,
                )
            )

    # -------- labels ----------
    ax.set_xlabel(r"$1/T$ / $\mathrm{K^{-1}}$")
    ax.set_ylabel(r"$H$ / $\mathrm{Oe}$")
    m = (z_mode or "ln").lower()
    if m in ("ln", "log", "log_e"):
        ax.set_zlabel(r"$\ln\,\tau$ / $\mathrm{s}$")
    elif m in ("log10", "lg"):
        ax.set_zlabel(r"$\log_{10}\,\tau$ / $\mathrm{s}$")
    else:
        ax.set_zlabel(r"$\tau$ / $\mathrm{s}$")

    # -------- view + limits ----------
    ax.view_init(elev=elev, azim=azim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    # -------- ticks / wall grid ----------
    ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))

    # Make box faces end exactly on the major grid/ticks (removes the “weird last cell”)
    _snap_3d_limits_to_major_ticks(ax, mode="enclose")  # or mode="inside"

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())

    # -------- pane (walls) styling: keep grid, remove grey walls ----------
    # Use alpha=0.0 for fully transparent, alpha=1.0 for solid white
    pane_alpha = 0.1  # <- set to 1.0 if you prefer white (not transparent)
    pane_rgba = (0.95, 0.94, 0.94, pane_alpha)

    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        # modern API
        try:
            a.pane.set_facecolor(pane_rgba)
            a.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))  # hide pane borders (optional)
            a.pane.fill = True
        except Exception:
            # fallback for older Matplotlib
            a._axinfo["pane"]["color"] = pane_rgba
            a._axinfo["pane"]["edgecolor"] = (1.0, 1.0, 1.0, 0.0)

    if show_pane_grid:
        grid_rgba = (0.0, 0.0, 0.0, 1.0)  # black
        grid_lw = 0.6  # adjust as you like
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = grid_rgba
            axis._axinfo["grid"]["linewidth"] = grid_lw
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.grid(False)

    # -------- legend ----------
    if legend and proxy:
        seen = set()
        uniq = []
        for h in proxy:
            lab = h.get_label()
            if lab not in seen:
                uniq.append(h)
                seen.add(lab)
        ax.legend(handles=uniq, loc=legend_loc, fontsize=legend_fontsize, frameon=False)

    return ax


def _snap_3d_limits_to_major_ticks(ax, *, mode: str = "enclose"):
    """
    Snap x/y/z limits to major tick positions to make pane grid end exactly on the box.

    mode:
      - "enclose": expand limits to outermost major ticks that cover the current range
      - "inside" : shrink limits to innermost major ticks inside the current range
    """

    def _snap_one(getlim, setlim, getloc, setticks):
        v0, v1 = getlim()
        lo, hi = (v0, v1) if v0 <= v1 else (v1, v0)

        loc = getloc()
        try:
            ticks = np.asarray(loc.tick_values(lo, hi), dtype=float)
        except Exception:
            ticks = np.asarray(setticks(), dtype=float)

        ticks = ticks[np.isfinite(ticks)]
        if ticks.size < 2:
            return

        ticks = np.unique(ticks)

        if mode == "inside":
            t = ticks[(ticks >= lo) & (ticks <= hi)]
            if t.size >= 2:
                lo2, hi2 = float(t[0]), float(t[-1])
                setlim(lo2, hi2)
                setticks(t)
        else:  # "enclose"
            lo2, hi2 = float(ticks[0]), float(ticks[-1])
            setlim(lo2, hi2)
            setticks(ticks)

    _snap_one(ax.get_xlim3d, ax.set_xlim3d, ax.xaxis.get_major_locator, ax.set_xticks)
    _snap_one(ax.get_ylim3d, ax.set_ylim3d, ax.yaxis.get_major_locator, ax.set_yticks)
    _snap_one(ax.get_zlim3d, ax.set_zlim3d, ax.zaxis.get_major_locator, ax.set_zticks)


def _figure_to_rgba_image(
    fig: plt.Figure,
    *,
    dpi: int = 600,
    tight: bool = True,
    pad_inches: float = 0.02,
    facecolor: str = "white",
) -> np.ndarray:
    """
    Rasterize a Matplotlib Figure to an RGBA image array.

    Notes
    -----
    - Uses savefig -> PNG in-memory -> mpimg.imread.
    - If `tight=True`, uses bbox_inches="tight" and pad_inches to crop whitespace.
    """
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight" if tight else None,
        pad_inches=pad_inches if tight else 0.0,
        facecolor=facecolor,
    )
    buf.seek(0)
    img = mpimg.imread(buf)  # RGBA float in [0,1] typically
    buf.close()
    return img


def panel_from_figures(
    figures: Sequence[plt.Figure],
    *,
    labels: Sequence[str] | None = None,
    titles: Sequence[str | None] | None = None,
    dpi_raster: int = 600,
    tight: bool = True,
    pad_inches: float = 0.02,
    onecol_width_in: float = 3.37,
    twocol_width_in: float = 7.00,
    gap_in: float = 0.10,
    label_pos: tuple[float, float] = (0.02, 0.98),
    label_fontsize: float = 8.0,
    label_mode: str = "title",  # "title" | "text"
    label_pad_pt: float = 2.0,
    label_bbox: bool = True,
    equalize_cell_aspect: bool = True,
    # --- titles tuning (requested) ---
    title_fontsize: float = 9.0,
    title_pad_pt: float = 4.0,
    # --- headroom so titles never overlap raster content ---
    top_inset_frac: float = 0.07,
    title_y: float = 0.99,
):
    """
    Build a panel figure from 1–4 inputs:
      - n=1..3  -> vertical 1-column panel
      - n=4     -> 2x2 2-column panel

    Inputs may be:
      - Matplotlib Figure objects
      - Paths/strings to image files (png/jpg/tif/...) or PDFs (if PyMuPDF is available)

    Labels are assigned in the ORDER of `figures`.
    """
    from pathlib import Path

    figs = list(figures)
    n = len(figs)
    if n < 1 or n > 4:
        raise ValueError("panel_from_figures supports only 1–4 figures.")
    if label_mode not in ("title", "text"):
        raise ValueError("label_mode must be 'title' or 'text'.")

    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels must have the same length as figures.")

    if titles is None:
        titles = [None] * n
    if len(titles) != n:
        raise ValueError("titles must have the same length as figures.")
    titles = list(titles)

    # --------- helpers: load inputs into RGBA arrays ---------

    def _as_rgba(img: np.ndarray) -> np.ndarray:
        """Ensure RGBA float array in [0,1]."""
        a = np.asarray(img)
        if a.ndim == 2:  # grayscale
            a = np.stack([a, a, a], axis=-1)
        if a.shape[-1] == 3:
            alpha = np.ones((*a.shape[:-1], 1), dtype=a.dtype)
            a = np.concatenate([a, alpha], axis=-1)
        # mpimg.imread gives float [0,1] for PNG; for some formats it can be uint8.
        if a.dtype.kind in ("u", "i"):
            a = a.astype(np.float32) / 255.0
        return a

    def _read_image_file(path: Path) -> np.ndarray:
        # common raster images
        ext = path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"):
            return _as_rgba(mpimg.imread(str(path)))

        if ext == ".pdf":
            # Try PyMuPDF (fitz) if available
            try:
                import fitz  # type: ignore
            except Exception as e:
                raise ImportError(
                    "PDF input requires PyMuPDF. Install with: pip install pymupdf"
                ) from e

            doc = fitz.open(str(path))
            if doc.page_count < 1:
                raise ValueError(f"PDF has no pages: {path}")
            page = doc.load_page(0)

            # Render at requested dpi (72 dpi is default user space)
            zoom = float(dpi_raster) / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=True)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            doc.close()

            img = _as_rgba(img)
            img = _crop_border_px(img, px=3)

            return img

        raise ValueError(
            f"Unsupported file type for panel input: {path.suffix!r} ({path})"
        )

    def _input_to_rgba(x) -> np.ndarray:
        # Matplotlib figure
        if isinstance(x, mpl.figure.Figure):
            return _figure_to_rgba_image(
                x, dpi=dpi_raster, tight=tight, pad_inches=pad_inches, facecolor="white"
            )

        # file path
        if isinstance(x, (str, Path)):
            p = Path(x)
            if not p.exists():
                raise FileNotFoundError(f"panel_from_figures: file not found: {p}")
            return _read_image_file(p)

        raise TypeError(
            "panel_from_figures expects each item to be a Matplotlib Figure or a path (str/Path) "
            f"to an image/PDF, got: {type(x)}"
        )

    imgs = [_input_to_rgba(x) for x in figs]
    ratios = [img.shape[0] / img.shape[1] for img in imgs]  # h/w

    # --------- label/title placement (unchanged logic) ---------

    def _place_label(ax: plt.Axes, lab: str):
        if label_mode == "title":
            ax.set_title(
                lab,
                loc="left",
                pad=label_pad_pt,
                fontsize=label_fontsize,
                fontweight="bold",
            )
        else:
            bbox = None
            if label_bbox:
                bbox = dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.20)
            ax.text(
                label_pos[0],
                label_pos[1],
                lab,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_fontsize,
                fontweight="bold",
                bbox=bbox,
                clip_on=False,
            )

    def _place_user_title(ax: plt.Axes, title: str | None):
        if title:
            ax.text(
                0.5,
                title_y,
                title,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=title_fontsize,
                fontweight="normal",
                clip_on=False,
            )

    def _draw_cell(ax: plt.Axes, img: np.ndarray, lab: str, title: str | None):
        ax.imshow(img, aspect="auto")

        # Reserve top headroom inside the axes so raster content is pushed down.
        if top_inset_frac and top_inset_frac > 0:
            ax.set_ylim(img.shape[0], -img.shape[0] * float(top_inset_frac))

        ax.set_axis_off()
        _place_label(ax, lab)
        _place_user_title(ax, title)

    # --------- layout (unchanged) ---------

    if n <= 3:
        W = float(onecol_width_in)

        if equalize_cell_aspect:
            r = float(np.max(ratios))
            cell_h = W * r
            H = float(n * cell_h + gap_in * (n - 1))
            height_ratios = [1.0] * n
        else:
            panel_heights = [W * r for r in ratios]
            H = float(sum(panel_heights) + gap_in * (n - 1))
            height_ratios = panel_heights

        panel_fig = plt.figure(figsize=(W, H))
        gs = panel_fig.add_gridspec(
            nrows=n,
            ncols=1,
            height_ratios=height_ratios,
            hspace=gap_in / max(1e-12, W),
        )

        axes: list[plt.Axes] = []
        for i in range(n):
            ax = panel_fig.add_subplot(gs[i, 0])
            _draw_cell(ax, imgs[i], labels[i], titles[i])
            axes.append(ax)

        panel_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return panel_fig, axes

    else:
        # n == 4
        W = float(twocol_width_in)
        cell_w = W / 2.0

        if equalize_cell_aspect:
            r = float(np.max(ratios))
            row1_h = cell_w * r
            row2_h = cell_w * r
            height_ratios = [1.0, 1.0]
        else:
            row1_h = cell_w * max(ratios[0], ratios[1])
            row2_h = cell_w * max(ratios[2], ratios[3])
            height_ratios = [row1_h, row2_h]

        H = float(row1_h + row2_h + gap_in)

        panel_fig = plt.figure(figsize=(W, H))
        gs = panel_fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=height_ratios,
            wspace=gap_in / max(1e-12, cell_w),
            hspace=gap_in / max(1e-12, min(row1_h, row2_h)),
        )

        axes: list[plt.Axes] = []
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (r0, c0) in enumerate(positions):
            ax = panel_fig.add_subplot(gs[r0, c0])
            _draw_cell(ax, imgs[i], labels[i], titles[i])
            axes.append(ax)

        panel_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return panel_fig, axes


def _crop_border_px(img: np.ndarray, *, px: int = 3) -> np.ndarray:
    """
    Crop a uniform border (in pixels) from all sides.
    Helps remove faint PDF page-edge boxes after rasterization.
    """
    if px <= 0:
        return img
    h, w = img.shape[:2]
    if 2 * px >= h or 2 * px >= w:
        return img
    return img[px : h - px, px : w - px, ...]
