#!/usr/bin/env python3

# SlothPy
# Copyright (C) 2026 Mikolaj Tadeusz Zychowicz (MTZ)

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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter

# ---------------------------------------------------------------------------
# Minimal JCP-like style (safe defaults, no external deps)
# ---------------------------------------------------------------------------


def single_column_size(height_ratio: float = 0.82) -> Tuple[float, float]:
    """Return (width, height) for a one-column figure. Width ≈ 3.37 in."""
    w = 3.37
    return (w, w * height_ratio)


def set_jcp_style(*, figsize: Tuple[float, float] | None = None) -> None:
    """
    Apply a Matplotlib style that matches common JCP/AIP expectations:
    - sans serif fonts, ~7–8 pt
    - inward ticks, four spines
    - no grid
    """
    rc = {
        "savefig.dpi": 600,
        "savefig.transparent": False,
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "mathtext.fontset": "cm",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
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
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "axes.grid": False,
    }
    if figsize is not None:
        rc["figure.figsize"] = figsize
    else:
        rc["figure.figsize"] = single_column_size()
    mpl.rcParams.update(rc)


def savefig_pub(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 600,
    formats: Sequence[str] = ("pdf",),
    bbox: str = "tight",
) -> None:
    """Save a figure in vector formats suitable for publication."""
    path = Path(path)
    for ext in formats:
        fig.savefig(path.with_suffix("." + ext.lower()), dpi=dpi, bbox_inches=bbox)


def _apply_axis_format(ax: plt.Axes, *, sci: bool = False, axis: str = "both") -> None:
    """Plain tick labels (no offset text) with optional scientific notation."""
    # --- NEW: don't override log-axis formatter (it breaks 0.1/0.01 labels)
    if ax.get_xscale() == "log" and axis == "both":
        axis = "y"
    elif ax.get_xscale() == "log" and axis == "x":
        return

    axes = []
    if axis in ("both", "x"):
        axes.append(ax.xaxis)
    if axis in ("both", "y"):
        axes.append(ax.yaxis)

    for axaxis in axes:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_useOffset(False)
        fmt.set_scientific(bool(sci))
        axaxis.set_major_formatter(fmt)
        axaxis.get_offset_text().set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _pretty_log_ticks(ax: plt.Axes, *, plain_until: float = 1e5) -> None:
    """Major ticks at 10^n; labels as plain numbers up to plain_until, else 10^n."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1), numticks=12)
    )

    def _fmt(val, pos):
        if val <= 0:
            return ""
        if val < plain_until:
            return f"{val:g}"
        n = int(np.round(np.log10(val)))
        return r"$10^{{{}}}$".format(n)

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))


# ---------------------------------------------------------------------------
# Reading + convergence logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BZConvergenceSummary:
    """Per-FWHM converged outcome."""

    fwhm: float
    n_points_conv: int
    log10_nu_peak_conv: float
    log10_tau_conv: float


def read_bz_convergence_csv(
    paths: str | Path | Sequence[str | Path],
    *,
    sep: str | None = None,
    fwhm_round: int | None = 6,
) -> pd.DataFrame:
    """
    Read one or more convergence summary CSVs and glue them together.

    Required columns (case-insensitive):
      - fwhm
      - n_points   (or npoints)
      - log10_peak (preferred) OR peak_freq_hz

    Optional:
      - thr_log10  (threshold used by your script; per-row or per-file)

    Notes
    -----
    - If the same (fwhm, n_points) appears multiple times, the *last* one wins.
    - Sorting is by (fwhm, n_points).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = []
    for p in paths:
        p = Path(p)
        if sep is None:
            # try common separators automatically
            try:
                df = pd.read_csv(p, sep=";")
                if df.shape[1] == 1:
                    df = pd.read_csv(p, sep=",")
            except Exception:
                df = pd.read_csv(p)
        else:
            df = pd.read_csv(p, sep=sep)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Normalize colnames (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # Lowercase map
    low = {c.lower(): c for c in df.columns}

    def _rename_if_present(target: str, *cands: str):
        for cand in cands:
            if cand.lower() in low and target not in df.columns:
                df.rename(columns={low[cand.lower()]: target}, inplace=True)
                return

    _rename_if_present("fwhm", "fwhm", "FWHM")
    _rename_if_present("n_points", "n_points", "npoints", "Npoints", "nPoints")
    _rename_if_present(
        "log10_peak", "log10_peak", "log10_nu_peak", "log10_nu", "log10_peak_nu"
    )
    _rename_if_present(
        "peak_freq_hz",
        "peak_freq_hz",
        "nu_peak_hz",
        "nu_peak",
        "peak_hz",
        "peak_frequency_hz",
    )
    _rename_if_present("thr_log10", "thr_log10", "threshold_log10", "thr", "thrlog10")

    if "fwhm" not in df.columns or "n_points" not in df.columns:
        raise ValueError("CSV must contain columns fwhm and n_points (or npoints).")

    if "log10_peak" not in df.columns:
        if "peak_freq_hz" in df.columns:
            df["log10_peak"] = np.log10(
                pd.to_numeric(df["peak_freq_hz"], errors="coerce")
            )
        else:
            raise ValueError("CSV must contain log10_peak or peak_freq_hz.")

    # Make numeric
    df["fwhm"] = pd.to_numeric(df["fwhm"], errors="coerce")
    df["n_points"] = pd.to_numeric(df["n_points"], errors="coerce").astype("Int64")
    df["log10_peak"] = pd.to_numeric(df["log10_peak"], errors="coerce")
    if "thr_log10" in df.columns:
        df["thr_log10"] = pd.to_numeric(df["thr_log10"], errors="coerce")

    df = df.dropna(subset=["fwhm", "n_points"]).copy()

    # Canonicalize FWHM to avoid tiny float mismatches when gluing files
    if fwhm_round is not None:
        df["fwhm"] = df["fwhm"].round(int(fwhm_round))
    df["n_points"] = df["n_points"].astype(int)

    # Deduplicate by (fwhm, n_points): last wins
    df = df.drop_duplicates(subset=["fwhm", "n_points"], keep="last")
    df = df.sort_values(["fwhm", "n_points"], kind="mergesort").reset_index(drop=True)
    return df


def summarize_bz_convergence(
    df: pd.DataFrame,
    *,
    tail_k: int = 5,
    thr_log10: float | None = None,
) -> list[BZConvergenceSummary]:
    """
    Determine converged log10_peak per FWHM, then convert to log10(tau).

    Convergence rule (matches the "stabilization" idea used in your scripts):
      - sort by n_points
      - require `tail_k` consecutive *points* (finite log10_peak) such that each
        new point differs from the previous one by at most `thr` in log10 space:
            |y[i] - y[i-1]| <= thr
      - convergence is declared at the n_points of the LAST point of the first
        streak that reaches length `tail_k`
      - converged log10_peak is the mean of the last `tail_k` values in that streak

    Threshold:
      - `thr_log10` if provided
      - else: last non-NaN df['thr_log10'] within that fwhm (if present)
      - else: 0.02

    Returns a list of BZConvergenceSummary.
    """
    needed = {"fwhm", "n_points", "log10_peak"}
    if not needed.issubset(df.columns):
        raise ValueError(f"df must contain {sorted(needed)}")

    # Require (tail_k+1) consecutive *successful steps* -> tail_k+2 stable points
    k_points = max(2, int(tail_k) + 2)
    out: list[BZConvergenceSummary] = []

    for fwhm, g in df.groupby("fwhm", sort=True):
        g = g.sort_values("n_points")
        y_all = g["log10_peak"].to_numpy(float)
        n_all = g["n_points"].to_numpy(int)

        # threshold
        if thr_log10 is not None:
            thr = float(thr_log10)
        elif "thr_log10" in g.columns and g["thr_log10"].notna().any():
            thr = float(g["thr_log10"].dropna().iloc[-1])
        else:
            thr = 0.02

        # walk through finite points only, preserving order
        finite_idx = np.where(np.isfinite(y_all))[0]
        if finite_idx.size == 0:
            ref = float("nan")
            n_conv = int(n_all[-1]) if n_all.size else 0
            out.append(
                BZConvergenceSummary(float(fwhm), int(n_conv), float(ref), float("nan"))
            )
            continue

        streak_vals: list[float] = []
        streak_n: list[int] = []

        best_ref = float(y_all[finite_idx[-1]])
        best_n = int(n_all[finite_idx[-1]])

        prev_y = None
        for idx in finite_idx:
            yy = float(y_all[idx])
            nn = int(n_all[idx])

            if prev_y is None:
                streak_vals = [yy]
                streak_n = [nn]
                prev_y = yy
                continue

            if abs(yy - prev_y) <= thr:
                streak_vals.append(yy)
                streak_n.append(nn)
            else:
                streak_vals = [yy]
                streak_n = [nn]

            prev_y = yy

            if len(streak_vals) >= k_points:
                # take the last k_points in this streak
                best_ref = float(np.mean(streak_vals[-k_points:]))
                best_n = int(streak_n[-1])  # convergence at the LAST point
                break

        log10_tau = float("nan")
        if np.isfinite(best_ref):
            log10_tau = -math.log10(2.0 * math.pi) - best_ref

        out.append(
            BZConvergenceSummary(
                float(fwhm), int(best_n), float(best_ref), float(log10_tau)
            )
        )

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_bz_convergence(
    paths_or_df: str | Path | Sequence[str | Path] | pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    tail_k: int = 5,
    thr_log10: float | None = None,
    annotate: bool = True,
    annotate_fmt: str = "{n:d}",
    annotate_offset_pts: Tuple[float, float] = (2.0, 2.0),
    xscale: str = "log",
    pretty_logx: bool = True,
    marker: str = "o",
    ms: float = 3.2,
    color: str = "k",
    label: str | None = None,
    fwhm_unit: str = r"$\mathrm{cm^{-1}}$",
    y_min: float = None,
    y_max: float = None,
) -> tuple[plt.Axes, pd.DataFrame]:
    """
    Publication-ready plot: x=FWHM, y=log10(tau), annotate with n_points_conv.

    Returns
    -------
    ax : matplotlib Axes
    summary_df : DataFrame with columns:
        fwhm, n_points_conv, log10_nu_peak_conv, log10_tau_conv
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=single_column_size(0.82), constrained_layout=True
        )

    if isinstance(paths_or_df, pd.DataFrame):
        df_long = paths_or_df.copy()
    else:
        df_long = read_bz_convergence_csv(paths_or_df)

    summaries = summarize_bz_convergence(df_long, tail_k=tail_k, thr_log10=thr_log10)

    summ_df = pd.DataFrame(
        [
            {
                "fwhm": s.fwhm,
                "n_points_conv": s.n_points_conv,
                "log10_nu_peak_conv": s.log10_nu_peak_conv,
                "log10_tau_conv": s.log10_tau_conv,
            }
            for s in summaries
        ]
    ).sort_values("fwhm")

    x = summ_df["fwhm"].to_numpy(float)
    y = summ_df["log10_tau_conv"].to_numpy(float)
    nconv = summ_df["n_points_conv"].to_numpy(int)

    fx = x[np.isfinite(x)]
    fy = y[np.isfinite(y)]
    if fx.size and fy.size:
        xmin, xmax = float(fx.min()), float(fx.max())
        ymin, ymax = float(fy.min()), float(fy.max())

        xpad = 2.2
        if xmin > 0 and xmax > 0:
            ax.set_xlim(xmin / xpad, xmax * xpad)

        ypad = 0.16
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(ymin - ypad, ymax + ypad)

    c0 = "#69A6D7"
    c1 = "#F1B960"

    first = True
    for i, (xx, yy) in enumerate(zip(x, y)):
        if not (np.isfinite(xx) and np.isfinite(yy)):
            continue
        ci = c0 if (i % 2 == 0) else c1
        ax.plot(
            xx,
            yy,
            ls="",
            marker=marker,
            ms=ms,
            color=ci,
            label=(label if (label and first) else None),
        )
        first = False

    if xscale:
        ax.set_xscale(xscale)
        if pretty_logx and xscale == "log":
            _pretty_log_ticks(ax)

    ax.set_xlabel(r"$\mathrm{FWHM}$ / " + fwhm_unit)
    ax.set_ylabel(r"$\log_{10}\,\tau$ / $\mathrm{s}$")
    ax.set_title("BZ grid vs FWHM convergence")
    _apply_axis_format(ax)

    if annotate:
        dx, dy = annotate_offset_pts
        for i, (xx, yy, nn) in enumerate(zip(x, y, nconv)):
            if not (np.isfinite(xx) and np.isfinite(yy)):
                continue
            ci = c0 if (i % 2 == 0) else c1
            s = annotate_fmt.format(n=int(nn))
            sgn = 1 if (i % 2 == 0) else -1
            ax.annotate(
                s,
                xy=(xx, yy),
                xytext=(dx, sgn * dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if sgn > 0 else "top",
                fontsize=4.5,
                color=ci,  # <- NEW
            )

    if label:
        ax.legend(frameon=False, fontsize=7, loc="best", handlelength=1.6)

    return ax, summ_df


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Plot BZ convergence: FWHM vs log10(tau) annotated by n_points_conv."
    )
    ap.add_argument("csv", nargs="+", help="One or more summary CSV files.")
    ap.add_argument(
        "--out",
        default="bz_convergence",
        help="Output basename (default: bz_convergence)",
    )
    ap.add_argument(
        "--tail-k", type=int, default=5, help="Tail window for reference (default: 5)"
    )
    ap.add_argument(
        "--thr-log10",
        type=float,
        default=None,
        help="Convergence threshold in log10 space",
    )
    ap.add_argument("--y-max", type=float, default=None, help="Y-axis limit max.")
    ap.add_argument("--y-min", type=float, default=None, help="Y-axis limit min.")
    args = ap.parse_args()

    set_jcp_style(figsize=single_column_size(0.82))
    fig, ax = plt.subplots(figsize=single_column_size(0.82), constrained_layout=True)
    ax, summ = plot_bz_convergence(
        args.csv,
        ax=ax,
        tail_k=args.tail_k,
        thr_log10=args.thr_log10,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    savefig_pub(fig, args.out, formats=("pdf",))
    print("Saved:", str(Path(args.out).with_suffix(".pdf")))
    print(summ.to_string(index=False))
