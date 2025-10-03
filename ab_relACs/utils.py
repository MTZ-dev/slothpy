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

import ast
import posixpath
import re
import os
import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import h5py
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from slothpy._general_utilities._grids_over_hemisphere import lebedev_laikov_grid_over_hemisphere
from slothpy._general_utilities._constants import B_AU_T

def h5_has_group(filepath: str | os.PathLike | Path, group_path: str) -> bool:
    try:
        with h5py.File(filepath, "r") as f:
            cls = f.get(group_path, getclass=True)
            return cls is h5py.Group
    except OSError:
        return False

_ALLOWED_FUNCS = {
    "linspace": np.linspace,
    "logspace": np.logspace,
    "arange":   np.arange,
    "array":    np.array,
}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)

def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return node.value
        raise ValueError("only int/float/bool/str constants allowed")
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(x) for x in node.elts)
    if isinstance(node, ast.List):
        return [_eval_node(x) for x in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        val = _eval_node(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left, right = _eval_node(node.left), _eval_node(node.right)
        return eval(compile(ast.Expression(node), "<ast-op>", "eval"))  # numeric only
    if isinstance(node, ast.Call):
        func = node.func
        # Expect np.<name>(...)
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "np"):
            raise ValueError("only calls like np.linspace(...) are allowed")
        name = func.attr
        if name not in _ALLOWED_FUNCS:
            raise ValueError(f"np.{name} not allowed")
        args = [_eval_node(a) for a in node.args]
        kwargs = {kw.arg: _eval_node(kw.value) for kw in node.keywords}
        return _ALLOWED_FUNCS[name](*args, **kwargs)
    raise ValueError(f"unsupported syntax: {ast.dump(node, include_attributes=False)}")

def eval_numpy_expr(expr: str):
    """Evaluate a very small, whitelisted subset of NumPy calls from a string."""
    expr = expr.strip()
    if not expr.startswith("np."):
        raise ValueError("expression must start with 'np.'")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)

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

@njit(nogil=True, cache=True, fastmath=True)
def _normalize_orientations(orientations: np.ndarray):
    for vector_index in range(orientations.shape[0]):
        length = orientations[vector_index][0] ** 2 + orientations[vector_index][1] ** 2 + orientations[vector_index][2] ** 2
        if length == 0:
            raise ValueError("Vector of length zero detected in the input orientations.")
        length = 1 / (np.sqrt(length)*B_AU_T)
        orientations[vector_index,:3] = orientations[vector_index,:3] * length
    return orientations

def get_normalized_orientations_weights(orientations):
    if isinstance(orientations, int):
        orientations = lebedev_laikov_grid_over_hemisphere(orientations, "double")
    else:
        orientations = np.asarray(orientations, dtype=np.float64)
    orientations = _normalize_orientations(orientations)
    return np.ascontiguousarray(orientations[:,:3]), np.ascontiguousarray(orientations[:,3])

@njit(nogil=True, cache=True, fastmath=True)
def dot_3d(M: np.ndarray, N: np.ndarray):
    return M[0] * N[0] + M[1] * N[1] + M[2] * N[2]

def half_bz_grid_aniso(
    b_len: Sequence[float],
    n_ref: int,
    start_q: float,
    end_q: float = 0.0,
    *,
    endpoint: bool = True,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Anisotropic half-BZ mesh (unique reps of {+q,−q}) inside an L∞ shell:
        end_q < max(|qx|,|qy|,|qz|) <= start_q
    If end_q == 0 the Γ point is included.
    """

    if n_ref <= 0:
        raise ValueError("n_ref must be positive.")
    b_len = np.asarray(b_len, float)
    if b_len.size != 3 or np.any(b_len <= 0):
        raise ValueError("b_len must contain three positive numbers.")
    if not (start_q > 0):
        raise ValueError("start_q must be positive.")
    if end_q < 0 or end_q >= start_q:
        raise ValueError("Require 0 <= end_q < start_q.")

    b_min = b_len.min()
    n_axis = []
    for L in b_len:
        n = int(round(n_ref * L / b_min))
        if n % 2 == 0:
            n += 1
        n_axis.append(n)

    ax = [np.linspace(-start_q, start_q, n, endpoint=endpoint, dtype=float)
          for n in n_axis]
    full = np.array(np.meshgrid(*ax, indexing="ij")).reshape(3, -1).T 
    maxabs = np.max(np.abs(full), axis=1)
    mask_shell = (maxabs > end_q + tol) & (maxabs <= start_q + tol)
    full = full[mask_shell]
    if full.size == 0:
        return full

    keep = np.zeros(full.shape[0], dtype=bool)
    include_gamma = (end_q <= tol)
    for i, (x, y, z) in enumerate(full):
        if include_gamma and abs(x) < tol and abs(y) < tol and abs(z) < tol:
            keep[i] = True
            continue
        if   x >  tol: keep[i] = True
        elif x < -tol: continue
        elif y >  tol: keep[i] = True
        elif y < -tol: continue
        elif z >  tol: keep[i] = True
    q_unique = full[keep]

    idx = np.lexsort(q_unique.T[::-1])
    return q_unique[idx]

def _set_equal_3d(ax, X, Y, Z):
    """Make 3D axes scale equal for nicer geometry perception."""
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = Z.max() - Z.min()
    max_range = max(x_range, y_range, z_range)
    x_mid = 0.5 * (X.max() + X.min())
    y_mid = 0.5 * (Y.max() + Y.min())
    z_mid = 0.5 * (Z.max() + Z.min())
    half = 0.5 * max_range
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

def multigrid_aniso(
    b_len: Sequence[float],
    n_ref: int,
    q_ranges: Sequence[float],
    *,
    endpoint: bool = True,
    tol: float = 1e-12,
    plot: bool = False,
    ax: mpl.axes.Axes | None = None,
    cmap: str = "viridis",
    s: float = 8.0,
    alpha: float = 0.9,
) -> np.ndarray:
    grids_list = []
    weights_list = []
    q_ranges.insert(0, 0.0)
    for i_q in range(1,len(q_ranges)):
        aniso_grid = half_bz_grid_aniso(b_len, n_ref, q_ranges[i_q], q_ranges[i_q-1],
                                         endpoint=endpoint, tol=tol)
        grid_weight = (2*q_ranges[i_q])**3
        aniso_weights = np.full(aniso_grid.shape[0], grid_weight)
        grids_list.append(aniso_grid)
        weights_list.append(aniso_weights)
    
    grid = np.vstack(grids_list)
    weights = np.concatenate(weights_list)

    q_ranges.pop(0)

    if plot:
        if ax is None:
            fig = plt.figure(figsize=(6.5, 5.5), constrained_layout=True)
            ax  = fig.add_subplot(111, projection="3d")
        vmin, vmax = weights.min(), weights.max()
        use_log = vmax / max(vmin, 1e-300) > 50
        norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else None

        sc = ax.scatter(
            grid[:,0], grid[:,1], grid[:,2],
            c=weights, cmap=cmap, norm=norm, s=s, alpha=alpha, edgecolors="none"
        )
        _set_equal_3d(ax, grid[:,0], grid[:,1], grid[:,2])
        ax.set_xlabel(r"$q_x$ (frac.)")
        ax.set_ylabel(r"$q_y$ (frac.)")
        ax.set_zlabel(r"$q_z$ (frac.)")
        ax.set_title("Multigrid in fractional BZ (colour = weight)")
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label("Weight")
        if ax.figure is not None:
            ax.figure.canvas.draw_idle()
            plt.show()
    
    return grid, weights

def make_npoints_fwhm_filename(filepath: Union[str, Path], npoints: int, fwhm: float) -> str:
    p = Path(filepath)
    suffixes = "".join(p.suffixes)
    base = p.name[:-len(suffixes)] if suffixes else p.name

    def fmt(v):
        if isinstance(v, (int, float)):
            s = f"{v:g}"
        else:
            s = str(v)
        return s.replace(" ", "").replace("/", "-")

    new_name = f"{base}_npoints_{fmt(npoints)}_fwhm_{fmt(fwhm)}{suffixes}"
    return str(p.with_name(new_name))

def make_npoints_fwhm_orient_filename(
    filepath: Union[str, Path],
    npoints: int,
    fwhm: float,
    orientation: Sequence[float],
    sig: int = 6,
    int_tol: float = 1e-12,
) -> str:

    p = Path(filepath)
    suffixes = "".join(p.suffixes)
    base = p.name[:-len(suffixes)] if suffixes else p.name

    vec = np.asarray(orientation, dtype=float)
    if vec.shape != (3,):
        raise ValueError("orientation must be length-3 (x, y, z).")
    if not np.all(np.isfinite(vec)) or not math.isfinite(fwhm):
        raise ValueError("Non-finite numbers (NaN/Inf) are not allowed in filenames.")

    def fmt_num(v):
        # ints stay ints, floats capped to `sig` significant digits
        if isinstance(v, (np.integer, int)):
            s = str(int(v))
        else:
            fv = float(v)
            if abs(fv - round(fv)) < int_tol:
                s = str(int(round(fv)))
            else:
                s = f"{fv:.{sig}g}"  # caps total significant digits
        # clean up filename-unfriendly artifacts
        if s in ("-0", "-0.0", "+0", "+0.0"): s = "0"
        return s.replace(" ", "").replace("+", "")

    x, y, z = (fmt_num(v) for v in vec)
    new_name = (
        f"{base}"
        f"_npoints_{fmt_num(npoints)}"
        f"_fwhm_{fmt_num(fwhm)}"
        f"_ori_x{x}_y{y}_z{z}"
        f"{suffixes}"
    )
    return str(p.with_name(new_name))


        
