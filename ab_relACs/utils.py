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

import ast
import posixpath
import re

import numpy as np
import h5py
from numba import njit

from slothpy._general_utilities._grids_over_hemisphere import lebedev_laikov_grid_over_hemisphere
from slothpy._general_utilities._constants import B_AU_T
from input_models import AppConfig

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

@njit
def _normalize_orientations(orientations: np.ndarray):
    for vector_index in range(orientations.shape[0]):
        length = orientations[vector_index][0] ** 2 + orientations[vector_index][1] ** 2 + orientations[vector_index][2] ** 2
        if length == 0:
            raise ValueError("Vector of length zero detected in the input orientations.")
        length = 1 / (np.sqrt(length)*B_AU_T)
        orientations[vector_index,:3] = orientations[vector_index,:3] * length
    return orientations

def get_normalized_orientations_weights(cfg: AppConfig):
    orientations = cfg.relacs.orientations
    if isinstance(orientations, int):
        orientations = lebedev_laikov_grid_over_hemisphere(orientations, "double")
    else:
        orientations = np.asarray(orientations, dtype=np.float64)
    orientations = _normalize_orientations(orientations)
    return np.ascontiguousarray(orientations[:,:3]), np.ascontiguousarray(orientations[:,3])

@njit
def dot_3d(M: np.ndarray, N: np.ndarray):
    return M[0] * N[0] + M[1] * N[1] + M[2] * N[2]

        
