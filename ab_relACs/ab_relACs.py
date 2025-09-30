#!/usr/bin/env python3

# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

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

import argparse
import json
import logging
import os
import sys
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
import tomllib

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("ab_relACs")

def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("env:"):
            key = value.split(":", 1)[1].strip()
            return os.environ.get(key, "")
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value

_ALLOWED_FUNCS = {
    "linspace": np.linspace,
    "logspace": np.logspace,
    "arange":   np.arange,
    "array":    np.array,
}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)

def _eval_node(node):
    if isinstance(node, ast.Constant):  # int/float/bool/str
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

def _walk_and_eval_numpy(d):
    if isinstance(d, dict):
        return {k: _walk_and_eval_numpy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_walk_and_eval_numpy(x) for x in d]
    if isinstance(d, str) and d.strip().startswith("np."):
        return eval_numpy_expr(d)  # returns np.ndarray or scalar
    return d

class InpRelacs(BaseModel):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    slt_filepath: str = ""

    orientations: np.ndarray | List[list] | str | int = 0
    fields: np.ndarray | List[float] | str = Field(default_factory=list)
    temperatures: np.ndarray | List[float] | str = Field(default_factory=list)
    frequencies: np.ndarray | List[float] | str = "np.logspace(0, 6, 100)"

    states_number: int = 0
    degeneracy_tolerance: float = 1e-5
    psi_frequency_shift: bool = False
    initial_correlation: bool = False
    omega_loop: bool = False
    chi_s: bool | np.ndarray | List[float] = False

    n_points: List[int] = [1]
    q_ranges: List[float] = [0.5]
    broadening: str = "gaussian" # "lorentzian"
    fwhm: List[float] = [0.1]
    adaptive_fwhm: bool = False
    modes_low: float = 0.0
    modes_high: float = 0.1
    cutoff_fwhm: float = 1000
    qtm: bool = True

    chi_csv_path: str = ""
    tau_21_csv_path: str = ""
    tau_41_csv_path: str= ""
    show_plot: bool = True

class InpSupercell(BaseModel):
    model_config = ConfigDict(extra='forbid')
    xyz_path: str = ""
    group_name: str = ""
    cell_params: List[float] = Field(default_factory=list)
    nx: int = 0
    ny: int = 0
    nz: int = 0
    replace_atoms: List[int] = Field(default_factory=list)
    new_atoms:List[str] = Field(default_factory=list)

class InpHessian(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str = ""
    group_name: str = ""
    displacement_number: int = 0
    step: float = 0.0
    accoustic_sum_rule: str = ""

class InpSpinPhonon(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str = ""
    group_name: str = ""
    orca_fragovl_path: str = ""
    displacement_number: int = 0
    step: float = 0.0
        
class AppConfig(BaseModel):
    relacs: InpRelacs = Field(default_factory=InpRelacs)
    supercell: InpSupercell = Field(default_factory=InpSupercell)
    hessian: InpHessian = Field(default_factory=InpHessian)
    spin_phonon: InpSpinPhonon = Field(default_factory=InpSpinPhonon)

    @model_validator(mode="after")
    def _post(self):
        #TODO: implement validation and custom dataclasses
        return self

def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() != ".toml":
        raise ValueError("Only TOML is supported. Please provide a .toml file.")

    with path.open("rb") as f:
        raw: Dict[str, Any] = tomllib.load(f)
    logger.debug("Raw TOML: %s", raw)

    raw = expand_env(raw)
    logger.debug("After env expansion: %s", raw)

    raw = _walk_and_eval_numpy(raw)
    logger.debug("After numpy evaluation: %s", raw)

    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"configuration validation error: {e}")

    logger.debug("Validated config: %s", cfg)
    return cfg

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ab_relACs",
        description="Read a TOML config and validate it with Pydantic v2 (slim).",
    )
    p.add_argument("--config", type=Path, required=False, help="Path to TOML config file")
    p.add_argument("--dump", action="store_true", help="Print the parsed config and exit")
    p.add_argument("--dry-run", action="store_true", help="Load/validate only; do not run app")
    p.add_argument("--example", action="store_true", help="Print example TOML and exit")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    return p

EXAMPLE_TOML = """
# Example configuration (TOML)
[relacs]
slt_filepath = "./test.slt"

orientations = [[0,0,1,0.333333333],[1,0,0,0.333333333],[0,1,0,0.333333333]]
fields = [0.3]
temperatures = [5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40]
frequencies = "np.logspace(-3, 7, 300)"

states_number = 13
degeneracy_tolerance = 1e-5
psi_frequency_shift = false
initial_correlation = false
omega_loop = false
chi_s = false

n_points = [19]
q_ranges = [0.125,0.25,0.5]
broadening = "gaussian" # "lorentzian"
fwhm = [0.021]
adaptive_fwhm = false
modes_low = 1e-5
modes_high = 2000
cutoff_fwhm = 1000
qtm = true

chi_csv_path = "./test_ac_relacs.dat"
tau_21_csv_path = "./test_tau_R21_relacs.dat"
tau_41_csv_path = "./test_tau_R41_relacs.dat"
show_plot = true

[supercell]
xyz_path = "./YCo_supercell_from_cell/dof_0_disp_0.xyz"
group_name = "YCo_supercell"
cell_params = [22.663134149075237, 22.663134149075233, 25.14851428466812, 90.0, 90.0, 120.0]
nx = 3
ny = 3
nz = 2
replace_atoms = [0]
new_atoms = ["Tb"]

[hessian]
path = "./YCo_supercell_from_cell"
group_name = "YCo_hessian"
displacement_number = 1
step = 0.01
accoustic_sum_rule = "symmetric"

[spin_phonon]
path = "./TbCo"
group_name = "TbCo_spin_phonon"
orca_fragovl_path = "/home/orca_6_1_0_avx2/orca_fragovl"
displacement_number = 1
step = 0.0005
""".strip()

def run_app(cfg: AppConfig) -> int:
    print(cfg.hessian.path)
    print(cfg.relacs.frequencies)
    logger.debug("Full config: %s", cfg)
    # TODO: put your real logic here
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.example:
        print(EXAMPLE_TOML)
        return 0

    if not args.config:
        parser.error("--config is required (or use --example)")

    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 2

    if args.dump:
        print(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False))
        if args.dry_run:
            return 0

    if args.dry_run:
        logger.info("Dry-run complete. Config OK.")
        return 0

    return run_app(cfg)


if __name__ == "__main__":
    sys.exit(main())
