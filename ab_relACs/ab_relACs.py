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

from __future__ import annotations

import os
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"] = "cores"
os.environ['NUMBA_OPT'] = '3'
os.environ['NUMBA_LOOP_VECTORIZE'] = '1'
os.environ['NUMBA_ENABLE_AVX'] = '1'

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from importlib.resources import files
from typing import Any, Dict, List, Optional
import tomllib

from pydantic import ValidationError

from run_ab_relACs import run_relacs
from input_models import AppConfig
from utils import eval_numpy_expr


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

def _walk_and_eval_numpy(d):
    if isinstance(d, dict):
        return {k: _walk_and_eval_numpy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_walk_and_eval_numpy(x) for x in d]
    if isinstance(d, str) and d.strip().startswith("np."):
        return eval_numpy_expr(d)
    return d

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

def run_app(cfg: AppConfig) -> int:
    logger.debug("Full config: %s", cfg)
    start = time.perf_counter()
    status = run_relacs(cfg)
    end = time.perf_counter()
    print(f"Running time: {end - start} s")
    return status

def main(argv: Optional[List[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.example:
        try:
            example_path = files("ab_relACs").joinpath("examples/config_example.toml")
            with example_path.open("r", encoding="utf-8") as f:
                print(f.read())
        except Exception:
            here = Path(__file__).resolve().parent
            example_path = here / "examples" / "config_example.toml"
            with example_path.open("r", encoding="utf-8") as f:
                print(f.read())
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
