from __future__ import annotations

from slothpy.core.slt_file import SltFile
from slothpy.types.aliases import PathLike

# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def create_slt_file(path: PathLike, *, overwrite: bool = False) -> SltFile:
    """
    Create a new xarray-backed ``.slt`` file and return an ``SltFile`` handle.
    """
    return SltFile._create(path, overwrite=overwrite)


def open_slt_file(path: PathLike) -> SltFile:
    """
    Open an existing ``.slt`` file and return an ``SltFile`` handle.
    """
    return SltFile._new(path)


def slt_file(path: PathLike) -> SltFile:
    """
    Backward-compatible alias for opening an existing ``.slt`` file.
    """
    return open_slt_file(path)
