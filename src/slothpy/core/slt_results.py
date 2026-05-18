from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SltResults:
    """
    Bundle for writing one SlothPy semantic xarray group.

    Passed to :meth:`SltFile._write_slothpy_group` together with the target
    group name. Producers supply an ``xr.Dataset`` or ``DataArray``, optional
    ``slt_type`` and ``primary`` (stored as SlothPy dataset metadata), and
    optional extra entries applied to the returned :class:`SltGroup` attributes.
    """

    dataset: xr.Dataset | xr.DataArray
    slt_type: str | None = None
    primary: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
