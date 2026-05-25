from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
import xarray as xr

from slothpy.core.slt_results import SltResultView
from slothpy.groups.typed_group import SltTypedGroup

MAGNETISATION_SLT_TYPE = "MAGNETISATION"


@dataclass(frozen=True, slots=True)
class MagnetisationOptions:
    """
    Input arrays for a magnetisation computation.
    """

    magnetic_fields: np.ndarray
    orientations: np.ndarray
    temperatures: np.ndarray
    states_cutoff: tuple[int, int] = (0, 0)
    rotation: np.ndarray | None = None
    electric_field_vector: np.ndarray | None = None
    steps_per_task: int = 8
    sleep_seconds: float = 0.05
    progress_interval_steps: int = 1


class MagnetisationVar(StrEnum):
    MAGNETISATION = "magnetisation"
    ORIENTATIONS = "orientations"


class MagnetisationCoord(StrEnum):
    TEMPERATURE = "temperature"
    FIELD = "field"
    ORIENTATION = "orientation"


class MagnetisationDataMixin:
    """
    Shared xarray accessors for magnetisation semantic groups and results.
    """

    expected_slt_type: ClassVar[str] = MAGNETISATION_SLT_TYPE

    @property
    def magnetisation(self) -> xr.DataArray:
        return self.dataset[MagnetisationVar.MAGNETISATION]  # type: ignore[attr-defined]

    @property
    def temperatures(self) -> xr.DataArray:
        return self.dataset[MagnetisationCoord.TEMPERATURE]  # type: ignore[attr-defined]

    @property
    def magnetic_fields(self) -> xr.DataArray:
        return self.dataset[MagnetisationCoord.FIELD]  # type: ignore[attr-defined]

    @property
    def orientations(self) -> xr.DataArray | None:
        dataset = self.dataset  # type: ignore[attr-defined]
        if MagnetisationVar.ORIENTATIONS in dataset:
            return dataset[MagnetisationVar.ORIENTATIONS]
        return None

    def to_dataframe(self) -> Any:
        return self.magnetisation.to_dataframe()


class SltMagnetisationGroup(MagnetisationDataMixin, SltTypedGroup):
    """
    Typed on-disk magnetisation semantic group.
    """

    expected_slt_type: ClassVar[str] = MAGNETISATION_SLT_TYPE


class SltMagnetisationResult(MagnetisationDataMixin, SltResultView):
    """
    Typed in-memory magnetisation computation result.
    """

    expected_slt_type: ClassVar[str] = MAGNETISATION_SLT_TYPE
