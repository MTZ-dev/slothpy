from dataclasses import dataclass
from typing import ClassVar

from slothpy.compute.magnetisation import MagnetisationDataMixin
from slothpy.groups.typed_group import SltTypedGroup


@dataclass(frozen=True, slots=True)
class SltMagnetisationGroup(MagnetisationDataMixin, SltTypedGroup):
    expected_slt_type: ClassVar[str] = "MAGNETISATION"


__all__ = [
    "SltMagnetisationGroup",
]
