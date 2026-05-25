"""
SlothPy semantic specs for typed groups and in-memory results.

Each module defines variable/coordinate names, shared data-access mixins,
typed group handles, and result views for one ``slt_type``.
"""

from slothpy.specs.magnetisation import (
    MAGNETISATION_SLT_TYPE,
    MagnetisationCoord,
    MagnetisationDataMixin,
    MagnetisationOptions,
    MagnetisationVar,
    SltMagnetisationGroup,
    SltMagnetisationResult,
)

__all__ = [
    "MAGNETISATION_SLT_TYPE",
    "MagnetisationCoord",
    "MagnetisationDataMixin",
    "MagnetisationOptions",
    "MagnetisationVar",
    "SltMagnetisationGroup",
    "SltMagnetisationResult",
]
