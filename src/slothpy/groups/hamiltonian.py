from __future__ import annotations

from typing import ClassVar

import xarray as xr

from slothpy.groups.hamiltonian_names import HamiltonianCoord, HamiltonianVar
from slothpy.groups.typed_group import SltTypedGroup
from slothpy.logic.rules import has_zeeman_matrix

__all__ = [
    "HamiltonianCoord",
    "HamiltonianVar",
    "SltHamiltonianGroup",
]


class SltHamiltonianGroup(SltTypedGroup):
    expected_slt_type: ClassVar[str] = "HAMILTONIAN"

    @property
    def state_energies(self) -> xr.DataArray:
        return self.to_dataset()[HamiltonianVar.STATE_ENERGIES.value]

    @property
    def spin_matrices(self) -> xr.DataArray:
        return self.to_dataset()[HamiltonianVar.SPIN_MATRICES.value]

    @property
    def angular_momentum_matrices(self) -> xr.DataArray:
        return self.to_dataset()[HamiltonianVar.ANGULAR_MOMENTUM_MATRICES.value]

    @property
    def electric_dipole_moment_matrices(self) -> xr.DataArray:
        return self.to_dataset()[HamiltonianVar.ELECTRIC_DIPOLE_MOMENT_MATRICES.value]

    def magnetisation(self) -> xr.DataArray:
        self.require_rule(has_zeeman_matrix, "magnetisation")
