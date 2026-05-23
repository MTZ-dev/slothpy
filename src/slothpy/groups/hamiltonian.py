from __future__ import annotations

from typing import Any, ClassVar

import xarray as xr
from numpy.typing import ArrayLike

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

    def magnetisation(
        self,
        magnetic_fields: ArrayLike,
        orientations: ArrayLike,
        temperatures: ArrayLike,
        *,
        states_cutoff: tuple[int, int] = (0, 0),
        rotation: ArrayLike | None = None,
        electric_field_vector: ArrayLike | None = None,
        num_processes: int = 1,
        num_threads: int = 1,
        save_as: str | None = None,
        overwrite: bool = False,
    ) -> Any:
        self.require_rule(has_zeeman_matrix, "magnetisation")

        from slothpy.compute.magnetisation import magnetisation

        return magnetisation(
            self,
            magnetic_fields,
            orientations,
            temperatures,
            states_cutoff=states_cutoff,
            rotation=rotation,
            electric_field_vector=electric_field_vector,
            num_processes=num_processes,
            num_threads=num_threads,
            save_as=save_as,
            output_slt=self.file_path,
            overwrite=overwrite,
        )
