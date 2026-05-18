from enum import StrEnum
from typing import ClassVar

import xarray as xr

from slothpy.groups.typed_group import SltTypedGroup


class HamiltonianVar(StrEnum):
    STATE_ENERGIES = "state_energies"
    SOC_MATRIX = "soc_matrix"
    SOC_SSC_MATRIX = "soc_ssc_matrix"
    SPIN_MATRICES = "spin_matrices"
    ANGULAR_MOMENTUM_MATRICES = "angular_momentum_matrices"
    ELECTRIC_DIPOLE_MOMENT_MATRICES = "electric_dipole_moment_matrices"

    @staticmethod
    def ci_alpha_occupations_mult(multiplicity: int) -> str:
        return f"ci_alpha_occupations_mult_{multiplicity}"

    @staticmethod
    def ci_beta_occupations_mult(multiplicity: int) -> str:
        return f"ci_beta_occupations_mult_{multiplicity}"

    @staticmethod
    def ci_coefficients_mult(multiplicity: int) -> str:
        return f"ci_coefficients_mult_{multiplicity}"


class HamiltonianCoord(StrEnum):
    STATE = "state"
    BRA_STATE = "bra_state"
    KET_STATE = "ket_state"
    COMPONENT = "component"
    CI_BRA_STATE = "ci_bra_state"
    CI_KET_STATE = "ci_ket_state"

    @staticmethod
    def determinant_mult(multiplicity: int) -> str:
        return f"determinant_mult_{multiplicity}"

    @staticmethod
    def root_mult(multiplicity: int) -> str:
        return f"root_mult_{multiplicity}"

    @staticmethod
    def active_orbital_mult(multiplicity: int) -> str:
        return f"active_orbital_mult_{multiplicity}"


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

    @property
    def has_spin_matrices(self) -> bool:
        return HamiltonianVar.SPIN_MATRICES.value in self.to_dataset()
