from __future__ import annotations

from enum import StrEnum


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
