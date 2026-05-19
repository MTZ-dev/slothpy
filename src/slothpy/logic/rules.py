from slothpy.groups.hamiltonian_names import HamiltonianVar
from slothpy.groups.typed_group import SltTypedGroup
from slothpy.logic.predicate import rule


@rule
def has_spin_matrices(group: SltTypedGroup) -> bool:
    return group.has_variable(HamiltonianVar.SPIN_MATRICES.value)


@rule
def has_angular_momentum_matrices(group: SltTypedGroup) -> bool:
    return group.has_variable(HamiltonianVar.ANGULAR_MOMENTUM_MATRICES.value)


@rule
def has_electric_dipole_moment_matrices(group: SltTypedGroup) -> bool:
    return group.has_variable(HamiltonianVar.ELECTRIC_DIPOLE_MOMENT_MATRICES.value)


@rule
def has_state_energies(group: SltTypedGroup) -> bool:
    return group.has_variable(HamiltonianVar.STATE_ENERGIES.value)


@rule
def has_soc_matrix(group: SltTypedGroup) -> bool:
    return group.has_variable(HamiltonianVar.SOC_MATRIX.value)


@rule
def has_soc_ssc_matrix(group: SltTypedGroup) -> bool:
    return group.has_variable(HamiltonianVar.SOC_SSC_MATRIX.value)


has_zeeman_matrix = (
    (has_state_energies | has_soc_matrix)
    & has_spin_matrices
    & has_angular_momentum_matrices
)

has_zeeman_matrix.name = "has_zeeman_matrix"
