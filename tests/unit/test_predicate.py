from __future__ import annotations

from slothpy.logic.predicate import rule
from slothpy.logic.rules import (
    has_angular_momentum_matrices,
    has_soc_matrix,
    has_spin_matrices,
    has_state_energies,
    has_zeeman_matrix,
)


def test_slt_predicate_stores_description() -> None:
    @rule
    def always_true(_: object) -> bool:
        return True

    assert repr(always_true) == "always_true"
    assert str(always_true) == "always_true"


def test_composite_predicate_repr_uses_bitwise_operators() -> None:
    combined = (has_state_energies | has_soc_matrix) & has_spin_matrices

    assert "has_state_energies" in repr(combined)
    assert "has_soc_matrix" in repr(combined)
    assert "has_spin_matrices" in repr(combined)
    assert combined is not has_spin_matrices


def test_has_zeeman_matrix_is_full_composite_not_single_rule() -> None:
    assert has_zeeman_matrix is not has_angular_momentum_matrices
    assert has_zeeman_matrix is not has_spin_matrices
    text = repr(has_zeeman_matrix)
    assert "has_angular_momentum_matrices" in text
    assert "has_spin_matrices" in text
