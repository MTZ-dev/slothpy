import pytest

from slothpy.io.readers.hamiltonian_reader import HamiltonianReader
from slothpy.io.readers.orca_hamiltonian_reader import (
    OrcaHamiltonianReader,
    hamiltonian_from_orca,
)


def test_orca_hamiltonian_reader_is_hamiltonian_reader_subclass() -> None:
    reader = OrcaHamiltonianReader()

    assert isinstance(reader, HamiltonianReader)


def test_hamiltonian_from_orca_rejects_reader_with_option_kwargs() -> None:
    with pytest.raises(TypeError, match="reader= is provided"):
        hamiltonian_from_orca(
            "x.out",
            "y.slt",
            "ham",
            reader=OrcaHamiltonianReader(),
            pt2=True,
        )


def test_hamiltonian_from_orca_rejects_both_dipole_aliases() -> None:
    with pytest.raises(TypeError, match="only one"):
        hamiltonian_from_orca(
            "x.out",
            "y.slt",
            "ham",
            electric_dipole_moment_matrices=True,
            include_electric_dipole_moment_matrices=False,
        )


def test_hamiltonian_from_orca_rejects_unknown_reader_kw() -> None:
    with pytest.raises(TypeError, match="Unknown reader option"):
        hamiltonian_from_orca("x.out", "y.slt", "ham", not_a_valid_field=True)
