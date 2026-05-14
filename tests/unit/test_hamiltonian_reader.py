from __future__ import annotations

from collections.abc import Iterable
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from slothpy.core.slt import SltGroup, SltResults, create_slt_file, open_slt_file
from slothpy.io.readers.hamiltonian_reader import (
    CIDeterminantExpansion,
    HamiltonianReader,
    HamiltonianReaderOptions,
    HamiltonianReaderResult,
    add_ci_expansion_variables_to_dataset,
    hamiltonian_reader_result_to_slt_results,
    write_hamiltonian_reader_result_to_slt_group,
    write_slt_results_to_group,
)
from slothpy.io.readers.orca_hamiltonian_reader import OrcaHamiltonianReaderOptions
from slothpy.types.aliases import PathLike


class _FakeHamiltonianReader(HamiltonianReader):
    def __init__(self, result: HamiltonianReaderResult) -> None:
        self.result = result
        self.sources: list[PathLike | Iterable[str]] = []

    def read(self, source: PathLike | Iterable[str]) -> HamiltonianReaderResult:
        self.sources.append(source)
        return self.result


def _diagonal_result() -> HamiltonianReaderResult:
    return HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        attrs={"slt_kind": "TEST", "shift_energies_applied": False},
    )


def _ci_result() -> HamiltonianReaderResult:
    return HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="CI",
        hamiltonian_matrix=np.eye(2, dtype=np.complex128),
        attrs={"slt_kind": "TEST"},
    )


def test_hamiltonian_reader_options_defaults() -> None:
    options = HamiltonianReaderOptions()

    assert options.parse_ci_expansions is False
    assert options.diagonalize is True
    assert options.shift_energies is True
    assert options.include_spin_matrices is True
    assert options.include_angular_momentum_matrices is True
    assert options.include_electric_dipole_moment_matrices is False


def test_hamiltonian_reader_options_are_shared_via_base_type() -> None:
    options = OrcaHamiltonianReaderOptions()

    assert isinstance(options, HamiltonianReaderOptions)
    assert options.parse_ci_expansions is False
    assert options.diagonalize is True
    assert options.shift_energies is True
    assert options.include_spin_matrices is True
    assert options.include_angular_momentum_matrices is True
    assert options.include_electric_dipole_moment_matrices is False


def test_slt_results_is_dataclass() -> None:
    assert is_dataclass(SltResults)


def test_hamiltonian_reader_is_abstract() -> None:
    with pytest.raises(TypeError):
        HamiltonianReader()  # type: ignore[abstract]


def test_hamiltonian_reader_read_as_slt_results() -> None:
    structured = _diagonal_result()
    reader = _FakeHamiltonianReader(structured)

    out = reader.read_as_slt_results(["dummy"])

    assert isinstance(out, SltResults)
    assert out.primary == "states_energies"
    assert out.slt_type == "HAMILTONIAN"
    assert reader.sources == [["dummy"]]


def test_hamiltonian_reader_write_to_group(tmp_path: Path) -> None:
    structured = _diagonal_result()
    reader = _FakeHamiltonianReader(structured)

    slt = create_slt_file(tmp_path / "reader_write.slt")
    group = reader.write_to_group(["dummy"], slt, "hamiltonian", overwrite=False)

    assert isinstance(group, SltGroup)
    assert group.path == "hamiltonian"
    assert reader.sources == [["dummy"]]

    dataset = group.to_dataset()
    try:
        assert "states_energies" in dataset
    finally:
        dataset.close()


def test_compose_diagonal_minimal_hamiltonian_only() -> None:
    structured = _diagonal_result()

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert out.primary == "states_energies"
    assert out.slt_type == "HAMILTONIAN"
    assert out.attrs["slt_kind"] == "TEST"
    assert set(out.dataset.data_vars) == {"states_energies"}
    assert set(out.dataset.coords) == {"state"}
    assert out.dataset["states_energies"].attrs["unit"] == "E_h"
    assert (
        out.dataset["states_energies"].attrs["long_name"] == "SOC eigenstate energies"
    )


def test_compose_diagonal_shifted_energies_long_name() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        attrs={"shift_energies_applied": True},
    )

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert (
        "shifted to the lowest state"
        in out.dataset["states_energies"].attrs["long_name"]
    )


def test_compose_diagonal_with_optional_operators() -> None:
    dim = 2
    eye = np.eye(dim, dtype=np.complex128)
    stack = np.stack([eye, eye, eye], axis=0)

    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        spin_matrices=stack,
        angular_momentum_matrices=stack,
        electric_dipole_moment_matrices=stack,
        attrs={"slt_kind": "TEST", "shift_energies_applied": False},
    )

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert "spin_matrices" in out.dataset
    assert "angular_momentum_matrices" in out.dataset
    assert "electric_dipole_moment_matrices" in out.dataset
    assert "state" in out.dataset.coords
    assert "bra_state" in out.dataset.coords
    assert "ket_state" in out.dataset.coords
    assert out.dataset["component"].values.tolist() == ["x", "y", "z"]


def test_compose_diagonal_rejects_missing_states_energies() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        attrs={},
    )

    with pytest.raises(ValueError, match="states_energies"):
        hamiltonian_reader_result_to_slt_results(structured)


def test_compose_diagonal_rejects_hamiltonian_matrix() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        hamiltonian_matrix=np.eye(2, dtype=np.complex128),
        attrs={},
    )

    with pytest.raises(ValueError, match="representation='DIAGONAL'"):
        hamiltonian_reader_result_to_slt_results(structured)


def test_compose_rejects_bad_operator_shape() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        spin_matrices=np.zeros((2, 2, 2), dtype=np.complex128),
        attrs={},
    )

    with pytest.raises(ValueError, match=r"spin_matrices must have shape"):
        hamiltonian_reader_result_to_slt_results(structured)


def test_compose_ci_minimal_matrix_only() -> None:
    structured = _ci_result()

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert out.primary == "soc_matrix"
    assert set(out.dataset.data_vars) == {"soc_matrix"}
    assert set(out.dataset.coords) == {"ci_bra_state", "ci_ket_state"}
    assert out.dataset["soc_matrix"].attrs["unit"] == "E_h"
    assert out.dataset["soc_matrix"].attrs["long_name"] == "SOC matrix in CI basis"


def test_compose_ci_soc_ssc_matrix() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC_SSC",
        representation="CI",
        hamiltonian_matrix=np.eye(2, dtype=np.complex128),
        attrs={"slt_kind": "TEST"},
    )

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert out.primary == "soc_ssc_matrix"
    assert set(out.dataset.data_vars) == {"soc_ssc_matrix"}
    assert (
        out.dataset["soc_ssc_matrix"].attrs["long_name"] == "SOC+SSC matrix in CI basis"
    )


def test_compose_ci_with_optional_operators() -> None:
    dim = 2
    eye = np.eye(dim, dtype=np.complex128)
    stack = np.stack([eye, eye, eye], axis=0)

    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="CI",
        hamiltonian_matrix=eye,
        spin_matrices=stack,
        angular_momentum_matrices=stack,
        electric_dipole_moment_matrices=stack,
        attrs={},
    )

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert "spin_matrices" in out.dataset
    assert "angular_momentum_matrices" in out.dataset
    assert "electric_dipole_moment_matrices" in out.dataset
    assert "ci_bra_state" in out.dataset.coords
    assert "ci_ket_state" in out.dataset.coords
    assert out.dataset["component"].values.tolist() == ["x", "y", "z"]


def test_compose_ci_with_ci_expansions() -> None:
    expansion = CIDeterminantExpansion(
        alpha_occupations=np.array(
            [
                [1, 1, 0],
                [1, 0, 1],
            ],
            dtype=np.int64,
        ),
        beta_occupations=np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.int64,
        ),
        ci_coefficients=np.array(
            [
                [0.8, 0.1],
                [0.2, 0.9],
            ],
            dtype=np.float64,
        ),
    )

    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="CI",
        hamiltonian_matrix=np.eye(2, dtype=np.complex128),
        ci_expansions_by_multiplicity={3: expansion},
        attrs={},
    )

    out = hamiltonian_reader_result_to_slt_results(structured)

    assert "determinant_mult_3" in out.dataset.coords
    assert "root_mult_3" in out.dataset.coords
    assert "active_orbital_mult_3" in out.dataset.coords
    assert "ci_alpha_occupations_mult_3" in out.dataset
    assert "ci_beta_occupations_mult_3" in out.dataset
    assert "ci_coefficients_mult_3" in out.dataset
    assert out.dataset["ci_coefficients_mult_3"].shape == (2, 2)


def test_add_ci_expansion_variables_to_dataset_preserves_input_dataset() -> None:
    dataset = xr.Dataset({"x": ("state", np.array([1.0, 2.0]))})

    expansion = CIDeterminantExpansion(
        alpha_occupations=np.array([[1, 0]], dtype=np.int64),
        beta_occupations=np.array([[0, 1]], dtype=np.int64),
        ci_coefficients=np.array([[1.0]], dtype=np.float64),
    )

    out = add_ci_expansion_variables_to_dataset(dataset, {1: expansion})

    assert "ci_alpha_occupations_mult_1" not in dataset
    assert "ci_alpha_occupations_mult_1" in out


def test_compose_rejects_ci_with_states_energies() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="CI",
        hamiltonian_matrix=np.eye(2, dtype=np.complex128),
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        attrs={},
    )

    with pytest.raises(ValueError, match="representation='CI'"):
        hamiltonian_reader_result_to_slt_results(structured)


def test_compose_rejects_ci_without_hamiltonian_matrix() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="CI",
        attrs={},
    )

    with pytest.raises(ValueError, match="hamiltonian_matrix"):
        hamiltonian_reader_result_to_slt_results(structured)


def test_compose_rejects_non_square_ci_hamiltonian_matrix() -> None:
    structured = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="CI",
        hamiltonian_matrix=np.zeros((2, 3), dtype=np.complex128),
        attrs={},
    )

    with pytest.raises(ValueError, match="must be square"):
        hamiltonian_reader_result_to_slt_results(structured)


def test_write_slt_results_to_group(tmp_path: Path) -> None:
    structured = _diagonal_result()
    results = hamiltonian_reader_result_to_slt_results(structured)

    path = tmp_path / "ham.slt"
    slt = create_slt_file(path)

    group = write_slt_results_to_group(
        slt,
        "ham_group",
        results,
        overwrite=False,
    )

    assert isinstance(group, SltGroup)

    dataset = group.to_dataset()
    try:
        assert "states_energies" in dataset
    finally:
        dataset.close()


def test_write_hamiltonian_reader_result_to_slt_group(tmp_path: Path) -> None:
    structured = _diagonal_result()

    path = tmp_path / "ham.slt"
    slt = create_slt_file(path)

    write_hamiltonian_reader_result_to_slt_group(
        slt,
        "ham_group",
        structured,
        overwrite=False,
    )

    slt2 = open_slt_file(path)

    assert "ham_group" in slt2

    group = slt2["ham_group"]
    assert isinstance(group, SltGroup)

    dataset = group.to_dataset()
    try:
        assert "states_energies" in dataset
    finally:
        dataset.close()


def test_write_hamiltonian_reader_result_to_slt_group_custom_slt_type(
    tmp_path: Path,
) -> None:
    structured = _diagonal_result()

    slt = create_slt_file(tmp_path / "ham.slt")

    group = write_hamiltonian_reader_result_to_slt_group(
        slt,
        "ham_group",
        structured,
        overwrite=False,
        slt_type="CUSTOM_HAMILTONIAN",
    )

    assert group.type == "CUSTOM_HAMILTONIAN"


def test_write_hamiltonian_reader_result_to_slt_group_with_encoding(
    tmp_path: Path,
) -> None:
    structured = _diagonal_result()

    slt = create_slt_file(tmp_path / "ham.slt")

    group = write_hamiltonian_reader_result_to_slt_group(
        slt,
        "ham_group",
        structured,
        overwrite=False,
        encoding={"states_energies": {"dtype": "float32"}},
    )

    dataset = group.to_dataset()
    try:
        assert dataset["states_energies"].dtype == np.float32
    finally:
        dataset.close()


def test_write_hamiltonian_reader_result_to_slt_group_rejects_duplicate(
    tmp_path: Path,
) -> None:
    structured = _diagonal_result()

    slt = create_slt_file(tmp_path / "ham.slt")

    write_hamiltonian_reader_result_to_slt_group(
        slt,
        "ham_group",
        structured,
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        write_hamiltonian_reader_result_to_slt_group(
            slt,
            "ham_group",
            structured,
            overwrite=False,
        )


def test_write_hamiltonian_reader_result_to_slt_group_overwrite(
    tmp_path: Path,
) -> None:
    slt = create_slt_file(tmp_path / "ham.slt")

    first = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 1.0], dtype=np.float64),
        attrs={"shift_energies_applied": False},
    )
    second = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        states_energies=np.array([0.0, 2.0], dtype=np.float64),
        attrs={"shift_energies_applied": False},
    )

    write_hamiltonian_reader_result_to_slt_group(
        slt,
        "ham_group",
        first,
        overwrite=False,
    )
    group = write_hamiltonian_reader_result_to_slt_group(
        slt,
        "ham_group",
        second,
        overwrite=True,
    )

    xr_obj = group.to_xarray()
    try:
        assert isinstance(xr_obj, xr.DataArray)
        assert xr_obj.values.tolist() == [0.0, 2.0]
    finally:
        xr_obj.close()
