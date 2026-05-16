from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import slothpy.io.readers.molcas_hamiltonian_reader as molcas_mod
from slothpy.core.slt import SltFile, SltGroup, create_slt_file, open_slt_file
from slothpy.io.readers.hamiltonian_reader import (
    HamiltonianReader,
    HamiltonianReaderOptions,
    hamiltonian_reader_result_to_slt_results,
)
from slothpy.io.readers.molcas_hamiltonian_reader import (
    MolcasHamiltonianReader,
    MolcasHamiltonianReaderOptions,
    hamiltonian_from_molcas,
)

# ---------------------------------------------------------------------------
# Fixtures/helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def molcas_h5_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.rassi.h5"
    _write_minimal_molcas_h5(path)
    return path


def _matrix_stack(
    real_offset: float, imag_offset: float, dim: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    real = np.arange(3 * dim * dim, dtype=np.float64).reshape(3, dim, dim)
    imag = np.arange(3 * dim * dim, dtype=np.float64).reshape(3, dim, dim)
    return real + real_offset, imag + imag_offset


def _write_minimal_molcas_h5(path: Path) -> None:
    energies = np.array([2.0, 5.0], dtype=np.float64)

    spin_real, spin_imag = _matrix_stack(100.0, 200.0)
    ang_real, ang_imag = _matrix_stack(300.0, 350.0)
    edip_real, edip_imag = _matrix_stack(400.0, 500.0)

    with h5py.File(path, "w") as h5:
        h5.attrs["MOLCAS_MODULE"] = "RASSI"
        h5.attrs["MOLCAS_VERSION"] = "24.10"
        h5.attrs["NSTATE"] = np.int64(2)
        h5.attrs["NSYM"] = np.int64(1)

        h5.create_dataset("SOS_ENERGIES", data=energies)

        h5.create_dataset("SOS_SPIN_REAL", data=spin_real)
        h5.create_dataset("SOS_SPIN_IMAG", data=spin_imag)

        h5.create_dataset("SOS_ANGMOM_REAL", data=ang_real)
        h5.create_dataset("SOS_ANGMOM_IMAG", data=ang_imag)

        h5.create_dataset("SOS_EDIPMOM_REAL", data=edip_real)
        h5.create_dataset("SOS_EDIPMOM_IMAG", data=edip_imag)

        h5.create_dataset("STATE_SPINMULT", data=np.array([2, 2], dtype=np.int64))
        h5.create_dataset("STATE_IRREPS", data=np.array([1, 1], dtype=np.int64))
        h5.create_dataset("STATE_LROOT", data=np.array([1, 2], dtype=np.int64))


def _safe_close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


# ---------------------------------------------------------------------------
# Options/base-reader tests
# ---------------------------------------------------------------------------


def test_molcas_reader_options_are_dataclass_and_base_options() -> None:
    options = MolcasHamiltonianReaderOptions()

    assert is_dataclass(options)
    assert isinstance(options, HamiltonianReaderOptions)
    assert options.parse_ci_expansions is False
    assert options.shift_energies is False
    assert options.include_spin_matrices is False
    assert options.include_angular_momentum_matrices is False
    assert options.include_electric_dipole_moment_matrices is False
    assert options.ci_basis is False


def test_molcas_reader_is_hamiltonian_reader_subclass() -> None:
    reader = MolcasHamiltonianReader()

    assert isinstance(reader, HamiltonianReader)


# ---------------------------------------------------------------------------
# Reader behavior
# ---------------------------------------------------------------------------


def test_reader_reads_diagonal_hamiltonian_with_default_options(
    molcas_h5_path: Path,
) -> None:
    reader = MolcasHamiltonianReader()

    result = reader.read(molcas_h5_path)

    assert result.hamiltonian_interaction == "SOC"
    assert result.representation == "DIAGONAL"
    assert result.hamiltonian_matrix is None
    assert result.states_energies is not None
    np.testing.assert_allclose(result.states_energies, np.array([2.0, 5.0]))

    assert result.spin_matrices is None
    assert result.angular_momentum_matrices is None
    assert result.electric_dipole_moment_matrices is None
    assert result.ci_expansions_by_multiplicity is None

    assert result.attrs["slt_kind"] == "MOLCAS"
    assert result.attrs["hamiltonian_type"] == "SOC"
    assert result.attrs["basis"] == "DIAGONAL"
    assert result.attrs["states"] == 2
    assert result.attrs["source_format"] == "OpenMolcas/MOLCAS RASSI HDF5 output"
    assert result.attrs["shift_energies_applied"] is False


def test_reader_can_shift_energies(molcas_h5_path: Path) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(shift_energies=True)
    )

    result = reader.read(molcas_h5_path)

    assert result.states_energies is not None
    np.testing.assert_allclose(result.states_energies, np.array([0.0, 3.0]))
    assert result.attrs["shift_energies_applied"] is True


def test_reader_can_read_spin_matrices(molcas_h5_path: Path) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(include_spin_matrices=True)
    )

    result = reader.read(molcas_h5_path)

    assert result.spin_matrices is not None

    with h5py.File(molcas_h5_path, "r") as h5:
        expected = h5["SOS_SPIN_REAL"][:] + 1j * h5["SOS_SPIN_IMAG"][:]

    np.testing.assert_allclose(result.spin_matrices, expected)


def test_reader_can_read_angular_momentum_matrices(molcas_h5_path: Path) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(include_angular_momentum_matrices=True)
    )

    result = reader.read(molcas_h5_path)

    assert result.angular_momentum_matrices is not None
    assert result.angular_momentum_matrices.shape == (3, 2, 2)
    assert np.iscomplexobj(result.angular_momentum_matrices)


def test_reader_can_read_electric_dipole_moment_matrices(
    molcas_h5_path: Path,
) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(
            include_electric_dipole_moment_matrices=True,
        )
    )

    result = reader.read(molcas_h5_path)

    assert result.electric_dipole_moment_matrices is not None

    with h5py.File(molcas_h5_path, "r") as h5:
        expected = h5["SOS_EDIPMOM_REAL"][:] + 1j * h5["SOS_EDIPMOM_IMAG"][:]

    np.testing.assert_allclose(result.electric_dipole_moment_matrices, expected)


def test_reader_can_read_all_optional_operator_matrices(
    molcas_h5_path: Path,
) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(
            include_spin_matrices=True,
            include_angular_momentum_matrices=True,
            include_electric_dipole_moment_matrices=True,
        )
    )

    result = reader.read(molcas_h5_path)

    assert result.spin_matrices is not None
    assert result.angular_momentum_matrices is not None
    assert result.electric_dipole_moment_matrices is not None
    assert result.spin_matrices.shape == (3, 2, 2)
    assert result.angular_momentum_matrices.shape == (3, 2, 2)
    assert result.electric_dipole_moment_matrices.shape == (3, 2, 2)


def test_reader_result_can_be_composed_to_slt_results(
    molcas_h5_path: Path,
) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(
            shift_energies=True,
            include_spin_matrices=True,
            include_angular_momentum_matrices=True,
        )
    )

    result = reader.read(molcas_h5_path)
    composed = hamiltonian_reader_result_to_slt_results(result)

    assert composed.slt_type == "HAMILTONIAN"
    assert composed.primary == "states_energies"
    assert isinstance(composed.dataset, xr.Dataset)
    assert "states_energies" in composed.dataset
    assert "spin_matrices" in composed.dataset
    assert "angular_momentum_matrices" in composed.dataset
    assert "electric_dipole_moment_matrices" not in composed.dataset


def test_reader_rejects_ci_basis(molcas_h5_path: Path) -> None:
    reader = MolcasHamiltonianReader(MolcasHamiltonianReaderOptions(ci_basis=True))

    with pytest.raises(NotImplementedError):
        reader.read(molcas_h5_path)


def test_reader_rejects_ci_expansion_parsing(molcas_h5_path: Path) -> None:
    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(parse_ci_expansions=True)
    )

    with pytest.raises(NotImplementedError):
        reader.read(molcas_h5_path)


def test_reader_raises_when_required_energy_dataset_is_missing(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_ENERGIES"]

    reader = MolcasHamiltonianReader()

    with pytest.raises(KeyError):
        reader.read(molcas_h5_path)


def test_reader_raises_when_requested_spin_dataset_is_missing(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_SPIN_REAL"]

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(include_spin_matrices=True)
    )

    with pytest.raises(KeyError):
        reader.read(molcas_h5_path)


def test_reader_raises_when_requested_angular_momentum_dataset_is_missing(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_ANGMOM_IMAG"]

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(include_angular_momentum_matrices=True)
    )

    with pytest.raises(KeyError):
        reader.read(molcas_h5_path)


def test_reader_raises_when_requested_electric_dipole_dataset_is_missing(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_EDIPMOM_IMAG"]

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(
            include_electric_dipole_moment_matrices=True,
        )
    )

    with pytest.raises(KeyError):
        reader.read(molcas_h5_path)


# ---------------------------------------------------------------------------
# Public function tests
# ---------------------------------------------------------------------------


def test_hamiltonian_from_molcas_creates_new_slt_file(
    molcas_h5_path: Path,
    tmp_path: Path,
) -> None:
    slt_path = tmp_path / "molcas.slt"

    slt = hamiltonian_from_molcas(
        molcas_h5_path,
        slt_path,
        "molcas_hamiltonian",
        overwrite=True,
    )

    assert isinstance(slt, SltFile)
    assert slt.path == slt_path
    assert slt.path.exists()
    assert "molcas_hamiltonian" in slt

    group = slt["molcas_hamiltonian"]
    assert isinstance(group, SltGroup)

    ds = group.to_dataset()
    try:
        assert "states_energies" in ds
        np.testing.assert_allclose(
            ds["states_energies"].values,
            np.array([0.0, 3.0]),
        )
    finally:
        ds.close()


def test_hamiltonian_from_molcas_opens_existing_slt_file(
    molcas_h5_path: Path,
    tmp_path: Path,
) -> None:
    slt_path = tmp_path / "existing.slt"
    create_slt_file(slt_path, overwrite=True)

    slt = hamiltonian_from_molcas(
        molcas_h5_path,
        slt_path,
        "molcas_hamiltonian",
        overwrite=True,
    )

    reopened = open_slt_file(slt_path)

    assert slt.path == reopened.path
    assert "molcas_hamiltonian" in reopened


def test_hamiltonian_from_molcas_writes_requested_optional_matrices(
    molcas_h5_path: Path,
    tmp_path: Path,
) -> None:
    slt_path = tmp_path / "molcas_with_operators.slt"

    slt = hamiltonian_from_molcas(
        molcas_h5_path,
        slt_path,
        "molcas_hamiltonian",
        include_spin_matrices=True,
        include_angular_momentum_matrices=True,
        include_electric_dipole_moment_matrices=True,
        overwrite=True,
    )

    group = slt["molcas_hamiltonian"]
    assert isinstance(group, SltGroup)

    ds = group.to_dataset()
    try:
        assert "states_energies" in ds
        assert "spin_matrices" in ds
        assert "angular_momentum_matrices" in ds
        assert "electric_dipole_moment_matrices" in ds
    finally:
        ds.close()


def test_hamiltonian_from_molcas_rejects_duplicate_group_without_overwrite(
    molcas_h5_path: Path,
    tmp_path: Path,
) -> None:
    slt_path = tmp_path / "duplicate.slt"

    hamiltonian_from_molcas(
        molcas_h5_path,
        slt_path,
        "molcas_hamiltonian",
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        hamiltonian_from_molcas(
            molcas_h5_path,
            slt_path,
            "molcas_hamiltonian",
            overwrite=False,
        )


def test_hamiltonian_from_molcas_rejects_non_string_group_name(
    molcas_h5_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        hamiltonian_from_molcas(
            molcas_h5_path,
            tmp_path / "bad_group_name.slt",
            123,  # type: ignore[arg-type]
        )


def test_hamiltonian_from_molcas_propagates_not_implemented_options(
    molcas_h5_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(NotImplementedError):
        hamiltonian_from_molcas(
            molcas_h5_path,
            tmp_path / "ci_basis.slt",
            "molcas_hamiltonian",
            ci_basis=True,
            overwrite=True,
        )

    with pytest.raises(NotImplementedError):
        hamiltonian_from_molcas(
            molcas_h5_path,
            tmp_path / "ci_expansions.slt",
            "molcas_hamiltonian",
            parse_ci_expansions=True,
            overwrite=True,
        )


# ---------------------------------------------------------------------------
# Internal helper / edge-case coverage
# ---------------------------------------------------------------------------


def test_decode_hdf5_attr_decodes_bytes() -> None:
    assert molcas_mod._decode_hdf5_attr(b"  RASSI  ") == "RASSI"
    assert molcas_mod._decode_hdf5_attr(np.bytes_(b"MOLCAS")) == "MOLCAS"
    assert molcas_mod._decode_hdf5_attr(42) == 42


def test_reader_reads_optional_rassi_file_attrs(molcas_h5_path: Path) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        h5.attrs["MOLCAS_MODULE"] = np.bytes_(b"  RASSI  ")
        h5.attrs["STATE_SPINMULT"] = np.array([2, 2], dtype=np.int64)
        h5.attrs["STATE_IRREPS"] = np.array([1, 1], dtype=np.int64)
        h5.attrs["STATE_LROOT"] = np.array([1, 2], dtype=np.int64)

    reader = MolcasHamiltonianReader()
    result = reader.read(molcas_h5_path)

    assert result.attrs["molcas_module"] == "RASSI"
    assert result.attrs["nstate"] == 2
    assert result.attrs["nsym"] == 1
    np.testing.assert_array_equal(result.attrs["state_spinmult"], [2, 2])
    np.testing.assert_array_equal(result.attrs["state_irreps"], [1, 1])
    np.testing.assert_array_equal(result.attrs["state_lroot"], [1, 2])


def test_read_required_dataset_raises_when_name_is_not_a_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "group_instead_of_dataset.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("SOS_ENERGIES")

    with h5py.File(path, "r") as h5:
        with pytest.raises(TypeError, match="is not a dataset"):
            molcas_mod._read_required_dataset(h5, "SOS_ENERGIES")


def test_reader_raises_when_energies_are_not_one_dimensional(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_ENERGIES"]
        h5.create_dataset("SOS_ENERGIES", data=np.array([[1.0, 2.0]], dtype=np.float64))

    reader = MolcasHamiltonianReader()

    with pytest.raises(ValueError, match="must be a 1D array"):
        reader.read(molcas_h5_path)


def test_reader_raises_when_energies_are_empty(molcas_h5_path: Path) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_ENERGIES"]
        h5.create_dataset("SOS_ENERGIES", data=np.array([], dtype=np.float64))

    reader = MolcasHamiltonianReader()

    with pytest.raises(ValueError, match="SOS_ENERGIES is empty"):
        reader.read(molcas_h5_path)


def test_reader_raises_when_spin_real_and_imag_shapes_differ(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_SPIN_IMAG"]
        h5.create_dataset(
            "SOS_SPIN_IMAG",
            data=np.zeros((3, 3, 3), dtype=np.float64),
        )

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(include_spin_matrices=True)
    )

    with pytest.raises(ValueError, match="must have the same shape"):
        reader.read(molcas_h5_path)


def test_read_complex_operator_stack_raises_when_shapes_differ(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shape_mismatch.h5"
    _write_minimal_molcas_h5(path)
    with h5py.File(path, "a") as h5:
        del h5["SOS_SPIN_IMAG"]
        h5.create_dataset(
            "SOS_SPIN_IMAG",
            data=np.zeros((3, 3, 3), dtype=np.float64),
        )

    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="must have the same shape"):
            molcas_mod._read_complex_operator_stack(
                h5,
                real_name="SOS_SPIN_REAL",
                imag_name="SOS_SPIN_IMAG",
                output_name="spin_matrices",
                dim=2,
            )


def test_reader_raises_when_operator_stack_has_wrong_shape(
    molcas_h5_path: Path,
) -> None:
    with h5py.File(molcas_h5_path, "a") as h5:
        del h5["SOS_SPIN_REAL"]
        del h5["SOS_SPIN_IMAG"]
        h5.create_dataset(
            "SOS_SPIN_REAL",
            data=np.zeros((3, 3, 3), dtype=np.float64),
        )
        h5.create_dataset(
            "SOS_SPIN_IMAG",
            data=np.zeros((3, 3, 3), dtype=np.float64),
        )

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(include_spin_matrices=True)
    )

    with pytest.raises(ValueError, match=r"spin_matrices must have shape \(3, 2, 2\)"):
        reader.read(molcas_h5_path)


def test_validate_operator_stack_raises_for_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"test_op must have shape \(3, 2, 2\)"):
        molcas_mod._validate_operator_stack(
            "test_op",
            np.zeros((3, 3, 3), dtype=np.complex128),
            2,
        )


# ---------------------------------------------------------------------------
# Real MOLCAS/OpenMolcas file regression tests
# ---------------------------------------------------------------------------


def _real_molcas_test_file() -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "molcas"

    candidates = sorted(data_dir.glob("*.rassi.h5"))
    if not candidates:
        candidates = sorted(data_dir.glob("*.h5"))

    if not candidates:
        pytest.skip(
            "No real MOLCAS/OpenMolcas HDF5 test file found under tests/data/molcas."
        )

    return candidates[0]


def test_reader_handles_real_molcas_hdf5_file() -> None:
    molcas_path = _real_molcas_test_file()

    with h5py.File(molcas_path, "r") as h5:
        expected_energies = np.asarray(h5["SOS_ENERGIES"][:], dtype=np.float64)
        dim = int(expected_energies.shape[0])

        has_spin = "SOS_SPIN_REAL" in h5 and "SOS_SPIN_IMAG" in h5
        has_angular_momentum = "SOS_ANGMOM_REAL" in h5 and "SOS_ANGMOM_IMAG" in h5
        has_electric_dipole = "SOS_EDIPMOM_REAL" in h5 and "SOS_EDIPMOM_IMAG" in h5

        expected_spin = (
            h5["SOS_SPIN_REAL"][:] + 1j * h5["SOS_SPIN_IMAG"][:] if has_spin else None
        )
        expected_electric = (
            h5["SOS_EDIPMOM_REAL"][:] + 1j * h5["SOS_EDIPMOM_IMAG"][:]
            if has_electric_dipole
            else None
        )

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(
            shift_energies=True,
            include_spin_matrices=has_spin,
            include_angular_momentum_matrices=has_angular_momentum,
            include_electric_dipole_moment_matrices=has_electric_dipole,
        )
    )

    result = reader.read(molcas_path)

    assert result.hamiltonian_interaction == "SOC"
    assert result.representation == "DIAGONAL"
    assert result.hamiltonian_matrix is None
    assert result.states_energies is not None

    np.testing.assert_allclose(
        result.states_energies,
        expected_energies - np.min(expected_energies),
    )

    assert result.attrs["slt_kind"] == "MOLCAS"
    assert result.attrs["hamiltonian_type"] == "SOC"
    assert result.attrs["basis"] == "DIAGONAL"
    assert result.attrs["states"] == dim
    assert result.attrs["shift_energies_applied"] is True

    if has_spin:
        assert result.spin_matrices is not None
        assert result.spin_matrices.shape == (3, dim, dim)
        assert expected_spin is not None
        np.testing.assert_allclose(result.spin_matrices, expected_spin)
    else:
        assert result.spin_matrices is None

    if has_angular_momentum:
        assert result.angular_momentum_matrices is not None
        assert result.angular_momentum_matrices.shape == (3, dim, dim)
    else:
        assert result.angular_momentum_matrices is None

    if has_electric_dipole:
        assert result.electric_dipole_moment_matrices is not None
        assert result.electric_dipole_moment_matrices.shape == (3, dim, dim)
        assert expected_electric is not None
        np.testing.assert_allclose(
            result.electric_dipole_moment_matrices,
            expected_electric,
        )
    else:
        assert result.electric_dipole_moment_matrices is None


def test_hamiltonian_from_molcas_writes_real_molcas_file(
    tmp_path: Path,
) -> None:
    molcas_path = _real_molcas_test_file()
    slt_path = tmp_path / "real_molcas.slt"

    with h5py.File(molcas_path, "r") as h5:
        expected_energies = np.asarray(h5["SOS_ENERGIES"][:], dtype=np.float64)

        has_spin = "SOS_SPIN_REAL" in h5 and "SOS_SPIN_IMAG" in h5
        has_angular_momentum = "SOS_ANGMOM_REAL" in h5 and "SOS_ANGMOM_IMAG" in h5
        has_electric_dipole = "SOS_EDIPMOM_REAL" in h5 and "SOS_EDIPMOM_IMAG" in h5

    slt = hamiltonian_from_molcas(
        molcas_path,
        slt_path,
        "real_molcas_hamiltonian",
        shift_energies=True,
        include_spin_matrices=has_spin,
        include_angular_momentum_matrices=has_angular_momentum,
        include_electric_dipole_moment_matrices=has_electric_dipole,
        overwrite=True,
    )

    assert isinstance(slt, SltFile)
    assert slt.path == slt_path
    assert "real_molcas_hamiltonian" in slt

    group = slt["real_molcas_hamiltonian"]
    assert isinstance(group, SltGroup)

    dataset = group.to_dataset()
    try:
        assert "states_energies" in dataset
        np.testing.assert_allclose(
            dataset["states_energies"].values,
            expected_energies - np.min(expected_energies),
        )

        if has_spin:
            assert "spin_matrices" in dataset
        else:
            assert "spin_matrices" not in dataset

        if has_angular_momentum:
            assert "angular_momentum_matrices" in dataset
        else:
            assert "angular_momentum_matrices" not in dataset

        if has_electric_dipole:
            assert "electric_dipole_moment_matrices" in dataset
        else:
            assert "electric_dipole_moment_matrices" not in dataset
    finally:
        dataset.close()
