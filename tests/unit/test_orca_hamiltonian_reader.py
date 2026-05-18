from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import slothpy.io.readers.orca_hamiltonian_reader as orca_mod
from slothpy.core.slt import create_slt_file, open_slt_file
from slothpy.core.slt_file import SltFile
from slothpy.core.slt_group import SltGroup
from slothpy.io.readers.hamiltonian_reader import HamiltonianReader
from slothpy.io.readers.orca_hamiltonian_reader import (
    OrcaHamiltonianReader,
    hamiltonian_from_orca,
)


def test_orca_hamiltonian_reader_is_hamiltonian_reader_subclass() -> None:
    reader = OrcaHamiltonianReader()

    assert isinstance(reader, HamiltonianReader)


def test_hamiltonian_from_orca_rejects_unknown_kwargs() -> None:
    with pytest.raises(ValidationError):
        hamiltonian_from_orca(
            "x.out",
            "y.slt",
            "ham",
            not_a_valid_field=True,  # type: ignore[call-arg]
        )


def test_hamiltonian_from_orca_rejects_removed_reader_kw() -> None:
    with pytest.raises(ValidationError):
        hamiltonian_from_orca(
            "x.out",
            "y.slt",
            "ham",
            reader=OrcaHamiltonianReader(),  # type: ignore[call-arg]
        )


def test_hamiltonian_from_orca_rejects_removed_dipole_alias() -> None:
    with pytest.raises(ValidationError):
        hamiltonian_from_orca(
            "x.out",
            "y.slt",
            "ham",
            electric_dipole_moment_matrices=True,  # type: ignore[call-arg]
        )


def test_hamiltonian_from_orca_rejects_parse_ci_expansions_without_ci_basis() -> None:
    with pytest.raises(
        ValueError, match="parse_ci_expansions=True requires ci_basis=True"
    ):
        hamiltonian_from_orca(
            "x.out",
            "y.slt",
            "ham",
            parse_ci_expansions=True,
            ci_basis=False,
        )


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "orca"
REAL_ORCA_SOC_FILE = DATA_DIR / "Pr_minimal.out"


# ---------------------------------------------------------------------------
# Synthetic ORCA-output helpers
# ---------------------------------------------------------------------------


def _input_block(*, mult: str = "1", nroots: str = "2") -> list[str]:
    return [
        "INPUT FILE",
        "|  1> %casscf",
        f"|  2>   mult {mult}",
        f"|  3>   nroots {nroots}",
        "|  4> end",
        "|  5> ****END OF INPUT****",
    ]


def _dimension_lines(
    *,
    active_orbitals: int = 2,
    total_orbitals: int = 5,
    inactive_orbitals: int = 3,
) -> list[str]:
    return [
        f"Number of active orbitals     ... {active_orbitals}",
        f"Total number of orbitals      ... {total_orbitals}",
        "Determined orbital ranges:",
        f"Internal    0 -    2    (   {inactive_orbitals} orbitals)",
        "Active      3 -    4    (   2 orbitals)",
        "External    5 -    9    (   5 orbitals)",
        "Number of rotation parameters",
    ]


def _matrix_block(values: Any) -> list[str]:
    array = np.asarray(values, dtype=np.float64)
    assert array.ndim == 2
    assert array.shape[0] == array.shape[1]

    dim = int(array.shape[0])
    lines = ["    " + "    ".join(str(idx) for idx in range(dim))]

    for row in range(dim):
        row_values = "    ".join(f"{array[row, col]: .12E}" for col in range(dim))
        lines.append(f"{row:5d}    {row_values}")

    return lines


def _soc_section(
    *,
    real: Any | None = None,
    imag: Any | None = None,
    ssc: bool = False,
    image_typo: bool = False,
) -> list[str]:
    if real is None:
        real = np.diag([0.0, 1.0])
    real_array = np.asarray(real, dtype=np.float64)

    if imag is None:
        imag = np.zeros_like(real_array)

    header = "SOC and SSC MATRIX (A.U.)" if ssc else "SOC MATRIX (A.U.)"
    imag_header = "Image part:" if image_typo else "Imag part:"

    return [
        header,
        "Real part:",
        *_matrix_block(real_array),
        imag_header,
        *_matrix_block(imag),
    ]


def _operator_section(label: str, values: Any | None = None) -> list[str]:
    if values is None:
        values = np.eye(2)

    return [
        label,
        "Real part:",
        *_matrix_block(values),
    ]


def _electric_dipole_section(label: str, values: Any | None = None) -> list[str]:
    if values is None:
        values = np.eye(2)

    return [
        label,
        *_matrix_block(values),
    ]


def _ci_block() -> list[str]:
    return [
        "Spin-Determinant CI Printing",
        "ROOT 0: E= -1.0",
        "",
        "[u0]     8.000000D-01",
        "[0d]     6.000000D-01",
        "ROOT 1: E= -0.5",
        "",
        "[u0]    -6.000000D-01",
        "[0d]     8.000000D-01",
    ]


def _minimal_orca_lines(
    *,
    mult: str = "1",
    nroots: str = "2",
    dim: int = 2,
    soc_real: Any | None = None,
    soc_imag: Any | None = None,
    ssc: bool = False,
    pt2: bool = False,
    include_spin: bool = False,
    include_angular_momentum: bool = False,
    include_electric_dipole: bool = False,
    include_ci: bool = False,
) -> list[str]:
    if soc_real is None:
        soc_real = np.diag(np.arange(dim, dtype=np.float64))
    if soc_imag is None:
        soc_imag = np.zeros((dim, dim), dtype=np.float64)

    lines: list[str] = []
    lines.extend(_input_block(mult=mult, nroots=nroots))
    lines.extend(_dimension_lines())

    if include_ci:
        lines.extend(_ci_block())

    if pt2:
        lines.extend(
            _soc_section(
                real=np.diag(np.arange(dim, dtype=np.float64) + 10.0),
                imag=np.zeros((dim, dim), dtype=np.float64),
                ssc=ssc,
            )
        )

    lines.extend(_soc_section(real=soc_real, imag=soc_imag, ssc=ssc))

    if include_spin:
        for label in orca_mod._SPIN_LABELS:
            lines.extend(_operator_section(label, np.eye(dim)))

    if include_angular_momentum:
        for label in orca_mod._ANGULAR_MOMENTUM_LABELS:
            lines.extend(_operator_section(label, np.eye(dim)))

    if include_electric_dipole:
        for label in orca_mod._ELECTRIC_DIPOLE_LABELS:
            lines.extend(_electric_dipole_section(label, np.eye(dim)))

    return lines


def _reader_without_optional_operators(
    **options: Any,
) -> orca_mod.OrcaHamiltonianReader:
    defaults = {
        "include_spin_matrices": False,
        "include_angular_momentum_matrices": False,
        "include_electric_dipole_moment_matrices": False,
    }
    defaults.update(options)
    return orca_mod.OrcaHamiltonianReader(
        orca_mod.OrcaHamiltonianReaderOptions(**defaults)
    )


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Options and class structure
# ---------------------------------------------------------------------------


def test_orca_reader_options_defaults() -> None:
    options = orca_mod.OrcaHamiltonianReaderOptions()

    assert isinstance(options, orca_mod.HamiltonianReaderOptions)
    assert options.parse_ci_expansions is False
    assert options.shift_energies is False
    assert options.include_spin_matrices is False
    assert options.include_angular_momentum_matrices is False
    assert options.include_electric_dipole_moment_matrices is False
    assert options.ci_basis is False
    assert options.pt2 is False
    assert options.ssc is False


def test_orca_reader_is_hamiltonian_reader_subclass() -> None:
    reader = orca_mod.OrcaHamiltonianReader()

    assert isinstance(reader, orca_mod.HamiltonianReader)


# ---------------------------------------------------------------------------
# Line-stream and source helpers
# ---------------------------------------------------------------------------


def test_line_stream_iteration_and_push_back() -> None:
    stream = orca_mod._LineStream(["a", "b"])

    assert iter(stream) is stream
    assert next(stream) == "a"

    stream.push_back("x")

    assert next(stream) == "x"
    assert list(stream) == ["b"]


def test_iter_source_lines_from_path_and_iterable(tmp_path: Path) -> None:
    source = tmp_path / "orca.out"
    source.write_text("a\r\nb\nc")

    assert list(orca_mod._iter_source_lines(source)) == ["a", "b", "c"]
    assert list(orca_mod._iter_source_lines(["x\n", "y\r\n"])) == ["x", "y"]


def test_strip_orca_input_prefix() -> None:
    assert orca_mod._strip_orca_input_prefix("|  12>   mult 1 3") == "mult 1 3"
    assert orca_mod._strip_orca_input_prefix("plain line") == "plain line"


# ---------------------------------------------------------------------------
# Input-block parsing
# ---------------------------------------------------------------------------


def test_parse_input_block_success() -> None:
    stream = orca_mod._LineStream(_input_block(mult="1 3", nroots="2 1")[1:])

    multiplicities, nroots = orca_mod._parse_input_block(stream)

    assert multiplicities.tolist() == [1, 3]
    assert nroots.tolist() == [2, 1]


def test_parse_input_block_missing_end_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "|  1> %casscf",
            "|  2> mult 1",
            "|  3> nroots 2",
        ]
    )

    with pytest.raises(ValueError, match="Unexpected end"):
        orca_mod._parse_input_block(stream)


def test_parse_input_block_missing_casscf_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "|  1> %scf",
            "|  2> end",
            "|  3> ****END OF INPUT****",
        ]
    )

    with pytest.raises(ValueError, match="%casscf"):
        orca_mod._parse_input_block(stream)


def test_parse_input_block_missing_mult_or_nroots_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "|  1> %casscf",
            "|  2> mult 1",
            "|  3> end",
            "|  4> ****END OF INPUT****",
        ]
    )

    with pytest.raises(ValueError, match="multiplicities and nroots"):
        orca_mod._parse_input_block(stream)


def test_parse_input_block_mismatched_mult_and_nroots_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "|  1> %casscf",
            "|  2> mult 1 3",
            "|  3> nroots 2",
            "|  4> end",
            "|  5> ****END OF INPUT****",
        ]
    )

    with pytest.raises(ValueError, match="different lengths"):
        orca_mod._parse_input_block(stream)


def test_parse_input_block_stops_at_next_percent_section() -> None:
    stream = orca_mod._LineStream(
        [
            "|  1> %casscf",
            "|  2> mult 1",
            "|  3> nroots 2",
            "|  4> %pal",
            "|  5> nprocs 16",
            "|  6> end",
            "|  7> ****END OF INPUT****",
        ]
    )

    multiplicities, nroots = orca_mod._parse_input_block(stream)

    assert multiplicities.tolist() == [1]
    assert nroots.tolist() == [2]


# ---------------------------------------------------------------------------
# Dimension-section parsing
# ---------------------------------------------------------------------------


def test_parse_inactive_orbitals_success() -> None:
    stream = orca_mod._LineStream(
        [
            "",
            "Active      3 -    4    (   2 orbitals)",
            "Internal    0 -    2    (   3 orbitals)",
        ]
    )

    assert orca_mod._parse_inactive_orbitals(stream) == 3


def test_parse_inactive_orbitals_failure() -> None:
    stream = orca_mod._LineStream(
        [
            "",
            "Active      3 -    4    (   2 orbitals)",
            "External    5 -    9    (   5 orbitals)",
            "Number of rotation parameters",
        ]
    )

    with pytest.raises(ValueError, match="inactive/internal"):
        orca_mod._parse_inactive_orbitals(stream)


# ---------------------------------------------------------------------------
# Generic stream search helpers
# ---------------------------------------------------------------------------


def test_consume_until_success() -> None:
    stream = orca_mod._LineStream(["a", "Real part:", "b"])

    assert (
        orca_mod._consume_until(stream, orca_mod.compile(r"Real part:")) == "Real part:"
    )


def test_consume_until_failure() -> None:
    stream = orca_mod._LineStream(["a", "b"])

    with pytest.raises(ValueError, match="Could not find ORCA output section"):
        orca_mod._consume_until(stream, orca_mod.compile(r"missing"))


def test_next_nonempty_line_success_and_failure() -> None:
    stream = orca_mod._LineStream(["", "   ", "value"])

    assert orca_mod._next_nonempty_line(stream) == "value"

    with pytest.raises(ValueError, match="Unexpected end"):
        orca_mod._next_nonempty_line(orca_mod._LineStream(["", "   "]))


def test_consume_until_root_success_and_failure() -> None:
    stream = orca_mod._LineStream(["ROOT 0: E= -1.0"])

    orca_mod._consume_until_root(stream, expected_root=0, multiplicity=1)

    with pytest.raises(RuntimeError, match="Could not find ROOT"):
        orca_mod._consume_until_root(
            orca_mod._LineStream(["ROOT 1: E= -1.0"]),
            expected_root=0,
            multiplicity=1,
        )


# ---------------------------------------------------------------------------
# Matrix parsing
# ---------------------------------------------------------------------------


def test_is_column_header() -> None:
    assert orca_mod._is_column_header("    0    1    2")
    assert not orca_mod._is_column_header("Real part:")


def test_read_block_matrix_from_stream_success() -> None:
    stream = orca_mod._LineStream(_matrix_block([[1.0, 2.0], [3.0, 4.0]]))

    matrix = orca_mod._read_block_matrix_from_stream(
        stream,
        dim=2,
        fix_negative_overlap=False,
    )

    np.testing.assert_allclose(matrix, [[1.0, 2.0], [3.0, 4.0]])


def test_read_block_matrix_from_stream_accepts_fortran_d_exponents() -> None:
    stream = orca_mod._LineStream(
        [
            "    0    1",
            "    0    1.0D+00    2.0D+00",
            "    1    3.0D+00    4.0D+00",
        ]
    )

    matrix = orca_mod._read_block_matrix_from_stream(
        stream,
        dim=2,
        fix_negative_overlap=False,
    )

    np.testing.assert_allclose(matrix, [[1.0, 2.0], [3.0, 4.0]])


def test_read_block_matrix_from_stream_fixes_negative_overlap() -> None:
    stream = orca_mod._LineStream(
        [
            "    0    1",
            "    0    1.0-2.0",
            "    1    3.0-4.0",
        ]
    )

    matrix = orca_mod._read_block_matrix_from_stream(
        stream,
        dim=2,
        fix_negative_overlap=True,
    )

    np.testing.assert_allclose(matrix, [[1.0, -2.0], [3.0, -4.0]])


def test_read_block_matrix_from_stream_unexpected_eof_raises() -> None:
    with pytest.raises(ValueError, match="Unexpected end"):
        orca_mod._read_block_matrix_from_stream(
            orca_mod._LineStream([]),
            dim=2,
            fix_negative_overlap=False,
        )


def test_read_block_matrix_from_stream_empty_header_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orca_mod, "_is_column_header", lambda line: True)

    with pytest.raises(ValueError, match="Empty ORCA matrix column header"):
        orca_mod._read_block_matrix_from_stream(
            orca_mod._LineStream([""]),
            dim=2,
            fix_negative_overlap=False,
        )


def test_read_block_matrix_from_stream_incomplete_block_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "    0    1",
            "    0    1.0    2.0",
        ]
    )

    with pytest.raises(ValueError, match="Incomplete"):
        orca_mod._read_block_matrix_from_stream(
            stream,
            dim=2,
            fix_negative_overlap=False,
        )


def test_read_block_matrix_from_stream_wrong_float_count_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "    0    1",
            "    0    1.0",
            "    1    2.0",
        ]
    )

    with pytest.raises(ValueError, match="Expected 4 floats"):
        orca_mod._read_block_matrix_from_stream(
            stream,
            dim=2,
            fix_negative_overlap=False,
        )


def test_read_soc_matrix_from_stream_success_with_image_typo() -> None:
    stream = orca_mod._LineStream(
        [
            "Real part:",
            *_matrix_block([[1.0, 0.0], [0.0, 2.0]]),
            "Image part:",
            *_matrix_block([[0.0, 3.0], [-3.0, 0.0]]),
        ]
    )

    matrix = orca_mod._read_soc_matrix_from_stream(stream, dim=2)

    np.testing.assert_allclose(matrix.real, [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(matrix.imag, [[0.0, 3.0], [-3.0, 0.0]])


def test_read_electric_dipole_matrix_from_stream() -> None:
    stream = orca_mod._LineStream(_matrix_block([[1.0, 2.0], [3.0, 4.0]]))

    matrix = orca_mod._read_electric_dipole_matrix_from_stream(stream, dim=2)

    assert matrix.dtype == np.complex128
    np.testing.assert_allclose(matrix.real, [[1.0, 2.0], [3.0, 4.0]])


def test_read_vector_operator_matrix_spin_sx() -> None:
    stream = orca_mod._LineStream(
        [
            "Real part:",
            *_matrix_block([[2.0, 0.0], [0.0, 2.0]]),
        ]
    )

    matrix = orca_mod._read_vector_operator_matrix_from_stream(
        stream,
        dim=2,
        label="SX MATRIX IN CI BASIS",
        operator="spin",
    )

    np.testing.assert_allclose(matrix, np.eye(2))


def test_read_vector_operator_matrix_spin_sy_phase() -> None:
    stream = orca_mod._LineStream(
        [
            "Real part:",
            *_matrix_block([[2.0, 0.0], [0.0, 2.0]]),
        ]
    )

    matrix = orca_mod._read_vector_operator_matrix_from_stream(
        stream,
        dim=2,
        label="SY MATRIX IN CI BASIS",
        operator="spin",
    )

    np.testing.assert_allclose(matrix, 1j * np.eye(2))


def test_read_vector_operator_matrix_angular_momentum_phase() -> None:
    stream = orca_mod._LineStream(
        [
            "Real part:",
            *_matrix_block([[1.0, 0.0], [0.0, 1.0]]),
        ]
    )

    matrix = orca_mod._read_vector_operator_matrix_from_stream(
        stream,
        dim=2,
        label="LX MATRIX IN CI BASIS",
        operator="angular_momentum",
    )

    np.testing.assert_allclose(matrix, 1j * np.eye(2))


# ---------------------------------------------------------------------------
# CI determinant parsing
# ---------------------------------------------------------------------------


def test_decode_orca_determinant_occupations_all_symbols() -> None:
    alpha, beta = orca_mod._decode_orca_determinant_occupations("ud20")

    assert alpha.tolist() == [1, 0, 1, 0]
    assert beta.tolist() == [0, 1, 1, 0]


def test_decode_orca_determinant_occupations_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        orca_mod._decode_orca_determinant_occupations("ud", active_orbitals=3)


def test_decode_orca_determinant_occupations_unknown_character_raises() -> None:
    with pytest.raises(ValueError, match="Unknown determinant character"):
        orca_mod._decode_orca_determinant_occupations("ux")


def test_parse_orca_spin_determinant_ci_block_success() -> None:
    stream = orca_mod._LineStream(_ci_block()[1:] + ["AFTER"])

    expansion = orca_mod._parse_orca_spin_determinant_ci_block(
        stream,
        multiplicity=1,
        nroots=2,
        active_orbitals=2,
    )

    assert expansion.alpha_occupations.tolist() == [[1, 0], [0, 0]]
    assert expansion.beta_occupations.tolist() == [[0, 0], [0, 1]]
    np.testing.assert_allclose(
        expansion.ci_coefficients,
        [[0.8, -0.6], [0.6, 0.8]],
    )
    assert next(stream) == "AFTER"


def test_parse_orca_spin_determinant_ci_block_inconsistent_determinants_raises() -> (
    None
):
    stream = orca_mod._LineStream(
        [
            "ROOT 0: E= -1.0",
            "[u0]     1.0",
            "ROOT 1: E= -0.5",
            "[0d]     1.0",
        ]
    )

    with pytest.raises(ValueError, match="Inconsistent determinant list"):
        orca_mod._parse_orca_spin_determinant_ci_block(
            stream,
            multiplicity=1,
            nroots=2,
            active_orbitals=2,
        )


def test_parse_orca_spin_determinant_ci_block_missing_root_raises() -> None:
    stream = orca_mod._LineStream(
        [
            "ROOT 0: E= -1.0",
            "[u0]     1.0",
        ]
    )

    with pytest.raises(RuntimeError, match="Could not find ROOT"):
        orca_mod._parse_orca_spin_determinant_ci_block(
            stream,
            multiplicity=1,
            nroots=2,
            active_orbitals=2,
        )


def test_parse_orca_spin_determinant_ci_block_zero_roots_raises() -> None:
    with pytest.raises(RuntimeError, match="No CI determinants parsed"):
        orca_mod._parse_orca_spin_determinant_ci_block(
            orca_mod._LineStream([]),
            multiplicity=1,
            nroots=0,
            active_orbitals=2,
        )


# ---------------------------------------------------------------------------
# Operator stack helpers
# ---------------------------------------------------------------------------


def test_stack_required_operator_parts_success() -> None:
    parts = [np.eye(2), np.eye(2) * 2.0, np.eye(2) * 3.0]

    stack = orca_mod._stack_required_operator_parts("operator", parts)

    assert stack.shape == (3, 2, 2)


def test_stack_required_operator_parts_missing_raises() -> None:
    parts = [np.eye(2), None, np.eye(2)]

    with pytest.raises(ValueError, match="missing \\[1\\]"):
        orca_mod._stack_required_operator_parts("operator", parts)


def test_transform_operator_stack() -> None:
    matrices = np.stack([np.diag([1.0, 2.0])] * 3).astype(np.complex128)
    eigenvectors = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    transformed = orca_mod._transform_operator_stack(matrices, eigenvectors)

    np.testing.assert_allclose(transformed[0], np.diag([2.0, 1.0]))


# ---------------------------------------------------------------------------
# High-level reader behavior
# ---------------------------------------------------------------------------


def test_reader_parses_minimal_diagonal_hamiltonian() -> None:
    reader = _reader_without_optional_operators(shift_energies=True)

    result = reader.read(_minimal_orca_lines())

    assert result.hamiltonian_interaction == "SOC"
    assert result.representation == "DIAGONAL"
    assert result.state_energies is not None
    assert result.hamiltonian_matrix is None
    np.testing.assert_allclose(result.state_energies, [0.0, 1.0])
    assert result.attrs["shift_energies_applied"] is True


def test_reader_parses_unshifted_diagonal_hamiltonian() -> None:
    reader = _reader_without_optional_operators(shift_energies=False)

    result = reader.read(
        _minimal_orca_lines(
            soc_real=np.diag([5.0, 6.0]),
        )
    )

    assert result.state_energies is not None
    np.testing.assert_allclose(result.state_energies, [5.0, 6.0])
    assert result.attrs["shift_energies_applied"] is False


def test_reader_parses_ci_basis_hamiltonian() -> None:
    reader = _reader_without_optional_operators(ci_basis=True)

    result = reader.read(
        _minimal_orca_lines(
            soc_real=[[1.0, 2.0], [2.0, 4.0]],
        )
    )

    assert result.representation == "CI"
    assert result.hamiltonian_matrix is not None
    assert result.state_energies is None
    np.testing.assert_allclose(result.hamiltonian_matrix.real, [[1.0, 2.0], [2.0, 4.0]])


def test_reader_parses_ssc_hamiltonian() -> None:
    reader = _reader_without_optional_operators(ssc=True)

    result = reader.read(_minimal_orca_lines(ssc=True))

    assert result.hamiltonian_interaction == "SOC_SSC"
    assert result.attrs["hamiltonian_type"] == "SOC_SSC"


def test_reader_pt2_reads_second_soc_occurrence() -> None:
    reader = _reader_without_optional_operators(ci_basis=True, pt2=True)

    result = reader.read(
        _minimal_orca_lines(
            pt2=True,
            soc_real=[[1.0, 0.0], [0.0, 2.0]],
        )
    )

    assert result.hamiltonian_matrix is not None
    np.testing.assert_allclose(result.hamiltonian_matrix.real, [[1.0, 0.0], [0.0, 2.0]])


def test_reader_parses_optional_operator_stacks() -> None:
    reader = orca_mod.OrcaHamiltonianReader(
        orca_mod.OrcaHamiltonianReaderOptions(
            include_spin_matrices=True,
            include_angular_momentum_matrices=True,
            include_electric_dipole_moment_matrices=True,
        )
    )

    result = reader.read(
        _minimal_orca_lines(
            include_spin=True,
            include_angular_momentum=True,
            include_electric_dipole=True,
        )
    )

    assert result.spin_matrices is not None
    assert result.angular_momentum_matrices is not None
    assert result.electric_dipole_moment_matrices is not None
    assert result.spin_matrices.shape == (3, 2, 2)
    assert result.angular_momentum_matrices.shape == (3, 2, 2)
    assert result.electric_dipole_moment_matrices.shape == (3, 2, 2)


def test_reader_parses_ci_expansions() -> None:
    reader = _reader_without_optional_operators(
        ci_basis=True,
        parse_ci_expansions=True,
    )

    result = reader.read(_minimal_orca_lines(include_ci=True))

    assert result.ci_expansions_by_multiplicity is not None
    assert set(result.ci_expansions_by_multiplicity) == {1}


def test_reader_read_as_slt_results() -> None:
    reader = _reader_without_optional_operators()

    results = reader.read_as_slt_results(_minimal_orca_lines())

    assert results.primary == "state_energies"
    assert results.slt_type == "HAMILTONIAN"
    assert isinstance(results.dataset, xr.Dataset)


def test_reader_write_to_group(tmp_path: Path) -> None:
    reader = _reader_without_optional_operators()
    slt = create_slt_file(tmp_path / "reader_write.slt", overwrite=True)

    group = reader.write_to_group(_minimal_orca_lines(), slt, "hamiltonian")

    assert isinstance(group, SltGroup)

    dataset = group.to_dataset()
    try:
        assert "state_energies" in dataset
    finally:
        dataset.close()


def test_reader_accepts_text_stream(tmp_path: Path) -> None:
    source = _write_lines(tmp_path / "orca.out", _minimal_orca_lines())
    reader = _reader_without_optional_operators()

    with source.open(errors="replace") as stream:
        result = reader.read(stream)

    assert result.state_energies is not None
    np.testing.assert_allclose(result.state_energies, [0.0, 1.0])


# ---------------------------------------------------------------------------
# High-level reader error branches
# ---------------------------------------------------------------------------


def test_reader_rejects_incomplete_dimension_information() -> None:
    reader = _reader_without_optional_operators()

    with pytest.raises(ValueError, match="complete ORCA dimension information"):
        reader.read(_input_block())


def test_reader_rejects_soc_before_dimension_information() -> None:
    reader = _reader_without_optional_operators()

    with pytest.raises(ValueError, match="before enough dimension information"):
        reader.read(["SOC MATRIX (A.U.)"])


def test_reader_rejects_missing_soc_matrix() -> None:
    reader = _reader_without_optional_operators()

    with pytest.raises(ValueError, match="Could not find requested occurrence"):
        reader.read([*_input_block(), *_dimension_lines()])


def test_reader_rejects_missing_second_pt2_occurrence() -> None:
    reader = _reader_without_optional_operators(pt2=True)

    with pytest.raises(ValueError, match="requested occurrence 2"):
        reader.read(_minimal_orca_lines())


def test_reader_rejects_ci_block_before_required_ci_dimensions() -> None:
    reader = _reader_without_optional_operators(
        ci_basis=True,
        parse_ci_expansions=True,
    )

    lines = [
        *_input_block(),
        "Spin-Determinant CI Printing",
    ]

    with pytest.raises(ValueError, match="before multiplicities"):
        reader.read(lines)


def test_reader_rejects_missing_requested_ci_blocks() -> None:
    reader = _reader_without_optional_operators(
        ci_basis=True,
        parse_ci_expansions=True,
    )

    dim = 4
    lines = _minimal_orca_lines(
        mult="1 3",
        nroots="1 1",
        dim=dim,
        soc_real=np.diag(np.arange(dim, dtype=np.float64)),
        soc_imag=np.zeros((dim, dim), dtype=np.float64),
        include_ci=True,
    )

    with pytest.raises(ValueError, match="Could not parse all requested"):
        reader.read(lines)


def test_reader_rejects_missing_operator_components() -> None:
    reader = orca_mod.OrcaHamiltonianReader(
        orca_mod.OrcaHamiltonianReaderOptions(
            include_spin_matrices=True,
            include_angular_momentum_matrices=False,
        )
    )

    with pytest.raises(ValueError, match="spin_matrices"):
        reader.read(_minimal_orca_lines(include_spin=False))


# ---------------------------------------------------------------------------
# Public hamiltonian_from_orca API
# ---------------------------------------------------------------------------


def test_hamiltonian_from_orca_creates_file_and_group(tmp_path: Path) -> None:
    source = _write_lines(tmp_path / "orca.out", _minimal_orca_lines())
    slt_path = tmp_path / "ham.slt"

    slt = orca_mod.hamiltonian_from_orca(
        source,
        slt_path,
        "hamiltonian",
        include_spin_matrices=False,
        include_angular_momentum_matrices=False,
    )

    assert slt.path == slt_path
    assert "hamiltonian" in slt

    opened = open_slt_file(slt_path)
    group = opened["hamiltonian"]
    assert isinstance(group, SltGroup)

    dataset = group.to_dataset()
    try:
        assert "state_energies" in dataset
    finally:
        dataset.close()


def test_hamiltonian_from_orca_opens_existing_file(tmp_path: Path) -> None:
    source = _write_lines(tmp_path / "orca.out", _minimal_orca_lines())
    slt_path = tmp_path / "existing.slt"
    create_slt_file(slt_path, overwrite=True)

    slt = orca_mod.hamiltonian_from_orca(
        source,
        slt_path,
        "hamiltonian",
        include_spin_matrices=False,
        include_angular_momentum_matrices=False,
    )

    assert slt.path == slt_path
    assert "hamiltonian" in slt


def test_hamiltonian_from_orca_accepts_open_slt_file(tmp_path: Path) -> None:
    source = _write_lines(tmp_path / "orca.out", _minimal_orca_lines())
    slt_path = tmp_path / "open.slt"
    opened = create_slt_file(slt_path, overwrite=True)

    slt = orca_mod.hamiltonian_from_orca(
        source,
        opened,
        "hamiltonian",
        include_spin_matrices=False,
        include_angular_momentum_matrices=False,
    )

    assert isinstance(slt, SltFile)
    assert slt is opened
    assert slt.path == slt_path
    assert "hamiltonian" in slt


def test_hamiltonian_from_orca_rejects_empty_group_name(tmp_path: Path) -> None:
    source = _write_lines(tmp_path / "orca.out", _minimal_orca_lines())

    with pytest.raises(ValidationError):
        orca_mod.hamiltonian_from_orca(
            source,
            tmp_path / "ham.slt",
            "",
            include_spin_matrices=False,
            include_angular_momentum_matrices=False,
        )


def test_hamiltonian_from_orca_rejects_parse_ci_without_ci_basis(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing.out"

    with pytest.raises(ValueError, match="parse_ci_expansions=True requires"):
        orca_mod.hamiltonian_from_orca(
            source,
            tmp_path / "ham.slt",
            "hamiltonian",
            parse_ci_expansions=True,
        )


def test_hamiltonian_from_orca_writes_ci_basis(tmp_path: Path) -> None:
    source = _write_lines(tmp_path / "orca.out", _minimal_orca_lines())
    slt_path = tmp_path / "ci.slt"

    slt = orca_mod.hamiltonian_from_orca(
        source,
        slt_path,
        "ci_hamiltonian",
        ci_basis=True,
        include_spin_matrices=False,
        include_angular_momentum_matrices=False,
    )

    group = slt["ci_hamiltonian"]
    assert isinstance(group, SltGroup)

    dataset = group.to_dataset()
    try:
        assert "soc_matrix" in dataset
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
# Optional real ORCA-output integration fixture
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_ORCA_SOC_FILE.exists(),
    reason="tests/data/orca/Pr_minimal.out is not available.",
)
def test_real_orca_soc_fixture_from_path() -> None:
    reader = _reader_without_optional_operators()

    result = reader.read(REAL_ORCA_SOC_FILE)

    assert result.representation == "DIAGONAL"
    assert result.state_energies is not None
    assert result.state_energies.ndim == 1
    assert result.state_energies.size == int(result.attrs["states"])


@pytest.mark.skipif(
    not REAL_ORCA_SOC_FILE.exists(),
    reason="tests/data/orca/Pr_minimal.out is not available.",
)
def test_real_orca_soc_fixture_from_stream() -> None:
    reader = _reader_without_optional_operators()

    with REAL_ORCA_SOC_FILE.open(errors="replace") as stream:
        result = reader.read(stream)

    assert result.representation == "DIAGONAL"
    assert result.state_energies is not None
    assert result.state_energies.size == int(result.attrs["states"])


def test_reader_ignores_repeated_dimension_lines_after_info_is_cached() -> None:
    """
    Once ``OrcaDimensionInfo`` is built, ``update_dimension_info`` returns
    immediately and later dimension headers must not change ``dim`` or attrs.
    """
    dim = 2
    lines = _minimal_orca_lines(dim=dim)
    soc_start = next(i for i, line in enumerate(lines) if line.startswith("SOC"))
    lines = (
        lines[:soc_start]
        + _input_block(mult="3", nroots="4")
        + _dimension_lines(
            active_orbitals=99,
            total_orbitals=99,
            inactive_orbitals=99,
        )
        + lines[soc_start:]
    )

    reader = _reader_without_optional_operators()
    result = reader.read(lines)

    assert int(result.attrs["states"]) == dim
    np.testing.assert_array_equal(result.attrs["multiplicities"], [1])
    np.testing.assert_array_equal(result.attrs["nroots"], [2])
    assert result.attrs["active_orbitals"] == 2
    assert result.attrs["total_orbitals"] == 5
    assert result.attrs["inactive_orbitals"] == 3
    assert result.state_energies is not None
    assert result.state_energies.size == dim


def test_reader_reuses_dimension_info_for_later_operator_blocks() -> None:
    reader = orca_mod.OrcaHamiltonianReader(
        orca_mod.OrcaHamiltonianReaderOptions(
            include_spin_matrices=True,
            include_angular_momentum_matrices=False,
            include_electric_dipole_moment_matrices=False,
        )
    )

    result = reader.read(
        _minimal_orca_lines(
            include_spin=True,
            include_angular_momentum=False,
            include_electric_dipole=False,
        )
    )

    assert result.spin_matrices is not None
    assert result.spin_matrices.shape == (3, 2, 2)


def test_parse_ci_block_raises_when_nroots_is_zero() -> None:
    with pytest.raises(RuntimeError, match="No CI determinants parsed"):
        orca_mod._parse_orca_spin_determinant_ci_block(
            orca_mod._LineStream([]),
            multiplicity=1,
            nroots=0,
            active_orbitals=2,
        )


def test_require_ci_coefficient_buffers_for_later_root_raises_when_uninitialized() -> (
    None
):
    with pytest.raises(RuntimeError, match="Internal CI parser state error"):
        orca_mod._require_ci_coefficient_buffers_for_later_root(None, None)
