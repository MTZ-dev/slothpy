from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from re import IGNORECASE, MULTILINE, Pattern, compile, findall
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, validate_call

from .hamiltonian_reader import (
    CIDeterminantExpansion,
    HamiltonianReader,
    HamiltonianReaderOptions,
    HamiltonianReaderResult,
    write_hamiltonian_reader_result_to_slt_group,
)

try:
    from slothpy.core.slt import SltFile, create_slt_file, open_slt_file
    from slothpy.types.aliases import PathLike
except Exception:  # pragma: no cover - for standalone parser tests outside SlothPy
    SltFile = Any  # type: ignore[misc,assignment]
    PathLike = str | Path  # type: ignore[misc,assignment]


_FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"

_ROW_INDEX_RE = compile(r"^\s*\d+\s*", MULTILINE)
_NEGATIVE_OVERLAP_RE = compile(r"(\d)(-)")
_COLUMN_HEADER_RE = compile(r"^\s*(?:\d+\s*)+$")

_INPUT_START_RE = compile(r"^\s*INPUT FILE")
_INPUT_END_RE = compile(r"^\s*\|?\s*\d*>?\s*\*{4}END OF INPUT\*{4}")

_ACTIVE_ORBITALS_RE = compile(r"Number of active orbitals\s+\.\.\.\s+(\d+)")
_TOTAL_ORBITALS_RE = compile(r"Total number of orbitals\s+\.\.\.\s+(\d+)")
_ORBITAL_RANGES_RE = compile(r"Determined orbital ranges:")
_INTERNAL_ORBITALS_RE = compile(r"Internal\s+\d+\s*-\s*\d+\s*\(\s*(\d+)\s+orbitals\)")

_SOC_MATRIX_RE = compile(r"^\s*SOC MATRIX \(A\.U\.\)\s*$")
_SOC_SSC_MATRIX_RE = compile(r"^\s*SOC and SSC MATRIX \(A\.U\.\)\s*$")
_MATRIX_PART_RE = compile(r"(?:Real|Imag|Image) part:")

_CI_START_RE = compile(r"^\s*Spin-Determinant CI Printing\s*$")
_ROOT_START_RE = compile(r"^ROOT\s+(\d+):\s+E=")
_DET_LINE_RE = compile(rf"^\s*\[([ud20]+)\]\s+({_FLOAT_RE})")

_SPIN_LABELS = (
    "SX MATRIX IN CI BASIS",
    "SY MATRIX IN CI BASIS",
    "SZ MATRIX IN CI BASIS",
)
_ANGULAR_MOMENTUM_LABELS = (
    "LX MATRIX IN CI BASIS",
    "LY MATRIX IN CI BASIS",
    "LZ MATRIX IN CI BASIS",
)
_ELECTRIC_DIPOLE_LABELS = (
    "Matrix EDX in CI Basis",
    "Matrix EDY in CI Basis",
    "Matrix EDZ in CI Basis",
)

_SPIN_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(_SPIN_LABELS)}
_ANGULAR_MOMENTUM_LABEL_TO_INDEX = {
    label: idx for idx, label in enumerate(_ANGULAR_MOMENTUM_LABELS)
}
_ELECTRIC_DIPOLE_LABEL_TO_INDEX = {
    label: idx for idx, label in enumerate(_ELECTRIC_DIPOLE_LABELS)
}


@dataclass(frozen=True, slots=True)
class OrcaHamiltonianReaderOptions(HamiltonianReaderOptions):
    """Options controlling ORCA Hamiltonian parsing."""

    pt2: bool = False
    ssc: bool = False


@dataclass(slots=True)
class OrcaHamiltonianReader(HamiltonianReader):
    """
    Read ORCA CASSCF/relativistic SOC/SSC output into structured Hamiltonian data.

    The parser is one-pass and stream-compatible. It does not materialize the
    whole ORCA output in memory.
    """

    options: OrcaHamiltonianReaderOptions = OrcaHamiltonianReaderOptions()

    def read(self, source: PathLike | Iterable[str]) -> HamiltonianReaderResult:
        stream = _LineStream(_iter_source_lines(source))

        multiplicities: np.ndarray | None = None
        nroots: np.ndarray | None = None
        active_orbitals: int | None = None
        total_orbitals: int | None = None
        inactive_orbitals: int | None = None

        info: OrcaDimensionInfo | None = None
        dim: int | None = None

        soc_matrix: np.ndarray | None = None
        soc_occurrence = 0
        target_soc_occurrence = 2 if self.options.pt2 else 1

        spin_parts: list[np.ndarray | None] = [None, None, None]
        angular_momentum_parts: list[np.ndarray | None] = [None, None, None]
        electric_dipole_parts: list[np.ndarray | None] = [None, None, None]

        ci_expansions: dict[int, CIDeterminantExpansion] = {}
        ci_block_index = 0

        soc_pattern = _SOC_SSC_MATRIX_RE if self.options.ssc else _SOC_MATRIX_RE

        def update_dimension_info() -> None:
            nonlocal info, dim

            if info is not None:
                return

            if (
                multiplicities is None
                or nroots is None
                or active_orbitals is None
                or total_orbitals is None
                or inactive_orbitals is None
            ):
                return

            info = OrcaDimensionInfo(
                multiplicities=multiplicities,
                nroots=nroots,
                active_orbitals=active_orbitals,
                total_orbitals=total_orbitals,
                inactive_orbitals=inactive_orbitals,
            )
            dim = int(np.sum(info.multiplicities * info.nroots))

        def require_dim(section: str) -> int:
            if dim is None:
                raise ValueError(
                    f"Found ORCA section {section!r} before enough dimension "
                    "information was parsed."
                )
            return dim

        for line in stream:
            stripped = line.strip()

            if _INPUT_START_RE.match(line):
                multiplicities, nroots = _parse_input_block(stream)
                update_dimension_info()
                continue

            match = _ACTIVE_ORBITALS_RE.search(line)
            if match:
                active_orbitals = int(match.group(1))
                update_dimension_info()
                continue

            match = _TOTAL_ORBITALS_RE.search(line)
            if match:
                total_orbitals = int(match.group(1))
                update_dimension_info()
                continue

            if _ORBITAL_RANGES_RE.search(line):
                inactive_orbitals = _parse_inactive_orbitals(stream)
                update_dimension_info()
                continue

            if (
                self.options.ci_basis
                and self.options.parse_ci_expansions
                and _CI_START_RE.match(line)
            ):
                if multiplicities is None or nroots is None or active_orbitals is None:
                    raise ValueError(
                        "Found ORCA CI determinant block before multiplicities, "
                        "nroots, and active orbital count were parsed."
                    )

                if ci_block_index < len(multiplicities):
                    mult = int(multiplicities[ci_block_index])
                    nr = int(nroots[ci_block_index])
                    ci_expansions[mult] = _parse_orca_spin_determinant_ci_block(
                        stream,
                        multiplicity=mult,
                        nroots=nr,
                        active_orbitals=active_orbitals,
                    )
                    ci_block_index += 1

                continue

            if soc_pattern.match(line):
                current_dim = require_dim(stripped)
                soc_occurrence += 1

                if soc_occurrence == target_soc_occurrence:
                    soc_matrix = _read_soc_matrix_from_stream(
                        stream,
                        dim=current_dim,
                    )

                continue

            if (
                self.options.include_electric_dipole_moment_matrices
                and stripped in _ELECTRIC_DIPOLE_LABEL_TO_INDEX
            ):
                current_dim = require_dim(stripped)
                idx = _ELECTRIC_DIPOLE_LABEL_TO_INDEX[stripped]
                electric_dipole_parts[idx] = _read_electric_dipole_matrix_from_stream(
                    stream,
                    dim=current_dim,
                )
                continue

            if self.options.include_spin_matrices and stripped in _SPIN_LABEL_TO_INDEX:
                current_dim = require_dim(stripped)
                idx = _SPIN_LABEL_TO_INDEX[stripped]
                spin_parts[idx] = _read_vector_operator_matrix_from_stream(
                    stream,
                    dim=current_dim,
                    label=stripped,
                    operator="spin",
                )
                continue

            if (
                self.options.include_angular_momentum_matrices
                and stripped in _ANGULAR_MOMENTUM_LABEL_TO_INDEX
            ):
                current_dim = require_dim(stripped)
                idx = _ANGULAR_MOMENTUM_LABEL_TO_INDEX[stripped]
                angular_momentum_parts[idx] = _read_vector_operator_matrix_from_stream(
                    stream,
                    dim=current_dim,
                    label=stripped,
                    operator="angular_momentum",
                )
                continue

        if info is None or dim is None:
            raise ValueError(
                "Could not parse complete ORCA dimension information: "
                "multiplicities, nroots, active orbitals, total orbitals, "
                "and inactive orbitals are required."
            )

        if soc_matrix is None:
            header = (
                "SOC and SSC MATRIX (A.U.)" if self.options.ssc else "SOC MATRIX (A.U.)"
            )
            raise ValueError(
                f"Could not find requested occurrence {target_soc_occurrence} "
                f"of ORCA section {header!r}."
            )

        spin_matrices = (
            _stack_required_operator_parts("spin_matrices", spin_parts)
            if self.options.include_spin_matrices
            else None
        )
        angular_momentum_matrices = (
            _stack_required_operator_parts(
                "angular_momentum_matrices",
                angular_momentum_parts,
            )
            if self.options.include_angular_momentum_matrices
            else None
        )
        electric_dipole_moment_matrices = (
            _stack_required_operator_parts(
                "electric_dipole_moment_matrices",
                electric_dipole_parts,
            )
            if self.options.include_electric_dipole_moment_matrices
            else None
        )

        if (
            self.options.ci_basis
            and self.options.parse_ci_expansions
            and len(ci_expansions) != len(info.multiplicities)
        ):
            raise ValueError(
                "Could not parse all requested ORCA CI determinant blocks. "
                f"Expected {len(info.multiplicities)}, got {len(ci_expansions)}."
            )

        interaction: Literal["SOC", "SOC_SSC"] = (
            "SOC_SSC" if self.options.ssc else "SOC"
        )
        representation: Literal["CI", "DIAGONAL"] = (
            "CI" if self.options.ci_basis else "DIAGONAL"
        )

        attrs: dict[str, Any] = {
            "slt_kind": "ORCA",
            "hamiltonian_type": "SOC_SSC" if self.options.ssc else "SOC",
            "basis": "CI" if self.options.ci_basis else "DIAGONAL",
            "states": dim,
            "multiplicities": info.multiplicities.astype(np.int64),
            "nroots": info.nroots.astype(np.int64),
            "active_orbitals": info.active_orbitals,
            "total_orbitals": info.total_orbitals,
            "inactive_orbitals": info.inactive_orbitals,
            "source_format": "ORCA 6.X.Y output",
        }

        vectors: np.ndarray | None = None
        hamiltonian_matrix: np.ndarray | None = None
        states_energies: np.ndarray | None = None

        if self.options.ci_basis:
            hamiltonian_matrix = soc_matrix
        else:
            energies, vectors = np.linalg.eigh(soc_matrix)

            if self.options.shift_energies:
                energies = energies - np.min(energies)

            states_energies = energies.real.astype(np.float64)
            attrs["shift_energies_applied"] = self.options.shift_energies

        if vectors is not None:
            if spin_matrices is not None:
                spin_matrices = _transform_operator_stack(spin_matrices, vectors)

            if angular_momentum_matrices is not None:
                angular_momentum_matrices = _transform_operator_stack(
                    angular_momentum_matrices,
                    vectors,
                )

            if electric_dipole_moment_matrices is not None:
                electric_dipole_moment_matrices = _transform_operator_stack(
                    electric_dipole_moment_matrices,
                    vectors,
                )

        return HamiltonianReaderResult(
            hamiltonian_interaction=interaction,
            representation=representation,
            hamiltonian_matrix=hamiltonian_matrix,
            states_energies=states_energies,
            spin_matrices=spin_matrices,
            angular_momentum_matrices=angular_momentum_matrices,
            electric_dipole_moment_matrices=electric_dipole_moment_matrices,
            ci_expansions_by_multiplicity=ci_expansions or None,
            attrs=attrs,
        )


@dataclass(frozen=True, slots=True)
class OrcaDimensionInfo:
    multiplicities: np.ndarray
    nroots: np.ndarray
    active_orbitals: int
    total_orbitals: int
    inactive_orbitals: int


class _LineStream:
    """
    One-pass line stream with a tiny local push-back buffer.

    This does not seek the source and does not store the output file. The buffer
    is used only when a parser reads one delimiter line too far, mainly in CI
    determinant parsing.
    """

    __slots__ = ("_iterator", "_buffer")

    def __init__(self, lines: Iterable[str]) -> None:
        self._iterator = iter(lines)
        self._buffer: list[str] = []

    def __iter__(self) -> _LineStream:
        return self

    def __next__(self) -> str:
        if self._buffer:
            return self._buffer.pop()
        return next(self._iterator)

    def push_back(self, line: str) -> None:
        self._buffer.append(line)


def _iter_source_lines(source: PathLike | Iterable[str]) -> Iterator[str]:
    """
    Yield source lines one by one without storing the whole ORCA output.
    """
    if isinstance(source, str | Path):
        with Path(source).open(errors="replace") as handle:
            for line in handle:
                yield line.rstrip("\r\n")
        return

    for line in source:
        yield line.rstrip("\r\n")


def _strip_orca_input_prefix(line: str) -> str:
    return compile(r"^\s*\|\s*\d+>\s*").sub("", line).strip()


def _parse_input_block(stream: _LineStream) -> tuple[np.ndarray, np.ndarray]:
    input_lines: list[str] = []

    for line in stream:
        input_lines.append(_strip_orca_input_prefix(line))
        if _INPUT_END_RE.match(line):
            break
    else:
        raise ValueError("Unexpected end of file while reading ORCA input block.")

    casscf_start = None
    for idx, line in enumerate(input_lines):
        if compile(r"^%casscf\b", IGNORECASE).match(line):
            casscf_start = idx
            break

    if casscf_start is None:
        raise ValueError("Could not find %casscf section in ORCA input block.")

    casscf_lines: list[str] = []
    for line in input_lines[casscf_start + 1 :]:
        if compile(r"^%\w+\b", IGNORECASE).match(line):
            break
        casscf_lines.append(line)

    multiplicities: np.ndarray | None = None
    nroots: np.ndarray | None = None

    for line in casscf_lines:
        mult_match = compile(r"^mult\s+(.+)$", IGNORECASE).match(line)
        if mult_match:
            multiplicities = np.asarray(
                list(map(int, findall(r"\d+", mult_match.group(1)))),
                dtype=np.int64,
            )
            continue

        nroots_match = compile(r"^nroots\s+(.+)$", IGNORECASE).match(line)
        if nroots_match:
            nroots = np.asarray(
                list(map(int, findall(r"\d+", nroots_match.group(1)))),
                dtype=np.int64,
            )
            continue

    if multiplicities is None or nroots is None:
        raise ValueError(
            "Could not find multiplicities and nroots in the %casscf section."
        )

    if multiplicities.shape != nroots.shape:
        raise ValueError(
            "ORCA multiplicities and nroots arrays have different lengths: "
            f"{multiplicities.tolist()} vs {nroots.tolist()}."
        )

    return multiplicities, nroots


def _parse_inactive_orbitals(stream: _LineStream) -> int:
    for line in stream:
        match = _INTERNAL_ORBITALS_RE.search(line)
        if match:
            return int(match.group(1))

        if line.strip() == "":
            continue

        if "Active" in line or "External" in line:
            continue

        if "Number of rotation parameters" in line:
            break

    raise ValueError("Could not find inactive/internal orbital count.")


def _consume_until(stream: _LineStream, pattern: Pattern[str]) -> str:
    for line in stream:
        if pattern.search(line):
            return line

    raise ValueError(
        f"Could not find ORCA output section matching {pattern.pattern!r}."
    )


def _is_column_header(line: str) -> bool:
    return _COLUMN_HEADER_RE.match(line) is not None


def _read_block_matrix_from_stream(
    stream: _LineStream,
    *,
    dim: int,
    fix_negative_overlap: bool,
) -> np.ndarray:
    matrix = np.empty((dim, dim), dtype=np.float64, order="C")
    col_start = 0

    while col_start < dim:
        for line in stream:
            if _is_column_header(line):
                header = line
                break
        else:
            raise ValueError("Unexpected end of file while reading ORCA matrix block.")

        columns = [int(value) for value in header.split()]
        ncols = len(columns)

        if ncols <= 0:
            raise ValueError("Empty ORCA matrix column header.")

        data_lines: list[str] = []
        for _ in range(dim):
            try:
                data_lines.append(next(stream))
            except StopIteration as exc:
                raise ValueError("Incomplete ORCA matrix block.") from exc

        data_str = "\n".join(data_lines)
        data_str = _ROW_INDEX_RE.sub("", data_str)

        if fix_negative_overlap:
            data_str = _NEGATIVE_OVERLAP_RE.sub(r"\1 -", data_str)

        data_str = data_str.replace("D", "E").replace("d", "e")

        data = np.fromstring(data_str, sep=" ", dtype=np.float64)
        expected = dim * ncols

        if data.size != expected:
            raise ValueError(
                "Failed to parse ORCA matrix block. "
                f"Expected {expected} floats, got {data.size}."
            )

        matrix[:, col_start : col_start + ncols] = data.reshape(dim, ncols)
        col_start += ncols

    return matrix


def _read_soc_matrix_from_stream(stream: _LineStream, *, dim: int) -> np.ndarray:
    _consume_until(stream, compile(r"Real part:"))
    real = _read_block_matrix_from_stream(
        stream,
        dim=dim,
        fix_negative_overlap=True,
    )

    _consume_until(stream, compile(r"(?:Imag|Image) part:"))
    imag = _read_block_matrix_from_stream(
        stream,
        dim=dim,
        fix_negative_overlap=True,
    )

    return real + 1j * imag


def _read_electric_dipole_matrix_from_stream(
    stream: _LineStream,
    *,
    dim: int,
) -> np.ndarray:
    matrix = _read_block_matrix_from_stream(
        stream,
        dim=dim,
        fix_negative_overlap=False,
    )
    return matrix.astype(np.complex128)


def _read_vector_operator_matrix_from_stream(
    stream: _LineStream,
    *,
    dim: int,
    label: str,
    operator: Literal["spin", "angular_momentum"],
) -> np.ndarray:
    _consume_until(stream, _MATRIX_PART_RE)

    matrix = _read_block_matrix_from_stream(
        stream,
        dim=dim,
        fix_negative_overlap=False,
    )

    # Preserve ORCA phase convention for angular momentum and Sy.
    if operator == "spin":
        if label.startswith("SY"):
            matrix = 1j * matrix
        return 0.5 * matrix

    return 1j * matrix


def _next_nonempty_line(stream: _LineStream) -> str:
    for line in stream:
        if line.strip() != "":
            return line

    raise ValueError("Unexpected end of file while reading ORCA section.")


def _consume_until_root(
    stream: _LineStream,
    *,
    expected_root: int,
    multiplicity: int,
) -> None:
    for line in stream:
        root_match = _ROOT_START_RE.match(line)
        if root_match and int(root_match.group(1)) == expected_root:
            return

    raise RuntimeError(
        f"Could not find ROOT {expected_root} for multiplicity {multiplicity}."
    )


def _require_ci_coefficient_buffers_for_later_root(
    determinant_patterns: list[str] | None,
    ci_coeffs: np.ndarray | None,
) -> None:
    if determinant_patterns is None or ci_coeffs is None:
        raise RuntimeError("Internal CI parser state error.")


def _parse_orca_spin_determinant_ci_block(
    stream: _LineStream,
    *,
    multiplicity: int,
    nroots: int,
    active_orbitals: int,
) -> CIDeterminantExpansion:
    determinant_patterns: list[str] | None = None
    ci_coeffs: np.ndarray | None = None

    for expected_root in range(nroots):
        _consume_until_root(
            stream,
            expected_root=expected_root,
            multiplicity=multiplicity,
        )

        current = _next_nonempty_line(stream)

        coeffs: list[float] = []
        patterns_this_root: list[str] = []

        while True:
            det_match = _DET_LINE_RE.match(current)
            if not det_match:
                stream.push_back(current)
                break

            det_str, coeff_str = det_match.groups()
            patterns_this_root.append(det_str)
            coeffs.append(float(coeff_str.replace("D", "E").replace("d", "e")))

            try:
                current = next(stream)
            except StopIteration:
                break

        if expected_root == 0:
            determinant_patterns = patterns_this_root
            ci_coeffs = np.zeros((len(coeffs), nroots), dtype=np.float64)
        else:
            _require_ci_coefficient_buffers_for_later_root(
                determinant_patterns,
                ci_coeffs,
            )

            if patterns_this_root != determinant_patterns:
                raise ValueError(
                    f"Inconsistent determinant list for ROOT {expected_root} "
                    f"of multiplicity {multiplicity}."
                )

        ci_coeffs[:, expected_root] = np.asarray(coeffs, dtype=np.float64)

    if determinant_patterns is None or ci_coeffs is None:
        raise RuntimeError(
            f"No CI determinants parsed for multiplicity {multiplicity}."
        )

    alpha = np.zeros((len(determinant_patterns), active_orbitals), dtype=np.int8)
    beta = np.zeros_like(alpha)

    for row, det_str in enumerate(determinant_patterns):
        alpha[row], beta[row] = _decode_orca_determinant_occupations(
            det_str,
            active_orbitals,
        )

    return CIDeterminantExpansion(
        alpha_occupations=alpha,
        beta_occupations=beta,
        ci_coefficients=ci_coeffs,
    )


def _decode_orca_determinant_occupations(
    det_str: str,
    active_orbitals: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if active_orbitals is None:
        active_orbitals = len(det_str)

    if len(det_str) != active_orbitals:
        raise ValueError(
            f"Determinant {det_str!r} has length {len(det_str)}, "
            f"expected {active_orbitals}."
        )

    alpha = np.zeros(active_orbitals, dtype=np.int8)
    beta = np.zeros(active_orbitals, dtype=np.int8)

    for idx, char in enumerate(det_str):
        if char == "u":
            alpha[idx] = 1
        elif char == "d":
            beta[idx] = 1
        elif char == "2":
            alpha[idx] = 1
            beta[idx] = 1
        elif char == "0":
            continue
        else:
            raise ValueError(f"Unknown determinant character {char!r} in {det_str!r}.")

    return alpha, beta


def _stack_required_operator_parts(
    name: str,
    parts: list[np.ndarray | None],
) -> np.ndarray:
    missing = [idx for idx, part in enumerate(parts) if part is None]

    if missing:
        raise ValueError(
            f"Could not parse all components of {name}; missing {missing}."
        )

    return np.stack([part for part in parts if part is not None], axis=0)


def _transform_operator_stack(
    matrices: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(matrices, dtype=np.complex128)
    vdag = eigenvectors.conj().T

    for idx in range(matrices.shape[0]):
        transformed[idx] = vdag @ matrices[idx] @ eigenvectors

    return transformed


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def hamiltonian_from_orca(
    orca_source: PathLike | Iterable[str],
    slt_filepath: PathLike,
    group_name: Annotated[str, Field(min_length=1)],
    *,
    shift_energies: bool = True,
    include_spin_matrices: bool = True,
    include_angular_momentum_matrices: bool = True,
    include_electric_dipole_moment_matrices: bool = False,
    ci_basis: bool = False,
    parse_ci_expansions: bool = False,
    pt2: bool = False,
    ssc: bool = False,
    overwrite: bool = False,
) -> SltFile:
    """
    Read an ORCA output source and write a SlothPy Hamiltonian group.

    Parameters
    ----------
    orca_source
        ORCA output file path or a one-pass iterable/stream of output lines.
    slt_filepath
        Target ``.slt`` file path. The file is created if it does not exist.
    group_name
        Root-level SlothPy group name for the Hamiltonian data.
    shift_energies
        Shift diagonalized energies so that the lowest state has energy zero.
    include_spin_matrices
        Parse spin matrices.
    include_angular_momentum_matrices
        Parse orbital angular momentum matrices.
    include_electric_dipole_moment_matrices
        Parse electric dipole moment matrices.
    ci_basis
        Store the SOC/SOC+SSC Hamiltonian in the CI basis as read. When
        ``False``, diagonalize the matrix and store eigenstate energies.
    parse_ci_expansions
        Parse spin-determinant CI expansions. Requires ``ci_basis=True``.
    pt2
        Read the second SOC/SOC+SSC matrix occurrence, used for PT2 output.
    ssc
        Read the SOC+SSC matrix instead of the SOC-only matrix.
    overwrite
        Replace an existing SlothPy group with the same name.
    """
    if parse_ci_expansions and not ci_basis:
        raise ValueError("parse_ci_expansions=True requires ci_basis=True.")

    reader = OrcaHamiltonianReader(
        OrcaHamiltonianReaderOptions(
            parse_ci_expansions=parse_ci_expansions,
            shift_energies=shift_energies,
            include_spin_matrices=include_spin_matrices,
            include_angular_momentum_matrices=include_angular_momentum_matrices,
            include_electric_dipole_moment_matrices=(
                include_electric_dipole_moment_matrices
            ),
            ci_basis=ci_basis,
            pt2=pt2,
            ssc=ssc,
        )
    )

    structured = reader.read(orca_source)

    try:
        slt = open_slt_file(slt_filepath)
    except FileNotFoundError:
        slt = create_slt_file(slt_filepath)

    write_hamiltonian_reader_result_to_slt_group(
        slt,
        group_name,
        structured,
        overwrite=overwrite,
    )

    return slt


__all__ = [
    "CIDeterminantExpansion",
    "HamiltonianReader",
    "HamiltonianReaderResult",
    "OrcaDimensionInfo",
    "OrcaHamiltonianReader",
    "OrcaHamiltonianReaderOptions",
    "hamiltonian_from_orca",
    "write_hamiltonian_reader_result_to_slt_group",
]
