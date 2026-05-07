from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from re import IGNORECASE, MULTILINE, compile, findall
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr

try:
    from slothpy.core.slt import PathLike, SltFile, create_slt_file, open_slt_file
except Exception:  # pragma: no cover - for standalone parser tests outside SlothPy
    SltFile = Any  # type: ignore
    PathLike = str | Path  # type: ignore


_FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
_ROW_INDEX_RE = compile(r"^\s*\d+\s*", MULTILINE)
_NEGATIVE_OVERLAP_RE = compile(r"(\d)(-)")
_COLUMN_HEADER_RE = compile(r"^\s*(?:\d+\s*)+$")


@dataclass(frozen=True, slots=True)
class OrcaReaderOptions:
    """Options controlling ORCA Hamiltonian parsing."""

    pt2: bool = False
    electric_dipole_momenta: bool = False
    ssc: bool = False
    ci_basis: bool = False
    parse_ci_expansions: bool = True
    diagonalize: bool = True
    shift_energies: bool = True
    preserve_legacy_electric_dipole_phase: bool = False


@dataclass(frozen=True, slots=True)
class ReaderResult:
    """Reader output ready to be written as one SlothPy xarray group."""

    dataset: xr.Dataset
    slt_type: str
    primary: str
    attrs: dict[str, Any]


@runtime_checkable
class HamiltonianReader(Protocol):
    """Protocol for reader dependency injection."""

    def read(self, source: PathLike | Iterable[str]) -> ReaderResult:
        """Read a source and return an xarray-backed SlothPy result."""


@dataclass(slots=True)
class OrcaHamiltonianReader:
    """Read ORCA CASSCF/relativistic SOC output into an xarray Dataset."""

    options: OrcaReaderOptions = OrcaReaderOptions()

    def read(self, source: PathLike | Iterable[str]) -> ReaderResult:
        lines = _read_lines(source)
        info = _get_orca_dimension_info(lines)
        dim = int(np.sum(info.multiplicities * info.nroots))

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
            "source_format": "ORCA output",
        }

        soc_matrix = _read_soc_matrix(
            lines,
            dim=dim,
            ssc=self.options.ssc,
            pt2=self.options.pt2,
        )

        vectors: np.ndarray | None = None
        if self.options.ci_basis:
            primary = "soc_ssc_matrix" if self.options.ssc else "soc_matrix"
            state_dim = "ci_state"
            bra_dim = "ci_bra_state"
            ket_dim = "ci_ket_state"
            state_coord = np.arange(dim, dtype=np.int64)
            data_vars: dict[str, Any] = {
                primary: (
                    (bra_dim, ket_dim),
                    soc_matrix,
                    {
                        "unit": "E_h",
                        "long_name": "SOC+SSC matrix in CI basis"
                        if self.options.ssc
                        else "SOC matrix in CI basis",
                    },
                )
            }
        else:
            if not self.options.diagonalize:
                raise ValueError(
                    "diagonalize=False is only meaningful with ci_basis=True."
                )
            energies, vectors = np.linalg.eigh(soc_matrix)
            if self.options.shift_energies:
                energies = energies - np.min(energies)
            primary = "states_energies"
            state_dim = "state"
            bra_dim = "bra_state"
            ket_dim = "ket_state"
            state_coord = np.arange(dim, dtype=np.int64)
            data_vars = {
                "states_energies": (
                    (state_dim,),
                    energies.real.astype(np.float64),
                    {
                        "unit": "E_h",
                        "long_name": "SOC eigenstate energies shifted to the lowest state"
                        if self.options.shift_energies
                        else "SOC eigenstate energies",
                    },
                )
            }

        spin_matrices = _read_vector_operator(
            lines,
            dim=dim,
            labels=(
                "SX MATRIX IN CI BASIS",
                "SY MATRIX IN CI BASIS",
                "SZ MATRIX IN CI BASIS",
            ),
            operator="spin",
        )
        angular_momenta = _read_vector_operator(
            lines,
            dim=dim,
            labels=(
                "LX MATRIX IN CI BASIS",
                "LY MATRIX IN CI BASIS",
                "LZ MATRIX IN CI BASIS",
            ),
            operator="angular_momentum",
        )

        if vectors is not None:
            spin_matrices = _transform_operator_stack(spin_matrices, vectors)
            angular_momenta = _transform_operator_stack(angular_momenta, vectors)

        data_vars["spins"] = (
            ("component", bra_dim, ket_dim),
            spin_matrices,
            {"long_name": "spin matrices", "component_order": "x,y,z"},
        )
        data_vars["angular_momenta"] = (
            ("component", bra_dim, ket_dim),
            angular_momenta,
            {
                "long_name": "orbital angular momentum matrices",
                "component_order": "x,y,z",
            },
        )

        if self.options.electric_dipole_momenta:
            electric = _read_electric_dipole_momenta(
                lines,
                dim=dim,
                legacy_phase=self.options.preserve_legacy_electric_dipole_phase,
            )
            if vectors is not None:
                electric = _transform_operator_stack(electric, vectors)
            data_vars["electric_dipole_momenta"] = (
                ("component", bra_dim, ket_dim),
                electric,
                {
                    "long_name": "electric dipole moment matrices",
                    "component_order": "x,y,z",
                    "legacy_phase_applied": str(
                        self.options.preserve_legacy_electric_dipole_phase
                    ).lower(),
                },
            )
            attrs["additional"] = "ELECTRIC_DIPOLE_MOMENTA"

        coords = {
            state_dim: state_coord,
            bra_dim: state_coord,
            ket_dim: state_coord,
            "component": np.array(["x", "y", "z"], dtype=object),
        }

        dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        if self.options.ci_basis and self.options.parse_ci_expansions:
            dataset = _add_ci_expansion_variables(dataset, lines, info)

        return ReaderResult(
            dataset=dataset,
            slt_type="HAMILTONIAN",
            primary=primary,
            attrs=attrs,
        )


@dataclass(frozen=True, slots=True)
class OrcaDimensionInfo:
    multiplicities: np.ndarray
    nroots: np.ndarray
    active_orbitals: int
    total_orbitals: int
    inactive_orbitals: int


def _read_lines(source: PathLike | Iterable[str]) -> list[str]:
    if isinstance(source, str | Path):
        return Path(source).read_text(errors="replace").splitlines()
    return [line.rstrip("\n") for line in source]


def _strip_orca_input_prefix(line: str) -> str:
    return compile(r"^\s*\|\s*\d+>\s*").sub("", line).strip()


def _get_orca_dimension_info(lines: list[str]) -> OrcaDimensionInfo:
    input_start_re = compile(r"^\s*INPUT FILE")
    input_end_re = compile(r"^\s*\|?\s*\d*>?\s*\*{4}END OF INPUT\*{4}")
    active_re = compile(r"Number of active orbitals\s+\.\.\.\s+(\d+)")
    total_re = compile(r"Total number of orbitals\s+\.\.\.\s+(\d+)")
    internal_re = compile(r"Internal\s+\d+\s*-\s*\d+\s*\(\s*(\d+)\s+orbitals\)")

    input_start = _find_line(lines, input_start_re)
    input_end = _find_line(lines, input_end_re, start=input_start + 1)
    input_lines = [
        _strip_orca_input_prefix(line) for line in lines[input_start : input_end + 1]
    ]

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
                list(map(int, findall(r"\d+", mult_match.group(1)))), dtype=np.int64
            )
            continue

        nroots_match = compile(r"^nroots\s+(.+)$", IGNORECASE).match(line)
        if nroots_match:
            nroots = np.asarray(
                list(map(int, findall(r"\d+", nroots_match.group(1)))), dtype=np.int64
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

    active_orbitals = _find_first_int(lines, active_re, "number of active orbitals")
    total_orbitals = _find_first_int(lines, total_re, "total number of orbitals")

    ranges_idx = _find_line(lines, compile(r"Determined orbital ranges:"))
    inactive_orbitals = None
    for line in lines[ranges_idx + 1 : ranges_idx + 10]:
        match = internal_re.search(line)
        if match:
            inactive_orbitals = int(match.group(1))
            break
    if inactive_orbitals is None:
        raise ValueError("Could not find inactive/internal orbital count.")

    return OrcaDimensionInfo(
        multiplicities=multiplicities,
        nroots=nroots,
        active_orbitals=active_orbitals,
        total_orbitals=total_orbitals,
        inactive_orbitals=inactive_orbitals,
    )


def _find_first_int(lines: list[str], pattern: Any, description: str) -> int:
    for line in lines:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not find {description}.")


def _find_line(lines: list[str], pattern: Any, *, start: int = 0) -> int:
    for idx in range(start, len(lines)):
        if pattern.search(lines[idx]):
            return idx
    raise ValueError(
        f"Could not find ORCA output section matching {pattern.pattern!r}."
    )


def _read_soc_matrix(lines: list[str], *, dim: int, ssc: bool, pt2: bool) -> np.ndarray:
    pattern = (
        compile(r"^\s*SOC and SSC MATRIX \(A\.U\.\)\s*$")
        if ssc
        else compile(r"^\s*SOC MATRIX \(A\.U\.\)\s*$")
    )
    occurrence = 2 if pt2 else 1
    header_idx = _find_nth_line(lines, pattern, occurrence=occurrence)
    real_idx = _find_line(lines, compile(r"Real part:"), start=header_idx + 1)
    real, after_real = _read_block_matrix(
        lines, start=real_idx + 1, dim=dim, fix_negative_overlap=True
    )
    imag_idx = _find_line(lines, compile(r"(?:Imag|Image) part:"), start=after_real)
    imag, _ = _read_block_matrix(
        lines, start=imag_idx + 1, dim=dim, fix_negative_overlap=True
    )
    return real + 1j * imag


def _find_nth_line(lines: list[str], pattern: Any, *, occurrence: int) -> int:
    count = 0
    for idx, line in enumerate(lines):
        if pattern.search(line):
            count += 1
            if count == occurrence:
                return idx
    raise ValueError(
        f"Could not find occurrence {occurrence} of ORCA section {pattern.pattern!r}."
    )


def _read_vector_operator(
    lines: list[str],
    *,
    dim: int,
    labels: tuple[str, str, str],
    operator: str,
) -> np.ndarray:
    matrices = np.empty((3, dim, dim), dtype=np.complex128)
    for idx, label in enumerate(labels):
        header_idx = _find_line(lines, compile(rf"^\s*{label}\s*$"))
        part_idx = _find_line(
            lines, compile(r"(?:Real|Imag|Image) part:"), start=header_idx + 1
        )
        matrix, _ = _read_block_matrix(
            lines, start=part_idx + 1, dim=dim, fix_negative_overlap=False
        )

        # Preserve ORCA/legacy phase convention for angular momentum and Sy.
        if operator == "spin":
            if label.startswith("SY"):
                matrix = 1j * matrix
            matrices[idx] = 0.5 * matrix
        elif operator == "angular_momentum":
            matrix = 1j * matrix
            matrices[idx] = matrix
        else:
            matrices[idx] = matrix
    return matrices


def _read_electric_dipole_momenta(
    lines: list[str], *, dim: int, legacy_phase: bool
) -> np.ndarray:
    labels = (
        "Matrix EDX in CI Basis",
        "Matrix EDY in CI Basis",
        "Matrix EDZ in CI Basis",
    )
    matrices = np.empty((3, dim, dim), dtype=np.complex128)
    for idx, label in enumerate(labels):
        header_idx = _find_line(lines, compile(rf"^\s*{label}\s*$"))
        matrix, _ = _read_block_matrix(
            lines, start=header_idx + 1, dim=dim, fix_negative_overlap=False
        )
        matrices[idx] = 1j * matrix if legacy_phase else matrix.astype(np.complex128)
    return matrices


def _is_column_header(line: str) -> bool:
    return _COLUMN_HEADER_RE.match(line) is not None


def _read_block_matrix(
    lines: list[str],
    *,
    start: int,
    dim: int,
    fix_negative_overlap: bool,
) -> tuple[np.ndarray, int]:
    matrix = np.empty((dim, dim), dtype=np.float64, order="C")
    col_start = 0
    idx = start

    while col_start < dim:
        while idx < len(lines) and not _is_column_header(lines[idx]):
            idx += 1
        if idx >= len(lines):
            raise ValueError("Unexpected end of file while reading ORCA matrix block.")

        columns = [int(value) for value in lines[idx].split()]
        ncols = len(columns)
        if ncols <= 0:
            raise ValueError("Empty ORCA matrix column header.")

        idx += 1
        data_lines = lines[idx : idx + dim]
        if len(data_lines) != dim:
            raise ValueError("Incomplete ORCA matrix block.")

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
        idx += dim

    return matrix, idx


def _transform_operator_stack(
    matrices: np.ndarray, eigenvectors: np.ndarray
) -> np.ndarray:
    transformed = np.empty_like(matrices, dtype=np.complex128)
    vdag = eigenvectors.conj().T
    for idx in range(matrices.shape[0]):
        transformed[idx] = vdag @ matrices[idx] @ eigenvectors
    return transformed


def _add_ci_expansion_variables(
    dataset: xr.Dataset, lines: list[str], info: OrcaDimensionInfo
) -> xr.Dataset:
    expansions = _parse_orca_spin_determinant_ci(
        lines, info.multiplicities, info.nroots, info.active_orbitals
    )
    result = dataset.copy(deep=False)
    for mult, expansion in expansions.items():
        det_dim = f"determinant_mult_{mult}"
        root_dim = f"root_mult_{mult}"
        orbital_dim = f"active_orbital_mult_{mult}"
        result.coords[det_dim] = np.arange(
            expansion.alpha_occupations.shape[0], dtype=np.int64
        )
        result.coords[root_dim] = np.arange(
            expansion.ci_coefficients.shape[1], dtype=np.int64
        )
        result.coords[orbital_dim] = np.arange(
            expansion.alpha_occupations.shape[1], dtype=np.int64
        )
        result[f"ci_alpha_occupations_mult_{mult}"] = (
            (det_dim, orbital_dim),
            expansion.alpha_occupations,
            {"long_name": f"alpha spin occupations for multiplicity {mult}"},
        )
        result[f"ci_beta_occupations_mult_{mult}"] = (
            (det_dim, orbital_dim),
            expansion.beta_occupations,
            {"long_name": f"beta spin occupations for multiplicity {mult}"},
        )
        result[f"ci_coefficients_mult_{mult}"] = (
            (det_dim, root_dim),
            expansion.ci_coefficients,
            {"long_name": f"spin-determinant CI coefficients for multiplicity {mult}"},
        )
    return result


@dataclass(frozen=True, slots=True)
class CIDeterminantExpansion:
    alpha_occupations: np.ndarray
    beta_occupations: np.ndarray
    ci_coefficients: np.ndarray


def _parse_orca_spin_determinant_ci(
    lines: list[str],
    multiplicities: np.ndarray,
    nroots: np.ndarray,
    active_orbitals: int,
) -> dict[int, CIDeterminantExpansion]:
    ci_start_re = compile(r"^\s*Spin-Determinant CI Printing\s*$")
    root_start_re = compile(r"^ROOT\s+(\d+):\s+E=")
    det_line_re = compile(rf"^\s*\[([ud20]+)\]\s+({_FLOAT_RE})")

    results: dict[int, CIDeterminantExpansion] = {}
    search_start = 0

    for mult_raw, nr_raw in zip(multiplicities, nroots, strict=True):
        mult = int(mult_raw)
        nr = int(nr_raw)
        ci_idx = _find_line(lines, ci_start_re, start=search_start)
        idx = ci_idx + 1

        determinant_patterns: list[str] | None = None
        ci_coeffs: np.ndarray | None = None

        for expected_root in range(nr):
            while idx < len(lines):
                root_match = root_start_re.match(lines[idx])
                if root_match and int(root_match.group(1)) == expected_root:
                    break
                idx += 1
            if idx >= len(lines):
                raise RuntimeError(
                    f"Could not find ROOT {expected_root} for multiplicity {mult}."
                )

            idx += 1
            while idx < len(lines) and lines[idx].strip() == "":
                idx += 1

            coeffs: list[float] = []
            patterns_this_root: list[str] = []
            while idx < len(lines):
                det_match = det_line_re.match(lines[idx])
                if not det_match:
                    break
                det_str, coeff_str = det_match.groups()
                patterns_this_root.append(det_str)
                coeffs.append(float(coeff_str.replace("D", "E").replace("d", "e")))
                idx += 1

            if expected_root == 0:
                determinant_patterns = patterns_this_root
                ci_coeffs = np.zeros((len(coeffs), nr), dtype=np.float64)
            else:
                if determinant_patterns is None or ci_coeffs is None:
                    raise RuntimeError("Internal CI parser state error.")
                if patterns_this_root != determinant_patterns:
                    raise ValueError(
                        f"Inconsistent determinant list for ROOT {expected_root} "
                        f"of multiplicity {mult}."
                    )

            if ci_coeffs is None:
                raise RuntimeError("Internal CI parser state error.")
            ci_coeffs[:, expected_root] = np.asarray(coeffs, dtype=np.float64)

        if determinant_patterns is None or ci_coeffs is None:
            raise RuntimeError(f"No CI determinants parsed for multiplicity {mult}.")

        alpha = np.zeros((len(determinant_patterns), active_orbitals), dtype=np.int8)
        beta = np.zeros_like(alpha)
        for row, det_str in enumerate(determinant_patterns):
            alpha[row], beta[row] = _decode_orca_determinant_occupations(
                det_str, active_orbitals
            )

        results[mult] = CIDeterminantExpansion(
            alpha_occupations=alpha,
            beta_occupations=beta,
            ci_coefficients=ci_coeffs,
        )
        search_start = idx

    return results


def _decode_orca_determinant_occupations(
    det_str: str, active_orbitals: int | None = None
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


def hamiltonian_from_orca(
    orca_filepath: PathLike,
    slt_filepath: PathLike,
    group_name: str,
    *,
    pt2: bool = False,
    electric_dipole_momenta: bool = False,
    ssc: bool = False,
    ci_basis: bool = False,
    overwrite: bool = False,
    reader: HamiltonianReader | None = None,
) -> SltFile:
    """
    Read an ORCA output file and write a SlothPy Hamiltonian group.

    The reader is dependency-injected: any object implementing
    ``HamiltonianReader.read(source) -> ReaderResult`` can be passed.
    """
    if not isinstance(group_name, str):
        raise TypeError(f"group_name must be a string, not {type(group_name)!r}.")

    if reader is None:
        reader = OrcaHamiltonianReader(
            OrcaReaderOptions(
                pt2=pt2,
                electric_dipole_momenta=electric_dipole_momenta,
                ssc=ssc,
                ci_basis=ci_basis,
            )
        )

    result = reader.read(orca_filepath)

    try:
        slt = open_slt_file(slt_filepath)
    except FileNotFoundError:
        slt = create_slt_file(slt_filepath)

    group = slt._write_slothpy_group(
        group_name,
        result.dataset,
        overwrite=overwrite,
        primary=result.primary,
        slt_type=result.slt_type,
    )

    for key, value in result.attrs.items():
        group.attrs[key] = value

    return slt


__all__ = [
    "CIDeterminantExpansion",
    "HamiltonianReader",
    "OrcaDimensionInfo",
    "OrcaHamiltonianReader",
    "OrcaReaderOptions",
    "ReaderResult",
    "hamiltonian_from_orca",
]

if __name__ == "__main__":
    slt = hamiltonian_from_orca(
        "Pr_minimal.out",
        "demo.slt",
        "orca_hamiltonian",
        electric_dipole_momenta=True,
        ci_basis=False,
        overwrite=True,
    )
