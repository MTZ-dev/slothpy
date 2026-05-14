from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from slothpy.core.slt import SltFile, SltGroup, SltResults
from slothpy.types.aliases import (
    HamiltonianInteractionKind,
    HamiltonianRepresentationKind,
    PathLike,
)


@dataclass(frozen=True, slots=True)
class HamiltonianReaderOptions:
    """
    Options shared by SlothPy Hamiltonian readers
    (program-specific subclasses extend this).
    """

    parse_ci_expansions: bool = False
    diagonalize: bool = True
    shift_energies: bool = True
    include_spin_matrices: bool = True
    include_angular_momentum_matrices: bool = True
    include_electric_dipole_moment_matrices: bool = False


@dataclass(frozen=True, slots=True)
class CIDeterminantExpansion:
    """Spin-determinant CI expansion for one multiplicity block."""

    alpha_occupations: np.ndarray
    beta_occupations: np.ndarray
    ci_coefficients: np.ndarray


@dataclass(frozen=True, slots=True)
class HamiltonianReaderResult:
    """
    Structured Hamiltonian content before conversion to an xarray group.

    Exactly one of ``hamiltonian_matrix`` (CI representation) or
    ``states_energies`` (diagonal / eigenstate representation) must be set.
    Optional operator matrices (S, L, electric dipole) and CI expansions may
    be omitted; downstream code should check their presence on the dataset.
    When present, operators use the same basis as the primary Hamiltonian data.
    """

    hamiltonian_interaction: HamiltonianInteractionKind
    representation: HamiltonianRepresentationKind
    hamiltonian_matrix: np.ndarray | None = None
    states_energies: np.ndarray | None = None
    spin_matrices: np.ndarray | None = None
    angular_momentum_matrices: np.ndarray | None = None
    electric_dipole_moment_matrices: np.ndarray | None = None
    ci_expansions_by_multiplicity: dict[int, CIDeterminantExpansion] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


class HamiltonianReader(ABC):
    """
    Base class for Hamiltonian readers: implement :meth:`read` only.

    Composed storage payload uses :func:`hamiltonian_reader_result_to_slt_results`;
    writing uses :meth:`write_to_group` or :func:`write_hamiltonian_reader_result_to_slt_group`.
    """

    @abstractmethod
    def read(self, source: PathLike | Iterable[str]) -> HamiltonianReaderResult:
        """Parse *source* into a :class:`HamiltonianReaderResult`."""

    def read_as_slt_results(self, source: PathLike | Iterable[str]) -> SltResults:
        """Compose :meth:`read` into :class:`~slothpy.core.slt.SltResults` for SlothPy storage."""
        return hamiltonian_reader_result_to_slt_results(self.read(source))

    def write_to_group(
        self,
        source: PathLike | Iterable[str],
        slt: SltFile,
        group_name: str,
        *,
        overwrite: bool = False,
        encoding: dict[str, Any] | None = None,
    ) -> SltGroup:
        """Read *source* and write one Hamiltonian semantic group on *slt*."""
        return write_hamiltonian_reader_result_to_slt_group(
            slt,
            group_name,
            self.read(source),
            overwrite=overwrite,
            encoding=encoding,
        )


def write_slt_results_to_group(
    slt: SltFile,
    group_name: str,
    results: SltResults,
    *,
    overwrite: bool = False,
    encoding: dict[str, Any] | None = None,
) -> SltGroup:
    """Write :class:`~slothpy.core.slt.SltResults` as one root-level SlothPy semantic group."""
    return slt._write_slothpy_group(
        group_name, results, overwrite=overwrite, encoding=encoding
    )


def write_hamiltonian_reader_result_to_slt_group(
    slt: SltFile,
    group_name: str,
    structured: HamiltonianReaderResult,
    *,
    overwrite: bool = False,
    slt_type: str = "HAMILTONIAN",
    encoding: dict[str, Any] | None = None,
) -> SltGroup:
    """Compose structured Hamiltonian data and write it as one SlothPy group."""
    composed = hamiltonian_reader_result_to_slt_results(structured, slt_type=slt_type)
    return write_slt_results_to_group(
        slt, group_name, composed, overwrite=overwrite, encoding=encoding
    )


def _dim_from_result(result: HamiltonianReaderResult) -> int:
    if result.representation == "CI":
        if result.hamiltonian_matrix is None:
            raise ValueError("representation='CI' requires hamiltonian_matrix.")
        n = int(result.hamiltonian_matrix.shape[0])
        if result.hamiltonian_matrix.shape != (n, n):
            raise ValueError(
                "hamiltonian_matrix must be square; got "
                f"{result.hamiltonian_matrix.shape}."
            )
        return n
    if result.states_energies is None:
        raise ValueError("representation='DIAGONAL' requires states_energies.")
    n = int(result.states_energies.shape[0])
    return n


def _validate_operator_stack(name: str, arr: np.ndarray, dim: int) -> None:
    if arr.shape != (3, dim, dim):
        raise ValueError(f"{name} must have shape (3, {dim}, {dim}); got {arr.shape}.")


def _validate_optional_operator_stack(
    name: str, arr: np.ndarray | None, dim: int
) -> None:
    if arr is not None:
        _validate_operator_stack(name, arr, dim)


def add_ci_expansion_variables_to_dataset(
    dataset: xr.Dataset, ci_expansions: dict[int, CIDeterminantExpansion]
) -> xr.Dataset:
    """Attach CI expansion variables and coordinates (SlothPy Hamiltonian layout)."""
    out = dataset.copy(deep=False)
    for mult, expansion in ci_expansions.items():
        det_dim = f"determinant_mult_{mult}"
        root_dim = f"root_mult_{mult}"
        orbital_dim = f"active_orbital_mult_{mult}"
        out.coords[det_dim] = np.arange(
            expansion.alpha_occupations.shape[0], dtype=np.int64
        )
        out.coords[root_dim] = np.arange(
            expansion.ci_coefficients.shape[1], dtype=np.int64
        )
        out.coords[orbital_dim] = np.arange(
            expansion.alpha_occupations.shape[1], dtype=np.int64
        )
        out[f"ci_alpha_occupations_mult_{mult}"] = (
            (det_dim, orbital_dim),
            expansion.alpha_occupations,
            {"long_name": f"alpha spin occupations for multiplicity {mult}"},
        )
        out[f"ci_beta_occupations_mult_{mult}"] = (
            (det_dim, orbital_dim),
            expansion.beta_occupations,
            {"long_name": f"beta spin occupations for multiplicity {mult}"},
        )
        out[f"ci_coefficients_mult_{mult}"] = (
            (det_dim, root_dim),
            expansion.ci_coefficients,
            {"long_name": f"spin-determinant CI coefficients for multiplicity {mult}"},
        )
    return out


def hamiltonian_reader_result_to_slt_results(
    result: HamiltonianReaderResult,
    *,
    slt_type: str = "HAMILTONIAN",
) -> SltResults:
    """
    Build :class:`~slothpy.core.slt.SltResults` from structured Hamiltonian data.

    This is the single place that defines how Hamiltonian groups are laid out
    in SlothPy; program-specific readers only fill :class:`HamiltonianReaderResult`.
    """
    dim = _dim_from_result(result)
    _validate_optional_operator_stack("spin_matrices", result.spin_matrices, dim)
    _validate_optional_operator_stack(
        "angular_momentum_matrices", result.angular_momentum_matrices, dim
    )
    _validate_optional_operator_stack(
        "electric_dipole_moment_matrices",
        result.electric_dipole_moment_matrices,
        dim,
    )

    has_operator_matrices = (
        result.spin_matrices is not None
        or result.angular_momentum_matrices is not None
        or result.electric_dipole_moment_matrices is not None
    )

    state_coord = np.arange(dim, dtype=np.int64)

    data_vars: dict[str, Any]
    if result.representation == "CI":
        if result.states_energies is not None:
            raise ValueError(
                "representation='CI' must not set states_energies; "
                "use hamiltonian_matrix only."
            )
        matrix = result.hamiltonian_matrix
        assert matrix is not None
        primary = (
            "soc_ssc_matrix"
            if result.hamiltonian_interaction == "SOC_SSC"
            else "soc_matrix"
        )
        bra_dim = "ci_bra_state"
        ket_dim = "ci_ket_state"
        long_h = (
            "SOC+SSC matrix in CI basis"
            if result.hamiltonian_interaction == "SOC_SSC"
            else "SOC matrix in CI basis"
        )
        data_vars = {
            primary: (
                (bra_dim, ket_dim),
                matrix,
                {"unit": "E_h", "long_name": long_h},
            )
        }
    else:
        if result.hamiltonian_matrix is not None:
            raise ValueError(
                "representation='DIAGONAL' must not set hamiltonian_matrix; "
                "use states_energies only."
            )
        energies = result.states_energies
        assert energies is not None
        primary = "states_energies"
        state_dim = "state"
        bra_dim = "bra_state"
        ket_dim = "ket_state"
        shift = bool(result.attrs.get("shift_energies_applied", False))
        data_vars = {
            "states_energies": (
                (state_dim,),
                energies.astype(np.float64, copy=False),
                {
                    "unit": "E_h",
                    "long_name": (
                        "SOC eigenstate energies shifted to the lowest state"
                        if shift
                        else "SOC eigenstate energies"
                    ),
                },
            )
        }

    attrs = dict(result.attrs)
    coords: dict[str, Any] = {}

    if result.representation == "DIAGONAL":
        coords[state_dim] = state_coord

    if result.representation == "CI":
        coords[bra_dim] = state_coord
        coords[ket_dim] = state_coord

    if has_operator_matrices:
        op_bra, op_ket = bra_dim, ket_dim
        coords[op_bra] = state_coord
        coords[op_ket] = state_coord
        coords["component"] = np.array(["x", "y", "z"], dtype=object)

        if result.spin_matrices is not None:
            data_vars["spin_matrices"] = (
                ("component", op_bra, op_ket),
                result.spin_matrices,
                {"long_name": "spin matrices", "component_order": "x,y,z"},
            )
        if result.angular_momentum_matrices is not None:
            data_vars["angular_momentum_matrices"] = (
                ("component", op_bra, op_ket),
                result.angular_momentum_matrices,
                {
                    "long_name": "orbital angular momentum matrices",
                    "component_order": "x,y,z",
                },
            )
        if result.electric_dipole_moment_matrices is not None:
            data_vars["electric_dipole_moment_matrices"] = (
                ("component", op_bra, op_ket),
                result.electric_dipole_moment_matrices,
                {
                    "long_name": "electric dipole moment matrices",
                    "component_order": "x,y,z",
                },
            )

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    if result.ci_expansions_by_multiplicity is not None:
        dataset = add_ci_expansion_variables_to_dataset(
            dataset, result.ci_expansions_by_multiplicity
        )

    return SltResults(
        dataset=dataset,
        slt_type=slt_type,
        primary=primary,
        attrs=attrs,
    )


__all__ = [
    "CIDeterminantExpansion",
    "HamiltonianInteractionKind",
    "HamiltonianReader",
    "HamiltonianReaderOptions",
    "HamiltonianReaderResult",
    "HamiltonianRepresentationKind",
    "add_ci_expansion_variables_to_dataset",
    "hamiltonian_reader_result_to_slt_results",
    "write_hamiltonian_reader_result_to_slt_group",
    "write_slt_results_to_group",
]
