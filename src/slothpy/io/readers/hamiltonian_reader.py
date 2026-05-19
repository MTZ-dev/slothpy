from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from slothpy.core.slt_file import SltFile
from slothpy.core.slt_group import SltGroup
from slothpy.core.slt_results import SltResults
from slothpy.groups.hamiltonian_names import HamiltonianCoord, HamiltonianVar
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
    shift_energies: bool = False
    include_spin_matrices: bool = False
    include_angular_momentum_matrices: bool = False
    include_electric_dipole_moment_matrices: bool = False
    ci_basis: bool = False


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
    ``state_energies`` (diagonal / eigenstate representation) must be set.
    Optional operator matrices (S, L, electric dipole) and CI expansions may
    be omitted; downstream code should check their presence on the dataset.
    When present, operators use the same basis as the primary Hamiltonian data.
    """

    hamiltonian_interaction: HamiltonianInteractionKind
    representation: HamiltonianRepresentationKind
    hamiltonian_matrix: np.ndarray | None = None
    state_energies: np.ndarray | None = None
    spin_matrices: np.ndarray | None = None
    angular_momentum_matrices: np.ndarray | None = None
    electric_dipole_moment_matrices: np.ndarray | None = None
    ci_expansions_by_multiplicity: dict[int, CIDeterminantExpansion] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_slt_results(
        self,
        *,
        slt_type: str = "HAMILTONIAN",
    ) -> SltResults:
        """
        Build :class:`~slothpy.core.slt.SltResults` from structured Hamiltonian data.

        This is the single place that defines how Hamiltonian groups are laid out
        in SlothPy; program-specific readers only fill :class:`HamiltonianReaderResult`.
        """
        dim = _dim_from_result(self)
        _validate_optional_operator_stack(
            HamiltonianVar.SPIN_MATRICES, self.spin_matrices, dim
        )
        _validate_optional_operator_stack(
            HamiltonianVar.ANGULAR_MOMENTUM_MATRICES,
            self.angular_momentum_matrices,
            dim,
        )
        _validate_optional_operator_stack(
            HamiltonianVar.ELECTRIC_DIPOLE_MOMENT_MATRICES,
            self.electric_dipole_moment_matrices,
            dim,
        )

        has_operator_matrices = (
            self.spin_matrices is not None
            or self.angular_momentum_matrices is not None
            or self.electric_dipole_moment_matrices is not None
        )

        state_coord = np.arange(dim, dtype=np.int64)

        data_vars: dict[str, Any]
        if self.representation == "CI":
            if self.state_energies is not None:
                raise ValueError(
                    "representation='CI' must not set state_energies; "
                    "use hamiltonian_matrix only."
                )
            matrix = self.hamiltonian_matrix
            assert matrix is not None
            primary_var = (
                HamiltonianVar.SOC_SSC_MATRIX.value
                if self.hamiltonian_interaction == "SOC_SSC"
                else HamiltonianVar.SOC_MATRIX.value
            )
            bra_dim = HamiltonianCoord.CI_BRA_STATE.value
            ket_dim = HamiltonianCoord.CI_KET_STATE.value
            long_h = (
                "SOC+SSC matrix in CI basis"
                if self.hamiltonian_interaction == "SOC_SSC"
                else "SOC matrix in CI basis"
            )
            data_vars = {
                primary_var: (
                    (bra_dim, ket_dim),
                    matrix,
                    {"unit": "E_h", "long_name": long_h},
                )
            }
        else:
            if self.hamiltonian_matrix is not None:
                raise ValueError(
                    "representation='DIAGONAL' must not set hamiltonian_matrix; "
                    "use state_energies only."
                )
            energies = self.state_energies
            assert energies is not None
            primary_var = HamiltonianVar.STATE_ENERGIES.value
            state_dim = HamiltonianCoord.STATE.value
            bra_dim = HamiltonianCoord.BRA_STATE.value
            ket_dim = HamiltonianCoord.KET_STATE.value
            shift = bool(self.attrs.get("shift_energies_applied", False))
            data_vars = {
                HamiltonianVar.STATE_ENERGIES.value: (
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

        attrs = dict(self.attrs)
        coords: dict[str, Any] = {}

        if self.representation == "DIAGONAL":
            coords[state_dim] = state_coord

        if self.representation == "CI":
            coords[bra_dim] = state_coord
            coords[ket_dim] = state_coord

        if has_operator_matrices:
            op_bra, op_ket = bra_dim, ket_dim
            coords[op_bra] = state_coord
            coords[op_ket] = state_coord
            coords[HamiltonianCoord.COMPONENT.value] = np.array(
                ["x", "y", "z"], dtype=object
            )
            component = HamiltonianCoord.COMPONENT.value

            if self.spin_matrices is not None:
                data_vars[HamiltonianVar.SPIN_MATRICES.value] = (
                    (component, op_bra, op_ket),
                    self.spin_matrices,
                    {"long_name": "spin matrices", "component_order": "x,y,z"},
                )
            if self.angular_momentum_matrices is not None:
                data_vars[HamiltonianVar.ANGULAR_MOMENTUM_MATRICES.value] = (
                    (component, op_bra, op_ket),
                    self.angular_momentum_matrices,
                    {
                        "long_name": "orbital angular momentum matrices",
                        "component_order": "x,y,z",
                    },
                )
            if self.electric_dipole_moment_matrices is not None:
                data_vars[HamiltonianVar.ELECTRIC_DIPOLE_MOMENT_MATRICES.value] = (
                    (component, op_bra, op_ket),
                    self.electric_dipole_moment_matrices,
                    {
                        "long_name": "electric dipole moment matrices",
                        "component_order": "x,y,z",
                    },
                )

        dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        if self.ci_expansions_by_multiplicity is not None:
            dataset = add_ci_expansion_variables_to_dataset(
                dataset, self.ci_expansions_by_multiplicity
            )

        return SltResults(
            dataset=dataset,
            slt_type=slt_type,
            primary=primary_var,
            attrs=attrs,
        )

    def write_to_slt_group(
        self,
        slt: SltFile,
        group_name: str,
        *,
        slt_type: str = "HAMILTONIAN",
        overwrite: bool = False,
        encoding: dict[str, Any] | None = None,
    ) -> SltGroup:
        """Compose :class:`HamiltonianReaderResult` and write it as one SlothPy group."""
        return slt._write_slothpy_group(
            group_name,
            self.to_slt_results(slt_type=slt_type),
            overwrite=overwrite,
            encoding=encoding,
        )


class HamiltonianReader(ABC):
    """
    Base class for Hamiltonian readers: implement :meth:`read` only.

    Storage uses :meth:`HamiltonianReaderResult.to_slt_results` on the object
    returned by :meth:`read`; writing uses :meth:`HamiltonianReaderResult.write_to_slt_group`
    or :meth:`write_to_group`.
    """

    @abstractmethod
    def read(self, source: PathLike | Iterable[str]) -> HamiltonianReaderResult:
        """Parse *source* into a :class:`HamiltonianReaderResult`."""

    def read_as_slt_results(self, source: PathLike | Iterable[str]) -> SltResults:
        """Compose :meth:`read` into :class:`~slothpy.core.slt.SltResults` for SlothPy storage."""
        return self.read(source).to_slt_results()

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
        return self.read(source).write_to_slt_group(
            slt,
            group_name,
            overwrite=overwrite,
            encoding=encoding,
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
    if result.state_energies is None:
        raise ValueError("representation='DIAGONAL' requires state_energies.")
    n = int(result.state_energies.shape[0])
    return n


def _validate_operator_stack(name: HamiltonianVar, arr: np.ndarray, dim: int) -> None:
    if arr.shape != (3, dim, dim):
        raise ValueError(
            f"{name.value} must have shape (3, {dim}, {dim}); got {arr.shape}."
        )


def _validate_optional_operator_stack(
    name: HamiltonianVar, arr: np.ndarray | None, dim: int
) -> None:
    if arr is not None:
        _validate_operator_stack(name, arr, dim)


def add_ci_expansion_variables_to_dataset(
    dataset: xr.Dataset, ci_expansions: dict[int, CIDeterminantExpansion]
) -> xr.Dataset:
    """Attach CI expansion variables and coordinates (SlothPy Hamiltonian layout)."""
    out = dataset.copy(deep=False)
    for mult, expansion in ci_expansions.items():
        det_dim = HamiltonianCoord.determinant_mult(mult)
        root_dim = HamiltonianCoord.root_mult(mult)
        orbital_dim = HamiltonianCoord.active_orbital_mult(mult)
        out.coords[det_dim] = np.arange(
            expansion.alpha_occupations.shape[0], dtype=np.int64
        )
        out.coords[root_dim] = np.arange(
            expansion.ci_coefficients.shape[1], dtype=np.int64
        )
        out.coords[orbital_dim] = np.arange(
            expansion.alpha_occupations.shape[1], dtype=np.int64
        )
        out[HamiltonianVar.ci_alpha_occupations_mult(mult)] = (
            (det_dim, orbital_dim),
            expansion.alpha_occupations,
            {"long_name": f"alpha spin occupations for multiplicity {mult}"},
        )
        out[HamiltonianVar.ci_beta_occupations_mult(mult)] = (
            (det_dim, orbital_dim),
            expansion.beta_occupations,
            {"long_name": f"beta spin occupations for multiplicity {mult}"},
        )
        out[HamiltonianVar.ci_coefficients_mult(mult)] = (
            (det_dim, root_dim),
            expansion.ci_coefficients,
            {"long_name": f"spin-determinant CI coefficients for multiplicity {mult}"},
        )
    return out


__all__ = [
    "CIDeterminantExpansion",
    "HamiltonianInteractionKind",
    "HamiltonianReader",
    "HamiltonianReaderOptions",
    "HamiltonianReaderResult",
    "HamiltonianRepresentationKind",
    "add_ci_expansion_variables_to_dataset",
]
