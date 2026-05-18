from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import h5py
import numpy as np
from pydantic import ConfigDict, Field, validate_call

from .hamiltonian_reader import (
    HamiltonianReader,
    HamiltonianReaderOptions,
    HamiltonianReaderResult,
)

try:
    from slothpy.core.slt import SltFile, SltPathOrFile
    from slothpy.types.aliases import PathLike
except Exception:  # pragma: no cover - for standalone parser tests outside SlothPy
    SltFile = Any  # type: ignore[misc,assignment]
    SltPathOrFile = Any  # type: ignore[misc,assignment]
    PathLike = str | Path  # type: ignore[misc,assignment]


@dataclass(frozen=True, slots=True)
class MolcasHamiltonianReaderOptions(HamiltonianReaderOptions):
    """Options controlling OpenMolcas/MOLCAS RASSI HDF5 Hamiltonian parsing."""


@dataclass(slots=True)
class MolcasHamiltonianReader(HamiltonianReader):
    """
    Read OpenMolcas/MOLCAS RASSI HDF5 output into structured Hamiltonian data.

    Only the spin-orbit-state / diagonal representation is supported. The reader
    expects already prepared spin-orbit-state datasets such as ``SOS_ENERGIES``,
    ``SOS_SPIN_REAL``, ``SOS_SPIN_IMAG``, ``SOS_ANGMOM_REAL``,
    ``SOS_ANGMOM_IMAG``, and optionally ``SOS_EDIPMOM_REAL`` /
    ``SOS_EDIPMOM_IMAG``.
    """

    options: MolcasHamiltonianReaderOptions = MolcasHamiltonianReaderOptions()

    def read(self, source: PathLike | Iterable[str]) -> HamiltonianReaderResult:
        if not isinstance(source, (str, Path)):
            raise TypeError(
                "MOLCAS Hamiltonian reader requires a file path, not a line iterable."
            )

        if self.options.ci_basis:
            raise NotImplementedError(
                "MOLCAS Hamiltonian reader currently supports only the "
                "spin-orbit-state / diagonal representation; ci_basis=True "
                "is not implemented."
            )

        if self.options.parse_ci_expansions:
            raise NotImplementedError(
                "MOLCAS Hamiltonian reader does not parse CI determinant "
                "expansions; parse_ci_expansions=True is not implemented."
            )

        source_path = Path(source)

        with h5py.File(source_path, "r") as rassi:
            state_energies = _read_state_energies(
                rassi,
                shift_energies=self.options.shift_energies,
            )
            dim = int(state_energies.shape[0])

            spin_matrices: np.ndarray | None = None
            if self.options.include_spin_matrices:
                spin_matrices = _read_complex_operator_stack(
                    rassi,
                    real_name="SOS_SPIN_REAL",
                    imag_name="SOS_SPIN_IMAG",
                    output_name="spin_matrices",
                    dim=dim,
                )

            angular_momentum_matrices: np.ndarray | None = None
            if self.options.include_angular_momentum_matrices:
                angular_momentum_matrices = _read_molcas_angular_momentum_stack(
                    rassi,
                    dim=dim,
                )

            electric_dipole_moment_matrices: np.ndarray | None = None
            if self.options.include_electric_dipole_moment_matrices:
                electric_dipole_moment_matrices = _read_complex_operator_stack(
                    rassi,
                    real_name="SOS_EDIPMOM_REAL",
                    imag_name="SOS_EDIPMOM_IMAG",
                    output_name="electric_dipole_moment_matrices",
                    dim=dim,
                )

            attrs = _read_molcas_attrs(rassi, dim=dim)
            attrs["shift_energies_applied"] = self.options.shift_energies

        return HamiltonianReaderResult(
            hamiltonian_interaction="SOC",
            representation="DIAGONAL",
            state_energies=state_energies,
            spin_matrices=spin_matrices,
            angular_momentum_matrices=angular_momentum_matrices,
            electric_dipole_moment_matrices=electric_dipole_moment_matrices,
            attrs=attrs,
        )


def _decode_hdf5_attr(value: Any) -> Any:
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8").strip()

    return value


def _read_molcas_attrs(rassi: h5py.File, *, dim: int) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "slt_kind": "MOLCAS",
        "hamiltonian_type": "SOC",
        "basis": "DIAGONAL",
        "states": dim,
        "source_format": "OpenMolcas/MOLCAS RASSI HDF5 output",
    }

    molcas_module = rassi.attrs.get("MOLCAS_MODULE")
    if molcas_module is not None:
        attrs["molcas_module"] = _decode_hdf5_attr(molcas_module)

    molcas_version = rassi.attrs.get("MOLCAS_VERSION")
    if molcas_version is not None:
        attrs["molcas_version"] = _decode_hdf5_attr(molcas_version)

    for key in (
        "NSTATE",
        "NSYM",
        "STATE_SPINMULT",
        "STATE_IRREPS",
        "STATE_LROOT",
    ):
        if key in rassi.attrs:
            attrs[key.lower()] = rassi.attrs[key]

    return attrs


def _read_required_dataset(rassi: h5py.File, name: str) -> h5py.Dataset:
    if name not in rassi:
        raise KeyError(f"Required MOLCAS/RASSI dataset {name!r} is missing.")

    dataset = rassi[name]

    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"MOLCAS/RASSI object {name!r} is not a dataset.")

    return dataset


def _read_state_energies(
    rassi: h5py.File,
    *,
    shift_energies: bool,
) -> np.ndarray:
    dataset = _read_required_dataset(rassi, "SOS_ENERGIES")
    energies = np.asarray(dataset[()], dtype=np.float64)

    if energies.ndim != 1:
        raise ValueError(
            f"SOS_ENERGIES must be a 1D array; got shape {energies.shape}."
        )

    if energies.size == 0:
        raise ValueError("SOS_ENERGIES is empty.")

    if shift_energies:
        energies = energies - np.min(energies)

    return energies.astype(np.float64, copy=False)


def _read_complex_operator_stack(
    rassi: h5py.File,
    *,
    real_name: str,
    imag_name: str,
    output_name: str,
    dim: int,
) -> np.ndarray:
    real = np.asarray(_read_required_dataset(rassi, real_name)[()], dtype=np.float64)
    imag = np.asarray(_read_required_dataset(rassi, imag_name)[()], dtype=np.float64)

    if real.shape != imag.shape:
        raise ValueError(
            f"{real_name!r} and {imag_name!r} must have the same shape; "
            f"got {real.shape} and {imag.shape}."
        )

    matrix = real + 1j * imag
    _validate_operator_stack(output_name, matrix, dim)

    return matrix.astype(np.complex128, copy=False)


def _read_molcas_angular_momentum_stack(
    rassi: h5py.File,
    *,
    dim: int,
) -> np.ndarray:
    """
    Read angular momentum matrices from MOLCAS/RASSI.

    MOLCAS stores ``SOS_ANGMOM_*`` as matrix elements of ``iL``.

        angular_momentum = 1j * SOS_ANGMOM_REAL - SOS_ANGMOM_IMAG

    which is equivalent to multiplying the stored complex ``iL`` matrix by
    ``1j``.
    """
    stored_i_l = _read_complex_operator_stack(
        rassi,
        real_name="SOS_ANGMOM_REAL",
        imag_name="SOS_ANGMOM_IMAG",
        output_name="angular_momentum_matrices",
        dim=dim,
    )

    angular_momentum = 1j * stored_i_l
    _validate_operator_stack("angular_momentum_matrices", angular_momentum, dim)

    return angular_momentum.astype(np.complex128, copy=False)


def _validate_operator_stack(name: str, arr: np.ndarray, dim: int) -> None:
    expected = (3, dim, dim)

    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {arr.shape}.")


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def hamiltonian_from_molcas(
    molcas_filepath: PathLike,
    slt_path_or_file: SltPathOrFile,
    group_name: Annotated[str, Field(min_length=1)],
    *,
    shift_energies: bool = True,
    include_spin_matrices: bool = True,
    include_angular_momentum_matrices: bool = True,
    include_electric_dipole_moment_matrices: bool = False,
    ci_basis: bool = False,
    parse_ci_expansions: bool = False,
    overwrite: bool = False,
) -> SltFile:
    """
    Read a MOLCAS/OpenMolcas RASSI HDF5 file and write a SlothPy Hamiltonian group.

    Parameters
    ----------
    molcas_filepath
        Path to the MOLCAS/OpenMolcas ``.rassi.h5`` file.
    slt_path_or_file
        Target ``.slt`` file path or an open :class:`~slothpy.core.slt.SltFile`
        handle. Paths open an existing file or create one if missing.
    group_name
        Root-level SlothPy group name for the Hamiltonian data.
    shift_energies
        Shift spin-orbit-state energies so that the lowest state has energy zero.
    include_spin_matrices
        Read ``SOS_SPIN_REAL`` and ``SOS_SPIN_IMAG``.
    include_angular_momentum_matrices
        Read ``SOS_ANGMOM_REAL`` and ``SOS_ANGMOM_IMAG``.
    include_electric_dipole_moment_matrices
        Read ``SOS_EDIPMOM_REAL`` and ``SOS_EDIPMOM_IMAG``.
    ci_basis
        Not implemented for MOLCAS. MOLCAS RASSI HDF5 reading currently supports
        only the spin-orbit-state / diagonal representation.
    parse_ci_expansions
        Not implemented for MOLCAS.
    overwrite
        Replace an existing SlothPy group with the same name.
    """
    if ci_basis:
        raise NotImplementedError(
            "MOLCAS Hamiltonian reader currently supports only the "
            "spin-orbit-state / diagonal representation; ci_basis=True "
            "is not implemented."
        )

    if parse_ci_expansions:
        raise NotImplementedError(
            "MOLCAS Hamiltonian reader does not parse CI determinant expansions; "
            "parse_ci_expansions=True is not implemented."
        )

    reader = MolcasHamiltonianReader(
        MolcasHamiltonianReaderOptions(
            parse_ci_expansions=parse_ci_expansions,
            shift_energies=shift_energies,
            include_spin_matrices=include_spin_matrices,
            include_angular_momentum_matrices=include_angular_momentum_matrices,
            include_electric_dipole_moment_matrices=(
                include_electric_dipole_moment_matrices
            ),
            ci_basis=ci_basis,
        )
    )

    result = reader.read(molcas_filepath)
    slt = cast(SltFile, slt_path_or_file)

    result.write_to_slt_group(slt, group_name, overwrite=overwrite)

    return slt


__all__ = [
    "HamiltonianReader",
    "HamiltonianReaderResult",
    "MolcasHamiltonianReader",
    "MolcasHamiltonianReaderOptions",
    "hamiltonian_from_molcas",
]
