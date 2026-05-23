from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from pydantic import ConfigDict

from slothpy.compute.mpi_job import MPIJobResources, MPIJobRunner, MPIJobSpec
from slothpy.core.slt_computation import (
    SltComputation,
    SltComputationResources,
)
from slothpy.core.slt_results import SltResults, SltResultView
from slothpy.groups.hamiltonian import SltHamiltonianGroup
from slothpy.io.shared_memory import SharedArrayBundle

_VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True, slots=True)
class MagnetisationOptions:
    magnetic_fields: np.ndarray
    orientations: np.ndarray
    temperatures: np.ndarray
    states_cutoff: tuple[int, int] = (0, 0)
    rotation: np.ndarray | None = None
    electric_field_vector: np.ndarray | None = None


class MagnetisationVar(StrEnum):
    MAGNETISATION = "magnetisation"
    ORIENTATIONS = "orientations"


class MagnetisationCoord(StrEnum):
    TEMPERATURE = "temperature"
    FIELD = "field"
    ORIENTATION = "orientation"


class MagnetisationDataMixin:
    expected_slt_type: ClassVar[str] = "MAGNETISATION"

    @property
    def dataset(self) -> xr.Dataset:
        raise NotImplementedError

    @property
    def magnetisation(self) -> xr.DataArray:
        return self.dataset[MagnetisationVar.MAGNETISATION]

    @property
    def temperatures(self) -> xr.DataArray:
        return self.dataset[MagnetisationCoord.TEMPERATURE]

    @property
    def magnetic_fields(self) -> xr.DataArray:
        return self.dataset[MagnetisationCoord.FIELD]

    @property
    def orientations(self) -> xr.DataArray | None:
        dataset = self.dataset
        if MagnetisationVar.ORIENTATIONS in dataset:
            return dataset[MagnetisationVar.ORIENTATIONS]
        return None

    def to_dataframe(self) -> Any:
        return self.magnetisation.to_dataframe()


@dataclass(frozen=True, slots=True)
class SltMagnetisationResult(MagnetisationDataMixin, SltResultView):
    expected_slt_type: ClassVar[str] = "MAGNETISATION"


@dataclass(slots=True)
class SltMagnetisationComputation(
    SltComputation[MagnetisationOptions, SltMagnetisationResult]
):
    computation_name: ClassVar[str] = "MagnetisationComputation"

    def _compute(self) -> SltResults:
        average = self.options.orientations.shape[1] == 4

        with TemporaryDirectory(prefix="slothpy-mpi-magnetisation-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            manifest_path = tmpdir_path / "shared_memory.json"
            job_spec_path = tmpdir_path / "mpi_job.json"

            bundle = self._stage_shared_arrays(average=average)

            try:
                bundle.write_manifest(manifest_path)

                spec = MPIJobSpec(
                    worker_module="slothpy.compute.workers.magnetisation_worker",
                    shared_memory_manifest=str(manifest_path),
                    payload={
                        "n_fields": int(self.options.magnetic_fields.shape[0]),
                        "n_orientations": int(self.options.orientations.shape[0]),
                        "n_temperatures": int(self.options.temperatures.shape[0]),
                        "average": average,
                    },
                )
                spec.write_json(job_spec_path)

                runner = MPIJobRunner(
                    resources=MPIJobResources(
                        num_processes=self.resources.num_processes,
                        num_threads=self.resources.num_threads,
                    )
                )
                runner.run(spec, job_spec_path)

                result = bundle["result"].copy()

            finally:
                bundle.release()

        return self._compose_results(result, average=average)

    def _stage_shared_arrays(self, *, average: bool) -> SharedArrayBundle:
        source = self.source
        if not isinstance(source, SltHamiltonianGroup):
            raise TypeError(
                "MagnetisationComputation source must be SltHamiltonianGroup."
            )

        n_fields = self.options.magnetic_fields.shape[0]
        n_orientations = self.options.orientations.shape[0]
        n_temperatures = self.options.temperatures.shape[0]

        result_shape: tuple[int, ...]
        if average:
            result_shape = (n_fields, n_temperatures)
        else:
            result_shape = (n_orientations, n_fields, n_temperatures)

        bundle = SharedArrayBundle()

        # These paths assume xarray/h5netcdf writes variables directly under
        # the semantic group. If your final constants are different, keep them
        # in one HamiltonianVar enum and use it here.
        group_path = source.group_name

        bundle.add_hdf5_dataset(
            "states_energies",
            source.file_path,
            f"{group_path}/states_energies",
            dtype=np.float64,
            readonly=True,
        )
        bundle.add_hdf5_dataset(
            "spin_matrices",
            source.file_path,
            f"{group_path}/spin_matrices",
            dtype=np.complex128,
            readonly=True,
        )
        bundle.add_hdf5_dataset(
            "angular_momentum_matrices",
            source.file_path,
            f"{group_path}/angular_momentum_matrices",
            dtype=np.complex128,
            readonly=True,
        )

        bundle.add_array(
            "magnetic_fields",
            np.asarray(self.options.magnetic_fields, dtype=np.float64),
            readonly=True,
        )
        bundle.add_array(
            "orientations",
            np.asarray(self.options.orientations, dtype=np.float64),
            readonly=True,
        )
        bundle.add_array(
            "temperatures",
            np.asarray(self.options.temperatures, dtype=np.float64),
            readonly=True,
        )

        bundle.add_empty(
            "result",
            result_shape,
            dtype=np.float64,
            readonly=False,
        )

        return bundle

    def _compose_results(self, result: np.ndarray, *, average: bool) -> SltResults:
        fields = self.options.magnetic_fields
        temperatures = self.options.temperatures
        orientations = self.options.orientations

        if average:
            # Worker writes result as (field, temperature).
            # Stored result uses (temperature, field), which is nicer for xarray.
            magnetisation = result.T

            dataset = xr.Dataset(
                data_vars={
                    MagnetisationVar.MAGNETISATION: (
                        (
                            MagnetisationCoord.TEMPERATURE,
                            MagnetisationCoord.FIELD,
                        ),
                        magnetisation,
                        {
                            "long_name": "powder-averaged magnetisation",
                        },
                    )
                },
                coords={
                    MagnetisationCoord.TEMPERATURE: (
                        MagnetisationCoord.TEMPERATURE,
                        temperatures,
                        {"unit": "K"},
                    ),
                    MagnetisationCoord.FIELD: (
                        MagnetisationCoord.FIELD,
                        fields,
                        {"unit": "T"},
                    ),
                },
                attrs={
                    "slt_kind": "AVERAGE",
                    "source_group": self.source.group_name,
                },
            )
        else:
            # Worker writes result as (orientation, field, temperature).
            # Stored result uses (orientation, temperature, field).
            magnetisation = np.transpose(result, (0, 2, 1))

            dataset = xr.Dataset(
                data_vars={
                    MagnetisationVar.MAGNETISATION: (
                        (
                            MagnetisationCoord.ORIENTATION,
                            MagnetisationCoord.TEMPERATURE,
                            MagnetisationCoord.FIELD,
                        ),
                        magnetisation,
                        {
                            "long_name": "directional magnetisation",
                        },
                    )
                },
                coords={
                    MagnetisationCoord.ORIENTATION: np.arange(
                        orientations.shape[0], dtype=np.int64
                    ),
                    MagnetisationCoord.TEMPERATURE: (
                        MagnetisationCoord.TEMPERATURE,
                        temperatures,
                        {"unit": "K"},
                    ),
                    MagnetisationCoord.FIELD: (
                        MagnetisationCoord.FIELD,
                        fields,
                        {"unit": "T"},
                    ),
                },
                attrs={
                    "slt_kind": "DIRECTIONAL",
                    "source_group": self.source.group_name,
                },
            )

            dataset[MagnetisationVar.ORIENTATIONS] = (
                (MagnetisationCoord.ORIENTATION, "orientation_component"),
                orientations,
                {"long_name": "magnetic-field orientation grid"},
            )

        return SltResults(
            dataset=dataset,
            slt_type="MAGNETISATION",
            primary=MagnetisationVar.MAGNETISATION,
            attrs=dict(dataset.attrs),
        )


def magnetisation(
    source: SltHamiltonianGroup,
    magnetic_fields: ArrayLike,
    orientations: ArrayLike,
    temperatures: ArrayLike,
    *,
    states_cutoff: tuple[int, int] = (0, 0),
    rotation: ArrayLike | None = None,
    electric_field_vector: ArrayLike | None = None,
    num_processes: int = 1,
    num_threads: int = 1,
    save_as: str | None = None,
    output_slt: Any = None,
    overwrite: bool = False,
) -> SltMagnetisationComputation:
    options = MagnetisationOptions(
        magnetic_fields=np.asarray(magnetic_fields, dtype=np.float64),
        orientations=np.asarray(orientations, dtype=np.float64),
        temperatures=np.asarray(temperatures, dtype=np.float64),
        states_cutoff=states_cutoff,
        rotation=None if rotation is None else np.asarray(rotation, dtype=np.float64),
        electric_field_vector=(
            None
            if electric_field_vector is None
            else np.asarray(electric_field_vector, dtype=np.float64)
        ),
    )

    return SltMagnetisationComputation(
        source=source,
        options=options,
        resources=SltComputationResources(
            num_processes=num_processes,
            num_threads=num_threads,
        ),
        save_as=save_as,
        output_slt=output_slt,
        overwrite=overwrite,
    )
