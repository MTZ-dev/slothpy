from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from pydantic import ConfigDict, validate_call

from slothpy.compute.mpi_job import MPIJobSpec
from slothpy.compute.mpi_progress import THREAD_PROGRESS_KEY, thread_progress_shape
from slothpy.core.slt import create_slt_file
from slothpy.core.slt_computation import (
    SltComputation,
    SltComputationResources,
)
from slothpy.core.slt_results import SltResults
from slothpy.core.slt_session import SltAllocation
from slothpy.groups.hamiltonian import SltHamiltonianGroup
from slothpy.groups.hamiltonian_names import HamiltonianVar
from slothpy.io.readers.hamiltonian_reader import HamiltonianReaderResult
from slothpy.io.shared_memory import SharedArrayBundle
from slothpy.specs.magnetisation import (
    MAGNETISATION_SLT_TYPE,
    MagnetisationCoord,
    MagnetisationOptions,
    MagnetisationVar,
    SltMagnetisationResult,
)
from slothpy.types.aliases import PathLike

_VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


@dataclass(slots=True)
class SltMagnetisationComputation(
    SltComputation[MagnetisationOptions, SltMagnetisationResult]
):
    computation_name: ClassVar[str] = "MagnetisationComputation"

    def _task_count(self) -> int:
        average = self.options.orientations.shape[1] == 4
        n_fields = self.options.magnetic_fields.shape[0]
        n_orientations = self.options.orientations.shape[0]
        return n_fields if average else n_fields * n_orientations

    def _total_work_units(self) -> int:
        return self._task_count() * self.options.steps_per_task

    def _before_run(self) -> None:
        self._ensure_progress_tracker(total=self._total_work_units())

    def _compute(self, *, allocation: SltAllocation) -> SltResults:
        average = self.options.orientations.shape[1] == 4

        with TemporaryDirectory(prefix="slothpy-mpi-magnetisation-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            manifest_path = tmpdir_path / "shared_memory.json"
            job_spec_path = tmpdir_path / "mpi_job.json"

            bundle = self._stage_shared_arrays(
                average=average,
                allocation=allocation,
            )

            try:
                bundle.write_manifest(manifest_path)

                request = self.resource_request
                payload: dict[str, Any] = {
                    "n_fields": int(self.options.magnetic_fields.shape[0]),
                    "n_orientations": int(self.options.orientations.shape[0]),
                    "n_temperatures": int(self.options.temperatures.shape[0]),
                    "average": average,
                    "n_tasks": self._task_count(),
                    "num_threads": request.num_threads,
                    "steps_per_task": self.options.steps_per_task,
                    "sleep_seconds": self.options.sleep_seconds,
                    "progress_interval_steps": self.options.progress_interval_steps,
                    "total_work": self._total_work_units(),
                }

                if self._progress_tracker is not None:
                    payload["progress"] = self._progress_tracker.spec.to_json_dict()

                spec = MPIJobSpec(
                    worker_module="slothpy.compute.workers.magnetisation",
                    shared_memory_manifest=str(manifest_path),
                    payload=payload,
                )
                spec.write_json(job_spec_path)

                self._run_mpi_process(
                    spec=spec,
                    job_spec_path=job_spec_path,
                    allocation=allocation,
                )

                result = bundle["result"].copy()

            finally:
                bundle.release()

        return self._compose_results(result, average=average)

    def _stage_shared_arrays(
        self,
        *,
        average: bool,
        allocation: SltAllocation,
    ) -> SharedArrayBundle:
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

        group_path = source.group_name

        bundle.add_hdf5_dataset(
            "state_energies",
            source.file_path,
            f"{group_path}/{HamiltonianVar.STATE_ENERGIES.value}",
            dtype=np.float64,
            readonly=True,
        )
        bundle.add_hdf5_dataset(
            "spin_matrices",
            source.file_path,
            f"{group_path}/{HamiltonianVar.SPIN_MATRICES.value}",
            dtype=np.complex128,
            readonly=True,
        )
        bundle.add_hdf5_dataset(
            "angular_momentum_matrices",
            source.file_path,
            f"{group_path}/{HamiltonianVar.ANGULAR_MOMENTUM_MATRICES.value}",
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

        progress_shape = thread_progress_shape(
            num_processes=allocation.num_processes,
            num_threads=allocation.num_threads,
        )
        bundle.add_empty(
            THREAD_PROGRESS_KEY,
            progress_shape,
            dtype=np.int64,
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
            slt_type=MAGNETISATION_SLT_TYPE,
            primary=MagnetisationVar.MAGNETISATION,
            attrs=dict(dataset.attrs),
        )


def create_demo_hamiltonian_group(
    path: PathLike,
    group_name: str = "demo_hamiltonian",
    *,
    n_states: int = 4,
    overwrite: bool = True,
) -> SltHamiltonianGroup:
    """
    Write a tiny diagonal Hamiltonian group for session / MPI demos.
    """
    if n_states < 1:
        raise ValueError("n_states must be >= 1.")

    dim = n_states
    result = HamiltonianReaderResult(
        hamiltonian_interaction="SOC",
        representation="DIAGONAL",
        state_energies=np.linspace(0.0, 0.1, dim, dtype=np.float64),
        spin_matrices=np.zeros((3, dim, dim), dtype=np.complex128),
        angular_momentum_matrices=np.zeros((3, dim, dim), dtype=np.complex128),
        attrs={"source": "slothpy demo"},
    )

    slt = create_slt_file(path, overwrite=overwrite)
    result.write_to_slt_group(slt, group_name, overwrite=overwrite)
    return slt.hamiltonian(group_name)


def demo_magnetisation(
    source: SltHamiltonianGroup,
    *,
    n_fields: int = 3,
    n_orientations: int = 4,
    n_temperatures: int = 2,
    num_processes: int = 2,
    num_threads: int = 2,
    steps_per_task: int = 10,
    sleep_seconds: float = 0.08,
) -> SltMagnetisationComputation:
    """
    Build a magnetisation computation with small grids and slow placeholder work.

    Intended for :class:`~slothpy.core.slt_session.SltSession` dashboard testing.
    """
    rng = np.random.default_rng(0)
    fields = rng.uniform(0.0, 5.0, size=n_fields)
    orientations = rng.normal(size=(n_orientations, 3))
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    temperatures = rng.uniform(1.0, 300.0, size=n_temperatures)

    return magnetisation(
        source,
        fields,
        orientations,
        temperatures,
        num_processes=num_processes,
        num_threads=num_threads,
        steps_per_task=steps_per_task,
        sleep_seconds=sleep_seconds,
        progress_interval_steps=1,
    )


@validate_call(config=_VALIDATE_CONFIG)
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
    steps_per_task: int = 8,
    sleep_seconds: float = 0.05,
    progress_interval_steps: int = 1,
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
        steps_per_task=steps_per_task,
        sleep_seconds=sleep_seconds,
        progress_interval_steps=progress_interval_steps,
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
