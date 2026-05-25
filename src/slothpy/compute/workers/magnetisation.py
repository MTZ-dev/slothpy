from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from mpi4py import MPI

from slothpy.compute.mpi_job import read_mpi_job_spec_from_cli
from slothpy.io.shared_memory import SharedArrayBundle


@dataclass(frozen=True, slots=True)
class RankChunk:
    start: int
    end: int


def _rank_chunk(n_items: int, size: int, rank: int) -> RankChunk:
    chunk_size = n_items // size
    remainder = n_items % size

    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)

    return RankChunk(start=start, end=end)


def _compute_magnetisation_chunk(
    *,
    start: int,
    end: int,
    n_fields: int,
    n_orientations: int,
    states_energies: np.ndarray,
    spin_matrices: np.ndarray,
    angular_momentum_matrices: np.ndarray,
    magnetic_fields: np.ndarray,
    orientations: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Placeholder for the real numerical kernel.

    This function should eventually call a Numba/numba-mpi implementation.
    It returns one row per task, where task index enumerates
    orientation-major, field-minor pairs:

        task = orientation_index * n_fields + field_index

    Returned shape:
        (end - start, n_temperatures)
    """
    out = np.empty((end - start, temperatures.shape[0]), dtype=np.float64)

    for local_index, task_index in enumerate(range(start, end)):
        orientation_index = task_index // n_fields
        field_index = task_index % n_fields

        field = magnetic_fields[field_index]
        orientation = orientations[orientation_index]

        # Replace this placeholder with the real kernel.
        # The expression only makes the worker testable structurally.
        out[local_index, :] = (
            np.linalg.norm(orientation[:3]) * float(field) / (temperatures + 1.0)
            + float(states_energies[0])
            + 0.0 * np.real(spin_matrices[0, 0, 0])
            + 0.0 * np.real(angular_momentum_matrices[0, 0, 0])
        )

    return out


def _write_gathered_result(
    *,
    result: np.ndarray,
    gathered: list[tuple[int, int, np.ndarray]],
    n_fields: int,
    n_orientations: int,
    average: bool,
) -> None:
    if average:
        result[...] = 0.0

    for start, end, partial in gathered:
        for local_index, task_index in enumerate(range(start, end)):
            orientation_index = task_index // n_fields
            field_index = task_index % n_fields

            if average:
                result[field_index, :] += partial[local_index, :] / n_orientations
            else:
                result[orientation_index, field_index, :] = partial[local_index, :]


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    spec = read_mpi_job_spec_from_cli()
    payload = spec.payload

    bundle: SharedArrayBundle | None = None

    try:
        if rank == 0:
            bundle = SharedArrayBundle.attach_from_manifest(
                spec.shared_memory_manifest,
                track=False,
            )

            arrays: dict[str, Any] = {
                "states_energies": np.asarray(bundle["states_energies"].array),
                "spin_matrices": np.asarray(bundle["spin_matrices"].array),
                "angular_momentum_matrices": np.asarray(
                    bundle["angular_momentum_matrices"].array
                ),
                "magnetic_fields": np.asarray(bundle["magnetic_fields"].array),
                "orientations": np.asarray(bundle["orientations"].array),
                "temperatures": np.asarray(bundle["temperatures"].array),
            }
        else:
            arrays = {}

        arrays = comm.bcast(arrays, root=0)

        n_fields = int(payload["n_fields"])
        n_orientations = int(payload["n_orientations"])
        average = bool(payload["average"])

        n_tasks = n_fields * n_orientations
        chunk = _rank_chunk(n_tasks, size, rank)

        partial = _compute_magnetisation_chunk(
            start=chunk.start,
            end=chunk.end,
            n_fields=n_fields,
            n_orientations=n_orientations,
            states_energies=arrays["states_energies"],
            spin_matrices=arrays["spin_matrices"],
            angular_momentum_matrices=arrays["angular_momentum_matrices"],
            magnetic_fields=arrays["magnetic_fields"],
            orientations=arrays["orientations"],
            temperatures=arrays["temperatures"],
        )

        gathered = comm.gather((chunk.start, chunk.end, partial), root=0)

        if rank == 0:
            assert bundle is not None
            assert gathered is not None
            result = bundle["result"].array

            _write_gathered_result(
                result=result,
                gathered=gathered,
                n_fields=n_fields,
                n_orientations=n_orientations,
                average=average,
            )

        comm.Barrier()

    finally:
        if bundle is not None:
            # Worker attaches only; parent owns unlinking.
            bundle.close()


if __name__ == "__main__":
    main()
