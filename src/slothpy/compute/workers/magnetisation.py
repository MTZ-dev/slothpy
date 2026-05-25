from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from mpi4py import MPI

from slothpy.compute.mpi_job import read_mpi_job_spec_from_cli
from slothpy.compute.mpi_progress import (
    THREAD_PROGRESS_KEY,
    attach_worker_progress,
    publish_mpi_thread_progress,
)
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
    state_energies: np.ndarray,
    spin_matrices: np.ndarray,
    angular_momentum_matrices: np.ndarray,
    magnetic_fields: np.ndarray,
    orientations: np.ndarray,
    temperatures: np.ndarray,
    num_threads: int,
    steps_per_task: int,
    sleep_seconds: float,
    progress_interval_steps: int,
    local_thread_done: np.ndarray,
    progress: Any,
    thread_progress: np.ndarray | None,
    total_work: int,
    comm: Any,
) -> tuple[np.ndarray, bool]:
    """
    Placeholder kernel: sleep-based work units with per-thread counters.

    Real implementation will replace the inner loop with Numba kernels while
    keeping the same progress reporting hooks.
    """
    out = np.empty((end - start, temperatures.shape[0]), dtype=np.float64)
    cancelled = False

    for local_index, task_index in enumerate(range(start, end)):
        orientation_index = task_index // n_fields
        field_index = task_index % n_fields

        field = magnetic_fields[field_index]
        orientation = orientations[orientation_index]

        for step in range(steps_per_task):
            if cancelled:
                break

            thread_id = step % num_threads
            time.sleep(sleep_seconds)
            local_thread_done[thread_id] += 1

            if step % progress_interval_steps == 0:
                cancelled = publish_mpi_thread_progress(
                    comm=comm,
                    progress=progress,
                    local_thread_done=local_thread_done,
                    thread_progress=thread_progress,
                    total_work=total_work,
                )

        if cancelled:
            break

        out[local_index, :] = (
            np.linalg.norm(orientation[:3]) * float(field) / (temperatures + 1.0)
            + float(state_energies[0])
            + 0.0 * np.real(spin_matrices[0, 0, 0])
            + 0.0 * np.real(angular_momentum_matrices[0, 0, 0])
        )

    publish_mpi_thread_progress(
        comm=comm,
        progress=progress,
        local_thread_done=local_thread_done,
        thread_progress=thread_progress,
        total_work=total_work,
    )

    return out, cancelled


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

    num_threads = int(payload.get("num_threads", 1))
    steps_per_task = int(payload.get("steps_per_task", 8))
    sleep_seconds = float(payload.get("sleep_seconds", 0.05))
    progress_interval_steps = int(payload.get("progress_interval_steps", 1))
    total_work = int(payload["total_work"])

    bundle: SharedArrayBundle | None = None
    progress = attach_worker_progress(payload)

    try:
        if rank == 0:
            bundle = SharedArrayBundle.attach_from_manifest(
                spec.shared_memory_manifest,
                track=False,
            )

            arrays: dict[str, Any] = {
                "state_energies": np.asarray(bundle["state_energies"].array),
                "spin_matrices": np.asarray(bundle["spin_matrices"].array),
                "angular_momentum_matrices": np.asarray(
                    bundle["angular_momentum_matrices"].array
                ),
                "magnetic_fields": np.asarray(bundle["magnetic_fields"].array),
                "orientations": np.asarray(bundle["orientations"].array),
                "temperatures": np.asarray(bundle["temperatures"].array),
            }
            thread_progress = (
                bundle[THREAD_PROGRESS_KEY].array
                if THREAD_PROGRESS_KEY in bundle
                else None
            )
        else:
            arrays = {}
            thread_progress = None

        arrays = comm.bcast(arrays, root=0)
        thread_progress = comm.bcast(thread_progress, root=0)

        n_fields = int(payload["n_fields"])
        n_orientations = int(payload["n_orientations"])
        average = bool(payload["average"])

        n_tasks = int(payload["n_tasks"])
        chunk = _rank_chunk(n_tasks, size, rank)
        local_thread_done = np.zeros(num_threads, dtype=np.int64)

        partial, cancelled = _compute_magnetisation_chunk(
            start=chunk.start,
            end=chunk.end,
            n_fields=n_fields,
            n_orientations=n_orientations,
            state_energies=arrays["state_energies"],
            spin_matrices=arrays["spin_matrices"],
            angular_momentum_matrices=arrays["angular_momentum_matrices"],
            magnetic_fields=arrays["magnetic_fields"],
            orientations=arrays["orientations"],
            temperatures=arrays["temperatures"],
            num_threads=num_threads,
            steps_per_task=steps_per_task,
            sleep_seconds=sleep_seconds,
            progress_interval_steps=progress_interval_steps,
            local_thread_done=local_thread_done,
            progress=progress,
            thread_progress=thread_progress,
            total_work=total_work,
            comm=comm,
        )

        if cancelled:
            if rank == 0 and progress is not None:
                progress.set_cancelled()
            comm.Barrier()
            return

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

            if progress is not None:
                progress.set_finished()

        comm.Barrier()

    finally:
        if progress is not None:
            progress.close()
        if bundle is not None:
            bundle.close()


if __name__ == "__main__":
    main()
