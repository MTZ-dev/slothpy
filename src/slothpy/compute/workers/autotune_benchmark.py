"""
MPI benchmark worker for local MPI/thread autotuning.

Each rank runs a bucket of tasks in parallel threads. Work units combine a
short sleep (simulating Numba loop overhead) with a small dense matrix
multiply (exercising BLAS with per-rank thread limits from the environment).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

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
class _RankChunk:
    start: int
    end: int


def _rank_chunk(n_items: int, size: int, rank: int) -> _RankChunk:
    chunk_size = n_items // size
    remainder = n_items % size
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    return _RankChunk(start=start, end=end)


def _run_task_steps(
    *,
    task_index: int,
    thread_id: int,
    steps_per_task: int,
    sleep_seconds: float,
    compute_size: int,
    progress_interval_steps: int,
    local_thread_done: np.ndarray,
) -> None:
    rng = np.random.default_rng(task_index * 9973 + thread_id * 17)
    left = np.asarray(rng.standard_normal(compute_size), dtype=np.float64)
    right = np.asarray(rng.standard_normal(compute_size), dtype=np.float64)

    for step in range(steps_per_task):
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        _ = float(left @ right)
        left = np.roll(left, 1)

        local_thread_done[thread_id] += 1

        if (
            progress_interval_steps > 0
            and (step + 1) % progress_interval_steps == 0
        ):
            pass


def _thread_bucket(
    *,
    thread_id: int,
    task_indices: list[int],
    steps_per_task: int,
    sleep_seconds: float,
    compute_size: int,
    progress_interval_steps: int,
    local_thread_done: np.ndarray,
) -> None:
    for task_index in task_indices:
        _run_task_steps(
            task_index=task_index,
            thread_id=thread_id,
            steps_per_task=steps_per_task,
            sleep_seconds=sleep_seconds,
            compute_size=compute_size,
            progress_interval_steps=progress_interval_steps,
            local_thread_done=local_thread_done,
        )


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    spec = read_mpi_job_spec_from_cli()
    payload = spec.payload

    num_threads = int(payload["num_threads"])
    n_tasks = int(payload["n_tasks"])
    steps_per_task = int(payload["steps_per_task"])
    sleep_seconds = float(payload.get("sleep_seconds", 0.0))
    compute_size = int(payload.get("compute_size", 64))
    progress_interval_steps = int(payload.get("progress_interval_steps", 1))
    total_work = int(payload["total_work"])
    progress = attach_worker_progress(payload)
    bundle: SharedArrayBundle | None = None
    thread_progress: np.ndarray | None = None

    try:
        if rank == 0:
            bundle = SharedArrayBundle.attach_from_manifest(
                spec.shared_memory_manifest,
                track=False,
            )
            thread_progress = (
                bundle[THREAD_PROGRESS_KEY].array
                if THREAD_PROGRESS_KEY in bundle
                else None
            )
        else:
            thread_progress = None

        thread_progress = comm.bcast(thread_progress, root=0)

        chunk = _rank_chunk(n_tasks, size, rank)
        local_thread_done = np.zeros(num_threads, dtype=np.int64)

        task_indices = list(range(chunk.start, chunk.end))
        buckets: list[list[int]] = [[] for _ in range(num_threads)]
        for index, task_index in enumerate(task_indices):
            buckets[index % num_threads].append(task_index)

        cancelled = False
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    _thread_bucket,
                    thread_id=thread_id,
                    task_indices=buckets[thread_id],
                    steps_per_task=steps_per_task,
                    sleep_seconds=sleep_seconds,
                    compute_size=compute_size,
                    progress_interval_steps=progress_interval_steps,
                    local_thread_done=local_thread_done,
                )
                for thread_id in range(num_threads)
                if buckets[thread_id]
            ]

            while futures and not cancelled:
                cancelled = publish_mpi_thread_progress(
                    comm=comm,
                    progress=progress,
                    local_thread_done=local_thread_done,
                    thread_progress=thread_progress,
                    total_work=total_work,
                )
                if all(future.done() for future in futures):
                    break
                time.sleep(0.01)

            for future in as_completed(futures):
                future.result()

        publish_mpi_thread_progress(
            comm=comm,
            progress=progress,
            local_thread_done=local_thread_done,
            thread_progress=thread_progress,
            total_work=total_work,
        )

        if progress is not None and rank == 0:
            progress.set_finished()

    finally:
        if progress is not None:
            progress.close()
        if bundle is not None:
            bundle.close()


if __name__ == "__main__":
    main()
