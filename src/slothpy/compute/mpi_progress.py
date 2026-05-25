from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI

from slothpy.core.slt_session import SltProgressSpec, SltWorkerProgress

THREAD_PROGRESS_KEY = "thread_progress"


def thread_progress_shape(
    *,
    num_processes: int,
    num_threads: int,
) -> tuple[int, int]:
    if num_processes < 1:
        raise ValueError("num_processes must be >= 1.")
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1.")
    return (num_processes, num_threads)


def publish_mpi_thread_progress(
    *,
    comm: Any,
    progress: SltWorkerProgress | None,
    local_thread_done: np.ndarray,
    thread_progress: np.ndarray | None,
    total_work: int,
    root: int = 0,
) -> bool:
    """
    Aggregate per-thread counters from all MPI ranks into shared memory.

    Each rank maintains ``local_thread_done`` with shape ``(num_threads,)``.
    Rank ``root`` optionally writes the gathered rows into ``thread_progress``
    with shape ``(comm.size, num_threads)`` and updates the parent-visible
    :class:`~slothpy.core.slt_session.SltProgressTracker`.

    Returns
    -------
    bool
        ``True`` when the parent requested cancellation.
    """
    rank = comm.Get_rank()

    local_board = np.asarray(local_thread_done, dtype=np.int64).reshape(-1)
    local_sum = int(local_board.sum())
    global_done = int(comm.allreduce(local_sum, op=MPI.SUM))

    gathered_rows = comm.gather(local_board, root=root)

    cancel = False
    if rank == root:
        if thread_progress is not None and gathered_rows is not None:
            n_cols = thread_progress.shape[1]
            for process_index, row in enumerate(gathered_rows):
                thread_progress[process_index, :n_cols] = row[:n_cols]

        if progress is not None:
            progress.set_total(int(total_work))
            progress.set_done(min(global_done, int(total_work)))
            progress.set_running()
            cancel = progress.cancel_requested()

    return bool(comm.bcast(cancel, root=root))


def attach_worker_progress(
    payload: dict[str, Any],
) -> SltWorkerProgress | None:
    """
    Attach to the session progress block described in the MPI job payload.
    """
    raw = payload.get("progress")
    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise TypeError("MPI payload progress must be a mapping.")

    return SltWorkerProgress.attach(SltProgressSpec.from_json_dict(raw), track=False)


__all__ = [
    "THREAD_PROGRESS_KEY",
    "attach_worker_progress",
    "publish_mpi_thread_progress",
    "thread_progress_shape",
]
