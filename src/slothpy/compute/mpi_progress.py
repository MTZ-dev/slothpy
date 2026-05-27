from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI

from slothpy.core.slt_progress import SltProgressSpec, SltWorkerProgress

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


def worker_thread_progress_view(
    bundle: Any,
    *,
    rank: int,
) -> np.ndarray | None:
    """
    Parent-visible ``thread_progress`` board for MPI rank 0 only.

    Other ranks keep per-thread counters locally and send them to rank 0 via
    :func:`publish_mpi_thread_progress`. Do not broadcast this array to other
    ranks; only rank 0's writes are visible to the session parent.
    """
    if rank != 0:
        return None

    if bundle is None or THREAD_PROGRESS_KEY not in bundle:
        return None

    return np.asarray(bundle[THREAD_PROGRESS_KEY].array, dtype=np.int64)


def _write_thread_progress_row(
    thread_progress: np.ndarray,
    process_index: int,
    local_board: np.ndarray,
) -> None:
    n_cols = thread_progress.shape[1]
    thread_progress[process_index, :n_cols] = np.asarray(
        local_board, dtype=np.int64
    ).reshape(-1)[:n_cols]


def _update_parent_progress(
    progress: SltWorkerProgress | None,
    *,
    global_done: int,
    total_work: int,
) -> bool:
    if progress is None:
        return False

    progress.set_total(int(total_work))
    progress.set_done(min(int(global_done), int(total_work)))
    progress.set_running()
    return progress.cancel_requested()


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
    Publish per-thread counters for the session parent.

    Rank ``root`` (with a shared-memory ``thread_progress`` view) writes its row
    directly into the parent-attached array and updates the scalar progress block.
    No MPI transfer is used when ``comm.size == 1``.

    Other ranks keep a local ``(num_threads,)`` counter array and periodically
    send it to rank ``root`` via ``MPI.gather``; rank ``root`` merges rows into
    ``thread_progress``. The scalar ``done`` count is always the sum of all ranks'
    counters (``MPI.Allreduce`` when size > 1).

    Returns
    -------
    bool
        ``True`` when the parent requested cancellation.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_board = np.asarray(local_thread_done, dtype=np.int64).reshape(-1)
    local_sum = int(local_board.sum())

    if size == 1:
        if rank != root:
            return False

        if thread_progress is not None:
            _write_thread_progress_row(thread_progress, root, local_board)

        cancel = _update_parent_progress(
            progress,
            global_done=local_sum,
            total_work=total_work,
        )
        return cancel

    global_done = int(comm.allreduce(local_sum, op=MPI.SUM))

    if rank == root:
        if thread_progress is not None:
            _write_thread_progress_row(thread_progress, rank, local_board)

        gathered_rows = comm.gather(local_board, root=root)
        if thread_progress is not None and gathered_rows is not None:
            n_cols = thread_progress.shape[1]
            for process_index, row in enumerate(gathered_rows):
                if process_index == root:
                    continue
                thread_progress[process_index, :n_cols] = np.asarray(
                    row, dtype=np.int64
                ).reshape(-1)[:n_cols]

        cancel = _update_parent_progress(
            progress,
            global_done=global_done,
            total_work=total_work,
        )
    else:
        comm.gather(local_board, root=root)
        cancel = False

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
    "worker_thread_progress_view",
]
