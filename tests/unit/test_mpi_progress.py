from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from slothpy.compute.mpi_progress import (
    publish_mpi_thread_progress,
    thread_progress_shape,
)
from slothpy.core.slt_progress import SltProgressTracker


def test_thread_progress_shape() -> None:
    assert thread_progress_shape(num_processes=4, num_threads=2) == (4, 2)


def test_progress_tracker_snapshot_fraction() -> None:
    tracker = SltProgressTracker.create(total=10)
    try:
        tracker.set_running()
        tracker.set_done(5)
        snapshot = tracker.snapshot()
        assert snapshot.fraction == pytest.approx(0.5)
    finally:
        tracker.release()


class _FakeComm:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def allreduce(self, value: int, op: object = None) -> int:
        return value

    def gather(self, data: np.ndarray, root: int = 0) -> list[np.ndarray] | None:
        if self._rank == root:
            return [np.array(data, copy=True)]
        return None

    def bcast(self, value: bool, root: int = 0) -> bool:
        return value


def test_publish_single_rank_writes_shm_without_mpi_transfer() -> None:
    comm = _FakeComm(rank=0, size=1)
    thread_progress = np.zeros((1, 2), dtype=np.int64)
    local_done = np.array([3, 2], dtype=np.int64)
    progress = SltProgressTracker.create(total=100)

    try:
        cancel = publish_mpi_thread_progress(
            comm=comm,
            progress=progress,
            local_thread_done=local_done,
            thread_progress=thread_progress,
            total_work=100,
        )

        assert cancel is False
        np.testing.assert_array_equal(thread_progress[0], [3, 2])
        assert progress.snapshot().done == 5
    finally:
        progress.release()
