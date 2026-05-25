from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from slothpy.compute.mpi_progress import thread_progress_shape
from slothpy.core.slt_session import SltProgressTracker


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
