from __future__ import annotations

from unittest.mock import MagicMock

from slothpy.compute.autotune_runtime import (
    abort_compute_runtime,
    begin_autotune_trial,
    cancel_active_autotune_trials,
    end_autotune_trial,
)
from slothpy.core.slt_progress import SltProgressTracker
from slothpy.io.shared_memory import release_all_parent_owned_shared_blocks


def test_cancel_active_autotune_trials_releases_registered_memory() -> None:
    trial = begin_autotune_trial()
    progress = SltProgressTracker.create(total=1, track=False)
    trial.register_release(progress.release)

    try:
        cancel_active_autotune_trials()
    finally:
        end_autotune_trial(trial)
        release_all_parent_owned_shared_blocks()


def test_abort_compute_runtime_shuts_down_open_session() -> None:
    session = MagicMock()
    session._closed = False

    abort_compute_runtime(session=session, cancel_session_jobs=True)

    session.shutdown.assert_called_once()
    session.shutdown.assert_called_with(
        cancel_running=True,
        hard_cancel=True,
        grace_seconds=0.0,
        wait=False,
        cleanup_timeout=0.0,
    )
