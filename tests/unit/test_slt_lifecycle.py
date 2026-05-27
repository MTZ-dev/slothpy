from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import MagicMock

from slothpy.core.slt_lifecycle import _shutdown_session
from slothpy.core.slt_session import MPIProcessHandle, SltSession


def test_shutdown_session_cancels_when_open() -> None:
    session = MagicMock()
    session._closed = False

    _shutdown_session(session, wait=False)

    session.shutdown.assert_called_once_with(
        cancel_running=True,
        hard_cancel=True,
        grace_seconds=2.0,
        wait=False,
        cleanup_timeout=10.0,
    )


def test_shutdown_session_skips_closed() -> None:
    session = MagicMock()
    session._closed = True

    _shutdown_session(session, wait=True)

    session.shutdown.assert_not_called()


def test_session_shutdown_is_idempotent() -> None:
    session = SltSession.local(
        cores=2,
        max_running_jobs=1,
        install_signal_handlers=False,
    )

    session.shutdown(wait=True)
    session.shutdown(wait=True)

    assert session.snapshot().closed


def test_mpi_process_handle_terminates_subprocess_tree() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    handle = MPIProcessHandle(process=process, command=("sleep",))

    try:
        handle.terminate(grace_seconds=0.2)
        assert process.wait(timeout=5) is not None
    finally:
        if process.poll() is None:
            handle.kill()
            process.wait(timeout=5)
